"""
YOLOv8n-P2 Training State Machine - Hailo8 / 1920x1080 Camera

Target  : 1024x576  (16:9, both /32, perfect 1920x1080 match)
Model   : YOLOv8n-P2 (nano + extra P2 small-object head)
Export  : ONNX float32  (Hailo quantizes externally)

Architecture:
  Phases are defined in PHASE_CONFIG (ordered list).
  Each phase references a LOSS_PRESETS entry and an AUGMENTATION_PRESETS entry.
  The state machine iterates PHASE_CONFIG in order, skipping completed phases.

Usage:
  python train_hailo.py                       # run all phases
  python train_hailo.py --from precision      # resume from "precision" phase
  python train_hailo.py --only export         # export only
  python train_hailo.py --only high_recall    # run single phase by name

Adding a new phase:
  1. Append a dict to PHASE_CONFIG
  2. Pick a loss_preset and augmentation preset (or define new ones)
  3. Re-run train_hailo.py - completed phases are auto-skipped

State is saved in {project}/train_state.json after each phase.
If training crashes, re-run - completed phases are skipped automatically.

Requirements:
  pip install ultralytics onnx onnxsim
"""

import argparse
import json
import sys
import torch
from pathlib import Path
from ultralytics import YOLO
from recall_trainer import (
    RecallFocusedTrainer,
    build_recall_model,
    compare_checkpoints,
    MODEL_YAML,
    MODEL_WEIGHTS,
    make_grayscale_callback,
)
from loss_function import (
    get_loss_summary,
    CLASS_CONFIG,
    NUM_CLASSES,
)


# ======================================================================
#  GLOBAL CONFIGURATION  <-  Edit this block for hardware/dataset
# ======================================================================

CFG = {
    # -- Dataset & project -------------------------------------------------
    "data":         "../Dataset/dataset.yaml",         # dataset yaml path
    "project":      "yolov8s-robocup2A/runs/hailo_train",  # output folder

    # -- Model -------------------------------------------------------------
    # YOLOv8n-P2: nano backbone (fast on Hailo8) with P2 head (stride 4)
    # for detecting objects as small as ~4px at 1024x576.
    # There is no yolov8n-p2.pt pretrained file - we load yolov8n.pt
    # weights into the yolov8n-p2.yaml architecture via .load()
    "base_model":   MODEL_YAML,         # "yolov8n-p2.yaml"
    "base_weights": MODEL_WEIGHTS,      # "yolov8n.pt"

    # -- Resolution --------------------------------------------------------
    # 1024x576 = exact 16:9, both divisible by 32
    # Matches 1920x1080 camera with zero distortion / letterboxing
    "imgsz":        (1024, 576),

    # -- Hardware ----------------------------------------------------------
    "device":       "0",                # GPU id - use "cpu" if no GPU
    "workers":      4,                  # data loader threads (8 caused OOM on Windows spawn)
    "cache":        "disk",             # "ram" causes OOM with mosaic aug; disk is safe
}


# ======================================================================
#  LOSS PRESETS
#
#  Each preset defines loss hyperparameters passed to the trainer.
#  Per-class weights come from loss_function.py CLASS_CONFIG unless
#  overridden here via "class_recall_overrides".
#
#  Keys:
#    box          - box regression loss weight (higher = tighter boxes)
#    cls          - classification loss weight (higher = stricter class ID)
#    dfl          - distribution focal loss weight (keep 1.5 usually)
#    gamma_neg    - ASL focal exponent for false positives (2-6)
#    gamma_pos    - ASL focal exponent for true positives (keep 0)
#    clip         - ASL probability margin delta (0.0-0.1, higher = more recall)
#    class_recall_overrides - dict {class_id: weight} to override recall_weight
#                             from loss_function.py for this preset only
# ======================================================================

LOSS_PRESETS = {
    # High recall bias: relaxed box/cls, aggressive ASL margin
    # Use when recall is the primary goal (early training)
    "recall_focused": {
        "box":       4.0,       # relaxed (default 7.5) - more proposals
        "cls":       0.3,       # relaxed (default 0.5) - less class penalty
        "dfl":       1.5,       # standard
        "gamma_neg": 4,         # aggressive FP penalty on confident predictions
        "gamma_pos": 0,         # never suppress true positives
        "clip":      0.05,      # soft negatives below 5% ignored -> higher recall
        "class_recall_overrides": {},  # use loss_function.py defaults
    },

    # Precision mode: tighter boxes, stricter classification
    # Use after recall is established to sharpen predictions
    "precision": {
        "box":       7.5,       # tight (default) - penalizes sloppy boxes
        "cls":       0.5,       # standard - full class discrimination
        "dfl":       1.5,       # standard
        "gamma_neg": 3,         # milder FP penalty (allow borderline detections)
        "gamma_pos": 0,         # never suppress true positives
        "clip":      0.02,      # small margin - most soft FPs penalized
        "class_recall_overrides": {},  # use loss_function.py defaults
    },

    # Targeted per-class surgery: boost specific weak classes
    # Use when specific classes underperform after general training
    "class_surgery": {
        "box":       6.0,       # moderate box weight
        "cls":       0.4,       # slightly relaxed classification
        "dfl":       1.5,       # standard
        "gamma_neg": 4,         # aggressive FP penalty
        "gamma_pos": 0,         # never suppress true positives
        "clip":      0.05,      # permissive margin for recall
        "class_recall_overrides": {
            0: 2.5,             # robot: boost recall (often confused with robot_rct)
            5: 2.0,            # tag_rouge: boost recall (small, often missed)
        },
    },
}


# ======================================================================
#  AUGMENTATION PRESETS
#
#  Each preset defines augmentation hyperparameters for model.train().
#
#  Keys:
#    mosaic       - probability of mosaic (4 images merged) - (0.0-1.0)
#    copy_paste   - probability of copy-paste augmentation - (0.0-0.5)
#    scale        - random scale factor range - (0.0-1.0, higher = more variation)
#    mixup        - probability of mixup (blend 2 images) - (0.0-0.3)
#    close_mosaic - disable mosaic for last N epochs (stabilizes convergence)
#    hsv_h        - hue shift range - (0.0-0.1)
#    hsv_s        - saturation shift range - (0.0-1.0)
#    hsv_v        - brightness/value shift range - (0.0-1.0)
#    fliplr       - probability of horizontal flip - (0.0-1.0)
# ======================================================================

AUGMENTATION_PRESETS = {
    # Maximum augmentation: exposes small objects, builds robustness
    # Use in early training when model needs diverse examples
    "aggressive": {
        "mosaic":       1.0,    # always merge 4 images - max object density
        "copy_paste":   0.3,    # paste objects onto other images - key for small obj
        "scale":        0.9,    # large scale variation - exposes tiny objects
        "mixup":        0.15,   # blend 2 images - regularization
        "close_mosaic": 10,     # disable mosaic last 10 epochs for stable convergence
        "hsv_h":        0.015,  # mild hue shift
        "hsv_s":        0.7,    # strong saturation variation
        "hsv_v":        0.4,    # moderate brightness variation
        "fliplr":       0.5,    # 50% horizontal flip
    },

    # Moderate augmentation: balanced robustness and convergence
    # Use in precision phases when model is refining
    "moderate": {
        "mosaic":       0.5,    # 50% mosaic - less aggressive
        "copy_paste":   0.2,    # moderate copy-paste
        "scale":        0.5,    # moderate scale variation
        "mixup":        0.05,   # minimal mixup
        "close_mosaic": 15,     # disable mosaic last 15 epochs
        "hsv_h":        0.015,  # mild hue shift
        "hsv_s":        0.7,    # strong saturation variation
        "hsv_v":        0.4,    # moderate brightness variation
        "fliplr":       0.5,    # 50% horizontal flip
    },

    # Minimal augmentation: let model consolidate learned features
    # Use in surgery/fine-tune phases with very low LR
    "minimal": {
        "mosaic":       0.0,    # off - no mosaic
        "copy_paste":   0.05,   # near-zero copy-paste
        "scale":        0.2,    # small scale variation
        "mixup":        0.0,    # off
        "close_mosaic": 0,      # N/A (mosaic already off)
        "hsv_h":        0.01,   # very mild hue shift
        "hsv_s":        0.4,    # moderate saturation
        "hsv_v":        0.4,    # moderate brightness
        "fliplr":       0.5,    # 50% horizontal flip
    },

    # Color stress test: extreme color variation + high grayscale_p
    # Forces model to rely on shape/texture not color
    # Critical for cameras that switch between day (color) and night (IR/gray)
    "color_stress": {
        "mosaic":       1.0,    # always merge 4 images
        "copy_paste":   0.3,    # strong copy-paste
        "scale":        0.9,    # large scale variation
        "mixup":        0.15,   # moderate mixup
        "close_mosaic": 10,     # disable mosaic last 10 epochs
        "hsv_h":        0.015,  # mild hue shift
        "hsv_s":        0.9,    # EXTREME saturation variation
        "hsv_v":        0.6,    # STRONG brightness variation
        "fliplr":       0.5,    # 50% horizontal flip
    },
}


# ======================================================================
#  PHASE CONFIGURATION
#
#  Ordered list of training phases. The state machine runs them in order.
#  Each phase dict has:
#    name           - folder name for outputs (must be unique)
#    epochs         - max training epochs for this phase
#    loss_preset    - key into LOSS_PRESETS
#    lr0            - initial learning rate
#    lrf            - final LR multiplier (final_lr = lr0 * lrf)
#    warmup_epochs  - epochs of linear LR warmup (protects backbone)
#    batch          - batch size (-1 = auto-detect from GPU memory)
#    grayscale_p    - fraction of batch converted to 3-channel grayscale
#    augmentation   - key into AUGMENTATION_PRESETS
#    patience       - early stopping patience (epochs without improvement)
#    stop_condition - human-readable description of when to stop
#                     (informational only, not executed)
#
#  To add a phase: append a dict here, pick presets, re-run.
#  To skip a phase: comment it out.
# ======================================================================

PHASE_CONFIG = [
    # -- Phase 1: High Recall -----------------------------------------------
    # Goal: establish strong recall baseline across all classes.
    # Relaxed loss (low box/cls weight) encourages more proposals.
    # Aggressive augmentation ensures small objects are seen.
    # Warmup protects backbone features in lieu of freezing.
    {
        "name":           "high_recall",
        "epochs":         60,               # generous - early stopping will cut short
        "loss_preset":    "recall_focused",  # relaxed box/cls, high ASL margin
        "lr0":            0.001,            # moderate LR with warmup protection
        "lrf":            0.01,             # decay to lr0 * 0.01 by end
        "warmup_epochs":  5,                # 5 epochs warmup - protects backbone
        "batch":          -1,               # auto-batch (fills ~60% VRAM)
        "grayscale_p":    0.2,              # 20% grayscale for mild color invariance
        "augmentation":   "aggressive",     # max augmentation for recall
        "patience":       20,               # generous patience for early phase
        "stop_condition": "ballon recall > 0.95, overall recall > 0.80",
    },

    # -- Phase 2: Color Invariance ------------------------------------------
    # Goal: make detections robust to color/lighting changes.
    # 70% grayscale forces shape/texture reliance over color.
    # Same recall-focused loss - we're not tightening precision yet.
    # Essential for cameras that switch day (color) -> night (IR/gray).
    {
        "name":           "color_invariance",
        "epochs":         40,               # shorter - building on phase 1
        "loss_preset":    "recall_focused",  # keep recall-focused loss
        "lr0":            0.0005,           # lower LR - refining not relearning
        "lrf":            0.01,             # decay to lr0 * 0.01
        "warmup_epochs":  5,                # warmup still important
        "batch":          -1,               # auto-batch
        "grayscale_p":    0.7,              # 70% grayscale - heavy color stress
        "augmentation":   "color_stress",   # extreme hsv_s/hsv_v variation
        "patience":       15,               # moderate patience
        "stop_condition": "metrics stable under grayscale augmentation",
    },

    # -- Phase 3: Precision -------------------------------------------------
    # Goal: tighten bounding boxes and classification accuracy.
    # Tight box weight (7.5) penalizes sloppy localization.
    # Lower ASL clip (0.02) penalizes most false positives.
    # Moderate augmentation for stable convergence.
    {
        "name":           "precision",
        "epochs":         40,               # moderate - fine-tuning
        "loss_preset":    "precision",      # tight box/cls, low ASL margin
        "lr0":            0.0002,           # very low LR - surgical refinement
        "lrf":            0.01,             # decay to lr0 * 0.01
        "warmup_epochs":  5,                # warmup protects learned features
        "batch":          -1,               # auto-batch
        "grayscale_p":    0.2,              # 20% grayscale maintenance
        "augmentation":   "moderate",       # balanced augmentation
        "patience":       15,               # moderate patience
        "stop_condition": "mAP50-95 > 0.60, precision > 0.82",
    },

    # -- Phase 4: Class Surgery (OPTIONAL - uncomment if needed) ------------
    # Goal: boost recall for specific underperforming classes.
    # Elevates recall_weight for robot and tag_rouge.
    # Very low LR + minimal augmentation = targeted fine-tune.
    # Only run if specific classes are still weak after phase 3.
    #
    # {
    #     "name":           "class_surgery",
    #     "epochs":         25,               # short - targeted fix
    #     "loss_preset":    "class_surgery",  # elevated weights for weak classes
    #     "lr0":            0.00005,          # near-zero LR
    #     "lrf":            0.01,             # minimal decay
    #     "warmup_epochs":  5,                # warmup still needed
    #     "batch":          -1,               # auto-batch
    #     "grayscale_p":    0.1,              # 10% grayscale maintenance
    #     "augmentation":   "minimal",        # let model consolidate
    #     "patience":       10,               # short patience
    #     "stop_condition": "robot recall > 0.75, tag_rouge recall > 0.80",
    # },
]


# ======================================================================
#  STATE PERSISTENCE
# ======================================================================

def state_path() -> Path:
    return Path(CFG["project"]) / "train_state.json"

def save_state(phase_name: str, weights: str):
    p = state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {"last_completed": phase_name, "weights": weights}
    with open(p, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n[state] saved -> last_completed={phase_name}, weights={weights}")

def load_state() -> dict:
    p = state_path()
    if p.exists():
        with open(p) as f:
            state = json.load(f)
        print(f"[state] found existing state: {state}")
        return state
    return {"last_completed": None, "weights": CFG["base_model"]}


# ======================================================================
#  TRAINING HELPER
# ======================================================================

def get_phase_names() -> list:
    """Return ordered list of active phase names."""
    return [p["name"] for p in PHASE_CONFIG]


def resolve_loss_preset(preset_name: str) -> dict:
    """Look up and return a copy of a loss preset by name."""
    if preset_name not in LOSS_PRESETS:
        sys.exit(f"[error] Unknown loss_preset '{preset_name}'. "
                 f"Available: {list(LOSS_PRESETS.keys())}")
    return dict(LOSS_PRESETS[preset_name])


def resolve_augmentation_preset(preset_name: str) -> dict:
    """Look up and return a copy of an augmentation preset by name."""
    if preset_name not in AUGMENTATION_PRESETS:
        sys.exit(f"[error] Unknown augmentation preset '{preset_name}'. "
                 f"Available: {list(AUGMENTATION_PRESETS.keys())}")
    return dict(AUGMENTATION_PRESETS[preset_name])


def run_phase(phase: dict, weights: str) -> str:
    """Run one training phase. Returns path to best.pt."""
    phase_name = phase["name"]
    loss_preset = resolve_loss_preset(phase["loss_preset"])
    aug_preset = resolve_augmentation_preset(phase["augmentation"])

    print(f"\n{'=' * 70}")
    print(f"  PHASE: {phase_name.upper()}")
    print(f"  loss_preset: {phase['loss_preset']}  |  augmentation: {phase['augmentation']}")
    print(f"  weights: {weights}")
    print(f"  stop_condition: {phase.get('stop_condition', 'N/A')}")
    print(f"{'=' * 70}\n")

    # Build model: first phase starts from yaml+pretrained, others from .pt
    if weights.endswith(".yaml"):
        model = build_recall_model(weights, CFG["base_weights"])
    else:
        model = YOLO(weights)

    # Register grayscale callback
    gray_cb = make_grayscale_callback(phase["grayscale_p"])
    if gray_cb:
        model.add_callback("on_train_batch_start", gray_cb)
        print(f"[grayscale] enabled - p={phase['grayscale_p']}")

    model.train(
        trainer=RecallFocusedTrainer,
        # -- Core ---------------------------------------------------------
        data=CFG["data"],
        project=CFG["project"],
        name=phase_name,
        exist_ok=True,
        imgsz=CFG["imgsz"],
        device=CFG["device"],
        workers=CFG["workers"],
        cache=CFG["cache"],

        # -- Optimizer & schedule -----------------------------------------
        optimizer="AdamW",
        cos_lr=True,
        amp=True,
        epochs=phase["epochs"],
        batch=phase["batch"],
        lr0=phase["lr0"],
        lrf=phase["lrf"],
        warmup_epochs=phase["warmup_epochs"],
        patience=phase["patience"],

        # -- Loss weights (from preset) -----------------------------------
        box=loss_preset["box"],
        cls=loss_preset["cls"],
        dfl=loss_preset["dfl"],

        # -- ASL parameters passed to trainer via extra args ---------------
        # These are picked up by RecallFocusedTrainer._setup_train()
        loss_preset_name=phase["loss_preset"],
        loss_preset_data=loss_preset,

        # -- Augmentation (from preset) -----------------------------------
        mosaic=aug_preset["mosaic"],
        close_mosaic=aug_preset["close_mosaic"],
        copy_paste=aug_preset["copy_paste"],
        mixup=aug_preset["mixup"],
        scale=aug_preset["scale"],
        hsv_h=aug_preset["hsv_h"],
        hsv_s=aug_preset["hsv_s"],
        hsv_v=aug_preset["hsv_v"],
        fliplr=aug_preset["fliplr"],
        flipud=0.0,

        # -- Regularization -----------------------------------------------
        label_smoothing=0.0,
        dropout=0.0,

        # -- Output --------------------------------------------------------
        val=True,
        plots=True,
        save_period=10,
    )

    best = str(Path(CFG["project"]) / phase_name / "weights" / "best.pt")
    if not Path(best).exists():
        sys.exit(f"[error] Expected best.pt at {best} - training may have failed.")

    print(f"\n[{phase_name}] best weights -> {best}")
    return best


# ======================================================================
#  VALIDATION + THRESHOLD SWEEP
# ======================================================================

def threshold_sweep(weights: str):
    print(f"\n{'=' * 62}")
    print("  THRESHOLD SWEEP  (no training - finds best conf & iou)")
    print(f"{'=' * 62}\n")
    print(f"  {'conf':>6}  {'iou':>6}  {'P':>7}  {'R':>7}  {'mAP50':>8}  {'mAP50-95':>10}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*10}")

    model = YOLO(weights)
    rows = []
    best_r = None

    for conf in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        for iou in [0.30, 0.45, 0.60]:
            m = model.val(
                data=CFG["data"],
                imgsz=CFG["imgsz"],
                conf=conf,
                iou=iou,
                device=CFG["device"],
                verbose=False,
            )
            row = dict(
                conf=conf,
                iou=iou,
                P=round(float(m.box.mp), 3),
                R=round(float(m.box.mr), 3),
                mAP50=round(float(m.box.map50), 3),
                mAP5095=round(float(m.box.map), 3),
            )
            rows.append(row)
            print(f"  {conf:6.2f}  {iou:6.2f}  {row['P']:7.3f}  {row['R']:7.3f}"
                  f"  {row['mAP50']:8.3f}  {row['mAP5095']:10.3f}")

            if best_r is None or row["R"] > best_r["R"]:
                best_r = row

    out = Path(CFG["project"]) / "threshold_sweep.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\n  Results saved -> {out}")
    print(f"\n  Best recall config:")
    print(f"    conf={best_r['conf']}  iou={best_r['iou']}"
          f"  ->  R={best_r['R']}  P={best_r['P']}  mAP50={best_r['mAP50']}")
    print(f"\n  -> Use these values in your Hailo inference pipeline.")


# ======================================================================
#  EXPORT - ONNX float32 for Hailo Dataflow Compiler
# ======================================================================

def export_for_hailo(weights: str) -> str:
    print(f"\n{'=' * 62}")
    print("  EXPORT -> ONNX float32  (Hailo quantizes externally)")
    print(f"{'=' * 62}\n")

    model = YOLO(weights)

    model.export(
        format="onnx",
        imgsz=CFG["imgsz"],         # (1024, 576) - matches training resolution
        opset=11,                   # Hailo Dataflow Compiler prefers opset 11
        simplify=True,              # onnx-simplifier: cleaner graph
        dynamic=False,              # fixed batch=1, fixed HxW - required for Hailo
        half=False,                 # float32 - Hailo does int8 quantization itself
        device="cpu",               # CPU export = cleaner ONNX graph
    )

    onnx_path = str(Path(weights).with_suffix(".onnx"))
    print(f"\n  ONNX model -> {onnx_path}")
    print(f"""
  -- Hailo next steps -----------------------------------------------
  1. Parse + optimize:
       hailo parser onnx {onnx_path} --hw-arch hailo8

  2. Quantize (provide calibration images, ~100-300 frames):
       hailo optimize model.har --hw-arch hailo8 --calib-path calib_images/

  3. Compile:
       hailo compiler model.har --hw-arch hailo8

  4. Output: model.hef  ->  deploy to Hailo8 runtime
  --------------------------------------------------------------------
    """)
    return onnx_path


# ======================================================================
#  QUICK VALIDATION REPORT
# ======================================================================

def quick_val(weights: str):
    print(f"\n[val] Running quick validation on {weights}")
    model = YOLO(weights)
    m = model.val(
        data=CFG["data"],
        imgsz=CFG["imgsz"],
        conf=0.001,
        iou=0.6,
        device=CFG["device"],
    )
    print(f"""
  -- Validation results --------------------------------------------
  Precision    : {m.box.mp:.4f}
  Recall       : {m.box.mr:.4f}   <- target > 0.85
  mAP50        : {m.box.map50:.4f}
  mAP50-95     : {m.box.map:.4f}

  Per-class mAP50:
    {dict(zip(m.names.values(), [round(x, 3) for x in m.box.maps]))}
  ------------------------------------------------------------------
    """)


# ======================================================================
#  STATE MACHINE
# ======================================================================

def should_run(phase_name: str, last_completed: str | None, start_from: str | None) -> bool:
    """Decide if a phase should run based on state and CLI args."""
    phase_names = get_phase_names()
    all_steps = phase_names + ["export"]

    if phase_name not in all_steps:
        return False

    phase_idx = all_steps.index(phase_name)

    if start_from:
        if start_from not in all_steps:
            return True
        start_idx = all_steps.index(start_from)
        if phase_idx < start_idx:
            return False

    if last_completed:
        if last_completed not in all_steps:
            return True
        last_idx = all_steps.index(last_completed)
        if phase_idx <= last_idx:
            if not start_from:
                return False

    return True


def main():
    phase_names = get_phase_names()
    all_choices = phase_names + ["export", "val", "sweep"]

    parser = argparse.ArgumentParser(description="YOLOv8n-P2 Hailo Training State Machine")
    parser.add_argument("--from", dest="start_from", default=None,
                        choices=phase_names + ["export"],
                        help="Force restart from this phase (skips earlier phases)")
    parser.add_argument("--only", dest="only", default=None,
                        choices=all_choices,
                        help="Run only this single step")
    parser.add_argument("--weights", default=None,
                        help="Override weights path (useful with --only val/sweep/export)")
    parser.add_argument("--skip", dest="skip_phases", nargs="*", default=[],
                        help="Phase names to skip (e.g. --skip color_invariance)")
    args = parser.parse_args()

    state = load_state()
    weights = args.weights or state["weights"]
    last = state.get("last_completed")

    # Print loss configuration at startup
    print(get_loss_summary())

    # -- --only shortcuts --------------------------------------------------
    if args.only == "val":
        quick_val(weights); return
    if args.only == "sweep":
        threshold_sweep(weights); return
    if args.only == "export":
        export_for_hailo(weights); return

    # --only <phase_name>: run a single phase
    if args.only and args.only in phase_names:
        phase = next(p for p in PHASE_CONFIG if p["name"] == args.only)
        weights = run_phase(phase, weights)
        save_state(phase["name"], weights)
        quick_val(weights)
        return

    # -- Full state machine ------------------------------------------------
    for phase in PHASE_CONFIG:
        phase_name = phase["name"]

        if phase_name in args.skip_phases:
            print(f"[skip] {phase_name} (--skip)")
            continue

        if should_run(phase_name, last, args.start_from):
            weights = run_phase(phase, weights)
            save_state(phase_name, weights)
            quick_val(weights)
        else:
            print(f"[skip] {phase_name} - already completed")

    # Threshold sweep
    threshold_sweep(weights)

    # Compare checkpoints from last completed phase
    last_phase_name = get_phase_names()[-1]
    for p in reversed(PHASE_CONFIG):
        run_dir = Path(CFG["project"]) / p["name"]
        if (run_dir / "weights" / "best.pt").exists():
            last_phase_name = p["name"]
            break
    compare_checkpoints(
        str(Path(CFG["project"]) / last_phase_name),
        CFG["data"], imgsz=CFG["imgsz"], device=CFG["device"]
    )

    # Export
    if should_run("export", last, args.start_from):
        export_for_hailo(weights)
        save_state("export", weights)

    print(f"\n{'=' * 62}")
    print("  ALL DONE")
    print(f"  Final weights : {weights}")
    print(f"  State file    : {state_path()}")
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    main()

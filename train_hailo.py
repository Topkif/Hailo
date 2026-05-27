"""
╔══════════════════════════════════════════════════════════════════╗
║  YOLOv8s-P2 Training State Machine — Hailo8 / 1920×1080 Camera  ║
║                                                                  ║
║  Target  : 1024×576  (16:9, both ×32, perfect 1920×1080 match)  ║
║  Model   : YOLOv8s-P2 (extra small-object P2 detection head)    ║
║  Export  : ONNX float32  (Hailo quantizes externally)           ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
  python train_hailo.py                  # run all phases
  python train_hailo.py --from phase2    # resume from phase 2
  python train_hailo.py --only export    # export only

State is saved in {project}/train_state.json after each phase.
If training crashes, re-run — completed phases are skipped automatically.

Requirements:
  pip install ultralytics onnx onnxsim
"""

import argparse
import json
import sys
import torch
from pathlib import Path
from ultralytics import YOLO
from recall_trainer import RecallFocusedTrainer, build_recall_model, compare_checkpoints


# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION  ←  Edit only this block
# ══════════════════════════════════════════════════════════════════

CFG = {
    # ── Dataset & project ─────────────────────────────────────────
    "data":         "data.yaml",            # your dataset yaml
    "project":      "runs/hailo_train",     # output folder

    # ── Model ─────────────────────────────────────────────────────
    # YOLOv8s-P2 adds a P2 head (stride 4) for small object detection
    # Download: ultralytics will auto-fetch yolov8s-p2.pt on first run
    "base_model":   "yolov8s-p2.yaml",  # architecture; weights loaded from yolov8s.pt

    # ── Resolution ────────────────────────────────────────────────
    # 1024×576 = exact 16:9, both divisible by 32
    # Matches 1920×1080 camera with zero distortion / letterboxing
    "imgsz":        (1024, 576),

    # ── Hardware ──────────────────────────────────────────────────
    "device":       "0",                    # GPU id — use "cpu" if no GPU
    "workers":      8,                      # data loader threads
    "cache":        "ram",                  # "ram" | "disk" | False
                                            # use "disk" if RAM < 16 GB

    # ── Grayscale augmentation ────────────────────────────────────
    # Fraction of each batch converted to grayscale (3-channel gray).
    # Teaches the model to rely on shape/texture, not color.
    # Very useful for IR cameras or day/night switching.
    # 0.0 = disabled, 0.3 = 30% of batch images affected
    "grayscale_p":  0.3,

    # ══════════════════════════════════════════════════════════════
    #  PHASE 1 — Backbone frozen, aggressive recall bias
    #  Goal: teach the detection head to find objects without
    #        destroying pretrained backbone features.
    # ══════════════════════════════════════════════════════════════
    "p1": {
        "epochs":           40,
        "freeze":           10,     # freeze layers 0-9 (backbone)
        "lr0":              0.005,  # higher OK — backbone is frozen
        "lrf":              0.1,    # final LR = lr0 × lrf
        "warmup_epochs":    3,
        "batch":            -1,     # auto-batch (fills ~60% VRAM)

        # Loss weights — biased toward recall (finding objects)
        # lower box/cls = model tries more proposals = higher recall
        "box":              4.0,    # default 7.5 — relaxed
        "cls":              0.3,    # default 0.5 — relaxed
        "dfl":              1.5,    # keep default

        # Augmentation — aggressive to expose small objects
        "mosaic":           1.0,
        "close_mosaic":     5,      # turn off mosaic last 5 epochs
        "copy_paste":       0.3,    # pastes objects onto other images — key for small obj
        "mixup":            0.15,   # blends 2 images
        "scale":            0.9,    # large scale variation → exposes small objects
        "hsv_h":            0.015,  # hue shift
        "hsv_s":            0.7,    # saturation variation
        "hsv_v":            0.4,    # brightness variation (exposure changes)
        "fliplr":           0.5,
        "flipud":           0.0,    # set 0.3 if objects can appear upside-down

        "patience":         15,     # early stopping
        "label_smoothing":  0.0,
        "dropout":          0.0,
    },

    # ══════════════════════════════════════════════════════════════
    #  PHASE 2 — Full model unfrozen, precision tightened
    #  Goal: fine-tune the whole network at target resolution,
    #        sharpen bounding boxes, improve mAP50-95.
    # ══════════════════════════════════════════════════════════════
    "p2": {
        "epochs":           70,
        "freeze":           None,   # unfreeze everything
        "lr0":              0.0005, # much lower — surgical fine-tune
        "lrf":              0.01,
        "warmup_epochs":    2,
        "batch":            -1,

        # Loss weights — back to balanced/default
        "box":              7.5,    # tighten box precision
        "cls":              0.5,
        "dfl":              1.5,

        # Augmentation — moderate
        "mosaic":           0.5,
        "close_mosaic":     15,     # disable mosaic last 15 epochs → stable convergence
        "copy_paste":       0.3,
        "mixup":            0.05,
        "scale":            0.5,
        "hsv_h":            0.015,
        "hsv_s":            0.7,
        "hsv_v":            0.4,
        "fliplr":           0.5,
        "flipud":           0.0,

        # Regularization
        "label_smoothing":  0.05,   # softens overconfident predictions
        "dropout":          0.1,    # dropout in detection head
        "patience":         20,
    },

    # ══════════════════════════════════════════════════════════════
    #  PHASE 3 — Recall surgery (OPTIONAL)
    #  Run only if recall is still below target after phase 2.
    #  Very low LR + very relaxed loss = final recall push.
    # ══════════════════════════════════════════════════════════════
    "p3": {
        "epochs":           30,
        "freeze":           None,
        "lr0":              0.0001, # near-zero LR
        "lrf":              0.01,
        "warmup_epochs":    1,
        "batch":            -1,

        # Loss — maximum recall bias
        "box":              3.5,    # very relaxed
        "cls":              0.2,    # minimal class penalty
        "dfl":              1.0,

        # Augmentation — minimal, let model consolidate
        "mosaic":           0.0,    # off
        "close_mosaic":     0,
        "copy_paste":       0.1,
        "mixup":            0.0,
        "scale":            0.3,
        "hsv_h":            0.01,
        "hsv_s":            0.4,
        "hsv_v":            0.4,
        "fliplr":           0.5,
        "flipud":           0.0,

        "label_smoothing":  0.0,
        "dropout":          0.0,
        "patience":         12,
    },
}

# ══════════════════════════════════════════════════════════════════
#  GRAYSCALE BATCH CALLBACK
# ══════════════════════════════════════════════════════════════════

def make_grayscale_callback(p: float):
    """
    Returns a callback that converts a fraction p of each training batch
    to grayscale (3-channel). Applied at batch level, no extra dependencies.
    """
    if p <= 0.0:
        return None

    def on_train_batch_start(trainer):
        if not hasattr(trainer, "batch") or "img" not in trainer.batch:
            return
        imgs = trainer.batch["img"]          # float tensor (B, 3, H, W), 0-1
        mask = torch.rand(imgs.shape[0], device=imgs.device) < p
        if mask.any():
            # Average RGB channels → luminance, expand back to 3 channels
            gray = imgs[mask].mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
            imgs[mask] = gray
        trainer.batch["img"] = imgs

    return on_train_batch_start


# ══════════════════════════════════════════════════════════════════
#  STATE PERSISTENCE
# ══════════════════════════════════════════════════════════════════

def state_path() -> Path:
    return Path(CFG["project"]) / "train_state.json"

def save_state(phase: str, weights: str):
    p = state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {"last_completed": phase, "weights": weights}
    with open(p, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n[state] saved → last_completed={phase}, weights={weights}")

def load_state() -> dict:
    p = state_path()
    if p.exists():
        with open(p) as f:
            state = json.load(f)
        print(f"[state] found existing state: {state}")
        return state
    return {"last_completed": None, "weights": CFG["base_model"]}


# ══════════════════════════════════════════════════════════════════
#  TRAINING HELPER
# ══════════════════════════════════════════════════════════════════

PHASE_ORDER = ["phase1", "phase2", "phase3"]

def run_phase(phase_name: str, weights: str, params: dict) -> str:
    """Run one training phase. Returns path to best.pt."""
    print(f"\n{'═' * 62}")
    print(f"  {phase_name.upper()}  |  weights: {weights}")
    print(f"{'═' * 62}\n")

    # Phase 1 starts from yaml+pretrained weights; later phases resume from .pt
    if weights.endswith(".yaml"):
        model = build_recall_model(weights, "yolov8s.pt")
    else:
        model = YOLO(weights)

    # Register grayscale callback if enabled
    gray_cb = make_grayscale_callback(CFG["grayscale_p"])
    if gray_cb:
        model.add_callback("on_train_batch_start", gray_cb)
        print(f"[grayscale] enabled — p={CFG['grayscale_p']}")

    model.train(
        trainer = RecallFocusedTrainer,   # ← recall loss + F2 fitness + extra checkpoints
        # ── Core ──────────────────────────────────────────────────
        data        = CFG["data"],
        project     = CFG["project"],
        name        = phase_name,
        exist_ok    = True,             # overwrite if re-running same phase
        imgsz       = CFG["imgsz"],
        device      = CFG["device"],
        workers     = CFG["workers"],
        cache       = CFG["cache"],

        # ── Optimizer & schedule ──────────────────────────────────
        optimizer       = "AdamW",
        cos_lr          = True,
        amp             = True,         # FP16 — 2× faster, same accuracy
        epochs          = params["epochs"],
        batch           = params.get("batch", -1),
        freeze          = params.get("freeze"),
        lr0             = params["lr0"],
        lrf             = params["lrf"],
        warmup_epochs   = params["warmup_epochs"],
        patience        = params["patience"],

        # ── Loss weights ──────────────────────────────────────────
        box     = params["box"],
        cls     = params["cls"],
        dfl     = params["dfl"],

        # ── Augmentation ──────────────────────────────────────────
        mosaic          = params["mosaic"],
        close_mosaic    = params["close_mosaic"],
        copy_paste      = params["copy_paste"],
        mixup           = params["mixup"],
        scale           = params["scale"],
        hsv_h           = params["hsv_h"],
        hsv_s           = params["hsv_s"],
        hsv_v           = params["hsv_v"],
        fliplr          = params.get("fliplr", 0.5),
        flipud          = params.get("flipud", 0.0),

        # ── Regularization ────────────────────────────────────────
        label_smoothing = params.get("label_smoothing", 0.0),
        dropout         = params.get("dropout", 0.0),

        # ── Output ────────────────────────────────────────────────
        val         = True,
        plots       = True,
        save_period = 10,               # checkpoint every 10 epochs
    )

    best = str(Path(CFG["project"]) / phase_name / "weights" / "best.pt")
    if not Path(best).exists():
        sys.exit(f"[error] Expected best.pt at {best} — training may have failed.")

    print(f"\n[{phase_name}] ✓ best weights → {best}")
    return best


# ══════════════════════════════════════════════════════════════════
#  VALIDATION + THRESHOLD SWEEP
#  No training — just finds the best conf/iou for deployment.
#  Output saved to {project}/threshold_sweep.json
# ══════════════════════════════════════════════════════════════════

def threshold_sweep(weights: str):
    print(f"\n{'═' * 62}")
    print("  THRESHOLD SWEEP  (no training — finds best conf & iou)")
    print(f"{'═' * 62}\n")
    print(f"  {'conf':>6}  {'iou':>6}  {'P':>7}  {'R':>7}  {'mAP50':>8}  {'mAP50-95':>10}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*10}")

    model  = YOLO(weights)
    rows   = []
    best_r = None

    for conf in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        for iou in [0.30, 0.45, 0.60]:
            m = model.val(
                data    = CFG["data"],
                imgsz   = CFG["imgsz"],
                conf    = conf,
                iou     = iou,
                device  = CFG["device"],
                verbose = False,
            )
            row = dict(
                conf     = conf,
                iou      = iou,
                P        = round(float(m.box.mp),    3),
                R        = round(float(m.box.mr),    3),
                mAP50    = round(float(m.box.map50), 3),
                mAP5095  = round(float(m.box.map),   3),
            )
            rows.append(row)

            print(f"  {conf:6.2f}  {iou:6.2f}  {row['P']:7.3f}  {row['R']:7.3f}"
                  f"  {row['mAP50']:8.3f}  {row['mAP5095']:10.3f}")

            if best_r is None or row["R"] > best_r["R"]:
                best_r = row

    out = Path(CFG["project"]) / "threshold_sweep.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\n  ✓ Results saved → {out}")
    print(f"\n  ★ Best recall config:")
    print(f"    conf={best_r['conf']}  iou={best_r['iou']}"
          f"  →  R={best_r['R']}  P={best_r['P']}  mAP50={best_r['mAP50']}")
    print(f"\n  → Use these values in your Hailo inference pipeline.")


# ══════════════════════════════════════════════════════════════════
#  EXPORT — ONNX float32 for Hailo Dataflow Compiler
# ══════════════════════════════════════════════════════════════════

def export_for_hailo(weights: str) -> str:
    print(f"\n{'═' * 62}")
    print("  EXPORT → ONNX float32  (Hailo quantizes externally)")
    print(f"{'═' * 62}\n")

    model = YOLO(weights)

    # Export with fixed shape — Hailo requires static input dimensions
    model.export(
        format   = "onnx",
        imgsz    = CFG["imgsz"],    # (1024, 576) — matches training resolution
        opset    = 11,              # Hailo Dataflow Compiler prefers opset 11
        simplify = True,            # onnx-simplifier: cleaner graph, fewer ops
        dynamic  = False,           # fixed batch=1, fixed H×W  ← required for Hailo
        half     = False,           # float32  — Hailo does int8 quantization itself
        device   = "cpu",           # CPU export = cleaner ONNX graph
    )

    onnx_path = str(Path(weights).with_suffix(".onnx"))
    print(f"\n  ✓ ONNX model → {onnx_path}")
    print(f"""
  ── Hailo next steps ─────────────────────────────────────────
  1. Parse + optimize:
       hailo parser onnx {onnx_path} --hw-arch hailo8

  2. Quantize (provide calibration images, ~100-300 frames):
       hailo optimize model.har \\
         --hw-arch hailo8 \\
         --calib-path calib_images/ \\
         --use-random-calib-set     # fallback if no calib images

  3. Compile:
       hailo compiler model.har --hw-arch hailo8

  4. Output: model.hef  →  deploy to Hailo8 runtime
  ─────────────────────────────────────────────────────────────
    """)
    return onnx_path


# ══════════════════════════════════════════════════════════════════
#  QUICK VALIDATION REPORT  (run at any time on any weights)
# ══════════════════════════════════════════════════════════════════

def quick_val(weights: str):
    print(f"\n[val] Running quick validation on {weights}")
    model = YOLO(weights)
    m = model.val(
        data   = CFG["data"],
        imgsz  = CFG["imgsz"],
        conf   = 0.001,         # very low — shows full model potential
        iou    = 0.6,
        device = CFG["device"],
    )
    print(f"""
  ── Validation results ───────────────────────────────────────
  Precision    : {m.box.mp:.4f}
  Recall       : {m.box.mr:.4f}   ← target > 0.80
  mAP50        : {m.box.map50:.4f}
  mAP50-95     : {m.box.map:.4f}  ← main accuracy metric

  Per-class mAP50:
    {dict(zip(m.names.values(), [round(x, 3) for x in m.box.maps]))}
  ─────────────────────────────────────────────────────────────
    """)


# ══════════════════════════════════════════════════════════════════
#  STATE MACHINE
# ══════════════════════════════════════════════════════════════════

def should_run(phase: str, last_completed: str | None, start_from: str | None) -> bool:
    """Decide if a phase should run based on state and CLI args."""
    order = PHASE_ORDER + ["export"]

    if start_from:
        # --from phaseX: skip everything before phaseX
        if order.index(phase) < order.index(start_from):
            return False

    if last_completed:
        # Skip already-completed phases unless --from forces a restart
        if order.index(phase) <= order.index(last_completed):
            if not start_from:
                return False

    return True


def main():
    parser = argparse.ArgumentParser(description="YOLOv8s-P2 Hailo Training State Machine")
    parser.add_argument("--from",    dest="start_from", default=None,
                        choices=["phase1", "phase2", "phase3", "export"],
                        help="Force restart from this phase (ignores saved state for this phase onward)")
    parser.add_argument("--only",    dest="only", default=None,
                        choices=["phase1", "phase2", "phase3", "export", "val", "sweep"],
                        help="Run only this single step")
    parser.add_argument("--weights", default=None,
                        help="Override weights path (useful with --only val/sweep/export)")
    parser.add_argument("--skip-p3", action="store_true",
                        help="Skip phase 3 (recall surgery) — use if recall is already good")
    args = parser.parse_args()

    state   = load_state()
    weights = args.weights or state["weights"]
    last    = state.get("last_completed")

    # ── --only shortcuts ──────────────────────────────────────────
    if args.only == "val":
        quick_val(weights); return
    if args.only == "sweep":
        threshold_sweep(weights); return
    if args.only == "export":
        export_for_hailo(weights); return
    if args.only in ["phase1", "phase2", "phase3"]:
        phase_cfg = {"phase1": CFG["p1"], "phase2": CFG["p2"], "phase3": CFG["p3"]}
        weights = run_phase(args.only, weights, phase_cfg[args.only])
        save_state(args.only, weights)
        quick_val(weights)
        return

    # ── Full state machine ────────────────────────────────────────

    # Phase 1 — Backbone frozen
    if should_run("phase1", last, args.start_from):
        weights = run_phase("phase1", weights, CFG["p1"])
        save_state("phase1", weights)
        quick_val(weights)
    else:
        print("[skip] phase1 — already completed")

    # Phase 2 — Full model
    if should_run("phase2", last, args.start_from):
        weights = run_phase("phase2", weights, CFG["p2"])
        save_state("phase2", weights)
        quick_val(weights)
    else:
        print("[skip] phase2 — already completed")

    # Phase 3 — Recall surgery (optional)
    if not args.skip_p3 and should_run("phase3", last, args.start_from):
        print("\n[phase3] Recall surgery — run 'quick_val' first to decide if needed.")
        print("         Skip with: python train_hailo.py --from export --skip-p3")
        weights = run_phase("phase3", weights, CFG["p3"])
        save_state("phase3", weights)
        quick_val(weights)
    else:
        print("[skip] phase3")

    # Threshold sweep
    threshold_sweep(weights)

    # Compare all checkpoints — pick best for Hailo export
    last_run_dir = Path(CFG["project"]) / PHASE_ORDER[PHASE_ORDER.index(
        state.get("last_completed", "phase1")
    )]
    compare_checkpoints(str(last_run_dir), CFG["data"],
                        imgsz=CFG["imgsz"], device=CFG["device"])

    # Export
    if should_run("export", last, args.start_from):
        export_for_hailo(weights)
        save_state("export", weights)

    print(f"\n{'═' * 62}")
    print("  ALL DONE")
    print(f"  Final weights : {weights}")
    print(f"  State file    : {state_path()}")
    print(f"{'═' * 62}\n")


if __name__ == "__main__":
    main()

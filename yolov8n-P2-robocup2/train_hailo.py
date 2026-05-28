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

# Patch check_amp to test AMP using the model already in memory instead of
# downloading yolo26n.pt (which ultralytics uses as its hardcoded test model).
def _patch_check_amp():
    import re
    import ultralytics.utils.checks as _checks
    from ultralytics.utils import ASSETS, LOGGER, colorstr
    from ultralytics.utils.torch_utils import autocast

    def check_amp(model):
        from ultralytics import YOLO as _YOLO
        device = next(model.parameters()).device
        prefix = colorstr("AMP: ")
        if device.type in {"cpu", "mps"}:
            return False
        pattern = re.compile(
            r"(nvidia|geforce|quadro|tesla).*?(1660|1650|1630|t400|t550|t600|t1000|t1200|t2000|k40m)",
            re.IGNORECASE,
        )
        if bool(pattern.search(torch.cuda.get_device_name(device))):
            LOGGER.warning(f"{prefix}checks failed ❌. AMP disabled for this GPU.")
            return False

        def amp_allclose(m, im):
            batch = [im] * 8
            imgsz = max(256, int(model.stride.max() * 4))
            a = m(batch, imgsz=imgsz, device=device, verbose=False)[0].boxes.data
            with autocast(enabled=True):
                b = m(batch, imgsz=imgsz, device=device, verbose=False)[0].boxes.data
            del m
            return a.shape == b.shape and torch.allclose(a, b.float(), atol=0.5)

        im = ASSETS / "bus.jpg"
        LOGGER.info(f"{prefix}running AMP checks with current model (no download)...")
        try:
            # Wrap the raw nn.Module in a callable that matches YOLO inference API
            wrapper = _YOLO.__new__(_YOLO)
            wrapper.model = model
            wrapper.task = "detect"
            assert amp_allclose(wrapper, im)
            LOGGER.info(f"{prefix}checks passed ✅")
        except Exception:
            LOGGER.warning(f"{prefix}checks skipped. If you see zero-mAP or NaN losses, use amp=False.")
        return True

    _checks.check_amp = check_amp
    # Patch the reference already bound in trainer.py's namespace at import time
    import ultralytics.engine.trainer as _trainer
    _trainer.check_amp = check_amp

_patch_check_amp()
from recall_trainer import (
    RecallFocusedTrainer,
    compare_checkpoints,
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
    "project":      str(Path(__file__).parent / "runs" / "hailo_train"),  # absolute so ultralytics doesn't prepend runs/detect/

    # -- Model -------------------------------------------------------------
    # Run yolov8n-p2_builder.py once to generate this file before training.
    "base_weights": MODEL_WEIGHTS,      # "yolov8n-p2.pt"

    # -- Resolution --------------------------------------------------------
    # 1024x576 = exact 16:9, both divisible by 32
    # Matches 1920x1080 camera with zero distortion / letterboxing
    "imgsz":        (1024, 576),

    # -- Hardware ----------------------------------------------------------
    "device":       "0",                # GPU id - use "cpu" if no GPU
    "workers":      4,                  # data loader threads (8 caused OOM on Windows spawn)
    "cache":        "disk",             # "ram" causes OOM with mosaic aug; disk is safe
    "batch":        8,                  # batch=8 safe for 1024x576 + mosaic on most GPUs
                                        # (-1 auto caused OOM: mosaic 4x images = 4x VRAM spike)
}


# ======================================================================
#  LOSS PRESETS
#
#  How each key affects training:
#
#  box   (default 7.5) — weight of the CIoU bounding-box regression loss.
#          Low  (4.0) → model learns "find objects first, boxes later" → more proposals → higher recall
#          High (7.5) → model is punished hard for sloppy boxes → better localization
#
#  cls   (default 0.5) — weight of the classification loss branch.
#          Low  (0.3) → model is lenient about uncertain class predictions → fires more detections
#          High (0.5) → stricter class identity required before firing
#
#  dfl   (default 1.5) — distribution focal loss for box edge refinement. Leave at 1.5.
#
#  gamma_neg (ASL, 2–6) — focal exponent that DOWN-WEIGHTS easy negatives (background).
#          High (4–6) → only very confident false positives get heavily penalized;
#                        low-confidence background preds are mostly ignored → model fires more → higher recall
#          Low  (2–3) → most false positives are penalized → fewer detections → higher precision
#          Rule: keep at 4 unless precision is the only goal.
#
#  gamma_pos (ASL, always 0) — focal exponent for true positives.
#          Must stay 0: any value > 0 would down-weight the gradient from real objects,
#          killing recall on rare/small objects like the ball.
#
#  clip  (ASL delta, 0.0–0.10) — probability margin below which a negative prediction
#          contributes ZERO loss. Think of it as a "free zone" for low-confidence detections.
#          High (0.05–0.08) → more free zone → model fires detections freely → higher recall
#          Low  (0.01–0.02) → almost every false positive is penalized → higher precision
#          Effect on recall is strong: +0.03 clip can shift recall by ~3–5 points.
#
#  class_recall_overrides — per-preset multiplier on the positive loss branch for
#          specific classes. Stacks ON TOP of recall_weight in loss_function.py.
#          Use for targeted surgery without touching the base config.
# ======================================================================

LOSS_PRESETS = {
    # ── recall_focused ────────────────────────────────────────────────────────
    # Used in: high_recall, color_invariance phases
    # Goal: maximize recall — model should fire detections freely.
    #
    # box=4.0  → half the default weight; model focuses on "finding" not "localizing"
    # cls=0.3  → lenient class identity; a detection doesn't need high class confidence
    # clip=0.05 → FP predictions below 5% confidence ignored; model isn't scared to fire
    # gamma_neg=4 → only very confident FPs (>5% after clip) get penalized
    "recall_focused": {
        "box":       4.0,
        "cls":       0.3,
        "dfl":       1.5,
        "gamma_neg": 4,    # punish only confident FPs — low-conf detections are free
        "gamma_pos": 0,    # NEVER suppress gradient from real objects
        "clip":      0.05, # free zone: FP predictions below 5% contribute zero loss
        "class_recall_overrides": {},
    },

    # ── precision ─────────────────────────────────────────────────────────────
    # Used in: precision phase
    # Goal: reduce ghost detections and tighten boxes, WITHOUT losing recall.
    #
    # box=7.5  → back to default; sloppy boxes are now penalized
    # cls=0.5  → full class discrimination required
    # clip=0.02 → almost all FPs penalized; the model must be more selective
    # gamma_neg=4 → kept high (not lowered to 3) so recall isn't sacrificed
    #               lowering to 3 during tests caused tag_rouge/humain recall to drop
    "precision": {
        "box":       7.5,
        "cls":       0.5,
        "dfl":       1.5,
        "gamma_neg": 4,    # kept at 4 — lowering to 3 hurts recall on weak classes
        "gamma_pos": 0,
        "clip":      0.02, # tight — penalize almost all FPs to reduce ghost detections
        "class_recall_overrides": {},
    },

    # ── class_surgery ─────────────────────────────────────────────────────────
    # Used in: class_surgery phase
    # Goal: push ballon recall/precision as high as possible.
    #
    # clip=0.06 → wider free zone — model can fire ballon detections freely
    # class_recall_overrides stacks on top of recall_weight in loss_function.py:
    #   ballon: base 3.0 × override 4.0 = effective weight 12.0
    "class_surgery": {
        "box":       6.0,
        "cls":       0.4,
        "dfl":       1.5,
        "gamma_neg": 4,
        "gamma_pos": 0,
        "clip":      0.06, # wider free zone — model can fire on ballon freely
        "class_recall_overrides": {
            1: 4.0,  # ballon: extra surgery boost on top of base recall_weight=3.0
        },
    },
}


# ======================================================================
#  AUGMENTATION PRESETS
#
#  How each key affects training:
#
#  mosaic (0.0–1.0) — probability of merging 4 images into one training sample.
#          Forces the model to detect objects at all scales and positions.
#          Key for small objects (ball, tags). Disable for last N epochs via
#          close_mosaic to let the model stabilize on clean single images.
#
#  copy_paste (0.0–0.5) — paste object instances from other images onto this one.
#          Especially effective for rare/small classes. Creates synthetic
#          examples of tag_rouge/humain appearing in diverse backgrounds.
#
#  scale (0.0–1.0) — random resize range. scale=0.9 means the image can be
#          resized to anywhere from 10% to 190% of original before cropping.
#          High values expose the model to very small objects (ball at distance).
#
#  mixup (0.0–0.3) — blend two images at alpha=0.5. Acts as a regularizer.
#          Too high disrupts convergence; 0.05–0.15 is the safe range.
#
#  close_mosaic (int) — disable mosaic for the last N epochs of the phase.
#          Mosaic creates "unnatural" composite images; the model needs a few
#          clean epochs to consolidate what it learned on mosaics.
#
#  hsv_h (0.0–0.05) — random hue rotation. Keep low (0.015); the ball is orange
#          and shouldn't shift to red/yellow in augmentation.
#
#  hsv_s (0.0–1.0) — random saturation scale. High = images can go near-grayscale
#          OR hyper-saturated. Critical for color invariance phase.
#
#  hsv_v (0.0–0.6) — random brightness scale. High = images can be very dark or
#          very bright. Simulates changing arena lighting.
#
#  grayscale_p (set per-phase, not here) — fraction of batch converted to
#          3-channel grayscale at the GPU tensor level. This is SEPARATE from
#          hsv_s=1.0 (which only desaturates, still uses RGB path). grayscale_p
#          forces the model through a true single-channel signal.
# ======================================================================

AUGMENTATION_PRESETS = {
    # ── aggressive ────────────────────────────────────────────────────────────
    # Used in: high_recall phase
    # Goal: maximum variety — expose the model to every object scale and position.
    # grayscale_p=0.2 (set in phase config) adds mild color invariance from day 1.
    "aggressive": {
        "mosaic":       1.0,   # every sample is a 4-image mosaic — max object density
        "copy_paste":   0.3,   # paste instances from other images — key for small/rare obj
        "scale":        0.9,   # extreme scale variation — exposes ball at all distances
        "mixup":        0.15,  # image blending — regularization, avoids overfitting
        "close_mosaic": 10,    # clean single images for last 10 epochs to stabilize
        "hsv_h":        0.015, # mild hue shift — keep ball color recognizable
        "hsv_s":        0.7,   # strong saturation variation — lighting robustness
        "hsv_v":        0.4,   # moderate brightness — indoor arena lighting range
        "fliplr":       0.5,   # 50% horizontal flip — symmetric field
    },

    # ── moderate ──────────────────────────────────────────────────────────────
    # Used in: precision phase
    # Goal: keep variety but reduce noise so the model can refine box quality.
    # Lower mosaic (0.5) and scale (0.5) mean cleaner, more natural samples.
    "moderate": {
        "mosaic":       0.5,   # 50% mosaic — balanced variety vs naturalness
        "copy_paste":   0.2,   # moderate copy-paste — still helps with rare classes
        "scale":        0.5,   # moderate scale — less extreme than aggressive
        "mixup":        0.05,  # minimal blending — keep gradients clean for box refinement
        "close_mosaic": 15,    # longer clean period — important for tight box convergence
        "hsv_h":        0.015,
        "hsv_s":        0.7,
        "hsv_v":        0.4,
        "fliplr":       0.5,
    },

    # ── minimal ───────────────────────────────────────────────────────────────
    # Used in: class_surgery phase
    # Goal: almost no augmentation — model must focus on the weak classes,
    # not adapt to new image variations at the same time.
    "minimal": {
        "mosaic":       0.0,   # off — surgery needs clean, realistic samples
        "copy_paste":   0.1,   # small amount — helps class_surgery see more tag/humain instances
        "scale":        0.2,   # very small scale jitter only
        "mixup":        0.0,   # off
        "close_mosaic": 0,
        "hsv_h":        0.01,
        "hsv_s":        0.4,
        "hsv_v":        0.3,
        "fliplr":       0.5,
    },

    # ── color_stress ──────────────────────────────────────────────────────────
    # Used in: color_invariance phase
    # Goal: force the model to detect by shape/texture, not color.
    # Combined with grayscale_p=0.85 (set in phase config), ~85% of batches
    # are full grayscale. The remaining 15% get extreme HSV variation.
    # This simulates: night IR cameras, overexposed white fields, underlit gyms.
    #
    # hsv_s=1.0 → saturation can swing from 0% (gray) to 200% (neon)
    # hsv_v=0.7 → brightness can swing from very dark to blown out
    # The combination with grayscale_p makes the color signal nearly useless —
    # the model MUST learn shape and texture cues to survive this phase.
    "color_stress": {
        "mosaic":       1.0,   # keep mosaic — still need object density
        "copy_paste":   0.3,   # keep copy-paste — rare classes need exposure
        "scale":        0.9,   # keep scale variation — small objects still important
        "mixup":        0.1,   # reduced from 0.15 — blending interferes with grayscale signal
        "close_mosaic": 10,
        "hsv_h":        0.015, # keep low — hue shift doesn't matter much for grayscale
        "hsv_s":        1.0,   # max saturation swing (was 0.9) — color is worthless as cue
        "hsv_v":        0.7,   # strong brightness swing (was 0.6) — simulate dark/bright arenas
        "fliplr":       0.5,
    },
}


# ======================================================================
#  PHASE CONFIGURATION
#
#  lr0            — starting learning rate for this phase
#                   Rule of thumb: halve it each phase as the model converges.
#                   Too high late in training → overshoots fine-tuned weights.
#                   Too low early → backbone never updates, recall stays low.
#
#  lrf            — final LR multiplier. Final LR = lr0 × lrf.
#                   With cos_lr=True the schedule is a cosine curve from lr0 → lr0×lrf.
#                   lrf=0.01 means the LR ends 100× lower than it started.
#
#  warmup_epochs  — linear LR ramp from near-zero to lr0 over N epochs.
#                   Protects backbone weights at the start of each phase.
#                   5 for early phases (big LR jump risk), 3 for late phases.
#
#  patience       — early stopping: halt if F2 fitness doesn't improve for N epochs.
#                   Generous (20) early — the model may plateau before breaking through.
#                   Tight (10) for surgery — if no movement in 10 epochs, it won't happen.
#
#  grayscale_p    — fraction of each GPU batch converted to 3-channel grayscale.
#                   Separate from hsv augmentation — this forces a truly colorless signal.
#                   0.0 = color only, 0.5 = half gray, 1.0 = always gray (too extreme).
#
#  To add a phase: append a dict, pick presets, re-run — completed phases are skipped.
#  To re-run a phase: python train_hailo.py --from <phase_name>
#  To run one phase: python train_hailo.py --only <phase_name>
# ======================================================================

PHASE_CONFIG = [
    # ── Phase 1: High Recall ──────────────────────────────────────────────────
    # COMPLETED: F2=0.826, R=0.811, ballon R=0.981 ✓
    #
    # Strategy:
    #   lr0=0.001 — full LR, model starts from pretrained weights so backbone
    #               can adapt quickly; warmup=5 prevents catastrophic forgetting
    #   box=4.0, cls=0.3 (recall_focused) — relaxed loss so model fires freely
    #   clip=0.05 — 5% free zone on FP; model isn't scared to propose detections
    #   grayscale_p=0.2 — mild color invariance from day 1; 20% of batches are gray
    #   mosaic+copy_paste (aggressive) — critical for ball/tag at all scales
    {
        "name":           "high_recall",
        "epochs":         60,              # 60 max; early stopping triggered ~epoch 40
        "loss_preset":    "recall_focused",
        "lr0":            0.001,           # full LR — backbone needs to adapt to 8 classes
        "lrf":            0.01,            # cosine decay to 1e-5 by end
        "warmup_epochs":  5,               # 5-epoch ramp — protects pretrained features
        "grayscale_p":    0.2,             # 20% gray — mild color stress from start
        "augmentation":   "aggressive",
        "patience":       20,              # generous — early recall plateau is normal
    },

    # ── Phase 2: Color Invariance ─────────────────────────────────────────────
    # COMPLETED: R=0.894, ballon R=0.981 ✓ | tag_rouge R=0.696 ✗ | humain R=0.709 ✗
    #
    # Strategy:
    #   lr0=0.0005 — half of phase 1; model is converged, fine-tune not relearn
    #   grayscale_p=0.85 — 85% of batches are full grayscale (raised from 0.7)
    #                       0.7 left too much color signal; model didn't fully commit
    #                       to shape cues. 0.85 is near-maximum useful pressure.
    #   hsv_s=1.0, hsv_v=0.7 (color_stress) — remaining 15% color batches are extreme
    #   Same recall_focused loss — not tightening precision here; recall is fragile
    #   Note: tag_rouge/humain weakness first appeared here; phase 4 will address them
    {
        "name":           "color_invariance",
        "epochs":         40,
        "loss_preset":    "recall_focused",
        "lr0":            0.0005,          # half of phase 1 — refining, not relearning
        "lrf":            0.01,
        "warmup_epochs":  5,               # still important — LR jump from phase 1 end
        "grayscale_p":    0.85,            # 85% gray (was 0.70) — model must rely on shape
        "augmentation":   "color_stress",  # extreme HSV on the remaining 15% color batches
        "patience":       15,
    },

    # ── Phase 3: Precision ────────────────────────────────────────────────────
    # Goal: reduce ghost detections and tighten bounding boxes.
    #
    # Strategy:
    #   lr0=0.0003 — slightly higher than naive 0.0002; the cosine schedule from
    #                phase 2 ended at ~5e-6, so 3e-4 gives a meaningful step without
    #                overshooting. Too low (2e-4) caused underfitting in earlier runs.
    #   box=7.5 (precision preset) — back to default; sloppy boxes are now penalized
    #   clip=0.02 — tight margin; almost all FP predictions penalized → fewer ghosts
    #   gamma_neg=4 — KEPT at 4 (not lowered to 3); lowering caused recall to drop
    #                 on tag_rouge/humain in testing. Precision comes from clip, not gamma.
    #   grayscale_p=0.2 — maintain color invariance, but not the heavy 85% stress
    #   warmup=3 — shorter; model is well-adapted, small LR ramp is enough
    {
        "name":           "precision",
        "epochs":         40,
        "loss_preset":    "precision",
        "lr0":            0.0003,          # slightly above naive 2e-4; avoid underfitting
        "lrf":            0.01,
        "warmup_epochs":  3,               # shorter warmup — model is already converged
        "grayscale_p":    0.2,             # maintain color invariance without stressing
        "augmentation":   "moderate",
        "patience":       15,
    },

    # ── Phase 4: Class Surgery ────────────────────────────────────────────────
    # Goal: push ballon recall and precision as high as possible.
    #
    # Strategy:
    #   lr0=5e-5 — very low; nudging the loss, not relearning.
    #   class_surgery preset — effective ballon weight = 3.0 (base) × 4.0 (override) = 12.0
    #              clip=0.06: model can fire ballon detections very freely
    #   minimal aug — surgery needs clean, realistic ball instances, not mosaics.
    #   patience=10 — tight; if ballon hasn't improved in 10 epochs, it won't.
    {
        "name":           "class_surgery",
        "epochs":         25,
        "loss_preset":    "class_surgery",
        "lr0":            0.00005,         # very low — surgical nudge, don't overwrite phase 3
        "lrf":            0.01,
        "warmup_epochs":  3,
        "grayscale_p":    0.1,             # minimal — weak classes need recognizable examples
        "augmentation":   "minimal",       # no mosaic — surgery needs clean images
        "patience":       10,              # tight — if it hasn't moved in 10 ep, increase LR
    },
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
    return {"last_completed": None, "weights": CFG["base_weights"]}


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


def run_phase(phase: dict, weights: str, test_mode: bool = False) -> str:
    """
    Run one training phase. Returns path to best.pt.
    test_mode=True: forces 1 epoch and disables mosaic-dependent settings
    so the full pipeline can be verified quickly without real training.
    """
    phase_name = phase["name"]
    loss_preset = resolve_loss_preset(phase["loss_preset"])
    aug_preset = resolve_augmentation_preset(phase["augmentation"])

    epochs = 1 if test_mode else phase["epochs"]
    # close_mosaic must be < epochs or ultralytics errors; in test mode just disable it
    close_mosaic = 0 if test_mode else aug_preset["close_mosaic"]

    print(f"\n{'=' * 70}")
    print(f"  PHASE: {phase_name.upper()}{' [TEST MODE — 1 epoch]' if test_mode else ''}")
    print(f"  loss_preset={phase['loss_preset']}  augmentation={phase['augmentation']}")
    print(f"  lr0={phase['lr0']}  epochs={epochs}  grayscale_p={phase['grayscale_p']}")
    print(f"  weights: {weights}")
    print(f"{'=' * 70}\n")

    model = YOLO(weights)

    gray_cb = make_grayscale_callback(phase["grayscale_p"])
    if gray_cb:
        model.add_callback("on_train_batch_start", gray_cb)
        print(f"[grayscale] p={phase['grayscale_p']}")

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
        epochs=epochs,
        batch=CFG["batch"],
        lr0=phase["lr0"],
        lrf=phase["lrf"],
        warmup_epochs=min(phase["warmup_epochs"], epochs),  # warmup can't exceed epochs
        patience=phase["patience"],

        # -- Loss weights (from preset) ------------------------------------
        box=loss_preset["box"],
        cls=loss_preset["cls"],
        dfl=loss_preset["dfl"],

        # -- ASL parameters → picked up by RecallFocusedTrainer -----------
        loss_preset_name=phase["loss_preset"],
        loss_preset_data=loss_preset,

        # -- Augmentation (from preset) ------------------------------------
        mosaic=aug_preset["mosaic"],
        close_mosaic=close_mosaic,
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
        verbose=False,         # suppress config dump + architecture table at startup
        plots=not test_mode,   # skip plot generation in test mode (saves time)
        save_period=-1 if test_mode else 10,
    )

    best = str(Path(CFG["project"]) / phase_name / "weights" / "best.pt")
    if not Path(best).exists():
        sys.exit(f"[error] Expected best.pt at {best} - training may have failed.")

    print(f"\n[{phase_name}] best.pt -> {best}")
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
    parser.add_argument("--test", action="store_true",
                        help="1 epoch per phase — verifies the full pipeline without real training")
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
        weights = run_phase(phase, weights, test_mode=args.test)
        save_state(phase["name"], weights)
        return

    # -- Full state machine ------------------------------------------------
    for phase in PHASE_CONFIG:
        phase_name = phase["name"]

        if phase_name in args.skip_phases:
            print(f"[skip] {phase_name} (--skip)")
            continue

        if should_run(phase_name, last, args.start_from):
            weights = run_phase(phase, weights, test_mode=args.test)
            save_state(phase_name, weights)
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

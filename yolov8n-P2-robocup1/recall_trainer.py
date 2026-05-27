"""
recall_trainer.py - Recall-focused YOLOv8n-P2 custom trainer

Drop-in replacement for the default DetectionTrainer with:
  1. PerClassAsymmetricLoss - per-class recall/cls weighting
  2. Per-class box weights  - applied to IoU regression loss
  3. F2 fitness             - best.pt saved by recall-weighted score
  4. Extra checkpoints      - best_recall.pt + best_f2.pt
  5. Grayscale callback     - batch-level color invariance augmentation

The trainer accepts a loss_preset dict (resolved from LOSS_PRESETS in
train_hailo.py) which controls ASL parameters and per-class overrides.
No ASL constants are hardcoded here - they come from the preset.

Usage in train_hailo.py:
    from recall_trainer import RecallFocusedTrainer, build_recall_model
    model = build_recall_model()
    model.train(
        data="data.yaml",
        trainer=RecallFocusedTrainer,
        loss_preset_name="recall_focused",
        loss_preset_data={...},
        ...
    )

Or standalone:
    python recall_trainer.py --data data.yaml --epochs 40
"""

import torch
import torch.nn as nn
from copy import deepcopy
from pathlib import Path

from torch.nn.parallel import DataParallel, DistributedDataParallel

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.loss import v8DetectionLoss, BboxLoss

from loss_function import (
    PerClassAsymmetricLoss,
    get_class_box_weights,
    get_loss_summary,
    NUM_CLASSES,
    CLASS_CONFIG,
    ASL_GAMMA_NEG,
    ASL_GAMMA_POS,
    ASL_CLIP,
)


def de_parallel(model):
    """Unwrap DataParallel/DistributedDataParallel to get the raw model."""
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        return model.module
    return model


# ======================================================================
#  MODEL CONFIGURATION
# ======================================================================

MODEL_YAML    = "yolov8n-p2.yaml"   # YOLOv8 nano with P2 head (stride 4)
MODEL_WEIGHTS = "yolov8n.pt"        # pretrained nano weights (transferred to P2 arch)


# ======================================================================
#  GRAYSCALE BATCH CALLBACK
#
#  Converts a fraction of each training batch to 3-channel grayscale.
#  Teaches the model to rely on shape/texture, not color.
#  Critical for cameras that switch day (color) -> night (IR/gray).
# ======================================================================

def make_grayscale_callback(p: float):
    """
    Returns a callback that converts fraction p of each batch to grayscale.
    Applied at batch level on GPU tensors. Returns None if p <= 0.

    Args:
        p: fraction of images per batch to convert (0.0-1.0)
    """
    if p <= 0.0:
        return None

    def on_train_batch_start(trainer):
        if not hasattr(trainer, "batch") or "img" not in trainer.batch:
            return
        imgs = trainer.batch["img"]          # float tensor (B, 3, H, W), 0-1
        mask = torch.rand(imgs.shape[0], device=imgs.device) < p
        if mask.any():
            gray = imgs[mask].mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
            imgs[mask] = gray
        trainer.batch["img"] = imgs

    return on_train_batch_start


# ======================================================================
#  PER-CLASS BBOX LOSS
#
#  Subclasses ultralytics BboxLoss to apply per-class box_weight.
#  Each foreground anchor's box loss is scaled by the weight of
#  its assigned class (from loss_function.CLASS_CONFIG).
# ======================================================================

class PerClassBboxLoss(BboxLoss):
    """
    BboxLoss with per-class weighting on the IoU regression loss.

    For each foreground anchor, the loss is multiplied by the
    box_weight of its assigned class. This allows relaxing box
    precision for classes where exact localization doesn't matter
    (e.g., large goals) while tightening it for critical small
    objects (e.g., the ball).
    """

    def __init__(self, reg_max):
        super().__init__(reg_max)
        self._box_weights = None

    def set_box_weights(self, device):
        """Load per-class box weights onto the correct device."""
        self._box_weights = get_class_box_weights(device)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, *args, **kwargs):
        """
        Same as parent BboxLoss.forward but with per-class box weight applied.
        """
        if self._box_weights is not None and fg_mask.any():
            anchor_classes = target_scores.detach().argmax(-1)[fg_mask]
            per_anchor_bw = self._box_weights[anchor_classes]
        else:
            per_anchor_bw = None

        loss_iou, loss_dfl = super().forward(
            pred_dist, pred_bboxes, anchor_points, target_bboxes,
            target_scores, target_scores_sum, fg_mask, *args, **kwargs
        )

        # Scale IoU loss by mean per-class box weight of foreground anchors
        if per_anchor_bw is not None and per_anchor_bw.numel() > 0:
            mean_box_w = per_anchor_bw.mean()
            loss_iou = loss_iou * mean_box_w

        return loss_iou, loss_dfl


# ======================================================================
#  RECALL-FOCUSED DETECTION LOSS
#
#  Subclasses v8DetectionLoss and swaps:
#    1. self.bce -> PerClassAsymmetricLoss (cls branch)
#    2. self.bbox_loss -> PerClassBboxLoss (box branch)
#
#  ASL parameters come from the loss_preset dict passed at construction.
# ======================================================================

class RecallFocusedDetectionLoss(v8DetectionLoss):
    """
    v8DetectionLoss with:
      - PerClassAsymmetricLoss replacing BCE for classification
      - PerClassBboxLoss for per-class box weight scaling
      - Configurable ASL parameters from loss preset
    """

    def __init__(self, model, loss_preset: dict = None):
        super().__init__(model)

        # Resolve ASL parameters from preset or fall back to loss_function defaults
        preset = loss_preset or {}
        gamma_neg = preset.get("gamma_neg", ASL_GAMMA_NEG)
        gamma_pos = preset.get("gamma_pos", ASL_GAMMA_POS)
        clip = preset.get("clip", ASL_CLIP)

        # Build class config with any per-class recall overrides from preset
        class_config = _build_class_config(preset.get("class_recall_overrides", {}))

        # Replace classification loss with per-class ASL
        self.bce = PerClassAsymmetricLoss(
            class_config=class_config,
            gamma_neg=gamma_neg,
            gamma_pos=gamma_pos,
            clip=clip,
            reduction="none",
        )

        # Replace box loss with per-class version
        self.bbox_loss = PerClassBboxLoss(self.reg_max)
        device = next(model.parameters()).device
        self.bbox_loss.set_box_weights(device)

        print(
            f"[RecallFocusedDetectionLoss] Active - "
            f"ASL(gamma_neg={gamma_neg}, gamma_pos={gamma_pos}, clip={clip}) "
            f"+ per-class box weights"
        )
        if preset.get("class_recall_overrides"):
            print(f"  class_recall_overrides: {preset['class_recall_overrides']}")


def _build_class_config(overrides: dict) -> dict:
    """
    Build a class config dict by starting from loss_function.CLASS_CONFIG
    and applying any per-class recall_weight overrides.

    Args:
        overrides: dict mapping class_id -> recall_weight value
                   e.g. {0: 2.5, 5: 2.0}
    """
    if not overrides:
        return CLASS_CONFIG

    config = {}
    for i in range(NUM_CLASSES):
        entry = dict(CLASS_CONFIG[i])
        if i in overrides:
            entry["recall_weight"] = overrides[i]
        config[i] = entry
    return config


# ======================================================================
#  F-BETA SCORE UTILITIES
# ======================================================================

def f_beta(precision: float, recall: float, beta: float = 2.0) -> float:
    """Fb score.  beta=2 = recall weighted 2x more than precision."""
    if precision + recall < 1e-9:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * (precision * recall) / (b2 * precision + recall)


# ======================================================================
#  RECALL-FOCUSED TRAINER
# ======================================================================

# Fitness parameters (not part of loss preset - always the same)
FITNESS_BETA = 2.0          # beta for Fb fitness.  2.0 = recall 2x more important.
FITNESS_WEIGHTS = {
    "f_beta":  0.6,         # Fb component (precision + recall)
    "map50":   0.3,         # mAP@50 - rewards finding objects
    "map5095": 0.1,         # mAP@50-95 - rewards box precision (low weight)
}


class RecallFocusedTrainer(DetectionTrainer):
    """
    DetectionTrainer with:
      1. RecallFocusedDetectionLoss (ASL + per-class weights)
      2. Fb-based fitness for best.pt selection
      3. Extra checkpoints: best_recall.pt + best_f2.pt
      4. Loss preset driven - no hardcoded ASL parameters

    Expects extra train() kwargs:
      loss_preset_name: str   - name of the active preset (for logging)
      loss_preset_data: dict  - resolved preset dict with ASL params
    """

    def __init__(self, *args, **kwargs):
        # Pop custom keys before ultralytics validates the kwargs dict
        self._loss_preset_name = kwargs.pop("loss_preset_name", None)
        self._loss_preset_data = kwargs.pop("loss_preset_data", None)
        # Also handle when passed through overrides dict (ultralytics >=8.4)
        overrides = kwargs.get("overrides") or (args[1] if len(args) > 1 else None) or {}
        if isinstance(overrides, dict):
            if self._loss_preset_name is None:
                self._loss_preset_name = overrides.pop("loss_preset_name", None)
            if self._loss_preset_data is None:
                self._loss_preset_data = overrides.pop("loss_preset_data", None)
        super().__init__(*args, **kwargs)
        self._best_recall = 0.0
        self._best_f2 = 0.0

    # -- 1. Swap loss function ---------------------------------------------
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        return model

    def _setup_train(self):
        super()._setup_train()

        # Retrieve loss preset stashed in __init__ before ultralytics validated args
        loss_preset_name = self._loss_preset_name or "default"
        loss_preset_data = self._loss_preset_data

        print(f"\n[RecallFocusedTrainer] Active preset: {loss_preset_name}")
        print(get_loss_summary())

        # Patch the loss after model is built
        self.loss_fn = RecallFocusedDetectionLoss(
            de_parallel(self.model),
            loss_preset=loss_preset_data,
        )
        self.compute_loss = self.loss_fn

    # -- 2. Fb-based fitness -----------------------------------------------
    def _metrics_to_prm(self, metrics: dict):
        """Extract (P, R, mAP50, mAP50-95) from the metrics dict returned by validate()."""
        P       = float(metrics.get("metrics/precision(B)", 0.0))
        R       = float(metrics.get("metrics/recall(B)",    0.0))
        map50   = float(metrics.get("metrics/mAP50(B)",     0.0))
        map5095 = float(metrics.get("metrics/mAP50-95(B)",  0.0))
        return P, R, map50, map5095

    def validate(self):
        """
        Override validate() to replace the default mAP-based fitness with F2.
        validate() now returns (metrics_dict, fitness_float).
        """
        metrics, _ = super().validate()
        if metrics is None:
            return None, None

        P, R, map50, map5095 = self._metrics_to_prm(metrics)
        fb = f_beta(P, R, beta=FITNESS_BETA)
        score = (
            FITNESS_WEIGHTS["f_beta"] * fb +
            FITNESS_WEIGHTS["map50"] * map50 +
            FITNESS_WEIGHTS["map5095"] * map5095
        )
        print(
            f"\n  [fitness] F{FITNESS_BETA}={fb:.4f}  P={P:.4f}  R={R:.4f}"
            f"  mAP50={map50:.4f}  mAP50-95={map5095:.4f}"
            f"  -> score={score:.4f}"
        )
        metrics["fitness"] = score
        return metrics, score

    # -- 3. Extra checkpoints ----------------------------------------------
    def save_metrics(self, metrics: dict):
        """Called after each validation. Save extra best checkpoints."""
        super().save_metrics(metrics)

        P, R, map50, map5095 = self._metrics_to_prm(metrics)
        fb = f_beta(P, R, beta=FITNESS_BETA)

        wdir = Path(self.save_dir) / "weights"
        wdir.mkdir(parents=True, exist_ok=True)

        if R > self._best_recall:
            self._best_recall = R
            ckpt_path = wdir / "best_recall.pt"
            self._save_ckpt(ckpt_path)
            print(f"  [checkpoint] best_recall.pt updated  R={R:.4f}")

        if fb > self._best_f2:
            self._best_f2 = fb
            ckpt_path = wdir / "best_f2.pt"
            self._save_ckpt(ckpt_path)
            print(f"  [checkpoint] best_f2.pt    updated  F{FITNESS_BETA}={fb:.4f}")

    def _save_ckpt(self, path: Path):
        """Save current model state to path."""
        ckpt = {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            "model": deepcopy(de_parallel(self.model)).half(),
            "ema": deepcopy(self.ema.ema).half(),
            "updates": self.ema.updates,
            "optimizer": None,
            "train_args": vars(self.args),
            "date": None,
        }
        torch.save(ckpt, path)


# ======================================================================
#  CONVENIENCE BUILDER
# ======================================================================

def build_recall_model(yaml_path: str = MODEL_YAML,
                       weights_path: str = MODEL_WEIGHTS) -> YOLO:
    """
    Build YOLOv8n-P2 model ready for recall-focused training.
    """
    model = YOLO(yaml_path).load(weights_path)
    print(
        f"[build_recall_model] Architecture: {yaml_path}  "
        f"Weights: {weights_path}\n"
        f"  Fitness : F{FITNESS_BETA} x {FITNESS_WEIGHTS['f_beta']} "
        f"+ mAP50 x {FITNESS_WEIGHTS['map50']} "
        f"+ mAP50-95 x {FITNESS_WEIGHTS['map5095']}"
    )
    return model


# ======================================================================
#  CHECKPOINT COMPARISON UTILITY
# ======================================================================

def compare_checkpoints(run_dir: str, data_yaml: str, imgsz=(1024, 576), device="0"):
    """
    Validate all checkpoints and print a comparison table.

        python recall_trainer.py --compare runs/hailo_train/precision --data data.yaml
    """
    run_dir = Path(run_dir)
    candidates = {
        "best.pt        (F2 fitness)": run_dir / "weights" / "best.pt",
        "best_recall.pt (max recall)": run_dir / "weights" / "best_recall.pt",
        "best_f2.pt     (max F2)    ": run_dir / "weights" / "best_f2.pt",
        "last.pt        (final ep.) ": run_dir / "weights" / "last.pt",
    }

    print(f"\n{'=' * 72}")
    print(f"  Checkpoint comparison - {run_dir}")
    print(f"{'=' * 72}")
    print(f"  {'Name':<34}  {'P':>6}  {'R':>6}  {'F2':>6}  {'mAP50':>7}  {'mAP50-95':>9}")
    print(f"  {'-'*34}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*9}")

    rows = []
    for label, pt in candidates.items():
        if not pt.exists():
            print(f"  {label:<34}  (not found)")
            continue
        m_obj = YOLO(str(pt)).val(
            data=data_yaml, imgsz=imgsz, conf=0.001, iou=0.6,
            device=device, verbose=False
        )
        P, R = float(m_obj.box.mp), float(m_obj.box.mr)
        map50, map5095 = float(m_obj.box.map50), float(m_obj.box.map)
        fb = f_beta(P, R, beta=FITNESS_BETA)
        rows.append((label, P, R, fb, map50, map5095, str(pt)))
        print(f"  {label:<34}  {P:6.3f}  {R:6.3f}  {fb:6.3f}  {map50:7.3f}  {map5095:9.3f}")

    if rows:
        best = max(rows, key=lambda x: x[3])
        print(f"\n  * Recommended for Hailo export (highest F2):")
        print(f"    {best[0].strip()}  ->  {best[6]}")
    print()


# ======================================================================
#  STANDALONE CLI
# ======================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, nargs=2, default=[1024, 576])
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/recall_test")
    parser.add_argument("--compare", default=None,
                        help="Path to a run dir - compare checkpoints and exit")
    args = parser.parse_args()

    if args.compare:
        compare_checkpoints(args.compare, args.data,
                            imgsz=tuple(args.imgsz), device=args.device)
    else:
        model = build_recall_model()
        model.train(
            data=args.data,
            trainer=RecallFocusedTrainer,
            epochs=args.epochs,
            imgsz=tuple(args.imgsz),
            device=args.device,
            project=args.project,
            name="recall_focused",
            batch=-1,
            amp=True,
            patience=20,
            plots=True,
            warmup_epochs=5,
            loss_preset_name="recall_focused",
            loss_preset_data={
                "box": 4.0, "cls": 0.3, "dfl": 1.5,
                "gamma_neg": 4, "gamma_pos": 0, "clip": 0.05,
                "class_recall_overrides": {},
            },
        )

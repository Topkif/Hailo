"""
recall_trainer.py — Custom trainer for recall-focused YOLOv8n-P2 training.

What it does on top of standard ultralytics DetectionTrainer:
  1. Replaces BCE with PerClassAsymmetricLoss (from loss_function.py)
  2. Saves best.pt by F2 fitness (recall weighted 2×) instead of mAP
  3. Suppresses ultralytics noise (config dump, architecture table, val class table)
  4. Supports grayscale batch augmentation via make_grayscale_callback()
"""

import torch
from copy import deepcopy
from pathlib import Path

from torch.nn.parallel import DataParallel, DistributedDataParallel
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.loss import v8DetectionLoss

from loss_function import (
    PerClassAsymmetricLoss,
    get_loss_summary,
    NUM_CLASSES,
    CLASS_CONFIG,
    ASL_GAMMA_NEG,
    ASL_GAMMA_POS,
    ASL_CLIP,
)

MODEL_WEIGHTS = "yolov8n-p2.pt"

FITNESS_BETA = 2.0
FITNESS_WEIGHTS = {"f_beta": 0.6, "map50": 0.3, "map5095": 0.1}


def de_parallel(model):
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        return model.module
    return model


def f_beta(p: float, r: float, beta: float = 2.0) -> float:
    if p + r < 1e-9:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * p * r / (b2 * p + r)


# ── Grayscale callback ────────────────────────────────────────────────────────

def make_grayscale_callback(p: float):
    """Convert fraction p of each training batch to 3-channel grayscale (on GPU)."""
    if p <= 0.0:
        return None

    def on_train_batch_start(trainer):
        if not hasattr(trainer, "batch") or "img" not in trainer.batch:
            return
        imgs = trainer.batch["img"]
        mask = torch.rand(imgs.shape[0], device=imgs.device) < p
        if mask.any():
            gray = imgs[mask].mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
            imgs[mask] = gray
        trainer.batch["img"] = imgs

    return on_train_batch_start


# ── ASL loss wrapper ──────────────────────────────────────────────────────────

class _ASLDetectionLoss(v8DetectionLoss):
    """v8DetectionLoss with PerClassAsymmetricLoss replacing BCE."""

    def __init__(self, model, loss_preset: dict = None):
        super().__init__(model)
        preset = loss_preset or {}

        # Build class config with any per-class recall overrides from the preset
        class_config = CLASS_CONFIG
        overrides = preset.get("class_recall_overrides", {})
        if overrides:
            class_config = {
                i: {**CLASS_CONFIG[i], "recall_weight": overrides[i]}
                if i in overrides else CLASS_CONFIG[i]
                for i in range(NUM_CLASSES)
            }

        self.bce = PerClassAsymmetricLoss(
            class_config=class_config,
            gamma_neg=preset.get("gamma_neg", ASL_GAMMA_NEG),
            gamma_pos=preset.get("gamma_pos", ASL_GAMMA_POS),
            clip=preset.get("clip", ASL_CLIP),
            reduction="none",
        )


# ── Trainer ───────────────────────────────────────────────────────────────────

class RecallFocusedTrainer(DetectionTrainer):
    """
    DetectionTrainer with F2 fitness and PerClassAsymmetricLoss.

    Pass extra kwargs to model.train():
      loss_preset_name: str   — name for logging
      loss_preset_data: dict  — preset dict with gamma_neg/clip/class_recall_overrides
    """

    def __init__(self, *args, **kwargs):
        # Pop custom keys before ultralytics validates kwargs
        self._preset_name = kwargs.pop("loss_preset_name", None)
        self._preset_data = kwargs.pop("loss_preset_data", None)
        overrides = kwargs.get("overrides") or (args[1] if len(args) > 1 else None) or {}
        if isinstance(overrides, dict):
            if self._preset_name is None:
                self._preset_name = overrides.pop("loss_preset_name", None)
            if self._preset_data is None:
                self._preset_data = overrides.pop("loss_preset_data", None)
        super().__init__(*args, **kwargs)
        self._best_f2 = 0.0

    def _setup_train(self):
        # Suppress the ultralytics config dump and architecture table
        self.args.verbose = False
        super()._setup_train()
        print(f"\n[RecallFocusedTrainer] preset={self._preset_name or 'default'}")
        print(get_loss_summary())
        loss = _ASLDetectionLoss(de_parallel(self.model), self._preset_data)
        self.loss_fn = loss
        self.compute_loss = loss

    def _get_prm(self, metrics: dict):
        P       = float(metrics.get("metrics/precision(B)", 0.0))
        R       = float(metrics.get("metrics/recall(B)",    0.0))
        map50   = float(metrics.get("metrics/mAP50(B)",     0.0))
        map5095 = float(metrics.get("metrics/mAP50-95(B)",  0.0))
        return P, R, map50, map5095

    def validate(self):
        # Suppress per-class table ultralytics prints during training validation
        self.args.verbose = False
        result = super().validate()
        metrics = result[0] if isinstance(result, tuple) else result
        if metrics is None:
            return None, None
        P, R, map50, map5095 = self._get_prm(metrics)
        fb = f_beta(P, R, FITNESS_BETA)
        score = (
            FITNESS_WEIGHTS["f_beta"]   * fb +
            FITNESS_WEIGHTS["map50"]    * map50 +
            FITNESS_WEIGHTS["map5095"]  * map5095
        )
        print(f"\n  [F2-fitness] F2={fb:.4f}  P={P:.4f}  R={R:.4f}"
              f"  mAP50={map50:.4f}  mAP50-95={map5095:.4f}  score={score:.4f}")

        # Per-class recall/precision from the validator object
        try:
            box = self.validator.metrics.box
            names = self.validator.names  # {0: "robot", 1: "ballon", ...}
            p_arr = box.p.tolist() if hasattr(box.p, "tolist") else list(box.p)
            r_arr = box.r.tolist() if hasattr(box.r, "tolist") else list(box.r)
            print(f"\n  {'Class':<14} {'P':>7} {'R':>7}")
            print(f"  {'-'*14} {'-'*7} {'-'*7}")
            for i in sorted(names):
                if i < len(p_arr):
                    print(f"  {names[i]:<14} {p_arr[i]:7.3f} {r_arr[i]:7.3f}")
        except Exception:
            pass

        metrics["fitness"] = score
        return metrics, score

    def save_metrics(self, metrics: dict):
        super().save_metrics(metrics)
        P, R, map50, map5095 = self._get_prm(metrics)
        fb = f_beta(P, R, FITNESS_BETA)
        wdir = Path(self.save_dir) / "weights"
        wdir.mkdir(parents=True, exist_ok=True)

        if fb > self._best_f2:
            self._best_f2 = fb
            self._save_ckpt(wdir / "best.pt")
            print(f"  [ckpt] best.pt  F2={fb:.4f}  P={P:.4f}  R={R:.4f}")

    def _save_ckpt(self, path: Path):
        torch.save({
            "epoch":        self.epoch,
            "best_fitness": self.best_fitness,
            "model":        deepcopy(de_parallel(self.model)).half(),
            "ema":          deepcopy(self.ema.ema).half(),
            "updates":      self.ema.updates,
            "optimizer":    None,
            "train_args":   vars(self.args),
            "date":         None,
        }, path)


# ── Checkpoint comparison utility ─────────────────────────────────────────────

def compare_checkpoints(run_dir: str, data_yaml: str, imgsz=(1024, 576), device="0"):
    run_dir = Path(run_dir)
    candidates = {
        "best.pt  (F2 fitness)": run_dir / "weights" / "best.pt",
        "last.pt  (final ep.) ": run_dir / "weights" / "last.pt",
    }
    print(f"\n{'='*70}")
    print(f"  Checkpoint comparison — {run_dir}")
    print(f"{'='*70}")
    print(f"  {'Name':<30}  {'P':>6}  {'R':>6}  {'F2':>6}  {'mAP50':>7}  {'mAP50-95':>9}")
    print(f"  {'-'*30}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*9}")
    rows = []
    for label, pt in candidates.items():
        if not pt.exists():
            print(f"  {label:<30}  (not found)")
            continue
        m = YOLO(str(pt)).val(data=data_yaml, imgsz=imgsz, conf=0.001, iou=0.6,
                              device=device, verbose=False)
        P, R = float(m.box.mp), float(m.box.mr)
        map50, map5095 = float(m.box.map50), float(m.box.map)
        fb = f_beta(P, R, FITNESS_BETA)
        rows.append((label, P, R, fb, map50, map5095, str(pt)))
        print(f"  {label:<30}  {P:6.3f}  {R:6.3f}  {fb:6.3f}  {map50:7.3f}  {map5095:9.3f}")
    if rows:
        best = max(rows, key=lambda x: x[3])
        print(f"\n  * Best for export (highest F2): {best[0].strip()} → {best[6]}")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--compare", default=None)
    parser.add_argument("--imgsz", type=int, nargs=2, default=[1024, 576])
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if args.compare:
        compare_checkpoints(args.compare, args.data, tuple(args.imgsz), args.device)

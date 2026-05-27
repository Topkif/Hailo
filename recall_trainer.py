"""
╔══════════════════════════════════════════════════════════════════╗
║  recall_trainer.py — Recall-focused YOLOv8 custom trainer       ║
║                                                                  ║
║  Drop-in replacement for the default DetectionTrainer with:     ║
║  1. AsymmetricLoss  — penalizes missed objects more than FP     ║
║  2. F2 fitness      — best.pt saved by recall-weighted score    ║
║  3. best_recall.pt  — extra checkpoint: highest raw recall      ║
╚══════════════════════════════════════════════════════════════════╝

Usage in train_hailo.py  (replace the YOLO().train() calls):

    from recall_trainer import build_recall_model
    model = build_recall_model("yolov8s-p2.yaml", "yolov8s.pt")
    model.train(data="data.yaml", trainer=RecallFocusedTrainer, ...)

Or standalone:
    python recall_trainer.py --data data.yaml --epochs 40
"""

import math
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from pathlib import Path
from typing import Tuple

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.metrics import DetMetrics
from ultralytics.utils.torch_utils import de_parallel


# ══════════════════════════════════════════════════════════════════
#  ASYMMETRIC LOSS
#
#  Replaces BCEWithLogitsLoss in the classification branch.
#
#  Key parameters:
#    gamma_neg  (default 4) — focal exponent for negatives (FP).
#                             Higher = harder penalty for confident FP.
#                             Range: 2–6. Start at 4.
#
#    gamma_pos  (default 0) — focal exponent for positives (TP/FN).
#                             0 = no dampening of positive samples.
#                             Keep at 0 to never discount real objects.
#
#    clip       (default 0.05) — probability margin δ.
#                             Negative predictions are clipped to
#                             max(0, p - δ) before loss is computed.
#                             Soft negatives (p < δ) contribute zero loss
#                             → model is free to fire more detections
#                             → higher recall.
#                             Range: 0.0–0.1. Start at 0.05.
#
#  Effect on your metrics:
#    Recall    ↑  (fewer false negatives penalized)
#    Precision ↑  (hard false positives still penalized by gamma_neg)
#    mAP50     ↑ or ≈ (better detections overall)
#    mAP50-95  ≈ (box precision unchanged — this only touches cls)
# ══════════════════════════════════════════════════════════════════

class AsymmetricLoss(nn.Module):
    """
    ASL: Asymmetric Loss for multi-label / object detection classification.
    Reference: "Asymmetric Loss For Multi-Label Classification" (Ben-Baruch et al., 2021)

    Replaces BCEWithLogitsLoss in v8DetectionLoss.cls branch.
    """
    def __init__(
        self,
        gamma_neg: float = 4.0,   # focal exp for negatives  — tune: 2 (mild) → 6 (aggressive)
        gamma_pos: float = 0.0,   # focal exp for positives  — keep 0 to never suppress TP
        clip: float      = 0.05,  # probability margin δ      — tune: 0.0 → 0.1
        reduction: str   = "none" # keep "none" — v8 applies its own reduction
    ):
        super().__init__()
        assert reduction == "none", "v8DetectionLoss expects reduction='none'"
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip      = clip

    def forward(self, pred_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        pred_logits : raw logits  (B, num_classes) — before sigmoid
        targets     : soft labels (B, num_classes) — 0.0 or IoU score
        returns     : per-element loss, same shape as inputs
        """
        p = torch.sigmoid(pred_logits)          # predicted probability

        # ── Positive branch (where target > 0) ─────────────────
        loss_pos = targets * torch.log(p.clamp(min=1e-8))
        if self.gamma_pos > 0:
            focal_pos = (1.0 - p) ** self.gamma_pos
            loss_pos  = loss_pos * focal_pos

        # ── Negative branch (where target == 0) ────────────────
        # Clip: shift probabilities down by δ — soft negatives vanish
        p_neg = p if self.clip == 0 else (p - self.clip).clamp(min=0.0)
        loss_neg = (1.0 - targets) * torch.log((1.0 - p_neg).clamp(min=1e-8))
        if self.gamma_neg > 0:
            focal_neg = p_neg ** self.gamma_neg
            loss_neg  = loss_neg * focal_neg

        return -(loss_pos + loss_neg)           # shape: (B, num_classes), reduction="none"


# ══════════════════════════════════════════════════════════════════
#  RECALL-FOCUSED DETECTION LOSS
#  Subclasses v8DetectionLoss and swaps the cls criterion.
# ══════════════════════════════════════════════════════════════════

class RecallFocusedDetectionLoss(v8DetectionLoss):
    """
    v8DetectionLoss with AsymmetricLoss replacing BCE for classification.

    All box / DFL loss is unchanged — only the cls branch is affected.
    """

    def __init__(self, model, asl_gamma_neg=4.0, asl_gamma_pos=0.0, asl_clip=0.05):
        super().__init__(model)
        # Replace the default BCEWithLogitsLoss with ASL
        self.bce = AsymmetricLoss(
            gamma_neg = asl_gamma_neg,
            gamma_pos = asl_gamma_pos,
            clip      = asl_clip,
            reduction = "none",
        )
        print(
            f"[RecallFocusedDetectionLoss] AsymmetricLoss active — "
            f"γ_neg={asl_gamma_neg}, γ_pos={asl_gamma_pos}, clip={asl_clip}"
        )


# ══════════════════════════════════════════════════════════════════
#  F-BETA SCORE UTILITIES
# ══════════════════════════════════════════════════════════════════

def f_beta(precision: float, recall: float, beta: float = 2.0) -> float:
    """
    Fβ score.
    β=2  → recall weighted 2× more than precision  (recommended)
    β=1  → F1, balanced
    β=0.5→ precision weighted 2× more
    """
    if precision + recall < 1e-9:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * (precision * recall) / (b2 * precision + recall)


# ══════════════════════════════════════════════════════════════════
#  RECALL-FOCUSED TRAINER
# ══════════════════════════════════════════════════════════════════

# ── Tunable constants ─────────────────────────────────────────────
FITNESS_BETA      = 2.0   # β for Fβ fitness.  2.0 = recall 2× more important.
                          # Change to 1.0 for balanced, 3.0 for extreme recall focus.

FITNESS_WEIGHTS   = {     # Additional mAP terms blended into fitness.
    "f_beta":   0.6,      # Fβ component (precision + recall)
    "map50":    0.3,      # mAP@50 — rewards finding objects
    "map5095":  0.1,      # mAP@50-95 — rewards box precision (low weight)
}

# ASL parameters (forwarded to loss)
ASL_GAMMA_NEG = 4.0
ASL_GAMMA_POS = 0.0
ASL_CLIP      = 0.05
# ─────────────────────────────────────────────────────────────────


class RecallFocusedTrainer(DetectionTrainer):
    """
    DetectionTrainer with:
      1. RecallFocusedDetectionLoss (AsymmetricLoss for cls)
      2. Fβ-based fitness for best.pt selection
      3. Extra checkpoint: best_recall.pt (highest raw recall)
      4. Extra checkpoint: best_f2.pt     (highest F2, redundant with best.pt here)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._best_recall  = 0.0
        self._best_f2      = 0.0

    # ── 1. Swap loss function ─────────────────────────────────────
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        return model

    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        # Patch the loss after model is built
        self.loss_fn = RecallFocusedDetectionLoss(
            de_parallel(self.model),
            asl_gamma_neg = ASL_GAMMA_NEG,
            asl_gamma_pos = ASL_GAMMA_POS,
            asl_clip      = ASL_CLIP,
        )
        self.compute_loss = self.loss_fn

    # ── 2. Fβ-based fitness ───────────────────────────────────────
    def fitness(self) -> float:
        """
        Default ultralytics fitness:
            w = [0.0, 0.0, 0.1, 0.9] × [P, R, mAP50, mAP50-95]
            → heavily mAP50-95 focused, recall weight = 0

        Ours:
            fitness = 0.6 × Fβ(P,R)  +  0.3 × mAP50  +  0.1 × mAP50-95
            → recall-weighted via Fβ, still checks box quality via mAP
        """
        results = self.metrics.mean_results()   # [P, R, mAP50, mAP50-95]
        P, R, map50, map5095 = results[0], results[1], results[2], results[3]

        fb    = f_beta(P, R, beta=FITNESS_BETA)
        score = (
            FITNESS_WEIGHTS["f_beta"]   * fb     +
            FITNESS_WEIGHTS["map50"]    * map50  +
            FITNESS_WEIGHTS["map5095"]  * map5095
        )

        # Log to console so you can track it
        print(
            f"\n  [fitness] F{FITNESS_BETA}={fb:.4f}  P={P:.4f}  R={R:.4f}"
            f"  mAP50={map50:.4f}  mAP50-95={map5095:.4f}"
            f"  → score={score:.4f}"
        )
        return score

    # ── 3. Extra checkpoints ──────────────────────────────────────
    def save_metrics(self, metrics: dict):
        """Called after each validation. Save extra best checkpoints."""
        super().save_metrics(metrics)

        results = self.metrics.mean_results()
        P, R, map50, map5095 = results[0], results[1], results[2], results[3]
        fb = f_beta(P, R, beta=FITNESS_BETA)

        wdir = Path(self.save_dir) / "weights"
        wdir.mkdir(parents=True, exist_ok=True)

        # best_recall.pt — saved whenever raw recall improves
        if R > self._best_recall:
            self._best_recall = R
            ckpt_path = wdir / "best_recall.pt"
            self._save_ckpt(ckpt_path)
            print(f"  [checkpoint] best_recall.pt updated  R={R:.4f}")

        # best_f2.pt — saved whenever F2 improves (same as best.pt here,
        #              but kept separate so you can compare with mAP-best)
        if fb > self._best_f2:
            self._best_f2 = fb
            ckpt_path = wdir / "best_f2.pt"
            self._save_ckpt(ckpt_path)
            print(f"  [checkpoint] best_f2.pt    updated  F{FITNESS_BETA}={fb:.4f}")

    def _save_ckpt(self, path: Path):
        """Save current model state to path."""
        import torch
        ckpt = {
            "epoch":        self.epoch,
            "best_fitness": self.best_fitness,
            "model":        deepcopy(de_parallel(self.model)).half(),
            "ema":          deepcopy(self.ema.ema).half(),
            "updates":      self.ema.updates,
            "optimizer":    None,           # not needed for inference
            "train_args":   vars(self.args),
            "date":         None,
        }
        torch.save(ckpt, path)


# ══════════════════════════════════════════════════════════════════
#  CONVENIENCE BUILDER
#  Returns a YOLO model configured to use RecallFocusedTrainer
# ══════════════════════════════════════════════════════════════════

def build_recall_model(yaml_path: str = "yolov8s-p2.yaml",
                       weights_path: str = "yolov8s.pt") -> YOLO:
    """
    Build YOLOv8s-P2 with recall-focused training.

    Usage:
        model = build_recall_model()
        model.train(
            data    = "data.yaml",
            trainer = RecallFocusedTrainer,
            epochs  = 40,
            ...
        )
    """
    model = YOLO(yaml_path).load(weights_path)
    print(
        f"[build_recall_model] Architecture: {yaml_path}  "
        f"Weights: {weights_path}\n"
        f"  Loss    : AsymmetricLoss (γ_neg={ASL_GAMMA_NEG}, clip={ASL_CLIP})\n"
        f"  Fitness : F{FITNESS_BETA} × {FITNESS_WEIGHTS['f_beta']} "
        f"+ mAP50 × {FITNESS_WEIGHTS['map50']} "
        f"+ mAP50-95 × {FITNESS_WEIGHTS['map5095']}"
    )
    return model


# ══════════════════════════════════════════════════════════════════
#  CHECKPOINT COMPARISON UTILITY
#  Run after training to pick the right .pt for Hailo export
# ══════════════════════════════════════════════════════════════════

def compare_checkpoints(run_dir: str, data_yaml: str, imgsz=(1024, 576), device="0"):
    """
    Validate all three checkpoints and print a comparison table.

        python recall_trainer.py --compare runs/hailo_train/phase2 --data data.yaml
    """
    run_dir = Path(run_dir)
    candidates = {
        "best.pt        (F2 fitness)": run_dir / "weights" / "best.pt",
        "best_recall.pt (max recall)": run_dir / "weights" / "best_recall.pt",
        "best_f2.pt     (max F2)    ": run_dir / "weights" / "best_f2.pt",
        "last.pt        (final ep.) ": run_dir / "weights" / "last.pt",
    }

    print(f"\n{'═'*72}")
    print(f"  Checkpoint comparison — {run_dir}")
    print(f"{'═'*72}")
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
        best = max(rows, key=lambda x: x[3])   # highest F2
        print(f"\n  ★ Recommended for Hailo export (highest F2):")
        print(f"    {best[0].strip()}  →  {best[6]}")
    print()


# ══════════════════════════════════════════════════════════════════
#  STANDALONE CLI
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data",     default="data.yaml")
    parser.add_argument("--epochs",   type=int, default=40)
    parser.add_argument("--imgsz",    type=int, nargs=2, default=[1024, 576])
    parser.add_argument("--device",   default="0")
    parser.add_argument("--project",  default="runs/recall_test")
    parser.add_argument("--compare",  default=None,
                        help="Path to a run dir — compare checkpoints and exit")
    args = parser.parse_args()

    if args.compare:
        compare_checkpoints(args.compare, args.data,
                            imgsz=tuple(args.imgsz), device=args.device)
    else:
        model = build_recall_model()
        model.train(
            data     = args.data,
            trainer  = RecallFocusedTrainer,
            epochs   = args.epochs,
            imgsz    = tuple(args.imgsz),
            device   = args.device,
            project  = args.project,
            name     = "recall_focused",
            batch    = -1,
            amp      = True,
            patience = 20,
            plots    = True,
        )

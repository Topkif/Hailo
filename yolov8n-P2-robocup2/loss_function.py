"""
loss_function.py — Per-class Asymmetric Loss for YOLOv8n-P2.
Single source of truth for all per-class loss weights.

  recall_weight : cost of MISSING the object (positive branch only)
                  Higher = model tries harder to find this class → higher recall
  cls_weight    : cost of misclassifying this class (both branches)
                  Lower = confusion with similar classes is tolerated
"""

import torch
import torch.nn as nn


# ── Global ASL parameters ─────────────────────────────────────────────────────
# Applied across all classes. Per-class tuning done in CLASS_CONFIG below.

ASL_GAMMA_NEG = 4.0   # focal exponent for false positives (2=mild, 6=aggressive)
ASL_GAMMA_POS = 0.0   # focal exponent for true positives — keep 0, never suppress real objects
ASL_CLIP      = 0.05  # probability margin: soft negatives below this threshold contribute zero loss


# ── Per-class configuration ───────────────────────────────────────────────────
# Edit these weights to tune recall/precision per class.
# Results after color_invariance phase (conf=0.20):
#   robot R=0.946  ballon R=0.981  but R=0.969  poteau R=0.960
#   tag_bleu R=0.889  tag_rouge R=0.696 ← weak  robot_rct R=1.000  humain R=0.709 ← weak

NUM_CLASSES = 8

CLASS_NAMES = [
    "robot",     # 0
    "ballon",    # 1
    "but",       # 2
    "poteau",    # 3
    "tag_bleu",  # 4
    "tag_rouge", # 5
    "robot_rct", # 6
    "humain",    # 7
]

CLASS_CONFIG = {
    0: {"name": "robot",     "recall_weight": 1.0, "cls_weight": 1.0},
    1: {
        "name":          "ballon",
        "recall_weight": 3.0,   # FOCUS — maximize ball recall and precision
        "cls_weight":    2.0,   # must not confuse ball with anything else
    },
    2: {"name": "but",       "recall_weight": 1.0, "cls_weight": 1.0},
    3: {"name": "poteau",    "recall_weight": 1.0, "cls_weight": 1.0},
    4: {"name": "tag_bleu",  "recall_weight": 1.0, "cls_weight": 1.0},
    5: {"name": "tag_rouge", "recall_weight": 1.0, "cls_weight": 1.0},
    6: {"name": "robot_rct", "recall_weight": 1.0, "cls_weight": 1.0},
    7: {"name": "humain",    "recall_weight": 1.0, "cls_weight": 1.0},
}


# ── Asymmetric Loss ───────────────────────────────────────────────────────────

class PerClassAsymmetricLoss(nn.Module):
    """
    Drop-in replacement for BCEWithLogitsLoss in ultralytics v8DetectionLoss.

    Standard BCE treats false positives and false negatives equally.
    ASL separates them:
      Positives: -(1-p)^γ_pos * log(p)          γ_pos=0 → full gradient always
      Negatives: -(p_neg)^γ_neg * log(1-p_neg)  γ_neg=4 → only penalize confident FP
                 p_neg = max(0, p - clip)         clip    → ignore low-confidence FP

    Per-class recall_weight amplifies the positive branch for critical classes.
    Per-class cls_weight scales the full loss for classes where confusion is OK.

    reduction must be "none" — ultralytics reduces via target_scores_sum.
    """

    def __init__(
        self,
        class_config: dict = None,
        gamma_neg: float = ASL_GAMMA_NEG,
        gamma_pos: float = ASL_GAMMA_POS,
        clip: float = ASL_CLIP,
        reduction: str = "none",
    ):
        super().__init__()
        assert reduction == "none", "PerClassAsymmetricLoss requires reduction='none'"
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

        cfg = class_config if class_config is not None else CLASS_CONFIG
        self.register_buffer("recall_weights", torch.tensor(
            [cfg[i]["recall_weight"] for i in range(NUM_CLASSES)], dtype=torch.float32))
        self.register_buffer("cls_weights", torch.tensor(
            [cfg[i]["cls_weight"] for i in range(NUM_CLASSES)], dtype=torch.float32))

    def forward(self, pred_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(pred_logits)

        # Positive branch — never suppress real objects (γ_pos=0 by default)
        log_p = torch.log(p.clamp(min=1e-8))
        loss_pos = targets * log_p
        if self.gamma_pos > 0:
            loss_pos = loss_pos * (1.0 - p) ** self.gamma_pos
        loss_pos = loss_pos * self.recall_weights   # amplify FN penalty for critical classes

        # Negative branch — only penalize confident false positives
        p_neg = (p - self.clip).clamp(min=0.0) if self.clip > 0 else p
        log_1mp = torch.log((1.0 - p_neg).clamp(min=1e-8))
        loss_neg = (1.0 - targets) * log_1mp
        if self.gamma_neg > 0:
            loss_neg = loss_neg * p_neg ** self.gamma_neg

        raw_loss = -(loss_pos + loss_neg)
        return raw_loss * self.cls_weights


# ── Summary ───────────────────────────────────────────────────────────────────

def get_loss_summary() -> str:
    lines = ["", "=" * 65, "  LOSS CONFIGURATION — PerClassAsymmetricLoss", "=" * 65, ""]
    lines.append(f"  gamma_neg={ASL_GAMMA_NEG}  gamma_pos={ASL_GAMMA_POS}  clip={ASL_CLIP}")
    lines.append("")
    lines.append(f"  {'ID':<3} {'Class':<12} {'recall_w':>9} {'cls_w':>7}  Notes")
    lines.append(f"  {'-'*3} {'-'*12} {'-'*9} {'-'*7}  {'-'*30}")
    notes = {
        1: "CRITICAL — ball",
        5: "was R=0.696 — boosted recall_w",
        6: "cls relaxed (confusion w/ robot OK)",
        7: "SAFETY — never miss humans",
    }
    for i in range(NUM_CLASSES):
        cfg = CLASS_CONFIG[i]
        lines.append(
            f"  {i:<3} {cfg['name']:<12} "
            f"{cfg['recall_weight']:>9.1f} "
            f"{cfg['cls_weight']:>7.1f}  {notes.get(i, '')}"
        )
    lines += ["", "=" * 65, ""]
    return "\n".join(lines)


if __name__ == "__main__":
    print(get_loss_summary())
    loss_fn = PerClassAsymmetricLoss()
    dummy_logits = torch.randn(16, NUM_CLASSES)
    dummy_targets = torch.zeros(16, NUM_CLASSES)
    dummy_targets[0, 1] = 0.85
    dummy_targets[1, 7] = 0.72
    loss = loss_fn(dummy_logits, dummy_targets)
    print(f"  shape={loss.shape}  sum={loss.sum().item():.4f}  OK")
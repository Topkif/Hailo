"""
loss_function.py — Per-class Asymmetric Loss for YOLOv8n-P2 Hailo8 training.
Single source of truth for all per-class loss behavior.

How the three per-class weights interact:
---------------------------------------------------------------------------
  recall_weight : Multiplies the POSITIVE branch of the classification loss.
                  This is the "cost of missing an object" for each class.
                  Higher = the model is punished more for failing to detect
                  this class, so it fires more detections → higher recall.
                  Does NOT affect false positives or box quality.

  box_weight    : Multiplies the bounding box regression loss (IoU-based).
                  Higher = the model is punished more for inaccurate boxes.
                  Lower = the model can be sloppy on box edges (useful for
                  large amorphous objects where box precision doesn't matter).
                  Does NOT affect whether the object is detected at all.

  cls_weight    : Multiplies the FULL classification loss for a class
                  (both positive and negative branches).
                  Lower = the model is less penalized for confusing this class
                  with similar classes. Useful when two classes look alike
                  and you'd rather detect both than distinguish them perfectly.
---------------------------------------------------------------------------

Usage:
    from loss_function import (
        PerClassAsymmetricLoss,
        get_class_box_weights,
        get_loss_summary,
        CLASS_CONFIG,
        NUM_CLASSES,
    )

Integration with recall_trainer.py:
    # In RecallFocusedDetectionLoss.__init__:
    self.bce = PerClassAsymmetricLoss()

    # For per-class box weighting in the trainer:
    box_weights = get_class_box_weights(device)
    anchor_classes = target_scores.argmax(-1)[fg_mask]
    per_anchor_bw = box_weights[anchor_classes]
"""

import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════
#  GLOBAL ASL HYPERPARAMETERS
#
#  These control the asymmetry of the loss across ALL classes.
#  Per-class tuning is done in CLASS_CONFIG below.
# ══════════════════════════════════════════════════════════════════

# Focal exponent for NEGATIVES (false positives).
# Higher = model punishes confident false positives harder.
# Lower = more tolerant of low-confidence false alarms.
# Range: 2 (mild) → 6 (aggressive). Start at 4.
ASL_GAMMA_NEG = 4.0

# Focal exponent for POSITIVES (true positives / false negatives).
# Keep at 0: we NEVER want to suppress the gradient from real objects.
# If you set >0, hard-to-detect objects get less gradient → bad for recall.
ASL_GAMMA_POS = 0.0

# Probability margin (delta) for soft negatives.
# Negative predictions with p < delta contribute ZERO loss.
# This gives the model "free" low-confidence detections → higher recall.
# Range: 0.0 (strict, every FP penalized) → 0.1 (very permissive).
# Start at 0.05. Increase to 0.08 if recall is too low.
ASL_CLIP = 0.05


# ══════════════════════════════════════════════════════════════════
#  CLASS CONFIGURATION
#
#  *** THIS IS THE ONLY SECTION YOU NEED TO EDIT ***
#
#  Each class has three weights. Ranges and effects:
#
#  recall_weight (0.5 → 3.0):
#    0.5 = "don't care if we miss this class"
#    1.0 = standard
#    2.0 = "really try to find this"
#    3.0 = "NEVER miss this — critical for safety/gameplay"
#
#  box_weight (0.2 → 2.0):
#    0.2 = "just find it, box can be rough"
#    1.0 = standard
#    1.5 = "tight box required"
#    2.0 = "pixel-precise box needed"
#
#  cls_weight (0.2 → 2.0):
#    0.2 = "confusion with similar classes is fine"
#    0.7 = "some confusion tolerated"
#    1.0 = standard
#    1.5 = "must know exact class"
#    2.0 = "zero tolerance for misclassification"
# ══════════════════════════════════════════════════════════════════

NUM_CLASSES = 8

CLASS_NAMES = [
    "robot",        # 0
    "ballon",       # 1
    "but",          # 2
    "poteau",       # 3
    "tag_bleu",     # 4
    "tag_rouge",    # 5
    "robot_rct",    # 6
    "humain",       # 7
]

CLASS_CONFIG = {
    # ── Class 0: robot (friendly robot) ──────────────────────────────
    # Standard detection. Baseline weights.
    0: {
        "name":          "robot",
        "recall_weight": 1.0,   # standard recall priority
        "box_weight":    1.0,   # standard box precision
        "cls_weight":    1.0,   # standard classification
    },

    # ── Class 1: ballon (the ball) ───────────────────────────────────
    # MOST CRITICAL object. Must be detected with high recall AND
    # well-localized for trajectory prediction. Never miss the ball.
    1: {
        "name":          "ballon",
        "recall_weight": 2.5,   # very high — never miss the ball
        "box_weight":    1.5,   # tight box — needed for trajectory estimation
        "cls_weight":    1.5,   # must not confuse ball with anything else
    },

    # ── Class 2: but (goal structure) ────────────────────────────────
    # Large static object. Detection matters, but box edges are not
    # critical — the goal doesn't move and approximate position suffices.
    2: {
        "name":          "but",
        "recall_weight": 1.5,   # slightly elevated — should always find goals
        "box_weight":    0.5,   # relaxed — goal is large, exact edges don't matter
        "cls_weight":    1.0,   # standard — no similar class to confuse with
    },

    # ── Class 3: poteau (vertical pole) ──────────────────────────────
    # Standard landmark. Used for localization.
    3: {
        "name":          "poteau",
        "recall_weight": 1.0,   # standard
        "box_weight":    1.0,   # standard — pole localization helps navigation
        "cls_weight":    1.0,   # standard — distinctive shape
    },

    # ── Class 4: tag_bleu (blue team marker) ─────────────────────────
    # Small marker on robots. Detection matters for team identification.
    # Box can be slightly relaxed since markers are small.
    4: {
        "name":          "tag_bleu",
        "recall_weight": 1.0,   # standard
        "box_weight":    0.8,   # slightly relaxed — small target
        "cls_weight":    1.0,   # must distinguish blue from red
    },

    # ── Class 5: tag_rouge (red team marker) ─────────────────────────
    # Same as tag_bleu — symmetric treatment for fairness.
    5: {
        "name":          "tag_rouge",
        "recall_weight": 1.0,   # standard
        "box_weight":    0.8,   # slightly relaxed — small target
        "cls_weight":    1.0,   # must distinguish red from blue
    },

    # ── Class 6: robot_rct (opponent robot) ──────────────────────────
    # Confusion with class 0 (friendly robot) is TOLERATED.
    # Both are robots — it's more important to detect them than to
    # perfectly distinguish friend vs foe at the detection level.
    6: {
        "name":          "robot_rct",
        "recall_weight": 0.8,   # slightly below standard — less critical than ball/human
        "box_weight":    1.0,   # standard box
        "cls_weight":    0.7,   # LOW — confusion with 'robot' is acceptable
    },

    # ── Class 7: humain (human referee) ──────────────────────────────
    # HIGH RECALL required for SAFETY. The robot must detect humans
    # to avoid collisions. Missing a human is dangerous.
    7: {
        "name":          "humain",
        "recall_weight": 2.0,   # high — safety critical, never miss a human
        "box_weight":    1.0,   # standard — approximate position is enough for avoidance
        "cls_weight":    1.2,   # slightly elevated — must not confuse with robot
    },
}


# ══════════════════════════════════════════════════════════════════
#  PER-CLASS ASYMMETRIC LOSS
#
#  Drop-in replacement for nn.BCEWithLogitsLoss in ultralytics
#  v8DetectionLoss (assigned to self.bce).
#
#  Reference: "Asymmetric Loss For Multi-Label Classification"
#             Ben-Baruch et al., 2021 (ICCV)
#
#  The standard BCE treats false positives and false negatives
#  EQUALLY. ASL separates them:
#    - Positives (real objects): gentle treatment → high recall
#    - Negatives (background): harsh treatment → filter FP
#
#  On top of ASL, this class adds per-class weighting so that
#  critical classes (ball, human) get extra recall emphasis.
# ══════════════════════════════════════════════════════════════════

class PerClassAsymmetricLoss(nn.Module):
    """
    Asymmetric Loss with per-class recall and classification weights.

    Replaces BCEWithLogitsLoss in ultralytics v8DetectionLoss.
    Interface: forward(pred_logits, targets) → loss tensor (same shape).
    Reduction is "none" — ultralytics applies its own reduction downstream.
    """

    def __init__(
        self,
        class_config: dict = None,
        gamma_neg: float = ASL_GAMMA_NEG,
        gamma_pos: float = ASL_GAMMA_POS,
        clip: float = ASL_CLIP,
        reduction: str = "none",
    ):
        """
        Args:
            class_config: Dict mapping class_id → {recall_weight, cls_weight}.
                          Defaults to CLASS_CONFIG defined above.
            gamma_neg:    Focal exponent for negatives. Higher = harsher on FP.
            gamma_pos:    Focal exponent for positives. Keep 0 for max recall.
            clip:         Probability margin δ. Soft negatives (p < δ) ignored.
            reduction:    Must be "none" for ultralytics compatibility.
        """
        super().__init__()
        assert reduction == "none", (
            "PerClassAsymmetricLoss requires reduction='none' — "
            "ultralytics applies its own reduction via target_scores_sum"
        )

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

        # Build per-class weight tensors from config
        cfg = class_config if class_config is not None else CLASS_CONFIG
        recall_w = torch.tensor(
            [cfg[i]["recall_weight"] for i in range(NUM_CLASSES)],
            dtype=torch.float32,
        )
        cls_w = torch.tensor(
            [cfg[i]["cls_weight"] for i in range(NUM_CLASSES)],
            dtype=torch.float32,
        )

        # register_buffer: these tensors move with the model's device (CPU→GPU)
        # automatically, but are NOT trainable parameters.
        self.register_buffer("recall_weights", recall_w)
        self.register_buffer("cls_weights", cls_w)

    def forward(self, pred_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute per-element ASL with per-class weighting.

        Args:
            pred_logits: Raw logits BEFORE sigmoid. Shape (..., num_classes).
                         In ultralytics v8: (num_fg_anchors, num_classes) or
                         (batch_size * num_anchors, num_classes).
            targets:     Soft labels from Task-Aligned Assigner (TAL).
                         Shape same as pred_logits.
                         Foreground anchors have one class column with IoU score.
                         Background anchors are all zeros.

        Returns:
            Per-element loss tensor, same shape as inputs.
            Ultralytics reduces this with: loss.sum() / target_scores_sum
        """
        # ── Sigmoid activation ─────────────────────────────────────────
        # Convert logits to probabilities
        p = torch.sigmoid(pred_logits)

        # ── POSITIVE branch ────────────────────────────────────────────
        # Loss for elements where the target > 0 (real objects exist).
        # Formula: -targets × (1-p)^γ_pos × log(p)
        #
        # When γ_pos = 0 (default): simplifies to -targets × log(p)
        # This means we NEVER down-weight the gradient from real objects,
        # even if the model is already somewhat confident. Maximum recall.
        log_p = torch.log(p.clamp(min=1e-8))
        loss_pos = targets * log_p

        if self.gamma_pos > 0:
            # Focal weighting on positives: down-weight easy positives.
            # NOT recommended for recall-focused training (keep γ_pos=0).
            pt_pos = (1.0 - p) ** self.gamma_pos
            loss_pos = loss_pos * pt_pos

        # Apply per-class recall_weight to the POSITIVE branch ONLY.
        # This amplifies the false-negative penalty for critical classes.
        # recall_weights shape: (num_classes,) → broadcasts over (..., num_classes)
        loss_pos = loss_pos * self.recall_weights

        # ── NEGATIVE branch ────────────────────────────────────────────
        # Loss for elements where target == 0 (background).
        # Formula: -(1-targets) × (p_neg)^γ_neg × log(1 - p_neg)
        #
        # The clip (δ) shifts probabilities DOWN before computing loss:
        #   p_neg = max(0, p - δ)
        # Effect: if the model's confidence is below δ, the loss is ZERO.
        # This gives the model "free" low-confidence detections, which
        # tend to become true detections with more training → higher recall.
        if self.clip > 0:
            p_neg = (p - self.clip).clamp(min=0.0)
        else:
            p_neg = p

        log_1mp = torch.log((1.0 - p_neg).clamp(min=1e-8))
        loss_neg = (1.0 - targets) * log_1mp

        if self.gamma_neg > 0:
            # Focal weighting on negatives: down-weight easy negatives,
            # focus on hard false positives (high confidence, wrong class).
            # γ_neg=4 is aggressive — only very confident FPs get penalized.
            pt_neg = p_neg ** self.gamma_neg
            loss_neg = loss_neg * pt_neg

        # ── Combine branches ───────────────────────────────────────────
        # Both loss_pos and loss_neg are negative (log of values in [0,1]).
        # Negate to get positive loss values.
        raw_loss = -(loss_pos + loss_neg)

        # Apply per-class cls_weight to the FULL per-class loss.
        # This scales the total importance of correctly classifying each class.
        # cls_weights shape: (num_classes,) → broadcasts over (..., num_classes)
        weighted_loss = raw_loss * self.cls_weights

        return weighted_loss


# ══════════════════════════════════════════════════════════════════
#  BOX WEIGHT HELPER
#
#  Returns a tensor of per-class box weights for use in the
#  bounding box regression loss inside the trainer.
#
#  The trainer uses this as:
#    anchor_classes = target_scores.argmax(-1)[fg_mask]
#    per_anchor_bw = box_weights[anchor_classes]
#    weight = base_weight * per_anchor_bw
# ══════════════════════════════════════════════════════════════════

def get_class_box_weights(device: str = "cpu") -> torch.Tensor:
    """
    Returns a (NUM_CLASSES,) tensor of box_weight values.

    Usage in recall_trainer.py:
        box_weights = get_class_box_weights(self.device)
        # For each foreground anchor, look up its assigned class:
        anchor_classes = target_scores.argmax(-1)[fg_mask]
        # Get per-anchor box weight:
        per_anchor_bw = box_weights[anchor_classes]
        # Multiply into the per-anchor IoU loss weight:
        weight = alignment_metric * per_anchor_bw
    """
    weights = torch.tensor(
        [CLASS_CONFIG[i]["box_weight"] for i in range(NUM_CLASSES)],
        dtype=torch.float32,
        device=device,
    )
    return weights


# ══════════════════════════════════════════════════════════════════
#  LOSS SUMMARY (for logging at training start)
# ══════════════════════════════════════════════════════════════════

def get_loss_summary() -> str:
    """
    Returns a formatted multi-line string showing all class weights
    and global ASL parameters. Print this at the start of training
    to verify your configuration.
    """
    lines = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("  LOSS CONFIGURATION - PerClassAsymmetricLoss")
    lines.append("=" * 70)
    lines.append("")
    lines.append("  Global ASL parameters:")
    lines.append(f"    gamma_neg = {ASL_GAMMA_NEG:<5}  (focal exp for FP - higher = harsher)")
    lines.append(f"    gamma_pos = {ASL_GAMMA_POS:<5}  (focal exp for TP - keep 0 for recall)")
    lines.append(f"    clip      = {ASL_CLIP:<5}  (probability margin delta)")
    lines.append("")
    lines.append(f"  {'ID':<3} {'Class':<12} {'recall_w':>9} {'box_w':>7} {'cls_w':>7}  Notes")
    lines.append(f"  {'-'*3} {'-'*12} {'-'*9} {'-'*7} {'-'*7}  {'-'*25}")

    notes = {
        0: "",
        1: "CRITICAL - ball",
        2: "box relaxed (large static)",
        3: "",
        4: "",
        5: "",
        6: "cls relaxed (confusion w/ robot OK)",
        7: "SAFETY - never miss humans",
    }

    for i in range(NUM_CLASSES):
        cfg = CLASS_CONFIG[i]
        note = notes.get(i, "")
        lines.append(
            f"  {i:<3} {cfg['name']:<12} "
            f"{cfg['recall_weight']:>9.1f} "
            f"{cfg['box_weight']:>7.1f} "
            f"{cfg['cls_weight']:>7.1f}  {note}"
        )

    lines.append("")
    lines.append("  Interpretation:")
    lines.append("    recall_w > 1 = model tries harder to FIND this class")
    lines.append("    box_w < 1    = model allowed to be sloppy on box edges")
    lines.append("    cls_w < 1    = model allowed to CONFUSE this with similar classes")
    lines.append("=" * 70)
    lines.append("")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
#  MAIN — verify configuration without starting training
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(get_loss_summary())

    # Quick sanity check: instantiate and run with dummy data
    print("  Sanity check: running forward pass with dummy data...")
    loss_fn = PerClassAsymmetricLoss()
    dummy_logits = torch.randn(16, NUM_CLASSES)   # 16 anchors, 8 classes
    dummy_targets = torch.zeros(16, NUM_CLASSES)
    dummy_targets[0, 1] = 0.85   # anchor 0 is a 'ballon' with IoU 0.85
    dummy_targets[1, 7] = 0.72   # anchor 1 is a 'humain' with IoU 0.72
    dummy_targets[2, 0] = 0.90   # anchor 2 is a 'robot' with IoU 0.90

    loss = loss_fn(dummy_logits, dummy_targets)
    print(f"  Input shape:  {dummy_logits.shape}")
    print(f"  Output shape: {loss.shape}")
    print(f"  Loss sum:     {loss.sum().item():.4f}")
    print(f"  Loss mean:    {loss.mean().item():.4f}")
    print("\n  OK - loss_function.py is correctly configured.\n")

# CLAUDE.md — YOLOv8n-P2 Hailo8 Training Project

## Project Goal
Train a YOLOv8n-P2 object detection model optimized for **high recall and real-life detection**
on a 1920x1080 camera (Hailo8 smart camera), exported as ONNX float32 for external int8 quantization.

---

## Hardware & Resolution
| Parameter | Value | Reason |
|---|---|---|
| Camera | 1920x1080 | Source resolution |
| Train/infer resolution | 1024x576 | Exact 16:9, both /32, no letterboxing |
| Target device | Hailo8 | CNN accelerator - no transformers |
| Target FPS | >=40 fps | Headroom from YOLOv7-tiny @ 100fps |
| Export format | ONNX float32, opset 11, static shape | Hailo Dataflow Compiler requirement |

---

## Model
- **Architecture** : `yolov8n-p2.yaml` (not yolov8s - nano for speed on Hailo8)
- **Pretrained weights** : `yolov8n.pt` (transferred into yaml via `.load()`)
- **P2 head** : adds a 4th detection head at stride 4 (256x144 feature map at 1024x576)
  -> detects objects as small as ~4px, vs 8px minimum for standard P3 head
- **Why not RTDETR/hybrid** : transformer attention does not map onto Hailo8 MAC arrays

```python
# Correct instantiation - there is no yolov8n-p2.pt pretrained file
model = YOLO("yolov8n-p2.yaml").load("yolov8n.pt")
```

---

## Classes
```
0  robot       - friendly robot
1  ballon      - the ball  <- CRITICAL: highest recall + precision required
2  but         - goal post (box precision not critical)
3  poteau      - vertical pole
4  tag_bleu    - blue team tag
5  tag_rouge   - red team tag
6  robot_rct   - opponent robot (confusion with 'robot' tolerated)
7  humain      - human referee
```

---

## File Structure
```
project/
├── train_hailo.py       - state machine with PHASE_CONFIG / LOSS_PRESETS / AUGMENTATION_PRESETS
├── recall_trainer.py    - custom DetectionTrainer with preset-driven ASL loss + F2 fitness
├── loss_function.py     - per-class weights + PerClassAsymmetricLoss (single source of truth)
├── evaluate.py          - validate .pt, annotate images, write results.txt
└── runs/hailo_train/
    ├── train_state.json - crash-safe phase tracking
    └── <phase_name>/weights/
        ├── best.pt          <- saved by F2 fitness (recall-weighted)
        ├── best_recall.pt   <- saved by highest raw recall
        ├── best_f2.pt       <- saved by highest F2 score
        └── last.pt
```

---

## Training Philosophy

### Core Problem
Default YOLO:
- `cls_loss` = BCE -> treats false positives and false negatives **equally**
- `best.pt` fitness = `0.9 x mAP50-95` -> **recall has zero weight**

This project fixes both.

### Loss Function - AsymmetricLoss (ASL)
Replaces BCE in the classification branch only. Box and DFL loss are unchanged.

```
Positives (object exists):  L = -(1-p)^gamma_pos x log(p)       gamma_pos=0 -> never discount real objects
Negatives (background):     L = -(p-d)^gamma_neg x log(1-(p-d)) gamma_neg=4 -> punish confident FP hard
                                                                 d=0.05  -> soft FP (p<d) ignored -> higher recall
```

Per-class weights multiply the loss per sample - classes like `ballon` get higher penalty for misses.

**Key tuning variables in `loss_function.py`:**
```python
ASL_GAMMA_NEG        # 2-6   - strictness on false positives
ASL_GAMMA_POS        # 0     - keep 0: never suppress true positives
ASL_CLIP             # 0-0.1 - probability margin; higher = more recall, less precision
CLASS_RECALL_WEIGHT  # per-class multiplier on positive loss
CLASS_BOX_WEIGHT     # per-class multiplier on box loss
CLASS_CLS_WEIGHT     # per-class multiplier on cls loss
```

### Fitness for best.pt - F-beta score
```
Default: 0.0xP + 0.0xR + 0.1xmAP50 + 0.9xmAP50-95   (recall weight = 0)
Ours:    0.6xF2(P,R) + 0.3xmAP50 + 0.1xmAP50-95

F2 = 5xPxR / (4P + R)   -> recall weighted 2x more than precision
```

### Grayscale Augmentation
Configurable per-phase via `grayscale_p` (0.0-1.0).
Converts a fraction of each training batch to 3-channel grayscale at the tensor level (GPU).
Teaches shape/texture invariance. Critical for cameras that switch day (color) -> night (IR/gray).

---

## Training Phases (Preset-Driven)

Training uses a **cost-function phase approach**: each phase applies a different
loss preset and augmentation preset. No backbone freezing - warmup_epochs=5
protects backbone features during the initial LR ramp-up.

### Phase 1 - "high_recall" (loss: recall_focused, aug: aggressive)
- Warmup=5 epochs protects backbone (no freezing needed with 5k+ images)
- Relaxed box/cls loss (4.0/0.3) encourages more proposals -> high recall
- Aggressive augmentation: mosaic=1.0, copy_paste=0.3, scale=0.9
- ASL clip=0.05: soft negatives ignored -> model free to fire detections
- Goal: ballon recall > 0.95, overall recall > 0.80

### Phase 2 - "color_invariance" (loss: recall_focused, aug: color_stress)
- 70% grayscale forces shape/texture reliance over color
- Extreme hsv_s=0.9, hsv_v=0.6 combined with grayscale stress
- Same recall-focused loss - not tightening precision yet
- Goal: metrics stable under color removal

### Phase 3 - "precision" (loss: precision, aug: moderate)
- Tight box weight (7.5) penalizes sloppy localization
- Lower ASL clip=0.02: most false positives now penalized
- Moderate augmentation for stable convergence
- Goal: mAP50-95 > 0.60, precision > 0.82

### Phase 4 - "class_surgery" (loss: class_surgery, aug: minimal) [OPTIONAL]
- Elevates recall_weight for specific weak classes (robot: 2.5, tag_rouge: 2.0)
- Very low LR (0.00005) + minimal augmentation = targeted fix
- Only run if specific classes underperform after phase 3
- Goal: boost recall on underperforming classes

---

## Loss Presets (defined in train_hailo.py)

| Preset | box | cls | gamma_neg | clip | Purpose |
|---|---|---|---|---|---|
| recall_focused | 4.0 | 0.3 | 4 | 0.05 | Max recall, relaxed box/cls |
| precision | 7.5 | 0.5 | 3 | 0.02 | Tight boxes, strict classification |
| class_surgery | 6.0 | 0.4 | 4 | 0.05 | Targeted per-class recall boost |

Each preset can override per-class recall_weight via `class_recall_overrides`
without modifying loss_function.py.

## Augmentation Presets (defined in train_hailo.py)

| Preset | mosaic | copy_paste | scale | hsv_s | hsv_v | Purpose |
|---|---|---|---|---|---|---|
| aggressive | 1.0 | 0.3 | 0.9 | 0.7 | 0.4 | Max variation, early training |
| moderate | 0.5 | 0.2 | 0.5 | 0.7 | 0.4 | Balanced, precision phase |
| minimal | 0.0 | 0.05 | 0.2 | 0.4 | 0.4 | Consolidation, surgery |
| color_stress | 1.0 | 0.3 | 0.9 | 0.9 | 0.6 | Color invariance stress test |

---

## How to Add a New Phase

1. **Append** a dict to `PHASE_CONFIG` in `train_hailo.py`:
   ```python
   {
       "name":           "my_new_phase",
       "epochs":         30,
       "loss_preset":    "recall_focused",   # pick from LOSS_PRESETS
       "lr0":            0.0001,
       "lrf":            0.01,
       "warmup_epochs":  5,
       "batch":          -1,
       "grayscale_p":    0.2,
       "augmentation":   "moderate",         # pick from AUGMENTATION_PRESETS
       "patience":       15,
       "stop_condition": "describe your goal here",
   }
   ```

2. **(Optional)** Define a new loss or augmentation preset if existing ones don't fit.

3. **Re-run** `python train_hailo.py` - completed phases are auto-skipped,
   the new phase runs next.

To remove a phase: comment it out of PHASE_CONFIG. To re-run a phase:
`python train_hailo.py --from <phase_name>`

---

## Metrics to Watch
| Metric | Target | Meaning |
|---|---|---|
| Recall | >0.85 | % of real objects found - primary goal |
| Precision | >0.70 | % of detections that are correct |
| F2 | >0.82 | Recall-weighted composite - used for best.pt |
| mAP@50 | >0.80 | Overall detection accuracy |
| mAP@50-95 | >0.60 | Box precision (secondary) |

---

## Inference Tuning (post-training, no retraining)
```python
conf=0.15-0.25   # lower = higher recall, more false alarms
iou=0.40-0.50    # lower = less aggressive box merging (helps small/touching objects)
max_det=1000     # raise if many objects per frame
```
Run `evaluate.py` with `--data dataset.yaml --split test` to sweep conf/iou.

---

## Hailo8 Export Checklist
```
1. ONNX export:  opset=11, float32, static shape (1,3,576,1024), simplify=True
2. Parse:        hailo parser onnx model.onnx --hw-arch hailo8
3. Quantize:     hailo optimize model.har --hw-arch hailo8 --calib-path calib_images/
4. Compile:      hailo compiler model.har --hw-arch hailo8
5. Output:       model.hef  -> deploy to Hailo8 runtime
```
Calibration images: 100-300 representative frames from the actual camera.

---

## Common Commands
```bash
# Full training run (all phases)
python train_hailo.py

# Resume from a specific phase
python train_hailo.py --from precision

# Run only one phase
python train_hailo.py --only high_recall

# Skip specific phases
python train_hailo.py --skip color_invariance

# Export only (uses last saved weights)
python train_hailo.py --only export

# Export specific weights
python train_hailo.py --only export --weights path/to/best.pt

# Quick validation
python train_hailo.py --only val --weights best.pt

# Threshold sweep for deployment
python train_hailo.py --only sweep

# Compare checkpoints from a phase
python recall_trainer.py --compare runs/hailo_train/precision --data data.yaml

# Verify loss configuration
python loss_function.py
```

---

## Key Design Decisions & Rationale
| Decision | Why |
|---|---|
| yolov8**n** not yolov8s | Faster on Hailo8; P2 head compensates for smaller backbone |
| 1024x576 not 960x512 | Exact 16:9 ratio - zero distortion from 1920x1080 |
| ONNX float32 not int8 | Hailo SDK quantizes more accurately with its own calibration |
| ASL not focal loss | ASL separates gamma for pos/neg - focal loss uses same gamma for both |
| F2 fitness not mAP | mAP50-95 rewards tight boxes; F2 rewards finding objects |
| loss_function.py separate | Single source of truth for loss; easier per-class versioning |
| No backbone freezing | 5k+ images + warmup=5 is sufficient; freezing complicates resume |
| Preset-driven phases | Adding/removing phases = editing a list, not rewriting logic |
| Grayscale per-phase | Different phases need different color invariance pressure |

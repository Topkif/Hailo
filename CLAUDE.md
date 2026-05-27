# CLAUDE.md — YOLOv8n-P2 Hailo8 Training Project

## Project Goal
Train a YOLOv8n-P2 object detection model optimized for **high recall and real-life detection**
on a 1920×1080 camera (Hailo8 smart camera), exported as ONNX float32 for external int8 quantization.

---

## Hardware & Resolution
| Parameter | Value | Reason |
|---|---|---|
| Camera | 1920×1080 | Source resolution |
| Train/infer resolution | 1024×576 | Exact 16:9, both ÷32, no letterboxing |
| Target device | Hailo8 | CNN accelerator — no transformers |
| Target FPS | ≥40 fps | Headroom from YOLOv7-tiny @ 100fps |
| Export format | ONNX float32, opset 11, static shape | Hailo Dataflow Compiler requirement |

---

## Model
- **Architecture** : `yolov8n-p2.yaml` (not yolov8s — nano for speed on Hailo8)
- **Pretrained weights** : `yolov8n.pt` (transferred into yaml via `.load()`)
- **P2 head** : adds a 4th detection head at stride 4 (256×144 feature map at 1024×576)
  → detects objects as small as ~4px, vs 8px minimum for standard P3 head
- **Why not RTDETR/hybrid** : transformer attention does not map onto Hailo8 MAC arrays

```python
# Correct instantiation — there is no yolov8n-p2.pt pretrained file
model = YOLO("yolov8n-p2.yaml").load("yolov8n.pt")
```

---

## Classes
```
0  robot       — friendly robot
1  ballon      — the ball  ← CRITICAL: highest recall + precision required
2  but         — goal post (box precision not critical)
3  poteau      — vertical pole
4  tag_bleu    — blue team tag
5  tag_rouge   — red team tag
6  robot_rct   — opponent robot (confusion with 'robot' tolerated)
7  humain      — human referee
```

---

## File Structure
```
project/
├── train_hailo.py       — state machine: phase1 → phase2 → phase3 → export
├── recall_trainer.py    — custom DetectionTrainer with ASL loss + F2 fitness
├── loss_function.py     — shared loss: AsymmetricLoss + per-class weights (single source of truth)
├── evaluate.py          — validate .pt, annotate images, write results.txt
└── runs/hailo_train/
    ├── train_state.json — crash-safe phase tracking
    └── phaseN/weights/
        ├── best.pt          ← saved by F2 fitness (recall-weighted)
        ├── best_recall.pt   ← saved by highest raw recall
        ├── best_f2.pt       ← saved by highest F2 score
        └── last.pt
```

---

## Training Philosophy

### Core Problem
Default YOLO:
- `cls_loss` = BCE → treats false positives and false negatives **equally**
- `best.pt` fitness = `0.9×mAP50-95` → **recall has zero weight**

This project fixes both.

### Loss Function — AsymmetricLoss (ASL)
Replaces BCE in the classification branch only. Box and DFL loss are unchanged.

```
Positives (object exists):  L = -(1-p)^γ_pos × log(p)       γ_pos=0 → never discount real objects
Negatives (background):     L = -(p-δ)^γ_neg × log(1-(p-δ)) γ_neg=4 → punish confident FP hard
                                                               δ=0.05  → soft FP (p<δ) ignored → higher recall
```

Per-class weights multiply the loss per sample — classes like `ballon` get higher penalty for misses.

**Key tuning variables in `loss_function.py`:**
```python
ASL_GAMMA_NEG        # 2–6   — strictness on false positives
ASL_GAMMA_POS        # 0     — keep 0: never suppress true positives
ASL_CLIP             # 0–0.1 — probability margin; higher = more recall, less precision
CLASS_RECALL_WEIGHT  # per-class multiplier on positive loss
CLASS_BOX_WEIGHT     # per-class multiplier on box loss
CLASS_CLS_WEIGHT     # per-class multiplier on cls loss
```

### Fitness for best.pt — F-beta score
```
Default: 0.0×P + 0.0×R + 0.1×mAP50 + 0.9×mAP50-95   (recall weight = 0)
Ours:    0.6×F2(P,R) + 0.3×mAP50 + 0.1×mAP50-95

F2 = 5×P×R / (4P + R)   → recall weighted 2× more than precision
```

### Grayscale Augmentation
30% of each training batch is converted to 3-channel grayscale at the tensor level (GPU).
Teaches shape/texture invariance. Critical for cameras that switch day (color) → night (IR/gray).

---

## Training Phases

### Phase 1 — Backbone frozen, recall bias
- Freeze layers 0–9 (backbone), train detection heads only
- High LR (0.005), aggressive augmentation (mosaic, copy_paste=0.3, scale=0.9)
- Loss: box=4.0, cls=0.3 → model tries more proposals → higher recall
- Goal: establish recall baseline without destroying pretrained backbone

### Phase 2 — Full model, precision tightened
- Unfreeze everything, LR=0.0005
- Loss: box=7.5, cls=0.5 → default, sharpen box positions
- close_mosaic=15 → disables mosaic last 15 epochs → stable convergence
- label_smoothing=0.05, dropout=0.1 → reduce overfitting
- Goal: tighten mAP50-95 while keeping recall from phase 1

### Phase 3 — Recall surgery (optional)
- Run only if recall < target after phase 2
- Very low LR (0.0001), no mosaic, minimal augmentation
- Loss: box=3.5, cls=0.2 → maximum recall push
- Goal: final recall recovery without undoing phase 2 box precision

---

## Metrics to Watch
| Metric | Target | Meaning |
|---|---|---|
| Recall | >0.85 | % of real objects found — primary goal |
| Precision | >0.70 | % of detections that are correct |
| F2 | >0.82 | Recall-weighted composite — used for best.pt |
| mAP@50 | >0.80 | Overall detection accuracy |
| mAP@50-95 | >0.60 | Box precision (secondary) |

---

## Inference Tuning (post-training, no retraining)
```python
conf=0.15–0.25   # lower = higher recall, more false alarms
iou=0.40–0.50    # lower = less aggressive box merging (helps small/touching objects)
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
5. Output:       model.hef  → deploy to Hailo8 runtime
```
Calibration images: 100–300 representative frames from the actual camera.

---

## Common Commands
```bash
# Full training run
python train_hailo.py

# Resume from a specific phase
python train_hailo.py --from phase2

# Skip recall surgery (phase 3) if recall already good
python train_hailo.py --skip-p3

# Validate + annotate with dataset yaml (test split)
python evaluate.py --model runs/hailo_train/phase2/weights/best.pt \
                   --data dataset.yaml --split test

# Validate with raw image+label folder
python evaluate.py --model best.pt --images dataset/images/test/

# Compare all checkpoints from a phase
python recall_trainer.py --compare runs/hailo_train/phase2 --data dataset.yaml
```

---

## Key Design Decisions & Rationale
| Decision | Why |
|---|---|
| yolov8**n** not yolov8s | Faster on Hailo8; P2 head compensates for smaller backbone |
| 1024×576 not 960×512 | Exact 16:9 ratio — zero distortion from 1920×1080 |
| ONNX float32 not int8 | Hailo SDK quantizes more accurately with its own calibration |
| ASL not focal loss | ASL separates γ for pos/neg — focal loss uses same γ for both |
| F2 fitness not mAP | mAP50-95 rewards tight boxes; F2 rewards finding objects |
| loss_function.py separate | Single source of truth for loss; easier per-class versioning |

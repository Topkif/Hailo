"""
yolov8n-p2_builder.py — Generate yolov8n-p2-pretrained.pt once.

Run this before the first training:
    python yolov8n-p2_builder.py

What it does:
  1. Downloads yolov8n.pt (ultralytics pretrained nano weights)
  2. Loads them into the yolov8n-p2.yaml architecture (adds the P2 head)
  3. Saves the result as yolov8n-p2-pretrained.pt

After this, train_hailo.py uses yolov8n-p2.pt directly and
never needs the yaml again.
"""

from pathlib import Path
from ultralytics import YOLO

OUT = Path("yolov8n-p2.pt")

if OUT.exists():
    print(f"[skip] {OUT} already exists — delete it to rebuild.")
else:
    print("Building yolov8n-p2-pretrained.pt ...")
    YOLO("yolov8n-p2.yaml").load("yolov8n.pt").save(str(OUT))
    print(f"[done] {OUT}  ({OUT.stat().st_size // 1024} KB)")
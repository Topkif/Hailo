"""
evaluate.py — YOLO model evaluation + annotated image export.

Three modes
───────────
1. dataset yaml  (ground-truth metrics, val or test split)
     python evaluate.py --model best.pt --data dataset.yaml
     python evaluate.py --model best.pt --data dataset.yaml --split test

2. images + labels folder  (ground-truth metrics, any folder pair)
     python evaluate.py --model best.pt --images path/to/images/test/
     → labels auto-located at  path/to/labels/test/  (YOLO convention)
     → if no labels found: inference-only, no P/R/mAP

3. images only  (inference + annotation, no metrics)
     python evaluate.py --model best.pt --images path/to/images/ --no-metrics

Output (always next to this script)
────────────────────────────────────
  test_annotation/   annotated images
  results.txt        metrics + per-image detections
"""

import argparse
import cv2
import numpy as np
import tempfile
import yaml
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO


# ── Visual config ─────────────────────────────────────────────────
LINE_THICKNESS = 2
FONT_SCALE     = 0.45
FONT_THICKNESS = 1
FONT           = cv2.FONT_HERSHEY_SIMPLEX
LABEL_PADDING  = 3
IMAGE_EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

PALETTE = [
    (255,  82,  82), ( 82, 255,  82), ( 82, 164, 255),
    (255, 200,  82), (200,  82, 255), ( 82, 255, 200),
    (255, 128,   0), (  0, 200, 255), (255,   0, 200),
    (160, 255,  82), ( 82,  82, 255), (255, 255,  82),
]

def class_color(cls_id):
    return PALETTE[int(cls_id) % len(PALETTE)]


# ── Drawing ───────────────────────────────────────────────────────

def draw_detections(img, boxes, names):
    out = img.copy()
    for box in boxes:
        cls_id          = int(box.cls[0])
        conf            = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color           = class_color(cls_id)
        label           = f"{names[cls_id]} {conf:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, LINE_THICKNESS)

        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
        ly1 = y1 - th - LABEL_PADDING * 2
        ly2 = y1
        if ly1 < 0:
            ly1, ly2 = y2, y2 + th + LABEL_PADDING * 2
        cv2.rectangle(out, (x1, ly1), (x1 + tw + LABEL_PADDING * 2, ly2), color, -1)
        cv2.putText(out, label, (x1 + LABEL_PADDING, ly2 - LABEL_PADDING),
                    FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA)
    return out


# ── Label auto-detection ──────────────────────────────────────────

def find_labels_dir(images_dir: Path) -> Path | None:
    """
    Given  .../images/test/
    Return .../labels/test/   if it exists and contains .txt files.

    Works for any depth — replaces the 'images' component in the path.
    """
    parts = images_dir.parts
    for i, part in enumerate(parts):
        if part == "images":
            candidate = Path(*parts[:i]) / "labels" / Path(*parts[i+1:])
            if candidate.exists() and list(candidate.glob("*.txt")):
                return candidate
    # Fallback: look for a 'labels' sibling of images_dir
    sibling = images_dir.parent / "labels"
    if sibling.exists() and list(sibling.glob("*.txt")):
        return sibling
    return None


def make_temp_yaml(images_dir: Path, labels_dir: Path, names: dict) -> Path:
    """
    Build a temporary data.yaml so model.val() can find images + labels.
    YOLO resolves labels by replacing 'images' in the path — so we
    structure the temp dirs to match that convention.
    """
    tmp = Path(tempfile.mkdtemp())

    # Symlink or note real paths — we just write absolute paths into yaml
    data = {
        "path": str(images_dir.parent),  # dataset root
        "val":  str(images_dir),         # absolute path is fine for val
        "nc":   len(names),
        "names": [names[i] for i in sorted(names)],
        # Tell YOLO exactly where labels are
        # (ultralytics uses img_path.replace('images','labels') internally,
        #  so we write a redirect key if available)
    }

    yaml_path = tmp / "eval_temp.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f)
    return yaml_path


# ── Inference + annotation ────────────────────────────────────────

def run_inference(model, image_paths, conf, iou, out_dir, log):
    out_dir.mkdir(parents=True, exist_ok=True)
    names  = model.names
    totals = {"images": 0, "detections": 0}

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            log(f"  [skip] cannot read {img_path.name}")
            continue

        results   = model.predict(img, conf=conf, iou=iou, verbose=False)[0]
        annotated = draw_detections(img, results.boxes, names)
        cv2.imwrite(str(out_dir / img_path.name), annotated)

        n = len(results.boxes)
        totals["images"]     += 1
        totals["detections"] += n

    return totals


# ── Validation metrics ────────────────────────────────────────────

def run_val(model, data_yaml, split, conf, iou, imgsz, device):
    """
    Run model.val() against a yaml split.
    split = 'val' | 'test'
    """
    m = model.val(
        data    = str(data_yaml),
        split   = split,
        conf    = conf,
        iou     = iou,
        imgsz   = imgsz,
        device  = device,
        verbose = False,
    )
    p_arr    = list(m.box.p)
    r_arr    = list(m.box.r)
    ap50_arr = list(getattr(m.box, "ap50", m.box.maps))
    maps_arr = list(m.box.maps)
    nt_arr   = list(getattr(m.box, "nt_per_class",
                   getattr(m.box, "nt", [0] * len(p_arr))))

    per_class = {}
    for i, idx in enumerate(sorted(m.names)):
        name = m.names[idx]
        per_class[name] = {
            "p":    round(float(p_arr[i])    if i < len(p_arr)    else 0.0, 4),
            "r":    round(float(r_arr[i])    if i < len(r_arr)    else 0.0, 4),
            "ap50": round(float(ap50_arr[i]) if i < len(ap50_arr) else 0.0, 4),
            "map":  round(float(maps_arr[i]) if i < len(maps_arr) else 0.0, 4),
            "n":    int(nt_arr[i])           if i < len(nt_arr)   else 0,
        }

    return {
        "precision": float(m.box.mp),
        "recall":    float(m.box.mr),
        "map50":     float(m.box.map50),
        "map50_95":  float(m.box.map),
        "per_class": per_class,
    }


# ── Pretty print ──────────────────────────────────────────────────

def print_metrics(metrics, log):
    P, R = metrics["precision"], metrics["recall"]
    f1   = 2 * P * R / (P + R + 1e-9)
    f2   = 5 * P * R / (4 * P + R + 1e-9)

    # ── Overall summary ──────────────────────────────────────────────
    log("")
    log("╔══════════════════════════════════════╗")
    log("║         VALIDATION METRICS           ║")
    log("╠══════════════════════════════════════╣")
    log(f"║  Precision    :   {P:.4f}               ║")
    log(f"║  Recall       :   {R:.4f}               ║")
    log(f"║  F1           :   {f1:.4f}               ║")
    log(f"║  F2 (recall²) :   {f2:.4f}               ║")
    log("╠══════════════════════════════════════╣")
    log(f"║  mAP@50       :   {metrics['map50']:.4f}               ║")
    log(f"║  mAP@50-95    :   {metrics['map50_95']:.4f}               ║")
    log("╚══════════════════════════════════════╝")

    # ── Per-class table ──────────────────────────────────────────────
    pc = metrics["per_class"]
    col_w = max((len(f"{c} (x{v['n']})") for c, v in pc.items()), default=12)
    col_w = max(col_w, 12)

    def _f1(p, r):  return 2 * p * r / (p + r + 1e-9)
    def _f2(p, r):  return 5 * p * r / (4 * p + r + 1e-9)

    H = col_w + 2
    sep = "─" * H
    hdr = f"{'Class':<{col_w}}"
    log("")
    log("╔══════════════════════════════════════════════════════════════════════════╗")
    log("║                      PER-CLASS METRICS                                  ║")
    log("╚══════════════════════════════════════════════════════════════════════════╝")
    log(f"╔{sep}╦{'─'*8}╦{'─'*8}╦{'─'*8}╦{'─'*8}╦{'─'*8}╦{'─'*8}╗")
    log(f"║ {hdr}║{'Prec':^8}║{'Recall':^8}║{'F1':^8}║{'F2':^8}║{'AP@50':^8}║{'mAP95':^8}║")
    log(f"╠{sep}╬{'═'*8}╬{'═'*8}╬{'═'*8}╬{'═'*8}╬{'═'*8}╬{'═'*8}╣")
    for cls, v in pc.items():
        cp, cr = v["p"], v["r"]
        cf1 = _f1(cp, cr)
        cf2 = _f2(cp, cr)
        label = f"{cls} (x{v['n']})"
        log(f"║ {label:<{col_w}}║{cp:^8.4f}║{cr:^8.4f}║{cf1:^8.4f}║{cf2:^8.4f}║{v['ap50']:^8.4f}║{v['map']:^8.4f}║")
    log(f"╠{sep}╬{'═'*8}╬{'═'*8}╬{'═'*8}╬{'═'*8}╬{'═'*8}╬{'═'*8}╣")
    log(f"║ {'ALL (mean)':<{col_w}}║{P:^8.4f}║{R:^8.4f}║{f1:^8.4f}║{f2:^8.4f}║{metrics['map50']:^8.4f}║{metrics['map50_95']:^8.4f}║")
    log(f"╚{sep}╩{'═'*8}╩{'═'*8}╩{'═'*8}╩{'═'*8}╩{'═'*8}╩{'═'*8}╝")

    log("")
    log("── Metric definitions ───────────────────────────────────────────")
    log("  Precision  : of all detections made, % that were correct (low = ghost detections)")
    log("  Recall     : of all real objects in scene, % the model found (low = missed objects)")
    log("  F1         : harmonic mean of P and R — balanced score")
    log("  F2         : recall-weighted score — missing an object counts 2x more than a false alarm")
    log("  AP@50      : per-class average precision at IoU=0.50 (lenient box match)")
    log("  mAP@50-95  : per-class mean AP across IoU thresholds 0.50→0.95 (strict box precision)")
    log("─────────────────────────────────────────────────────────────────")


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      required=True,
                        help=".pt weights file")
    parser.add_argument("--data",       default=None,
                        help="dataset.yaml (YOLO format) — uses --split to choose val or test")
    parser.add_argument("--split",      default="test", choices=["train", "val", "test"],
                        help="Which split to evaluate when using --data (default: test)")
    parser.add_argument("--images",     default=None,
                        help="Folder of images — labels auto-located at ../labels/ or sibling labels/")
    parser.add_argument("--no-metrics", action="store_true",
                        help="Skip metric computation even if labels are found")
    parser.add_argument("--conf",       type=float, default=0.20)
    parser.add_argument("--iou",        type=float, default=0.45)
    parser.add_argument("--imgsz",      type=int, nargs=2, default=[1024, 576])
    parser.add_argument("--device",     default="0")
    args = parser.parse_args()

    if not args.data and not args.images:
        parser.error("Provide --data and/or --images")

    script_dir  = Path(__file__).parent
    out_dir     = script_dir / "test_annotation"
    results_txt = script_dir / "results.txt"

    _lines = []
    def log(text=""):
        print(text)
        _lines.append(text)

    # ── Header ────────────────────────────────────────────────────
    log(f"date    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"model   : {args.model}")
    log(f"conf    : {args.conf}   iou : {args.iou}")

    log(f"\n[load] {args.model}")
    model = YOLO(args.model)

    # ══════════════════════════════════════════════════════════════
    #  MODE A — dataset.yaml provided
    #  Uses the split defined in the yaml (val or test key)
    # ══════════════════════════════════════════════════════════════
    if args.data:
        yaml_path = Path(args.data)
        with open(yaml_path) as f:
            ds = yaml.safe_load(f)

        log(f"data    : {yaml_path}  (split: {args.split})")
        log(f"classes : {ds.get('names', [])}")

        # -- Metrics --
        if not args.no_metrics:
            log(f"\n[val] split='{args.split}' ...")
            metrics = run_val(model, yaml_path, args.split,
                              args.conf, args.iou, tuple(args.imgsz), args.device)
            print_metrics(metrics, log)

        # -- Annotate the split images --
        dataset_root = Path(ds.get("path", yaml_path.parent))
        split_rel    = ds.get(args.split)

        if split_rel:
            split_imgs = dataset_root / split_rel
            # yaml may point to an images/ subfolder or directly to images
            if not split_imgs.exists():
                split_imgs = dataset_root / "images" / split_rel
            if split_imgs.exists():
                paths = sorted(p for p in split_imgs.rglob("*")
                               if p.suffix.lower() in IMAGE_EXTS)
                if paths:
                    log(f"\n[annotate] {len(paths)} images from {args.split} split → {out_dir}/")
                    totals = run_inference(model, paths, args.conf,
                                          args.iou, out_dir, log)
                    log(f"\n  Total images     : {totals['images']}")
                    log(f"  Total detections : {totals['detections']}")
            else:
                log(f"[warn] Could not locate split images at {split_imgs}")
        else:
            log(f"[warn] '{args.split}' key not found in {yaml_path.name}")

    # ══════════════════════════════════════════════════════════════
    #  MODE B — images folder provided directly
    #  Auto-detect labels for metrics, annotate regardless
    # ══════════════════════════════════════════════════════════════
    if args.images:
        img_dir = Path(args.images)
        paths   = sorted(p for p in img_dir.rglob("*")
                         if p.suffix.lower() in IMAGE_EXTS)

        if not paths:
            log(f"[warn] No images found in {img_dir}")
        else:
            log(f"\nimages  : {img_dir}  ({len(paths)} found)")

            # -- Auto-locate labels --
            labels_dir  = find_labels_dir(img_dir)
            has_labels  = labels_dir is not None and not args.no_metrics

            if has_labels:
                log(f"labels  : {labels_dir}  (auto-detected)")
                log(f"\n[val] Computing metrics from images + labels ...")

                # Build a minimal temp yaml so model.val() can use these paths
                # Ultralytics resolves labels by replacing 'images' → 'labels'.
                # We create a yaml with the images path as 'val'.
                # For label resolution to work the path must contain 'images';
                # if it doesn't, we symlink into a temp structure.

                parts = img_dir.parts
                has_images_component = any(p == "images" for p in parts)

                if has_images_component:
                    # Standard layout — val() will find labels automatically
                    tmp_yaml = make_temp_yaml(img_dir, labels_dir, model.names)
                    metrics  = run_val(model, tmp_yaml, "val",
                                       args.conf, args.iou,
                                       tuple(args.imgsz), args.device)
                else:
                    # Non-standard layout — create symlink temp structure
                    import tempfile, os
                    tmp_root  = Path(tempfile.mkdtemp())
                    link_imgs = tmp_root / "images" / "val"
                    link_lbls = tmp_root / "labels" / "val"
                    link_imgs.mkdir(parents=True)
                    link_lbls.mkdir(parents=True)
                    # Symlink each file (Windows needs developer mode or admin)
                    for p in paths:
                        lnk = link_imgs / p.name
                        if not lnk.exists():
                            try:
                                os.symlink(p.resolve(), lnk)
                            except (OSError, NotImplementedError):
                                import shutil
                                shutil.copy2(p, lnk)   # fallback: copy
                    for lbl in labels_dir.glob("*.txt"):
                        lnk = link_lbls / lbl.name
                        if not lnk.exists():
                            try:
                                os.symlink(lbl.resolve(), lnk)
                            except (OSError, NotImplementedError):
                                import shutil
                                shutil.copy2(lbl, lnk)

                    tmp_yaml_data = {
                        "path":  str(tmp_root),
                        "val":   "images/val",
                        "nc":    len(model.names),
                        "names": [model.names[i] for i in sorted(model.names)],
                    }
                    tmp_yaml = tmp_root / "eval_temp.yaml"
                    with open(tmp_yaml, "w") as f:
                        yaml.dump(tmp_yaml_data, f)

                    metrics = run_val(model, tmp_yaml, "val",
                                      args.conf, args.iou,
                                      tuple(args.imgsz), args.device)

                print_metrics(metrics, log)

            else:
                if args.no_metrics:
                    log("labels  : skipped (--no-metrics)")
                else:
                    log("labels  : not found — inference only, no P/R/mAP")
                    log("          (expected labels/ folder next to images/ folder)")

            # -- Always annotate --
            log(f"\n[annotate] {len(paths)} images → {out_dir}/")
            totals = run_inference(model, paths, args.conf, args.iou, out_dir, log)
            log(f"\n  Total images     : {totals['images']}")
            log(f"  Total detections : {totals['detections']}")

    # ── Write results.txt ─────────────────────────────────────────
    results_txt.write_text("\n".join(_lines), encoding="utf-8")
    print(f"\n[saved] {results_txt}")
    print(f"[saved] {out_dir}/")


if __name__ == "__main__":
    main()

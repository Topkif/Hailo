#!/usr/bin/env python3
"""
auto_label_sam3.py
------------------
Labelise automatiquement les images d'un dossier avec SAM 3
(Segment Anything Model 3, Meta - novembre 2025) via la
Promptable Concept Segmentation (PCS) basée sur des prompts texte.

Classes détectées :
  - Ballons de foot  → class_id = 1   (prompt "soccer ball")
  - Humains          → class_id = 7   (prompt "person")

Contrairement à SAM/SAM 2, SAM 3 trouve et segmente TOUTES les instances
d'un concept à partir d'une simple phrase, sans détecteur externe.

Sorties : annotations au format YOLO (.txt) + visualisations optionnelles.

Dépendances :
    pip install -U ultralytics            # >= 8.3.237
    pip install opencv-python numpy tqdm
    # CLIP correct pour les prompts texte SAM 3 :
    pip uninstall clip -y
    pip install git+https://github.com/ultralytics/CLIP.git

Poids du modèle :
    Les poids sam3.pt ne sont PAS téléchargés automatiquement.
    Demander l'accès puis télécharger depuis :
        https://huggingface.co/facebook/sam3
    Placer sam3.pt dans le dossier de travail (ou utiliser --model).

Usage :
    python auto_label_sam3.py --input_dir /chemin/images [--output_dir /chemin/labels]
                               [--visualize] [--model sam3.pt] [--conf 0.25]
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ── Configuration des classes ───────────────────────────────────────────────
#
# Chaque entrée : prompt texte SAM 3  →  (class_id YOLO, nom lisible)
#
PROMPT_TO_CLASS = {
    "soccer ball": (1, "ball"),
    "person":      (7, "person"),
}

# Ordre stable des prompts envoyés au prédicteur
TEXT_PROMPTS = list(PROMPT_TO_CLASS.keys())

# Index dans la liste TEXT_PROMPTS → (class_id, nom)
IDX_TO_CLASS = {i: PROMPT_TO_CLASS[p] for i, p in enumerate(TEXT_PROMPTS)}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# Couleurs de visualisation (BGR)
VIS_COLORS = {1: (0, 100, 255), 7: (255, 180, 0)}  # orange ballon, bleu humain


# ── Conversion masque → YOLO ─────────────────────────────────────────────────

def mask_to_yolo_bbox(mask: np.ndarray, img_h: int, img_w: int):
    """Masque binaire → bbox YOLO normalisée (cx, cy, w, h)."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    cx = (x_min + x_max) / 2 / img_w
    cy = (y_min + y_max) / 2 / img_h
    bw = (x_max - x_min) / img_w
    bh = (y_max - y_min) / img_h
    return cx, cy, bw, bh


def mask_to_yolo_polygon(mask: np.ndarray, img_h: int, img_w: int):
    """Masque binaire → polygone YOLO normalisé (segmentation)."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.005 * cv2.arcLength(contour, True)  # simplification légère
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) < 3:
        return None
    pts = approx.reshape(-1, 2)
    normalized = []
    for x, y in pts:
        normalized.extend([x / img_w, y / img_h])
    return normalized


# ── Chargement de SAM 3 ──────────────────────────────────────────────────────

def load_sam3_predictor(model_path: str, conf: float, half: bool):
    """Initialise le SAM3SemanticPredictor pour les prompts texte."""
    try:
        from ultralytics.models.sam import SAM3SemanticPredictor
    except ImportError:
        print(
            "[ERREUR] ultralytics >= 8.3.237 avec support SAM 3 introuvable.\n"
            "         pip install -U ultralytics"
        )
        sys.exit(1)

    if not Path(model_path).exists():
        print(
            f"[ERREUR] Poids '{model_path}' introuvables.\n"
            "         SAM 3 ne télécharge PAS les poids automatiquement.\n"
            "         Demandez l'accès puis téléchargez sam3.pt depuis :\n"
            "             https://huggingface.co/facebook/sam3"
        )
        sys.exit(1)

    overrides = dict(
        conf=conf,
        task="segment",
        mode="predict",
        model=model_path,
        half=half,
        save=False,
        verbose=False,
    )
    predictor = SAM3SemanticPredictor(overrides=overrides)
    print(f"[OK] SAM 3 chargé depuis {model_path}")
    return predictor


# ── Extraction des annotations d'un résultat SAM 3 ──────────────────────────

def extract_annotations(results):
    """
    Transforme la sortie SAM3SemanticPredictor en liste de (class_id, mask, score).
    results est une liste de Results par image; chaque détection porte un .boxes.cls
    qui est l'index dans TEXT_PROMPTS → mapped via IDX_TO_CLASS.
    """
    annotations = []

    # results[0] = Results pour l'unique image envoyée
    res = results[0] if isinstance(results, (list, tuple)) else results

    if res.masks is None or len(res.masks) == 0:
        return annotations

    masks = res.masks.data.cpu().numpy()                    # (N, H, W)
    scores = (res.boxes.conf.cpu().numpy()
              if res.boxes is not None and res.boxes.conf is not None
              else np.ones(len(masks), dtype=float))
    cls_indices = (res.boxes.cls.cpu().numpy().astype(int)
                   if res.boxes is not None and res.boxes.cls is not None
                   else np.zeros(len(masks), dtype=int))

    for mask, score, cls_idx in zip(masks, scores, cls_indices):
        if cls_idx not in IDX_TO_CLASS:
            continue
        class_id, _ = IDX_TO_CLASS[cls_idx]
        annotations.append((class_id, mask.astype(bool), float(score)))

    return annotations


# ── Sauvegarde YOLO ──────────────────────────────────────────────────────────

def save_yolo_labels(annotations, img_h, img_w, out_path: Path,
                     use_segmentation: bool = True):
    """Écrit les annotations au format YOLO (.txt). Retourne le nb de lignes."""
    lines = []
    for class_id, mask, _ in annotations:
        if use_segmentation:
            pts = mask_to_yolo_polygon(mask, img_h, img_w)
            if pts is None:
                continue
            coords = " ".join(f"{v:.6f}" for v in pts)
            lines.append(f"{class_id} {coords}")
        else:
            bbox = mask_to_yolo_bbox(mask, img_h, img_w)
            if bbox is None:
                continue
            cx, cy, bw, bh = bbox
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    out_path.write_text("\n".join(lines))
    return len(lines)


# ── Visualisation ────────────────────────────────────────────────────────────

def visualize_annotations(image_bgr, annotations, out_path: Path):
    """Superpose masques + labels sur l'image et l'enregistre."""
    NAMES = {1: "ball", 7: "person"}
    vis = image_bgr.copy()
    overlay = vis.copy()

    for class_id, mask, score in annotations:
        color = VIS_COLORS.get(class_id, (200, 200, 200))
        overlay[mask] = color

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, color, 2)

        rows = np.any(mask, axis=1)
        if rows.any():
            y_top = int(np.where(rows)[0][0])
            x_left = int(np.where(np.any(mask, axis=0))[0][0])
            label = f"{NAMES.get(class_id, class_id)} {score:.2f}"
            cv2.putText(vis, label, (x_left, max(y_top - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)
    cv2.imwrite(str(out_path), vis)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Labelise des images avec SAM 3 (ballons id=1, humains id=7)"
    )
    p.add_argument("--input_dir",  required=True, help="Dossier des images")
    p.add_argument("--output_dir", default=None,  help="Sortie labels (défaut: input_dir/labels)")
    p.add_argument("--vis_dir",    default=None,  help="Sortie visualisations")
    p.add_argument("--model",      default="sam3.pt", help="Chemin des poids SAM 3")
    p.add_argument("--conf",       type=float, default=0.25, help="Seuil de confiance")
    p.add_argument("--visualize",  action="store_true", help="Génère les visualisations")
    p.add_argument("--segmentation", action="store_true", help="Polygone YOLO au lieu de bbox")
    p.add_argument("--no_half",    action="store_true", help="Désactive le FP16 (utile sur CPU)")
    return p.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"[ERREUR] Dossier introuvable : {input_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "labeled_output"
    images_out = output_dir / "images"
    labels_out = output_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    vis_dir = None
    if args.visualize:
        vis_dir = Path(args.vis_dir) if args.vis_dir else output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

    # ── Chargement du modèle
    predictor = load_sam3_predictor(args.model, args.conf, half=not args.no_half)

    # ── Liste des images
    images = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        print(f"[AVERTISSEMENT] Aucune image dans {input_dir}")
        sys.exit(0)

    print(f"[INFO] {len(images)} image(s) à traiter")
    print(f"[INFO] Prompts SAM 3 : {TEXT_PROMPTS}\n")

    stats = {"total": len(images), "annotated": 0, "skipped": 0,
             "balls": 0, "persons": 0}

    for img_path in tqdm(images, unit="img"):
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            tqdm.write(f"[SKIP] illisible : {img_path.name}")
            stats["skipped"] += 1
            continue
        img_h, img_w = image_bgr.shape[:2]

        try:
            predictor.set_image(str(img_path))
            results = predictor(text=TEXT_PROMPTS)
        except Exception as e:
            tqdm.write(f"[SKIP] erreur inférence {img_path.name} : {e}")
            stats["skipped"] += 1
            continue

        annotations = extract_annotations(results)

        # ── Copie de l'image dans le dossier de sortie
        shutil.copy2(img_path, images_out / img_path.name)

        # ── Écriture du fichier label (même vide → image négative)
        label_path = labels_out / img_path.with_suffix(".txt").name
        n = save_yolo_labels(
            annotations, img_h, img_w, label_path,
            use_segmentation=args.segmentation
        )

        if n > 0:
            stats["annotated"] += 1
            stats["balls"]   += sum(1 for c, *_ in annotations if c == 1)
            stats["persons"] += sum(1 for c, *_ in annotations if c == 7)

        if vis_dir is not None and annotations:
            visualize_annotations(image_bgr, annotations, vis_dir / img_path.name)

    # ── Résumé
    print("\n── Résumé ──────────────────────────────────────────────")
    print(f"  Images traitées  : {stats['total']}")
    print(f"  Avec annotations : {stats['annotated']}")
    print(f"  Images ignorées  : {stats['skipped']}")
    print(f"  Ballons (id=1)   : {stats['balls']}")
    print(f"  Humains (id=7)   : {stats['persons']}")
    print(f"  Images copiées → {images_out}")
    print(f"  Labels écrits  → {labels_out}")
    if vis_dir:
        print(f"  Visualisations → {vis_dir}")
    print("────────────────────────────────────────────────────────")

    # ── classes.txt (index = class_id, donc on remplit les trous)
    max_id = max(cid for cid, _ in PROMPT_TO_CLASS.values())
    names = ["unused"] * (max_id + 1)
    for cid, name in PROMPT_TO_CLASS.values():
        names[cid] = name
    (labels_out / "classes.txt").write_text("\n".join(names) + "\n")
    print(f"[OK] classes.txt écrit (id=1 → ball, id=7 → person)")


if __name__ == "__main__":
    main()

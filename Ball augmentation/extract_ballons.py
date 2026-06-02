"""
extract_ballons.py — Script 1 : Extraction de la banque de ballons via SAM2.

Parcourt les splits train + val du dataset, segmente chaque instance "ballon"
avec SAM2 (bbox prompt), et sauvegarde le résultat en RGBA PNG dans balloon_bank/.

Usage :
    python extract_ballons.py
    python extract_ballons.py --yaml ../Dataset/dataset.yaml --model sam2.1_b.pt
"""

import argparse
import sys
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import yaml

# tqdm est optionnel — fallback silencieux si absent
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):  # type: ignore[misc]
        return it

try:
    from ultralytics import SAM
except ImportError:
    sys.exit("[erreur] ultralytics non installé — pip install ultralytics")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXTRACT_CONFIG: dict = {
    "sam_model": "sam2.1_b.pt",
    "device": "cuda",
    "min_bbox_px": 8,       # côté minimum (px) pour traiter une instance
    "crop_padding_px": 10,  # pixels de contexte autour de la bbox pour SAM
    "splits": ["train", "val"],
    "output_dir": "balloon_bank",
    "sam_imgsz": 1024,
}

# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------

def load_dataset_config(yaml_path: Path) -> tuple[Path, dict[str, Path], int]:
    """
    Lit dataset.yaml et retourne (dataset_root, splits_dirs, ballon_idx).

    splits_dirs : dict split → répertoire images absolu.
    ballon_idx  : index numérique de la classe "ballon" dans ds["names"].
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"dataset.yaml introuvable : {yaml_path}")

    with open(yaml_path, encoding="utf-8") as f:
        ds = yaml.safe_load(f)

    # Résolution du chemin racine
    raw_path = ds.get("path", "")
    dataset_root = Path(raw_path) if raw_path else yaml_path.parent
    if not dataset_root.is_absolute():
        dataset_root = (yaml_path.parent / dataset_root).resolve()

    names: list[str] = ds.get("names", [])
    if "ballon" not in names:
        raise ValueError(
            f"Classe 'ballon' absente de dataset.yaml. Classes trouvées : {names}"
        )
    ballon_idx = names.index("ballon")

    splits_dirs: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        rel = ds.get(split)
        if rel:
            splits_dirs[split] = dataset_root / rel

    return dataset_root, splits_dirs, ballon_idx


def iter_balloon_annotations(
    images_dir: Path,
    labels_dir: Path,
    ballon_idx: int,
    min_bbox_px: int,
) -> Iterator[tuple[Path, Path, int, tuple[float, float, float, float]]]:
    """
    Yield (img_path, lbl_path, bbox_index, yolo_bbox) pour chaque annotation ballon.

    yolo_bbox = (cx, cy, w, h) normalisés.
    Filtre les instances dont min(w_px, h_px) < min_bbox_px.
    """
    if not images_dir.exists():
        print(f"  [avertissement] images_dir inexistant : {images_dir}")
        return

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue

        lbl_path = labels_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        # Lire dimensions image sans décoder entièrement
        img_header = cv2.imread(str(img_path))
        if img_header is None:
            continue
        img_h, img_w = img_header.shape[:2]

        lines = lbl_path.read_text(encoding="utf-8").splitlines()
        bbox_index = 0
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            if cls_id != ballon_idx:
                continue

            cx, cy, w, h = map(float, parts[1:5])

            # Filtre taille minimale
            w_px = w * img_w
            h_px = h * img_h
            if min(w_px, h_px) < min_bbox_px:
                bbox_index += 1
                continue

            yield img_path, lbl_path, bbox_index, (cx, cy, w, h)
            bbox_index += 1


def yolo_bbox_to_xyxy_abs(
    cx: float, cy: float, w: float, h: float, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    """Convertit bbox YOLO normalisée → coordonnées pixels absolues (x1,y1,x2,y2)."""
    half_w = w * img_w / 2
    half_h = h * img_h / 2
    cx_px = cx * img_w
    cy_px = cy * img_h

    x1 = max(0, int(cx_px - half_w))
    y1 = max(0, int(cy_px - half_h))
    x2 = min(img_w - 1, int(cx_px + half_w))
    y2 = min(img_h - 1, int(cy_px + half_h))
    return x1, y1, x2, y2


def crop_with_padding(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    padding: int,
) -> tuple[np.ndarray, int, int]:
    """
    Extrait un crop avec padding, clampé aux bords de l'image.

    Retourne (crop_bgr, offset_x, offset_y) — les offsets permettent de
    recalculer le prompt bbox en coordonnées locales du crop.
    """
    img_h, img_w = img.shape[:2]
    ox = max(0, x1 - padding)
    oy = max(0, y1 - padding)
    ex = min(img_w, x2 + padding)
    ey = min(img_h, y2 + padding)
    return img[oy:ey, ox:ex].copy(), ox, oy


def run_sam_on_crop(
    sam_model: SAM,
    crop_bgr: np.ndarray,
    orig_x1: int, orig_y1: int,
    orig_x2: int, orig_y2: int,
    offset_x: int, offset_y: int,
    device: str,
    imgsz: int,
) -> np.ndarray | None:
    """
    Lance SAM avec un prompt bbox sur le crop.

    Le prompt est converti en coordonnées locales du crop.
    Retourne un masque binaire uint8 (H, W) [0 ou 255], ou None.
    """
    local_x1 = orig_x1 - offset_x
    local_y1 = orig_y1 - offset_y
    local_x2 = orig_x2 - offset_x
    local_y2 = orig_y2 - offset_y

    # Sécurité : bbox doit être dans le crop
    crop_h, crop_w = crop_bgr.shape[:2]
    local_x1 = max(0, min(local_x1, crop_w - 1))
    local_y1 = max(0, min(local_y1, crop_h - 1))
    local_x2 = max(local_x1 + 1, min(local_x2, crop_w))
    local_y2 = max(local_y1 + 1, min(local_y2, crop_h))

    try:
        results = sam_model.predict(
            source=crop_bgr,
            bboxes=[[local_x1, local_y1, local_x2, local_y2]],
            device=device,
            imgsz=imgsz,
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [SAM erreur] {exc}")
        return None

    if not results or results[0].masks is None:
        return None

    mask_data = results[0].masks.data
    if mask_data is None or len(mask_data) == 0:
        return None

    # Premier masque (confiance la plus haute)
    mask_tensor = mask_data[0]
    mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)

    # SAM peut retourner un masque à la résolution d'entrée ou interne — on
    # s'assure qu'il correspond aux dimensions du crop.
    if mask_np.shape[:2] != (crop_bgr.shape[0], crop_bgr.shape[1]):
        mask_np = cv2.resize(
            mask_np,
            (crop_bgr.shape[1], crop_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    return mask_np


def mask_to_rgba(crop_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fusionne un crop BGR + masque SAM → RGBA PNG avec alpha strictement binaire."""
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    # SAM2 retourne des valeurs float [0,1] converties en uint8 — s'assurer que
    # l'alpha est 0 ou 255 uniquement (pas de valeurs intermédiaires issues de
    # l'interpolation interne de SAM).
    _, alpha = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    rgba = np.dstack([crop_rgb, alpha])
    return rgba.astype(np.uint8)


def save_balloon_rgba(
    rgba: np.ndarray,
    output_dir: Path,
    source_stem: str,
    bbox_index: int,
) -> Path:
    """Sauvegarde un patch RGBA en PNG dans output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{source_stem}_{bbox_index}.png"
    # cv2.imwrite attend BGRA — conversion depuis RGBA
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(str(out_path), bgra)
    return out_path


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extraction des ballons en RGBA PNG via SAM2."
    )
    script_dir = Path(__file__).parent
    default_yaml = script_dir.parent / "Dataset" / "dataset.yaml"

    parser.add_argument(
        "--yaml", type=Path, default=default_yaml,
        help=f"Chemin vers dataset.yaml (défaut : {default_yaml})",
    )
    parser.add_argument(
        "--output", type=Path,
        default=script_dir / EXTRACT_CONFIG["output_dir"],
        help="Dossier de sortie balloon_bank/ (défaut : ./balloon_bank)",
    )
    parser.add_argument(
        "--model", type=str, default=EXTRACT_CONFIG["sam_model"],
        help=f"Modèle SAM (défaut : {EXTRACT_CONFIG['sam_model']})",
    )
    parser.add_argument(
        "--device", type=str, default=EXTRACT_CONFIG["device"],
        help="Device torch (cuda / cpu)",
    )
    parser.add_argument(
        "--splits", nargs="+", default=EXTRACT_CONFIG["splits"],
        help="Splits à traiter (défaut : train val)",
    )
    parser.add_argument(
        "--min-px", type=int, default=EXTRACT_CONFIG["min_bbox_px"],
        help="Côté minimum en pixels pour accepter une instance",
    )
    parser.add_argument(
        "--padding", type=int, default=EXTRACT_CONFIG["crop_padding_px"],
        help="Padding en pixels autour de la bbox avant le crop",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Surcharge de la config avec les args CLI
    config = dict(EXTRACT_CONFIG)
    config["sam_model"] = args.model
    config["device"] = args.device
    config["splits"] = args.splits
    config["min_bbox_px"] = args.min_px
    config["crop_padding_px"] = args.padding

    print(f"[1/4] Lecture de {args.yaml}")
    dataset_root, splits_dirs, ballon_idx = load_dataset_config(args.yaml)
    print(f"      dataset_root = {dataset_root}")
    print(f"      ballon_idx   = {ballon_idx}")

    print(f"[2/4] Chargement de SAM : {config['sam_model']} → {config['device']}")
    sam_model = SAM(config["sam_model"])

    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[3/4] Extraction → {output_dir}")

    total = 0
    saved = 0
    skipped_no_mask = 0

    for split in config["splits"]:
        if split not in splits_dirs:
            print(f"  [avertissement] split '{split}' absent de dataset.yaml — ignoré")
            continue

        images_dir = splits_dirs[split]
        labels_dir = dataset_root / "labels" / split

        annotations = list(
            iter_balloon_annotations(
                images_dir, labels_dir, ballon_idx, config["min_bbox_px"]
            )
        )
        print(f"  [{split}] {len(annotations)} instances trouvées")

        for img_path, _lbl_path, bbox_idx, yolo_bbox in tqdm(
            annotations, desc=f"  [{split}]", unit="ballon"
        ):
            total += 1
            img = cv2.imread(str(img_path))
            if img is None:
                skipped_no_mask += 1
                continue

            img_h, img_w = img.shape[:2]
            x1, y1, x2, y2 = yolo_bbox_to_xyxy_abs(*yolo_bbox, img_w, img_h)

            crop, off_x, off_y = crop_with_padding(
                img, x1, y1, x2, y2, config["crop_padding_px"]
            )

            mask = run_sam_on_crop(
                sam_model, crop,
                x1, y1, x2, y2,
                off_x, off_y,
                config["device"], config["sam_imgsz"],
            )

            if mask is None:
                skipped_no_mask += 1
                continue

            rgba = mask_to_rgba(crop, mask)
            save_balloon_rgba(rgba, output_dir, img_path.stem, bbox_idx)
            saved += 1

    print("[4/4] Résumé")
    print(f"      Total analysé   : {total}")
    print(f"      Sauvegardés     : {saved}")
    print(f"      Sans masque     : {skipped_no_mask}")
    print(f"      Sortie          : {output_dir}")


if __name__ == "__main__":
    main()

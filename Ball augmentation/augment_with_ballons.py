"""
augment_with_ballons.py — Script 2 : Incrustation de ballons "en l'air".

Lit la balloon_bank/ générée par extract_ballons.py et incruste 1 à 3 ballons
dans la moitié supérieure de chaque image du dataset. Résultat écrit dans
dataset_augmented/ sans modifier l'original.

Usage :
    python augment_with_ballons.py
    python augment_with_ballons.py --yaml ../Dataset/dataset.yaml --max 2 --seed 42
"""

import argparse
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):  # type: ignore[misc]
        return it

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AUGMENTATION_CONFIG: dict = {
    "balloons_per_image_min": 1,
    "balloons_per_image_max": 3,
    "scale_min": 0.4,
    "scale_max": 1.8,
    "rotation_min_deg": 0.0,
    "rotation_max_deg": 360.0,
    "placement_y_min_frac": 0.02,   # bord supérieur (marge)
    "placement_y_max_frac": 0.50,   # mi-hauteur → "en l'air"
    "blur_kernel_min_px": 5,
    "blur_kernel_max_px": 25,
    "blur_diameter_frac": 0.45,
    "blur_probability": 0.75,       # chance d'appliquer le motion blur
    "edge_feather_px": 3,           # rayon de feathering des bords du masque pour le blend
    "color_match": True,            # adapter la colorimétrie du ballon à la scène
    "luma_strength": 0.85,          # 0=pas d'adaptation luminance, 1=copie exacte
    "chroma_strength": 0.25,        # 0=pas d'adaptation couleur, 1=copie exacte
    "aug_splits": ["train"],
    "output_dir": "dataset_augmented",
    # "bank_dir": "balloon_bank",
    "bank_dir": "filtered_bank",
    "random_seed": None,
    "jpeg_quality": 95,
    "min_placed_area_px": 16,
}

# ---------------------------------------------------------------------------
# Chargement du dataset
# ---------------------------------------------------------------------------

def load_dataset_config(yaml_path: Path) -> tuple[Path, dict[str, Path], int, list[str]]:
    """
    Lit dataset.yaml.

    Retourne (dataset_root, splits_dirs, ballon_idx, names).
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"dataset.yaml introuvable : {yaml_path}")

    with open(yaml_path, encoding="utf-8") as f:
        ds = yaml.safe_load(f)

    raw_path = ds.get("path", "")
    dataset_root = Path(raw_path) if raw_path else yaml_path.parent
    if not dataset_root.is_absolute():
        dataset_root = (yaml_path.parent / dataset_root).resolve()

    names: list[str] = ds.get("names", [])
    if "ballon" not in names:
        raise ValueError(f"Classe 'ballon' absente de dataset.yaml. Classes : {names}")
    ballon_idx = names.index("ballon")

    splits_dirs: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        rel = ds.get(split)
        if rel:
            splits_dirs[split] = dataset_root / rel

    return dataset_root, splits_dirs, ballon_idx, names


# ---------------------------------------------------------------------------
# Chargement de la banque
# ---------------------------------------------------------------------------

def load_balloon_bank(bank_dir: Path) -> list[np.ndarray]:
    """
    Charge tous les PNG RGBA de balloon_bank/.

    Retourne une liste de tableaux uint8 (H, W, 4).
    Lève FileNotFoundError si vide ou absent.
    """
    if not bank_dir.exists():
        raise FileNotFoundError(
            f"balloon_bank/ introuvable : {bank_dir}\n"
            "Lancez d'abord extract_ballons.py."
        )

    pngs = sorted(bank_dir.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"Aucun PNG dans {bank_dir}")

    bank: list[np.ndarray] = []
    for p in pngs:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)  # conserve le canal alpha
        if img is None or img.ndim != 3 or img.shape[2] != 4:
            continue
        # Convertir BGRA → RGBA pour cohérence interne
        img_rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        bank.append(img_rgba)

    if not bank:
        raise ValueError(f"0 PNG RGBA valides chargés depuis {bank_dir}")

    return bank


# ---------------------------------------------------------------------------
# Transformations géométriques
# ---------------------------------------------------------------------------

def transform_balloon_patch(
    rgba: np.ndarray,
    angle_deg: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applique rotation + scaling sur un patch RGBA.

    Canvas carré de côté diag = ceil(sqrt((W*s)² + (H*s)²)) — aucun coin coupé.
    borderValue=(0,0,0,0) → alpha=0 en dehors du ballon.

    Retourne (warped_rgba, warped_alpha).
    """
    h, w = rgba.shape[:2]
    diag = math.ceil(math.sqrt((w * scale) ** 2 + (h * scale) ** 2))
    diag = max(diag, 1)

    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, scale)
    # Centrer le patch dans le canvas carré
    M[0, 2] += (diag - w) / 2.0
    M[1, 2] += (diag - h) / 2.0

    warped = cv2.warpAffine(
        rgba, M, (diag, diag),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    # INTER_LINEAR interpolates alpha → interior pixels get values like 230 instead of 255.
    # Hard-threshold to binary: seamlessClone handles its own boundary blending via the
    # Poisson solver — it does NOT need a pre-feathered mask, and GaussianBlur on small
    # balls (< 40 px) would make the entire body semi-transparent.
    raw_alpha = warped[:, :, 3]
    binary = np.where(raw_alpha > 10, np.uint8(255), np.uint8(0))
    warped[:, :, 3] = binary
    warped_alpha = binary
    return warped, warped_alpha


# ---------------------------------------------------------------------------
# Motion blur
# ---------------------------------------------------------------------------

def _force_odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def compute_blur_kernel_size(diameter_px: float, config: dict) -> int:
    """Taille du noyau proportionnelle au diamètre du ballon, clampée et forcée impaire."""
    ks = int(diameter_px * config["blur_diameter_frac"])
    ks = max(config["blur_kernel_min_px"], min(ks, config["blur_kernel_max_px"]))
    return _force_odd(ks)


def build_motion_blur_kernel(angle_deg: float, kernel_size: int) -> np.ndarray:
    """
    Noyau de flou directionnel (ligne) à angle_deg degrés.

    1. Canvas zeros (ks × ks).
    2. Ligne horizontale → rotation via warpAffine.
    3. Normalisation somme=1.
    """
    ks = _force_odd(kernel_size)
    center = ks // 2
    kernel = np.zeros((ks, ks), dtype=np.float32)
    cv2.line(kernel, (0, center), (ks - 1, center), 1.0, 1)

    M = cv2.getRotationMatrix2D(
        (float(center), float(center)), angle_deg, 1.0
    )
    kernel = cv2.warpAffine(kernel, M, (ks, ks), flags=cv2.INTER_LINEAR)

    total = kernel.sum()
    if total > 0:
        kernel /= total
    return kernel


def apply_motion_blur_rgb(rgba: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Applique le flou directionnel sur les canaux RGB uniquement (alpha inchangé)."""
    result = rgba.copy()
    for c in range(3):
        result[:, :, c] = cv2.filter2D(rgba[:, :, c], -1, kernel)
    return result


# ---------------------------------------------------------------------------
# Calcul de la bbox YOLO après transformation
# ---------------------------------------------------------------------------

def compute_tight_bbox_from_mask(
    alpha_mask: np.ndarray,
    offset_x: int,
    offset_y: int,
    img_w: int,
    img_h: int,
    min_area_px: int = 16,
) -> tuple[float, float, float, float] | None:
    """
    Dérive la bbox YOLO à partir des pixels non-transparents du masque,
    une fois le patch positionné à (offset_x, offset_y) dans l'image destination.

    Algorithme :
        1. findNonZero sur le masque local.
        2. boundingRect pour obtenir (x_l, y_l, w_l, h_l) dans le repère du patch.
        3. Ajout de l'offset → coordonnées absolues dans l'image.
        4. Clamp aux bords puis conversion YOLO normalisée.
    """
    pts = cv2.findNonZero(alpha_mask)
    if pts is None or len(pts) < min_area_px:
        return None

    x_l, y_l, w_l, h_l = cv2.boundingRect(pts)

    abs_x1 = offset_x + x_l
    abs_y1 = offset_y + y_l
    abs_x2 = abs_x1 + w_l
    abs_y2 = abs_y1 + h_l

    # Clamp
    abs_x1 = max(0, min(abs_x1, img_w - 1))
    abs_y1 = max(0, min(abs_y1, img_h - 1))
    abs_x2 = max(0, min(abs_x2, img_w))
    abs_y2 = max(0, min(abs_y2, img_h))

    bw = abs_x2 - abs_x1
    bh = abs_y2 - abs_y1
    if bw <= 0 or bh <= 0:
        return None

    cx = (abs_x1 + bw / 2) / img_w
    cy = (abs_y1 + bh / 2) / img_h
    w = bw / img_w
    h = bh / img_h
    return cx, cy, w, h


# ---------------------------------------------------------------------------
# Incrustation via seamlessClone
# ---------------------------------------------------------------------------

def match_color_to_scene(
    patch_bgr: np.ndarray,
    binary_mask: np.ndarray,
    dst_bgr: np.ndarray,
    ox: int,
    oy: int,
    luma_strength: float,
    chroma_strength: float,
) -> np.ndarray:
    """
    Adapte la colorimétrie du patch à la zone de destination via transfert LAB.

    Principe :
        1. Convertir patch et fond local en LAB (L=luminance, a/b=chrominance).
        2. Calculer mean/std de L, a, b du patch (pixels masqués) et du fond local.
        3. Normaliser le patch : (patch - mean_patch) / std_patch.
        4. Reéchantillonner avec les stats de la scène : * std_scene + mean_scene.
        5. Mélanger avec l'original selon luma_strength / chroma_strength.
           → luma_strength élevé : luminance se fond dans l'ombre/lumière locale.
           → chroma_strength faible : le ballon garde sa couleur distinctive.
    """
    # Région du fond sous le patch (clampée)
    dst_h, dst_w = dst_bgr.shape[:2]
    ph, pw = patch_bgr.shape[:2]
    x1 = max(0, ox);      y1 = max(0, oy)
    x2 = min(dst_w, ox + pw); y2 = min(dst_h, oy + ph)
    if x2 <= x1 or y2 <= y1:
        return patch_bgr

    bg_roi = dst_bgr[y1:y2, x1:x2]
    # Recadrer le patch et masque à la même région si patch déborde
    sx1, sy1 = x1 - ox, y1 - oy
    sx2, sy2 = sx1 + (x2 - x1), sy1 + (y2 - y1)
    patch_crop = patch_bgr[sy1:sy2, sx1:sx2]
    mask_crop  = binary_mask[sy1:sy2, sx1:sx2]

    if cv2.countNonZero(mask_crop) < 9:
        return patch_bgr

    # Conversion en LAB float32 [0,1] range
    patch_lab = cv2.cvtColor(patch_crop.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    bg_lab    = cv2.cvtColor(bg_roi.astype(np.float32)    / 255.0, cv2.COLOR_BGR2Lab)

    fg_pixels = mask_crop > 0

    result_lab = patch_lab.copy()
    for c, strength in enumerate([luma_strength, chroma_strength, chroma_strength]):
        if strength <= 0.0:
            continue
        src_vals = patch_lab[:, :, c][fg_pixels]
        bg_vals  = bg_lab[:, :, c]

        src_mean, src_std = src_vals.mean(), src_vals.std() + 1e-6
        bg_mean,  bg_std  = bg_vals.mean(),  bg_vals.std()  + 1e-6

        # Transfert : centrer sur les stats du fond
        channel = patch_lab[:, :, c].copy()
        normalized = (channel - src_mean) / src_std
        transferred = normalized * bg_std + bg_mean

        # Mélanger : strength=1 → fond, strength=0 → original
        result_lab[:, :, c] = (1.0 - strength) * channel + strength * transferred

    result_bgr_f = cv2.cvtColor(result_lab, cv2.COLOR_Lab2BGR)
    result_bgr = np.clip(result_bgr_f * 255.0, 0, 255).astype(np.uint8)

    # Remettre le résultat dans le patch complet (seule la zone recadrée a changé)
    out = patch_bgr.copy()
    out[sy1:sy2, sx1:sx2] = result_bgr
    return out


def _feather_alpha(binary_alpha: np.ndarray, radius: int) -> np.ndarray:
    """
    Adoucit les bords d'un masque binaire par erosion + blur gaussien.

    L'erosion réduit d'abord le masque de `radius` px pour que le flou
    ne déborde pas au-delà du contour original du ballon.
    """
    if radius <= 0:
        return binary_alpha.astype(np.float32) / 255.0
    ks = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    eroded = cv2.erode(binary_alpha, kernel, iterations=1)
    blurred = cv2.GaussianBlur(eroded.astype(np.float32), (ks * 2 + 1, ks * 2 + 1), radius * 0.8)
    return np.clip(blurred / 255.0, 0.0, 1.0)


def composite_balloon(
    dst_bgr: np.ndarray,
    warped_rgba: np.ndarray,
    warped_alpha: np.ndarray,
    placement_cx: int,
    placement_cy: int,
    edge_feather_px: int = 3,
    luma_strength: float = 0.0,
    chroma_strength: float = 0.0,
) -> tuple[np.ndarray, int, int] | None:
    """
    Incruste le patch dans dst_bgr par alpha blend avec feathering des bords.

    warped_alpha (binaire) est utilisé pour la bbox.
    Un alpha adouci séparé est utilisé uniquement pour le blend visuel.

    Retourne (image_résultat, offset_x, offset_y) ou None si le patch déborde.
    """
    patch_h, patch_w = warped_rgba.shape[:2]
    dst_h, dst_w = dst_bgr.shape[:2]

    ox = placement_cx - patch_w // 2
    oy = placement_cy - patch_h // 2

    dst_x1 = max(0, ox)
    dst_y1 = max(0, oy)
    dst_x2 = min(dst_w, ox + patch_w)
    dst_y2 = min(dst_h, oy + patch_h)

    if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
        return None
    if cv2.countNonZero(warped_alpha) == 0:
        return None

    src_x1 = dst_x1 - ox
    src_y1 = dst_y1 - oy
    src_x2 = src_x1 + (dst_x2 - dst_x1)
    src_y2 = src_y1 + (dst_y2 - dst_y1)

    # Alpha adouci pour le blend visuel uniquement
    blend_alpha_full = _feather_alpha(warped_alpha, edge_feather_px)
    alpha3 = blend_alpha_full[src_y1:src_y2, src_x1:src_x2, np.newaxis]

    result = dst_bgr.copy()
    patch_bgr = cv2.cvtColor(warped_rgba[:, :, :3], cv2.COLOR_RGB2BGR)

    if luma_strength > 0.0 or chroma_strength > 0.0:
        patch_bgr = match_color_to_scene(
            patch_bgr, warped_alpha, dst_bgr, ox, oy, luma_strength, chroma_strength
        )

    roi = result[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.float32)
    patch_roi = patch_bgr[src_y1:src_y2, src_x1:src_x2].astype(np.float32)

    result[dst_y1:dst_y2, dst_x1:dst_x2] = (alpha3 * patch_roi + (1.0 - alpha3) * roi).astype(np.uint8)

    return result, ox, oy


# ---------------------------------------------------------------------------
# Traitement d'une image
# ---------------------------------------------------------------------------

def augment_single_image(
    img_path: Path,
    label_path: Path | None,
    balloon_bank: list[np.ndarray],
    ballon_idx: int,
    config: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str]] | None:
    """
    Ajoute 1 à 3 ballons à une image. Retourne (img_augmentée, lignes_labels) ou None.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    img_h, img_w = img.shape[:2]

    existing_lines: list[str] = []
    if label_path is not None and label_path.exists():
        existing_lines = [
            l for l in label_path.read_text(encoding="utf-8").splitlines() if l.strip()
        ]

    n_balloons = int(rng.integers(
        config["balloons_per_image_min"],
        config["balloons_per_image_max"] + 1,
    ))

    current_img = img.copy()
    added_lines: list[str] = []
    placed_bboxes_abs: list[tuple[int, int, int, int]] = []  # (x1,y1,x2,y2) des ballons placés

    for _ in range(n_balloons):
        # Pioche un RGBA dans la banque
        bank_idx = int(rng.integers(0, len(balloon_bank)))
        rgba = balloon_bank[bank_idx]

        # Transformation géométrique
        angle = float(rng.uniform(config["rotation_min_deg"], config["rotation_max_deg"]))
        scale = float(rng.uniform(config["scale_min"], config["scale_max"]))
        warped_rgba, warped_alpha = transform_balloon_patch(rgba, angle, scale)

        # Motion blur (probabiliste)
        patch_h, patch_w = warped_rgba.shape[:2]
        if rng.random() < config["blur_probability"]:
            diameter = (patch_w + patch_h) / 2.0
            ks = compute_blur_kernel_size(diameter, config)
            blur_angle = float(rng.uniform(0.0, 180.0))
            kernel = build_motion_blur_kernel(blur_angle, ks)
            warped_rgba = apply_motion_blur_rgb(warped_rgba, kernel)

        # Calcul des marges de placement
        half_w = patch_w // 2
        half_h = patch_h // 2

        cx_min = half_w
        cx_max = img_w - half_w - 1
        cy_min_abs = max(half_h, int(img_h * config["placement_y_min_frac"]))
        cy_max_abs = min(
            int(img_h * config["placement_y_max_frac"]),
            img_h - half_h - 1,
        )

        if cx_min >= cx_max or cy_min_abs >= cy_max_abs:
            continue  # patch trop grand pour l'image → on passe

        # Placement sans overlap — on essaie plusieurs positions
        for _ in range(10):
            cx = int(rng.integers(cx_min, cx_max + 1))
            cy = int(rng.integers(cy_min_abs, cy_max_abs + 1))

            # Vérifier l'overlap avec les ballons déjà placés (IoU des bbox absolues)
            ox_try = cx - patch_w // 2
            oy_try = cy - patch_h // 2
            overlap = False
            for prev_bbox in placed_bboxes_abs:
                px1, py1, px2, py2 = prev_bbox
                ix1 = max(ox_try, px1); iy1 = max(oy_try, py1)
                ix2 = min(ox_try + patch_w, px2); iy2 = min(oy_try + patch_h, py2)
                if ix2 > ix1 and iy2 > iy1:
                    overlap = True
                    break
            if overlap:
                continue

            # Incrustation
            composite_result = composite_balloon(
                current_img, warped_rgba, warped_alpha, cx, cy,
                edge_feather_px=config["edge_feather_px"],
                luma_strength=config["luma_strength"] if config["color_match"] else 0.0,
                chroma_strength=config["chroma_strength"] if config["color_match"] else 0.0,
            )
            if composite_result is None:
                continue

            composited, offset_x, offset_y = composite_result

            yolo_bbox = compute_tight_bbox_from_mask(
                warped_alpha, offset_x, offset_y, img_w, img_h,
                config["min_placed_area_px"],
            )
            if yolo_bbox is None:
                continue

            current_img = composited
            bx, by, bw, bh = yolo_bbox
            added_lines.append(f"{ballon_idx} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}")
            placed_bboxes_abs.append((offset_x, offset_y, offset_x + patch_w, offset_y + patch_h))
            break

    if not added_lines:
        return None

    all_lines = existing_lines + added_lines
    return current_img, all_lines


# ---------------------------------------------------------------------------
# Écriture des résultats
# ---------------------------------------------------------------------------

def write_augmented_pair(
    aug_img: np.ndarray,
    label_lines: list[str],
    src_stem: str,
    out_images_dir: Path,
    out_labels_dir: Path,
    jpeg_quality: int,
) -> None:
    """Écrit l'image augmentée en JPEG et les annotations en TXT."""
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(
        str(out_images_dir / f"{src_stem}_aug.jpg"),
        aug_img,
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
    )
    (out_labels_dir / f"{src_stem}_aug.txt").write_text(
        "\n".join(label_lines), encoding="utf-8"
    )


def build_output_yaml(
    original_yaml_path: Path,
    output_root: Path,
    dataset_root: Path,
    names: list[str],
    aug_splits: list[str],
    original_splits_dirs: dict[str, Path],
) -> None:
    """
    Écrit dataset_aug.yaml dans output_root.

    Les splits augmentés pointent vers output_root/images/<split>.
    Les splits non-augmentés pointent vers le dataset original.
    """
    nc = len(names)
    data: dict = {
        "path": str(output_root),
        "nc": nc,
        "names": names,
    }

    for split in ("train", "val", "test"):
        if split in aug_splits:
            data[split] = f"images/{split}"
        elif split in original_splits_dirs:
            data[split] = str(original_splits_dirs[split])

    out_yaml = output_root / "dataset_aug.yaml"
    output_root.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
    print(f"  dataset_aug.yaml → {out_yaml}")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incrustation de ballons 'en l'air' dans le dataset."
    )
    script_dir = Path(__file__).parent
    default_yaml = script_dir.parent / "Dataset" / "dataset.yaml"

    parser.add_argument(
        "--yaml", type=Path, default=default_yaml,
        help=f"Chemin vers dataset.yaml (défaut : {default_yaml})",
    )
    parser.add_argument(
        "--bank", type=Path,
        default=script_dir / AUGMENTATION_CONFIG["bank_dir"],
        help="Dossier balloon_bank/ (défaut : ./balloon_bank)",
    )
    parser.add_argument(
        "--output", type=Path,
        default=script_dir / AUGMENTATION_CONFIG["output_dir"],
        help="Dossier de sortie dataset_augmented/ (défaut : ./dataset_augmented)",
    )
    parser.add_argument(
        "--splits", nargs="+",
        default=AUGMENTATION_CONFIG["aug_splits"],
        help="Splits à augmenter (défaut : train)",
    )
    parser.add_argument(
        "--min", type=int,
        default=AUGMENTATION_CONFIG["balloons_per_image_min"],
        help="Nombre minimum de ballons à ajouter par image",
    )
    parser.add_argument(
        "--max", type=int,
        default=AUGMENTATION_CONFIG["balloons_per_image_max"],
        help="Nombre maximum de ballons à ajouter par image",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Graine aléatoire pour reproductibilité",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = dict(AUGMENTATION_CONFIG)
    config["aug_splits"] = args.splits
    config["balloons_per_image_min"] = args.min
    config["balloons_per_image_max"] = args.max
    config["random_seed"] = args.seed

    rng = np.random.default_rng(config["random_seed"])

    print(f"[1/5] Lecture de {args.yaml}")
    dataset_root, splits_dirs, ballon_idx, names = load_dataset_config(args.yaml)
    print(f"      dataset_root = {dataset_root}")
    print(f"      ballon_idx   = {ballon_idx}  ({names[ballon_idx]})")

    print(f"[2/5] Chargement de la banque : {args.bank}")
    balloon_bank = load_balloon_bank(args.bank)
    print(f"      {len(balloon_bank)} patches RGBA chargés")

    output_root: Path = args.output

    print(f"[3/5] Augmentation → {output_root}")

    total = 0
    augmented = 0
    skipped = 0

    for split in config["aug_splits"]:
        if split not in splits_dirs:
            print(f"  [avertissement] split '{split}' absent — ignoré")
            continue

        images_dir = splits_dirs[split]
        labels_dir = dataset_root / "labels" / split
        out_images_dir = output_root / "images" / split
        out_labels_dir = output_root / "labels" / split

        img_paths = sorted(
            p for p in images_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        print(f"  [{split}] {len(img_paths)} images à traiter")

        for img_path in tqdm(img_paths, desc=f"  [{split}]", unit="img"):
            # Skip COCO-derived images whose names start with zeros (e.g. 000000123456 or 000000123456_1)
            if img_path.stem.startswith('0'):
                skipped += 1
                continue
            total += 1
            label_path = labels_dir / (img_path.stem + ".txt")

            result = augment_single_image(
                img_path, label_path if label_path.exists() else None,
                balloon_bank, ballon_idx, config, rng,
            )

            if result is None:
                skipped += 1
                continue

            aug_img, label_lines = result
            write_augmented_pair(
                aug_img, label_lines, img_path.stem,
                out_images_dir, out_labels_dir, config["jpeg_quality"],
            )
            augmented += 1

    print(f"[4/5] Génération du YAML de sortie")
    build_output_yaml(
        args.yaml, output_root, dataset_root, names,
        config["aug_splits"], splits_dirs,
    )

    print(f"[5/5] Résumé")
    print(f"      Total traité  : {total}")
    print(f"      Augmentées    : {augmented}")
    print(f"      Ignorées      : {skipped}")
    print(f"      Sortie        : {output_root}")


if __name__ == "__main__":
    main()

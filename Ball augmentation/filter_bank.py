"""
filter_bank.py — Déplace dans un dossier trash les ballons trop petits ou trop allongés.

Critères (configurables via dialogs) :
    - Taille minimale : au moins N px sur le côté le plus court
    - Ratio carré    : min(w,h) / max(w,h) >= seuil  (ex: 0.4 = pas plus de 2.5:1)

Usage : python filter_bank.py
"""

import shutil
from pathlib import Path

import cv2
import numpy as np

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog
except ImportError:
    raise SystemExit("tkinter requis (inclus dans Python standard)")


# ---------------------------------------------------------------------------
# Dialogs
# ---------------------------------------------------------------------------

def ask_folder(title: str, initial: Path | None = None) -> Path | None:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(
        title=title,
        initialdir=str(initial) if initial else None,
    )
    root.destroy()
    return Path(folder) if folder else None


def ask_float(title: str, prompt: str, default: float, minval: float, maxval: float) -> float | None:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    val = simpledialog.askfloat(
        title, prompt,
        initialvalue=default,
        minvalue=minval,
        maxvalue=maxval,
    )
    root.destroy()
    return val


def ask_int(title: str, prompt: str, default: int, minval: int, maxval: int) -> int | None:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    val = simpledialog.askinteger(
        title, prompt,
        initialvalue=default,
        minvalue=minval,
        maxvalue=maxval,
    )
    root.destroy()
    return val


def show_summary(moved: int, kept: int, trash_dir: Path) -> None:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    messagebox.showinfo(
        "Résultat",
        f"Terminé.\n\n"
        f"  Conservés : {kept}\n"
        f"  Déplacés  : {moved}\n\n"
        f"Corbeille : {trash_dir}",
    )
    root.destroy()


# ---------------------------------------------------------------------------
# Logique de filtrage
# ---------------------------------------------------------------------------

def get_content_size(rgba: np.ndarray) -> tuple[int, int]:
    """Retourne (w, h) de la bbox des pixels non-transparents."""
    alpha = rgba[:, :, 3]
    pts = cv2.findNonZero(alpha)
    if pts is None:
        return 0, 0
    _, _, w, h = cv2.boundingRect(pts)
    return w, h


def should_trash(rgba: np.ndarray, min_side_px: int, min_square_ratio: float) -> tuple[bool, str]:
    """
    Retourne (True, raison) si le patch doit être mis à la corbeille.

    min_side_px      : côté minimum en pixels (sur le plus court côté)
    min_square_ratio : min(w,h)/max(w,h) minimum (0.4 = pas plus de 2.5:1)
    """
    w, h = get_content_size(rgba)

    if w == 0 or h == 0:
        return True, "masque vide"

    short = min(w, h)
    long_ = max(w, h)

    if short < min_side_px:
        return True, f"trop petit ({short}px < {min_side_px}px)"

    ratio = short / long_
    if ratio < min_square_ratio:
        return True, f"trop allongé (ratio={ratio:.2f} < {min_square_ratio:.2f})"

    return False, ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    script_dir = Path(__file__).parent

    # --- Choisir le dossier source ---
    bank_dir = ask_folder(
        "Sélectionner le dossier balloon_bank (source)",
        initial=script_dir,
    )
    if bank_dir is None:
        print("Annulé.")
        return

    # --- Choisir (ou créer) le dossier trash ---
    default_trash = bank_dir.parent / (bank_dir.name + "_trash")
    trash_dir = ask_folder(
        "Sélectionner le dossier de destination (trash) — sera créé si absent",
        initial=bank_dir.parent,
    )
    if trash_dir is None:
        trash_dir = default_trash
        print(f"Dossier trash par défaut : {trash_dir}")

    # --- Seuils ---
    min_side = ask_int(
        "Taille minimale",
        "Côté minimum en pixels\n(le plus court côté du contenu):",
        default=40, minval=1, maxval=1000,
    )
    if min_side is None:
        print("Annulé.")
        return

    min_ratio = ask_float(
        "Ratio carré minimum",
        "min(w,h) / max(w,h) minimum\n0.4 → accepte jusqu'à 2.5:1\n1.0 → cercle parfait uniquement:",
        default=0.40, minval=0.01, maxval=1.0,
    )
    if min_ratio is None:
        print("Annulé.")
        return

    # --- Scan ---
    pngs = sorted(bank_dir.glob("*.png"))
    if not pngs:
        print(f"Aucun PNG dans {bank_dir}")
        return

    trash_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    kept = 0

    for p in pngs:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim != 3 or img.shape[2] != 4:
            shutil.move(str(p), trash_dir / p.name)
            moved += 1
            print(f"  TRASH (illisible)       {p.name}")
            continue

        trash, reason = should_trash(img, min_side, min_ratio)
        if trash:
            shutil.move(str(p), trash_dir / p.name)
            moved += 1
            print(f"  TRASH ({reason:<30s}) {p.name}")
        else:
            kept += 1

    print(f"\nConservés : {kept}  |  Déplacés : {moved}  |  Trash : {trash_dir}")
    show_summary(moved, kept, trash_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Tao brain mask cho Dataset_Ban (dinh dang kaggle_3m style).

Tai sao can script nay:
- Dataset Kaggle LGG goc chi co image + tumor mask.
- Khong co file brain mask rieng.
- Script nay suy ra brain mask tu giai phau tren anh, roi ghi ra:
    <case>_<slice>_brain_mask.tif

File dau vao moi slice:
- image: <case>_<slice>.tif
- tumor mask: <case>_<slice>_mask.tif

Brain mask duoc tao nhu the nao (infer_brain_mask):
1) Chuyen RGB sang grayscale va lam muot bang Gaussian blur.
2) Tim bien manh bang Canny edge detector.
3) Tim contour ngoai, cham diem moi contour theo:
         score = area - 2.5 * distance_to_image_center
   Sau do giu contour tot nhat (to va gan tam anh).
4) To day contour do thanh vung nhi phan.
5) Hau xu ly morphology:
   - close (lap khe nho), open (loai bo nhieu nho)
   - giu connected component lon nhat
6) Dam bao vung u nam trong brain mask:
         brain = brain OR tumor_mask

Pipeline trong file nay:
- collect_pairs(): quet toan bo cap image/tumor hop le.
- preview mode (mac dinh):
  - chon 1 slice dai dien
  - tao 1 brain mask va 1 anh preview PNG
  - dung lai, chua ghi full dataset
- full mode (--full):
  - tao brain mask cho tat ca cap anh
  - luu *_brain_mask.tif cho tung slice
  - co the dung --overwrite / --max-pairs de kiem soat

Mac dinh la preview truoc (1 mau) de tranh ghi nham hang loat.
Dung --full de chay toan bo.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


@dataclass
class SlicePair:
    case_name: str
    image_path: Path
    tumor_mask_path: Path


def is_tif(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".tif") or name.endswith(".tiff")


def collect_pairs(root: Path) -> List[SlicePair]:
    pairs: List[SlicePair] = []
    for case_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for img_path in sorted([p for p in case_dir.iterdir() if p.is_file() and is_tif(p)]):
            stem = img_path.stem
            if stem.endswith("_mask") or stem.endswith("_brain_mask"):
                continue
            mask_path = img_path.with_name(f"{stem}_mask{img_path.suffix}")
            if not mask_path.exists():
                continue
            pairs.append(
                SlicePair(
                    case_name=case_dir.name,
                    image_path=img_path,
                    tumor_mask_path=mask_path,
                )
            )
    return pairs


def load_rgb(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]
    return arr.astype(np.uint8)


def load_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return (arr > 0).astype(np.uint8)


def keep_largest_component(binary_u8: np.ndarray) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_u8, connectivity=8)
    if n_labels <= 1:
        return binary_u8
    largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(binary_u8)
    out[labels == largest_idx] = 1
    return out


def fill_holes(binary_u8: np.ndarray) -> np.ndarray:
    h, w = binary_u8.shape
    flood = (binary_u8 * 255).copy().astype(np.uint8)
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, mask, seedPoint=(0, 0), newVal=255)
    flood_inv = cv2.bitwise_not(flood)
    filled = ((binary_u8 * 255) | flood_inv) > 0
    return filled.astype(np.uint8)


def infer_brain_mask(image_rgb: np.ndarray, tumor_mask: np.ndarray) -> np.ndarray:
    # Build brain region from strong anatomical boundaries (skull/head contour).
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    low = max(10, int(np.percentile(blur, 30)))
    high = max(low + 20, int(np.percentile(blur, 70)))
    edges = cv2.Canny(blur, low, high)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    brain = np.zeros((h, w), dtype=np.uint8)

    best_contour = None
    best_score = -1.0
    cx, cy = w / 2.0, h / 2.0
    min_area = 0.08 * h * w

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        m = cv2.moments(contour)
        if m["m00"] == 0:
            continue

        x = m["m10"] / m["m00"]
        y = m["m01"] / m["m00"]
        dist = float(np.hypot(x - cx, y - cy))
        score = float(area - 2.5 * dist)

        if score > best_score:
            best_score = score
            best_contour = contour

    if best_contour is not None:
        cv2.drawContours(brain, [best_contour], contourIdx=-1, color=1, thickness=-1)
    else:
        # Fallback for difficult slices: simple foreground threshold.
        thr = max(5, int(np.percentile(blur, 25)))
        brain = (blur > thr).astype(np.uint8)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    brain = cv2.morphologyEx(brain, cv2.MORPH_CLOSE, kernel_close)
    brain = cv2.morphologyEx(brain, cv2.MORPH_OPEN, kernel_open)
    brain = keep_largest_component(brain)

    # Ensure tumor region is always inside brain mask.
    brain = np.where(tumor_mask > 0, 1, brain).astype(np.uint8)
    return brain


def save_mask(mask01: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask01 * 255).astype(np.uint8)).save(out_path)


def overlay_mask(image_rgb: np.ndarray, mask01: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    base = image_rgb.astype(np.float32) / 255.0
    overlay = base.copy()
    alpha = 0.45
    c = np.array(color, dtype=np.float32) / 255.0
    m = mask01.astype(bool)
    overlay[m] = (1.0 - alpha) * overlay[m] + alpha * c
    return np.clip(overlay, 0.0, 1.0)


def pick_preview_pair(pairs: Iterable[SlicePair], case_name: str = "", slice_id: str = "") -> Optional[SlicePair]:
    pairs = list(pairs)
    if not pairs:
        return None

    if case_name and slice_id:
        target_stem = f"{case_name}_{slice_id}"
        for pair in pairs:
            if pair.image_path.stem == target_stem:
                return pair

    # Prefer a sample with positive tumor mask for a meaningful preview.
    for pair in pairs:
        tumor = load_mask(pair.tumor_mask_path)
        if int(tumor.sum()) > 0:
            return pair

    return pairs[0]


def render_preview(pair: SlicePair, brain_mask: np.ndarray, out_path: Path) -> None:
    image = load_rgb(pair.image_path)
    tumor = load_mask(pair.tumor_mask_path)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(tumor, cmap="gray")
    axes[1].set_title("Tumor Mask")
    axes[1].axis("off")

    axes[2].imshow(brain_mask, cmap="gray")
    axes[2].set_title("Generated Brain Mask")
    axes[2].axis("off")

    combo = overlay_mask(image, brain_mask, (0, 180, 255))
    combo = overlay_mask((combo * 255).astype(np.uint8), tumor, (255, 80, 80))
    axes[3].imshow(combo)
    axes[3].set_title("Overlay (Brain + Tumor)")
    axes[3].axis("off")

    fig.suptitle(f"{pair.case_name} | {pair.image_path.name}", fontsize=11)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_full(pairs: Iterable[SlicePair], overwrite: bool, max_pairs: int) -> Tuple[int, int]:
    saved = 0
    skipped = 0
    count = 0

    for pair in pairs:
        if max_pairs > 0 and count >= max_pairs:
            break
        count += 1

        out_path = pair.image_path.with_name(f"{pair.image_path.stem}_brain_mask{pair.image_path.suffix}")
        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        image = load_rgb(pair.image_path)
        tumor = load_mask(pair.tumor_mask_path)
        brain = infer_brain_mask(image, tumor)
        save_mask(brain, out_path)
        saved += 1

    return saved, skipped


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_root = script_dir / "lgg-mri-segmentation" / "kaggle_3m"
    default_preview = script_dir / "previews" / "brain_mask_preview.png"

    parser = argparse.ArgumentParser(description="Generate brain masks for Dataset_Ban.")
    parser.add_argument("--root", type=Path, default=default_root, help="Path to kaggle_3m root.")
    parser.add_argument("--full", action="store_true", help="Process full dataset and save *_brain_mask.tif files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing *_brain_mask.tif files.")
    parser.add_argument("--max-pairs", type=int, default=0, help="Limit number of pairs in full mode (0 = all).")
    parser.add_argument("--preview-case", type=str, default="", help="Case id for preview, e.g. TCGA_CS_4941_19960909.")
    parser.add_argument("--preview-slice", type=str, default="", help="Slice id for preview, e.g. 11.")
    parser.add_argument("--preview-output", type=Path, default=default_preview, help="Output PNG for preview.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()

    if not root.exists() or not root.is_dir():
        print(f"Invalid root folder: {root}")
        return 1

    pairs = collect_pairs(root)
    if not pairs:
        print(f"No valid image/mask pairs found under: {root}")
        return 1

    print(f"Found {len(pairs)} image/mask pairs.")

    preview_pair = pick_preview_pair(pairs, case_name=args.preview_case, slice_id=args.preview_slice)
    if preview_pair is None:
        print("Could not pick preview sample.")
        return 1

    preview_image = load_rgb(preview_pair.image_path)
    preview_tumor = load_mask(preview_pair.tumor_mask_path)
    preview_brain = infer_brain_mask(preview_image, preview_tumor)
    render_preview(preview_pair, preview_brain, args.preview_output.resolve())

    print(f"Preview saved: {args.preview_output.resolve()}")
    print(f"Preview pair: {preview_pair.case_name} / {preview_pair.image_path.name}")

    if not args.full:
        print("Preview-only mode completed.")
        print("Use --full to generate *_brain_mask.tif for all slices.")
        return 0

    saved, skipped = process_full(pairs, overwrite=args.overwrite, max_pairs=args.max_pairs)
    print(f"Full mode completed. Saved={saved}, Skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

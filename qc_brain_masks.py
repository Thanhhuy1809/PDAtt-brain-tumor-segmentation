#!/usr/bin/env python3
"""Generate a quality-check grid for Dataset_Ban brain masks."""

from __future__ import annotations

from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def collect_pairs(root: Path):
    pairs = []
    for case in sorted([p for p in root.iterdir() if p.is_dir()]):
        for img in sorted(case.glob("*.tif")):
            stem = img.stem
            if stem.endswith("_mask") or stem.endswith("_brain_mask"):
                continue

            tumor = case / f"{stem}_mask.tif"
            brain = case / f"{stem}_brain_mask.tif"
            if tumor.exists() and brain.exists():
                pairs.append((case.name, img, tumor, brain))
    return pairs


def load_rgb(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path)).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[-1] > 3:
        arr = arr[:, :, :3]
    return arr


def main() -> int:
    root = Path("PBL4/PDAtt-Unet-main2/PDAtt-Unet-main2/Dataset_Ban/lgg-mri-segmentation/kaggle_3m")
    out = Path("PBL4/PDAtt-Unet-main2/PDAtt-Unet-main2/Dataset_Ban/previews/brain_mask_qc_grid.png")

    pairs = collect_pairs(root)
    if not pairs:
        print("No valid image/tumor/brain pairs found.")
        return 1

    pos = []
    neg = []
    for item in pairs:
        _, _, tumor_path, _ = item
        tumor = np.array(Image.open(tumor_path)) > 0
        if tumor.any():
            pos.append(item)
        else:
            neg.append(item)

    random.seed(42)
    random.shuffle(pos)
    random.shuffle(neg)

    selected = pos[:8] + neg[:4]
    if len(selected) < 12:
        selected = (pos + neg)[:12]

    cols = 4
    rows = (len(selected) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.3 * cols, 4.3 * rows))
    axes = np.atleast_2d(axes)

    for ax in axes.flat:
        ax.axis("off")

    for idx, (case_name, img_path, tumor_path, brain_path) in enumerate(selected):
        r, c = divmod(idx, cols)
        ax = axes[r, c]

        img = load_rgb(img_path)
        tumor = np.array(Image.open(tumor_path)) > 0
        brain = np.array(Image.open(brain_path)) > 0

        overlay = img.astype(np.float32) / 255.0
        overlay[brain] = 0.65 * overlay[brain] + 0.35 * np.array([0.10, 0.75, 1.00], dtype=np.float32)
        overlay[tumor] = 0.35 * overlay[tumor] + 0.65 * np.array([1.00, 0.35, 0.35], dtype=np.float32)

        tp = int(tumor.sum())
        bp = int(brain.sum())
        ratio = 100.0 * tp / max(1, bp)

        ax.imshow(np.clip(overlay, 0.0, 1.0))
        ax.set_title(f"{case_name}\n{img_path.name}\nTumor/Brain: {ratio:.2f}%", fontsize=8)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved QC grid: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Standalone prediction script for Dataset_Ban using PYAttUNet checkpoint.

Supports:
- Single image inference (--image)
- Batch inference from kaggle_3m-style root (--root)

Output:
- Predicted tumor mask PNG
- Predicted brain mask PNG
- Overlay visualization PNG
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


@dataclass
class Sample:
    case_name: str
    image_path: Path
    tumor_mask_path: Optional[Path]
    brain_mask_path: Optional[Path]


def is_tif(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".tif") or name.endswith(".tiff")


def collect_samples(root: Path) -> List[Sample]:
    samples: List[Sample] = []
    case_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

    for case_dir in case_dirs:
        for image_path in sorted([p for p in case_dir.iterdir() if p.is_file() and is_tif(p)]):
            stem = image_path.stem
            if stem.endswith("_mask") or stem.endswith("_brain_mask"):
                continue

            tumor_mask_path = image_path.with_name(f"{stem}_mask{image_path.suffix}")
            brain_mask_path = image_path.with_name(f"{stem}_brain_mask{image_path.suffix}")

            samples.append(
                Sample(
                    case_name=case_dir.name,
                    image_path=image_path,
                    tumor_mask_path=tumor_mask_path if tumor_mask_path.exists() else None,
                    brain_mask_path=brain_mask_path if brain_mask_path.exists() else None,
                )
            )

    return samples


def load_rgb(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]
    return arr.astype(np.uint8)


def load_binary_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return (arr > 0).astype(np.uint8)


def add_project_root_to_syspath(script_path: Path) -> Path:
    project_root = script_path.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


def preprocess_image(image_rgb: np.ndarray, img_size: int) -> Tuple[np.ndarray, torch.Tensor]:
    pil_img = Image.fromarray(image_rgb)
    pil_img = pil_img.resize((img_size, img_size), resample=Image.BILINEAR)
    resized_rgb = np.array(pil_img).astype(np.uint8)

    x = resized_rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = torch.from_numpy(x).unsqueeze(0)
    return resized_rgb, x


def resize_mask(mask01: np.ndarray, width: int, height: int) -> np.ndarray:
    pil = Image.fromarray((mask01 > 0).astype(np.uint8) * 255)
    pil = pil.resize((width, height), resample=Image.NEAREST)
    return (np.array(pil) > 0).astype(np.uint8)


def overlay_mask(image_rgb: np.ndarray, mask01: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    base = image_rgb.astype(np.float32) / 255.0
    out = base.copy()
    c = np.array(color, dtype=np.float32) / 255.0
    m = mask01.astype(bool)
    out[m] = (1.0 - alpha) * out[m] + alpha * c
    return np.clip(out, 0.0, 1.0)


def predict_one(
    model: torch.nn.Module,
    image_rgb: np.ndarray,
    img_size: int,
    device: torch.device,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    resized_rgb, x = preprocess_image(image_rgb, img_size)
    x = x.to(device)

    with torch.no_grad():
        out_tumor, out_brain = model(x)
        pred_tumor = (torch.sigmoid(out_tumor) > threshold).float()
        pred_brain = (torch.sigmoid(out_brain) > threshold).float()

    pred_tumor_np = pred_tumor.squeeze().cpu().numpy().astype(np.uint8)
    pred_brain_np = pred_brain.squeeze().cpu().numpy().astype(np.uint8)
    return resized_rgb, pred_tumor_np, pred_brain_np


def _mask_to_rgb(mask01: np.ndarray) -> np.ndarray:
    m = (mask01 > 0).astype(np.uint8) * 255
    return np.stack([m, m, m], axis=-1)


def _hstack_with_gap(images: List[np.ndarray], gap: int = 6) -> np.ndarray:
    if not images:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    h = images[0].shape[0]
    gap_img = np.full((h, gap, 3), 255, dtype=np.uint8)
    out = []
    for idx, img in enumerate(images):
        out.append(img)
        if idx < len(images) - 1:
            out.append(gap_img)
    return np.concatenate(out, axis=1)


def save_visualization(
    image_rgb: np.ndarray,
    pred_tumor: np.ndarray,
    pred_brain: np.ndarray,
    out_png: Path,
    gt_tumor: Optional[np.ndarray] = None,
    gt_brain: Optional[np.ndarray] = None,
) -> None:
    overlay = overlay_mask(image_rgb, pred_brain, (0, 180, 255))
    overlay = overlay_mask((overlay * 255).astype(np.uint8), pred_tumor, (255, 80, 80))
    row1 = _hstack_with_gap(
        [
            image_rgb.astype(np.uint8),
            _mask_to_rgb(pred_tumor),
            _mask_to_rgb(pred_brain),
            overlay.astype(np.uint8),
        ]
    )

    if gt_tumor is None and gt_brain is None:
        canvas = row1
    else:
        gt_t = _mask_to_rgb(gt_tumor if gt_tumor is not None else np.zeros_like(pred_tumor))
        gt_b = _mask_to_rgb(gt_brain if gt_brain is not None else np.zeros_like(pred_brain))
        blank = np.full_like(image_rgb, 255, dtype=np.uint8)
        row2 = _hstack_with_gap([blank, gt_t, gt_b, blank])
        gap_h = np.full((8, row1.shape[1], 3), 255, dtype=np.uint8)
        canvas = np.concatenate([row1, gap_h, row2], axis=0)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out_png)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_root = script_dir / "lgg-mri-segmentation" / "kaggle_3m"
    default_ckpt = script_dir / "outputs" / "pdatt_dataset_ban" / "best.pt"
    default_out = script_dir / "outputs" / "predictions"

    parser = argparse.ArgumentParser(description="Predict tumor/brain masks using PYAttUNet checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=default_ckpt, help="Path to checkpoint (best.pt/final.pt)")
    parser.add_argument("--output", type=Path, default=default_out, help="Output directory")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--image", type=Path, default=None, help="Single image path (*.tif)")
    parser.add_argument("--root", type=Path, default=default_root, help="kaggle_3m root for batch prediction")
    parser.add_argument("--num-samples", type=int, default=12, help="Number of samples in batch mode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full", action="store_true", help="Predict all samples under --root")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        return 1

    script_path = Path(__file__).resolve()
    add_project_root_to_syspath(script_path)
    import Architectures as networks  # pylint: disable=import-error

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    model = getattr(networks, "PYAttUNet")().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(model_state)
    model.eval()

    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.image is not None:
        if not args.image.exists():
            print(f"Image not found: {args.image}")
            return 1

        image_path = args.image.resolve()
        image_rgb = load_rgb(image_path)
        resized_rgb, pred_tumor, pred_brain = predict_one(model, image_rgb, args.img_size, device, args.threshold)

        stem = image_path.stem
        pred_tumor_path = out_dir / f"{stem}_pred_tumor.png"
        pred_brain_path = out_dir / f"{stem}_pred_brain.png"
        overlay_path = out_dir / f"{stem}_overlay.png"

        Image.fromarray((pred_tumor * 255).astype(np.uint8)).save(pred_tumor_path)
        Image.fromarray((pred_brain * 255).astype(np.uint8)).save(pred_brain_path)
        save_visualization(resized_rgb, pred_tumor, pred_brain, overlay_path)

        print(f"Saved: {pred_tumor_path}")
        print(f"Saved: {pred_brain_path}")
        print(f"Saved: {overlay_path}")
        return 0

    root = args.root.resolve()
    if not root.exists() or not root.is_dir():
        print(f"Invalid root folder: {root}")
        return 1

    samples = collect_samples(root)
    if not samples:
        print(f"No valid samples found under: {root}")
        return 1

    if args.full:
        selected = samples
    else:
        rng = random.Random(args.seed)
        k = min(max(1, args.num_samples), len(samples))
        selected = rng.sample(samples, k)

    print(f"Predicting {len(selected)} samples...")

    for idx, sample in enumerate(selected, start=1):
        image_rgb = load_rgb(sample.image_path)
        resized_rgb, pred_tumor, pred_brain = predict_one(model, image_rgb, args.img_size, device, args.threshold)

        rel_name = f"{sample.case_name}_{sample.image_path.stem}"
        pred_tumor_path = out_dir / f"{rel_name}_pred_tumor.png"
        pred_brain_path = out_dir / f"{rel_name}_pred_brain.png"
        overlay_path = out_dir / f"{rel_name}_overlay.png"

        Image.fromarray((pred_tumor * 255).astype(np.uint8)).save(pred_tumor_path)
        Image.fromarray((pred_brain * 255).astype(np.uint8)).save(pred_brain_path)

        gt_tumor = None
        gt_brain = None
        if sample.tumor_mask_path is not None:
            gt_tumor = load_binary_mask(sample.tumor_mask_path)
            gt_tumor = resize_mask(gt_tumor, args.img_size, args.img_size)
        if sample.brain_mask_path is not None:
            gt_brain = load_binary_mask(sample.brain_mask_path)
            gt_brain = resize_mask(gt_brain, args.img_size, args.img_size)

        save_visualization(resized_rgb, pred_tumor, pred_brain, overlay_path, gt_tumor=gt_tumor, gt_brain=gt_brain)

        print(f"[{idx}/{len(selected)}] Saved overlay: {overlay_path.name}")

    print(f"Done. Outputs in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

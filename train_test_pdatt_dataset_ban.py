#!/usr/bin/env python3
"""
Train/Test PYAttUNet directly on Dataset_Ban (kaggle_3m TIFF folders).

Expected files per slice:
- image: <case>_<slice>.tif
- tumor mask: <case>_<slice>_mask.tif
- brain mask: <case>_<slice>_brain_mask.tif

Examples:
- Sanity train: python train_test_pdatt_dataset_ban.py --mode train --epochs 1 --max-train-batches 2 --max-val-batches 2
- Full train  : python train_test_pdatt_dataset_ban.py --mode train --epochs 60
- Test best   : python train_test_pdatt_dataset_ban.p7y --mode test --checkpoint outputs/pdatt_dataset_ban/best.pt
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import albumentations as A
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


@dataclass
class Sample:
    case_name: str
    image_path: Path
    tumor_mask_path: Path
    brain_mask_path: Path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

            if not tumor_mask_path.exists() or not brain_mask_path.exists():
                continue

            samples.append(
                Sample(
                    case_name=case_dir.name,
                    image_path=image_path,
                    tumor_mask_path=tumor_mask_path,
                    brain_mask_path=brain_mask_path,
                )
            )
    return samples


def split_by_case(
    samples: Sequence[Sample],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, List[Sample]]:
    case_to_samples: Dict[str, List[Sample]] = {}
    for s in samples:
        case_to_samples.setdefault(s.case_name, []).append(s)

    cases = list(case_to_samples.keys())
    rng = random.Random(seed)
    rng.shuffle(cases)

    n_cases = len(cases)
    n_train = int(n_cases * train_ratio)
    n_val = int(n_cases * val_ratio)

    train_cases = set(cases[:n_train])
    val_cases = set(cases[n_train:n_train + n_val])
    test_cases = set(cases[n_train + n_val:])

    train_samples: List[Sample] = []
    val_samples: List[Sample] = []
    test_samples: List[Sample] = []

    for case in train_cases:
        train_samples.extend(case_to_samples[case])
    for case in val_cases:
        val_samples.extend(case_to_samples[case])
    for case in test_cases:
        test_samples.extend(case_to_samples[case])

    return {
        "train": sorted(train_samples, key=lambda x: str(x.image_path)),
        "val": sorted(val_samples, key=lambda x: str(x.image_path)),
        "test": sorted(test_samples, key=lambda x: str(x.image_path)),
    }


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


def build_edge_mask(mask01: np.ndarray) -> np.ndarray:
    kernel = np.ones((5, 5), np.uint8)
    edge = cv2.morphologyEx((mask01 * 255).astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
    return (edge > 0).astype(np.uint8)


class DatasetBan(Dataset):
    def __init__(self, samples: Sequence[Sample], transform: A.Compose):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        s = self.samples[index]

        image = load_rgb(s.image_path)
        tumor = load_binary_mask(s.tumor_mask_path)
        brain = load_binary_mask(s.brain_mask_path)
        edge = build_edge_mask(tumor)

        aug = self.transform(image=image, masks=[tumor, brain, edge])

        x = aug["image"]
        y_tumor = aug["masks"][0].float().unsqueeze(0)
        y_brain = aug["masks"][1].float().unsqueeze(0)
        y_edge = aug["masks"][2].float().unsqueeze(0)

        return x, y_tumor, y_brain, y_edge


def get_transforms(img_size: int) -> Tuple[A.Compose, A.Compose]:
    train_tf = A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Rotate(limit=35, p=0.8),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )
    val_tf = A.Compose(
        [
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )
    return train_tf, val_tf


def dice_iou_from_logits(logits: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    probs = torch.sigmoid(logits)
    pred = (probs > 0.5).float()

    inter = (pred * target).sum(dim=(1, 2, 3))
    union_dice = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    union_iou = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter

    dice = ((2.0 * inter + 1e-8) / (union_dice + 1e-8)).mean().item()
    iou = ((inter + 1e-8) / (union_iou + 1e-8)).mean().item()
    return dice, iou


def confusion_from_logits(logits: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    pred = (probs > 0.5).float()
    tgt = (target > 0.5).float()

    tp = float(((pred == 1) & (tgt == 1)).sum().item())
    tn = float(((pred == 0) & (tgt == 0)).sum().item())
    fp = float(((pred == 1) & (tgt == 0)).sum().item())
    fn = float(((pred == 0) & (tgt == 1)).sum().item())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def metrics_from_confusion(tp: float, tn: float, fp: float, fn: float) -> Dict[str, float]:
    eps = 1e-8
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    sens = tp / (tp + fn + eps)
    spec = tn / (tn + fp + eps)
    f1 = 2.0 * prec * sens / (prec + sens + eps)
    return {
        "acc": float(acc),
        "f1": float(f1),
        "prec": float(prec),
        "sens": float(sens),
        "spec": float(spec),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int,
) -> Dict[str, float]:
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_tp = 0.0
    total_tn = 0.0
    total_fp = 0.0
    total_fn = 0.0
    steps = 0

    for batch_idx, batch in enumerate(tqdm(loader, leave=False)):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        x, y_tumor, y_brain, y_edge = batch
        x = x.to(device)
        y_tumor = y_tumor.to(device)
        y_brain = y_brain.to(device)
        y_edge = y_edge.to(device)

        with torch.set_grad_enabled(is_train):
            out_tumor, out_brain = model(x)

            loss_tumor = criterion(out_tumor, y_tumor)
            loss_brain = criterion(out_brain, y_brain)
            edge_logits = out_tumor * y_edge
            loss_edge = criterion(edge_logits, y_edge)

            loss = 0.7 * loss_tumor + 0.3 * loss_brain + 2.0 * loss_edge

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        dice, iou = dice_iou_from_logits(out_tumor.detach(), y_tumor)
        conf = confusion_from_logits(out_tumor.detach(), y_tumor)
        total_loss += float(loss.item())
        total_dice += dice
        total_iou += iou
        total_tp += conf["tp"]
        total_tn += conf["tn"]
        total_fp += conf["fp"]
        total_fn += conf["fn"]
        steps += 1

    if steps == 0:
        return {
            "loss": 0.0,
            "dice": 0.0,
            "iou": 0.0,
            "acc": 0.0,
            "f1": 0.0,
            "prec": 0.0,
            "sens": 0.0,
            "spec": 0.0,
        }

    cls_metrics = metrics_from_confusion(total_tp, total_tn, total_fp, total_fn)

    return {
        "loss": total_loss / steps,
        "dice": total_dice / steps,
        "iou": total_iou / steps,
        "acc": cls_metrics["acc"],
        "f1": cls_metrics["f1"],
        "prec": cls_metrics["prec"],
        "sens": cls_metrics["sens"],
        "spec": cls_metrics["spec"],
    }


def add_project_root_to_syspath(script_path: Path) -> Path:
    project_root = script_path.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


def save_history_csv(history: List[Dict[str, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "epoch",
        "train_loss",
        "val_loss",
        "train_dice",
        "val_dice",
        "train_iou",
        "val_iou",
        "train_acc",
        "val_acc",
        "train_f1",
        "val_f1",
        "train_prec",
        "val_prec",
        "train_sens",
        "val_sens",
        "train_spec",
        "val_spec",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def save_history_plot(history: List[Dict[str, float]], out_path: Path) -> None:
    if not history:
        return

    epochs = [int(h["epoch"]) for h in history]
    train_loss = [float(h["train_loss"]) for h in history]
    val_loss = [float(h["val_loss"]) for h in history]
    train_dice = [float(h["train_dice"]) for h in history]
    val_dice = [float(h["val_dice"]) for h in history]
    train_iou = [float(h["train_iou"]) for h in history]
    val_iou = [float(h["val_iou"]) for h in history]
    train_acc = [float(h["train_acc"]) for h in history]
    val_acc = [float(h["val_acc"]) for h in history]
    train_f1 = [float(h["train_f1"]) for h in history]
    val_f1 = [float(h["val_f1"]) for h in history]
    train_prec = [float(h["train_prec"]) for h in history]
    val_prec = [float(h["val_prec"]) for h in history]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.0))

    axes[0, 0].plot(epochs, train_loss, label="train", linewidth=2)
    axes[0, 0].plot(epochs, val_loss, label="val", linewidth=2)
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, train_dice, label="train", linewidth=2)
    axes[0, 1].plot(epochs, val_dice, label="val", linewidth=2)
    axes[0, 1].set_title("Dice")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[0, 2].plot(epochs, train_iou, label="train", linewidth=2)
    axes[0, 2].plot(epochs, val_iou, label="val", linewidth=2)
    axes[0, 2].set_title("IoU")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()

    axes[1, 0].plot(epochs, train_acc, label="train", linewidth=2)
    axes[1, 0].plot(epochs, val_acc, label="val", linewidth=2)
    axes[1, 0].set_title("Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, train_f1, label="train", linewidth=2)
    axes[1, 1].plot(epochs, val_f1, label="val", linewidth=2)
    axes[1, 1].set_title("F1")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    axes[1, 2].plot(epochs, train_prec, label="train", linewidth=2)
    axes[1, 2].plot(epochs, val_prec, label="val", linewidth=2)
    axes[1, 2].set_title("Precision")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def train_main(args: argparse.Namespace) -> int:
    seed_everything(args.seed)
    script_path = Path(__file__).resolve()
    project_root = add_project_root_to_syspath(script_path)

    import Architectures as networks  # pylint: disable=import-error

    root = args.root.resolve()
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_samples(root)
    if not samples:
        print(f"No valid samples found under: {root}")
        return 1

    split = split_by_case(samples, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed)
    print(f"Project root: {project_root}")
    print(f"Total={len(samples)} | Train={len(split['train'])} | Val={len(split['val'])} | Test={len(split['test'])}")

    train_tf, val_tf = get_transforms(args.img_size)

    train_set = DatasetBan(split["train"], train_tf)
    val_set = DatasetBan(split["val"], val_tf)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = getattr(networks, "PYAttUNet")().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_dice = -1.0
    best_path = out_dir / "best.pt"
    final_path = out_dir / "final.pt"
    history_csv = out_dir / "history.csv"
    history_png = out_dir / "history.png"
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            max_batches=args.max_train_batches,
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            criterion=criterion,
            device=device,
            max_batches=args.max_val_batches,
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train loss={train_metrics['loss']:.4f} dice={train_metrics['dice']:.4f} iou={train_metrics['iou']:.4f} "
            f"acc={train_metrics['acc']:.4f} f1={train_metrics['f1']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} dice={val_metrics['dice']:.4f} iou={val_metrics['iou']:.4f} "
            f"acc={val_metrics['acc']:.4f} f1={val_metrics['f1']:.4f}"
        )

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_dice": train_metrics["dice"],
                "val_dice": val_metrics["dice"],
                "train_iou": train_metrics["iou"],
                "val_iou": val_metrics["iou"],
                "train_acc": train_metrics["acc"],
                "val_acc": val_metrics["acc"],
                "train_f1": train_metrics["f1"],
                "val_f1": val_metrics["f1"],
                "train_prec": train_metrics["prec"],
                "val_prec": val_metrics["prec"],
                "train_sens": train_metrics["sens"],
                "val_sens": val_metrics["sens"],
                "train_spec": train_metrics["spec"],
                "val_spec": val_metrics["spec"],
            }
        )
        save_history_csv(history, history_csv)
        save_history_plot(history, history_png)

        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
            "val_metrics": val_metrics,
        }

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(ckpt, best_path)
            print(f"Saved best checkpoint: {best_path}")

    torch.save({"model_state": model.state_dict(), "args": vars(args)}, final_path)
    print(f"Saved final checkpoint: {final_path}")
    print(f"Saved history csv: {history_csv}")
    print(f"Saved history plot: {history_png}")
    return 0


def test_main(args: argparse.Namespace) -> int:
    seed_everything(args.seed)
    script_path = Path(__file__).resolve()
    add_project_root_to_syspath(script_path)

    import Architectures as networks  # pylint: disable=import-error

    root = args.root.resolve()

    samples = collect_samples(root)
    if not samples:
        print(f"No valid samples found under: {root}")
        return 1

    split = split_by_case(samples, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed)
    test_samples = split["test"]
    if not test_samples:
        print("Test split is empty. Adjust train/val ratios.")
        return 1

    _, val_tf = get_transforms(args.img_size)
    test_set = DatasetBan(test_samples, val_tf)
    test_loader = DataLoader(
        test_set,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = getattr(networks, "PYAttUNet")().to(device)

    ckpt_path = args.checkpoint.resolve()
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return 1

    ckpt = torch.load(ckpt_path, map_location=device)
    model_state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(model_state)

    criterion = nn.BCEWithLogitsLoss()
    metrics = run_epoch(
        model=model,
        loader=test_loader,
        optimizer=None,
        criterion=criterion,
        device=device,
        max_batches=args.max_val_batches,
    )

    print(
        f"Test  | loss={metrics['loss']:.4f} dice={metrics['dice']:.4f} iou={metrics['iou']:.4f} "
        f"acc={metrics['acc']:.4f} f1={metrics['f1']:.4f} prec={metrics['prec']:.4f} "
        f"sens={metrics['sens']:.4f} spec={metrics['spec']:.4f} "
        f"on {len(test_samples)} samples"
    )
    return 0


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_root = script_dir / "lgg-mri-segmentation" / "kaggle_3m"
    default_out = script_dir / "outputs" / "pdatt_dataset_ban"

    parser = argparse.ArgumentParser(description="Train/Test PYAttUNet on Dataset_Ban")
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--output", type=Path, default=default_out)
    parser.add_argument("--checkpoint", type=Path, default=default_out / "best.pt")

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--val-batch-size", type=int, default=12)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--max-train-batches", type=int, default=0, help="0 means full epoch")
    parser.add_argument("--max-val-batches", type=int, default=0, help="0 means full epoch")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "train":
        return train_main(args)
    return test_main(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""
eval_checkpoint.py — честная оценка чекпоинта на val-сете с ЕДИНЫМ (исправленным)
метрическим определением: пустые патчи исключаются из Dice/IoU.
Используется, чтобы сравнивать модели между собой объективно (старый код завышал
метрику, считая пустые патчи идеальными).
"""
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import AstroDataset
from common import load_model_and_meta, MASK_THR_DEFAULT, BOUND_THR_DEFAULT, BOUND_DILATE_DEFAULT
from train import run_epoch, BCEDice
from model import build_model

ROOT = os.getcwd()
VAL_DIR = os.path.join(ROOT, "data", "val")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, size, tile_meta = load_model_and_meta(args.model, device)
    imagenet_norm = tile_meta["imagenet_norm"]
    print(f"[eval] model={args.model} size={size} imagenet_norm={imagenet_norm} device={device}")

    val_ds = AstroDataset(VAL_DIR, size=size, augment=False, has_labels=True, imagenet_norm=imagenet_norm)
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    cm = BCEDice().to(device)
    cb = BCEDice().to(device)
    va_loss, va_parts, m = run_epoch(
        model, loader, cm, cb, device, 1.0, 1.0, optimizer=None, compute_metrics=True,
        mask_thr=MASK_THR_DEFAULT, bound_thr=BOUND_THR_DEFAULT, bound_dilate=BOUND_DILATE_DEFAULT)

    print(f"val_loss={va_loss:.4f}  mask_loss={va_parts['mask']:.4f} bound_loss={va_parts['bound']:.4f}")
    print(f"dice_mask ={m['dice_mask']:.3f}  iou_mask ={m['iou_mask']:.3f}")
    print(f"dice_bound={m['dice_bound']:.3f}  iou_bound={m['dice_bound']:.3f}")
    print(f"comp_before={m['comp_before']:.2f}  comp_after={m['comp_after']:.2f}")


if __name__ == "__main__":
    main()

r"""
train.py — обучение простой U-Net сегментации: вход RGB, таргет [mask, bound].
Никакого DAB/gap-канала/skeleton/gap_aux. Crop — равномерная сетка тайлов
с нахлестом (crop_tiles.py), автоматизация + кэш кропов сохранены (--use_crop).

val_loss считается ОДНОВРЕМЕННО по точности предсказания маски астроцита
(канал 0) и по точности предсказания границ (канал 1) — BCE+Dice на каждый
канал, суммируются. Метрики dice/iou логируются отдельно для mask и bound.
"""
import os
import sys
import argparse
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import AstroDataset, IMG_SIZE, IN_CHANNELS
from common import (
    MASK_THR_DEFAULT, BOUND_THR_DEFAULT, BOUND_DILATE_DEFAULT,
    TILE_SIZE_DEFAULT, TILE_OVERLAP_DEFAULT, separate_by_boundary,
)
from model import build_model, N_CLASSES
from crop_tiles import process_split as crop_process_split

ROOT = os.getcwd()
TRAIN_DIR = os.path.join(ROOT, "data", "train")
VAL_DIR = os.path.join(ROOT, "data", "val")
MODEL_OUT_DEFAULT = os.path.join(ROOT, "models", "best_model.pth")
CROPPED_SUBDIR_NAME = "_auto_cropped"


def prepare_crop_cache(split_dir, tile_size, overlap, size, force_rebuild, skip_empty, empty_min_frac):
    images_dir = os.path.join(split_dir, "images")
    masks_dir = os.path.join(split_dir, "masks")
    bounds_dir = os.path.join(split_dir, "boundaries")
    boundaries_dir = bounds_dir if os.path.isdir(bounds_dir) else None

    out_dir = os.path.join(split_dir, f"{CROPPED_SUBDIR_NAME}_{tile_size}")
    marker = os.path.join(out_dir, ".crop_params.txt")
    # size тоже в параметрах: тайлы физически tile_size px, но датасет ресайзит их
    # до size; кэш, собранный под одно size, нельзя переиспользовать под другое.
    params_str = (f"tile_size={tile_size} overlap={overlap} size={size} "
                  f"skip_empty={skip_empty} empty_min_frac={empty_min_frac}")

    already_ok = (
        os.path.isdir(os.path.join(out_dir, "images")) and
        os.path.isfile(marker) and
        open(marker, encoding='utf-8').read().strip() == params_str
    )
    if force_rebuild and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
        already_ok = False
    if already_ok:
        print(f"[--use_crop] Кэш тайлов уже готов ({params_str}): {out_dir}")
        return out_dir

    print(f"[--use_crop] Строю сетку тайлов для {split_dir} -> {out_dir} ({params_str}) ...")
    total = crop_process_split(
        images_dir, masks_dir, out_dir,
        tile_size=tile_size, overlap=overlap,
        boundaries_dir=boundaries_dir, verbose=True,
        skip_empty=skip_empty, empty_min_frac=empty_min_frac)
    if total == 0:
        raise RuntimeError(
            f"--use_crop: в {split_dir} не получилось сохранить ни одного тайла "
            f"(проверьте --empty_min_frac или --no_skip_empty)."
        )
    os.makedirs(out_dir, exist_ok=True)
    with open(marker, "w", encoding='utf-8') as f:
        f.write(params_str)
    return out_dir


def dice_loss(prob, target, eps=1e-6):
    prob = prob.contiguous().view(prob.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    inter = (prob * target).sum(dim=1)
    union = prob.sum(dim=1) + target.sum(dim=1)
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


class BCEDice(nn.Module):
    def __init__(self, pos_weight: float | None = None):
        super().__init__()
        # pos_weight усиливает вклад редкого положительного класса (важно для
        # канала границ, где позитивов ~0.4%). None -> обычный BCE.
        self.pos_weight = pos_weight

    def forward(self, logits, target):
        pw = None
        if self.pos_weight is not None:
            pw = torch.as_tensor(self.pos_weight, dtype=logits.dtype, device=logits.device)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
        prob = torch.sigmoid(logits)
        dice = dice_loss(prob, target)
        return bce + dice


@torch.no_grad()
def dice_iou(prob, target, thr=0.5, eps=1e-6):
    """Dice/IoU по батчу. Пустые таргеты (нет объектов) исключаются из усреднения,
    иначе (2*0+eps)/(0+0+eps)=1.0 ложно завышает метрику на фоне пустых тайлов."""
    pred = (prob > thr).float()
    tgt = (target > 0.5).float()
    pred = pred.view(pred.size(0), -1)
    tgt = tgt.view(tgt.size(0), -1)
    tgt_sum = tgt.sum(dim=1)
    valid = tgt_sum > 0  # только патчи, где есть хоть один позитивный пиксель
    if not valid.any():
        return float("nan"), float("nan")
    pred = pred[valid]
    tgt = tgt[valid]
    inter = (pred * tgt).sum(dim=1)
    union = pred.sum(dim=1) + tgt.sum(dim=1) - inter
    dice = (2 * inter + eps) / (pred.sum(dim=1) + tgt.sum(dim=1) + eps)
    iou = (inter + eps) / (union + eps)
    return dice.mean().item(), iou.mean().item()


@torch.no_grad()
def boundary_separation_components(mask_prob, bound_prob, mask_thr=MASK_THR_DEFAULT,
                                    bound_thr=BOUND_THR_DEFAULT, bound_dilate=BOUND_DILATE_DEFAULT):
    n_before_list, n_after_list = [], []
    mask_np = mask_prob.cpu().numpy().astype("float32")
    bound_np = bound_prob.cpu().numpy().astype("float32")
    for i in range(mask_np.shape[0]):
        _, _, _, n_before, n_after = separate_by_boundary(mask_np[i], bound_np[i], mask_thr, bound_thr, bound_dilate)
        n_before_list.append(n_before)
        n_after_list.append(n_after)
    import numpy as np
    return float(np.mean(n_before_list)), float(np.mean(n_after_list))


def run_epoch(model, loader, criterion_mask, criterion_bound, device,
              mask_weight=1.0, bound_weight=1.0, optimizer=None, compute_metrics=False,
              mask_thr=MASK_THR_DEFAULT, bound_thr=BOUND_THR_DEFAULT, bound_dilate=BOUND_DILATE_DEFAULT,
              scaler=None):  # <-- Добавлен scaler
    device = torch.device(device)  # принимаем и строку, и torch.device
    train = optimizer is not None
    model.train() if train else model.eval()
    total, n = 0.0, 0
    mask_loss_sum, bound_loss_sum = 0.0, 0.0
    dice_mask_sum, iou_mask_sum = 0.0, 0.0
    dice_bound_sum, iou_bound_sum = 0.0, 0.0
    dice_mask_n, dice_bound_n = 0.0, 0.0
    comp_before_sum, comp_after_sum = 0.0, 0.0
    use_amp = (scaler is not None and scaler.is_enabled())
    
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for image, target, _ in loader:
            # Используем non_blocking=True для ускорения передачи на GPU
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            bs = image.size(0)

            if train:
                # AMP только на CUDA; на CPU скейлер отключён (enabled=False).
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(image)
                    loss_mask = criterion_mask(logits[:, 0:1], target[:, 0:1])
                    loss_bound = criterion_bound(logits[:, 1:2], target[:, 1:2])
                    loss = mask_weight * loss_mask + bound_weight * loss_bound

                # Оптимизация зануления градиентов (set_to_none=True работает быстрее)
                optimizer.zero_grad(set_to_none=True)
                
                if use_amp:
                    # Масштабируем loss и делаем шаг
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            else:
                # Без AMP для валидации (для максимальной точности метрик)
                logits = model(image)
                loss_mask = criterion_mask(logits[:, 0:1], target[:, 0:1])
                loss_bound = criterion_bound(logits[:, 1:2], target[:, 1:2])
                loss = mask_weight * loss_mask + bound_weight * loss_bound

            total += loss.item() * bs
            mask_loss_sum += loss_mask.item() * bs
            bound_loss_sum += loss_bound.item() * bs
            n += bs

            if compute_metrics and not train:
                # Вычисления метрик оставляем в FP32
                mask_prob = torch.sigmoid(logits[:, 0].float())
                bound_prob = torch.sigmoid(logits[:, 1].float())
                dm, im_ = dice_iou(mask_prob, target[:, 0], thr=mask_thr)
                db, ib = dice_iou(bound_prob, target[:, 1], thr=bound_thr)
                # Пустые батчи дают NaN — не включаем их в усреднение.
                if not np.isnan(dm):
                    dice_mask_sum += dm * bs; iou_mask_sum += im_ * bs; dice_mask_n += bs
                if not np.isnan(db):
                    dice_bound_sum += db * bs; iou_bound_sum += ib * bs; dice_bound_n += bs
                nb, na = boundary_separation_components(mask_prob, bound_prob, mask_thr, bound_thr, bound_dilate)
                comp_before_sum += nb * bs
                comp_after_sum += na * bs

    parts = {"mask": mask_loss_sum / n, "bound": bound_loss_sum / n}
    metrics = {}
    if compute_metrics and not train:
        metrics["dice_mask"] = dice_mask_sum / dice_mask_n if dice_mask_n else float("nan")
        metrics["iou_mask"] = iou_mask_sum / dice_mask_n if dice_mask_n else float("nan")
        metrics["dice_bound"] = dice_bound_sum / dice_bound_n if dice_bound_n else float("nan")
        metrics["iou_bound"] = iou_bound_sum / dice_bound_n if dice_bound_n else float("nan")
        metrics["comp_before"] = comp_before_sum / n
        metrics["comp_after"] = comp_after_sum / n
    return total / n, parts, metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--size", type=int, default=None,
                    help="размер входа модели; по умолчанию = tile_size при --use_crop, иначе IMG_SIZE")
    ap.add_argument("--encoder", type=str, default="resnet34")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42, help="фиксация случайности для воспроизводимости")

    ap.add_argument("--mask_weight", type=float, default=1.0, help="вес loss маски в val_loss")
    ap.add_argument("--bound_weight", type=float, default=1.0, help="вес loss границ в val_loss")

    ap.add_argument("--mask_thr", type=float, default=MASK_THR_DEFAULT)
    ap.add_argument("--bound_thr", type=float, default=BOUND_THR_DEFAULT)
    ap.add_argument("--bound_dilate", type=int, default=BOUND_DILATE_DEFAULT)
    ap.add_argument("--bound_pos_weight", type=float, default=None,
                    help="pos_weight для BCE канала границ (борьба с дисбалансом ~0.4%% позитивов). "
                         "None = без взвешивания.")

    ap.add_argument("--imagenet_norm", dest="imagenet_norm", action="store_true", default=True,
                    help="нормализация входа под resnet-imagenet (рекомендуется для новых прогонов)")
    ap.add_argument("--no_imagenet_norm", dest="imagenet_norm", action="store_false",
                    help="оставить старую нормализацию [0,1] (совместимо со старыми чекпоинтами)")

    ap.add_argument("--use_crop", action="store_true", help="автоматическая нарезка на тайлы с кэшированием")
    ap.add_argument("--tile_size", type=int, default=TILE_SIZE_DEFAULT, help="256, 512 и т.д.")
    ap.add_argument("--tile_overlap", type=int, default=TILE_OVERLAP_DEFAULT)
    ap.add_argument("--crop_no_skip_empty", action="store_true")
    ap.add_argument("--crop_empty_min_frac", type=float, default=0.0005)
    ap.add_argument("--crop_force_rebuild", action="store_true")
    args = ap.parse_args()

    # Воспроизводимость
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Включаем автонастройку алгоритмов cuDNN для ускорения сверток
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_out = args.out if args.out else MODEL_OUT_DEFAULT

    # Разрешение size: при тайлинге size должен совпадать с tile_size, иначе
    # тайлы бесполезно ресайзятся вниз и теряют детали.
    if args.size is None:
        args.size = args.tile_size if args.use_crop else IMG_SIZE

    train_data_dir = TRAIN_DIR
    val_data_dir = VAL_DIR
    if args.use_crop:
        train_data_dir = prepare_crop_cache(TRAIN_DIR, args.tile_size, args.tile_overlap, args.size,
                                            args.crop_force_rebuild, not args.crop_no_skip_empty,
                                            args.crop_empty_min_frac)
        val_data_dir = prepare_crop_cache(VAL_DIR, args.tile_size, args.tile_overlap, args.size,
                                          args.crop_force_rebuild, not args.crop_no_skip_empty,
                                          args.crop_empty_min_frac)

    print(f"Device: {device} | epochs={args.epochs} bs={args.batch_size} size={args.size} "
          f"in_channels={IN_CHANNELS} n_classes={N_CLASSES} (mask, bound) | "
          f"imagenet_norm={args.imagenet_norm} | "
          f"use_crop={args.use_crop} tile_size={args.tile_size} tile_overlap={args.tile_overlap} | "
          f"train_dir={train_data_dir} val_dir={val_data_dir}")

    train_ds = AstroDataset(train_data_dir, size=args.size, augment=True, has_labels=True,
                            imagenet_norm=args.imagenet_norm)
    val_ds = AstroDataset(val_data_dir, size=args.size, augment=False, has_labels=True,
                          imagenet_norm=args.imagenet_norm)
    print(f"train={len(train_ds)} val={len(val_ds)}")

    # Оптимизация потоков загрузки данных (до 8 потоков, в зависимости от процессора)
    import multiprocessing
    num_workers = min(8, multiprocessing.cpu_count())

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )

    encoder_weights = "imagenet" if args.resume is None else None
    model = build_model(args.encoder, encoder_weights, IN_CHANNELS).to(device)

    # Стартовые значения для resume (переопределятся, если есть чекпоинт)
    start_epoch = 1
    best_val = float("inf")
    epochs_no_improve = 0
    ckpt_imagenet_norm = None

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        if ckpt.get("in_channels") != IN_CHANNELS:
            raise ValueError(f"Несовместимость каналов при --resume: чекпоинт in_channels={ckpt.get('in_channels')}, ожидается {IN_CHANNELS}.")
        model.load_state_dict(ckpt["model_state"])
        ckpt_imagenet_norm = ckpt.get("imagenet_norm", False)
        if ckpt_imagenet_norm != args.imagenet_norm:
            print(f"[--resume] ВНИМАНИЕ: чекпоинт imagenet_norm={ckpt_imagenet_norm}, "
                  f"но CLI imagenet_norm={args.imagenet_norm}. Используется значение чекпоинта.")
        args.imagenet_norm = ckpt_imagenet_norm
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", float("inf")))
        epochs_no_improve = int(ckpt.get("epochs_no_improve", 0))
        print(f"Дообучение: веса загружены из {args.resume} "
              f"(epoch {ckpt.get('epoch', '?')}, best_val={best_val:.4f}, imagenet_norm={ckpt_imagenet_norm})")

    criterion_mask = BCEDice().to(device)
    criterion_bound = BCEDice(pos_weight=args.bound_pos_weight).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if args.resume is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        print("[--resume] optimizer/scheduler восстановлены.")

    # Скейлер AMP: включён только на CUDA.
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    os.makedirs(os.path.dirname(model_out), exist_ok=True)

    for ep in range(start_epoch, args.epochs + 1):
        tr_loss, tr_parts, _ = run_epoch(model, train_loader, criterion_mask, criterion_bound, device,
                                         args.mask_weight, args.bound_weight, optimizer, compute_metrics=False,
                                         scaler=scaler)
        va_loss, va_parts, va_metrics = run_epoch(model, val_loader, criterion_mask, criterion_bound, device,
                                                  args.mask_weight, args.bound_weight, optimizer=None,
                                                  compute_metrics=True, mask_thr=args.mask_thr,
                                                  bound_thr=args.bound_thr, bound_dilate=args.bound_dilate)
        scheduler.step()

        flag = ""
        if va_loss < best_val - 1e-4:
            best_val = va_loss
            epochs_no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "encoder": args.encoder,
                "size": args.size,
                "in_channels": IN_CHANNELS,
                "n_classes": N_CLASSES,
                "channel_names": ["mask", "bound"],
                "imagenet_norm": args.imagenet_norm,
                "epoch": ep,
                "best_val": best_val,
                "epochs_no_improve": epochs_no_improve,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "mask_thr": args.mask_thr,
                "bound_thr": args.bound_thr,
                "bound_dilate": args.bound_dilate,
                "mask_weight": args.mask_weight,
                "bound_weight": args.bound_weight,
                "bound_pos_weight": args.bound_pos_weight,
                "use_crop": args.use_crop,
                "tile_size": args.tile_size,
                "tile_overlap": args.tile_overlap,
            }, model_out)
            flag = " <-- saved best"
        else:
            epochs_no_improve += 1

        if ep == 1 or ep % 5 == 0 or ep == args.epochs or flag:
            tr_str = " ".join(f"{k}={v:.3f}" for k, v in tr_parts.items())
            va_str = " ".join(f"{k}={v:.3f}" for k, v in va_parts.items())
            metrics_str = (f"dice_mask={va_metrics.get('dice_mask', float('nan')):.3f} "
                           f"iou_mask={va_metrics.get('iou_mask', float('nan')):.3f} "
                           f"dice_bound={va_metrics.get('dice_bound', float('nan')):.3f} "
                           f"iou_bound={va_metrics.get('iou_bound', float('nan')):.3f} "
                           f"comp_before={va_metrics.get('comp_before', float('nan')):.2f} "
                           f"comp_after={va_metrics.get('comp_after', float('nan')):.2f}")
            print(f"ep {ep:3d} | train {tr_loss:.4f} ({tr_str}) | val {va_loss:.4f} ({va_str}) | "
                  f"{metrics_str} | lr {scheduler.get_last_lr()[0]:.2e}{flag}")

        if epochs_no_improve >= args.patience:
            print(f"\nРанняя остановка: val_loss не улучшался {args.patience} эпох.")
            break

    print(f"\nЛучший val_loss = {best_val:.4f}. Модель сохранена: {model_out}")


if __name__ == "__main__":
    main()

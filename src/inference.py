r"""
inference.py — инференс по папке test/images.

Реализована модификация из PDF-разбора проекта: разделение соприкасающихся
клеток теперь строится на предсказанном канале границ (bound), а не на
устаревшем механизме gap_aux/skeleton. Модель предсказывает [mask, bound],
затем separate_by_boundary() вычитает бинаризованные границы из маски и
считает связные компоненты до/после — аналогично прежней separate_by_gap,
но источник сигнала иной (bound вместо gap_aux).

Если модель обучалась с --use_crop, инференс тоже может идти по равномерной
сетке тайлов (--use_tiled) с усреднением вероятностей в зонах перекрытия.
"""
import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import (
    find_images, load_model_and_meta, to_tensor, build_tensor_from_rgb,
    prob_to_fullsize, separate_by_boundary, colorize_classes, label_map_from_probs,
    MASK_THR_DEFAULT, BOUND_THR_DEFAULT, BOUND_DILATE_DEFAULT,
)
from crop_tiles import grid_boxes

ROOT = os.getcwd()
TEST_DIR = os.path.join(ROOT, "data", "test", "images")
OUT_DIR = os.path.join(ROOT, "outputs")
MODEL_PATH_DEFAULT = os.path.join(ROOT, "models", "best_model.pth")


def render_output(final_mask_uint8):
    """Простой рендер: клетки — белым на черном фоне (без красного скелета,
    без DAB — эти навороты убраны согласно ТЗ)."""
    h, w = final_mask_uint8.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[final_mask_uint8 > 0] = (255, 255, 255)
    return out


@torch.no_grad()
def predict_full_frame(model, path, size, device, imagenet_norm=False):
    """Возвращает полный тензор вероятностей (C, H, W), C — число классов модели.
    Каналы интерпретируются вызывающим в зависимости от channel_names чекпоинта."""
    t, meta, orig_wh = to_tensor(path, size, imagenet_norm)
    logits = model(t.to(device))
    C = logits.shape[1]
    probs_small = torch.sigmoid(logits[0]).cpu().numpy().astype(np.float32)  # (C, h, w)
    probs_full = np.stack(
        [prob_to_fullsize(probs_small[c], meta, orig_wh) for c in range(C)], axis=0)
    return probs_full


@torch.no_grad()
def predict_tiled(model, path, size, device, tile_size, tile_overlap, imagenet_norm=False):
    """Инференс по равномерной сетке тайлов с нахлестом; вероятности
    усредняются в зонах перекрытия. Возвращает (probs_full (C,H,W), n_tiles)."""
    orig_rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    h, w = orig_rgb.shape[:2]
    boxes = grid_boxes(h, w, tile_size=tile_size, overlap=tile_overlap)

    probs_sum = None
    weight_sum = np.zeros((h, w), dtype=np.float32)

    for (y0, y1, x0, x1) in boxes:
        crop_rgb = orig_rgb[y0:y1, x0:x1]
        t, meta = build_tensor_from_rgb(crop_rgb, size, imagenet_norm)
        logits = model(t.to(device))
        C = logits.shape[1]
        probs_small = torch.sigmoid(logits[0]).cpu().numpy().astype(np.float32)  # (C, h, w)
        probs_crop = np.stack(
            [prob_to_fullsize(probs_small[c], meta, (x1 - x0, y1 - y0)) for c in range(C)], axis=0)

        if probs_sum is None:
            probs_sum = np.zeros((C, h, w), dtype=np.float32)
        probs_sum[:, y0:y1, x0:x1] += probs_crop
        weight_sum[y0:y1, x0:x1] += 1.0

    probs_full = probs_sum / np.maximum(weight_sum, 1.0)
    return probs_full, len(boxes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=MODEL_PATH_DEFAULT)
    ap.add_argument("--mask_thr", type=float, default=MASK_THR_DEFAULT)
    ap.add_argument("--bound_thr", type=float, default=BOUND_THR_DEFAULT)
    ap.add_argument("--bound_dilate", type=int, default=BOUND_DILATE_DEFAULT)
    ap.add_argument("--use_tiled", action="store_true",
                     help="инференс по сетке тайлов вместо полного кадра целиком")
    ap.add_argument("--tile_size", type=int, default=None, help="по умолчанию берётся из чекпоинта")
    ap.add_argument("--tile_overlap", type=int, default=None, help="по умолчанию берётся из чекпоинта")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Нет модели {args.model}. Сначала запустите train.py")

    model, size, tile_meta = load_model_and_meta(args.model, device)
    imagenet_norm = tile_meta["imagenet_norm"]
    tile_size = args.tile_size if args.tile_size is not None else tile_meta["tile_size"]
    tile_overlap = args.tile_overlap if args.tile_overlap is not None else tile_meta["tile_overlap"]

    files = find_images(TEST_DIR)
    if not files:
        raise RuntimeError(f"Нет тестовых изображений в {TEST_DIR}")
    print(f"Device: {device} | размер входа модели: {size} | файлов: {len(files)} | "
          f"imagenet_norm={imagenet_norm} | "
          f"mask_thr={args.mask_thr} bound_thr={args.bound_thr} bound_dilate={args.bound_dilate} | "
          f"use_tiled={args.use_tiled} (tile_size={tile_size}, overlap={tile_overlap})")

    with torch.no_grad():
        for path in files:
            name = os.path.splitext(os.path.basename(path))[0]
            if args.use_tiled:
                probs_full, n_tiles = predict_tiled(
                    model, path, size, device, tile_size, tile_overlap, imagenet_norm)
                tiles_info = f" (тайлов: {n_tiles})"
            else:
                probs_full = predict_full_frame(model, path, size, device, imagenet_norm)
                tiles_info = ""

            if tile_meta["is_gfap"]:
                # Наша задача GFAP: канал 0 — маска, канал 1 — границы.
                mask_prob_full = probs_full[0]
                bound_prob_full = probs_full[1]
                _, final_mask, _, n_before, n_after = separate_by_boundary(
                    mask_prob_full, bound_prob_full, args.mask_thr, args.bound_thr, args.bound_dilate)

                out = render_output(final_mask)
                out_path = os.path.join(OUT_DIR, name + ".png")
                Image.fromarray(out).save(out_path)
                status = "разделение сработало" if n_after > n_before else "разделения не произошло"
                print(f"  {name}{tiles_info}: объектов до={n_before} после={n_after} ({status}) -> {out_path}")
            else:
                # Multi-class: цветная маска (black->серый, gray->чёрный, white->белый)
                # ПЛЮС целочисленная метка-карта *_label.png (0=просвет,1=серое,2=белое).
                combined = colorize_classes(probs_full, tile_meta["channel_names"])
                label_map = label_map_from_probs(probs_full)
                out_path = os.path.join(OUT_DIR, name + ".png")
                Image.fromarray(combined, mode="RGB").save(out_path)
                Image.fromarray(label_map, mode="L").save(os.path.join(OUT_DIR, name + "_label.png"))
                print(f"  {name}{tiles_info}: классы={tile_meta['channel_names']} -> {out_path} + метка {name}_label.png")

    print(f"\nГотово. Результаты в {OUT_DIR}")


if __name__ == "__main__":
    main()

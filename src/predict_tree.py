r"""
predict_tree.py — рекурсивный инференс по дереву папок с СОХРАНЕНИЕМ структуры.

Находит ВСЕ изображения во входном дереве (любая глубина вложенности),
прогоняет каждое через модель [mask, bound] и пишет полученную маску клеток
в выходное дерево с ТЕМИ ЖЕ относительными путями и тем же именем файла.

Пример:
    python src/predict_tree.py --input "D:/astro_seg_light/GFAP_irr_study" --model models/astro_2.pth
    -> маски появятся в D:/astro_seg_light/GFAP_irr_study_masks/... зеркально структуре input.

Нормализация входа берётся из чекпоинта (imagenet_norm), чтобы НЕ расходиться
с тем, при какой нормализации модель училась. Для старых чекпоинтов astro_*.pth
это старая [0,1] — именно она и используется.
"""
import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import (
    find_images_recursive, load_model_and_meta, separate_by_boundary,
    colorize_classes, label_map_from_probs,
    MASK_THR_DEFAULT, BOUND_THR_DEFAULT, BOUND_DILATE_DEFAULT,
)
from inference import predict_full_frame, predict_tiled, render_output


def resolve_model(model_arg):
    """Если --model задан и существует — используем его. Если задан, но не существует —
    явная ошибка (НЕ тихая подмена другой моделью!). Если не задан — best_model.pth,
    иначе самый свежий astro_*.pth."""
    if model_arg:
        if os.path.exists(model_arg):
            return model_arg
        # Путь не разрешился (напр. запуск не из корня проекта) — падаем понятно,
        # вместо того чтобы молча взять astro_*.pth и выдать результат не той модели.
        raise FileNotFoundError(
            f"--model не найден по пути: {model_arg!r}. Укажите существующий файл "
            f"(абсолютный путь или относительно рабочей папки). "
            f"Модели лежат в: {os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models'))}")
    here = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.abspath(os.path.join(here, "..", "models"))
    best = os.path.join(models_dir, "best_model.pth")
    if os.path.exists(best):
        return best
    cands = sorted(
        (os.path.join(models_dir, f) for f in os.listdir(models_dir)
         if f.startswith("astro_") and f.endswith(".pth")),
        key=lambda p: os.path.getmtime(p), reverse=True)
    if cands:
        return cands[0]
    raise FileNotFoundError(
        f"Модель не найдена. Укажите --model явно (например --model models/astro_2.pth). "
        f"Проверено: {best}")


def save_mask(out_path, final_mask, mode):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if mode == "overlay":
        Image.fromarray(render_output(final_mask)).save(out_path)
    else:
        # 'mask': одноканальная бинарная маска 0/255 в том же расширении.
        Image.fromarray(final_mask, mode="L").save(out_path)


def run(input_dir, model_path, output_dir=None,
        mask_thr=MASK_THR_DEFAULT, bound_thr=BOUND_THR_DEFAULT,
        bound_dilate=BOUND_DILATE_DEFAULT, min_object_area=0,
        mode="mask", use_tiled=False, tile_size=None, tile_overlap=None,
        skip_existing=False, verbose_prefix=""):
    """Ядро инференса по дереву. Вынесено из main() чтобы переиспользоваться
    оркестратором predict_all.py в одном процессе."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = resolve_model(model_path)
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Нет входной папки: {input_dir}")
    out_root = output_dir if output_dir else (input_dir.rstrip(os.sep) + "_masks")

    model, size, tile_meta = load_model_and_meta(model_path, device, verbose_prefix=verbose_prefix)
    imagenet_norm = tile_meta["imagenet_norm"]
    tile_size = tile_size if tile_size is not None else tile_meta["tile_size"]
    tile_overlap = tile_overlap if tile_overlap is not None else tile_meta["tile_overlap"]

    files = find_images_recursive(input_dir)
    if not files:
        raise RuntimeError(f"Изображения не найдены в {input_dir}")
    print(f"Device: {device} | model: {model_path}\n"
          f"input: {input_dir}\noutput: {out_root}\n"
          f"файлов: {len(files)} | input_size={size} | imagenet_norm={imagenet_norm} | "
          f"min_object_area={min_object_area} | mode={mode}\n"
          f"use_tiled={use_tiled} (tile_size={tile_size}, overlap={tile_overlap})")

    done = 0
    skipped = 0
    n_total = len(files)
    n_w = len(str(n_total))
    with torch.no_grad():
        for path in files:
            rel = os.path.relpath(path, input_dir)
            out_path = os.path.join(out_root, rel)
            if skip_existing and os.path.exists(out_path):
                skipped += 1
                continue

            if use_tiled:
                probs_full, n_tiles = predict_tiled(
                    model, path, size, device, tile_size, tile_overlap, imagenet_norm)
            else:
                probs_full = predict_full_frame(
                    model, path, size, device, imagenet_norm)

            if tile_meta["is_gfap"]:
                mask_prob_full = probs_full[0]
                bound_prob_full = probs_full[1]
                _, final_mask, _, n_before, n_after = separate_by_boundary(
                    mask_prob_full, bound_prob_full, mask_thr, bound_thr,
                    bound_dilate, min_object_area=min_object_area)

                save_mask(out_path, final_mask, mode)
                done += 1
                status = "разделение" if n_after > n_before else "без разделения"
                print(f"  [{done + skipped:>{n_w}}/{n_total}] {rel}: "
                      f"до={n_before} после={n_after} ({status}) -> {out_path}")
            elif tile_meta["is_focus"]:
                # Одноканальная вероятность "in-focus": порог -> бинарная маска,
                # плюс целочисленная метка 0/1 (1=в фокусе) для analyze_masks.
                mask_prob_full = probs_full[0]
                mthr = tile_meta.get("mask_thr", MASK_THR_DEFAULT)
                in_focus = (mask_prob_full > mthr)
                save_mask(out_path, (in_focus.astype(np.uint8) * 255), mode)
                base, _ext = os.path.splitext(out_path)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                Image.fromarray(in_focus.astype(np.uint8), mode="L").save(base + "_label.png")
                done += 1
                print(f"  [{done + skipped:>{n_w}}/{n_total}] {rel}: "
                      f"focus mask (thr={mthr:.2f}) -> {out_path} + метка {base}_label.png")
            else:
                # Multi-class (серое/белое вещество + просвет): цветная маска
                # (black->серый, gray->чёрный, white->белый) ПЛЮС целочисленная
                # метка-карта *_label.png (0=просвет,1=серое,2=белое) для последующего
                # анализа/композитинга по областям ткани.
                combined = colorize_classes(probs_full, tile_meta["channel_names"])
                label_map = label_map_from_probs(probs_full)
                base, _ext = os.path.splitext(out_path)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                Image.fromarray(combined, mode="RGB").save(out_path)
                Image.fromarray(label_map, mode="L").save(base + "_label.png")
                done += 1
                print(f"  [{done + skipped:>{n_w}}/{n_total}] {rel}: "
                      f"классы={tile_meta['channel_names']} -> цветная маска {out_path} + метка {base}_label.png")

    print(f"\nГотово. Обработано: {done}, пропущено (уже есть): {skipped}. Маски в: {out_root}")
    return out_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="корень дерева с изображениями (например D:/astro_seg_light/GFAP_irr_study)")
    ap.add_argument("--output", default=None,
                    help="куда писать маски; по умолчанию <input>_masks")
    ap.add_argument("--model", default=None, help="путь к чекпоинту (.pth)")
    ap.add_argument("--mask_thr", type=float, default=MASK_THR_DEFAULT)
    ap.add_argument("--bound_thr", type=float, default=BOUND_THR_DEFAULT)
    ap.add_argument("--bound_dilate", type=int, default=BOUND_DILATE_DEFAULT)
    ap.add_argument("--min_object_area", type=int, default=0,
                    help="отбрасывать компоненты меньше N пикселей после разделения (0 = не отсекать)")
    ap.add_argument("--mode", choices=["mask", "overlay"], default="mask",
                    help="mask = бинарная одноканальная маска; overlay = белые клетки на чёрном")
    ap.add_argument("--use_tiled", action="store_true",
                    help="инференс по сетке тайлов вместо полного кадра целиком")
    ap.add_argument("--tile_size", type=int, default=None, help="по умолчанию из чекпоинта")
    ap.add_argument("--tile_overlap", type=int, default=None, help="по умолчанию из чекпоинта")
    ap.add_argument("--skip_existing", action="store_true",
                    help="пропускать файлы, для которых маска уже существует")
    args = ap.parse_args()

    run(args.input, args.model, args.output,
        mask_thr=args.mask_thr, bound_thr=args.bound_thr, bound_dilate=args.bound_dilate,
        min_object_area=args.min_object_area, mode=args.mode, use_tiled=args.use_tiled,
        tile_size=args.tile_size, tile_overlap=args.tile_overlap, skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()

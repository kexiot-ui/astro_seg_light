"""
crop_tiles.py — разбиение изображений на РАВНЫЕ тайлы фиксированного размера
(256x256, 512x512 и т.д.) с настраиваемым нахлестом (overlap). Без привязки
к белым объектам / детекции клеток — обычная sliding-window сетка по всей
картинке.

Используется как для подготовки train/val (кэш кропов), так и потенциально
для инференса по тайлам (см. inference.py: predict_tiled).
"""
import os
import argparse
import numpy as np
from PIL import Image

IMG_EXTS = (".bmp", ".png", ".tif", ".tiff", ".jpg", ".jpeg")


def _find_images(d):
    return sorted(
        os.path.join(d, f) for f in os.listdir(d)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    )


def _match_label_path(name, folder):
    if not folder or not os.path.isdir(folder):
        return None
    entries = {}
    for f in os.listdir(folder):
        p = os.path.join(folder, f)
        if not os.path.isfile(p) or os.path.splitext(f)[1].lower() not in IMG_EXTS:
            continue
        entries[os.path.splitext(f)[0].lower()] = p
    nl = name.lower()
    # Только ТОЧНОЕ совпадение стема со списком суффиксов — без startswith-фолбэка,
    # иначе "A1" находил бы метку "A10_mask".
    for suf in ("_mask", "", "_masks", "-mask", "mask", "_boundary", "_boundaries"):
        cand = nl + suf
        if cand in entries:
            return entries[cand]
    return None


def _read_gray(path):
    """Читает масть/границу и приводит к бинарному uint8 (0/255) независимо от
    исходного dtype (uint8/uint16/float), чтобы сохранение в PNG не падало."""
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = np.asarray(arr)
    if arr.dtype == np.bool_:
        return (arr.astype(np.float32) * 255).astype(np.uint8)
    if np.issubdtype(arr.dtype, np.floating):
        amax = float(arr.max()) if arr.size else 0.0
        scale = 255.0 if amax > 1.0 else 1.0
        return (arr > 0.5 * scale).astype(np.uint8) * 255
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        return (arr > info.max / 2.0).astype(np.uint8) * 255
    raise TypeError(f"Неподдерживаемый dtype маски: {arr.dtype} ({path})")


def grid_boxes(img_h, img_w, tile_size=256, overlap=32):
    """Равномерная сетка окон tile_size x tile_size с нахлестом overlap (px).
    Последний тайл в каждой строке/столбце сдвигается так, чтобы точно
    прилегать к правому/нижнему краю изображения (без выхода за границы).
    Если изображение меньше tile_size — один тайл на всю картинку (без ресайза,
    будет дополнен letterbox-ом на этапе построения тензора).
    """
    step = max(1, tile_size - overlap)

    def axis_starts(length):
        if length <= tile_size:
            return [0]
        starts = list(range(0, length - tile_size + 1, step))
        last = length - tile_size
        if starts[-1] != last:
            starts.append(last)
        return starts

    ys = axis_starts(img_h)
    xs = axis_starts(img_w)

    boxes = []
    for y0 in ys:
        y1 = min(img_h, y0 + tile_size)
        for x0 in xs:
            x1 = min(img_w, x0 + tile_size)
            boxes.append((y0, y1, x0, x1))
    return boxes


def crop_and_resize(arr, box, tile_size, is_mask=False):
    y0, y1, x0, x1 = box
    patch = arr[y0:y1, x0:x1]
    ph, pw = patch.shape[:2]
    if ph == tile_size and pw == tile_size:
        return patch
    img = Image.fromarray(patch)
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    return np.asarray(img.resize((tile_size, tile_size), resample))


def process_split(images_dir, masks_dir, out_dir, tile_size=256, overlap=32,
                   boundaries_dir=None, verbose=True, skip_empty=True,
                   empty_min_frac=0.0005):
    """Режет images/masks/(boundaries) на равные тайлы с нахлестом и
    сохраняет их в out_dir/{images,masks,boundaries}. Тайлы, где в маске
    почти нет объекта (доля белых пикселей < empty_min_frac), по умолчанию
    пропускаются, чтобы не засорять обучение пустым фоном.
    """
    out_images = os.path.join(out_dir, "images")
    out_masks = os.path.join(out_dir, "masks")
    out_bounds = os.path.join(out_dir, "boundaries") if boundaries_dir else None
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_masks, exist_ok=True)
    if out_bounds:
        os.makedirs(out_bounds, exist_ok=True)

    files = _find_images(images_dir)
    if not files:
        raise RuntimeError(f"Нет изображений в {images_dir}")

    total_crops = 0
    for img_path in files:
        name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = _match_label_path(name, masks_dir)
        if mask_path is None:
            if verbose:
                print(f"  [skip] нет маски для {name}")
            continue

        rgb = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        h, w = rgb.shape[:2]
        mask_arr = _read_gray(mask_path)
        if mask_arr.shape[:2] != (h, w):
            raise RuntimeError(f"Размер mask не совпадает с image для {name}: image={(h,w)} mask={mask_arr.shape[:2]}")

        bound_arr = None
        if boundaries_dir:
            bpath = _match_label_path(name, boundaries_dir)
            if bpath is not None:
                bound_arr = _read_gray(bpath)
                if bound_arr.shape[:2] != (h, w):
                    raise RuntimeError(f"Размер boundary не совпадает с image для {name}: image={(h,w)} boundary={bound_arr.shape[:2]}")

        boxes = grid_boxes(h, w, tile_size=tile_size, overlap=overlap)
        saved = 0
        for i, crop_box in enumerate(boxes):
            mask_crop_check = crop_and_resize(mask_arr, crop_box, tile_size, is_mask=True)
            if skip_empty:
                frac = (mask_crop_check > 127).mean()
                if frac < empty_min_frac:
                    continue

            rgb_crop = crop_and_resize(rgb, crop_box, tile_size, is_mask=False)
            out_name = f"{name}_tile{i:03d}"
            Image.fromarray(rgb_crop).save(os.path.join(out_images, out_name + ".png"))
            Image.fromarray(mask_crop_check).save(os.path.join(out_masks, out_name + "_mask.png"))
            if bound_arr is not None:
                bound_crop = crop_and_resize(bound_arr, crop_box, tile_size, is_mask=True)
                Image.fromarray(bound_crop).save(os.path.join(out_bounds, out_name + "_boundary.png"))
            saved += 1
            total_crops += 1

        if verbose:
            print(f"  {name}: {len(boxes)} тайлов сетки -> сохранено {saved}")

    if verbose:
        print(f"\nГотово. Всего тайлов: {total_crops}. Результат в {out_dir}")
    return total_crops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--masks_dir", required=True)
    ap.add_argument("--boundaries_dir", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tile_size", type=int, default=256, help="256, 512 и т.д.")
    ap.add_argument("--overlap", type=int, default=32)
    ap.add_argument("--no_skip_empty", action="store_true")
    ap.add_argument("--empty_min_frac", type=float, default=0.0005)
    args = ap.parse_args()
    process_split(args.images_dir, args.masks_dir, args.out_dir,
                  tile_size=args.tile_size, overlap=args.overlap,
                  boundaries_dir=args.boundaries_dir,
                  skip_empty=not args.no_skip_empty,
                  empty_min_frac=args.empty_min_frac)


if __name__ == "__main__":
    main()

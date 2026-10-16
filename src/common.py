"""
common.py — общий код: загрузка модели/чекпоинта, сборка входного тензора
из RGB, свёртка вероятностей в исходный размер, разделение маски по
boundary-каналу (реализация модификации инференса из PDF-разбора проекта:
разделение клеток теперь строится на предсказанных границах — канале
'bound' — а не на устаревшем gap_aux/skeleton-механизме).
"""
import os
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import label as cc_label, binary_dilation, generate_binary_structure

from model import build_model, N_CLASSES
from dataset import letterbox_resize, unletterbox

IMG_EXTS = (".bmp", ".png", ".tif", ".tiff", ".jpg", ".jpeg")
_STRUCT8 = generate_binary_structure(2, 2)

MASK_THR_DEFAULT = 0.5
BOUND_THR_DEFAULT = 0.35
BOUND_DILATE_DEFAULT = 2

TILE_SIZE_DEFAULT = 256
TILE_OVERLAP_DEFAULT = 32

# Статистика ImageNet (нужна, только если модель тренировалась с imagenet_norm).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def find_images(d):
    files = [os.path.join(d, f) for f in os.listdir(d)
             if os.path.splitext(f)[1].lower() in IMG_EXTS]
    return sorted(files)


def find_images_recursive(root):
    """Рекурсивный обход дерева: возвращает все изображения с полными путями."""
    out = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                out.append(os.path.join(dirpath, f))
    return sorted(out)


def load_model_and_meta(model_path, device, verbose_prefix=""):
    ckpt = torch.load(model_path, map_location=device, weights_only=True)
    encoder = ckpt.get("encoder", "resnet34")
    size = ckpt.get("size", 256)
    in_ch = ckpt.get("in_channels", 3)
    # Для старых чекпоинтов (без этого ключа) нормализация — старая [0,1].
    imagenet_norm = bool(ckpt.get("imagenet_norm", False))
    # Число классов и их имена — берём из чекпоинта (модели друга бывают 1/3/... классов,
    # а не только наши 2 [mask, bound]). Без этого load_state_dict падает на несовпадении
    # размера выходного слоя.
    n_classes = int(ckpt.get("n_classes", N_CLASSES))
    channel_names = ckpt.get("channel_names", ["mask", "bound"])
    if len(channel_names) != n_classes:
        channel_names = [f"class_{i}" for i in range(n_classes)]
    is_gfap = (channel_names == ["mask", "bound"])  # наша задача: маска + границы
    # Одноканальная модель (n_classes==1, напр. focus1: ['mask']) — бинарная маска
    # "in-focus" (высокая вероятность = в фокусе). Не путать с multi-class тканью
    # (white_and_grey, n_classes>=2) и с GFAP (['mask','bound']).
    is_focus = (n_classes == 1)

    tile_meta = {
        "tile_size": ckpt.get("tile_size", TILE_SIZE_DEFAULT),
        "tile_overlap": ckpt.get("tile_overlap", TILE_OVERLAP_DEFAULT),
        "imagenet_norm": imagenet_norm,
        "n_classes": n_classes,
        "channel_names": channel_names,
        "is_gfap": is_gfap,
        "is_focus": is_focus,
        "mask_thr": float(ckpt.get("mask_thr", MASK_THR_DEFAULT)),
    }

    model = build_model(encoder_name=encoder, encoder_weights=None, in_channels=in_ch, n_classes=n_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    if is_gfap:
        task_label = "GFAP[mask,bound]"
    elif is_focus:
        task_label = "focus[in-focus mask]"
    else:
        task_label = "multi-class"
    print(f"{verbose_prefix}Загружен чекпоинт: in_channels={in_ch} n_classes={n_classes} "
          f"channel_names={channel_names} input_size={size} tile_size={tile_meta['tile_size']} "
          f"tile_overlap={tile_meta['tile_overlap']} imagenet_norm={imagenet_norm} "
          f"task={task_label}")

    return model, size, tile_meta


def build_tensor_from_rgb(rgb_uint8_full, size, imagenet_norm=False):
    pad_val = tuple(int(v) for v in rgb_uint8_full.reshape(-1, 3).mean(axis=0))
    rgb_uint8, meta = letterbox_resize(rgb_uint8_full, size, is_mask=False, pad_value=pad_val)
    rgb = rgb_uint8.astype(np.float32) / 255.0
    if imagenet_norm:
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    t = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).float()
    return t, meta


def to_tensor(path, size, imagenet_norm=False):
    img = Image.open(path).convert("RGB")
    orig_wh = img.size
    rgb_uint8_full = np.asarray(img, dtype=np.uint8)
    t, meta = build_tensor_from_rgb(rgb_uint8_full, size, imagenet_norm)
    return t, meta, orig_wh


def prob_to_fullsize(prob_small, meta, orig_wh):
    cropped = unletterbox(prob_small, meta)
    W, H = orig_wh
    cropped_img = Image.fromarray(cropped.astype(np.float32), mode="F")
    resized = np.asarray(cropped_img.resize((W, H), Image.BICUBIC), dtype=np.float32)
    return np.clip(resized, 0.0, 1.0)


def separate_by_boundary(mask_prob_full, bound_prob_full, mask_thr, bound_thr, bound_dilate,
                         min_object_area=0):
    """Реализация модификации инференса из PDF: разделяем маску клеток по
    предсказанному каналу границ (bound), а не по устаревшему gap_aux.
    Логика идентична прежней separate_by_gap, но источник сигнала — bound.

    min_object_area > 0 — отбрасываем шумовые компоненты меньше заданной площади
    (после вычитания границ), чтобы в маске не было мусорных «клеток».
    """
    mask_bin = mask_prob_full > mask_thr
    bound_bin = bound_prob_full > bound_thr
    if bound_dilate > 0:
        bound_bin = binary_dilation(bound_bin, structure=_STRUCT8, iterations=bound_dilate)

    _, n_before = cc_label(mask_bin, structure=_STRUCT8)
    separated = mask_bin & (~bound_bin)

    if min_object_area and min_object_area > 0:
        lab, n = cc_label(separated, structure=_STRUCT8)
        if n:
            sizes = np.bincount(lab.ravel())[1:]
            for i in range(1, n + 1):
                if sizes[i - 1] < min_object_area:
                    separated[lab == i] = False

    _, n_after = cc_label(separated, structure=_STRUCT8)

    return (mask_bin.astype(np.uint8) * 255,
            separated.astype(np.uint8) * 255,
            (bound_bin & mask_bin).astype(np.uint8) * 255,
            int(n_before), int(n_after))


# Цвета по умолчанию для имён классов ткани (серое/белое вещество + просвет).
# Схема пользователя: просвет(black) -> серый, серое вещество(gray) -> чёрный,
# белое вещество(white) -> белый. (Позже поверх накладывается маска астроцитов.)
_TISSUE_COLORS = {
    "black": (128, 128, 128),       # просвет / пустота -> серый
    "background": (128, 128, 128),
    "bg": (128, 128, 128),
    "gray": (0, 0, 0),              # серое вещество -> чёрный
    "grey": (0, 0, 0),
    "white": (255, 255, 255),       # белое вещество -> белый
}

# Запасная палитра для прочих (неизвестных) имён классов.
_FALLBACK_PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
]


def colorize_classes(probs_full, channel_names, palette=None):
    """Комбинированная цветная маска (H, W, 3) uint8 для multi-class моделей.

    Каждый пиксель получает класс argmax по каналам вероятностей и раскрашивается
    по цветовой карте. По умолчанию (модель [black, gray, white]):
      black/background -> чёрный (0,0,0)        — просвет / пустота на стекле
      gray/grey        -> серый   (128,128,128) — серое вещество
      white            -> белый   (255,255,255) — белое вещество
    Неизвестные имена классов получают различающиеся цвета из _FALLBACK_PALETTE.
    """
    C = probs_full.shape[0]
    label_map = probs_full.argmax(axis=0).astype(np.int32)

    if palette is None:
        palette = [_TISSUE_COLORS.get(name.lower(), None) for name in channel_names]
        fi = 0
        for i in range(C):
            if palette[i] is None:
                palette[i] = _FALLBACK_PALETTE[fi % len(_FALLBACK_PALETTE)]
                fi += 1

    h, w = probs_full.shape[1], probs_full.shape[2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(C):
        out[label_map == i] = palette[i]
    return out


def label_map_from_probs(probs_full):
    """Целочисленная метка-карта (H, W) uint8: argmax по каналам вероятностей.
    Значения 0..C-1 = порядок channel_names чекпоинта
    (white_and_grey: 0=просвет, 1=серое вещество, 2=белое вещество).
    Используется для композитинга/анализа по областям ткани (analyze_masks.py)."""
    return probs_full.argmax(axis=0).astype(np.uint8)

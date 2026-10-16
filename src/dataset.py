"""
dataset.py — датасет с интеграцией Albumentations: вход RGB (3 канала), таргет [mask, bound].
Маски клетки и маски границ искажаются строго синхронно с фото, 
пиксельные шумы применяются только к оригинальному изображению.
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import tifffile
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_SIZE = 256
IN_CHANNELS = 3  # RGB only
IMG_EXTS = (".bmp", ".png", ".tif", ".tiff", ".jpg", ".jpeg")

# Статистика ImageNet. Энкодеры с encoder_weights="imagenet" ОЖИДАЮТ именно такую
# нормализацию; подача сырого [0,1] (mean=0, std=1) ломает transfer learning.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_normalize(imagenet_norm=True):
    """A.Normalize, совместимый с pretrained-энкодером.
    imagenet_norm=True  -> (x/255 - mean)/std  (для encoder_weights='imagenet')
    imagenet_norm=False -> x/255               (устаревшее поведение [0,1],
                            оставлено для совместимости со старыми чекпоинтами)
    """
    if imagenet_norm:
        return A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0)
    return A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0)


def _find_images(images_dir):
    files = []
    for f in os.listdir(images_dir):
        if os.path.splitext(f)[1].lower() in IMG_EXTS:
            files.append(os.path.join(images_dir, f))
    return sorted(files)


def letterbox_resize(arr, size, is_mask=False, pad_value=None):
    if arr.ndim == 2:
        h, w = arr.shape
    else:
        h, w = arr.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    img = Image.fromarray(arr)
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    img_r = img.resize((new_w, new_h), resample)
    arr_r = np.asarray(img_r)
    pad_left = (size - new_w) // 2
    pad_top = (size - new_h) // 2
    if arr.ndim == 2:
        canvas = np.zeros((size, size), dtype=arr.dtype)
        fill = 0 if (is_mask or pad_value is None) else pad_value
        canvas[:] = fill
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = arr_r
    else:
        c = arr.shape[2]
        canvas = np.zeros((size, size, c), dtype=arr.dtype)
        fill = 0 if (is_mask or pad_value is None) else pad_value
        canvas[:] = fill
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = arr_r
    meta = (new_w, new_h, pad_left, pad_top, w, h)
    return canvas, meta


def unletterbox(arr, meta):
    new_w, new_h, pad_left, pad_top, orig_w, orig_h = meta
    return arr[pad_top:pad_top + new_h, pad_left:pad_left + new_w]


def _read_image(path, size):
    img = Image.open(path).convert("RGB")
    rgb_uint8_full = np.asarray(img, dtype=np.uint8)
    if size is not None:
        pad_val = tuple(int(v) for v in rgb_uint8_full.reshape(-1, 3).mean(axis=0))
        rgb_uint8, meta = letterbox_resize(rgb_uint8_full, size, is_mask=False, pad_value=pad_val)
    else:
        rgb_uint8, meta = rgb_uint8_full, None
    
    # ВАЖНО: Больше не делим на 255.0 здесь. 
    # Albumentations работает с изображениями в формате uint8 (0-255).
    # Нормализация будет выполнена внутри конвейера.
    return rgb_uint8, meta


def binarize_mask(arr):
    """Бинаризация маски произвольного типа/диапазона -> float32 в {0,1}.
    Порог выбирается по dtype, а НЕ фиксированный 127: корректно обрабатывает
    uint8 (0/255), uint16 (0/65535), bool и float-маски ([0,1] и [0,255]).
    Прежняя логика `(arr.astype(np.uint8) > 127)` молча обнуляла float-маски.
    """
    arr = np.asarray(arr)
    if arr.dtype == np.bool_:
        return arr.astype(np.float32)
    if np.issubdtype(arr.dtype, np.floating):
        amax = float(arr.max()) if arr.size else 0.0
        scale = 255.0 if amax > 1.0 else 1.0
        return (arr > 0.5 * scale).astype(np.float32)
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        return (arr > info.max / 2.0).astype(np.float32)
    raise TypeError(f"Неподдерживаемый dtype маски: {arr.dtype}")


def _read_mask(path, size):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        arr = tifffile.imread(path)
    else:
        arr = np.asarray(Image.open(path))
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[..., 0]
    # Сначала бинаризуем с учётом исходного dtype, потом меняем размер,
    # чтобы NEAREST-интерполяция не задела полутона uint16/float.
    arr = (binarize_mask(arr) * 255).astype(np.uint8)
    if size is not None:
        arr, _ = letterbox_resize(arr, size, is_mask=True)
    return (arr > 127).astype(np.float32)


class AstroDataset(Dataset):
    """Датасет с онлайн-аугментацией: images/ + masks/ (+ boundaries/).
    Таргет — 2 канала [mask, bound]. Если has_labels=False, возвращает dummy target.
    """

    def __init__(self, split_dir, size=IMG_SIZE, augment=False, has_labels=True, imagenet_norm=True):
        self.images_dir = os.path.join(split_dir, "images")
        self.masks_dir = os.path.join(split_dir, "masks")
        self.bound_dir = os.path.join(split_dir, "boundaries")
        self.size = size
        self.augment = augment
        self.has_labels = has_labels
        self.imagenet_norm = imagenet_norm
        self.files = _find_images(self.images_dir)
        if len(self.files) == 0:
            raise RuntimeError(f"Нет изображений в {self.images_dir}")

        # Безопасная инициализация трансформаций по умолчанию
        self.transform = A.Compose([
            make_normalize(self.imagenet_norm),
            ToTensorV2()
        ])

        # 1. Задаем пайплайны Albumentations для обучения
        if self.augment:
            self.transform = A.Compose([
                # Геометрия
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Affine(
                    scale=(0.95, 1.05), translate_percent=(-0.03, 0.03), rotate=(-180, 180),
                    # Отражающие границы вместо чёрных (чёрные углы после поворота —
                    # лишний артефакт поверх mean-padding из letterbox).
                    border_mode=cv2.BORDER_REFLECT, fill=0, fill_mask=0, p=0.6
                ),
                # ВАЖНО: ElasticTransform с alpha=0.5 не даёт видимой деформации
                # (сдвиг <0.5px исчезает после nearest). Реально работающие значения:
                # alpha — амплитуда сдвига в пикселях, sigma — радиус сглаживания поля.
                # albumentations 2.0.8 принимает ТОЛЬКО скалярные float (кортежи -> pydantic
                # ValidationError). alpha=50, sigma=10 -> заметная, но мягкая деформация ткани.
                A.ElasticTransform(alpha=50, sigma=10, p=0.4),

                # Пиксельные искажения (только на изображении, маска не трогается)
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.4),
                # GaussNoise — image-only трансформ, НЕ портит маску (в отличие от CoarseDropout,
                # который занулял метки в выпавших областях -> несогласованные пары image/mask).
                # GaussNoise — image-only трансформ, НЕ портит маску (в отличие от CoarseDropout,
                # который занулял метки в выпавших областях -> несогласованные пары image/mask).
                # albumentations 2.0.8: параметр НЕ var_limit, а std_range — в пикселях исходника
                # (0-255). std_range=(5,15) даёт заметный, но мягкий сенсорный шум.
                A.GaussNoise(std_range=(0.02, 0.06), p=0.2),

                # Нормализация (согласована с encoder_weights='imagenet')
                make_normalize(self.imagenet_norm),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.files)

    def _label_path(self, name, folder, suffix):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Нет папки меток: {folder}")
        entries = {}
        for f in os.listdir(folder):
            p = os.path.join(folder, f)
            if not os.path.isfile(p):
                continue
            if os.path.splitext(f)[1].lower() not in IMG_EXTS:
                continue
            stem = os.path.splitext(f)[0]
            entries[stem.lower()] = p
        nl = name.lower()
        base_suf = suffix.lower()
        short = base_suf.lstrip("_")
        candidates = [nl + base_suf, nl, nl + base_suf + "s", nl + "-" + short, nl + short]
        for c in candidates:
            if c in entries:
                return entries[c]
        raise FileNotFoundError(
            f"Не найдена метка для '{name}' в {folder}. "
            f"Ожидались файлы вида '{name}{suffix}.tif/.png/.bmp' или '{name}.tif'. "
            f"Найдено меток в папке: {len(entries)}."
        )

    def __getitem__(self, idx):
        img_path = self.files[idx]
        name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Получаем изображение в формате uint8 (H, W, C)
        image, _meta = _read_image(img_path, self.size)

        if self.has_labels:
            mask = _read_mask(self._label_path(name, self.masks_dir, "_mask"), self.size)
            bound = _read_mask(self._label_path(name, self.bound_dir, "_boundary"), self.size)
            
            # Объединяем две маски в одну матрицу с 2 каналами: (H, W, 2)
            target = np.stack([mask, bound], axis=-1)

            # Применяем трансформации (синхронно для фото и обоих слоев маски)
            augmented = self.transform(image=image, mask=target)
            
            image_t = augmented['image']
            
            # Albumentations возвращает маску в виде тензора (H, W, C). 
            # Нам нужно поменять оси на (C, H, W) для PyTorch.
            target_t = augmented['mask'].permute(2, 0, 1).float()
            
            return image_t, target_t, name
        else:
            augmented = self.transform(image=image)
            image_t = augmented['image']
            
            # Генерируем пустой таргет, сохраняя нужную размерность
            dummy = torch.zeros((2, image_t.shape[1], image_t.shape[2]), dtype=torch.float32)
            
            return image_t, dummy, name
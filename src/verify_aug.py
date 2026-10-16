"""
verify_aug.py — проверка, что аугментация в dataset.py реально работает и не портит маску.
Использует синтетические данные, чтобы не зависеть от наличия реальных изображений.
"""
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_normalize(imagenet_norm=True):
    if imagenet_norm:
        return A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0)
    return A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0)


def build_aug(imagenet_norm=True):
    """Точная копия пайплайна из dataset.py (augment=True)."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
            scale=(0.95, 1.05), translate_percent=(-0.03, 0.03), rotate=(-180, 180),
            border_mode=1, fill=0, fill_mask=0, p=0.6  # 1==cv2.BORDER_REFLECT
        ),
        A.ElasticTransform(alpha=50, sigma=10, p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.4),
        A.GaussNoise(std_range=(0.02, 0.06), p=0.2),
        make_normalize(imagenet_norm),
        ToTensorV2()
    ])


def synth_image():
    # простое цветное поле с кругом
    h = w = 256
    img = np.zeros((h, w, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    circle = (xx - 128) ** 2 + (yy - 128) ** 2 < 60 ** 2
    img[circle] = (200, 100, 50)
    img[:, 0:10] = (10, 200, 10)  # цветная полоса, чтобы заметить сдвиги
    return img


def synth_target():
    # [mask, bound] бинарные
    h = w = 256
    mask = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    mask[(xx - 128) ** 2 + (yy - 128) ** 2 < 60 ** 2] = 1.0
    bound = np.zeros((h, w), dtype=np.float32)
    ring = ((xx - 128) ** 2 + (yy - 128) ** 2 < 62 ** 2) & ((xx - 128) ** 2 + (yy - 128) ** 2 > 58 ** 2)
    bound[ring] = 1.0
    return np.stack([mask, bound], axis=-1).astype(np.float32)


def main():
    print("== build pipeline ==")
    tf = build_aug(imagenet_norm=True)
    print("OK: pipeline built (no pydantic ValidationError)")

    img0 = synth_image()
    tgt0 = synth_target()
    mask0 = tgt0[..., 0]

    n = 200
    img_changed = 0        # изображение изменилось хотя бы в 1 пикселе
    mask_changed = 0       # маска изменилась (ожидаем >0 из-за геометрии/elastic)
    mask_expected_same = 0 # маска НЕ должна меняться от GaussNoise/BrightnessContrast
    mask_binary_violations = 0  # маска должна остаться {0,1}

    for i in range(n):
        out = tf(image=img0.copy(), mask=tgt0.copy())
        oimg = out["image"].permute(1, 2, 0).numpy()  # (H,W,C) float, normalized
        otgt = out["mask"].numpy()  # (H,W,2) float
        omask = otgt[..., 0]
        # изображение меняется относительно исходного uint8? сравниваем до/после нормализации
        # проще: сравниваем маску с исходной бинарной
        if not np.array_equal(omask, mask0):
            mask_changed += 1
        # маска должна остаться бинарной {0,1}
        if not np.allclose(omask, omask.astype(bool).astype(np.float32)):
            mask_binary_violations += 1

    print(f"trials={n}")
    print(f"mask changed (geometry/elastic deform): {mask_changed}/{n}")
    print(f"mask binary violations (must be 0): {mask_binary_violations}")

    # Проверка, что GaussNoise не меняет маску, но МЕНЯЕТ изображение
    print("\n== GaussNoise isolation (p=1.0) ==")
    tf2 = A.Compose([
        A.GaussNoise(std_range=(0.02, 0.06), p=1.0),
        make_normalize(True),
        ToTensorV2()
    ])
    gnoise_mask_changed = 0
    gnoise_img_mean_abs = 0.0
    for i in range(50):
        out = tf2(image=img0.copy(), mask=tgt0.copy())
        omask = out["mask"].numpy()[..., 0]
        oimg = out["image"].permute(1, 2, 0).numpy()  # normalized, ~[-2,2]
        if not np.array_equal(omask, mask0):
            gnoise_mask_changed += 1
        # денормализуем обратно к 0-255 для оценки величины шума
        den = (oimg * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)) * 255.0
        gnoise_img_mean_abs += np.mean(np.abs(den.astype(np.float32) - img0.astype(np.float32)))
    gnoise_img_mean_abs /= 50
    print(f"GaussNoise mask changed (must be 0): {gnoise_mask_changed}/50")
    print(f"GaussNoise mean |img diff| (pixels 0-255, want >0): {gnoise_img_mean_abs:.2f}")

    # Проверка ElasticTransform в одиночку деформирует маску
    print("\n== ElasticTransform isolation (p=1.0, alpha=50, sigma=10) ==")
    tf3 = A.Compose([
        A.ElasticTransform(alpha=50, sigma=10, p=1.0),
        make_normalize(True),
        ToTensorV2()
    ])
    elastic_mask_changed = 0
    for i in range(50):
        out = tf3(image=img0.copy(), mask=tgt0.copy())
        omask = out["mask"].numpy()[..., 0]
        if not np.array_equal(omask, mask0):
            elastic_mask_changed += 1
    print(f"ElasticTransform mask changed (want ~50/50): {elastic_mask_changed}/50")

    ok = (mask_changed > 0) and (mask_binary_violations == 0) and (gnoise_mask_changed == 0) and (gnoise_img_mean_abs > 0) and (elastic_mask_changed > 0)
    print("\nRESULT:", "PASS" if ok else "FAIL")
    assert ok, "augmentation verification failed"


if __name__ == "__main__":
    main()

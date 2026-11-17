r"""
analyze_masks.py — этап АНАЛИЗА масок в пайплайне astro_seg_light.

Адаптирован из C:/Users/Vanya_meduha/Desktop/Astrocyte_Analysis_Toolkit/astrocyte_analyzer.py,
но сделан tissue-aware: учитывает целочисленную метку-карту ткани
(серое / белое вещество / просвет) и считает метрики астроцитов в ВЫБРАННОЙ
ОБЛАСТИ, отрезая просвет.

Модель фокуса/расфокуса исключена из пайплайна (недостаточная доработка
чекпоинта). Оценка резкости более не применяется, ограничение по фокусу
полностью убрано.

Вход:
  --masks  дерево БИНАРНЫХ масок астроцитов (вых. predict_tree с GFAP-моделью, astro_*.pth).
  --tissue дерево ЦЕЛОЧИСЛЕННЫХ меток-карт *_label.png (вых. predict_tree с white_and_grey.pth).
           Семантика: 0=просвет, 1=серое вещество, 2=белое вещество.
  --region {all, tissue, gray, white}  (по умолчанию all)
           all    — по всей маске астроцитов
           tissue — исключая просвет (label != 0)
           gray   — только label == 1 (серое вещество)
           white  — только label == 2 (белое вещество)
  --min_area  мин. площадь компоненты (px), по умолчанию 100.
  --out       куда писать результаты (по умолчанию <masks>_analysis).

Типичный сценарий (пайплайн predict_all.py):
  считать астроциты в СЕРОМ веществе:  --region gray
  считать астроциты в БЕЛОМ веществе:  --region white
  -> просвет (tissue=0) исключается автоматически.

Выход:
  - Global_Metrics.csv / Cell_Metrics.csv — всегда (без внешних зависимостей).
  - Astrocyte_Analysis_Results.xlsx (Global_Metrics + Cell_Metrics) — ВСЕГДА,
    через xlsx_writer (чистый стандартный Python, без pandas/openpyxl).
    Настоящая структура колонок + тёмная шапка + границы + формат 0.000,
    как в образцовом Astrocyte_Analysis_Results.xlsx.
  - QC_<имя>.png — маска области с контурами (зелёный=валидная клетка, красный=отбраковка).
"""
import os
import sys
import csv
import argparse
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import find_images_recursive
from xlsx_writer import write_workbook

# Коды классов в метке-карте ткани (порядок channel_names модели white_and_grey:
# [black, gray, white] -> 0=просвет, 1=серое, 2=белое).
REGION_CODES = {"gray": 1, "white": 2}
REGION_NAMES = {
    "all": "All",
    "tissue": "Tissue (no lumen)",
    "gray": "Gray matter",
    "white": "White matter",
}


def calculate_fd_lacunarity(img, min_size=2):
    img_bool = img > 0
    p = min(img_bool.shape)
    if p < min_size:
        return 0, 0

    n = 2 ** int(np.floor(np.log2(p)))
    sizes = 2 ** np.arange(int(np.log2(n)), int(np.log2(min_size)) - 1, -1)

    counts, lacunarities = [], []
    sizes_used = []

    for size in sizes:
        shape = (img_bool.shape[0] // size, size, img_bool.shape[1] // size, size)
        truncated = img_bool[0:shape[0] * size, 0:shape[2] * size]
        blocks = truncated.reshape(shape).sum(axis=(1, 3))

        active_boxes = np.sum(blocks > 0)
        if active_boxes > 0:
            counts.append(active_boxes)
            sizes_used.append(size)
            active_masses = blocks[blocks > 0]
            mean_sq = np.mean(active_masses) ** 2
            var = np.var(active_masses)
            lac = (var / mean_sq) + 1 if mean_sq > 0 else 0
            lacunarities.append(lac)

    # ВАЖНО: counts собирается только для шкал с active_boxes>0, поэтому размерность
    # counts может быть меньше sizes (на грубых шкалах блоб может попасть в отброшенный
    # краевой кусок truncated). Для polyfit берём только те шкалы, что реально вошли в counts.
    if len(sizes_used) > 1:
        coeffs = np.polyfit(np.log(1. / np.asarray(sizes_used, dtype=float)),
                            np.log(np.asarray(counts, dtype=float)), 1)
        fd = coeffs[0]
    else:
        fd = 0

    mean_lac = np.mean(lacunarities) if lacunarities else 0
    return fd, mean_lac


def analyze_one(mask_bin, region_mask, region_label, min_area, img_name, out_dir):
    """Считает метрики астроцитов на mask_bin, ограниченном region_mask (0/255 или None).
    region_label — строка для колонки Region в отчёте."""
    if region_mask is None:
        restricted = mask_bin.copy()
    else:
        restricted = (mask_bin & region_mask).astype(np.uint8)

    h, w = restricted.shape
    valid_mask = np.zeros_like(restricted)

    qc_image = cv2.cvtColor(restricted, cv2.COLOR_GRAY2BGR)
    valid_cell_count = 0
    cell_metrics = []

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(restricted, connectivity=8)

    for i in range(1, num_labels):
        x, y, w_box, h_box, area = stats[i]

        touches_border = (x == 0) or (y == 0) or (x + w_box == w) or (y + h_box == h)

        cell_mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if touches_border or area < min_area:
            if contours:
                cv2.drawContours(qc_image, contours, -1, (0, 0, 255), 1)
            continue

        if contours:
            cnt = contours[0]
            cv2.drawContours(qc_image, [cnt], -1, (0, 255, 0), 1)
            valid_mask = cv2.bitwise_or(valid_mask, cell_mask)
            valid_cell_count += 1

            perimeter = cv2.arcLength(cnt, True)
            hull = cv2.convexHull(cnt)
            convex_area = cv2.contourArea(hull)

            solidity = area / convex_area if convex_area > 0 else 0
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

            cell_metrics.append({
                'File_Name': img_name,
                'Region': region_label,
                'Cell_ID': valid_cell_count,
                'Area': area,
                'Perimeter': perimeter,
                'Convex_Area': convex_area,
                'Solidity': solidity,
                'Circularity': circularity,
            })

            cx, cy = int(centroids[i][0]), int(centroids[i][1])
            cv2.putText(qc_image, str(valid_cell_count), (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    qc_path = os.path.join(out_dir, f"QC_{img_name}")
    cv2.imwrite(qc_path, qc_image)

    area_fraction = (np.sum(valid_mask > 0) / (h * w)) * 100
    fd, lac = calculate_fd_lacunarity(valid_mask)

    global_metric = {
        'File_Name': img_name,
        'Region': region_label,
        'Valid_Cell_Count': valid_cell_count,
        'Area_Fraction_%': area_fraction,
        'Global_Fractal_Dim': fd,
        'Global_Lacunarity': lac,
    }
    return global_metric, cell_metrics


def run_analysis(masks_root, tissue_root=None, region="all", min_area=100, out_dir=None):
    masks_root = os.path.abspath(masks_root)
    if not os.path.isdir(masks_root):
        raise FileNotFoundError(f"Нет папки масок: {masks_root}")
    out_dir = os.path.abspath(out_dir) if out_dir else (masks_root.rstrip(os.sep) + "_analysis")
    os.makedirs(out_dir, exist_ok=True)

    tissue_root = os.path.abspath(tissue_root) if tissue_root else None

    if region != "all" and tissue_root is None:
        print(f"Внимание: регион '{region}' требует --tissue (метки-карты ткани). "
              f"Анализ пойдёт по всей маске (region=all).")
        region = "all"

    mask_files = find_images_recursive(masks_root)
    if not mask_files:
        print(f"Внимание: в '{masks_root}' нет изображений.")
        return

    region_label = REGION_NAMES[region]

    print(f"masks={masks_root}\n"
          f"tissue={tissue_root}\n"
          f"region={region_label}\n"
          f"min_area={min_area}\nout={out_dir}\nфайлов: {len(mask_files)}")

    global_metrics = []
    cell_metrics = []

    for path in mask_files:
        rel = os.path.relpath(path, masks_root)
        img_name = os.path.basename(path)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  пропущено (не читается): {rel}")
            continue
        _, mask_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # --- ограничение по ткани: серое (1) / белое (2), просвет (0) исключаем ---
        region_mask = None
        if tissue_root is not None and region != "all":
            stem = os.path.splitext(rel)[0]
            label_rel = stem + "_label.png"
            label_path = os.path.join(tissue_root, label_rel)
            if not os.path.exists(label_path):
                print(f"  нет метки-карты ткани для {rel} (ожидалось {label_rel}) -> регион all")
            else:
                label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                if region == "tissue":
                    region_mask = (label > 0).astype(np.uint8) * 255
                else:
                    code = REGION_CODES[region]
                    region_mask = (label == code).astype(np.uint8) * 255

        try:
            g, cells = analyze_one(mask_bin, region_mask, region_label, min_area, img_name, out_dir)
        except Exception as e:
            # Один проблемный снимок не должен обрывать весь пакет (1811 изображений).
            print(f"  [ОШИБКА] {rel}: {type(e).__name__}: {e} -> пропущено")
            g = {'File_Name': img_name, 'Region': region_label,
                 'Valid_Cell_Count': 0, 'Area_Fraction_%': 0.0,
                 'Global_Fractal_Dim': 0.0, 'Global_Lacunarity': 0.0}
            cells = []
        global_metrics.append(g)
        cell_metrics.extend(cells)
        print(f"  {rel} [{region_label}]: клеток={g['Valid_Cell_Count']} "
              f"area%={g['Area_Fraction_%']:.2f} FD={g['Global_Fractal_Dim']:.3f}")

    # CSV — всегда (без зависимостей)
    if global_metrics:
        g_csv = os.path.join(out_dir, "Global_Metrics.csv")
        c_csv = os.path.join(out_dir, "Cell_Metrics.csv")
        _write_csv(g_csv, global_metrics)
        if cell_metrics:
            _write_csv(c_csv, cell_metrics)
        print(f"CSV сохранён: {g_csv}" + (f" и {c_csv}" if cell_metrics else ""))
    else:
        print("Нет данных для сохранения.")
        return

    # Excel — ВСЕГДА, без внешних зависимостей (xlsx_writer, чистый stdlib).
    excel_path = os.path.join(out_dir, "Astrocyte_Analysis_Results.xlsx")
    g_cols = list(global_metrics[0].keys())
    if cell_metrics:
        c_cols = list(cell_metrics[0].keys())
        c_rows = cell_metrics
    else:
        # пустой, но валидный лист Cell_Metrics (как в образце)
        c_cols = ["File_Name", "Region", "Cell_ID", "Area", "Perimeter",
                  "Convex_Area", "Solidity", "Circularity"]
        c_rows = []
    try:
        write_workbook(excel_path, [
            {"name": "Global_Metrics", "columns": g_cols, "rows": global_metrics},
            {"name": "Cell_Metrics", "columns": c_cols, "rows": c_rows},
        ])
        print(f"Excel сохранён: {excel_path}")
    except Exception as e:
        print(f"[ВНИМАНИЕ] не удалось сформировать Excel: {e}. CSV сохранён.")


def _write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True,
                    help="дерево бинарных масок астроцитов (вых. predict_tree с GFAP-моделью)")
    ap.add_argument("--tissue", default=None,
                    help="дерево меток-карт ткани *_label.png (вых. predict_tree с white_and_grey.pth)")
    ap.add_argument("--region", choices=["all", "tissue", "gray", "white"], default="all",
                    help="область ткани: all / tissue(без просвета) / gray / white")
    ap.add_argument("--min_area", type=int, default=100,
                    help="мин. площадь компоненты (px)")
    ap.add_argument("--out", default=None, help="выходная папка (по умолчанию <masks>_analysis)")
    args = ap.parse_args()

    run_analysis(args.masks, args.tissue, args.region, args.min_area, args.out)


if __name__ == "__main__":
    main()

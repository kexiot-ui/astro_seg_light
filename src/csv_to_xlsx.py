r"""
csv_to_xlsx.py — пересобрать Astrocyte_Analysis_Results.xlsx из существующих CSV
(Global_Metrics.csv / Cell_Metrics.csv) в папке анализа, через xlsx_writer.

Полезно, когда метрики уже посчитаны, а Excel нужен «прямо сейчас» — без
повторного прогона analyze_masks (1811 изображений заново считать не нужно).

Пример:
  python src/csv_to_xlsx.py "D:/astro_seg_light/GFAP_irr_study_seg_analysis/gray_focus"
  python src/csv_to_xlsx.py "D:/astro_seg_light/GFAP_irr_study_seg_analysis/white_focus"
"""
import os
import sys
import csv
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from xlsx_writer import write_workbook


def _read_csv(path):
    if not os.path.exists(path):
        return [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            # приводим числовые колонки к float/int по возможности
            clean = {}
            for k, v in r.items():
                if v is None or v == "":
                    clean[k] = ""
                    continue
                try:
                    if "." in v or "e" in v.lower():
                        clean[k] = float(v)
                    else:
                        clean[k] = int(v)
                except ValueError:
                    clean[k] = v
            rows.append(clean)
        return list(reader.fieldnames) if False else (list(rows[0].keys()) if rows else []), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis_dir", help="папка с Global_Metrics.csv / Cell_Metrics.csv")
    args = ap.parse_args()

    d = os.path.abspath(args.analysis_dir)
    g_cols, g_rows = _read_csv(os.path.join(d, "Global_Metrics.csv"))
    c_cols, c_rows = _read_csv(os.path.join(d, "Cell_Metrics.csv"))

    if not g_rows:
        print(f"Нет данных в {d}/Global_Metrics.csv — выход.")
        return

    if not c_cols:
        c_cols = ["File_Name", "Region", "Cell_ID", "Area", "Perimeter",
                  "Convex_Area", "Solidity", "Circularity"]
        c_rows = []

    out = os.path.join(d, "Astrocyte_Analysis_Results.xlsx")
    write_workbook(out, [
        {"name": "Global_Metrics", "columns": g_cols, "rows": g_rows},
        {"name": "Cell_Metrics", "columns": c_cols, "rows": c_rows},
    ])
    print(f"Excel пересобран: {out}  (Global_Metrics={len(g_rows)} строк, "
          f"Cell_Metrics={len(c_rows)} строк)")


if __name__ == "__main__":
    main()

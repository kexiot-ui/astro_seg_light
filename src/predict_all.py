r"""
predict_all.py — ЕДИНАЯ КОМАНДА: сегментация по трём моделям + авто-анализ.

Гонит по одному дереву RGB-изображений сразу три модели:
  1. Астроциты (GFAP)        -> models/astro_2.pth     (бинарная маска),   выход <base>_astro
  2. Ткань (серое/белое/      -> models/white_and_grey.pth (цвет + метка), выход <base>_tissue
     просвет)
  3. Фокус (в фокусе/         -> models/focus1.pth     (бинарная + метка), выход <base>_focus
     расфокус)

Затем автоматически считает астроциты:
  - в СЕРОМ веществе В ФОКУСЕ  -> <base>_analysis/gray_focus
  - в БЕЛОМ веществе В ФОКУСЕ  -> <base>_analysis/white_focus
Просвет (tissue=0) и расфокус исключаются автоматически — считать их нет нужды.

Пример (одна команда делает всё):
  python src/predict_all.py --input "D:/astro_seg_light/GFAP_irr_study"

Переопределить модель астроцитов:
  python src/predict_all.py --input "..." --astro models/astro_6.pth
"""
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from predict_tree import run as predict_run
from analyze_masks import run_analysis

DEFAULT_ASTRO = "models/astro_2.pth"
DEFAULT_WHITE_GREY = "models/white_and_grey.pth"
DEFAULT_FOCUS = "models/focus1.pth"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="корень дерева RGB-изображений")
    ap.add_argument("--astro", default=DEFAULT_ASTRO, help="модель астроцитов (GFAP), .pth")
    ap.add_argument("--white_grey", default=DEFAULT_WHITE_GREY, help="модель ткани, .pth")
    ap.add_argument("--focus", default=DEFAULT_FOCUS, help="модель фокуса, .pth")
    ap.add_argument("--out", default=None,
                    help="базовая папка выхода (по умолчанию <input>_seg)")
    ap.add_argument("--min_area", type=int, default=100, help="мин. площадь компоненты (px)")
    ap.add_argument("--skip_existing", action="store_true",
                    help="пропускать уже существующие маски на каждом этапе")
    ap.add_argument("--mode", choices=["mask", "overlay"], default="mask",
                    help="формат масок (mask = бинарная одноканальная)")
    args = ap.parse_args()

    base = args.out if args.out else (args.input.rstrip(os.sep) + "_seg")
    astro_out = base + "_astro"
    tissue_out = base + "_tissue"
    focus_out = base + "_focus"
    analysis_base = base + "_analysis"

    print("=" * 72)
    print("ЭТАП 1: сегментация трёмя моделями")
    print("=" * 72)
    predict_run(args.input, args.astro, astro_out,
                mode=args.mode, skip_existing=args.skip_existing, verbose_prefix="[astro] ")
    predict_run(args.input, args.white_grey, tissue_out,
                mode=args.mode, skip_existing=args.skip_existing, verbose_prefix="[tissue] ")
    predict_run(args.input, args.focus, focus_out,
                mode=args.mode, skip_existing=args.skip_existing, verbose_prefix="[focus] ")

    print("\n" + "=" * 72)
    print("ЭТАП 2: анализ астроцитов (серое / белое вещество В ФОКУСЕ)")
    print("=" * 72)
    for region in ["gray", "white"]:
        out_dir = os.path.join(analysis_base, f"{region}_focus")
        print(f"\n--- астроциты в {region} веществе В ФОКУСЕ ---")
        run_analysis(astro_out, tissue_out, focus_out, region=region,
                     focus_region="infocus", focus_in_code=1,
                     min_area=args.min_area, out_dir=out_dir)

    print("\n" + "=" * 72)
    print("ГОТОВО.")
    print(f"  маски астроцитов : {astro_out}")
    print(f"  маски ткани      : {tissue_out}  (цвет + *_label.png)")
    print(f"  маски фокуса     : {focus_out}  (бинарь + *_label.png)")
    print(f"  отчёты анализа   : {analysis_base}/gray_focus  и  {analysis_base}/white_focus")
    print("=" * 72)


if __name__ == "__main__":
    main()

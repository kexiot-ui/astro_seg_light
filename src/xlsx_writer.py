r"""
xlsx_writer.py — минимальный, БЕЗВНЕШНИЗАВИСИМОСТЕЙ генератор .xlsx (только стандартная библиотека).

Нужен, потому что в окружении astro_seg НЕТ ни pandas, ни openpyxl, и CSV-вывод в
русифицированном Excel открывается «в одну длинную строку» (нет настоящей структуры
колонок). Этот модуль пишет настоящий Office Open XML (.xlsx) с:
  - несколькими листами,
  - тёмной заливкой шапки (2C3E50) + белый жирный шрифт,
  - тонкими границами (BDC3C7) у всех ячеек,
  - числовыми колонками с форматом 0.000 (дробные) / общим (целые),
  - закреплённой строкой-шапкой.

Структура совместима с Excel / LibreOffice / WPS (проверено на тех же листах, что и
Astrocyte_Analysis_Results.xlsx из Astrocyte_Analysis_Toolkit).

API:
    from xlsx_writer import write_workbook
    write_workbook(path, sheets)
    # sheets = [ {"name": "Global_Metrics",
    #             "columns": ["File_Name", "Valid_Cell_Count", ...],
    #             "rows": [ {"File_Name": "...", ...}, ... ]}, ... ]
"""
import os
import zipfile
import html
from xml.sax.saxutils import escape as _xml_escape

# Стиль шапки / границ взят из образцового Astrocyte_Analysis_Results.xlsx:
HEADER_FILL = "2C3E50"   # тёмно-синий
BORDER_COLOR = "BDC3C7"  # светло-серый

# Ширины колонок (в символах Excel). Текстовые — шире.
COL_WIDTH = {
    "File_Name": 26,
    "Region": 30,
}


def _col_letter(idx):
    """1->A, 27->AA, ..."""
    s = ""
    while idx > 0:
        idx, r = divmod(idx - 1, 26)
        s = chr(65 + r) + s
    return s


def _infer_coltypes(columns, rows):
    """Вернуть dict col -> 'text' | 'int' | 'float' по фактическим данным."""
    types = {}
    for col in columns:
        vals = [r.get(col) for r in rows if r.get(col) is not None and r.get(col) != ""]
        is_int = all(isinstance(v, int) and not isinstance(v, bool) for v in vals)
        is_num = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals)
        if is_int:
            types[col] = "int"
        elif is_num:
            types[col] = "float"
        else:
            types[col] = "text"
    return types


def _cell_xml(ref, value, style, coltype):
    """Сформировать <c> для одной ячейки."""
    s = str(style)
    if coltype == "text" or not isinstance(value, (int, float)) or isinstance(value, bool):
        text = "" if value is None else str(value)
        # inline string; xml:space=preserve чтобы пробелы (в Region) не съедались
        return (f'<c r="{ref}" s="{s}" t="inlineStr">'
                f'<is><t xml:space="preserve">{_xml_escape(text)}</t></is></c>')
    else:
        num = value
        if isinstance(num, float):
            v = repr(num)
        else:
            v = str(int(num))
        return f'<c r="{ref}" s="{s}"><v>{v}</v></c>'


def _sheet_xml(name, columns, rows, coltypes):
    parts = []
    parts.append('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>')
    parts.append('<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">')
    # закрепляем шапку (первую строку)
    parts.append('<sheetViews><sheetView workbookViewId="0">'
                 '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
                 '</sheetView></sheetViews>')
    # ширины колонок
    parts.append('<cols>')
    for i, col in enumerate(columns, start=1):
        w = COL_WIDTH.get(col, 16)
        parts.append(f'<col min="{i}" max="{i}" width="{w}" customWidth="1"/>')
    parts.append('</cols>')
    parts.append('<sheetData>')
    # строка 1 — шапка (стиль 1)
    hdr_cells = []
    for i, col in enumerate(columns, start=1):
        ref = f"{_col_letter(i)}1"
        hdr_cells.append(f'<c r="{ref}" s="1" t="inlineStr">'
                         f'<is><t xml:space="preserve">{_xml_escape(str(col))}</t></is></c>')
    parts.append('<row r="1">' + "".join(hdr_cells) + '</row>')
    # данные
    for ri, row in enumerate(rows, start=2):
        cells = []
        for i, col in enumerate(columns, start=1):
            ref = f"{_col_letter(i)}{ri}"
            value = row.get(col)
            ct = coltypes.get(col, "text")
            # стиль: 2 = текст-граница, 3 = дробь 0.000-граница, 4 = целое-граница
            style = 2 if ct == "text" else (3 if ct == "float" else 4)
            cells.append(_cell_xml(ref, value, style, ct))
        parts.append(f'<row r="{ri}">' + "".join(cells) + '</row>')
    parts.append('</sheetData>')
    parts.append('</worksheet>')
    return "".join(parts)


_STYLES_XML = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<numFmts count="1"><numFmt numFmtId="164" formatCode="0.000"/></numFmts>
<fonts count="2">
<font><sz val="11"/><name val="Calibri"/></font>
<font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/></font>
</fonts>
<fills count="3">
<fill><patternFill patternType="none"/></fill>
<fill><patternFill patternType="gray125"/></fill>
<fill><patternFill patternType="solid"><fgColor rgb="FF''' + HEADER_FILL + '''"/><bgColor indexed="64"/></patternFill></fill>
</fills>
<borders count="2">
<border><left/><right/><top/><bottom/><diagonal/></border>
<border>
<left style="thin"><color rgb="FF''' + BORDER_COLOR + '''"/></left>
<right style="thin"><color rgb="FF''' + BORDER_COLOR + '''"/></right>
<top style="thin"><color rgb="FF''' + BORDER_COLOR + '''"/></top>
<bottom style="thin"><color rgb="FF''' + BORDER_COLOR + '''"/></bottom>
<diagonal/>
</border>
</borders>
<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
<cellXfs count="5">
<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
<xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyFont="1" applyFill="1" applyBorder="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>
<xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyBorder="1" applyAlignment="1"><alignment horizontal="left" vertical="center"/></xf>
<xf numFmtId="164" fontId="0" fillId="0" borderId="1" xfId="0" applyNumberFormat="1" applyBorder="1" applyAlignment="1"><alignment horizontal="right" vertical="center"/></xf>
<xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyBorder="1" applyAlignment="1"><alignment horizontal="right" vertical="center"/></xf>
</cellXfs>
<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>'''


def write_workbook(path, sheets):
    """sheets: list of dicts {name, columns, rows}. Пишет настоящий .xlsx."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # workbook.xml
    sheet_tags = []
    for i, sh in enumerate(sheets, start=1):
        safe = _xml_escape(sh["name"])
        sheet_tags.append(f'<sheet name="{safe}" sheetId="{i}" r:id="rId{i}"/>')
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets>' + "".join(sheet_tags) + '</sheets></workbook>'
    )

    # workbook rels: worksheets + styles
    rels = []
    for i in range(1, len(sheets) + 1):
        rels.append(f'<Relationship Id="rId{i}" '
                    f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
                    f'Target="worksheets/sheet{i}.xml"/>')
    rels.append(f'<Relationship Id="rId{len(sheets)+1}" '
                f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
                f'Target="styles.xml"/>')
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(rels) + '</Relationships>'
    )

    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
    )
    for i in range(1, len(sheets) + 1):
        content_types += (f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
                           f'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
    content_types += ('<Override PartName="/xl/styles.xml" '
                      'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
                      '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
                      '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
                      '</Types>')

    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
        '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
        '</Relationships>'
    )

    core_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        '<dc:creator>astro_seg_light</dc:creator><cp:lastModifiedBy>astro_seg_light</cp:lastModifiedBy>'
        '</cp:coreProperties>'
    )
    app_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
        '<Application>astro_seg_light</Application></Properties>'
    )

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", root_rels)
        z.writestr("docProps/core.xml", core_xml)
        z.writestr("docProps/app.xml", app_xml)
        z.writestr("xl/workbook.xml", workbook_xml)
        z.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        z.writestr("xl/styles.xml", _STYLES_XML)
        for i, sh in enumerate(sheets, start=1):
            columns = list(sh["columns"])
            rows = list(sh.get("rows", []))
            coltypes = _infer_coltypes(columns, rows)
            z.writestr(f"xl/worksheets/sheet{i}.xml", _sheet_xml(sh["name"], columns, rows, coltypes))
    return path

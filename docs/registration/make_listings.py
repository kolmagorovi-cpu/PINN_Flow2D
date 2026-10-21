# -*- coding: utf-8 -*-
"""Сборка листингов исходного кода по модулям для заявки на регистрацию программы.

Для каждой программы (Airfoil PINN — «профиль», Pipe-Bend PINN — «отвод») формирует:
  * <tag>_modules.csv   — реестр модулей (группа, файл, строк, байт);
  * <tag>_listing.txt   — полный листинг с заголовками модулей;
  * <tag>_listing.docx  — листинг по модулям (Courier New, нумерация страниц).

Запуск:  python make_listings.py
"""
import csv
import datetime
import os

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.dirname(os.path.abspath(__file__))

PROJECTS = [
    dict(
        tag="airfoil",
        short="Airfoil PINN",
        title=("Программа для ЭВМ «Нейросетевое моделирование обтекания "
               "аэродинамического профиля (PINN, Airfoil)»"),
        root=os.path.join(BASE, "Airfoil PINN"),
    ),
    dict(
        tag="pipebend",
        short="Pipe-Bend PINN",
        title=("Программа для ЭВМ «Нейросетевое моделирование течения в канале "
               "с отводом 90° (PINN, Pipe-Bend)»"),
        root=os.path.join(BASE, "Pipe-Bend PINN"),
    ),
]

SKIP_DIRS = {
    "__pycache__", ".venv", "venv", ".git", "workspace", "Sampler",
    "experiments", "backups", ".idea", ".vscode", "node_modules",
}
GROUP_ORDER = [
    "app", "config", "core", "data", "geometry", "physics",
    "sampling", "training", "utils", "scripts", "tests",
]


def group_of(rel):
    """Определяет группу модуля по вложенному подпакету (app, physics, ...)."""
    parts = rel.split("/")
    dirs = parts[:-1]
    for token in GROUP_ORDER:
        if token in dirs:
            return token
    return dirs[0] if dirs else "(корень)"


def collect(root):
    """Собирает .py-файлы проекта, сортируя их по группам модулей."""
    items = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            path = os.path.join(dirpath, fname)
            rel = os.path.relpath(path, root).replace("\\", "/")
            grp = group_of(rel)
            try:
                text = open(path, encoding="utf-8").read()
            except UnicodeDecodeError:
                text = open(path, encoding="cp1251", errors="replace").read()
            items.append(dict(
                rel=rel, grp=grp, lines=text.count("\n") + 1,
                bytes=os.path.getsize(path), text=text,
            ))

    def sort_key(item):
        if item["grp"] in GROUP_ORDER:
            order = GROUP_ORDER.index(item["grp"])
        elif item["grp"] == "(корень)":
            order = 999
        else:
            order = 500
        return (order, item["grp"], item["rel"])

    items.sort(key=sort_key)
    return items


def write_csv(proj, items):
    out = os.path.join(OUT, proj["tag"] + "_modules.csv")
    with open(out, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["№", "Группа (модуль)", "Файл", "Строк", "Байт"])
        for idx, item in enumerate(items, 1):
            writer.writerow([idx, item["grp"], item["rel"], item["lines"], item["bytes"]])
        writer.writerow([])
        writer.writerow(["ИТОГО", "", "%d файлов" % len(items),
                         sum(i["lines"] for i in items), sum(i["bytes"] for i in items)])
    return out


def write_txt(proj, items):
    out = os.path.join(OUT, proj["tag"] + "_listing.txt")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(proj["title"] + "\n")
        fh.write("Листинг исходного кода по модулям\n")
        fh.write("Дата сборки: %s\n" % datetime.date.today().isoformat())
        fh.write("Состав: %d модулей, %d строк\n\n" % (len(items), sum(i["lines"] for i in items)))
        for idx, item in enumerate(items, 1):
            fh.write("\n" + "=" * 100 + "\n")
            fh.write("Модуль %d: %s   (%d строк)\n" % (idx, item["rel"], item["lines"]))
            fh.write("=" * 100 + "\n\n")
            fh.write(item["text"].rstrip("\n") + "\n")
    return out


def write_docx(proj, items):
    """Формирует DOCX-листинг по модулям (Courier New 9, нумерация страниц)."""
    from docx import Document
    from docx.enum.text import WD_BREAK
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt, Cm

    doc = Document()

    normal = doc.styles["Normal"]
    normal.font.name = "Courier New"
    normal.font.size = Pt(9)
    rpr = normal.element.get_or_add_rPr()
    rfonts = rpr.get_or_add_rFonts()
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(attr), "Courier New")
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.line_spacing = 1.0

    for section in doc.sections:
        section.top_margin = Cm(1.5)
        section.bottom_margin = Cm(1.5)
        section.left_margin = Cm(2.0)
        section.right_margin = Cm(1.5)

    # Колонтитул: слева — программа, справа — номер страницы
    footer_p = doc.sections[0].footer.paragraphs[0]
    footer_p.text = proj["short"] + "  —  листинг исходного кода  —  стр. "
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), " PAGE ")
    footer_p._p.append(fld)

    def force_preserve(paragraph):
        for node in paragraph._p.iter(qn("w:t")):
            node.set(qn("xml:space"), "preserve")

    # Титульный блок
    doc.add_paragraph(proj["title"]).runs[0].bold = True
    doc.add_paragraph("")
    doc.add_paragraph("ЛИСТИНГ ИСХОДНОГО КОДА (по модулям)").runs[0].bold = True
    total_lines = sum(i["lines"] for i in items)
    doc.add_paragraph("Дата сборки: %s" % datetime.date.today().isoformat())
    doc.add_paragraph("Число модулей: %d" % len(items))
    doc.add_paragraph("Общий объём: %d строк" % total_lines)
    doc.add_paragraph("Язык реализации: Python 3")
    doc.add_paragraph("")
    doc.add_paragraph("Правообладатель: ______________________________")
    doc.add_paragraph("Авторы: ______________________________________")
    doc.add_paragraph("Год: ______")
    doc.add_page_break()

    # Реестр модулей
    doc.add_paragraph("РЕЕСТР МОДУЛЕЙ").runs[0].bold = True
    doc.add_paragraph("")
    table = doc.add_table(rows=1, cols=5)
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    for cell, txt in zip(hdr, ("№", "Группа (модуль)", "Файл", "Строк", "Байт")):
        cell.text = txt
        cell.paragraphs[0].runs[0].bold = True
    for idx, item in enumerate(items, 1):
        row = table.add_row().cells
        row[0].text = str(idx)
        row[1].text = item["grp"]
        row[2].text = item["rel"]
        row[3].text = str(item["lines"])
        row[4].text = str(item["bytes"])
    doc.add_page_break()

    # Листинг по модулям
    for idx, item in enumerate(items, 1):
        header = doc.add_paragraph()
        run = header.add_run("Модуль %d: %s   (%d строк)" % (idx, item["rel"], item["lines"]))
        run.bold = True
        force_preserve(header)
        code_lines = item["text"].rstrip("\n").split("\n")
        for line in code_lines:
            par = doc.add_paragraph(line if line.strip() else "")
            force_preserve(par)
        if idx != len(items):
            doc.add_page_break()

    out = os.path.join(OUT, proj["tag"] + "_listing.docx")
    doc.save(out)
    return out


if __name__ == "__main__":
    for project in PROJECTS:
        modules = collect(project["root"])
        paths = [write_csv(project, modules), write_txt(project, modules)]
        try:
            paths.append(write_docx(project, modules))
        except ImportError:
            print("!!! python-docx не установлен — DOCX пропущен")
        print("== %s: %d модулей, %d строк" % (
            project["short"], len(modules), sum(m["lines"] for m in modules)))
        for p in paths:
            print("    %s (%.1f КБ)" % (os.path.basename(p), os.path.getsize(p) / 1024.0))

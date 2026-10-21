# -*- coding: utf-8 -*-
"""Простое преобразование сводки Markdown -> DOCX (с встроенными рисунками)."""
import os

from docx import Document
from docx.shared import Cm, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.dirname(HERE)
SRC = os.path.join(DOCS, "pipe_bend_hypotheses_summary.md")
DST = os.path.join(DOCS, "pipe_bend_hypotheses_summary.docx")


def clean(text):
    return text.replace("**", "").replace("`", "").strip()


def main():
    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)

    table_rows = []

    def flush_table():
        if not table_rows:
            return
        rows = [r for r in table_rows if not set(r.replace("|", "").strip()) <= {"-", ":", " "}]
        table_rows.clear()
        if not rows:
            return
        cols = len([c for c in rows[0].split("|") if c.strip() != ""])
        table = doc.add_table(rows=0, cols=max(cols, 1))
        table.style = "Table Grid"
        for row in rows:
            cells = [clean(c) for c in row.strip().strip("|").split("|")]
            while len(cells) < cols:
                cells.append("")
            row_cells = table.add_row().cells
            for i, value in enumerate(cells[:cols]):
                row_cells[i].text = value
        doc.add_paragraph("")

    for raw in open(SRC, encoding="utf-8").read().splitlines():
        line = raw.rstrip()
        if line.startswith("|"):
            table_rows.append(line)
            continue
        flush_table()

        if not line.strip():
            continue
        if line.startswith("!["):
            path = line[line.index("(") + 1:line.rindex(")")]
            image = os.path.join(DOCS, path.replace("/", os.sep))
            if os.path.exists(image):
                doc.add_picture(image, width=Cm(16.5))
            continue
        if line.startswith("### "):
            doc.add_heading(clean(line[4:]), level=3)
        elif line.startswith("## "):
            doc.add_heading(clean(line[3:]), level=2)
        elif line.startswith("# "):
            doc.add_heading(clean(line[2:]), level=1)
        elif line.lstrip().startswith(("- ", "* ")):
            doc.add_paragraph(clean(line.lstrip()[2:]), style="List Bullet")
        elif line.lstrip()[:2].isdigit() and "." in line.lstrip()[:4]:
            doc.add_paragraph(clean(line), style="List Number")
        else:
            doc.add_paragraph(clean(line))

    flush_table()
    doc.save(DST)
    print("saved:", DST)
    check = Document(DST)
    print("paragraphs:", len(check.paragraphs), "images:", len(check.inline_shapes),
          "tables:", len(check.tables))


if __name__ == "__main__":
    main()

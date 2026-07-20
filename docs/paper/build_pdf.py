"""Render docs/paper/paper.md to a styled academic PDF (no LaTeX needed).

Pipeline: markdown -> HTML (+ academic CSS) -> PDF via xhtml2pdf (pure Python).
Figure image lines (``![caption](src)``) become centered figures with a visible
caption. Run with the throwaway PDF venv:

    ~/.venv_pdf/bin/python docs/paper/build_pdf.py
"""
from __future__ import annotations

import os
import re

import markdown
from xhtml2pdf import pisa

HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(HERE, "paper.md")
OUT = os.path.join(HERE, "paper.pdf")

CSS = """
@page {
    size: a4 portrait;
    margin: 2.0cm 1.9cm 2.2cm 1.9cm;
    @frame footer { -pdf-frame-content: footerContent; bottom: 1.0cm;
                    margin-left: 1.9cm; margin-right: 1.9cm; height: 1cm; }
}
body { font-family: "Times New Roman", serif; font-size: 10.5pt;
       line-height: 1.42; text-align: justify; color: #111; }
h1 { font-size: 17pt; text-align: center; margin: 0 0 2pt 0; line-height: 1.25; }
h2 { font-size: 13pt; border-bottom: 0.6pt solid #999; padding-bottom: 2pt;
     margin: 16pt 0 6pt 0; }
h3 { font-size: 11.5pt; margin: 11pt 0 4pt 0; }
p { margin: 0 0 6pt 0; }
a { color: #14387f; text-decoration: none; }
code { font-family: "Courier New", monospace; font-size: 9pt; background: #f2f2f2; }
pre { font-family: "Courier New", monospace; font-size: 8.5pt; background: #f5f5f5;
      border: 0.5pt solid #ddd; padding: 5pt; }
table { -pdf-keep-with-next: true; border: 0.5pt solid #999; border-collapse: collapse;
        margin: 6pt 0; width: 100%; font-size: 9.5pt; }
th { background: #e9eef7; border: 0.5pt solid #999; padding: 3pt 5pt; text-align: left; }
td { border: 0.5pt solid #999; padding: 3pt 5pt; }
.subtitle { text-align: center; font-size: 10pt; color: #444; margin-bottom: 10pt; }
.figure { -pdf-keep-with-next: true; text-align: center; margin: 9pt 0; }
.figure img { max-width: 80%; }
.caption { font-size: 8.8pt; color: #333; text-align: center; margin-top: 3pt; }
hr { border: none; border-top: 0.5pt solid #ccc; margin: 8pt 0; }
"""

FOOTER = ('<div id="footerContent" style="text-align:center;font-size:8pt;color:#777;">'
          'jax_marl3 &mdash; final-project manuscript &mdash; '
          'page <pdf:pagenumber> of <pdf:pagecount></div>')


def figurize(md_text: str) -> str:
    """Turn `![caption](src)` lines into figure divs with a visible caption."""
    def repl(m):
        cap, src = m.group(1), m.group(2)
        return (f'\n<div class="figure"><img src="{src}" />'
                f'<div class="caption">{cap}</div></div>\n')
    return re.sub(r'^!\[(.*?)\]\((.*?)\)\s*$', repl, md_text, flags=re.M)


def link_callback(uri, rel):
    if uri.startswith("figs/") or uri.startswith("./figs/"):
        return os.path.join(HERE, uri.lstrip("./"))
    p = os.path.join(HERE, uri)
    return p if os.path.exists(p) else uri


def main():
    src = open(MD, encoding="utf-8").read()
    src = re.sub(r"<!--.*?-->", "", src, flags=re.S)   # drop marker comments
    src = figurize(src)
    body = markdown.markdown(
        src, extensions=["tables", "fenced_code", "sane_lists", "attr_list"])
    html = (f"<html><head><meta charset='utf-8'><style>{CSS}</style></head>"
            f"<body>{FOOTER}{body}</body></html>")
    with open(OUT, "wb") as f:
        result = pisa.CreatePDF(html, dest=f, link_callback=link_callback,
                                encoding="utf-8")
    if result.err:
        raise SystemExit(f"PDF generation had {result.err} error(s)")
    print("wrote", OUT, f"({os.path.getsize(OUT)//1024} KB)")


if __name__ == "__main__":
    main()

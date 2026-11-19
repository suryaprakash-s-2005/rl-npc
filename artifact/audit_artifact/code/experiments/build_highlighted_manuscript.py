"""Build a print-highlighted copy of the revised manuscript.

Yellow marks new or substantively revised material in the reconstructed source.
Run from the project root:
    python -m experiments.build_highlighted_manuscript
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "paper", "main.html")
OUT = os.path.join(ROOT, "paper", "highlighted.html")

STYLE = """
<style id="revision-highlight">
  .rev { background: #fff3a8; }
  @media print {
    .rev {
      background: #fff3a8 !important;
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
    }
  }
</style>
"""

LEGEND = """<div style="font-size:7.2pt;line-height:1.1;text-align:center;margin:0 0 5pt;">
<b>Revision highlights:</b> yellow = new or substantively revised material. The submitted source was not editable; highlighting is semantic, not a character-level diff.
</div>"""


def main():
    with open(SRC, encoding="utf-8") as f:
        html = f.read()
    html = html.replace("</head>", STYLE + "</head>", 1)
    html = html.replace("<div class=\"abstract rev\">", LEGEND + "<div class=\"abstract rev\">", 1)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(OUT)


if __name__ == "__main__":
    main()

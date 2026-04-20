"""Convert MODELING_PIPELINE_RESEARCH_DRAFT.md to PDF via HTML + headless Chrome."""
from pathlib import Path
import subprocess
import sys

import markdown

HERE = Path(__file__).resolve().parent
MD = HERE / "MODELING_PIPELINE_RESEARCH_DRAFT.md"
HTML = HERE / "MODELING_PIPELINE_RESEARCH_DRAFT.html"
PDF = HERE / "MODELING_PIPELINE_RESEARCH_DRAFT.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = r"""
@page { size: Letter; margin: 0.85in 0.9in; }
html { font-size: 10.5pt; }
body {
  font-family: "Charter", "Georgia", "Times New Roman", serif;
  color: #111;
  line-height: 1.45;
  max-width: 100%;
}
h1, h2, h3, h4 {
  font-family: "Helvetica Neue", "Arial", sans-serif;
  color: #0b2545;
  line-height: 1.2;
  page-break-after: avoid;
}
h1 { font-size: 1.9rem; border-bottom: 2px solid #0b2545; padding-bottom: .25rem; margin-top: 0; }
h2 { font-size: 1.35rem; border-bottom: 1px solid #c8d3e0; padding-bottom: .15rem; margin-top: 1.6rem; }
h3 { font-size: 1.1rem; margin-top: 1.2rem; }
h4 { font-size: 1.0rem; margin-top: 1.0rem; }
p { margin: 0.5rem 0; text-align: justify; }
code {
  font-family: "Menlo", "Monaco", "Courier New", monospace;
  font-size: 0.88em;
  background: #f4f4f6;
  padding: 1px 4px;
  border-radius: 3px;
  color: #b03a2e;
}
pre {
  background: #f7f8fa;
  border: 1px solid #dde2e8;
  border-radius: 4px;
  padding: 10px 12px;
  overflow-x: auto;
  font-size: 0.82em;
  line-height: 1.35;
  page-break-inside: avoid;
}
pre code { background: none; color: #111; padding: 0; }
table {
  border-collapse: collapse;
  margin: 0.6rem 0;
  font-size: 0.88em;
  page-break-inside: avoid;
}
th, td {
  border: 1px solid #c8d3e0;
  padding: 5px 8px;
  text-align: left;
  vertical-align: top;
}
th { background: #eef2f7; font-weight: 600; }
tr:nth-child(even) td { background: #fafbfc; }
blockquote {
  border-left: 3px solid #0b2545;
  margin: 0.8rem 0;
  padding: 0.1rem 0.9rem;
  color: #333;
  background: #f6f8fb;
  page-break-inside: avoid;
}
ul, ol { margin: 0.4rem 0 0.4rem 1.3rem; }
li { margin: 0.15rem 0; }
hr { border: 0; border-top: 1px solid #c8d3e0; margin: 1.2rem 0; }
a { color: #1a4a8a; text-decoration: none; }
.math { font-family: "Latin Modern Math", "Cambria Math", serif; }
.footer-note { color: #666; font-size: 0.85em; }
"""

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>EEG Seizure Detection — Modeling Pipeline Research Draft</title>
<style>{css}</style>
</head>
<body>
{body}
</body>
</html>
"""


def main() -> int:
    md_text = MD.read_text(encoding="utf-8")
    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc", "sane_lists", "attr_list"],
        output_format="html5",
    )
    HTML.write_text(HTML_TEMPLATE.format(css=CSS, body=body), encoding="utf-8")
    print(f"[ok] wrote {HTML.name} ({HTML.stat().st_size:,} bytes)")

    cmd = [
        CHROME,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF}",
        HTML.as_uri(),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    if result.returncode != 0 or not PDF.exists():
        print("[err] Chrome exited non-zero", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return 1
    print(f"[ok] wrote {PDF.name} ({PDF.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

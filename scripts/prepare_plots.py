"""Shrink the raster plots in assets/charts/ and size their <img> tags.

Run after copying a new or updated plot (e.g. matplotlib output) into
assets/charts/:

    python3 scripts/prepare_plots.py

1. Every PNG that isn't already palette-based is quantized to 256 colours.
   Plots use few colours, so this cuts them to about a third of the size
   with no visible change.
2. Every Markdown image that points at one of these PNGs gets width and
   height (so the page doesn't jump while it loads) and lazy loading, in
   its kramdown attribute list: ![alt](/assets/charts/x.png){: .chart ...}
3. Wide plots (.chart-wide) are wrapped in a link to the full-size image,
   since multi-panel plots are hard to read at phone width.

Needs Pillow. Safe to run repeatedly.
"""
import re
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CHARTS = ROOT / "assets" / "charts"

sizes = {}
for png in sorted(CHARTS.glob("*.png")):
    im = Image.open(png)
    if im.mode != "P":
        before = png.stat().st_size
        im.convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(png, optimize=True)
        print(f"quantized {png.name}: {before // 1024} KB -> {png.stat().st_size // 1024} KB")
    sizes[png.name] = im.size

IMAGE = re.compile(r"(!\[[^\]]*\]\(/assets/charts/([\w.-]+\.png)\))\{:([^}]*)\}")
UNLINKED_WIDE = re.compile(r"(?<!\[)(!\[[^\]]*\]\((/assets/charts/[\w.-]+\.png)\)\{:[^}]*chart-wide[^}]*\})")


def size_attrs(match):
    name = match.group(2)
    if name not in sizes:
        return match.group(0)
    attrs = re.sub(r'\s*(width|height|loading|decoding)="[^"]*"', "", match.group(3)).rstrip()
    w, h = sizes[name]
    return f'{match.group(1)}{{:{attrs} width="{w}" height="{h}" loading="lazy" decoding="async"}}'


for md in sorted(ROOT.glob("**/*.md")):
    if any(part in {"_site", "node_modules", "vendor"} for part in md.parts):
        continue
    text = md.read_text()
    new = IMAGE.sub(size_attrs, text)
    new = UNLINKED_WIDE.sub(r"[\1](\2)", new)
    if new != text:
        md.write_text(new)
        print(f"sized images in {md.relative_to(ROOT)}")

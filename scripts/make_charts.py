"""Generate the SVG result charts in assets/charts/ for the project write-ups.

The numbers are the approximate results reported in the tables of
projects/vmas-blindness.md and projects/marl-elimination.md. Edit them here
and rerun `python3 scripts/make_charts.py` if those results change.
"""

from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent.parent / "assets" / "charts"

CHARTS = {
    "vmas-tuning": {
        "title": "Final reward after hyperparameter tuning",
        "bars": [
            ("Original settings", 280),
            ("Best combination", 520),
            ("Best, but clip 0.2", 480),
            ("Best, but batch 200", 490),
            ("Best, but 20 epochs", 470),
        ],
        "highlight": [1],
    },
    "marl-elimination": {
        "title": "Cumulative reward at 500K steps by agents needed to eliminate",
        "bars": [
            ("No elimination", 12000),
            ("1 agent", 11500),
            ("2 agents", 10000),
            ("3 agents", 9500),
            ("4 agents", 9000),
        ],
        "highlight": [0],
    },
    "marl-algorithms": {
        "title": "Cumulative reward by algorithm (3 agents to eliminate)",
        "bars": [
            ("SAC", 11800),
            ("PPO", 10500),
            ("POCA", 10200),
        ],
        "highlight": [0],
    },
}

# Labels sit above their bars so the chart stays narrow and readable on phones.
# Each chart draws its own background so it stays readable even when the
# browser picks a different colour scheme for the image than for the page.
WIDTH = 520
LABEL_H = 18
BAR_H = 20
GAP = 14
TOP = 40
VALUE_W = 64
PAD = 16  # space between the chart's own background and its contents

STYLE = """
  .title { font: 600 16px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; fill: #111; }
  .label, .value { font: 14px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; fill: #333; }
  .bg { fill: #fff; }
  .bar { fill: #8fb3e3; }
  .bar.hi { fill: #1f5fae; }
  @media (prefers-color-scheme: dark) {
    .bg { fill: #1b1e23; }
    .title { fill: #e3e5e8; }
    .label, .value { fill: #c9ccd1; }
    .bar { fill: #3d5f8f; }
    .bar.hi { fill: #7cb0f5; }
  }
"""


def fmt(v):
    return f"~{v:,}"


def chart(title, bars, highlight):
    row = LABEL_H + BAR_H + GAP
    height = TOP + len(bars) * row
    max_v = max(v for _, v in bars)
    plot_w = WIDTH - VALUE_W
    desc = "; ".join(f"{label}: {fmt(v)}" for label, v in bars)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH + 2 * PAD} {height + 2 * PAD}" '
        f'role="img" aria-labelledby="t d">',
        f"<title id=\"t\">{escape(title)}</title>",
        f"<desc id=\"d\">{escape(desc)}</desc>",
        f"<style>{STYLE}</style>",
        f'<rect class="bg" x="0" y="0" width="{WIDTH + 2 * PAD}" height="{height + 2 * PAD}" rx="8"/>',
        f'<g transform="translate({PAD} {PAD})">',
        f'<text class="title" x="0" y="20">{escape(title)}</text>',
    ]
    for i, (label, v) in enumerate(bars):
        y = TOP + i * row
        bar_y = y + LABEL_H
        w = max(2, round(plot_w * v / max_v))
        cls = "bar hi" if i in highlight else "bar"
        parts += [
            f'<text class="label" x="0" y="{y + 13}">{escape(label)}</text>',
            f'<rect class="{cls}" x="0" y="{bar_y}" width="{w}" height="{BAR_H}" rx="3"/>',
            f'<text class="value" x="{w + 8}" y="{bar_y + 15}">{fmt(v)}</text>',
        ]
    parts += ["</g>", "</svg>"]
    return "\n".join(parts) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, spec in CHARTS.items():
        (OUT / f"{name}.svg").write_text(chart(spec["title"], spec["bars"], spec["highlight"]))
        print(f"wrote assets/charts/{name}.svg")


if __name__ == "__main__":
    main()

"""
methodology_figure.py -- plan deliverable 2.10, the methodology flowchart.

Draws the pipeline as it is actually implemented, with one thing made
deliberately prominent: the split happens BEFORE any transform is fitted,
and it happens twice, once per protocol. That ordering is the whole
methodological claim of the project, so the figure is arranged to show it
rather than to list the modules.

Run:
    python src/methodology_figure.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures" / "methodology_flowchart.png"

# Palette slots 1 and 2 from the documented reference palette, plus ink.
# Blue = the shared pipeline. Orange = the two evaluation protocols, which
# is what the reader is meant to notice.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_SOFT, RULE = "#0b0b0b", "#52514e", "#b9b9b4"
FILL_BLUE, FILL_ORANGE, FILL_GREY = "#eaf2fc", "#fdeee7", "#f4f4f2"

# --- Geometry ----------------------------------------------------------
# Explicit row centres rather than arithmetic on a pitch, so the fork and
# the rejoin can have deliberately larger gaps than the straight runs.
H = 0.054                      # node height
PAD = 0.012                    # boxstyle pad, needed to find the true edge
EDGE = H / 2 + PAD             # centre -> drawn edge
W, WN = 0.40, 0.31             # full-width and branch-width nodes
LX, RX, CX = 0.265, 0.735, 0.5

Y_SRC, Y_C1, Y_C2 = 0.960, 0.862, 0.764
Y_C3 = 0.648
Y_PROTO = 0.522
Y_C4, Y_C5, Y_C6 = 0.396, 0.298, 0.200
Y_OUT = 0.080


def box(ax, x, y, w, title, sub=None, edge=BLUE, fill=FILL_BLUE):
    """One rounded node. `sub` is the small grey second line."""
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - H / 2), w, H,
        boxstyle=f"round,pad={PAD},rounding_size=0.018",
        linewidth=1.4, edgecolor=edge, facecolor=fill, zorder=3))
    ax.text(x, y + (0.011 if sub else 0), title, ha="center", va="center",
            fontsize=9.5, color=INK, zorder=4)
    if sub:
        ax.text(x, y - 0.017, sub, ha="center", va="center",
                fontsize=7.3, color=INK_SOFT, zorder=4)


def down(ax, x, y_from, y_to, color=RULE):
    """Vertical arrow between two node centres, trimmed to their edges."""
    ax.add_patch(FancyArrowPatch(
        (x, y_from - EDGE), (x, y_to + EDGE),
        arrowstyle="-|>", mutation_scale=11, linewidth=1.2,
        color=color, zorder=2))


def diag(ax, p0, p1, color=RULE):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=11, linewidth=1.2,
        color=color, zorder=2))


def main():
    fig, ax = plt.subplots(figsize=(9.2, 10.6), dpi=200)
    fig.patch.set_facecolor("#ffffff")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # --- shared ingestion stack ---------------------------------------
    box(ax, CX, Y_SRC, W, "CIC-IoT2023 CSV release",
        "309 files  ·  46,776,700 rows  ·  39 features",
        edge=RULE, fill=FILL_GREY)
    box(ax, CX, Y_C1, W, "C1   load.py",
        "Parquet conversion  ·  source_file capture key")
    box(ax, CX, Y_C2, W, "C2   sample.py",
        "class cap 200k  ·  4,458,051 rows")

    down(ax, CX, Y_SRC, Y_C1)
    down(ax, CX, Y_C1, Y_C2)

    # --- the split, and the point of the whole project -----------------
    box(ax, CX, Y_C3, W, "C3   split.py",
        "the split happens HERE, before anything is fitted",
        edge=ORANGE, fill=FILL_ORANGE)
    down(ax, CX, Y_C2, Y_C3, color=ORANGE)

    ax.text(0.995, Y_C3,
            "Rule 3.3 — every transform is\n"
            "fitted on the training partition\n"
            "only. Fitting before the split\n"
            "leaks the test set into the scaler.",
            ha="right", va="center", fontsize=7.4, color=INK_SOFT,
            linespacing=1.6)

    # Fork into the two protocols.
    diag(ax, (CX - 0.08, Y_C3 - EDGE), (LX, Y_PROTO + EDGE), color=ORANGE)
    diag(ax, (CX + 0.08, Y_C3 - EDGE), (RX, Y_PROTO + EDGE), color=ORANGE)

    box(ax, LX, Y_PROTO, WN, "Protocol A  (random)",
        "stratified random  ·  the control condition",
        edge=ORANGE, fill=FILL_ORANGE)
    box(ax, RX, Y_PROTO, WN, "Protocol B  (grouped)",
        "GroupShuffleSplit on source_file",
        edge=ORANGE, fill=FILL_ORANGE)

    # --- per-protocol preprocessing ------------------------------------
    for x in (LX, RX):
        down(ax, x, Y_PROTO, Y_C4)
        box(ax, x, Y_C4, WN, "C4   preprocess.py",
            "scaler fitted on train only")
        down(ax, x, Y_C4, Y_C5)
        box(ax, x, Y_C5, WN, "C5   features.py",
            "39 / 15 / 8 feature subsets")
        down(ax, x, Y_C5, Y_C6)
        box(ax, x, Y_C6, WN, "C6   train.py",
            "6 models  ×  3 seeds")

    # --- rejoin --------------------------------------------------------
    diag(ax, (LX, Y_C6 - EDGE), (CX - 0.08, Y_OUT + EDGE))
    diag(ax, (RX, Y_C6 - EDGE), (CX + 0.08, Y_OUT + EDGE))

    box(ax, CX, Y_OUT, W, "C7   bench.py   →   results.csv",
        "macro-recall  ·  macro-F1  ·  model size  ·  latency",
        edge=RULE, fill=FILL_GREY)

    ax.text(CX, 0.012,
            "RQ1 is the difference between the two branches.",
            ha="center", va="center", fontsize=9, color=INK)

    ax.set_title("Experimental methodology", fontsize=13, color=INK,
                 pad=2, loc="left", x=0.0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

"""
confusion_figures.py -- plan deliverable 3.5, confusion matrices as figures.

324 confusion matrices already exist as CSVs in `results/confusion/`. What
the report needs is not 324 figures but the two or three that carry the
argument, rendered legibly.

The headline pair is Protocol A against Protocol B for the same model, same
seed, same features -- because that single image is the RQ1 result. Anything
that changes between the two panels is attributable to the split and to
nothing else.

Seed 42 is used for the six-class figure. That choice is not arbitrary and
is not cherry-picking the flattering case: seed 42 is the ONLY seed whose
grouped split places all six classes in the test set, so it is the only one
where a six-by-six matrix can be drawn at all. Seed 44's grouped test set
contains three classes.

Outputs
-------
results/figures/confusion_xgboost_multiclass6.png
results/figures/confusion_random_forest_multiclass6.png

Run:
    python src/confusion_figures.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONF = ROOT / "results" / "confusion"
FIGS = ROOT / "results" / "figures"

TASK, FEATURES, SEED = "multiclass6", "full_39", 42
INK, INK_SOFT = "#0b0b0b", "#52514e"


def load(model: str, protocol: str) -> pd.DataFrame:
    path = CONF / f"{model}_{protocol}_{TASK}_{FEATURES}_seed{SEED}.csv"
    return pd.read_csv(path, index_col=0)


def panel(ax, cm: pd.DataFrame, title: str, subtitle: str,
          show_ylabel: bool = True):
    """One row-normalised confusion heatmap.

    Row-normalised, not raw counts: each row then sums to 1 and the diagonal
    IS per-class recall, which is the metric this project compares across
    protocols. Raw counts would be dominated by DDoS (421,739 test rows
    against Brute Force's 13,064) and show nothing.

    Colour is a single hue, light to dark -- the sequential rule. A
    diverging or multi-hue map would imply a midpoint that does not exist
    here; 0% and 100% are simply low and high.
    """
    counts = cm.to_numpy(dtype=float)
    totals = counts.sum(axis=1, keepdims=True)
    # A class with no test rows would divide by zero; keep it as NaN rather
    # than silently rendering it as 0% correct, which would be a lie.
    with np.errstate(invalid="ignore", divide="ignore"):
        norm = np.where(totals > 0, counts / totals, np.nan)

    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)

    labels = list(cm.index)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8,
                       color=INK)
    ax.set_yticklabels(labels, fontsize=8, color=INK)
    ax.set_xlabel("Predicted", fontsize=9, color=INK_SOFT, labelpad=6)
    if show_ylabel:
        ax.set_ylabel("Actual", fontsize=9, color=INK_SOFT, labelpad=6)
    # Title and subtitle as one two-line block in AXES coordinates. An
    # earlier version placed the subtitle in DATA coordinates, which put it
    # on top of the title.
    ax.set_title(title + "\n" + subtitle, fontsize=10.5, color=INK,
                 pad=12, loc="left", linespacing=1.9)

    # Cell labels. Text colour flips on dark cells so it stays readable --
    # the contrast requirement, not a style choice.
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            v = norm[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7.5,
                        color=INK_SOFT)
                continue
            ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center",
                    fontsize=8, color="#ffffff" if v > 0.55 else INK)

    # Thin white gaps between cells rather than gridlines on top of them.
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def make(model: str, pretty: str):
    a, b = load(model, "A"), load(model, "B")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.9), dpi=200)
    fig.patch.set_facecolor("#ffffff")

    diag_a = np.diag(a.to_numpy(dtype=float)) / a.to_numpy(dtype=float).sum(axis=1)
    diag_b = np.diag(b.to_numpy(dtype=float)) / b.to_numpy(dtype=float).sum(axis=1)

    panel(axes[0], a, "Protocol A — random split",
          f"macro-recall {np.nanmean(diag_a):.3f}  ·  the control condition")
    im = panel(axes[1], b, "Protocol B — grouped split",
               f"macro-recall {np.nanmean(diag_b):.3f}  ·  no capture file in both partitions",
               show_ylabel=False)

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Share of the true class, %", fontsize=8.5, color=INK_SOFT)
    cbar.ax.tick_params(labelsize=8, colors=INK_SOFT)
    cbar.outline.set_visible(False)

    fig.suptitle(f"{pretty} — six-class confusion, seed {SEED}",
                 fontsize=12.5, color=INK, x=0.065, ha="left", y=1.03)

    out = FIGS / f"confusion_{model}_{TASK}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    print(f"wrote {out}")
    return np.nanmean(diag_a), np.nanmean(diag_b)


def main():
    for model, pretty in [("xgboost", "XGBoost"),
                          ("random_forest", "Random Forest")]:
        ma, mb = make(model, pretty)
        print(f"  {pretty}: A {ma:.3f} -> B {mb:.3f}  ({mb - ma:+.3f})")


if __name__ == "__main__":
    main()

"""
dataset_stats.py -- plan deliverables 2.8, 2.9 (and the numbers 2.10 needs).

Produces the dataset-description artifacts the supervisor's plan asks for,
from the same Parquet files the experiments ran on. Nothing here is
transcribed by hand (CLAUDE.md rule 3.9) -- run the script, get the table.

Outputs
-------
results/tables/dataset_stats_overall.csv     one row: corpus-level totals
results/tables/dataset_stats_by_category.csv one row per attack category
results/tables/dataset_stats_by_subtype.csv  one row per raw 34-way label
results/figures/class_distribution.png       deliverable 2.9

Run:
    python src/dataset_stats.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from preprocess import CATEGORY_MAP, NON_FEATURE_COLS

# --- Paths -------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "data" / "processed" / "cic_iot2023.parquet"
SUB = ROOT / "data" / "processed" / "cic_iot2023_subsample.parquet"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"


def label_counts(path: Path) -> pd.DataFrame:
    """Row counts and distinct capture-file counts per raw label.

    Reads ONLY the `label` and `source_file` columns. The full corpus is
    46.8M rows and ~0.69 GB on disk; pulling two columns instead of all 41
    keeps this in a few hundred MB rather than filling RAM for no reason.
    """
    df = pq.read_table(path, columns=["label", "source_file"]).to_pandas()
    out = df.groupby("label").agg(
        rows=("source_file", "size"),
        # n_files matters because `source_file` is Protocol B's grouping key:
        # a class with one capture file CANNOT be split into train and test
        # without that whole class landing on one side. C3 flags this.
        n_files=("source_file", "nunique"),
    )
    return out.sort_values("rows", ascending=False)


def add_category(counts: pd.DataFrame) -> pd.DataFrame:
    """Attach the 7-category (+ Benign) taxonomy to per-subtype counts."""
    counts = counts.copy()
    # Benign has no entry in CATEGORY_MAP (it is not an attack), so any
    # label that doesn't map is kept under its own name. If a genuinely
    # unexpected label ever appeared, it would show up here rather than
    # being silently folded into another category.
    counts["category"] = counts.index.map(lambda s: CATEGORY_MAP.get(s, s))
    return counts


def nonfinite_rows(path: Path) -> tuple[int, int]:
    """Count rows carrying any inf/NaN in a feature column, and the columns.

    These are the single-packet capture windows found during C4: with one
    packet in the window, Std and Variance are undefined and Rate divides
    by a zero duration. Reported because plan item 2.2 asks for it.
    """
    df = pd.read_parquet(path)
    feats = [c for c in df.columns if c not in NON_FEATURE_COLS]
    bad_mask = ~np.isfinite(df[feats].to_numpy(dtype="float64")).all(axis=1)
    n_bad_cols = int(
        (~np.isfinite(df[feats].to_numpy(dtype="float64"))).any(axis=0).sum()
    )
    return int(bad_mask.sum()), n_bad_cols


def duplicate_stats(path: Path) -> dict:
    """Duplicate-feature-vector analysis (plan item 2.3), recomputed here.

    The headline number is the share of rows whose 39-number feature vector
    is identical to another row's but whose label differs -- those rows are
    unlearnable in principle and set a ceiling on achievable accuracy.
    """
    df = pd.read_parquet(path)
    feats = [c for c in df.columns if c not in NON_FEATURE_COLS]

    # Grouping by 39 separate float columns is pathologically slow (it was
    # still running after 10 minutes). Instead, view each row's 39 float32
    # values as ONE opaque fixed-width record and group on that. This is an
    # exact byte-level comparison, not a hash -- no collision risk -- and
    # np.unique then does the whole job in one sort.
    arr = np.ascontiguousarray(df[feats].to_numpy(dtype="float32"))
    row_view = arr.view([("", arr.dtype)] * arr.shape[1]).ravel()
    _, group_id, counts = np.unique(
        row_view, return_inverse=True, return_counts=True)

    n_distinct = int(len(counts))
    dup_mask = counts[group_id] > 1          # rows sharing a vector
    n_dup_rows = int(dup_mask.sum())

    # Among duplicated vectors, how many groups disagree about the label?
    # Grouping on the integer group id is fast where grouping on 39 floats
    # was not.
    cats = df["label"].map(lambda s: CATEGORY_MAP.get(s, s)).to_numpy()
    per_group = pd.Series(cats).groupby(group_id).nunique()
    conflicting = per_group.index[per_group.to_numpy() > 1]

    # Rows sitting inside a label-conflicting group.
    n_conflict_rows = int(np.isin(group_id, conflicting).sum())

    # --- Bayes error ----------------------------------------------------
    # NOT the same as "rows in conflicting groups". Inside a group of
    # identical feature vectors, the best possible classifier predicts that
    # group's MAJORITY label and is right for every majority row -- only
    # the minority rows are necessarily wrong. So the irreducible error is
    #     sum over groups of (group size - majority count)
    # Counting every row in a conflicting group as an error overstates the
    # floor badly (12.36% instead of 3.80% on this subsample).
    sizes = pd.DataFrame({"g": group_id, "c": cats}).groupby(
        ["g", "c"], observed=True).size()
    totals = sizes.groupby(level=0).sum()
    majority = sizes.groupby(level=0).max()
    n_bayes_errors = int((totals - majority).sum())

    return {
        "n_bayes_error_rows": n_bayes_errors,
        "pct_bayes_error": round(100 * n_bayes_errors / len(df), 2),
        "n_rows": len(df),
        "n_distinct_vectors": n_distinct,
        "n_duplicated_rows": n_dup_rows,
        "pct_duplicated_rows": round(100 * n_dup_rows / len(df), 2),
        "n_conflicting_groups": int(len(conflicting)),
        "n_rows_in_conflicting_groups": n_conflict_rows,
        "pct_rows_in_conflicting_groups": round(
            100 * n_conflict_rows / len(df), 2),
    }


def make_figure(by_cat: pd.DataFrame, out: Path) -> None:
    """Deliverable 2.9 -- the class-distribution graph.

    Design notes (see the dataviz method):
      * Horizontal bars, sorted by magnitude -- the job is comparing
        magnitudes across named categories, and the names are long.
      * LOG x-axis. DDoS is ~2,500x Brute Force; on a linear axis every
        minority class is an invisible sliver. Because a log axis breaks
        the "bar length is proportional to value" contract, EVERY bar is
        directly labelled with its exact count -- the label carries the
        magnitude, the bar carries the order of magnitude. The caption
        must state the axis is logarithmic.
      * Two series (full corpus, capped subsample) using slots 1 and 2 of
        the documented categorical palette. Legend present, as required
        for >= 2 series.
    """
    import matplotlib
    matplotlib.use("Agg")  # no display on this machine; write straight to file
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator

    # Palette slots 1 and 2, light mode. Taken unchanged from the
    # documented reference palette rather than invented, so the
    # adjacent-pair CVD separation is already established.
    C_FULL, C_SUB = "#2a78d6", "#eb6834"
    INK, INK_SOFT = "#0b0b0b", "#52514e"

    d = by_cat.sort_values("rows_full")
    y = np.arange(len(d))
    h = 0.38

    fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=200)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    ax.barh(y + h / 2, d["rows_full"], height=h, color=C_FULL,
            label="Full corpus (46.8M rows)", zorder=3)
    ax.barh(y - h / 2, d["rows_sub"], height=h, color=C_SUB,
            label="Capped subsample (4.46M rows)", zorder=3)

    ax.set_xscale("log")
    ax.set_xlim(1e3, 2e8)  # headroom on the right for the value labels
    ax.set_yticks(y)
    ax.set_yticklabels(d.index, fontsize=10, color=INK)

    # Recessive grid: x only, thin, solid (never dashed), behind the bars.
    ax.xaxis.grid(True, which="major", color="#e4e4e1", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(LogLocator(base=10))
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d6d6d2")
    ax.tick_params(axis="x", colors=INK_SOFT, labelsize=9)
    ax.tick_params(axis="y", length=0)

    # Direct value labels on every bar -- mandatory here because the log
    # axis means bar length alone does not communicate the magnitude.
    for yi, v in zip(y + h / 2, d["rows_full"]):
        ax.text(v * 1.25, yi, f"{v:,}", va="center", fontsize=8, color=INK_SOFT)
    for yi, v in zip(y - h / 2, d["rows_sub"]):
        ax.text(v * 1.25, yi, f"{v:,}", va="center", fontsize=8, color=INK_SOFT)

    ax.set_xlabel("Records (logarithmic scale)", fontsize=10, color=INK_SOFT,
                  labelpad=8)
    ax.set_title("CIC-IoT2023 class distribution before and after class capping",
                 fontsize=12, color=INK, pad=14, loc="left")
    leg = ax.legend(frameon=False, fontsize=9, loc="lower right")
    for t in leg.get_texts():
        t.set_color(INK_SOFT)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)

    print("counting full corpus ...")
    full = add_category(label_counts(FULL))
    print(f"  {full['rows'].sum():,} rows, {len(full)} labels")

    print("counting subsample ...")
    sub = add_category(label_counts(SUB))
    print(f"  {sub['rows'].sum():,} rows, {len(sub)} labels")

    # --- per-subtype table (34 rows) -----------------------------------
    by_sub = full.join(
        sub[["rows", "n_files"]].rename(
            columns={"rows": "rows_sub", "n_files": "n_files_sub"}),
        how="left",
    ).rename(columns={"rows": "rows_full", "n_files": "n_files_full"})
    by_sub["pct_of_corpus"] = (
        100 * by_sub["rows_full"] / by_sub["rows_full"].sum()).round(4)
    by_sub["capped"] = by_sub["rows_full"] > by_sub["rows_sub"]
    by_sub = by_sub[["category", "rows_full", "pct_of_corpus", "rows_sub",
                     "n_files_full", "capped"]]
    by_sub.to_csv(TABLES / "dataset_stats_by_subtype.csv")
    print(f"  wrote by_subtype ({len(by_sub)} rows)")

    # --- per-category table (8 rows) -----------------------------------
    by_cat = by_sub.groupby("category").agg(
        n_subtypes=("rows_full", "size"),
        rows_full=("rows_full", "sum"),
        rows_sub=("rows_sub", "sum"),
        n_files_full=("n_files_full", "sum"),
    )
    by_cat["pct_corpus"] = (
        100 * by_cat["rows_full"] / by_cat["rows_full"].sum()).round(3)
    by_cat["pct_subsample"] = (
        100 * by_cat["rows_sub"] / by_cat["rows_sub"].sum()).round(3)
    # Imbalance ratio against the smallest class -- the single number that
    # justifies class weighting (rule 3.7) and macro metrics (rule 3.6).
    by_cat["imbalance_vs_smallest"] = (
        by_cat["rows_full"] / by_cat["rows_full"].min()).round(1)
    by_cat = by_cat.sort_values("rows_full", ascending=False)
    by_cat.to_csv(TABLES / "dataset_stats_by_category.csv")
    print(f"  wrote by_category ({len(by_cat)} rows)")

    # --- corpus-level row ----------------------------------------------
    print("scanning subsample for non-finite values and duplicates ...")
    n_bad, n_bad_cols = nonfinite_rows(SUB)
    dup = duplicate_stats(SUB)

    schema = pq.read_schema(FULL)
    n_feature_cols = len([n for n in schema.names if n not in NON_FEATURE_COLS])

    overall = {
        "corpus_rows": int(by_sub["rows_full"].sum()),
        "corpus_capture_files": int(by_sub["n_files_full"].sum()),
        "feature_columns": n_feature_cols,
        "independent_feature_signals": 36,  # see DECISIONS.md, C5
        "raw_labels": int(len(by_sub)),
        "categories": int(len(by_cat)),
        "subsample_rows": int(by_sub["rows_sub"].sum()),
        "subsample_cap_per_class": 200_000,
        "subsample_pct_of_corpus": round(
            100 * by_sub["rows_sub"].sum() / by_sub["rows_full"].sum(), 3),
        "imbalance_ratio_corpus": float(by_cat["imbalance_vs_smallest"].max()),
        "nonfinite_rows_subsample": n_bad,
        "nonfinite_columns_affected": n_bad_cols,
        **{f"dup_{k}": v for k, v in dup.items()},
        # The Bayes-error ceiling: the best accuracy any classifier could
        # reach on this data, however good the model.
        "max_achievable_accuracy_pct": round(100 - dup["pct_bayes_error"], 2),
    }
    pd.DataFrame([overall]).T.rename(columns={0: "value"}).to_csv(
        TABLES / "dataset_stats_overall.csv")
    print(f"  wrote overall ({len(overall)} statistics)")

    make_figure(by_cat, FIGURES / "class_distribution.png")

    print("\n--- by category ---")
    print(by_cat.to_string())
    print("\n--- overall ---")
    for k, v in overall.items():
        print(f"  {k:38s} {v}")


if __name__ == "__main__":
    main()

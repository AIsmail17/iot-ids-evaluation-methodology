"""
analyse.py — turns results/results.csv into the report's tables.

Not one of the C1-C7 components, but required by rule 3.9: "No result is
ever transcribed by hand." Every table in the report is generated here from
results.csv and written to results/tables/, so a number in the report can
always be traced back to the run that produced it.

Produces:
  tables/aggregated.csv   mean +/- std over seeds for every cell
  tables/rq1_protocol.csv Protocol A vs B, the headline comparison
  tables/rq2_footprint.csv macro-F1 against feature count and artifact size
  tables/coverage.csv     seed counts and absent-class warnings

Reading a file the sweep may still be appending to is handled: a torn final
line is dropped rather than crashing the analysis.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_CSV = Path("results/results.csv")
TABLES_DIR = Path("results/tables")
PER_CLASS_DIR = Path("results/per_class")
BENCH_CSV = Path("results/bench.csv")

# Metrics aggregated across seeds. accuracy is included ONLY so the report
# can show the accuracy-vs-macro-F1 gap; rule 3.6 forbids it as a headline.
METRICS = ["macro_f1", "macro_recall", "macro_f1_trained_only",
           "pr_auc_macro", "fpr_benign", "accuracy"]

CELL = ["model", "protocol", "task", "feature_set"]


def load_results() -> pd.DataFrame:
    """Read results.csv, refusing to silently drop rows.

    An earlier version fell back to `on_bad_lines="skip"` on any
    ParserError. That is dangerous here: when the code gains a column, every
    NEW row has more fields than the header, so "skip the bad lines" quietly
    discards exactly the newest results while reporting success. A results
    table that silently loses runs is worse than one that fails to load.

    So a ragged file is repaired, not skipped -- rows with extra fields are
    read positionally against the widest schema found.
    """
    try:
        df = pd.read_csv(RESULTS_CSV)
    except pd.errors.ParserError as exc:
        print(f"results.csv is ragged ({exc}); repairing rather than skipping rows")
        df = _read_ragged(RESULTS_CSV)
    return df.drop_duplicates("run_id", keep="last")


# Columns the schema has gained over time, in the order train.py writes
# them. Used only to name the extra fields in rows written after a change.
LATE_COLUMNS = ["features_used"]


def _read_ragged(path: Path) -> pd.DataFrame:
    """Read a CSV whose later rows carry more fields than its header."""
    import csv

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    widest = max(len(r) for r in body) if body else len(header)

    if widest > len(header):
        extra = LATE_COLUMNS[: widest - len(header)]
        if len(extra) < widest - len(header):          # unknown new columns
            extra += [f"unknown_col_{i}" for i in range(len(extra), widest - len(header))]
        header = header + extra
        print(f"  widened header by {len(extra)} column(s): {extra}")

    padded = [r + [""] * (len(header) - len(r)) for r in body]
    df = pd.DataFrame(padded, columns=header)
    # csv.reader gives strings; restore numeric dtypes where possible.
    for c in df.columns:
        converted = pd.to_numeric(df[c], errors="coerce")
        if converted.notna().sum() >= df[c].replace("", pd.NA).notna().sum():
            df[c] = converted
    print(f"  recovered {len(df)} rows (none dropped)")
    return df


def add_macro_recall(df: pd.DataFrame) -> pd.DataFrame:
    """Attach macro-recall (balanced accuracy) per run.

    WHY THIS EXISTS. macro-F1 is NOT comparable across protocols here.
    Protocol A's test set is a stratified 20% sample, so benign is 4.49% of
    it. Protocol B assigns whole capture files to a side, so benign is 6.86%
    -- 1.5x as much. Precision, and therefore F1, moves with class
    prevalence, so part of any A-vs-B macro-F1 difference is just the two
    test sets having different compositions.

    Recall is invariant to class prevalence: it only asks "of the rows that
    really are class X, how many did we find". Macro-recall is therefore the
    metric that can legitimately be compared ACROSS protocols. It is added
    here rather than in train.py so no run needs repeating -- per-class
    recall was already saved for every run.
    """
    recalls = {}
    for path in PER_CLASS_DIR.glob("*.csv"):
        pc = pd.read_csv(path)
        recalls[path.stem] = pc["recall"].mean()
    df = df.copy()
    df["macro_recall"] = df["run_id"].map(recalls)
    missing = int(df["macro_recall"].isna().sum())
    if missing:
        print(f"note: {missing} run(s) had no per-class file; macro_recall is NA there")
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std across seeds for each cell (rule 3.8).

    `n_seeds` is carried through deliberately: a mean over 1 or 2 seeds is
    not the same claim as a mean over 3, and the report must show which is
    which rather than presenting them identically.
    """
    grouped = df.groupby(CELL, dropna=False)
    out = grouped[METRICS].agg(["mean", "std"]).round(4)
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out["n_seeds"] = grouped.size()
    out["model_mb"] = (grouped["model_bytes"].mean() / 1e6).round(3)
    out["train_seconds_mean"] = grouped["train_seconds"].mean().round(1)
    # Formatted for direct pasting into the report, so nobody retypes a number.
    out["macro_f1_report"] = [
        f"{m:.3f} ± {0.0 if pd.isna(s) else s:.3f}" + ("" if n >= 3 else f" (n={n})")
        for m, s, n in zip(out["macro_f1_mean"], out["macro_f1_std"], out["n_seeds"])
    ]
    return out.reset_index()


def rq1_table(agg: pd.DataFrame) -> pd.DataFrame:
    """Protocol A vs Protocol B -- the project's headline comparison.

    delta = B - A. The brief's recorded prior (section 8) is that every
    model loses MORE than 5 points moving A -> B, i.e. delta < -0.05.
    `prior_holds` records that test explicitly per cell rather than leaving
    it to be eyeballed.
    """
    idx = ["model", "task", "feature_set"]
    wide = agg.pivot_table(index=idx, columns="protocol",
                           values=["macro_f1_mean", "macro_f1_std",
                                   "macro_recall_mean", "n_seeds"])
    wide.columns = [f"{m}_{p}" for m, p in wide.columns]
    wide = wide.reset_index()

    if "macro_f1_mean_A" in wide and "macro_f1_mean_B" in wide:
        wide["delta_B_minus_A"] = (wide["macro_f1_mean_B"] - wide["macro_f1_mean_A"]).round(4)

        # The delta that is actually VALID across protocols. The macro-F1
        # delta above is confounded by the two protocols' test sets having
        # different class compositions (see add_macro_recall); this one is
        # not, because recall does not move with class prevalence. Read this
        # column, not the macro-F1 one, when comparing A against B.
        wide["delta_recall_B_minus_A"] = (
            wide["macro_recall_mean_B"] - wide["macro_recall_mean_A"]).round(4)

        # Must stay NA where one protocol has not run. A plain
        # `delta < -0.05` reads NaN as False, i.e. reports an untested cell
        # as evidence AGAINST the prior.
        wide["prior_holds"] = pd.array(
            np.where(wide["delta_B_minus_A"].notna(),
                     wide["delta_B_minus_A"] < -0.05, None),
            dtype="boolean",
        )
    return wide.sort_values(["task", "feature_set", "model"])


def rq2_table(agg: pd.DataFrame) -> pd.DataFrame:
    """Detection performance against footprint -- macro-F1 per megabyte.

    Section 8's recorded prior is that gradient-boosted trees with a reduced
    feature set give the best macro-F1 per kilobyte; this is the table that
    tests it.
    """
    cols = CELL + ["n_seeds", "macro_f1_mean", "macro_f1_std", "model_mb", "train_seconds_mean"]
    t = agg[cols].copy()
    # Guard against the majority baseline's ~0 MB artifact producing inf.
    t["macro_f1_per_mb"] = np.where(
        t["model_mb"] > 0.001, (t["macro_f1_mean"] / t["model_mb"]).round(2), np.nan
    )
    return t.sort_values(["task", "protocol", "macro_f1_mean"], ascending=[True, True, False])


def rq2_cost_table(agg: pd.DataFrame) -> pd.DataFrame:
    """Join C7's cost profile to detection performance -- the RQ2 answer.

    Uses macro-RECALL, not macro-F1, as the performance term: C7 profiles
    Protocol B artifacts, and recall is the metric that is comparable
    across protocols (see add_macro_recall). Efficiency is reported per MB
    of artifact.
    """
    if not BENCH_CSV.exists():
        print(f"note: {BENCH_CSV} not found; skipping the cost join")
        return pd.DataFrame()

    bench = pd.read_csv(BENCH_CSV)
    keys = ["model", "protocol", "task", "feature_set"]
    merged = bench.merge(agg[keys + ["macro_f1_mean", "macro_recall_mean", "n_seeds"]],
                         on=keys, how="left")

    merged["recall_per_mb"] = (merged["macro_recall_mean"] / merged["artifact_mb"]).round(2)
    merged["f1_per_mb"] = (merged["macro_f1_mean"] / merged["artifact_mb"]).round(2)
    cols = ["model", "task", "feature_set", "macro_recall_mean", "macro_f1_mean",
            "artifact_mb", "latency_p50_ms", "latency_p99_ms",
            "throughput_flows_per_s", "recall_per_mb", "f1_per_mb"]
    return merged[cols].sort_values(["task", "feature_set", "macro_recall_mean"],
                                    ascending=[True, True, False])


def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """Which cells are under-seeded, and where classes vanished from train.

    Logged as a requirement back in C3: under Protocol B a class can have
    zero training rows, and a mean over fewer than 3 seeds is a weaker claim
    than one over 3. Both must be visible in the report, not averaged away.
    """
    grouped = df.groupby(CELL, dropna=False)
    rows = []
    for cell, g in grouped:
        absent = sorted({
            c for entry in g["classes_absent_from_train"].dropna().astype(str)
            for c in entry.split("|") if c
        })
        rows.append({
            **dict(zip(CELL, cell)),
            "n_seeds": len(g),
            "under_seeded": len(g) < 3,
            "classes_absent_from_train": "|".join(absent),
            "max_classes_absent": int(g["n_classes_absent_from_train"].max()),
        })
    return pd.DataFrame(rows).sort_values(["task", "protocol", "model", "feature_set"])


def main():
    df = add_macro_recall(load_results())
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{len(df)} runs in {RESULTS_CSV}")

    agg = aggregate(df)
    agg.to_csv(TABLES_DIR / "aggregated.csv", index=False)

    rq1 = rq1_table(agg)
    rq1.to_csv(TABLES_DIR / "rq1_protocol.csv", index=False)

    rq2 = rq2_table(agg)
    rq2.to_csv(TABLES_DIR / "rq2_footprint.csv", index=False)

    cov = coverage_report(df)
    cov.to_csv(TABLES_DIR / "coverage.csv", index=False)

    cost = rq2_cost_table(agg)
    if len(cost):
        cost.to_csv(TABLES_DIR / "rq2_cost.csv", index=False)

    print(f"\nwrote 4 tables to {TABLES_DIR}/")

    print("\n" + "=" * 78)
    print("RQ1 - macro-F1 under Protocol A (random) vs Protocol B (grouped)")
    print("=" * 78)
    show = [c for c in ["model", "task", "feature_set", "macro_f1_mean_A",
                        "macro_f1_mean_B", "delta_B_minus_A",
                        "macro_recall_mean_A", "macro_recall_mean_B",
                        "delta_recall_B_minus_A", "prior_holds"] if c in rq1]
    print(rq1[show].to_string(index=False))

    if "prior_holds" in rq1:
        # Only cells where BOTH protocols have run are evidence either way.
        tested = int(rq1["prior_holds"].notna().sum())
        holds = int(rq1["prior_holds"].fillna(False).sum())
        untested = int(rq1["prior_holds"].isna().sum())
        print(f"\nRecorded prior ('every model loses >5 points A->B'): "
              f"holds in {holds}/{tested} testable cells; "
              f"{untested} cell(s) not yet testable (Protocol B still running).")

    print("\n" + "=" * 78)
    print("RQ2 - macro-F1 vs footprint (top rows by macro-F1 per MB)")
    print("=" * 78)
    print(rq2.dropna(subset=["macro_f1_per_mb"])
            .sort_values("macro_f1_per_mb", ascending=False)
            .head(12).to_string(index=False))

    under = cov[cov["under_seeded"]]
    if len(under):
        print(f"\n{len(under)} cell(s) still under-seeded (<3 seeds) - "
              "expected while the sweep is running.")


if __name__ == "__main__":
    main()

"""
duplicates_full.py -- duplicate analysis on the FULL corpus, not the subsample.

WHY THIS EXISTS
---------------
`dataset_stats.py` computes the duplicate rate on the 4.46M-row class-capped
subsample and gets 24.38 %. A published audit of CIC-IoT2023 reports 54.8 %
exact duplicates on the 39-feature representation. Those are probably not in
conflict: capping each subtype at 200,000 records strips out most of the
duplicate-heavy DDoS traffic, which is 72.65 % of the corpus. But the report
must state a corpus-level figure and a subsample-level figure with their
scopes attached, rather than implying one is the other.

This script computes the corpus-level figures so both can be reported.

Memory note: 46.78M rows x 39 float32 is ~7.3 GB, and np.unique sorts it, so
peak usage is roughly 20-25 GB. That fits in 64 GB but not much else should be
running.

Run:
    python src/duplicates_full.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from preprocess import CATEGORY_MAP, NON_FEATURE_COLS

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "data" / "processed" / "cic_iot2023.parquet"
OUT = ROOT / "results" / "tables" / "duplicates_full_corpus.csv"


def main():
    t0 = time.time()
    schema = pq.read_schema(FULL)
    feats = [n for n in schema.names if n not in NON_FEATURE_COLS]
    print(f"features: {len(feats)}", flush=True)

    print("reading feature columns ...", flush=True)
    arr = pq.read_table(FULL, columns=feats).to_pandas().to_numpy(dtype="float32")
    n_rows = len(arr)
    print(f"  {n_rows:,} rows x {arr.shape[1]} cols "
          f"({arr.nbytes / 1e9:.2f} GB)  [{time.time() - t0:.0f}s]", flush=True)

    print("reading labels ...", flush=True)
    labels = pq.read_table(FULL, columns=["label"]).to_pandas()["label"]
    # Category codes as int8 -- keeping 46.8M python strings would cost far
    # more memory than the feature array itself.
    cat_names = sorted(set(CATEGORY_MAP.values()) | {"Benign"})
    code = {n: i for i, n in enumerate(cat_names)}
    cats = labels.map(lambda s: code[CATEGORY_MAP.get(s, s)]).to_numpy(dtype="int8")
    del labels
    print(f"  {len(cat_names)} categories  [{time.time() - t0:.0f}s]", flush=True)

    # Byte-exact grouping: view each row's 39 float32 values as one opaque
    # fixed-width record, exactly as dataset_stats.py does on the subsample.
    print("grouping identical feature vectors (this is the slow part) ...",
          flush=True)
    arr = np.ascontiguousarray(arr)
    row_view = arr.view([("", arr.dtype)] * arr.shape[1]).ravel()
    _, group_id, counts = np.unique(row_view, return_inverse=True,
                                    return_counts=True)
    del arr, row_view
    print(f"  {len(counts):,} distinct vectors  [{time.time() - t0:.0f}s]",
          flush=True)

    n_distinct = int(len(counts))
    n_dup_rows = int((counts[group_id] > 1).sum())

    print("finding label-conflicting groups ...", flush=True)
    sizes = pd.DataFrame({"g": group_id, "c": cats}).groupby(
        ["g", "c"], observed=True).size()
    totals = sizes.groupby(level=0).sum()
    majority = sizes.groupby(level=0).max()
    nunique = sizes.groupby(level=0).size()

    conflicting = nunique.index[nunique.to_numpy() > 1]
    n_conflicting_groups = int(len(conflicting))
    n_rows_in_conflicting = int(totals.loc[conflicting].sum())
    # Bayes error: within a group of identical vectors the best possible
    # classifier predicts the MAJORITY label, so only the minority rows are
    # necessarily wrong.
    n_bayes = int((totals - majority).sum())

    result = {
        "scope": "full corpus",
        "n_rows": n_rows,
        "n_distinct_vectors": n_distinct,
        "n_duplicated_rows": n_dup_rows,
        "pct_duplicated_rows": round(100 * n_dup_rows / n_rows, 2),
        "n_conflicting_groups": n_conflicting_groups,
        "n_rows_in_conflicting_groups": n_rows_in_conflicting,
        "pct_rows_in_conflicting_groups": round(
            100 * n_rows_in_conflicting / n_rows, 2),
        "n_bayes_error_rows": n_bayes,
        "pct_bayes_error": round(100 * n_bayes / n_rows, 2),
        "max_achievable_accuracy_pct": round(100 - 100 * n_bayes / n_rows, 2),
        "wall_seconds": round(time.time() - t0, 1),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).T.rename(columns={0: "value"}).to_csv(OUT)

    print("\n--- full corpus ---", flush=True)
    for k, v in result.items():
        print(f"  {k:34s} {v}", flush=True)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

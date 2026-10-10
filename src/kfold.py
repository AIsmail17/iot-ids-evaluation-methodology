"""
kfold.py -- pooled k-fold evaluation, replacing the three-seed hold-out.

WHY THIS EXISTS
---------------
The hold-out comparison in results.csv has a confound that `report_tables.py`
exposed. Protocol B assigns whole capture files to a side, so which classes
land in the test partition depends on the seed: seed 42 tested six classes,
seed 43 five, seed 44 only three. Macro-recall averages over the classes
PRESENT, so those three numbers are different quantities and averaging them
is not meaningful. Comparing Protocol B's three-class average against
Protocol A's six-class average even made the grouped split look better.

K-fold removes the confound rather than averaging over it:

  * Every row is in the test partition exactly ONCE, across all folds.
  * Out-of-fold predictions are pooled, giving one prediction per row.
  * Macro-recall is then computed ONCE over all six classes, from a
    prediction set covering the entire dataset.

Both protocols are run this way, so the two are exactly comparable:
StratifiedKFold for A, GroupKFold on `source_file` for B. GroupKFold is
deterministic and takes no seed, which is the point -- there is no split
lottery left to average over.

WHAT THIS CANNOT FIX
--------------------
Brute Force occupies exactly ONE capture file. Under any grouped scheme that
file is in the test fold once and absent from training when it is, so its
recall is zero. That is a property of the dataset, not of the method, and it
is reported rather than worked around.

Run:
    python src/kfold.py                 # all models, both protocols
    python src/kfold.py xgboost lightgbm
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold

import preprocess
import train as T

ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "data" / "processed" / "cic_iot2023_subsample.parquet"
OUT_DIR = ROOT / "results" / "kfold"
OUT_CSV = ROOT / "results" / "tables" / "kfold_results.csv"

TASK = "multiclass6"
FEATURE_SET = "full_39"
N_SPLITS = 5
SEED = 42                      # used only by StratifiedKFold's shuffle
MODELS = ["majority", "logreg", "decision_tree", "random_forest",
          "lightgbm", "xgboost"]


def fold_indices(df, protocol):
    """Yield (fold_number, train_idx, test_idx) for the chosen protocol.

    Protocol A stratifies on the raw 34-way label, matching split.py's
    choice: a split proportional at the finest grain is proportional at any
    coarser grouping built from it.

    Protocol B groups on `source_file`. GroupKFold has no shuffle and no
    random_state -- it partitions the groups deterministically, which is
    exactly what removes the seed dependence.
    """
    idx = np.arange(len(df))
    if protocol == "A":
        splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                   random_state=SEED)
        pairs = splitter.split(idx, df["label"].to_numpy())
    elif protocol == "B":
        splitter = GroupKFold(n_splits=N_SPLITS)
        pairs = splitter.split(idx, groups=df["source_file"].to_numpy())
    else:
        raise ValueError(f"protocol must be 'A' or 'B', got {protocol!r}")

    for k, (tr, te) in enumerate(pairs, start=1):
        yield k, tr, te


def run_protocol(df, model_name, protocol):
    """Fit across all folds and pool the out-of-fold predictions.

    Returns a dict of pooled metrics plus the per-class recall table.
    """
    y_true_all, y_pred_all = [], []
    absent_by_fold = []
    t0 = time.time()

    for k, tr, te in fold_indices(df, protocol):
        X_train, X_test, y_train, y_test, class_names, _ = preprocess.preprocess(
            df, tr, te, task=TASK)
        n_classes = len(class_names)

        X_train, X_test, _ = T.select_features(
            X_train, y_train, X_test, FEATURE_SET, SEED)

        # Which classes this fold's training partition never saw. Under
        # Protocol B this is the interesting quantity; under A it is empty.
        seen = set(np.unique(y_train).tolist())
        absent = [class_names[i] for i in range(n_classes) if i not in seen]
        absent_by_fold.append(absent)

        model = T.build_model(model_name, SEED)
        model, classes = T.fit_model(model, model_name, X_train, y_train)
        proba = T.build_full_proba(model, model_name, X_test, classes, n_classes)
        y_pred = proba.argmax(axis=1)

        y_true_all.append(y_test)
        y_pred_all.append(y_pred)

        print(f"    fold {k}/{N_SPLITS}  train={len(y_train):>9,}  "
              f"test={len(y_test):>9,}  absent from train: "
              f"{absent if absent else 'none'}", flush=True)

        del model, X_train, X_test, proba

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    class_names = list(preprocess.TASK_CATEGORIES[TASK])

    # Per-class recall over the POOLED predictions -- every row predicted
    # exactly once, so this is a single number per class, not an average of
    # averages over differing class sets.
    rows = []
    for i, name in enumerate(class_names):
        mask = y_true == i
        support = int(mask.sum())
        recall = float((y_pred[mask] == i).mean()) if support else float("nan")
        rows.append({"class": name, "recall": recall, "support": support})
    per_class = pd.DataFrame(rows)

    macro_recall = float(per_class["recall"].mean())
    accuracy = float((y_pred == y_true).mean())

    return {
        "model": model_name,
        "protocol": protocol,
        "macro_recall": round(macro_recall, 4),
        "accuracy": round(accuracy, 4),
        "n_predicted": int(len(y_true)),
        "n_folds": N_SPLITS,
        "classes_ever_absent_from_train": "|".join(
            sorted({c for f in absent_by_fold for c in f})) or "",
        "wall_seconds": round(time.time() - t0, 1),
    }, per_class


def main():
    models = sys.argv[1:] or MODELS
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading {SUB.name} ...", flush=True)
    df = pd.read_parquet(SUB)
    print(f"  {len(df):,} rows, {df['source_file'].nunique()} capture files\n",
          flush=True)

    results = []
    for model_name in models:
        for protocol in ("A", "B"):
            print(f"{model_name}  protocol {protocol}", flush=True)
            summary, per_class = run_protocol(df, model_name, protocol)
            per_class.to_csv(
                OUT_DIR / f"{model_name}_{protocol}_{TASK}_kfold.csv",
                index=False)
            results.append(summary)
            print(f"    -> macro-recall {summary['macro_recall']:.4f}   "
                  f"accuracy {summary['accuracy']:.4f}   "
                  f"({summary['wall_seconds']:.0f} s)\n", flush=True)

            pd.DataFrame(results).to_csv(OUT_CSV, index=False)  # checkpoint

    out = pd.DataFrame(results)
    print("\n" + "=" * 62)
    print(out.to_string(index=False))

    piv = out.pivot_table(index="model", columns="protocol",
                          values="macro_recall")
    if {"A", "B"} <= set(piv.columns):
        piv["delta"] = (piv["B"] - piv["A"]).round(4)
        print("\nPooled k-fold macro-recall, A vs B:")
        print(piv.reindex([m for m in MODELS if m in piv.index]).to_string())
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()

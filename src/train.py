"""
train.py — Component C6: model training and evaluation.

One call to run_one() = one cell of the experiment matrix (section 9) =
one row appended to results/results.csv.

This pass covers the SUPERVISED roster (T0 + T1 in section 8). The T2
benign-only detectors (Isolation Forest, Autoencoder) train under a
different protocol entirely -- fit on benign, score anomalies -- and get
their own path rather than being forced through this one.

Three things here are subtle enough to be worth reading carefully:

  1. ABSENT CLASSES. Under Protocol B a whole class can have zero training
     rows (C3/C4 showed Brute Force does). A fitted model then knows fewer
     classes than exist, so predict_proba returns a narrower matrix, and
     XGBoost rejects non-contiguous label sets outright. Both silently
     corrupt metrics if unhandled. See build_full_proba() and fit_model().

  2. MACRO-F1 CONVENTION. Classes with no TEST rows are excluded from the
     macro average (F1 is genuinely undefined with zero support). Classes
     absent from TRAIN but present in test are INCLUDED at their real
     (~0) score -- that is a true detection failure, not a measurement
     artifact, and it is exactly what RQ3 asks about. Both numbers are
     reported: `macro_f1` (headline) and `macro_f1_trained_only`
     (restricted to classes the model actually saw). The gap between them
     is itself a result.

  3. FPR is ambiguous for multiclass, so it is defined operationally: the
     fraction of true-Benign rows predicted as ANY attack class -- the
     false-alarm rate an operator actually experiences. Collapses to the
     standard definition in the binary case.
"""

import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
import lightgbm as lgb
import xgboost as xgb

import split       # C3
import preprocess  # C4
import features    # C5

RESULTS_CSV = Path("results/results.csv")
MODELS_DIR = Path("models")
PER_CLASS_DIR = Path("results/per_class")
CONFUSION_DIR = Path("results/confusion")


def build_model(name: str, seed: int, params: dict | None = None):
    """Model roster from CLAUDE.md section 8 (supervised tiers only).

    Hyperparameters come from configs/experiment.yaml (rule 3.9), not from
    constants here. `params` is that config's model_params[name] block.

    Rule 3.7: imbalance is handled by class WEIGHTING by default, never by
    resampling. Every model that supports class_weight gets 'balanced'.
    XGBoost has no class_weight parameter -- it is handled with
    sample_weight at fit time instead (see fit_model).
    """
    params = dict(params or {})

    if name == "majority":
        # T0 floor. A model that cannot beat this has shown nothing.
        # Deliberately NOT class-weighted -- the whole point is to show what
        # ignoring minority classes does to accuracy vs macro-F1.
        return DummyClassifier(strategy="most_frequent", **params)
    if name == "logreg":
        return LogisticRegression(
            class_weight="balanced", random_state=seed, n_jobs=-1, **params
        )
    if name == "decision_tree":
        return DecisionTreeClassifier(class_weight="balanced", random_state=seed, **params)
    if name == "random_forest":
        return RandomForestClassifier(
            class_weight="balanced", random_state=seed, n_jobs=-1, **params
        )
    if name == "lightgbm":
        return lgb.LGBMClassifier(
            class_weight="balanced", random_state=seed,
            n_jobs=-1, verbose=-1, **params
        )
    if name == "xgboost":
        return xgb.XGBClassifier(
            random_state=seed, n_jobs=-1,
            tree_method="hist", eval_metric="mlogloss", **params
        )
    raise ValueError(f"unknown model {name!r}")


def fit_model(model, name: str, X_train: pd.DataFrame, y_train: np.ndarray):
    """Fit, returning (model, classes_) where classes_ are the ORIGINAL
    label codes the model learned, in the column order of predict_proba.

    XGBoost requires labels to be contiguous 0..k-1. When a class is absent
    from train (Protocol B), y_train is e.g. [0,2,3,4,5,6,7] and XGBoost
    refuses it. So for XGBoost only, labels are compressed to contiguous
    positions before fitting and the original codes are kept for mapping
    predictions back afterwards.
    """
    classes = np.unique(y_train)

    if name == "xgboost":
        y_compressed = np.searchsorted(classes, y_train)
        # Rule 3.7 equivalent for XGBoost: per-sample weights inversely
        # proportional to class frequency, since it has no class_weight arg.
        counts = np.bincount(y_compressed)
        weights = len(y_compressed) / (len(counts) * counts)
        sample_weight = weights[y_compressed]
        model.fit(X_train.to_numpy(), y_compressed, sample_weight=sample_weight)
        return model, classes

    model.fit(X_train.to_numpy(), y_train)
    # sklearn/LightGBM expose the learned classes directly.
    return model, np.asarray(getattr(model, "classes_", classes))


def build_full_proba(model, name: str, X_test: pd.DataFrame,
                     model_classes: np.ndarray, n_classes: int) -> np.ndarray:
    """Predicted probabilities widened to ALL classes, not just the ones
    the model saw in training.

    A model that never saw Brute Force cannot output a Brute Force column.
    Left narrow, that misaligns every downstream metric. Here the narrow
    matrix is scattered into full-width columns and unseen classes keep
    probability 0 -- which is the truthful statement: the model assigns
    that class zero probability because it does not know it exists.
    """
    narrow = model.predict_proba(X_test.to_numpy())
    full = np.zeros((len(X_test), n_classes), dtype=float)
    for col, cls in enumerate(model_classes):
        full[:, int(cls)] = narrow[:, col]
    return full


def compute_metrics(y_true, y_pred, proba_full, class_names, classes_in_train):
    """All metrics required by rule 3.6, plus the RQ3 gap described above."""
    n_classes = len(class_names)
    classes_in_test = np.unique(y_true)
    trained = set(int(c) for c in classes_in_train)

    # Classes with zero test support have undefined F1 -- excluded from the
    # macro average rather than silently counted as 0.
    labels_scored = [int(c) for c in classes_in_test]
    # Classes the model actually had training data for, AND that appear in
    # test -- the "generous" view used for the RQ3 comparison.
    labels_trained = [c for c in labels_scored if c in trained]

    macro_f1 = f1_score(y_true, y_pred, labels=labels_scored,
                        average="macro", zero_division=0)
    macro_f1_trained_only = (
        f1_score(y_true, y_pred, labels=labels_trained, average="macro", zero_division=0)
        if labels_trained else float("nan")
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_scored, zero_division=0
    )
    per_class = pd.DataFrame({
        "class": [class_names[i] for i in labels_scored],
        "precision": precision, "recall": recall, "f1": f1, "test_support": support,
        "in_train": [int(c in trained) for c in labels_scored],
    })

    # PR-AUC. Binary uses the positive-class column; multiclass uses
    # one-vs-rest macro over classes that actually have test support.
    if n_classes == 2:
        pr_auc = average_precision_score(y_true, proba_full[:, 1])
    else:
        y_onehot = np.zeros((len(y_true), n_classes), dtype=int)
        y_onehot[np.arange(len(y_true)), y_true] = 1
        pr_auc = average_precision_score(
            y_onehot[:, labels_scored], proba_full[:, labels_scored], average="macro"
        )

    # Operational false-alarm rate: true-Benign predicted as ANY attack.
    benign_idx = class_names.index("Benign")
    benign_mask = (y_true == benign_idx)
    fpr = float((y_pred[benign_mask] != benign_idx).mean()) if benign_mask.any() else float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    metrics = {
        "macro_f1": macro_f1,
        "macro_f1_trained_only": macro_f1_trained_only,
        "pr_auc_macro": pr_auc,
        "fpr_benign": fpr,
        # Recorded but NEVER the headline (rule 3.6) -- kept precisely so the
        # accuracy-vs-macro-F1 gap can be shown in the report.
        "accuracy": accuracy_score(y_true, y_pred),
        "n_classes_in_test": len(labels_scored),
        "n_classes_absent_from_train": len(labels_scored) - len(labels_trained),
        "classes_absent_from_train": "|".join(
            class_names[c] for c in labels_scored if c not in trained
        ),
    }
    return metrics, per_class, cm_df


def append_result_row(row: dict):
    """Append one result row, keeping results.csv schema-consistent.

    A plain `to_csv(mode="a")` writes the row's values against whatever
    header the file already has. If the code has since GAINED a column --
    `features_used` was added after the first 216 runs -- every new row then
    carries more fields than the header, and the file stops being valid CSV:
    pandas raises "Expected 26 fields, saw 27", and any reader that copes by
    skipping bad lines silently discards the newest results, which is worse
    than crashing.

    So: if the new row introduces columns the file does not have, rewrite the
    file once with the union of both schemas (old rows get empty values for
    the new columns) and then append. Self-healing, and the cost is one
    rewrite on the first run after a schema change.
    """
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])

    if not RESULTS_CSV.exists():
        new.to_csv(RESULTS_CSV, index=False)
        return

    existing_cols = list(pd.read_csv(RESULTS_CSV, nrows=0).columns)
    missing = [c for c in new.columns if c not in existing_cols]

    if missing:
        print(f"results.csv: schema changed, adding column(s) {missing} — "
              "rewriting the file with the union of both schemas")
        old = pd.read_csv(RESULTS_CSV)
        combined = pd.concat([old, new], ignore_index=True)
        combined.to_csv(RESULTS_CSV, index=False)
        return

    # Reindex to the file's column order so values never land under the
    # wrong header, even if the dict's key order differs.
    new.reindex(columns=existing_cols).to_csv(
        RESULTS_CSV, mode="a", header=False, index=False)


def git_state():
    """Commit hash plus a dirty flag -- a hash alone is misleading if there
    are uncommitted changes, which would make the run unreproducible."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                    text=True, check=True).stdout.strip())
    except Exception:
        commit, dirty = "unknown", True
    return commit, dirty


def select_features(X_train, y_train, X_test, feature_set, seed, ranking=None):
    """Full set needs no ranking; reduced sets rank on TRAIN ONLY (rule 3.3).

    `ranking` may be supplied pre-computed. That is not a shortcut around
    rule 3.3: the ranking depends only on the training partition, so it is
    identical for every model that consumes the same partition. Computing it
    once per (protocol, task, seed) instead of once per (model, feature_set)
    is the same number, ~12x fewer LightGBM fits.
    """
    if feature_set.startswith("full"):
        return X_train, X_test, list(X_train.columns)
    n = int(feature_set.split("_")[1])          # "top_15" -> 15
    if ranking is None:
        ranking = features.rank_features(X_train, y_train, seed=seed)
    chosen = ranking["feature"].tolist()[:n]
    return X_train[chosen], X_test[chosen], chosen


def prepare_partition(df, protocol, task, seed):
    """Split + preprocess for one (protocol, task, seed).

    Split apart from run_one so the matrix runner can compute this ONCE and
    reuse it across every (feature_set, model) in that cell -- these steps
    depend only on the partition, never on the model.
    """
    train_idx, test_idx = split.split_dataset(df, protocol=protocol, seed=seed)
    X_train, X_test, y_train, y_test, class_names, _ = preprocess.preprocess(
        df, train_idx, test_idx, task=task
    )
    return X_train, X_test, y_train, y_test, class_names


def run_one(df, model_name, protocol, task, feature_set, seed=42, save_model=True,
            params=None, prepared=None, ranking=None):
    """One experiment cell -> one results.csv row.

    `prepared` and `ranking` are optional caches supplied by run_matrix;
    when absent this function does the work itself, so a single run is still
    a one-line call.
    """
    run_id = f"{model_name}_{protocol}_{task}_{feature_set}_seed{seed}"
    print("\n" + "=" * 70)
    print(f"RUN: {run_id}")
    print("=" * 70)

    if prepared is None:
        prepared = prepare_partition(df, protocol, task, seed)
    X_train, X_test, y_train, y_test, class_names = prepared

    X_train, X_test, chosen = select_features(
        X_train, y_train, X_test, feature_set, seed, ranking=ranking
    )
    print(f"features used ({len(chosen)}): {chosen}")

    model = build_model(model_name, seed, params)

    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    t0 = time.perf_counter()
    model, model_classes = fit_model(model, model_name, X_train, y_train)
    train_seconds = time.perf_counter() - t0
    train_peak_rss_delta_mb = (proc.memory_info().rss - rss_before) / 1e6

    t0 = time.perf_counter()
    proba_full = build_full_proba(model, model_name, X_test, model_classes, len(class_names))
    # Predict from the widened probabilities so predictions and probabilities
    # can never disagree about which class won.
    y_pred = np.argmax(proba_full, axis=1)
    predict_seconds = time.perf_counter() - t0

    metrics, per_class, cm_df = compute_metrics(
        y_test, y_pred, proba_full, class_names, model_classes
    )

    for d in (MODELS_DIR, PER_CLASS_DIR, CONFUSION_DIR, RESULTS_CSV.parent):
        d.mkdir(parents=True, exist_ok=True)
    per_class.to_csv(PER_CLASS_DIR / f"{run_id}.csv", index=False)
    cm_df.to_csv(CONFUSION_DIR / f"{run_id}.csv")

    model_bytes = ""
    if save_model:
        model_path = MODELS_DIR / f"{run_id}.joblib"
        joblib.dump(model, model_path)
        model_bytes = model_path.stat().st_size

    commit, dirty = git_state()
    row = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model_name, "protocol": protocol, "task": task,
        "feature_set": feature_set, "n_features": len(chosen), "seed": seed,
        # WHICH features, not just how many. Without this, reproducing or
        # profiling a top_15 model later means re-deriving the ranking and
        # trusting it lands on the same 15 -- C7 hit exactly that gap.
        # Rows written before 2026-09-02 lack this column.
        "features_used": "|".join(chosen),
        **metrics,
        "n_train_rows": len(X_train), "n_test_rows": len(X_test),
        "train_seconds": round(train_seconds, 3),
        "predict_seconds": round(predict_seconds, 3),
        "train_peak_rss_delta_mb": round(train_peak_rss_delta_mb, 1),
        "model_bytes": model_bytes,
        "git_commit": commit, "git_dirty": dirty,
        "per_class_path": str(PER_CLASS_DIR / f"{run_id}.csv"),
        "confusion_path": str(CONFUSION_DIR / f"{run_id}.csv"),
    }

    append_result_row(row)

    print(f"\nmacro_f1={metrics['macro_f1']:.4f}  "
          f"macro_f1_trained_only={metrics['macro_f1_trained_only']:.4f}  "
          f"accuracy={metrics['accuracy']:.4f}  "
          f"pr_auc={metrics['pr_auc_macro']:.4f}  fpr_benign={metrics['fpr_benign']:.4f}")
    if metrics["classes_absent_from_train"]:
        print(f"classes absent from train: {metrics['classes_absent_from_train']}")
    print("\n--- per-class ---")
    print(per_class.to_string(index=False))
    return row


def dedupe_results():
    """Keep only the last row per run_id.

    `matrix` mode skips run_ids it has already done, but `probe` and
    `verify` modes deliberately re-run cells, which appends a second row
    with the same run_id. Two rows for one run_id would silently
    double-count that cell in any later mean, so the file is normalised
    after every write. Last wins: a rerun supersedes an earlier attempt.
    """
    if not RESULTS_CSV.exists():
        return
    df = pd.read_csv(RESULTS_CSV)
    n_before = len(df)
    df = df.drop_duplicates("run_id", keep="last").reset_index(drop=True)
    if len(df) != n_before:
        df.to_csv(RESULTS_CSV, index=False)
        print(f"results.csv: removed {n_before - len(df)} duplicate run_id row(s)")


def completed_run_ids() -> set:
    """run_ids already present in results.csv, so a long sweep can resume
    after an interruption instead of redoing hours of finished work."""
    if not RESULTS_CSV.exists():
        return set()
    return set(pd.read_csv(RESULTS_CSV)["run_id"].astype(str))


def needs_ranking(feature_sets) -> bool:
    return any(not fs.startswith("full") for fs in feature_sets)


def run_matrix(config_path="configs/experiment.yaml", limit=None):
    """Sweep the experiment matrix from config (rule 3.9).

    Loop order matters: (protocol, task, seed) outermost, so the split,
    the preprocessing, and the feature ranking are each computed ONCE per
    partition and reused across every (feature_set, model) inside it.
    """
    cfg = yaml.safe_load(Path(config_path).read_text())
    df = pd.read_parquet(cfg["subsample_path"])
    print(f"Loaded subsample: {df.shape}")

    done = completed_run_ids()
    print(f"{len(done)} run(s) already in {RESULTS_CSV} -- these will be skipped.")

    planned = [
        (p, t, s, fs, m)
        for p in cfg["protocols"] for t in cfg["tasks"] for s in cfg["seeds"]
        for fs in cfg["feature_sets"] for m in cfg["models"]
    ]
    todo = [x for x in planned
            if f"{x[4]}_{x[0]}_{x[1]}_{x[3]}_seed{x[2]}" not in done]
    print(f"{len(planned)} cells in the matrix, {len(todo)} still to run.")
    if limit:
        todo = todo[:limit]
        print(f"limit={limit}: running only the first {len(todo)}.")

    completed = 0
    for protocol in cfg["protocols"]:
        for task in cfg["tasks"]:
            for seed in cfg["seeds"]:
                cell = [x for x in todo if x[0] == protocol and x[1] == task and x[2] == seed]
                if not cell:
                    continue
                # Computed once per partition, reused by every run below.
                prepared = prepare_partition(df, protocol, task, seed)
                ranking = None
                if needs_ranking({x[3] for x in cell}):
                    ranking = features.rank_features(prepared[0], prepared[2], seed=seed)

                for _, _, _, feature_set, model_name in cell:
                    try:
                        run_one(
                            df, model_name, protocol, task, feature_set, seed=seed,
                            params=cfg.get("model_params", {}).get(model_name, {}),
                            prepared=prepared, ranking=ranking,
                        )
                        completed += 1
                    except Exception as exc:
                        # One failing cell must not abandon the sweep. Failures
                        # are printed loudly and left ABSENT from results.csv,
                        # so a rerun retries them rather than silently
                        # recording a bad row.
                        print(f"!! FAILED {model_name}/{protocol}/{task}/"
                              f"{feature_set}/seed{seed}: {type(exc).__name__}: {exc}")
    dedupe_results()
    print(f"\nMatrix run complete: {completed} run(s) added to {RESULTS_CSV}")


def run_probe(config_path="configs/experiment.yaml"):
    """Measure per-model cost on one cheap cell before committing to the
    full sweep -- caps should come from measurement, not guesswork."""
    cfg = yaml.safe_load(Path(config_path).read_text())
    probe = cfg["probe"]
    df = pd.read_parquet(cfg["subsample_path"])
    print(f"Loaded subsample: {df.shape}")
    print(f"PROBE cell: {probe}")

    prepared = prepare_partition(df, probe["protocol"], probe["task"], probe["seed"])
    timings = []
    for model_name in cfg["models"]:
        try:
            row = run_one(
                df, model_name, probe["protocol"], probe["task"], probe["feature_set"],
                seed=probe["seed"],
                params=cfg.get("model_params", {}).get(model_name, {}),
                prepared=prepared,
            )
            timings.append({
                "model": model_name,
                "train_seconds": row["train_seconds"],
                "predict_seconds": row["predict_seconds"],
                "model_mb": (row["model_bytes"] or 0) / 1e6,
                "macro_f1": round(row["macro_f1"], 4),
            })
        except Exception as exc:
            print(f"!! FAILED {model_name}: {type(exc).__name__}: {exc}")
            timings.append({"model": model_name, "train_seconds": float("nan"),
                            "predict_seconds": float("nan"), "model_mb": float("nan"),
                            "macro_f1": float("nan")})

    dedupe_results()
    print("\n" + "=" * 70)
    print("PROBE TIMINGS (one cell, binary task, 39 features)")
    print("=" * 70)
    print(pd.DataFrame(timings).to_string(index=False))

    n_cells = (len(cfg["protocols"]) * len(cfg["tasks"]) * len(cfg["feature_sets"])
               * len(cfg["seeds"]))
    total_train = sum(t["train_seconds"] for t in timings
                      if t["train_seconds"] == t["train_seconds"])
    print(f"\nOne full pass over all {len(cfg['models'])} models cost "
          f"{total_train:.0f}s of fitting on this cell.")
    print(f"The full matrix has {n_cells} cells -> a rough lower bound of "
          f"{total_train * n_cells / 3600:.1f}h of model fitting alone "
          f"(multiclass and ranking fits will add to this).")




# =====================================================================
# T2 - benign-only anomaly detectors (RQ3)
# =====================================================================
# These answer the second half of RQ3: "do unsupervised detectors hold up
# better on attack families absent from training?"
#
# The training protocol is fundamentally different from everything above:
# fit on BENIGN ROWS ONLY, then score how anomalous each test row looks.
# Because no attack is ever seen in training, EVERY attack family is an
# unseen family to these models -- exactly the condition the supervised
# models failed under (Spoofing -40 pts, Brute Force -> 0.000).
#
# Results go to their own CSV: these rows have different semantics from the
# supervised matrix (no 8-class macro-F1), and mixing them would invite a
# meaningless average across incompatible runs.

UNSUP_RESULTS_CSV = Path("results/results_unsupervised.csv")

# Operational false-alarm budget used to place the decision threshold.
# 5% is a convention, not a derived value -- flagged as a judgment call.
DEFAULT_FPR_TARGET = 0.05


def build_unsupervised(name: str, seed: int):
    """Isolation Forest (primary) and an autoencoder (secondary, section 8).

    The autoencoder is an MLPRegressor trained to reconstruct its own input,
    because section 5 excludes PyTorch at this scope. Reconstruction error
    is the anomaly score -- rows unlike the benign traffic it was fitted on
    reconstruct badly.
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.neural_network import MLPRegressor

    if name == "isolation_forest":
        return IsolationForest(n_estimators=100, random_state=seed, n_jobs=-1)
    if name == "autoencoder":
        return MLPRegressor(hidden_layer_sizes=(16,), max_iter=50,
                            random_state=seed, early_stopping=True)
    raise ValueError(f"unknown unsupervised model {name!r}")


def fit_unsupervised(model, name: str, X_benign: pd.DataFrame):
    """Fit on benign rows only."""
    Xb = X_benign.to_numpy()
    if name == "autoencoder":
        model.fit(Xb, Xb)          # reconstruct the input
    else:
        model.fit(Xb)
    return model


def anomaly_scores(model, name: str, X: pd.DataFrame) -> np.ndarray:
    """Higher = more anomalous, for both model types."""
    Xa = X.to_numpy()
    if name == "autoencoder":
        recon = model.predict(Xa)
        return ((recon - Xa) ** 2).mean(axis=1)     # per-row reconstruction error
    # IsolationForest.score_samples is higher for NORMAL points, so negate.
    return -model.score_samples(Xa)


def run_unsupervised_one(df, model_name, protocol, feature_set="full_39",
                         seed=42, fpr_target=DEFAULT_FPR_TARGET,
                         cat_task="multiclass7", out_csv=None):
    """One benign-only detector run -> one results row.

    `cat_task` selects the CATEGORY vocabulary used for the per-family
    breakdown, and therefore which rows are in scope at all:
    "multiclass7" is the 8-category superset, "multiclass6" the six
    categories the project brief specifies.

    `out_csv` overrides the destination. The six-class runs go to their
    OWN file rather than appending to results_unsupervised.csv, because
    that file is append-only and its rows carry per-family columns named
    after the 8-category vocabulary. Appending six-class rows would give
    the file two different schemas -- the exact fault that corrupted
    results.csv on 2026-09-19.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    scope_tag = "" if cat_task == "multiclass7" else f"_{cat_task}"
    run_id = f"{model_name}_{protocol}_benignonly{scope_tag}_{feature_set}_seed{seed}"
    print("\n" + "=" * 70)
    print(f"RUN (T2): {run_id}")
    print("=" * 70)

    # Preprocess at multiclass7 granularity so the CATEGORY of each test row
    # is available for the per-family breakdown RQ3 actually asks about;
    # the binary target is then derived from it.
    train_idx, test_idx = split.split_dataset(df, protocol=protocol, seed=seed)
    X_train, X_test, y_train_cat, y_test_cat, class_names, _ = preprocess.preprocess(
        df, train_idx, test_idx, task=cat_task
    )
    benign_idx = class_names.index("Benign")

    # THE defining step: train on benign rows only.
    X_benign = X_train[y_train_cat == benign_idx]
    print(f"training on {len(X_benign):,} benign rows only "
          f"(of {len(X_train):,} available)")
    if len(X_benign) == 0:
        raise ValueError("no benign rows in this training partition")

    model = build_unsupervised(model_name, seed)
    t0 = time.perf_counter()
    model = fit_unsupervised(model, model_name, X_benign)
    train_seconds = time.perf_counter() - t0

    # Threshold placed on TRAINING benign scores only (rule 3.3): the score
    # below which (1 - fpr_target) of benign training traffic falls.
    train_benign_scores = anomaly_scores(model, model_name, X_benign)
    threshold = float(np.quantile(train_benign_scores, 1 - fpr_target))

    t0 = time.perf_counter()
    test_scores = anomaly_scores(model, model_name, X_test)
    predict_seconds = time.perf_counter() - t0

    y_true_attack = (y_test_cat != benign_idx).astype(int)   # 1 = attack
    y_pred_attack = (test_scores > threshold).astype(int)

    tp = int(((y_pred_attack == 1) & (y_true_attack == 1)).sum())
    fp = int(((y_pred_attack == 1) & (y_true_attack == 0)).sum())
    fn = int(((y_pred_attack == 0) & (y_true_attack == 1)).sum())
    tn = int(((y_pred_attack == 0) & (y_true_attack == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    row = {
        "run_id": run_id, "model": model_name, "protocol": protocol,
        "task": "benign_only", "feature_set": feature_set, "seed": seed,
        "fpr_target": fpr_target, "threshold": round(threshold, 6),
        # Threshold-free, so independent of the 5% choice above.
        "roc_auc": round(roc_auc_score(y_true_attack, test_scores), 4),
        "pr_auc": round(average_precision_score(y_true_attack, test_scores), 4),
        "attack_precision": round(precision, 4),
        "attack_recall": round(recall, 4),
        "attack_f1": round(f1, 4),
        "fpr_benign_actual": round(fp / (fp + tn), 4) if fp + tn else float("nan"),
        "n_train_benign": len(X_benign), "n_test_rows": len(X_test),
        "train_seconds": round(train_seconds, 2),
        "predict_seconds": round(predict_seconds, 2),
    }

    # Per-family detection rate -- the RQ3 comparison against the supervised
    # models, every one of these families being unseen during training.
    for i, cname in enumerate(class_names):
        if i == benign_idx:
            continue
        mask = (y_test_cat == i)
        key = "recall_" + cname.replace(" ", "_")
        row[key] = round(float(y_pred_attack[mask].mean()), 4) if mask.any() else float("nan")

    commit, dirty = git_state()
    row["git_commit"], row["git_dirty"] = commit, dirty

    dest = Path(out_csv) if out_csv else UNSUP_RESULTS_CSV
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(dest, mode="a",
                               header=not dest.exists(), index=False)

    print(f"ROC-AUC={row['roc_auc']}  PR-AUC={row['pr_auc']}  "
          f"attack recall={row['attack_recall']}  benign FPR={row['fpr_benign_actual']}")
    fam = {k[len('recall_'):]: v for k, v in row.items() if k.startswith("recall_")}
    print("per-family detection rate:", fam)
    return row


def run_unsupervised_matrix(seeds=(42, 43, 44), protocols=("A", "B"),
                            models=("isolation_forest", "autoencoder"),
                            cat_task="multiclass7", out_csv=None):
    """T2 sweep. Full feature set only -- feature reduction is an RQ2
    question, not the RQ3 one these models exist to answer."""
    df = pd.read_parquet("data/processed/cic_iot2023_subsample.parquet")
    print(f"Loaded subsample: {df.shape}")
    # The resume guard must read the SAME file the runs will be written to,
    # and must build the same run_id that run_unsupervised_one generates --
    # scope tag included. An earlier version did neither, so every six-class
    # run was skipped as "already done" because the eight-class file happened
    # to contain the untagged id. Same class of fault as the results.csv
    # field-shift on 2026-09-19: an identifier missing a dimension that varies.
    dest = Path(out_csv) if out_csv else UNSUP_RESULTS_CSV
    scope_tag = "" if cat_task == "multiclass7" else f"_{cat_task}"
    done = set()
    if dest.exists():
        done = set(pd.read_csv(dest)["run_id"].astype(str))
    for protocol in protocols:
        for model_name in models:
            for seed in seeds:
                rid = (f"{model_name}_{protocol}_benignonly{scope_tag}"
                       f"_full_39_seed{seed}")
                if rid in done:
                    print(f"skip (done): {rid}")
                    continue
                try:
                    run_unsupervised_one(df, model_name, protocol, seed=seed,
                                         cat_task=cat_task, out_csv=out_csv)
                except Exception as exc:
                    print(f"!! FAILED {rid}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "verify"

    if mode == "probe":
        run_probe()
    elif mode == "matrix":
        # Optional cap on how many cells to run in one invocation.
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
        run_matrix(limit=limit)
    elif mode == "unsup":
        run_unsupervised_matrix()
    else:
        # Pass-1 verification. The majority-class predictor is used first
        # deliberately: its numbers are hand-checkable, so if the metric
        # machinery is wrong, it shows up immediately rather than hiding
        # inside a plausible-looking tree result.
        df = pd.read_parquet("data/processed/cic_iot2023_subsample.parquet")
        print(f"Loaded subsample: {df.shape}")
        run_one(df, "majority", protocol="B", task="binary", feature_set="full_39")
        run_one(df, "decision_tree", protocol="B", task="binary", feature_set="full_39")

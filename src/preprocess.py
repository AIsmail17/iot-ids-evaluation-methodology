"""
preprocess.py — Component C4: transforms fitted on the training partition only.

Two separate jobs live here, and they behave differently on purpose:

  1. Feature scaling (StandardScaler) -- fit on X_train ONLY, then applied
     to both train and test. This is rule 3.3 in CLAUDE.md: fitting a
     scaler on the full dataset lets its mean/std absorb information from
     the test partition, which is a real, demonstrated leakage path.

  2. Label/task encoding (binary / 7-category / 34-way) -- the OPPOSITE:
     uses a FIXED, split-independent lookup table, not something fit per
     split. Reason: C3 already showed classes can be entirely absent from
     train under Protocol B. An encoder "fit" on train labels would have no
     code for a class it never saw, and would break the moment that class
     shows up in test. A static lookup table has no such blind spot.

Input: the C2 subsample DataFrame + train/test index arrays from C3.
Output: scaled feature DataFrames and encoded label arrays for whichever
task (binary / multiclass7 / multiclass34) the caller asks for.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

import split  # C3 -- used only in the __main__ demo below


# --- Columns that are never model features -----------------------------
# `label` is the target. `source_file` is grouping metadata WE added in
# C1 -- if it leaked into X it would hand Protocol B's answer key straight
# to the model (and be a meaningless ~300-level ID to a real one anyway).
NON_FEATURE_COLS = ["label", "source_file"]

# Protocol Type is a nominal protocol number (6=TCP, 17=UDP, 1=ICMP, ...),
# not an ordered quantity -- scaling it doesn't fix the real issue with
# treating it as numeric, so it's left as the dataset's native raw code
# rather than scaled or one-hot encoded. See preprocess walkthrough.
NON_SCALED_FEATURE_COLS = ["Protocol Type"]


# --- 34-subtype -> 7-category taxonomy ----------------------------------
# From the CICIoT2023 paper's own category grouping. Keys are the raw
# `label` values (folder names) exactly as C1 wrote them.
CATEGORY_MAP = {
    # DDoS (12 subtypes)
    "DDoS-ACK_Fragmentation": "DDoS", "DDoS-HTTP_Flood": "DDoS",
    "DDoS-ICMP_Flood": "DDoS", "DDoS-ICMP_Fragmentation": "DDoS",
    "DDoS-PSHACK_FLOOD": "DDoS", "DDoS-RSTFINFLOOD": "DDoS",
    "DDoS-SYN_Flood": "DDoS", "DDoS-SlowLoris": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS", "DDoS-TCP_Flood": "DDoS",
    "DDoS-UDP_Flood": "DDoS", "DDoS-UDP_Fragmentation": "DDoS",
    # DoS (4 subtypes)
    "DoS-HTTP_Flood": "DoS", "DoS-SYN_Flood": "DoS",
    "DoS-TCP_Flood": "DoS", "DoS-UDP_Flood": "DoS",
    # Reconnaissance (5 subtypes) -- VulnerabilityScan belongs here despite
    # its folder name not carrying the "Recon-" prefix.
    "Recon-HostDiscovery": "Reconnaissance", "Recon-OSScan": "Reconnaissance",
    "Recon-PingSweep": "Reconnaissance", "Recon-PortScan": "Reconnaissance",
    "VulnerabilityScan": "Reconnaissance",
    # Web-based (6 subtypes)
    "BrowserHijacking": "Web-based", "CommandInjection": "Web-based",
    "SqlInjection": "Web-based", "XSS": "Web-based",
    "Uploading_Attack": "Web-based", "Backdoor_Malware": "Web-based",
    # Brute Force (1 subtype) -- the single-file class flagged in
    # DECISIONS.md 2026-09-02; excluded from the C6 multiclass Protocol B
    # config, but the mapping itself is defined here regardless.
    "DictionaryBruteForce": "Brute Force",
    # Spoofing (2 subtypes)
    "DNS_Spoofing": "Spoofing", "MITM-ArpSpoofing": "Spoofing",
    # Mirai (3 subtypes)
    "Mirai-greeth_flood": "Mirai", "Mirai-greip_flood": "Mirai",
    "Mirai-udpplain": "Mirai",
    # Benign -- its own class, not one of the 7 attack categories.
    "Benign_Final": "Benign",
}


# --- Task class scopes --------------------------------------------------
# The supervisor's plan (Resources and References/Lightweight Machine
# Learning.docx) specifies a SIX-class problem:
#   Benign + DDoS + DoS + Reconnaissance + Spoofing + Brute Force
# Mirai and Web-based are outside that scope, so `multiclass6` DROPS those
# rows rather than relabelling them -- the plan asks for a subset, not a
# remapping. `multiclass7` keeps all eight and is retained as a documented
# superset. This matters: Mirai is the easiest class in the data
# (recall 0.998), so including it lifts every macro average.
TASK_CATEGORIES = {
    "multiclass6": ["Benign", "Brute Force", "DDoS", "DoS",
                    "Reconnaissance", "Spoofing"],
}


def get_feature_columns(df: pd.DataFrame) -> list:
    """Every column that's actually a model feature."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def filter_to_task_classes(df_slice: pd.DataFrame, task: str) -> pd.DataFrame:
    """Drop rows whose category is out of scope for this task.

    Applied AFTER the split, per partition. That is sound for both
    protocols: Protocol A stratifies on the fine-grained label so removing
    whole labels leaves the rest proportional, and Protocol B assigns whole
    capture files, and the dropped categories occupy their own files
    entirely. So no row changes side as a result of filtering.
    """
    allowed = TASK_CATEGORIES.get(task)
    if allowed is None:
        return df_slice
    keep = df_slice["label"].map(CATEGORY_MAP).isin(allowed)
    n_dropped = int((~keep).sum())
    if n_dropped:
        print(f"task={task!r}: dropping {n_dropped:,} rows outside the "
              f"{len(allowed)}-class scope ({n_dropped/len(df_slice)*100:.1f}%)")
    return df_slice[keep]


def clean_nonfinite_rows(df_slice: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Drop rows where any feature value is inf or NaN.

    Root cause (checked directly on the real data, not assumed): rows with
    Number == 1 -- a flow "window" containing exactly one packet -- have
    Std/Variance undefined (variance of a single value has no meaning) and
    sometimes Rate == inf (a rate computed against a ~zero time delta).
    This is a tiny, well-understood edge case: ~350 of 4,458,051 rows
    (~0.008%), spread across almost every class, never more than a couple
    of rows out of any single class -- including the rarest ones. Dropping
    is simpler than inventing an imputation rule for a fraction this small,
    and just as defensible.

    Applied separately to train and test (both calls happen before
    scaling) -- this is safe under rule 3.3 because "is this row
    degenerate" is a fact about that row alone, not a statistic fitted
    across rows, so there's no leakage in checking train and test the same
    way independently.
    """
    finite_mask = np.isfinite(df_slice[feature_cols]).all(axis=1)
    n_dropped = int((~finite_mask).sum())
    if n_dropped:
        print(f"Dropping {n_dropped} rows with non-finite feature values "
              f"({n_dropped / len(df_slice) * 100:.4f}% of this partition)")
    return df_slice[finite_mask]


def encode_task_labels(labels: pd.Series, task: str):
    """Map raw 34-way labels to numeric targets for the requested task.

    Returns (y, class_names) where y[i] is the integer code of
    class_names[y[i]] -- class_names gives the human-readable order, which
    C6 will need later for confusion matrices and per-class metric tables.

    Uses a fixed lookup, independent of which split this is called on --
    see the module docstring for why that matters here specifically.
    """
    if task == "binary":
        # Judgment call: "Benign" is class 0, everything else (all 33
        # attack subtypes) is class 1 "Attack". Matches the brief's own
        # phrasing in section 4/9 ("Binary (benign/attack)").
        class_names = ["Benign", "Attack"]
        y = (labels != "Benign_Final").astype(int).to_numpy()
        return y, class_names

    elif task == "multiclass6":
        # The supervisor's six-class scope. Rows outside it are already gone
        # (filter_to_task_classes), so the vocabulary is exactly these six.
        class_names = list(TASK_CATEGORIES["multiclass6"])
        categories = labels.map(CATEGORY_MAP)
        unknown = set(categories.dropna().unique()) - set(class_names)
        if unknown:
            raise ValueError(f"multiclass6 got out-of-scope categories: {unknown}")
        lookup = {name: i for i, name in enumerate(class_names)}
        y = categories.map(lookup).to_numpy()
        return y, class_names

    elif task == "multiclass7":
        # Judgment call, flagged explicitly: 8 classes in practice (the 7
        # attack categories + Benign), not 7 -- see walkthrough. Fixed
        # alphabetical order so the encoding is stable across runs.
        class_names = sorted(set(CATEGORY_MAP.values()))
        categories = labels.map(CATEGORY_MAP)
        if categories.isna().any():
            unmapped = labels[categories.isna()].unique()
            raise ValueError(f"CATEGORY_MAP is missing these labels: {unmapped}")
        lookup = {name: i for i, name in enumerate(class_names)}
        y = categories.map(lookup).to_numpy()
        return y, class_names

    elif task == "multiclass34":
        # Raw subtype labels, fixed alphabetical order.
        class_names = sorted(CATEGORY_MAP.keys())
        lookup = {name: i for i, name in enumerate(class_names)}
        y = labels.map(lookup).to_numpy()
        return y, class_names

    else:
        raise ValueError(f"task must be 'binary', 'multiclass7', or 'multiclass34', got {task!r}")


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Fit StandardScaler on X_train only; apply to both. Rule 3.3."""
    scale_cols = [c for c in X_train.columns if c not in NON_SCALED_FEATURE_COLS]

    scaler = StandardScaler()
    scaler.set_output(transform="pandas")  # keep column names / DataFrame shape
    scaler.fit(X_train[scale_cols])  # <-- fit on TRAIN ONLY, this is the rule

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[scale_cols] = scaler.transform(X_train[scale_cols])
    X_test_scaled[scale_cols] = scaler.transform(X_test[scale_cols])

    return X_train_scaled, X_test_scaled, scaler


def preprocess(df: pd.DataFrame, train_idx, test_idx, task: str):
    """Orchestrator: slice by C3's indices, clean, scale features, encode labels."""
    feature_cols = get_feature_columns(df)

    train_slice = df.iloc[train_idx].reset_index(drop=True)
    test_slice = df.iloc[test_idx].reset_index(drop=True)
    print(f"\nX_train (before cleaning): {train_slice.shape}   X_test (before cleaning): {test_slice.shape}")

    train_slice = clean_nonfinite_rows(train_slice, feature_cols)
    test_slice = clean_nonfinite_rows(test_slice, feature_cols)
    print(f"X_train (after cleaning):  {train_slice.shape}   X_test (after cleaning):  {test_slice.shape}")

    # Restrict to the task's class scope before anything is fitted, so the
    # scaler sees only the data the task is actually about.
    train_slice = filter_to_task_classes(train_slice, task)
    test_slice = filter_to_task_classes(test_slice, task)
    if task in TASK_CATEGORIES:
        print(f"X_train (in scope):        {train_slice.shape}   X_test (in scope):        {test_slice.shape}")

    X_train = train_slice[feature_cols].reset_index(drop=True)
    X_test = test_slice[feature_cols].reset_index(drop=True)

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    y_train, class_names = encode_task_labels(train_slice["label"].reset_index(drop=True), task)
    y_test, _ = encode_task_labels(test_slice["label"].reset_index(drop=True), task)

    print(f"task={task!r}  classes={class_names}")
    train_counts = pd.Series(y_train).value_counts().sort_index()
    test_counts = pd.Series(y_test).value_counts().sort_index()
    report = pd.DataFrame({"train": train_counts, "test": test_counts}).fillna(0).astype(int)
    report.index = [class_names[i] for i in report.index]
    print(report.to_string())
    missing_train = set(range(len(class_names))) - set(np.unique(y_train))
    missing_test = set(range(len(class_names))) - set(np.unique(y_test))
    if missing_train:
        print(f"WARNING -- classes entirely absent from train: {[class_names[i] for i in missing_train]}")
    if missing_test:
        print(f"WARNING -- classes entirely absent from test:  {[class_names[i] for i in missing_test]}")

    return X_train_scaled, X_test_scaled, y_train, y_test, class_names, scaler


if __name__ == "__main__":
    # Demo / independent test, same pattern as split.py's __main__: run on
    # the real C2 subsample, through C3's split, so this is checked against
    # real data, not a toy example.
    subsample_path = Path("data/processed/cic_iot2023_subsample.parquet")
    df = pd.read_parquet(subsample_path)
    print(f"Loaded {subsample_path}: {df.shape}")

    print("\n" + "=" * 70)
    print("Protocol A, task=multiclass7 (the clean case -- no dropped classes expected)")
    print("=" * 70)
    train_idx, test_idx = split.split_dataset(df, protocol="A")
    preprocess(df, train_idx, test_idx, task="multiclass7")

    print("\n" + "=" * 70)
    print("Protocol B, task=multiclass7 (Brute Force expected to drop out)")
    print("=" * 70)
    train_idx, test_idx = split.split_dataset(df, protocol="B")
    preprocess(df, train_idx, test_idx, task="multiclass7")

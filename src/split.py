"""
split.py — Component C3: Protocol A and Protocol B behind one interface.

This is the project's central mechanism (CLAUDE.md section 1): every model
gets evaluated under BOTH protocols, and the difference between them is the
headline result.

  Protocol A (random)  -- stratified random split. The control condition,
                           included specifically to be criticised. Not a
                           valid result on its own (CLAUDE.md rule 3.1).
  Protocol B (grouped)  -- GroupShuffleSplit on `source_file`, so every row
                           from one capture file lands entirely in train or
                           entirely in test. This is the leakage-free split.

Both protocols stratify/group on the raw 34-way `label` column (not a
binary or 7-category target) -- a split that's proportional at the finest
grain is automatically proportional at any coarser grouping built from it
later. This keeps split.py agnostic of whatever task C4/C6 build on top.

Output is index arrays into the subsample DataFrame, not copies of it --
C4 will use these to fit transforms on the train indices only (rule 3.3).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GroupShuffleSplit


def split_protocol_a(df: pd.DataFrame, label_col: str, test_size: float, seed: int):
    """Stratified random split -- the control condition."""
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=seed,
        stratify=df[label_col].values,  # keeps class proportions equal in train/test
    )
    return train_idx, test_idx


def split_protocol_b(df: pd.DataFrame, group_col: str, test_size: float, seed: int):
    """Grouped split -- every source_file lands entirely on one side."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    # .split() is a generator over splits; n_splits=1 means exactly one to take.
    train_idx, test_idx = next(gss.split(df, groups=df[group_col].values))
    return train_idx, test_idx


def split_dataset(
    df: pd.DataFrame,
    protocol: str,
    label_col: str = "label",
    group_col: str = "source_file",
    test_size: float = 0.2,
    seed: int = 42,
):
    """One interface for both protocols. protocol must be 'A' or 'B'."""
    if protocol == "A":
        return split_protocol_a(df, label_col, test_size, seed)
    elif protocol == "B":
        return split_protocol_b(df, group_col, test_size, seed)
    else:
        raise ValueError(f"protocol must be 'A' or 'B', got {protocol!r}")


def print_split_report(df: pd.DataFrame, train_idx, test_idx, label_col: str, protocol_name: str):
    """Sizes, per-class train/test distribution, and any class that's
    completely missing from one side -- the live check for the
    single-source-file problem logged in DECISIONS.md (2026-09-02)."""
    print(f"\n--- Protocol {protocol_name} ---")
    print(f"train: {len(train_idx):,} rows   test: {len(test_idx):,} rows")

    train_counts = df[label_col].iloc[train_idx].value_counts()
    test_counts = df[label_col].iloc[test_idx].value_counts()
    report = pd.DataFrame({"train": train_counts, "test": test_counts}).fillna(0).astype(int)
    report["test_share"] = (report["test"] / (report["train"] + report["test"])).round(3)
    print(report.sort_values("train", ascending=False).to_string())

    all_labels = set(df[label_col].unique())
    missing_from_train = all_labels - set(df[label_col].iloc[train_idx].unique())
    missing_from_test = all_labels - set(df[label_col].iloc[test_idx].unique())
    if missing_from_train:
        print(f"WARNING -- classes entirely absent from train: {sorted(missing_from_train)}")
    if missing_from_test:
        print(f"WARNING -- classes entirely absent from test:  {sorted(missing_from_test)}")
    if not missing_from_train and not missing_from_test:
        print("Every class present in both train and test.")


if __name__ == "__main__":
    # Demo / independent test, per CLAUDE.md's "each component is
    # independently testable" -- run on the real C2 subsample, both
    # protocols, so the diagnostics above are checked against real data
    # rather than a toy example.
    subsample_path = Path("data/processed/cic_iot2023_subsample.parquet")
    df = pd.read_parquet(subsample_path)
    print(f"Loaded {subsample_path}: {df.shape}")

    for protocol in ("A", "B"):
        train_idx, test_idx = split_dataset(df, protocol=protocol)
        print_split_report(df, train_idx, test_idx, "label", protocol)

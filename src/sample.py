"""
sample.py — Component C2: stratified, class-capped subsampling.

Takes the full C1 parquet (46.78M rows) and produces a smaller subsample
that's actually practical to iterate on, WITHOUT letting the sampling itself
erase the minority classes that make this dataset's imbalance interesting.

The method (class-capped, not class-uniform):
    For each class (the `label` column from C1):
      - if it has <= cap_per_class rows, keep every row.
      - if it has more, take a uniform random sample of exactly
        cap_per_class rows.

This is deliberately NOT "take X% of every class" — that would shrink the
rarest classes (e.g. Uploading_Attack, 1,252 rows) down to almost nothing.
Capping only touches the classes that have rows to spare.

Parameters (cap_per_class, seed, paths) live in configs/sample.yaml, not
here, per CLAUDE.md rule 3.9.
"""

import pandas as pd
import yaml
from pathlib import Path


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def print_class_summary(df: pd.DataFrame, stage_name: str) -> None:
    """Per-class row counts AND distinct source_file counts.

    The source_file count matters here specifically because C2 removes rows,
    and we want to confirm sampling didn't accidentally erase a whole file's
    worth of a class by chance (see the walkthrough for why that's unlikely
    at these row counts, but it's cheap to just check rather than assume).
    """
    print(f"\n--- {stage_name} ---")
    print(f"shape: {df.shape}")
    summary = df.groupby("label", observed=True).agg(
        rows=("label", "size"),
        distinct_source_files=("source_file", "nunique"),
    ).sort_values("rows", ascending=False)
    print(summary.to_string())


def cap_per_class(df: pd.DataFrame, label_col: str, cap: int, seed: int) -> pd.DataFrame:
    """Return a new DataFrame where no class has more than `cap` rows.

    Groups once (single pass over the full frame), then for each group
    either keeps it whole or draws a fixed-seed uniform random sample.
    Looping + concatenating rather than a groupby().apply() one-liner,
    to match load.py's style and keep each step visible.
    """
    kept_groups = []
    for label, group in df.groupby(label_col, sort=False, observed=True):
        if len(group) <= cap:
            kept_groups.append(group)
        else:
            kept_groups.append(group.sample(n=cap, random_state=seed))
    return pd.concat(kept_groups, ignore_index=True)


def build_subsample(config: dict) -> None:
    input_path = Path(config["input_path"])
    output_path = Path(config["output_path"])
    cap = config["cap_per_class"]
    seed = config["seed"]

    if output_path.exists():
        raise FileExistsError(
            f"{output_path} already exists. Delete it first to regenerate — "
            "this script never overwrites silently."
        )

    print(f"Reading {input_path} ...")
    full = pd.read_parquet(input_path)
    print_class_summary(full, "Full dataset (C1 output)")

    print(f"\nApplying cap_per_class={cap:,}, seed={seed} ...")
    subsample = cap_per_class(full, "label", cap, seed)
    del full  # free the 12+ GB full frame before we write anything

    print_class_summary(subsample, "After class-capped subsampling")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    subsample.to_parquet(output_path, index=False)

    # Verify by reading back, same as C1 does.
    written = pd.read_parquet(output_path)
    print_class_summary(written, f"After writing {output_path}")
    print(f"\nParquet file size on disk: {output_path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    config = load_config(Path("configs/sample.yaml"))
    build_subsample(config)

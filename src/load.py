"""
load.py — Component C1: data ingestion for CIC-IoT2023.

Converts the raw per-attack CSV files under CSV/ into a single Parquet file,
once. Every downstream component (C2 onward) reads that Parquet file —
nobody re-parses CSV after this script has run (CLAUDE.md section 11).

Why CSV/ and not MERGED_CSV/: MERGED_CSV's files are pre-shuffled across
attack types and across original capture sessions, so "which Merged file a
row landed in" is not a meaningful grouping key. CSV/ preserves one capture
segment per file. See DECISIONS.md, 2026-09-02, for the full reasoning.

What this script does, in order:
  1. Walk CSV/<attack_folder>/*.csv (globbing *.csv skips the two README
     PDFs that live alongside the folders).
  2. For each file: read it, tag every row with:
       - `label`       = the parent folder name, verbatim (e.g.
                         "DDoS-ICMP_Flood"). These files carry no Label
                         column of their own — every row in a file belongs
                         to the same attack type, so the folder name IS the
                         label. Not normalized or mapped to the 7 broad
                         attack categories here — that mapping belongs to
                         whichever downstream component actually needs it.
       - `source_file` = "<folder>/<filename>" — the grouping key C3's
                         GroupShuffleSplit / GroupKFold will use for
                         Protocol B. This is the single most important
                         column this script produces.
     then downcast that file's float64 columns to float32.
  3. Concatenate every file's DataFrame into one, and write it to
     data/processed/cic_iot2023.parquet.
  4. Print shape and class (label) distribution after each stage.
"""

import pandas as pd
from pathlib import Path

# --- Config -----------------------------------------------------------
# Plain constants, not a YAML config: CLAUDE.md's "config files drive
# results" rule (section 3.9) is about experiment runs (C6/C7), where a
# forgotten hardcoded constant can silently change a reported number. This
# script has exactly one input and one output path and runs once per
# machine — a YAML file here would be indirection with no payoff.
RAW_ROOT = Path("D:/data/CICIoT2023/CSV")
OUT_PATH = Path("data/processed/cic_iot2023.parquet")


def find_csv_files(raw_root: Path) -> list[Path]:
    """Every per-attack CSV file, one level down inside each attack folder.

    Non-recursive-beyond-one-level on purpose: raw_root/<attack>/*.csv.
    This naturally skips the two README PDFs (they don't match *.csv) and
    would loudly fail (empty list) if the layout ever changes shape.
    """
    files = sorted(raw_root.glob("*/*.csv"))
    return files


def load_and_tag_one(csv_path: Path, raw_root: Path) -> pd.DataFrame:
    """Read one raw CSV, attach `label` and `source_file`, downcast floats."""
    df = pd.read_csv(csv_path)

    # label = parent folder name. E.g. .../CSV/DDoS-ICMP_Flood/DDoS-ICMP_Flood1.pcap.csv
    # -> parent folder is "DDoS-ICMP_Flood".
    df["label"] = csv_path.parent.name

    # source_file = "<folder>/<filename>", relative to raw_root, so it's a
    # stable identifier regardless of where the dataset is mounted.
    df["source_file"] = f"{csv_path.parent.name}/{csv_path.name}"

    # Downcast float64 -> float32. Safe for ML feature use (still ~7
    # significant decimal digits of precision); not safe if you needed the
    # values to round-trip exactly, which we don't.
    float_cols = df.select_dtypes(include="float64").columns
    df[float_cols] = df[float_cols].astype("float32")

    return df


def print_stage(name: str, df: pd.DataFrame) -> None:
    """Shape + class distribution, per CLAUDE.md section 11."""
    print(f"\n--- {name} ---")
    print(f"shape: {df.shape}")
    print("class distribution (label):")
    print(df["label"].value_counts())
    mem_gb = df.memory_usage(deep=True).sum() / 1e9
    print(f"memory (deep): {mem_gb:.2f} GB")


def build_dataset(raw_root: Path, out_path: Path) -> None:
    if out_path.exists():
        raise FileExistsError(
            f"{out_path} already exists. Delete it first if you want to "
            "re-convert — this script never overwrites silently, since "
            "re-parsing the full corpus takes several minutes."
        )

    csv_files = find_csv_files(raw_root)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {raw_root}/*/*.csv")
    print(f"Found {len(csv_files)} CSV files under {raw_root}")

    frames = []
    for i, path in enumerate(csv_files, start=1):
        df = load_and_tag_one(path, raw_root)
        frames.append(df)
        print(f"[{i}/{len(csv_files)}] {df.shape[0]:>8,} rows  {df['label'].iloc[0]:<28} <- {path.name}")

    # Stage 1: everything loaded and tagged (floats already downcast per-file
    # above — cheaper than downcasting after concatenation).
    full = pd.concat(frames, ignore_index=True)
    del frames  # free the per-file copies before we print/write
    print_stage("After load + tag + downcast (all files concatenated)", full)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out_path, index=False)

    # Stage 2: confirm what actually landed on disk, by reading it back.
    written = pd.read_parquet(out_path)
    print_stage(f"After writing {out_path}", written)
    print(f"\nParquet file size on disk: {out_path.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    build_dataset(RAW_ROOT, OUT_PATH)

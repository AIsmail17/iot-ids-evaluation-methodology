"""
features.py — Component C5: feature ranking and subset generation.

Produces the Full (39) / Reduced (~15) / Minimal (~8) feature sets that
section 9's experiment matrix treats as a factor.

The rule that shapes this module: CLAUDE.md 3.3 lists *feature selection*
alongside scalers and encoders as something that must be fitted on the
TRAINING PARTITION ONLY. Ranking features using data that includes the
test partition, then evaluating on that same test partition, is a
documented leakage path -- so this module is a set of functions that take
a training partition, NOT a script that computes one global subset and
freezes it for every run. C6 calls rank_features() inside each run.

Consequence, stated openly: the "Reduced (15)" set is not guaranteed to be
the same 15 features in every cell of the experiment matrix. That makes it
a slightly less clean controlled factor -- which is why the demo at the
bottom measures ranking STABILITY across protocols and tasks, so we can
say how much it actually varies rather than assuming it doesn't.

Ranking method: LightGBM gain importance (primary, makes the cut) plus
ANOVA F-scores (secondary, cross-check only). Permutation importance would
be more reliable but is far too expensive at 3.5M rows; mutual information
is k-NN based and slower still. Both rejected on cost, not principle.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_selection import f_classif
import lightgbm as lgb

import split       # C3
import preprocess  # C4

# Ranking is a coarse ordering -- it does not need all 3.5M training rows.
# Capping keeps the LightGBM fit to a couple of minutes. Judgment call.
RANKING_SAMPLE_CAP = 500_000

# The brief asks for "~15" and "~8" (section 9).
DEFAULT_SUBSET_SIZES = (15, 8)


def find_degenerate_features(X: pd.DataFrame, var_threshold: float = 1e-12) -> list:
    """Features with (near-)zero variance carry no information.

    Worth knowing before cutting to 8 features -- a constant column
    shouldn't occupy one of only eight slots.
    """
    variances = X.var(axis=0)
    return sorted(variances[variances <= var_threshold].index.tolist())


def find_redundant_pairs(X: pd.DataFrame, threshold: float = 0.99) -> pd.DataFrame:
    """Feature pairs with |Pearson r| >= threshold -- near-duplicates.

    Relevant to rule 3.5 ("every retained feature must be justifiable"):
    if two columns are the same signal twice, the nominal "39 features" is
    overstating how much independent information the dataset carries.
    """
    corr = X.corr(numeric_only=True).abs()
    # Take only the upper triangle so each pair is reported once, not twice.
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "feature_a", "level_1": "feature_b", 0: "abs_corr"})
    )
    return pairs[pairs["abs_corr"] >= threshold].sort_values("abs_corr", ascending=False)


def rank_features(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int = 42,
    sample_cap: int = RANKING_SAMPLE_CAP,
) -> pd.DataFrame:
    """Rank features by LightGBM gain importance, fit on TRAIN ONLY.

    Returns a DataFrame sorted best-first with columns:
      feature, lgbm_gain, lgbm_rank, f_score, f_rank
    """
    # Subsample the training partition for speed. Uses a fixed seed so the
    # ranking is reproducible for a given training partition.
    if len(X_train) > sample_cap:
        rng = np.random.default_rng(seed)
        take = rng.choice(len(X_train), size=sample_cap, replace=False)
        X_rank = X_train.iloc[take]
        y_rank = y_train[take]
    else:
        X_rank, y_rank = X_train, y_train

    n_classes = len(np.unique(y_rank))
    model = lgb.LGBMClassifier(
        n_estimators=100,
        random_state=seed,
        # Rule 3.7: imbalance is handled by weighting, not resampling. Without
        # this, importance would be dominated by the multi-million-row DDoS
        # rows and would tell us nothing about detecting rare classes.
        class_weight="balanced",
        verbose=-1,
        n_jobs=-1,
    )
    # Pass numpy rather than the DataFrame: several column names contain
    # spaces ("Protocol Type", "Tot sum"), and LightGBM is fussy about
    # characters in feature names. Positions map back to X.columns below.
    model.fit(X_rank.to_numpy(), y_rank)

    gains = model.booster_.feature_importance(importance_type="gain")

    # Secondary cross-check only -- ANOVA F detects mean differences between
    # classes (linear-ish signal). It does NOT make the cut; it's here so we
    # can see whether the LightGBM ordering is method-specific.
    f_scores, _ = f_classif(X_rank.to_numpy(), y_rank)

    ranking = pd.DataFrame({
        "feature": X_train.columns,
        "lgbm_gain": gains,
        "f_score": np.nan_to_num(f_scores),  # f_classif returns NaN for constant columns
    })
    ranking = ranking.sort_values("lgbm_gain", ascending=False).reset_index(drop=True)
    ranking["lgbm_rank"] = np.arange(1, len(ranking) + 1)
    ranking["f_rank"] = ranking["f_score"].rank(ascending=False).astype(int)
    print(f"Ranked {len(ranking)} features on {len(X_rank):,} training rows "
          f"({n_classes} classes, class_weight='balanced')")
    return ranking


def make_feature_subsets(ranking: pd.DataFrame, sizes=DEFAULT_SUBSET_SIZES) -> dict:
    """Cut the ranking into the Full / Reduced / Minimal subsets."""
    all_features = ranking["feature"].tolist()
    subsets = {f"full_{len(all_features)}": all_features}
    for size in sizes:
        subsets[f"top_{size}"] = all_features[:size]
    return subsets


def rank_for_partition(df: pd.DataFrame, protocol: str, task: str, seed: int = 42):
    """Convenience: split -> preprocess -> rank, for one experiment cell."""
    train_idx, test_idx = split.split_dataset(df, protocol=protocol, seed=seed)
    X_train, _, y_train, _, class_names, _ = preprocess.preprocess(
        df, train_idx, test_idx, task=task
    )
    ranking = rank_features(X_train, y_train, seed=seed)
    return X_train, ranking


if __name__ == "__main__":
    subsample_path = Path("data/processed/cic_iot2023_subsample.parquet")
    df = pd.read_parquet(subsample_path)
    print(f"Loaded {subsample_path}: {df.shape}")

    # --- Canonical cell: the leakage-free protocol, the headline task -----
    print("\n" + "=" * 70)
    print("Ranking on Protocol B / multiclass7 training partition")
    print("=" * 70)
    X_train_b7, ranking_b7 = rank_for_partition(df, protocol="B", task="multiclass7")

    print("\n--- Feature ranking (LightGBM gain, best first) ---")
    print(ranking_b7.to_string(index=False))

    subsets = make_feature_subsets(ranking_b7)
    print("\n--- Feature subsets ---")
    for name, feats in subsets.items():
        print(f"{name} ({len(feats)}): {feats}")

    # --- Diagnostics (rule 3.5: retained features must be justifiable) ----
    print("\n--- Degenerate (near-zero-variance) features in this training partition ---")
    degenerate = find_degenerate_features(X_train_b7)
    print(degenerate if degenerate else "None.")

    print("\n--- Near-duplicate feature pairs (|r| >= 0.99) ---")
    redundant = find_redundant_pairs(X_train_b7.sample(n=min(200_000, len(X_train_b7)), random_state=42))
    print(redundant.to_string(index=False) if len(redundant) else "None.")

    # --- Stability: does the top-15 depend on protocol or task? ----------
    print("\n" + "=" * 70)
    print("Ranking stability check")
    print("=" * 70)

    print("\n[Protocol A / multiclass7]")
    _, ranking_a7 = rank_for_partition(df, protocol="A", task="multiclass7")
    print("\n[Protocol B / binary]")
    _, ranking_bb = rank_for_partition(df, protocol="B", task="binary")

    top15_b7 = set(ranking_b7["feature"][:15])
    top15_a7 = set(ranking_a7["feature"][:15])
    top15_bb = set(ranking_bb["feature"][:15])

    print(f"\ntop-15 overlap, Protocol B vs A (same task):   "
          f"{len(top15_b7 & top15_a7)}/15")
    print(f"  only in B: {sorted(top15_b7 - top15_a7)}")
    print(f"  only in A: {sorted(top15_a7 - top15_b7)}")
    print(f"\ntop-15 overlap, multiclass7 vs binary (same protocol): "
          f"{len(top15_b7 & top15_bb)}/15")
    print(f"  only in multiclass7: {sorted(top15_b7 - top15_bb)}")
    print(f"  only in binary:      {sorted(top15_bb - top15_b7)}")

    # Save the canonical ranking so the report can cite it without re-running
    # a multi-minute LightGBM fit. Clearly named as a demo artifact -- the
    # authoritative per-run rankings get logged by C6.
    out_path = Path("results/feature_ranking_demo.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ranking_b7.to_csv(out_path, index=False)
    print(f"\nSaved canonical (Protocol B / multiclass7) ranking to {out_path}")

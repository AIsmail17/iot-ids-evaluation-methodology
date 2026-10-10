"""
bench.py — Component C7: computational cost profiling.

Measures what "lightweight IDS" papers claim and rarely measure (CLAUDE.md
section 1): artifact size, model complexity, batch=1 inference latency
percentiles, batched throughput, and memory.

CRITICAL, and the reason the imports below look odd: the thread-limit
environment variables MUST be set BEFORE numpy / sklearn / lightgbm /
xgboost are imported. Those libraries build their thread pools at import
time and read these variables exactly once. Setting them afterwards is a
no-op that silently produces multi-threaded numbers labelled as
single-threaded -- the easiest way to publish a meaningless latency figure.
"""

import os

# --- MUST come before any scientific import. Do not move below. ----------
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import time                      # noqa: E402
from pathlib import Path         # noqa: E402

import joblib                    # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402
import psutil                    # noqa: E402

import split                     # noqa: E402  (C3)
import preprocess                # noqa: E402  (C4)
import features                  # noqa: E402  (C5)

MODELS_DIR = Path("models")
BENCH_CSV = Path("results/bench.csv")

# Section 10 asks for >=10,000 single-sample inferences.
N_LATENCY = 10_000
N_WARMUP = 200          # discard: first calls pay cache/lazy-init costs
THROUGHPUT_BATCH = 100_000

# Which artifacts to profile. Latency depends on the model, the task and the
# feature count -- not meaningfully on which seed produced the artifact -- so
# one seed is profiled rather than all three. Protocol B is chosen because it
# is the methodologically valid protocol. Artifact SIZE is already recorded
# for all 216 runs in results.csv, so nothing is lost by narrowing here.
BENCH_PROTOCOL = "B"
BENCH_SEED = 42


def pin_to_single_core():
    """Pin this process to one core (section 10's constrained profile)."""
    try:
        psutil.Process().cpu_affinity([0])
        return True
    except Exception as exc:                      # not supported everywhere
        print(f"note: could not set CPU affinity ({exc}); threads still capped to 1")
        return False


def force_single_thread(model, name):
    """Override the n_jobs the model was TRAINED with.

    The sweep trained with n_jobs=-1 and joblib preserves that attribute, so
    a model loaded from disk will happily use every core unless this is
    reset -- which would invalidate the whole point of the measurement.
    """
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1
    if name == "xgboost":
        try:
            model.set_params(n_jobs=1)
            model.get_booster().set_param({"nthread": 1})
        except Exception:
            pass
    if name == "lightgbm":
        try:
            model.set_params(n_jobs=1)
        except Exception:
            pass
    return model


def model_complexity(model, name) -> dict:
    """Tree count / depth / node count, or parameter count (section 10)."""
    out = {"n_trees": np.nan, "max_depth": np.nan, "total_nodes": np.nan,
           "n_parameters": np.nan}
    try:
        if name == "decision_tree":
            out["n_trees"] = 1
            out["max_depth"] = model.tree_.max_depth
            out["total_nodes"] = model.tree_.node_count
        elif name == "random_forest":
            out["n_trees"] = len(model.estimators_)
            out["max_depth"] = max(e.tree_.max_depth for e in model.estimators_)
            out["total_nodes"] = sum(e.tree_.node_count for e in model.estimators_)
        elif name == "lightgbm":
            booster = model.booster_
            out["n_trees"] = booster.num_trees()
            dump = booster.dump_model()["tree_info"]
            # A LightGBM tree dump nests dicts; count every node reachable.
            def count(node):
                if "left_child" not in node:
                    return 1
                return 1 + count(node["left_child"]) + count(node["right_child"])
            out["total_nodes"] = sum(count(t["tree_structure"]) for t in dump)
            out["max_depth"] = max(t.get("num_leaves", 0) for t in dump)
        elif name == "xgboost":
            tdf = model.get_booster().trees_to_dataframe()
            out["n_trees"] = int(tdf["Tree"].nunique())
            out["total_nodes"] = int(len(tdf))
            out["max_depth"] = int(tdf["Node"].max())
        elif name == "logreg":
            out["n_parameters"] = int(model.coef_.size + model.intercept_.size)
        elif name == "majority":
            out["n_parameters"] = 0
    except Exception as exc:
        print(f"  note: complexity extraction failed for {name}: {exc}")
    return out


def measure_latency(model, X: np.ndarray, n=N_LATENCY) -> dict:
    """Batch=1 latency percentiles over >=n single-sample inferences.

    Each call passes ONE row, reshaped to (1, n_features) -- this measures
    the per-flow cost a gateway would actually pay, including per-call
    framework overhead, which is most of the cost for small models.
    """
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(X), size=n + N_WARMUP)
    samples = [X[i].reshape(1, -1) for i in idx]

    for s in samples[:N_WARMUP]:                  # warm-up, discarded
        model.predict(s)

    timings = np.empty(n, dtype=float)
    for i, s in enumerate(samples[N_WARMUP:]):
        t0 = time.perf_counter_ns()
        model.predict(s)
        timings[i] = time.perf_counter_ns() - t0

    return {
        "latency_p50_ms": round(np.percentile(timings, 50) / 1e6, 4),
        "latency_p95_ms": round(np.percentile(timings, 95) / 1e6, 4),
        "latency_p99_ms": round(np.percentile(timings, 99) / 1e6, 4),
        "latency_mean_ms": round(timings.mean() / 1e6, 4),
        "n_latency_samples": n,
    }


def measure_throughput(model, X: np.ndarray, batch=THROUGHPUT_BATCH) -> dict:
    """Batched flows/second -- the other end of the cost picture."""
    b = X[:batch]
    model.predict(b[:1000])                       # warm-up
    t0 = time.perf_counter()
    model.predict(b)
    elapsed = time.perf_counter() - t0
    return {
        "throughput_flows_per_s": round(len(b) / elapsed),
        "throughput_batch_size": len(b),
    }


def bench_one(run_id: str, X_test: pd.DataFrame, model_name: str) -> dict:
    """Profile one artifact."""
    path = MODELS_DIR / f"{run_id}.joblib"
    proc = psutil.Process()

    rss_before = proc.memory_info().rss
    model = joblib.load(path)
    model = force_single_thread(model, model_name)
    rss_after_load = proc.memory_info().rss

    X = X_test.to_numpy()
    row = {
        "run_id": run_id,
        "model": model_name,
        "n_features": X.shape[1],
        "artifact_bytes": path.stat().st_size,
        "artifact_mb": round(path.stat().st_size / 1e6, 3),
        "load_rss_delta_mb": round((rss_after_load - rss_before) / 1e6, 1),
    }
    row.update(model_complexity(model, model_name))
    row.update(measure_latency(model, X))
    rss_peak = proc.memory_info().rss
    row.update(measure_throughput(model, X))
    row["inference_rss_delta_mb"] = round((rss_peak - rss_after_load) / 1e6, 1)

    del model
    return row


def main():
    pinned = pin_to_single_core()
    print(f"single-core pinned: {pinned}; OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}")

    df_sub = pd.read_parquet("data/processed/cic_iot2023_subsample.parquet")
    rows = []

    # multiclass6 added 2026-09-24: the six-class scope is the one the
    # project brief specifies, so the reported cost figures must match it.
    for task in ("binary", "multiclass7", "multiclass6"):
        # Rebuild the exact partition the artifacts were trained against.
        prepared = None
        ranking = None
        for feature_set in ("full_39", "top_15", "top_8"):
            for model_name in ("majority", "logreg", "decision_tree",
                               "random_forest", "xgboost", "lightgbm"):
                run_id = f"{model_name}_{BENCH_PROTOCOL}_{task}_{feature_set}_seed{BENCH_SEED}"
                if not (MODELS_DIR / f"{run_id}.joblib").exists():
                    print(f"skip (no artifact): {run_id}")
                    continue

                if prepared is None:
                    prepared = split.split_dataset(
                        df_sub, protocol=BENCH_PROTOCOL, seed=BENCH_SEED)
                    tr_idx, te_idx = prepared
                    Xtr, Xte, ytr, yte, _, _ = preprocess.preprocess(
                        df_sub, tr_idx, te_idx, task=task)
                    prepared = (Xtr, Xte, ytr)

                Xtr, Xte, ytr = prepared
                if feature_set.startswith("full"):
                    X_use = Xte
                else:
                    # Re-derive the training-time selection. Deterministic:
                    # same partition + same seed => same ranking, so these
                    # are the columns the artifact was actually trained on.
                    if ranking is None:
                        ranking = features.rank_features(Xtr, ytr, seed=BENCH_SEED)
                    n = int(feature_set.split("_")[1])
                    X_use = Xte[ranking["feature"].tolist()[:n]]

                print(f"benchmarking {run_id} ({X_use.shape[1]} features) ...")
                r = bench_one(run_id, X_use, model_name)
                r.update({"protocol": BENCH_PROTOCOL, "task": task,
                          "feature_set": feature_set, "seed": BENCH_SEED})
                rows.append(r)
                print(f"   p50={r['latency_p50_ms']}ms p99={r['latency_p99_ms']}ms "
                      f"{r['throughput_flows_per_s']:,} flows/s  {r['artifact_mb']}MB")
        prepared = None
        ranking = None

    out = pd.DataFrame(rows)
    BENCH_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(BENCH_CSV, index=False)
    print(f"\nwrote {len(out)} rows to {BENCH_CSV}")

    print("\n" + "=" * 100)
    print("COST PROFILE (single pinned core, single thread)")
    print("=" * 100)
    show = ["model", "task", "feature_set", "artifact_mb", "n_trees", "total_nodes",
            "latency_p50_ms", "latency_p99_ms", "throughput_flows_per_s"]
    print(out[show].to_string(index=False))


if __name__ == "__main__":
    main()

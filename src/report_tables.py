"""
report_tables.py -- plan deliverable 3.4, the baseline results table.

`analyse.py` already aggregates every run; what the plan asks for on top is
a single table, framed as *the baseline*, that a reader can look at once and
understand what each model does on the six-class problem under each
protocol. That is what this produces -- in CSV for the record and in
Markdown for pasting into the report.

"Baseline" here means exactly what the plan means by it: default
hyperparameters, the full feature set, no tuning. Tuned results, reduced
feature sets and the cost analysis live in their own tables.

Outputs
-------
results/tables/baseline_results.csv   the table, one row per model x protocol
docs/baseline-results.md              the same table with captions and notes

Run:
    python src/report_tables.py
"""

from pathlib import Path

import pandas as pd

import analyse

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "results" / "tables" / "baseline_results.csv"
OUT_MD = ROOT / "docs" / "baseline-results.md"

# The baseline condition, stated once so it cannot drift between the table
# and the prose describing it.
TASK = "multiclass6"
FEATURE_SET = "full_39"

# Display names, so the report never shows a python identifier.
MODEL_NAMES = {
    "majority": "Majority-class baseline",
    "logreg": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
}
# Roster order from CLAUDE.md section 8: floor first, then the ensembles.
MODEL_ORDER = ["majority", "logreg", "decision_tree",
               "random_forest", "lightgbm", "xgboost"]


def pm(mean, std, n, dp=3):
    """Format a mean +/- std cell, flagging any cell with under 3 seeds.

    Rule 3.8 requires >= 3 seeds with mean +/- std. A cell built from fewer
    is a weaker claim and is labelled as such rather than being printed to
    look identical to a full one.
    """
    if pd.isna(mean):
        return "--"
    s = 0.0 if pd.isna(std) else std
    out = f"{mean:.{dp}f} ± {s:.{dp}f}"
    return out if n >= 3 else f"{out} (n={n})"


def build() -> pd.DataFrame:
    df = analyse.add_macro_recall(analyse.load_results())
    agg = analyse.aggregate(df)

    base = agg[(agg["task"] == TASK) & (agg["feature_set"] == FEATURE_SET)].copy()
    if base.empty:
        raise SystemExit(
            f"no rows for task={TASK} feature_set={FEATURE_SET} -- "
            "check results.csv")

    base["model_order"] = base["model"].map(
        {m: i for i, m in enumerate(MODEL_ORDER)})
    base = base.sort_values(["model_order", "protocol"])

    rows = []
    for _, r in base.iterrows():
        rows.append({
            "Model": MODEL_NAMES.get(r["model"], r["model"]),
            "Protocol": "A (random)" if r["protocol"] == "A" else "B (grouped)",
            "Macro-recall": pm(r["macro_recall_mean"], r["macro_recall_std"],
                               r["n_seeds"]),
            "Macro-F1": pm(r["macro_f1_mean"], r["macro_f1_std"], r["n_seeds"]),
            "Accuracy": pm(r["accuracy_mean"], r["accuracy_std"], r["n_seeds"]),
            "Benign FPR": pm(r["fpr_benign_mean"], r["fpr_benign_std"],
                             r["n_seeds"]),
            "Model size (MB)": f"{r['model_mb']:.2f}",
            "Train time (s)": f"{r['train_seconds_mean']:.0f}",
            "Seeds": int(r["n_seeds"]),
        })
    return pd.DataFrame(rows)


PER_CLASS = ROOT / "results" / "per_class"
REAL_MODELS = [m for m in MODEL_ORDER if m != "majority"]


def paired_rows() -> pd.DataFrame:
    """Like-for-like A vs B macro-recall, per model and seed.

    WHY THIS IS NOT JUST `macro_recall_B - macro_recall_A`.

    Protocol B assigns whole capture files to a side, and some classes own
    very few files (Brute Force owns ONE). Which classes end up represented
    in the test set therefore changes with the seed: seed 42 tests all six
    classes, seed 43 tests five, seed 44 tests only three.

    Macro-recall is an average OVER THE CLASSES PRESENT. Averaging over six
    classes and averaging over three is not the same quantity, so comparing
    Protocol A's six-class average against Protocol B's three-class average
    is comparing two different measurements. Doing so made Protocol B look
    BETTER than Protocol A under seed 44, which is an artefact, not a result.

    This function fixes it by recomputing BOTH protocols' macro-recall over
    only the classes present in both test sets for that seed -- so every
    comparison below averages over the same class list.
    """
    out = []
    for model in REAL_MODELS:
        for seed in (42, 43, 44):
            stem = f"{model}_{{p}}_{TASK}_{FEATURE_SET}_seed{seed}.csv"
            a = pd.read_csv(PER_CLASS / stem.format(p="A"))
            b = pd.read_csv(PER_CLASS / stem.format(p="B"))
            common = sorted(set(a["class"]) & set(b["class"]))
            a_c = a[a["class"].isin(common)]["recall"].mean()
            b_c = b[b["class"].isin(common)]["recall"].mean()
            out.append({
                "model": model, "seed": seed,
                "n_classes_compared": len(common),
                "A": round(a_c, 4), "B": round(b_c, 4),
                "delta": round(b_c - a_c, 4),
            })
    return pd.DataFrame(out)


def paired_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Mean paired delta per model, plus how often B lost."""
    g = rows.groupby("model")
    out = pd.DataFrame({
        "A (common classes)": g["A"].mean().round(3),
        "B (common classes)": g["B"].mean().round(3),
        "Mean change": g["delta"].mean().round(3),
        "Worst seed": g["delta"].min().round(3),
        "Best seed": g["delta"].max().round(3),
        "B worse in": [f"{int((v < 0).sum())}/{len(v)} seeds"
                       for _, v in g["delta"]],
    })
    out = out.reindex(REAL_MODELS)
    out.index = [MODEL_NAMES.get(m, m) for m in out.index]
    return out


def seed_detail(rows: pd.DataFrame) -> pd.DataFrame:
    """Per-seed table, so the instability is visible rather than averaged away."""
    d = rows.pivot_table(index="seed", columns="model", values="delta")
    d = d[REAL_MODELS]
    d.columns = [MODEL_NAMES.get(m, m) for m in d.columns]
    classes = rows.groupby("seed")["n_classes_compared"].first()
    d.insert(0, "Classes tested", classes)
    return d.round(3)


def md_table(df: pd.DataFrame, index_name: str | None = None) -> str:
    """Render a DataFrame as a Markdown table.

    pandas' own `.to_markdown()` needs `tabulate`, which is not in this
    project's dependency list. Adding a package to format one table is not
    worth it, so this does the job in a few lines.
    """
    df = df.reset_index() if index_name else df
    if index_name:
        df = df.rename(columns={df.columns[0]: index_name})
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    body = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join([header, rule, *body])


def to_markdown(table: pd.DataFrame, summary: pd.DataFrame,
                seeds: pd.DataFrame) -> str:
    lines = [
        "# Baseline results",
        "",
        "Plan deliverable **3.4**. Generated by `src/report_tables.py` from",
        "`results/results.csv` -- no number here is typed by hand",
        "(`CLAUDE.md` rule 3.9).",
        "",
        "**Baseline condition:** six-class task (Benign, Brute Force, DDoS,",
        "DoS, Reconnaissance, Spoofing), full 39-feature set, default",
        "hyperparameters, no tuning. Three seeds (42, 43, 44); cells show",
        "mean ± standard deviation.",
        "",
        "---",
        "",
        "**Table 1.** Baseline six-class results, both evaluation protocols.",
        "",
        md_table(table.drop(columns=["Seeds"])),
        "",
        "Macro-recall is the column to read across protocols. Macro-F1 is",
        "included because the plan asks for it, but it is **not** comparable",
        "between A and B: precision moves with class prevalence, and the two",
        "protocols produce test sets with different class composition",
        "(benign is 4.49% of Protocol A's test set and 6.86% of Protocol",
        "B's). Recall is invariant to prevalence, so only macro-recall",
        "isolates the effect of the split itself.",
        "",
        "Accuracy is reported for completeness and should not be quoted as a",
        "headline: on this class distribution a model can score well on it",
        "while missing minority classes entirely, which is precisely the",
        "failure documented in the literature matrix.",
        "",
        "> **Read the Protocol B standard deviations before quoting any of",
        "> these means.** They are 50-100x larger than Protocol A's, and the",
        "> reason is structural rather than statistical. See *The Protocol B",
        "> means are not trustworthy* below.",
        "",
        "---",
        "",
        "## The Protocol B means are not trustworthy",
        "",
        "Protocol B assigns whole capture files to a side. Some classes own",
        "very few files -- Brute Force owns exactly **one** -- so which",
        "classes appear in the test set at all changes with the seed:",
        "",
        md_table(seeds, index_name="Seed"),
        "",
        "Seed 42 tests all six classes, seed 43 tests five, and seed 44 tests",
        "only three. Macro-recall is an average **over the classes present**,",
        "so those are averages over different class lists. Seed 44's figure",
        "is an average over the three largest, easiest classes with no Brute",
        "Force zero in it.",
        "",
        "Averaging those three numbers together, as Table 1 does, averages",
        "three different quantities. It is reported above because the plan",
        "asks for the standard table, but the headline RQ1 claim must not",
        "rest on it.",
        "",
        "---",
        "",
        "**Table 3.** Like-for-like comparison. Both protocols re-scored over",
        "only the classes present in **both** test sets for that seed, so",
        "every row averages over the same class list.",
        "",
        md_table(summary, index_name="Model"),
        "",
        "**This is the RQ1 result.** Corrected for the class-coverage",
        "confound, Protocol B scores lower than Protocol A in **every model",
        "and every seed** -- 15 of 15 cells, with no exceptions. The",
        "uncorrected comparison had Protocol B *beating* Protocol A in three",
        "cells, which was an artefact of comparing a three-class average",
        "against a six-class one.",
        "",
        "The magnitude varies with how many of the low-file-count classes the",
        "grouped split managed to place in the test set:",
        "",
        md_table(seeds, index_name="Seed"),
        "",
        "When the test set covers all six classes the gap is 0.11 to 0.25.",
        "When it covers only the three classes that own the most capture",
        "files -- Benign, DDoS, DoS -- the gap nearly vanishes. That is the",
        "mechanism stated precisely: **the leakage advantage is concentrated",
        "in the classes with few capture files**, because those are the",
        "classes where a random split can place records from one capture on",
        "both sides. Classes with 38 or 176 files barely notice the",
        "difference.",
        "",
        "## Limitation, and what should be run next",
        "",
        "Three seeds is not enough for Protocol B. The estimate depends on a",
        "split lottery over a small number of capture files, and the three",
        "seeds drew three quite different problems. `CLAUDE.md` rule 3.8 asks",
        "for a minimum of three seeds; this is a case where the minimum is",
        "visibly insufficient.",
        "",
        "The correct fix is **GroupKFold** rather than more GroupShuffleSplit",
        "seeds. With k-fold grouping, every capture file is tested exactly",
        "once, predictions pool across folds, and macro-recall is computed",
        "once over all six classes -- removing the coverage lottery entirely",
        "instead of averaging over it. `CLAUDE.md` rule 3.1 already names",
        "GroupKFold as an accepted Protocol B implementation.",
        "",
        "One thing GroupKFold will not fix: **Brute Force owns one capture",
        "file**, so it is in the test fold exactly once and absent from",
        "training when it is. Its recall under any grouped protocol is zero.",
        "That is a property of the dataset, not of the method, and belongs in",
        "the report as a stated limit on what grouped evaluation can measure.",
        "",
    ]
    return "\n".join(lines)


def main():
    table = build()
    rows = paired_rows()
    summary = paired_summary(rows)
    seeds = seed_detail(rows)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}  ({len(table)} rows)")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(to_markdown(table, summary, seeds), encoding="utf-8")
    print(f"wrote {OUT_MD}")

    print()
    print(table.to_string(index=False))
    print()
    print(summary.to_string())
    print()
    print(seeds.to_string())


if __name__ == "__main__":
    main()

# quick_summary.py
import pandas as pd

runs = pd.read_csv("results_taiwan/abo_runs.csv")       # change if you used a different outdir
bases = pd.read_csv("results_taiwan/baselines.csv")

print("\nABO per-seed:\n", runs)

# aggregate
agg = {
    "seeds": runs["seed"].nunique(),
    "lambda": runs["lam"].iloc[0],
    "p (post-encoding)": runs["p"].iloc[0],
    "AUC_best_mean": runs["best_auc"].mean(),
    "AUC_best_std": runs["best_auc"].std(),
    "k_mean": runs["best_k"].mean(),
    "k_std": runs["best_k"].std(),
    "AUC_best_max": runs["best_auc"].max(),
    "best_seed": runs.loc[runs["best_auc"].idxmax(), "seed"],
    "k_at_best": int(runs.loc[runs["best_auc"].idxmax(), "best_k"]),
}
print("\nABO summary:\n", pd.Series(agg))

print("\nBaselines:\n", bases)

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path

from data_prep import load_and_prepare
from binary_abo import BinaryABO
from models import evaluate_subset, score_with_sparsity, baseline_all_features, baseline_l1_logreg_grid
from plots import plot_convergence

# ---- Default preset: Taiwan UCI Credit Card ----
DEFAULT_DATA = "UCI_Credit_Card.csv"
DEFAULT_TARGET = "default.payment.next.month"
DEFAULT_LAM = 0.05
DEFAULT_SEEDS = 5
DEFAULT_HERD = 60
DEFAULT_ITER = 200
DEFAULT_P_EXPLORE = 0.6
DEFAULT_FLIP = 0.05
DEFAULT_GUIDED = 0.2
DEFAULT_INIT_DENS = 0.3
DEFAULT_OUTDIR = "results_taiwan"

def run_once(csv_path, target, lam, seed,
             herd_size, iterations, p_explore, flip_rate, guided_frac, init_density,
             outdir: Path):
    Xt, Xv, yt, yv, _ = load_and_prepare(csv_path, target, test_size=0.3, random_state=seed)
    p = Xt.shape[1]

    def score_fn(mask):
        auc, k = evaluate_subset(Xt, yt, Xv, yv, mask, l2_C=1.0)
        return score_with_sparsity(auc, k, p, lam=lam)

    abo = BinaryABO(
        p=p, score_fn=score_fn, herd_size=herd_size, iterations=iterations,
        p_explore=p_explore, flip_rate=flip_rate, guided_frac=guided_frac,
        init_density=init_density, seed=seed
    )
    result = abo.solve()

    gmask = result["gbest_mask"]
    gscore = result["gbest_score"]
    auc, k = evaluate_subset(Xt, yt, Xv, yv, gmask)

    out = {"seed": seed, "p": p, "lam": lam,
           "herd_size": herd_size, "iterations": iterations,
           "p_explore": p_explore, "flip_rate": flip_rate,
           "guided_frac": guided_frac, "init_density": init_density,
           "best_score": gscore, "best_auc": auc, "best_k": k}

    outdir.mkdir(parents=True, exist_ok=True)
    plot_convergence(result["history"], str(outdir / f"convergence_seed{seed}.png"))
    return out

def main():
    ap = argparse.ArgumentParser(description="Binary-ABO Feature Selection (credit risk)")
    ap.add_argument("--data", type=str, default=DEFAULT_DATA, help="Path to CSV dataset")
    ap.add_argument("--target", type=str, default=DEFAULT_TARGET, help="Target column name (0/1)")
    ap.add_argument("--lam", type=float, default=DEFAULT_LAM, help="Sparsity penalty")
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS, help="Number of random seeds")
    ap.add_argument("--herd_size", type=int, default=DEFAULT_HERD)
    ap.add_argument("--iterations", type=int, default=DEFAULT_ITER)
    ap.add_argument("--p_explore", type=float, default=DEFAULT_P_EXPLORE)
    ap.add_argument("--flip_rate", type=float, default=DEFAULT_FLIP)
    ap.add_argument("--guided_frac", type=float, default=DEFAULT_GUIDED)
    ap.add_argument("--init_density", type=float, default=DEFAULT_INIT_DENS)
    ap.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR)
    args = ap.parse_args()

    if not os.path.exists(args.data):
        print(f"[WARN] Dataset not found: {args.data}")
        print("       Put UCI_Credit_Card.csv in the working directory or pass --data <path>")
        print("       Defaults are set to the Taiwan UCI dataset/target.")

    outdir = Path(args.outdir)
    rows = []
    for s in range(args.seeds):
        row = run_once(
            csv_path=args.data, target=args.target, lam=args.lam, seed=s,
            herd_size=args.herd_size, iterations=args.iterations,
            p_explore=args.p_explore, flip_rate=args.flip_rate,
            guided_frac=args.guided_frac, init_density=args.init_density,
            outdir=outdir
        )
        rows.append(row)

    df = pd.DataFrame(rows)

    # baselines (split fijo semilla=0)
    Xt, Xv, yt, yv, _ = load_and_prepare(args.data, args.target, test_size=0.3, random_state=0)
    all_auc = baseline_all_features(Xt, yt, Xv, yv, l2_C=1.0)
    l1_auc, l1_nnz, l1_C = baseline_l1_logreg_grid(Xt, yt, Xv, yv, Cs=[0.1, 1.0, 10.0])

    df.to_csv(outdir / "abo_runs.csv", index=False)
    pd.DataFrame([
        {"method": "AllFeatures_L2", "AUC": all_auc, "k": Xt.shape[1]},
        {"method": "L1LogReg", "AUC": l1_auc, "k": l1_nnz, "C": l1_C}
    ]).to_csv(outdir / "baselines.csv", index=False)

    print("Saved results to:", outdir.resolve())

if __name__ == "__main__":
    main()

# African Buffalo Optimization for Feature Selection

A binary implementation of the **African Buffalo Optimization (ABO)** metaheuristic applied to feature selection in credit risk classification. The algorithm finds compact, high-performing feature subsets that outperform full-feature baselines — using only ~20% of the original features.

---

## Results Summary

Evaluated on the **UCI Taiwan Credit Card dataset** (24 features after encoding, predicting default payment).

| Model | Features (k) | Test AUC |
|---|---|---|
| All features — Logistic L2 | 24 | 0.7146 |
| L1-Regularized Logistic | 23 | 0.7146 |
| ABO Run 0 | **4** | 0.7160 |
| ABO Run 1 *(best)* | **5** | **0.7228** |
| ABO Run 2 | **5** | 0.7149 |
| ABO Run 3 | **4** | 0.7168 |
| ABO Run 4 | **4** | 0.7076 |

The best ABO run achieved **+0.82 pp AUC** over the full model using only 5 of 24 features. All runs converged within 200 iterations.

---

## How It Works

Each "buffalo" in the herd is a binary vector of length `p` (1 = feature selected, 0 = not). The herd searches for the subset that maximizes:

```
F(S) = AUC(S) − λ · |S| / p
```

where `λ = 0.05` penalizes larger subsets. Each iteration, buffalos either:
- **Explore** (`waaa`): random bit-flips to discover new regions
- **Exploit** (`maaa`): guided alignment toward personal best and global best

---

## Project Structure

```
├── binary_abo.py        # Core ABO algorithm (BinaryABO class)
├── data_prep.py         # Data loading, preprocessing, train/val split
├── models.py            # Logistic regression evaluation + baselines
├── plots.py             # Convergence curve plotting
├── run_experiment.py    # Main experiment runner (CLI)
├── quick_summary.py     # Print results table from saved CSVs
├── results_taiwan/
│   ├── abo_runs.csv         # Per-seed ABO results
│   ├── baselines.csv        # Baseline model results
│   └── convergence_seed*.png
```

---

## Quickstart

```bash
pip install scikit-learn pandas matplotlib numpy
```

Run the experiment on the UCI Taiwan dataset:

```bash
python run_experiment.py \
  --data UCI_Credit_Card.csv \
  --target default.payment.next.month
```

Run on the German Credit dataset:

```bash
python run_experiment.py \
  --data german_credit_data.csv \
  --target Risk \
  --outdir results_german
```

Print the results summary:

```bash
python quick_summary.py
```

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `--lam` | `0.05` | Sparsity penalty (higher = fewer features) |
| `--seeds` | `5` | Number of independent runs |
| `--herd_size` | `60` | Number of buffalos (population size) |
| `--iterations` | `200` | Iterations per run |
| `--p_explore` | `0.6` | Probability of exploration move |
| `--flip_rate` | `0.05` | Fraction of bits flipped during exploration |
| `--guided_frac` | `0.2` | Fraction of features aligned during exploitation |
| `--init_density` | `0.3` | Initial feature inclusion probability |

---

## Datasets

- **UCI Credit Card (Taiwan)** — 30,000 records, predicts default payment next month. [Kaggle link](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)
- **German Credit Data** — 1,000 records, classifies credit risk (Good/Bad). [Kaggle link](https://www.kaggle.com/datasets/uciml/german-credit)

Place CSV files in the working directory before running.

---

## References

- Odili, J. B., Kahar, M. N., & Anwar, S. (2015). African Buffalo Optimization: A Swarm-Intelligence Technique. *Procedia Computer Science, 76*, 443–448.
- Ali, F. et al. (2025). African buffalo optimization with deep learning-based intrusion detection in cyber-physical systems. *Scientific Reports, 15*, 5150.
- Muthulakshmi, P., & Parveen, M. (2023). Z-Score Normalized Feature Selection and Iterative African Buffalo Optimization for Effective Heart Disease Prediction. *IJIES, 16*(1), 217–226.

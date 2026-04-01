import numpy as np
from typing import List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def evaluate_subset(X_train, y_train, X_valid, y_valid, mask, l2_C: float = 1.0):
    idx = np.where(mask == 1)[0]
    k = len(idx)
    if k == 0:
        return 0.5, 0
    clf = LogisticRegression(penalty="l2", C=l2_C, solver="liblinear", max_iter=1000)
    clf.fit(X_train[:, idx], y_train)
    proba = clf.predict_proba(X_valid[:, idx])[:, 1]
    auc = roc_auc_score(y_valid, proba)
    return float(auc), k

def score_with_sparsity(auc: float, k: int, p: int, lam: float = 0.05) -> float:
    return auc - lam * (k / max(1, p))

def baseline_all_features(X_train, y_train, X_valid, y_valid, l2_C: float = 1.0):
    clf = LogisticRegression(penalty="l2", C=l2_C, solver="liblinear", max_iter=1000)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_valid)[:, 1]
    return float(roc_auc_score(y_valid, proba))

def baseline_l1_logreg_grid(X_train, y_train, X_valid, y_valid, Cs: List[float] = [0.1,1.0,10.0]) -> Tuple[float,int,float]:
    best = (-1.0, 0, None)
    for C in Cs:
        clf = LogisticRegression(penalty="l1", C=C, solver="liblinear", max_iter=2000)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_valid)[:, 1]
        auc = float(roc_auc_score(y_valid, proba))
        nnz = int((np.abs(clf.coef_) > 1e-8).sum())
        if auc > best[0]:
            best = (auc, nnz, C)
    return best

"""Pair-level ROC evaluation and the frozen 0.05-grid fusion rule."""
import numpy as np


def roc(scores, labels):
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels, dtype=int)
    if scores.ndim != 1 or labels.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Scores must be a finite vector with matching binary labels")
    if not np.isin(labels, [0, 1]).all() or set(labels.tolist()) != {0, 1}:
        raise ValueError("ROC evaluation requires both positive and negative pairs")
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], labels[order]
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    return np.r_[0.0, np.cumsum(1 - y)[last] / (1 - y).sum()], \
        np.r_[0.0, np.cumsum(y)[last] / y.sum()]


def metrics(scores, labels):
    fpr, tpr = roc(scores, labels)
    return {"tpr1": float(np.interp(0.01, fpr, tpr)),
            "tpr01": float(np.interp(0.001, fpr, tpr)), "auroc": float(np.trapezoid(tpr, fpr))}


def fit(hash_scores, dense_scores, labels):
    h, d, y = np.asarray(hash_scores, float), np.asarray(dense_scores, float), np.asarray(labels, int)
    if h.shape != d.shape:
        raise ValueError("Hash and dense scores must have matching shapes")
    best = None
    for alpha in np.round(np.arange(0, 1.0001, 0.05), 2):
        score = alpha * h + (1 - alpha) * d
        result = metrics(score, y)
        key = (result["tpr1"], result["auroc"], -alpha)
        if best is None or key > best[0]:
            best = (key, float(alpha), score)
    negatives = np.sort(best[2][y == 0])
    threshold = negatives[int(np.ceil(0.999 * len(negatives))) - 1]
    return {"alpha": best[1], "threshold_tau": float(threshold), "pair_fpr_target": 0.001}


def fit_pairs(path, dense_key="D_B1"):
    with np.load(path, allow_pickle=False) as pairs:
        if "role" not in pairs:
            raise ValueError("NPZ requires a per-pair 'role' array with fit/test values")
        role = pairs["role"]
        selected = role == "fit"
        result = fit(pairs["H"][selected], pairs[dense_key][selected], pairs["label"][selected])
        test = role == "test"
        score = result["alpha"] * pairs["H"].astype(float) + (1 - result["alpha"]) * pairs[dense_key].astype(float)
        result["held_out_verification"] = metrics(score[test], pairs["label"][test])
    return result

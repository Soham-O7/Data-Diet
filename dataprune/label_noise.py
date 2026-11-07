"""Stage 3: label-noise removal by confident learning (Northcutt et al., 2021) with out-of-fold probabilities.

Own implementation of the 'confident joint off-diagonal' rule (cleanlab's filter_by='confident_learning'):
  t_j  = mean predicted probability of class j over samples labelled j
  a sample is a label issue if its most-likely class among {j : p_j >= t_j} differs from its given label.
If `cleanlab` is installed and requested, its find_label_issues() supplies the candidate set instead
(that path is NOT exercised in the build sandbox - cleanlab is not installed there).
Candidates are ranked by margin (self-confidence minus confidence in the suggested class) and removal is capped
globally and per class, so a bad proxy cannot wipe out the data.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold


def oof_probabilities(
    Z: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    model_factory: Callable[[int], object],
    folds: int = 5,
    seed: int = 0,
) -> np.ndarray:
    n = len(y)
    counts = np.bincount(y, minlength=n_classes)
    present = counts[counts > 0]
    folds = int(max(2, min(folds, present.min() if len(present) else 2, n)))
    if present.min() >= folds:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = splitter.split(Z, y)
    else:
        splits = KFold(n_splits=folds, shuffle=True, random_state=seed).split(Z)
    P = np.zeros((n, n_classes), dtype=np.float64)
    for tr, va in splits:
        model = model_factory(seed)
        model.fit(Z[tr], y[tr])
        proba = model.predict_proba(Z[va])
        P[np.ix_(va, np.asarray(model.classes_, dtype=int))] = proba
    return P


def find_label_issues(
    P: np.ndarray,
    y: np.ndarray,
    max_remove_frac: float = 0.10,
    max_remove_frac_per_class: float = 0.30,
    use_cleanlab: bool = False,
):
    """Return (issue_mask, info). issue_mask marks rows to REMOVE."""
    n, C = P.shape
    idx = np.arange(n)
    self_conf = P[idx, y]

    backend = "internal_confident_learning"
    cand_mask = None
    if use_cleanlab:
        try:
            from cleanlab.filter import find_label_issues as cl_find  # type: ignore

            cand_mask = np.asarray(cl_find(labels=y, pred_probs=P), dtype=bool)
            backend = "cleanlab"
        except Exception:  # not installed / API mismatch -> fall back silently to the internal rule
            cand_mask = None

    # suggested label = most likely class among those above their per-class threshold
    t = np.array([P[y == c, c].mean() if np.any(y == c) else 1.0 for c in range(C)])
    above = P >= t[None, :]
    masked = np.where(above, P, -np.inf)
    cand_label = masked.argmax(axis=1)
    has = above.any(axis=1)
    if cand_mask is None:
        cand_mask = has & (cand_label != y)
    # margin of the suggested class over the given one; for cleanlab candidates fall back to argmax class
    sugg = np.where(has, cand_label, P.argmax(axis=1))
    margin = self_conf - P[idx, sugg]          # more negative = more likely mislabelled

    cand_idx = np.flatnonzero(cand_mask & (sugg != y))
    cand_idx = cand_idx[np.argsort(margin[cand_idx], kind="stable")]

    cap_total = int(np.floor(max_remove_frac * n))
    class_n = np.bincount(y, minlength=C)
    cap_cls = np.floor(max_remove_frac_per_class * class_n).astype(int)
    removed_cls = np.zeros(C, dtype=int)
    chosen = []
    for i in cand_idx:
        if len(chosen) >= cap_total:
            break
        c = y[i]
        if removed_cls[c] >= cap_cls[c]:
            continue
        removed_cls[c] += 1
        chosen.append(i)
    mask = np.zeros(n, dtype=bool)
    mask[np.array(chosen, dtype=int)] = True
    info = {
        "backend": backend,
        "n_candidates": int(len(cand_idx)),
        "n_removed": int(mask.sum()),
        "capped": bool(len(cand_idx) > mask.sum()),
        "per_class_removed": removed_cls.tolist(),
    }
    return mask, info, margin

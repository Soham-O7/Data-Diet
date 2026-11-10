"""Reports built from numbers the pipeline already produces: removal export, class warnings, health score,
keep-ratio recommender."""
from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .config import PruneConfig

_REASON = {
    "1_hash": ("exact or perceptual duplicate of an earlier row with the same label", None),
    "2_semdedup": ("near-duplicate of an earlier same-class row", "max_cosine_to_earlier_item"),
    "3_label_noise": ("label looks wrong (out-of-fold model disagrees)", "margin_given_minus_suggested"),
    "5_selection": ("not selected to fit the keep budget", "difficulty (higher = harder)"),
}


def removal_export(result, X=None) -> pd.DataFrame:
    """One row per dropped row: original_index, label, stage, reason, score, score_meaning.

    `score` is stage-specific: SemDeDup = max cosine similarity to an earlier kept-side item; label noise = margin
    (given-label confidence minus suggested-label confidence; more negative = more likely mislabelled); selection =
    difficulty score. Hash removals have no score. Pass the original X (DataFrame) to append its columns.
    """
    rows = []
    for stage, idx in result.removed_by_stage.items():
        reason, meaning = _REASON.get(stage, (stage, None))
        det = result.removal_details.get(stage, {})
        for i in np.asarray(idx, dtype=int):
            rows.append({"original_index": int(i), "label": result.classes_[result.y_int[i]], "stage": stage,
                         "reason": reason, "score": det.get(int(i), np.nan), "score_meaning": meaning})
    cols = ["original_index", "label", "stage", "reason", "score", "score_meaning"]
    df = pd.DataFrame(rows, columns=cols).sort_values("original_index").reset_index(drop=True)
    if isinstance(X, pd.DataFrame) and len(df):
        feats = X.iloc[df["original_index"].to_numpy()].reset_index(drop=True)
        df = pd.concat([df, feats], axis=1)
    return df


def class_warnings(result, min_ratio: float = 0.5) -> list:
    """Plain-language alerts when a class loses much more than its share (or vanishes)."""
    y = result.y_int
    n_cls = len(result.classes_)
    full = np.bincount(y, minlength=n_cls)
    kept = np.bincount(y[result.keep_indices], minlength=n_cls)
    overall = len(result.keep_indices) / max(len(y), 1)
    out = []
    for c in range(n_cls):
        if full[c] == 0:
            continue
        name = result.classes_[c]
        frac = kept[c] / full[c]
        if kept[c] == 0:
            out.append(f"Class '{name}' has NO rows left (it had {full[c]}).")
        elif overall < 1.0 and frac < min_ratio * overall:
            out.append(f"Class '{name}' kept only {kept[c]} of {full[c]} rows ({frac:.0%}) while the dataset as a "
                       f"whole kept {overall:.0%}; it may be under-represented in the pruned data.")
        elif kept[c] < 10 and full[c] >= 10:
            out.append(f"Class '{name}' is down to {kept[c]} rows; models may struggle to learn it.")
    return out


def health_score(X: pd.DataFrame, y) -> dict:
    """0-100 data-quality score from duplicates, missing cells, class imbalance and label conflicts."""
    y = np.asarray(y)
    n = len(y)
    if n == 0:
        return {"score": 0.0, "n_rows": 0}
    dup_rows = int(pd.concat([X.reset_index(drop=True), pd.Series(y, name="__y")], axis=1).duplicated().sum())
    missing = int(X.isna().sum().sum())
    cells = max(X.size, 1)
    classes, counts = np.unique(y, return_counts=True)
    imbalance = float(counts.max() / counts.min())
    # rows identical in X but with different labels
    conflict = int(X.assign(__y=y).groupby(list(X.columns), dropna=False)["__y"].nunique().gt(1).sum()) \
        if X.shape[1] else 0
    score = 100.0
    score -= 40.0 * dup_rows / n
    score -= 30.0 * missing / cells
    score -= 20.0 * conflict / n
    score -= min(10.0, 2.5 * np.log10(imbalance) * 2) if imbalance > 1 else 0.0
    return {"score": round(max(0.0, min(100.0, score)), 1), "n_rows": n, "duplicate_rows": dup_rows,
            "missing_cells": missing, "missing_fraction": missing / cells, "label_conflict_groups": conflict,
            "n_classes": int(len(classes)), "imbalance_ratio": imbalance,
            "weights_note": "40*dup_frac + 30*missing_frac + 20*conflict_frac + up to 10 for imbalance"}


def recommend_keep_ratio(
    X, y, config: Optional[PruneConfig] = None,
    ratios: Sequence[float] = (0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1),
    tolerance: float = 0.02, eval_seeds: Sequence[int] = (0, 1, 2), **experiment_kwargs,
) -> dict:
    """Sweep keep ratios from large to small with the real prune + proof pipeline and stop at the first ratio whose
    accuracy retained vs full drops by more than `tolerance`. Recommendation = smallest ratio that still passed.

    Uses the same held-out test split for every ratio (same seed). Costs one full prune + proof run per ratio tried.
    """
    from .experiment import run_experiment

    base = config or PruneConfig()
    rows, rec = [], None
    for r in sorted(ratios, reverse=True):
        out = run_experiment(X, y, replace(base, keep_ratio=float(r)), eval_seeds=eval_seeds, **experiment_kwargs)
        v = out["verdict"]
        ok = v["pruned_retained_vs_full"] >= 1.0 - tolerance
        rows.append({"keep_ratio": float(r), "retained_vs_full": v["pruned_retained_vs_full"],
                     "random_stratified_retained": v["random_stratified_retained_vs_full"],
                     "worst_class_recall_drop": v["worst_per_class_recall_drop_vs_full"], "within_tolerance": ok,
                     "n_kept": int(len(out["result"].keep_indices))})
        if not ok:
            break
        rec = float(r)
    return {"recommended_keep_ratio": rec, "tolerance": tolerance, "sweep": pd.DataFrame(rows),
            "note": ("None = even the largest ratio tried lost more than the tolerance." if rec is None else
                     "Stopped at the first ratio that failed; based on a single held-out split, so small gaps "
                     "are noisy - confirm with run_experiment.")}

"""'Selector race': run each candidate selector on a small probe budget and pick the winner automatically.

Leakage rules: the probe is a stratified sample of the cleaned TRAINING data, split 75/25 into probe-train / probe-val.
Difficulty scores are recomputed on probe-train only; selectors pick `keep_ratio` of probe-train; a fresh model is
trained on each pick and scored on probe-val. The real test set is never touched.
Random selection is always scored as a reference. `beats_random` tells you whether the winner actually beat it on
the probe (with few repeats that can be noise - treat small gaps as ties).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

from .difficulty import score_difficulty
from .quotas import allocate_quotas, resolve_mode
from .selection import select_per_class


def _metric_fn(name: str):
    return balanced_accuracy_score if name == "balanced_accuracy" else accuracy_score


def run_race(
    Z: np.ndarray,
    geom: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    keep_ratio: float,
    cfg,
    modality: str,
    model_factory: Callable[[int], object],
    quota_mode: str,
    scorer_kwargs: Optional[dict] = None,
):
    rng = np.random.default_rng(cfg.seed + 7)
    n = len(y)
    counts = np.bincount(y, minlength=n_classes)
    nz = counts[counts > 0]
    imb = nz.max() / nz.min() if len(nz) else 1.0
    metric_name = cfg.race_metric
    if metric_name == "auto":
        metric_name = "balanced_accuracy" if imb >= cfg.imbalance_auto_ratio else "accuracy"
    metric = _metric_fn(metric_name)

    probe_n = int(min(cfg.race_probe_size, n))
    all_idx = np.arange(n)
    strat = y if counts[counts > 0].min() >= 2 else None
    if probe_n < n:
        probe_idx, _ = train_test_split(all_idx, train_size=probe_n, stratify=strat, random_state=cfg.seed)
    else:
        probe_idx = all_idx
    ystrat = y[probe_idx]
    pc = np.bincount(ystrat, minlength=n_classes)
    n_val = int(np.ceil(0.25 * len(probe_idx)))
    strat2 = ystrat if (pc[pc > 0].min() >= 2 and n_val >= int((pc > 0).sum())) else None   # tiny probe: unstratified
    tr_idx, va_idx = train_test_split(probe_idx, test_size=0.25, stratify=strat2, random_state=cfg.seed)

    Ztr, ytr, gtr = Z[tr_idx], y[tr_idx], geom[tr_idx]
    Zva, yva = Z[va_idx], y[va_idx]

    ds = score_difficulty(Ztr, ytr, n_classes, modality=modality, proxy=cfg.proxy,
                          epochs=cfg.proxy_epochs, seeds=max(1, cfg.proxy_seeds - 1), seed=cfg.seed,
                          el2n_epoch_frac=cfg.el2n_epoch_frac, metric=cfg.difficulty_metric)

    tr_counts = {c: int((ytr == c).sum()) for c in range(n_classes)}
    K = int(round(keep_ratio * len(ytr)))
    K = max(K, min(len(ytr), n_classes))
    mode = resolve_mode(quota_mode, tr_counts, cfg.imbalance_auto_ratio)
    quotas = allocate_quotas(tr_counts, tr_counts, K, mode, cfg.min_per_class)

    names = list(cfg.race_candidates) + ["random"]
    rows = []
    for r in range(cfg.race_repeats):
        for name in names:
            rr = np.random.default_rng(cfg.seed + 100 * r + 13)
            sel = select_per_class(name, ytr, quotas, ds.difficulty, ds.el2n, gtr, cfg, rr)
            model = model_factory(cfg.seed + r)
            model.fit(Ztr[sel], ytr[sel])
            score = float(metric(yva, model.predict(Zva)))
            rows.append({"selector": name, "repeat": r, "score": score, "n_selected": int(len(sel))})
    table = pd.DataFrame(rows)
    summary = table.groupby("selector")["score"].agg(["mean", "std"]).reset_index()
    summary["std"] = summary["std"].fillna(0.0)
    cand = summary[summary["selector"].isin(cfg.race_candidates)].copy()
    cand["order"] = cand["selector"].map({n_: i for i, n_ in enumerate(cfg.race_candidates)})
    cand = cand.sort_values(["mean", "order"], ascending=[False, True])
    winner = str(cand.iloc[0]["selector"])
    rand_mean = float(summary.loc[summary["selector"] == "random", "mean"].iloc[0])
    info = {
        "winner": winner,
        "metric": metric_name,
        "probe_train": int(len(tr_idx)),
        "probe_val": int(len(va_idx)),
        "quota_mode": mode,
        "random_reference": rand_mean,
        "winner_score": float(cand.iloc[0]["mean"]),
        "beats_random": bool(float(cand.iloc[0]["mean"]) > rand_mean),
        "summary": summary.to_dict(orient="records"),
    }
    return winner, info, table

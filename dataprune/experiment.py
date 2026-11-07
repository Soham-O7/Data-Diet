"""End-to-end experiment: split -> prune (training part only) -> three-way proof run on the held-out test part."""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import PruneConfig
from .models import make_eval_model
from .pipeline import PruneResult, prune, _detect_modality
from .verify import proof_run


def _take(X, idx):
    return X.iloc[idx].reset_index(drop=True) if isinstance(X, pd.DataFrame) else np.asarray(X)[idx]


def run_experiment(
    X,
    y,
    config: Optional[PruneConfig] = None,
    modality: Optional[str] = None,
    test_size: float = 0.2,
    eval_seeds: Sequence[int] = (0, 1, 2),
    X_test=None,
    y_test=None,
    embedder=None,
    model_factory: Optional[Callable[[int], object]] = None,
):
    """Returns dict(result, table, raw, verdict). The test set is split off BEFORE any pruning stage runs."""
    cfg = config or PruneConfig()
    modality = modality or _detect_modality(X)
    y = np.asarray(y)
    if X_test is None:
        idx = np.arange(len(y))
        _, cnt = np.unique(y, return_counts=True)
        strat = y if cnt.min() >= 2 else None
        tr, te = train_test_split(idx, test_size=test_size, stratify=strat, random_state=cfg.seed)
        X_train, y_train = _take(X, tr), y[tr]
        X_test, y_test = _take(X, te), y[te]
    else:
        X_train, y_train = X, y
        y_test = np.asarray(y_test)

    result: PruneResult = prune(X_train, y_train, cfg, modality=modality, embedder=embedder)
    lut = {c: i for i, c in enumerate(result.classes_)}
    unseen = set(np.unique(y_test)) - set(lut)
    if unseen:
        raise ValueError(f"test set has classes never seen in training data: {sorted(unseen)}")
    y_test_int = np.array([lut[c] for c in y_test], dtype=int)

    Z_all = result.transform(X_train)
    Z_test = result.transform(X_test)
    factory = model_factory or (lambda s: make_eval_model(modality, s))
    table, raw, verdict = proof_run(
        Z_all, result.y_int, result.keep_indices, Z_test, y_test_int, len(result.classes_), cfg, factory,
        eval_seeds, pruning_overhead_kwh=result.energy["total_kwh_est"],
    )
    return {"result": result, "table": table, "raw": raw, "verdict": verdict}

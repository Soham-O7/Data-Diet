"""Stage 4: difficulty scoring from proxy-model training dynamics.

A cheap proxy is trained for E epochs / boosting rounds; after every epoch the class probabilities on the TRAINING
set are recorded (Swayamdipta et al. 'data maps'; Paul et al. EL2N; Toneva et al. forgetting). Per sample:
  confidence   mean p(true label) over epochs           (low  = hard)
  variability  std  of p(true label) over epochs        (high = ambiguous)
  el2n         mean ||p - onehot|| over the FIRST `el2n_epoch_frac` of epochs   (high = hard)
  loss         mean cross-entropy over epochs           (high = hard)
  forget       forgetting events (correct -> incorrect) normalised by epochs; never-learned samples get 1.0
Everything is averaged over `seeds` proxy runs. 'difficulty' (higher = harder) is the metric the selectors use.
Proxy kinds: 'hgb' (gradient boosting, staged predictions) or 'mlp' (sklearn MLP, one partial_fit per epoch).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


@dataclass
class DifficultyScores:
    el2n: np.ndarray
    loss: np.ndarray
    forget: np.ndarray
    confidence: np.ndarray
    variability: np.ndarray
    difficulty: np.ndarray
    metric: str

    def as_frame(self):
        import pandas as pd

        return pd.DataFrame({"el2n": self.el2n, "loss": self.loss, "forget": self.forget,
                             "confidence": self.confidence, "variability": self.variability,
                             "difficulty": self.difficulty})


def default_proxy(modality: str) -> str:
    return "hgb" if modality == "tabular" else "mlp"


def default_metric(modality: str) -> str:
    return "datamap" if modality == "tabular" else "combined"


def _pct_rank(a: np.ndarray) -> np.ndarray:
    return (rankdata(a, method="average") - 1.0) / max(len(a) - 1, 1)


def _epoch_probs(Z: np.ndarray, y: np.ndarray, n_classes: int, proxy: str, epochs: int, seed: int):
    """Yield (n, C) probability matrices on the training set, one per epoch."""
    if proxy == "hgb":
        clf = HistGradientBoostingClassifier(
            max_iter=epochs, learning_rate=0.1, max_leaf_nodes=15, max_features=0.8,
            early_stopping=False, random_state=seed,
        ).fit(Z, y)
        cls = np.asarray(clf.classes_, dtype=int)
        for P in clf.staged_predict_proba(Z):
            full = np.zeros((len(Z), n_classes), dtype=np.float32)
            full[:, cls] = P
            yield full
    else:
        Zs = StandardScaler().fit_transform(Z)
        mlp = MLPClassifier(hidden_layer_sizes=(128,), batch_size=256, learning_rate_init=1e-3,
                            random_state=seed)
        classes = np.arange(n_classes)
        for _ in range(epochs):
            mlp.partial_fit(Zs, y, classes=classes)
            yield mlp.predict_proba(Zs).astype(np.float32)


def score_difficulty(
    Z: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    modality: str = "tabular",
    proxy: Optional[str] = None,
    epochs: int = 20,
    seeds: int = 3,
    seed: int = 0,
    el2n_epoch_frac: float = 0.5,
    metric: Optional[str] = None,
) -> DifficultyScores:
    proxy = proxy or default_proxy(modality)
    metric = metric or default_metric(modality)
    n = len(y)
    idx = np.arange(n)
    onehot = np.zeros((n, n_classes), dtype=np.float32)
    onehot[idx, y] = 1.0
    n_el2n = max(1, int(round(epochs * el2n_epoch_frac)))

    acc = {k: np.zeros(n, dtype=np.float64) for k in ("el2n", "loss", "forget", "confidence", "variability")}
    for s in range(seeds):
        p_true = np.zeros((epochs, n), dtype=np.float32)
        correct = np.zeros((epochs, n), dtype=bool)
        el2n_e = np.zeros((epochs, n), dtype=np.float32)
        e_done = 0
        for e, P in enumerate(_epoch_probs(Z, y, n_classes, proxy, epochs, seed + 1000 * s)):
            if e >= epochs:
                break
            p_true[e] = P[idx, y]
            correct[e] = P.argmax(1) == y
            el2n_e[e] = np.linalg.norm(P - onehot, axis=1)
            e_done = e + 1
        if e_done < epochs:  # proxy produced fewer rounds than asked (should not happen) -> trim
            p_true, correct, el2n_e = p_true[:e_done], correct[:e_done], el2n_e[:e_done]
        E = len(p_true)
        acc["confidence"] += p_true.mean(0)
        acc["variability"] += p_true.std(0)
        acc["el2n"] += el2n_e[:max(1, min(n_el2n, E))].mean(0)
        acc["loss"] += -np.log(np.clip(p_true, 1e-7, 1.0)).mean(0)
        trans = (correct[:-1] & ~correct[1:]).sum(0).astype(np.float64) if E > 1 else np.zeros(n)
        never = ~correct.any(0)
        forget = trans / max(E - 1, 1)
        forget[never] = 1.0
        acc["forget"] += forget
    for k in acc:
        acc[k] /= seeds

    if metric == "datamap":
        difficulty = 1.0 - acc["confidence"]
    elif metric == "el2n":
        difficulty = acc["el2n"]
    elif metric == "loss":
        difficulty = acc["loss"]
    elif metric == "forget":
        difficulty = acc["forget"]
    elif metric == "combined":
        difficulty = 0.5 * (_pct_rank(acc["el2n"]) + _pct_rank(acc["forget"]))
    else:
        raise ValueError(f"unknown difficulty metric {metric!r}")
    return DifficultyScores(
        el2n=acc["el2n"], loss=acc["loss"], forget=acc["forget"], confidence=acc["confidence"],
        variability=acc["variability"], difficulty=difficulty, metric=metric,
    )

"""Model factories used for (a) out-of-fold label-noise probabilities and (b) evaluation / selector race / proof run.

Tabular -> HistGradientBoosting. Images (on embeddings) -> standardised multinomial logistic regression.
Any object with sklearn's fit / predict / predict_proba can be swapped in through the `model_factory` arguments.
"""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_eval_model(modality: str, seed: int = 0):
    if modality == "tabular":
        return HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, early_stopping=False,
                                              random_state=seed)
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, random_state=seed))


def make_oof_model(modality: str, seed: int = 0):
    return make_eval_model(modality, seed)

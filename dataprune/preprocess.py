"""Tabular preprocessing: 'the columns themselves' (encode categories, scale numbers). No embedding."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replace a high-cardinality categorical by its training frequency (unseen -> 0)."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.maps_ = [X[c].value_counts(normalize=True).to_dict() for c in X.columns]
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        out = np.zeros((len(X), X.shape[1]), dtype=np.float64)
        for j, c in enumerate(X.columns):
            out[:, j] = X[c].map(self.maps_[j]).fillna(0.0).to_numpy(dtype=np.float64)
        return out


class TabularPreprocessor:
    def __init__(self, max_onehot_cardinality: int = 30):
        self.max_onehot_cardinality = max_onehot_cardinality
        self.ct_: ColumnTransformer | None = None

    def _prep_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.cat_cols_:
            col = X[c].astype(object)
            X[c] = col.where(col.notna(), np.nan)
        for c in self.num_cols_:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(np.float64)
        return X

    def fit(self, X: pd.DataFrame) -> "TabularPreprocessor":
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])]
        cat = [c for c in X.columns if c not in num]
        low = [c for c in cat if X[c].nunique(dropna=True) <= self.max_onehot_cardinality]
        high = [c for c in cat if c not in low]
        self.num_cols_, self.cat_cols_ = num, cat
        Xp = self._prep_frame(X)
        parts = []
        if num:
            parts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                           ("sc", StandardScaler())]), num))
        if low:
            parts.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                           ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), low))
        if high:
            parts.append(("hi", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                          ("fe", FrequencyEncoder()),
                                          ("sc", StandardScaler())]), high))
        if not parts:
            raise ValueError("no usable feature columns")
        self.ct_ = ColumnTransformer(parts, remainder="drop")
        self.ct_.fit(Xp)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.columns_)
        missing = [c for c in self.columns_ if c not in X.columns]
        if missing:
            raise ValueError(f"missing columns at transform time: {missing}")
        Z = self.ct_.transform(self._prep_frame(X[self.columns_]))
        return np.asarray(Z, dtype=np.float32)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

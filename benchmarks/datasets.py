"""Benchmark datasets.

Runnable offline (bundled with scikit-learn):
  synth_tabular_corrupted  imbalanced synthetic tabular, TRAIN split corrupted with label flips + duplicates
  digits_corrupted         sklearn digits (8x8 'images'), TRAIN split corrupted with flips + exact + near duplicates
  digits_clean             sklearn digits as-is
  breast_cancer            real tabular (569 rows - tiny, expect noisy results)

Need a download (NOT run in the sandbox this was written in - loaders provided for your machine):
  adult      OpenML 'adult' (CSV-style, the plan's tabular benchmark)
  mnist      OpenML 'mnist_784'
  cifar10    torchvision CIFAR10 (needs torch + torchvision)

The TEST split is always clean and is split off before any corruption or pruning.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, make_classification
from sklearn.model_selection import train_test_split


def _corrupt(X_train, y_train, rng, flip=0.0, dup=0.0, near_dup=0.0, near_scale=0.05, n_classes=None):
    """Return corrupted (X, y). Corruption is applied to TRAIN only."""
    y = y_train.copy()
    n = len(y)
    classes = np.unique(y)
    if flip > 0:
        f = rng.random(n) < flip
        for i in np.flatnonzero(f):
            y[i] = rng.choice(classes[classes != y[i]])
    is_df = isinstance(X_train, pd.DataFrame)
    parts_X, parts_y = [X_train], [y]
    if dup > 0:
        idx = rng.choice(n, size=int(dup * n), replace=False)
        parts_X.append(X_train.iloc[idx] if is_df else X_train[idx]); parts_y.append(y[idx])
    if near_dup > 0:
        idx = rng.choice(n, size=int(near_dup * n), replace=False)
        base = X_train.iloc[idx].to_numpy(dtype=float) if is_df else X_train[idx].astype(float)
        noisy = base + rng.normal(scale=near_scale, size=base.shape)
        parts_X.append(pd.DataFrame(noisy, columns=X_train.columns) if is_df else noisy); parts_y.append(y[idx])
    X_out = pd.concat(parts_X, ignore_index=True) if is_df else np.concatenate(parts_X)
    y_out = np.concatenate(parts_y)
    perm = rng.permutation(len(y_out))
    X_out = X_out.iloc[perm].reset_index(drop=True) if is_df else X_out[perm]
    return X_out, y_out[perm]


def _split(X, y, seed, test_size=0.25):
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    if isinstance(X, pd.DataFrame):
        return X.iloc[tr].reset_index(drop=True), y[tr], X.iloc[te].reset_index(drop=True), y[te]
    return X[tr], y[tr], X[te], y[te]


def load(name: str, seed: int = 0, max_rows: int | None = None):
    """Return dict(modality, X_train, y_train, X_test, y_test, notes)."""
    rng = np.random.default_rng(seed)
    if name == "synth_tabular_corrupted":
        X, y = make_classification(n_samples=12000, n_features=20, n_informative=8, n_redundant=4,
                                   weights=[0.75, 0.25], flip_y=0.0, class_sep=1.0, random_state=seed)
        X = pd.DataFrame(X, columns=[f"f{i}" for i in range(20)])
        Xtr, ytr, Xte, yte = _split(X, y, seed)
        Xtr, ytr = _corrupt(Xtr, ytr, rng, flip=0.08, dup=0.10)
        return dict(modality="tabular", X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte,
                    notes="8% train label flips + 10% exact duplicates added to train")
    if name in ("digits_corrupted", "digits_clean"):
        d = load_digits()
        Xtr, ytr, Xte, yte = _split(d.images, d.target, seed)
        if name == "digits_corrupted":
            Xtr, ytr = _corrupt(Xtr, ytr, rng, flip=0.05, dup=0.10, near_dup=0.10, near_scale=0.05)
            note = "5% train label flips + 10% exact + 10% near duplicates added to train"
        else:
            note = "no corruption"
        return dict(modality="image", X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte, notes=note)
    if name == "breast_cancer":
        d = load_breast_cancer(as_frame=True)
        Xtr, ytr, Xte, yte = _split(d.data, d.target.to_numpy(), seed)
        return dict(modality="tabular", X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte, notes="no corruption")
    if name == "adult":
        from sklearn.datasets import fetch_openml

        d = fetch_openml("adult", version=2, as_frame=True)
        X, y = d.data, d.target.to_numpy()
        if max_rows:
            X, y = X.iloc[:max_rows].reset_index(drop=True), y[:max_rows]
        Xtr, ytr, Xte, yte = _split(X, y, seed)
        return dict(modality="tabular", X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte, notes="OpenML adult")
    if name == "mnist":
        from sklearn.datasets import fetch_openml

        d = fetch_openml("mnist_784", version=1, as_frame=False)
        X = d.data.reshape(-1, 28, 28).astype(np.uint8)
        y = d.target.astype(int)
        if max_rows:
            X, y = X[:max_rows], y[:max_rows]
        Xtr, ytr, Xte, yte = _split(X, y, seed, test_size=0.2)
        return dict(modality="image", X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte, notes="OpenML MNIST")
    if name == "cifar10":
        import torchvision  # type: ignore

        tr = torchvision.datasets.CIFAR10("./data_cache", train=True, download=True)
        te = torchvision.datasets.CIFAR10("./data_cache", train=False, download=True)
        Xtr, ytr = np.asarray(tr.data, dtype=np.uint8), np.asarray(tr.targets)
        Xte, yte = np.asarray(te.data, dtype=np.uint8), np.asarray(te.targets)
        if max_rows:
            Xtr, ytr = Xtr[:max_rows], ytr[:max_rows]
        return dict(modality="image", X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte,
                    notes="CIFAR-10 (pixel-PCA embedder unless you pass a DINOv2 embedder)")
    raise ValueError(f"unknown dataset {name!r}")


OFFLINE = ["synth_tabular_corrupted", "digits_corrupted", "digits_clean", "breast_cancer"]
NEEDS_DOWNLOAD = ["adult", "mnist", "cifar10"]

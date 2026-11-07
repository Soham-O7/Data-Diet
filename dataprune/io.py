"""Loaders. Adding a new data type = one loader here + one embedder in embedders.py."""
from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp")


def load_csv(path: str, target: str, drop: Sequence[str] = ()) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"target column {target!r} not in CSV columns {list(df.columns)}")
    if df[target].isna().any():
        raise ValueError(f"target column {target!r} contains missing values; drop or fix them first")
    y = df[target]
    X = df.drop(columns=[target, *[c for c in drop if c in df.columns]])
    return X.reset_index(drop=True), y.reset_index(drop=True)


def load_image_folder(
    root: str,
    size: Optional[Tuple[int, int]] = None,
    max_per_class: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Folder layout: root/<class_name>/*.png. Returns (images uint8 (N,H,W,3), labels (N,) str, paths)."""
    from PIL import Image

    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    if not classes:
        raise ValueError(f"no class sub-folders found in {root}")
    imgs, labels, paths = [], [], []
    first_size = None
    for cls in classes:
        files = sorted(f for f in os.listdir(os.path.join(root, cls)) if f.lower().endswith(IMG_EXT))
        if max_per_class:
            files = files[:max_per_class]
        for f in files:
            p = os.path.join(root, cls, f)
            im = Image.open(p).convert("RGB")
            if size is not None:
                im = im.resize((size[1], size[0]))
            if first_size is None:
                first_size = im.size
            elif im.size != first_size:
                raise ValueError(
                    f"image sizes differ ({im.size} vs {first_size}); pass size=(H, W) to resize them"
                )
            imgs.append(np.asarray(im, dtype=np.uint8))
            labels.append(cls)
            paths.append(p)
    if not imgs:
        raise ValueError(f"no images with extensions {IMG_EXT} found under {root}")
    return np.stack(imgs), np.array(labels), paths

"""Stage 1: exact duplicates (SHA-256) and perceptual-hash duplicates for images.

All functions return a boolean KEEP mask over the input rows (first occurrence is kept).
Duplicates are label-aware: two identical rows with DIFFERENT labels are both kept
(that is a label conflict, which stage 3 is responsible for).
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def to_uint8(images: np.ndarray) -> np.ndarray:
    if images.dtype == np.uint8:
        return images
    a = images.astype(np.float64)
    lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        return np.zeros(images.shape, dtype=np.uint8)
    return np.clip((a - lo) / (hi - lo) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def exact_duplicate_mask_tabular(X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    seen = set()
    keep = np.ones(len(X), dtype=bool)
    y = np.asarray(y)
    for i, row in enumerate(X.itertuples(index=False, name=None)):
        h = hashlib.sha256((repr(row) + "|" + repr(y[i].item() if hasattr(y[i], "item") else y[i])).encode()).digest()
        if h in seen:
            keep[i] = False
        else:
            seen.add(h)
    return keep


def exact_duplicate_mask_images(images: np.ndarray, y: np.ndarray) -> np.ndarray:
    seen = set()
    keep = np.ones(len(images), dtype=bool)
    y = np.asarray(y)
    for i in range(len(images)):
        h = hashlib.sha256(np.ascontiguousarray(images[i]).tobytes() + b"|" + str(y[i]).encode()).digest()
        if h in seen:
            keep[i] = False
        else:
            seen.add(h)
    return keep


def dhash(image: np.ndarray, hash_size: int = 8) -> int:
    """Difference hash (64-bit for hash_size=8) of an (H,W) or (H,W,C) uint8 image."""
    from PIL import Image

    im = Image.fromarray(image)
    if im.mode not in ("L",):
        im = im.convert("L")
    im = im.resize((hash_size + 1, hash_size), Image.LANCZOS)
    a = np.asarray(im, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def phash_duplicate_mask(images: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Drop later images whose dHash AND label equal an earlier image's (Hamming distance 0)."""
    u8 = to_uint8(images)
    seen = set()
    keep = np.ones(len(u8), dtype=bool)
    y = np.asarray(y)
    for i in range(len(u8)):
        key = (dhash(u8[i]), str(y[i]))
        if key in seen:
            keep[i] = False
        else:
            seen.add(key)
    return keep

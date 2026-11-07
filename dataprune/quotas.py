"""Stage 6: per-class quotas ('constant proportion' rule).

Targets are defined on the ORIGINAL class counts so stages 1-3 (which may remove unevenly) cannot silently
skew the class mix. If a class no longer has enough rows after cleaning, its shortfall is redistributed to the
classes that still have spare rows. The integer quotas always sum to min(K, total available).

modes
  proportional : q_c ~ K * n_c / N          (every class keeps the same fraction)
  balanced_cap : q_c = min(n_c, T) with T chosen so sum q_c = K  (minority classes kept whole, majority pruned harder)
  auto         : balanced_cap when max/min class count >= imbalance_auto_ratio, else proportional
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def resolve_mode(mode: str, orig_counts: Dict[int, int], ratio_threshold: float) -> str:
    if mode != "auto":
        return mode
    counts = np.array([v for v in orig_counts.values() if v > 0], dtype=float)
    if len(counts) < 2:
        return "proportional"
    return "balanced_cap" if counts.max() / counts.min() >= ratio_threshold else "proportional"


def _water_fill(counts: np.ndarray, K: float) -> np.ndarray:
    """Return q_c = min(n_c, T) with sum = K (continuous T)."""
    n = counts.astype(float)
    if K >= n.sum():
        return n.copy()
    order = np.argsort(n)
    remaining = K
    q = np.zeros_like(n)
    m = len(n)
    for pos, c in enumerate(order):
        share = remaining / (m - pos)
        if n[c] <= share:
            q[c] = n[c]
            remaining -= n[c]
        else:
            q[order[pos:]] = share
            break
    return q


def allocate_quotas(
    orig_counts: Dict[int, int],
    avail_counts: Dict[int, int],
    K: int,
    mode: str = "proportional",
    min_per_class: int = 1,
) -> Dict[int, int]:
    classes = sorted(orig_counts)
    n_orig = np.array([orig_counts[c] for c in classes], dtype=float)
    avail = np.array([avail_counts.get(c, 0) for c in classes], dtype=float)
    K = int(min(K, avail.sum()))
    if K <= 0:
        return {c: 0 for c in classes}

    if mode == "proportional":
        target = K * n_orig / n_orig.sum()
    elif mode == "balanced_cap":
        target = _water_fill(n_orig, float(K))
    else:
        raise ValueError(f"unknown quota mode {mode!r} (resolve 'auto' first)")

    # cap by availability, redistribute the deficit to classes with spare rows
    q = np.minimum(target, avail)
    for _ in range(100):
        deficit = K - q.sum()
        if deficit < 1e-9:
            break
        spare = avail - q
        tot = spare.sum()
        if tot <= 1e-12:
            break
        q = q + np.minimum(spare, deficit * spare / tot)

    # integerise with largest remainders (never exceeding availability)
    base = np.floor(q + 1e-9).astype(int)
    base = np.minimum(base, avail.astype(int))
    rem = K - int(base.sum())
    if rem > 0:
        frac = q - base
        for i in np.argsort(-frac, kind="stable"):
            if rem == 0:
                break
            if base[i] < avail[i]:
                base[i] += 1
                rem -= 1
        # still short (can only happen via rounding corner cases): top up wherever there is room
        i = 0
        while rem > 0 and i < 10 * len(base) + 10:
            j = i % len(base)
            if base[j] < avail[j]:
                base[j] += 1
                rem -= 1
            i += 1

    # keep at least min_per_class rows of every non-empty class (take from the largest quota)
    for i in range(len(base)):
        floor_i = int(min(min_per_class, avail[i]))
        while base[i] < floor_i:
            donor = int(np.argmax(base))
            if base[donor] <= int(min(min_per_class, avail[donor])) or donor == i:
                break
            base[donor] -= 1
            base[i] += 1
    return {c: int(b) for c, b in zip(classes, base)}

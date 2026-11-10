"""Stage 2: SemDeDup-style near/semantic duplicate removal.

Cluster embeddings with k-means; inside each (cluster, class) group sort items by cosine similarity to the
centroid (ascending, so the item least like the centroid comes first and is the one that survives); an item is a
duplicate if its max cosine similarity to any EARLIER item in the group exceeds `threshold`.
This mirrors the SemDeDup recipe (Abbas et al., 2023). Differences from the paper: duplicates are only searched
within the same class (a near-identical pair with different labels is a stage-3 problem, not a duplicate), and a
safety cap limits how much may be removed. FAISS is not used (not installed); numpy block matmul is used instead.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans


def semdedup(
    emb: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.98,
    max_remove_frac: float = 0.25,
    n_clusters: int | None = None,
    seed: int = 0,
    block: int = 1024,
):
    """Return (keep_mask, info). `emb` should be (roughly) L2-normalised; it is re-normalised here."""
    n = len(emb)
    info = {"n_flagged": 0, "n_removed": 0, "capped": False, "n_clusters": 0, "threshold": threshold}
    if n < 3:
        return np.ones(n, dtype=bool), info
    E = emb.astype(np.float32)
    nrm = np.linalg.norm(E, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    E = E / nrm

    k = n_clusters or max(1, int(round(np.sqrt(n))))
    k = min(k, n)
    if n <= 20000:
        km = KMeans(n_clusters=k, n_init=1, random_state=seed).fit(E)
    else:
        km = MiniBatchKMeans(n_clusters=k, n_init=3, random_state=seed, batch_size=4096).fit(E)
    assign = km.labels_
    cent = km.cluster_centers_.astype(np.float32)
    cn = np.linalg.norm(cent, axis=1, keepdims=True)
    cn[cn == 0] = 1.0
    cent = cent / cn
    info["n_clusters"] = int(k)

    y = np.asarray(y)
    dup_score = np.full(n, -1.0, dtype=np.float32)
    # group by (cluster, class)
    order_all = np.lexsort((y, assign))
    a_sorted, y_sorted = assign[order_all], y[order_all]
    boundaries = np.flatnonzero((np.diff(a_sorted) != 0) | (np.diff(y_sorted) != 0)) + 1
    for grp in np.split(order_all, boundaries):
        m = len(grp)
        if m < 2:
            continue
        sim_c = E[grp] @ cent[assign[grp[0]]]
        o = np.argsort(sim_c, kind="stable")
        g = grp[o]
        Eg = E[g]
        for s in range(0, m, block):
            e = min(s + block, m)
            S = Eg[s:e] @ Eg.T
            rows = np.arange(s, e)[:, None]
            cols = np.arange(m)[None, :]
            S = np.where(cols < rows, S, -1.0)
            dup_score[g[s:e]] = S.max(axis=1)

    info["_dup_score"] = dup_score          # per-row max cosine to an earlier same-group item (popped by pipeline)
    flagged = dup_score > threshold
    n_flag = int(flagged.sum())
    info["n_flagged"] = n_flag
    cap = int(np.floor(max_remove_frac * n))
    if n_flag > cap:
        info["capped"] = True
        idx = np.flatnonzero(flagged)
        top = idx[np.argsort(-dup_score[idx], kind="stable")[:cap]]
        flagged = np.zeros(n, dtype=bool)
        flagged[top] = True
    keep = ~flagged
    info["n_removed"] = int(flagged.sum())
    return keep, info

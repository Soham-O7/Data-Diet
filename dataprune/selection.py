"""Stage 5 selectors (+ stage 6 application of per-class quotas).

All selectors work inside ONE class at a time (that is how the class quota is enforced) and return LOCAL indices
into the arrays they were given. Higher difficulty score = harder sample.

  random : uniform sample.
  el2n   : EL2N-only baseline - keep the `budget` highest-EL2N (hardest) samples.
  ccs    : Coverage-centric selection (Zheng et al., ICLR 2023): drop the hardest `beta` fraction, split the rest
           of the score range into equal-width strata, fill strata with the paper's budget-allocation rule
           (smallest strata first, each gets min(size, remaining_budget / remaining_strata), random within stratum).
  d2     : two modes (cfg.d2_mode). "paper" = select_d2_paper, the algorithm as published (checked against
           arXiv 2310.07931). "simplified" (default, what earlier benchmarks used) = "D2-style" pruning (after Maharana et al., 2023): kNN graph in embedding space, node score = difficulty
           plus forward message passing from neighbours; then greedy selection of the highest score, with the
           selected node's graph neighbours penalised (reverse message passing) so picks stay spread out.
           This is a simplified reading of the paper - edge weights/hyper-parameters are mine; check the paper
           before quoting 'D2 Pruning' results on slides.
"""
from __future__ import annotations

import heapq
from typing import Dict

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def _norm01(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return np.zeros_like(a, dtype=np.float64) if hi <= lo else (a - lo) / (hi - lo)


def select_random(n: int, budget: int, rng: np.random.Generator) -> np.ndarray:
    return np.sort(rng.choice(n, size=budget, replace=False))


def select_hardest(score: np.ndarray, budget: int) -> np.ndarray:
    return np.sort(np.argsort(-score, kind="stable")[:budget])


def select_ccs(score: np.ndarray, budget: int, rng: np.random.Generator,
               beta: float = 0.1, n_strata: int = 50) -> np.ndarray:
    n = len(score)
    n_prune = int(np.floor(beta * n))
    n_prune = min(n_prune, n - budget)            # beta is only valid while enough samples remain
    order = np.argsort(-score, kind="stable")
    pool = order[n_prune:]                         # drop the n_prune hardest
    s = score[pool]
    lo, hi = float(s.min()), float(s.max())
    if hi <= lo:
        return np.sort(rng.choice(pool, size=budget, replace=False))
    edges = np.linspace(lo, hi, n_strata + 1)
    which = np.clip(np.digitize(s, edges[1:-1]), 0, n_strata - 1)
    strata = [pool[which == b] for b in range(n_strata)]
    strata = [st for st in strata if len(st) > 0]
    strata.sort(key=len)
    chosen = []
    remaining = budget
    for pos, st in enumerate(strata):
        m = min(len(st), int(np.floor(remaining / (len(strata) - pos))))
        m = min(m, remaining)
        if m > 0:
            chosen.append(rng.choice(st, size=m, replace=False))
            remaining -= m
    sel = np.concatenate(chosen) if chosen else np.array([], dtype=int)
    if len(sel) < budget:                          # rounding leftovers: fill from the unselected pool
        left = np.setdiff1d(pool, sel, assume_unique=False)
        sel = np.concatenate([sel, rng.choice(left, size=budget - len(sel), replace=False)])
    return np.sort(sel[:budget])


def _knn_sq_dist_graph(geom: np.ndarray, k_eff: int):
    """Symmetric kNN graph; returns (csr of squared distances / median^2, structure). Explicit zeros are kept."""
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(geom)
    dist, nbr = nn.kneighbors(geom)
    dist, nbr = dist[:, 1:], nbr[:, 1:]
    sigma = float(np.median(dist)) + 1e-12            # scale normalisation (paper uses raw embedding distances)
    n = len(geom)
    rows = np.repeat(np.arange(n), k_eff)
    D = sparse.coo_matrix((((dist / sigma) ** 2).ravel() + 1e-12, (rows, nbr.ravel())), shape=(n, n)).tocsr()
    return D.maximum(D.T).tocsr()


def select_d2_paper(score: np.ndarray, geom: np.ndarray, budget: int, k: int = 10,
                    gamma_f: float = 1.0, gamma_r: float = 0.9) -> np.ndarray:
    """D2 Pruning as described in Maharana et al. (ICLR 2024), Eq. 4-5 + reverse message passing:
       e_ij = exp(-gamma * d_ij^2) on kNN edges (undirected); forward: x_i += sum_j e^f_ij x_j (gamma_f);
       loop: pick argmax x, then x_j -= e^r_kj * x_k for graph neighbours j (gamma_r).
       Only deviation: distances are divided by the median kNN distance so the kernel scale does not depend on the
       embedding's units. Scores are used raw (non-negative difficulty), not rescaled."""
    n = len(score)
    if budget >= n:
        return np.arange(n)
    k_eff = int(min(k, n - 1))
    if k_eff < 1:
        return select_hardest(score, budget)
    D = _knn_sq_dist_graph(geom, k_eff)
    Wf, Wr = D.copy(), D.copy()
    Wf.data = np.exp(-gamma_f * D.data)
    Wr.data = np.exp(-gamma_r * D.data)
    x0 = np.maximum(score.astype(np.float64), 0.0)
    s = x0 + Wf @ x0
    version = np.zeros(n, dtype=np.int64)
    selected = np.zeros(n, dtype=bool)
    heap = [(-s[i], 0, i) for i in range(n)]
    heapq.heapify(heap)
    out = []
    indptr, indices, data = Wr.indptr, Wr.indices, Wr.data
    while len(out) < budget and heap:
        neg, ver, i = heapq.heappop(heap)
        if selected[i]:
            continue
        if ver != version[i]:
            heapq.heappush(heap, (-s[i], version[i], i))
            continue
        selected[i] = True
        out.append(i)
        xk = s[i]
        for p in range(indptr[i], indptr[i + 1]):
            j = indices[p]
            if not selected[j]:
                s[j] -= data[p] * xk
                version[j] += 1
    return np.sort(np.array(out, dtype=int))


def select_d2(score: np.ndarray, geom: np.ndarray, budget: int, k: int = 10,
              gamma_forward: float = 0.5, gamma_reverse: float = 1.0) -> np.ndarray:
    n = len(score)
    if budget >= n:
        return np.arange(n)
    k_eff = int(min(k, n - 1))
    if k_eff < 1:
        return select_hardest(score, budget)
    nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(geom)
    dist, nbr = nn.kneighbors(geom)
    dist, nbr = dist[:, 1:], nbr[:, 1:]
    sigma = float(np.median(dist)) + 1e-12
    w = np.exp(-(dist / sigma) ** 2)
    rows = np.repeat(np.arange(n), k_eff)
    W = sparse.csr_matrix((w.ravel(), (rows, nbr.ravel())), shape=(n, n))
    W = W.maximum(W.T).tocsr()                     # symmetric graph

    s0 = _norm01(score.astype(np.float64))
    deg = np.maximum(np.asarray((W > 0).sum(axis=1)).ravel(), 1)
    s = s0 + gamma_forward * (W @ s0) / deg        # forward message passing

    version = np.zeros(n, dtype=np.int64)
    selected = np.zeros(n, dtype=bool)
    heap = [(-s[i], 0, i) for i in range(n)]
    heapq.heapify(heap)
    out = []
    indptr, indices, data = W.indptr, W.indices, W.data
    while len(out) < budget and heap:
        neg, ver, i = heapq.heappop(heap)
        if selected[i]:
            continue
        if ver != version[i]:                      # stale entry: scores only ever decrease -> lazy greedy is exact
            heapq.heappush(heap, (-s[i], version[i], i))
            continue
        selected[i] = True
        out.append(i)
        for p in range(indptr[i], indptr[i + 1]):  # reverse message passing
            j = indices[p]
            if not selected[j]:
                s[j] -= gamma_reverse * data[p]
                version[j] += 1
    return np.sort(np.array(out, dtype=int))


def select_per_class(
    selector: str,
    y: np.ndarray,
    quotas: Dict[int, int],
    difficulty: np.ndarray,
    el2n: np.ndarray,
    geom: np.ndarray,
    cfg,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply `selector` inside each class with that class's quota. Returns sorted global positions."""
    picked = []
    for c, q in quotas.items():
        pos = np.flatnonzero(y == c)
        if q <= 0 or len(pos) == 0:
            continue
        q = min(q, len(pos))
        if q == len(pos):
            picked.append(pos)
            continue
        if selector == "random":
            loc = select_random(len(pos), q, rng)
        elif selector == "el2n":
            loc = select_hardest(el2n[pos], q)
        elif selector == "ccs":
            loc = select_ccs(difficulty[pos], q, rng, beta=cfg.ccs_beta, n_strata=cfg.ccs_strata)
        elif selector == "d2" and getattr(cfg, "d2_mode", "simplified") == "paper":
            loc = select_d2_paper(difficulty[pos], geom[pos], q, k=cfg.d2_k, gamma_f=cfg.d2_paper_gamma_f,
                                  gamma_r=cfg.d2_paper_gamma_r)
        elif selector == "d2":
            loc = select_d2(difficulty[pos], geom[pos], q, k=cfg.d2_k,
                            gamma_forward=cfg.d2_gamma_forward, gamma_reverse=cfg.d2_gamma_reverse)
        else:
            raise ValueError(f"unknown selector {selector!r}")
        picked.append(pos[loc])
    return np.sort(np.concatenate(picked)) if picked else np.array([], dtype=int)

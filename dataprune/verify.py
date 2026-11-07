"""Stage 7: verification.

distribution_fidelity(): no-training checks (class-prior drift, feature KS, embedding coverage).
proof_run():            three-way proof - train the same model on FULL vs PRUNED vs RANDOM (same size) data and
                        score on a held-out test set. Reports accuracy, balanced accuracy, per-class recall,
                        training time and estimated energy, plus a pass/fail on the configured fidelity thresholds.
Two random baselines are reported: 'random_stratified' (same per-class counts as the pruned set - the stricter,
fairer baseline) and 'random_uniform' (plain random sample of the same size).
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score
from sklearn.neighbors import NearestNeighbors

from .energy import break_even_runs, savings_card


def distribution_fidelity(
    y_all: np.ndarray,
    keep_idx: np.ndarray,
    n_classes: int,
    Z_all: Optional[np.ndarray] = None,
    geom_all: Optional[np.ndarray] = None,
    seed: int = 0,
    max_query: int = 3000,
) -> dict:
    out = {}
    p_full = np.bincount(y_all, minlength=n_classes) / len(y_all)
    p_keep = np.bincount(y_all[keep_idx], minlength=n_classes) / max(len(keep_idx), 1)
    out["class_prior_tv_distance"] = float(0.5 * np.abs(p_full - p_keep).sum())
    out["class_prior_full"] = p_full.round(4).tolist()
    out["class_prior_kept"] = p_keep.round(4).tolist()

    if Z_all is not None and Z_all.shape[1] <= 512 and len(keep_idx) > 10:
        n_feat = min(Z_all.shape[1], 50)
        ks = [ks_2samp(Z_all[:, j], Z_all[keep_idx, j]).statistic for j in range(n_feat)]
        out["mean_feature_ks"] = float(np.mean(ks))
        out["max_feature_ks"] = float(np.max(ks))

    if geom_all is not None and len(keep_idx) > 10:
        rng = np.random.default_rng(seed)
        q = rng.choice(len(geom_all), size=min(max_query, len(geom_all)), replace=False)
        nn = NearestNeighbors(n_neighbors=1).fit(geom_all[keep_idx])
        d_kept = nn.kneighbors(geom_all[q])[0].mean()
        rnd = rng.choice(len(geom_all), size=len(keep_idx), replace=False)
        d_rand = NearestNeighbors(n_neighbors=1).fit(geom_all[rnd]).kneighbors(geom_all[q])[0].mean()
        out["coverage_mean_nn_dist_kept"] = float(d_kept)
        out["coverage_mean_nn_dist_random"] = float(d_rand)
        out["coverage_ratio_vs_random"] = float(d_kept / d_rand) if d_rand > 0 else float("nan")
    return out


def _fit_eval(model, Ztr, ytr, Zte, yte, n_classes, watts):
    w0, c0 = time.perf_counter(), time.process_time()
    model.fit(Ztr, ytr)
    wall, cpu = time.perf_counter() - w0, time.process_time() - c0
    pred = model.predict(Zte)
    rec = recall_score(yte, pred, labels=list(range(n_classes)), average=None, zero_division=0)
    return {
        "acc": float(accuracy_score(yte, pred)),
        "bal_acc": float(balanced_accuracy_score(yte, pred)),
        "recall": rec,
        "fit_wall_s": wall,
        "fit_cpu_s": cpu,
        "fit_kwh_est": cpu * watts / 3.6e6,
    }


def proof_run(
    Z_all: np.ndarray,
    y_all: np.ndarray,
    keep_idx: np.ndarray,
    Z_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    cfg,
    model_factory: Callable[[int], object],
    eval_seeds: Sequence[int] = (0, 1, 2),
    pruning_overhead_kwh: Optional[float] = None,
):
    keep_idx = np.asarray(keep_idx)
    K = len(keep_idx)
    kept_counts = np.bincount(y_all[keep_idx], minlength=n_classes)
    rows = []
    for s in eval_seeds:
        rng = np.random.default_rng(10_000 + s)
        # random uniform, same size
        r_uni = rng.choice(len(y_all), size=K, replace=False)
        # random stratified, same per-class counts as the pruned subset
        parts = []
        for c in range(n_classes):
            pool = np.flatnonzero(y_all == c)
            m = int(min(kept_counts[c], len(pool)))
            if m > 0:
                parts.append(rng.choice(pool, size=m, replace=False))
        r_str = np.concatenate(parts)
        subsets = {
            "full": np.arange(len(y_all)),
            "pruned": keep_idx,
            "random_stratified": r_str,
            "random_uniform": r_uni,
        }
        for name, idx in subsets.items():
            res = _fit_eval(model_factory(s), Z_all[idx], y_all[idx], Z_test, y_test, n_classes,
                            cfg.cpu_watts_per_core)
            res.update({"method": name, "seed": s, "n_train": int(len(idx))})
            rows.append(res)

    raw = pd.DataFrame(rows)
    recall_cols = np.stack(raw["recall"].to_numpy())
    raw = raw.drop(columns=["recall"])
    agg = raw.groupby("method").agg(
        n_train=("n_train", "first"),
        acc_mean=("acc", "mean"), acc_std=("acc", "std"),
        bal_acc_mean=("bal_acc", "mean"),
        fit_wall_s=("fit_wall_s", "mean"), fit_cpu_s=("fit_cpu_s", "mean"),
        fit_kwh_est=("fit_kwh_est", "mean"),
    ).fillna(0.0)
    per_class = {m: recall_cols[(raw["method"] == m).to_numpy()].mean(axis=0) for m in agg.index}
    full_acc = float(agg.loc["full", "acc_mean"])
    retained = {m: float(agg.loc[m, "acc_mean"] / full_acc) if full_acc > 0 else float("nan") for m in agg.index}
    worst_drop = float(np.max(per_class["full"] - per_class["pruned"]))
    verdict = {
        "pruned_retained_vs_full": retained["pruned"],
        "random_stratified_retained_vs_full": retained["random_stratified"],
        "random_uniform_retained_vs_full": retained["random_uniform"],
        "pruned_minus_random_stratified_acc": float(agg.loc["pruned", "acc_mean"] - agg.loc["random_stratified", "acc_mean"]),
        "worst_per_class_recall_drop_vs_full": worst_drop,
        "fidelity_pass": bool(retained["pruned"] >= cfg.fidelity_min_retained
                              and worst_drop <= cfg.fidelity_max_class_drop),
        "thresholds": {"min_retained": cfg.fidelity_min_retained,
                       "max_class_drop": cfg.fidelity_max_class_drop},
        "per_class_recall": {m: v.round(4).tolist() for m, v in per_class.items()},
        "eval_seeds": list(eval_seeds),
    }
    if pruning_overhead_kwh is not None:
        j = lambda kwh: kwh * 3.6e6
        verdict["pruning_overhead_kwh_est"] = pruning_overhead_kwh
        verdict["train_kwh_est_full"] = float(agg.loc["full", "fit_kwh_est"])
        verdict["train_kwh_est_pruned"] = float(agg.loc["pruned", "fit_kwh_est"])
        be = break_even_runs(j(pruning_overhead_kwh), j(float(agg.loc["full", "fit_kwh_est"])),
                             j(float(agg.loc["pruned", "fit_kwh_est"])))
        verdict["break_even_training_runs"] = None if np.isinf(be) else float(be)
        verdict["savings_card"] = savings_card(
            pruning_overhead_kwh, float(agg.loc["full", "fit_kwh_est"]), float(agg.loc["pruned", "fit_kwh_est"]),
            cfg.grid_kg_co2_per_kwh, cfg.price_per_kwh)
        verdict["break_even_note"] = ("training runs of THIS evaluation model needed before pruning overhead is "
                                      "repaid; scale with your real model's cost. null = no per-run saving.")
    return agg.reset_index(), raw, verdict

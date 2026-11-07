"""The 7-stage pruning pipeline.

  1 hash (SHA-256 / dHash) -> [embed or preprocess] -> 2 SemDeDup -> 3 confident learning -> 4 difficulty scoring
  -> 5 selection (fixed selector or 'auto' selector race) -> 6 per-class quotas -> 7 verification (see verify.py)

Row accounting: `keep_indices` index the ORIGINAL rows passed in. `removed_by_stage` maps stage name -> original
indices removed there, so every dropped row is explainable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .config import PruneConfig
from .difficulty import DifficultyScores, score_difficulty
from .embedders import PixelPCAEmbedder
from .energy import EnergyMeter
from .hashing import exact_duplicate_mask_images, exact_duplicate_mask_tabular, phash_duplicate_mask
from .label_noise import find_label_issues, oof_probabilities
from .models import make_eval_model, make_oof_model
from .preprocess import TabularPreprocessor
from .quotas import allocate_quotas, resolve_mode
from .race import run_race
from .selection import select_per_class
from .semdedup import semdedup
from .verify import distribution_fidelity


@dataclass
class StageReport:
    name: str
    n_in: int
    n_out: int
    seconds: float = 0.0
    notes: Dict[str, Any] = field(default_factory=dict)

    @property
    def removed(self) -> int:
        return self.n_in - self.n_out


@dataclass
class PruneResult:
    keep_indices: np.ndarray
    removed_by_stage: Dict[str, np.ndarray]
    stages: List[StageReport]
    selector_used: str
    race_table: Optional[pd.DataFrame]
    race_info: Optional[dict]
    quota_mode: str
    quotas: Dict[Any, int]
    energy: dict
    fidelity: dict
    classes_: np.ndarray
    y_int: np.ndarray                 # integer-encoded labels of the ORIGINAL rows
    modality: str
    transformer: Any                  # fitted TabularPreprocessor / embedder: .transform(X) -> Z
    config: PruneConfig
    warnings: List[str] = field(default_factory=list)

    def transform(self, X):
        return self.transformer.transform(X)

    def summary(self) -> dict:
        return {
            "modality": self.modality,
            "n_original": int(len(self.y_int)),
            "n_kept": int(len(self.keep_indices)),
            "kept_fraction": float(len(self.keep_indices) / len(self.y_int)),
            "target_keep_ratio": self.config.keep_ratio,
            "selector_used": self.selector_used,
            "selector_chosen_automatically": self.config.selector == "auto",
            "quota_mode": self.quota_mode,
            "quotas": {str(self.classes_[c]): int(q) for c, q in self.quotas.items()},
            "stages": [
                {"name": s.name, "n_in": s.n_in, "n_out": s.n_out, "removed": s.removed,
                 "seconds": round(s.seconds, 4), **({"notes": s.notes} if s.notes else {})}
                for s in self.stages
            ],
            "race": self.race_info,
            "energy": self.energy,
            "fidelity": self.fidelity,
            "warnings": self.warnings,
            "config": self.config.to_dict(),
        }

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2, default=_jsonable)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _detect_modality(X) -> str:
    if isinstance(X, pd.DataFrame):
        return "tabular"
    a = np.asarray(X)
    if a.ndim in (3, 4):
        return "image"
    if a.ndim == 2:
        return "tabular"
    raise ValueError(f"cannot infer modality from array of shape {a.shape}")


def prune(
    X,
    y,
    config: Optional[PruneConfig] = None,
    modality: Optional[str] = None,
    embedder=None,
) -> PruneResult:
    cfg = config or PruneConfig()
    cfg.validate()
    modality = modality or _detect_modality(X)
    if modality not in ("tabular", "image"):
        raise ValueError("modality must be 'tabular' or 'image' (text/audio are roadmap)")

    if modality == "tabular":
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
        X = X.reset_index(drop=True)
    else:
        X = np.asarray(X)
    y = np.asarray(y)
    N = len(y)
    if len(X) != N:
        raise ValueError(f"X has {len(X)} rows but y has {N}")
    if N < 4:
        raise ValueError("need at least 4 rows")
    classes, y_int = np.unique(y, return_inverse=True)
    y_int = y_int.astype(int)
    C = len(classes)

    meter = EnergyMeter(cfg.cpu_watts_per_core, cfg.energy_backend, cfg.grid_kg_co2_per_kwh,
                        cfg.price_per_kwh, cfg.country_iso_code)
    stages: List[StageReport] = []
    removed: Dict[str, np.ndarray] = {}
    warns: List[str] = []
    pos = np.arange(N)                         # original index of every surviving row

    def _sec(name):
        return meter.stages[name].wall_s

    # ------------------------------------------------------------ stage 1: exact duplicates
    n0 = len(pos)
    if cfg.do_hash:
        with meter.stage("1_hash"):
            if modality == "tabular":
                keep = exact_duplicate_mask_tabular(X.iloc[pos], y_int[pos])
            else:
                keep = exact_duplicate_mask_images(X[pos], y_int[pos])
            notes = {"exact_duplicates": int((~keep).sum())}
            removed["1_hash"] = pos[~keep]
            pos = pos[keep]
            if modality == "image" and cfg.use_phash and min(X.shape[1], X.shape[2]) >= cfg.phash_min_side:
                keep2 = phash_duplicate_mask(X[pos], y_int[pos])
                notes["perceptual_duplicates"] = int((~keep2).sum())
                removed["1_hash"] = np.concatenate([removed["1_hash"], pos[~keep2]])
                pos = pos[keep2]
        stages.append(StageReport("1_hash", n0, len(pos), _sec("1_hash"), notes))
    else:
        removed["1_hash"] = np.array([], dtype=int)

    pos1 = pos.copy()                          # base set after stage 1
    y1 = y_int[pos1]

    # ------------------------------------------------------------ features (preprocess / embed)
    with meter.stage("features"):
        if modality == "tabular":
            transformer = TabularPreprocessor().fit(X.iloc[pos1])
            Z1 = transformer.transform(X.iloc[pos1])
            if Z1.shape[1] > 32:
                geom_pca = PCA(n_components=32, random_state=cfg.seed).fit(Z1)
                G1 = geom_pca.transform(Z1).astype(np.float32)
            else:
                G1 = Z1.copy()
        else:
            transformer = embedder or PixelPCAEmbedder(seed=cfg.seed)
            Z1 = np.asarray(transformer.fit_transform(X[pos1]), dtype=np.float32)
            G1 = Z1
    stages.append(StageReport("features", len(pos1), len(pos1), _sec("features"),
                              {"embedder": getattr(transformer, "name", "tabular_preprocessor"),
                               "dim": int(Z1.shape[1])}))
    if modality == "image" and getattr(transformer, "name", "") == "pixel_pca":
        warns.append("Using the pixel-PCA fallback embedder, not DINOv2: semantic duplicate detection and "
                     "coverage are weaker than with a pretrained embedding.")

    alive = np.arange(len(pos1))               # indices into the post-stage-1 base arrays

    # ------------------------------------------------------------ stage 2: SemDeDup
    if cfg.do_semdedup and (modality == "image" or cfg.semdedup_tabular):
        n_in = len(alive)
        with meter.stage("2_semdedup"):
            keep, info = semdedup(G1[alive], y1[alive], cfg.semdedup_threshold, cfg.semdedup_max_remove_frac,
                                  cfg.semdedup_n_clusters, cfg.seed)
            removed["2_semdedup"] = pos1[alive[~keep]]
            alive = alive[keep]
        stages.append(StageReport("2_semdedup", n_in, len(alive), _sec("2_semdedup"), info))
    else:
        removed["2_semdedup"] = np.array([], dtype=int)

    # ------------------------------------------------------------ stage 3: label noise
    present = np.bincount(y1[alive], minlength=C)
    if cfg.do_label_noise and (present > 0).sum() >= 2 and len(alive) >= 20:
        n_in = len(alive)
        with meter.stage("3_label_noise"):
            P = oof_probabilities(Z1[alive], y1[alive], C, lambda s: make_oof_model(modality, s),
                                  folds=cfg.noise_cv_folds, seed=cfg.seed)
            issue, info, _ = find_label_issues(P, y1[alive], cfg.noise_max_remove_frac,
                                               cfg.noise_max_remove_frac_per_class, cfg.use_cleanlab_if_available)
            removed["3_label_noise"] = pos1[alive[issue]]
            alive = alive[~issue]
        stages.append(StageReport("3_label_noise", n_in, len(alive), _sec("3_label_noise"), info))
    else:
        removed["3_label_noise"] = np.array([], dtype=int)

    Zc, Gc, yc = Z1[alive], G1[alive], y1[alive]
    m = len(alive)

    # ------------------------------------------------------------ budget & quotas (stage 6 setup)
    K = int(round(cfg.keep_ratio * N))
    K = max(K, min(C, m))
    if K > m:
        warns.append(f"target of {K} rows is larger than the {m} rows left after cleaning; "
                     f"kept all {m} cleaned rows (keep ratio {m / N:.3f} instead of {cfg.keep_ratio}).")
        K = m
    base_counts = {c: int((y1 == c).sum()) for c in range(C)}       # class mix measured after exact-dup removal
    avail_counts = {c: int((yc == c).sum()) for c in range(C)}
    quota_mode = resolve_mode(cfg.quota_mode, base_counts, cfg.imbalance_auto_ratio)
    quotas = allocate_quotas(base_counts, avail_counts, K, quota_mode, cfg.min_per_class)

    # ------------------------------------------------------------ stage 4: difficulty
    selector = cfg.selector
    ds: Optional[DifficultyScores] = None
    needs_scores = selector != "random"
    if needs_scores and K < m:
        with meter.stage("4_difficulty"):
            ds = score_difficulty(Zc, yc, C, modality=modality, proxy=cfg.proxy, epochs=cfg.proxy_epochs,
                                  seeds=cfg.proxy_seeds, seed=cfg.seed, el2n_epoch_frac=cfg.el2n_epoch_frac,
                                  metric=cfg.difficulty_metric)
        stages.append(StageReport("4_difficulty", m, m, _sec("4_difficulty"),
                                  {"metric": ds.metric, "proxy": cfg.proxy or ("hgb" if modality == "tabular" else "mlp"),
                                   "epochs": cfg.proxy_epochs, "seeds": cfg.proxy_seeds}))
    if ds is None:
        zeros = np.zeros(m)
        ds = DifficultyScores(zeros, zeros, zeros, zeros, zeros, zeros, "none")

    # ------------------------------------------------------------ stage 5: selector (race or fixed)
    race_table, race_info = None, None
    if K >= m:
        selector = "keep_all"
        sel = np.arange(m)
    else:
        if selector == "auto":
            with meter.stage("5_selector_race"):
                selector, race_info, race_table = run_race(
                    Zc, Gc, yc, C, K / m, cfg, modality, lambda s: make_eval_model(modality, s), quota_mode)
            stages.append(StageReport("5_selector_race", m, m, _sec("5_selector_race"),
                                      {"winner": selector, "beats_random": race_info["beats_random"]}))
            if not race_info["beats_random"]:
                warns.append("Selector race: no candidate beat random on the probe; differences on small probes "
                             "can be noise - check the proof run before trusting the pruned subset.")
        rng = np.random.default_rng(cfg.seed + 1)
        with meter.stage("5_select"):
            sel = select_per_class(selector, yc, quotas, ds.difficulty, ds.el2n, Gc, cfg, rng)
    stages.append(StageReport("5_6_select_with_quotas", m, len(sel), meter.stages["5_select"].wall_s
                              if "5_select" in meter.stages else 0.0,
                              {"selector": selector, "quota_mode": quota_mode}))
    removed["5_selection"] = pos1[np.setdiff1d(alive, alive[sel], assume_unique=True)]
    keep_base = alive[sel]
    keep_orig = np.sort(pos1[keep_base])

    # ------------------------------------------------------------ stage 7a: distribution fidelity (no training)
    kept_in_base = np.flatnonzero(np.isin(pos1, keep_orig))
    fid = distribution_fidelity(y1, kept_in_base, C, Z1, G1, seed=cfg.seed)
    fid["class_prior_tv_distance_vs_original"] = float(
        0.5 * np.abs(np.bincount(y_int, minlength=C) / N
                     - np.bincount(y_int[keep_orig], minlength=C) / max(len(keep_orig), 1)).sum())

    return PruneResult(
        keep_indices=keep_orig,
        removed_by_stage=removed,
        stages=stages,
        selector_used=selector,
        race_table=race_table,
        race_info=race_info,
        quota_mode=quota_mode,
        quotas=quotas,
        energy=meter.summary(),
        fidelity=fid,
        classes_=classes,
        y_int=y_int,
        modality=modality,
        transformer=transformer,
        config=cfg,
        warnings=warns,
    )

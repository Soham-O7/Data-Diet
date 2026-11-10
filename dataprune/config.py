"""Configuration for the data-pruning pipeline."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Tuple


@dataclass
class PruneConfig:
    # ---- global ----
    keep_ratio: float = 0.4          # fraction of the ORIGINAL rows to keep, in (0, 1]
    seed: int = 0

    # ---- stage toggles ----
    do_hash: bool = True             # stage 1
    do_semdedup: bool = True         # stage 2 (images by default, see semdedup_tabular)
    semdedup_tabular: bool = False   # plan: CSV path skips SemDeDup
    do_label_noise: bool = True      # stage 3

    # ---- stage 1: exact duplicates ----
    use_phash: bool = True           # perceptual hash for images
    phash_min_side: int = 16         # skip phash for tiny images (e.g. 8x8 digits)

    # ---- stage 2: SemDeDup ----
    semdedup_threshold: float = 0.98     # cosine similarity above which two same-class items are duplicates
    semdedup_max_remove_frac: float = 0.25   # safety cap on how much stage 2 may remove
    semdedup_n_clusters: Optional[int] = None  # default ~ sqrt(n)

    # ---- stage 3: label noise (confident learning) ----
    noise_cv_folds: int = 5
    noise_max_remove_frac: float = 0.10      # safety cap on total removal
    noise_max_remove_frac_per_class: float = 0.30
    use_cleanlab_if_available: bool = True

    # ---- stage 4: difficulty scoring ----
    proxy: Optional[str] = None          # 'hgb' | 'mlp' | None -> hgb for tabular, mlp for images
    proxy_epochs: int = 20
    proxy_seeds: int = 3
    el2n_epoch_frac: float = 0.5         # EL2N averaged over the first fraction of epochs
    difficulty_metric: Optional[str] = None   # 'combined' | 'datamap' | 'el2n' | 'loss' | 'forget'
                                              # None -> combined for images, datamap for tabular

    # ---- stage 5: selection ----
    selector: str = "auto"               # 'auto' (selector race) | 'random' | 'el2n' | 'ccs' | 'd2'
    race_candidates: Tuple[str, ...] = ("el2n", "ccs", "d2")
    race_probe_size: int = 4000
    race_repeats: int = 2
    race_metric: str = "auto"            # 'auto' | 'accuracy' | 'balanced_accuracy'
    ccs_beta: float = 0.1                # fraction of hardest samples dropped before stratified sampling
    ccs_strata: int = 50
    d2_k: int = 10
    d2_gamma_forward: float = 0.5
    d2_gamma_reverse: float = 1.0
    d2_mode: str = "simplified"          # 'simplified' (own variant, default) | 'paper' (Maharana et al. as published)
    d2_paper_gamma_f: float = 1.0        # paper mode: forward kernel exp(-gamma_f d^2) (paper fixes 1.0)
    d2_paper_gamma_r: float = 0.9        # paper mode: reverse kernel; paper tuned 0.0-1.0 per pruning rate (0.9 at 30%, 0 at 90%)

    # ---- stage 6: per-class quotas ----
    quota_mode: str = "auto"             # 'proportional' | 'balanced_cap' | 'auto'
    imbalance_auto_ratio: float = 5.0    # auto -> balanced_cap if max/min class count >= this
    min_per_class: int = 1

    # ---- energy accounting (ESTIMATE, not a power meter) ----
    cpu_watts_per_core: float = 15.0
    energy_backend: str = "estimate"     # 'estimate' | 'codecarbon' (falls back to estimate if not installed)
    grid_kg_co2_per_kwh: float = 0.7     # PLACEHOLDER - set your local grid carbon intensity
    price_per_kwh: float = 8.0           # PLACEHOLDER - set your tariff (currency of your choice)
    country_iso_code: str = "IND"        # used by codecarbon's offline tracker

    # ---- stage 7: verification ----
    fidelity_min_retained: float = 0.98      # pruned_acc / full_acc
    fidelity_max_class_drop: float = 0.10    # worst per-class recall drop vs full

    def validate(self) -> None:
        if not (0.0 < self.keep_ratio <= 1.0):
            raise ValueError(f"keep_ratio must be in (0, 1], got {self.keep_ratio}")
        if self.selector not in ("auto", "random", "el2n", "ccs", "d2"):
            raise ValueError(f"unknown selector {self.selector!r}")
        for c in self.race_candidates:
            if c not in ("el2n", "ccs", "d2"):
                raise ValueError(f"unknown race candidate {c!r}")
        if self.quota_mode not in ("proportional", "balanced_cap", "auto"):
            raise ValueError(f"unknown quota_mode {self.quota_mode!r}")
        if self.proxy not in (None, "hgb", "mlp"):
            raise ValueError(f"unknown proxy {self.proxy!r}")
        if self.difficulty_metric not in (None, "combined", "datamap", "el2n", "loss", "forget"):
            raise ValueError(f"unknown difficulty_metric {self.difficulty_metric!r}")
        if self.energy_backend not in ("estimate", "codecarbon"):
            raise ValueError(f"unknown energy_backend {self.energy_backend!r}")
        if self.d2_mode not in ("simplified", "paper"):
            raise ValueError(f"unknown d2_mode {self.d2_mode!r}")
        if self.race_metric not in ("auto", "accuracy", "balanced_accuracy"):
            raise ValueError(f"unknown race_metric {self.race_metric!r}")
        if not (0.0 < self.semdedup_threshold <= 1.0):
            raise ValueError("semdedup_threshold must be in (0, 1]")

    def to_dict(self) -> dict:
        return asdict(self)

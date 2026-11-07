import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_digits, make_classification

from dataprune import PruneConfig, prune
from dataprune.difficulty import score_difficulty
from dataprune.energy import EnergyMeter, break_even_runs
from dataprune.hashing import (dhash, exact_duplicate_mask_images, exact_duplicate_mask_tabular,
                               phash_duplicate_mask)
from dataprune.label_noise import find_label_issues, oof_probabilities
from dataprune.models import make_oof_model
from dataprune.quotas import allocate_quotas, resolve_mode
from dataprune.selection import select_ccs, select_d2, select_hardest, select_per_class, select_random
from dataprune.semdedup import semdedup


# ----------------------------------------------------------------------------- stage 1
def test_exact_dupes_tabular_label_aware():
    df = pd.DataFrame({"a": [1, 1, 1, 2, 2], "b": ["x", "x", "x", "y", "y"]})
    y = np.array([0, 0, 1, 0, 0])
    keep = exact_duplicate_mask_tabular(df, y)
    # row1 duplicates row0 (same label) -> dropped; row2 same features but different label -> kept; row4 dup of row3
    assert keep.tolist() == [True, False, True, True, False]


def test_exact_dupes_images():
    rng = np.random.default_rng(0)
    im = rng.integers(0, 255, (5, 8, 8), dtype=np.uint8)
    imgs = np.concatenate([im, im[:2]])
    y = np.array([0, 1, 0, 1, 0, 0, 1])
    keep = exact_duplicate_mask_images(imgs, y)
    assert keep.tolist() == [True] * 5 + [False, False]
    # same pixels, different label -> kept
    y2 = y.copy(); y2[5] = 1
    assert exact_duplicate_mask_images(imgs, y2)[5]


def test_phash_catches_brightness_shift_not_different_images():
    rng = np.random.default_rng(1)
    base = (rng.random((20, 32, 32)) * 200).astype(np.uint8)
    shifted = np.clip(base[:3].astype(int) + 20, 0, 255).astype(np.uint8)
    imgs = np.concatenate([base, shifted])
    y = np.zeros(len(imgs), dtype=int)
    keep = phash_duplicate_mask(imgs, y)
    assert not keep[20:].any()          # shifted copies removed
    assert keep[:20].all()              # originals kept
    assert isinstance(dhash(base[0]), int)


# ----------------------------------------------------------------------------- stage 2
def test_semdedup_removes_near_duplicates_only_same_class():
    rng = np.random.default_rng(0)
    E = rng.normal(size=(300, 32)).astype(np.float32)
    y = rng.integers(0, 3, 300)
    near = E[:30] + rng.normal(scale=1e-3, size=(30, 32)).astype(np.float32)
    E2 = np.concatenate([E, near]); y2 = np.concatenate([y, y[:30]])
    keep, info = semdedup(E2, y2, threshold=0.98, max_remove_frac=0.5, seed=0)
    assert info["n_removed"] >= 28           # ~all 30 injected near-copies found
    assert keep[:300].sum() >= 270 or keep[300:].sum() <= 3   # one of each pair survives, not both
    pair_both_kept = sum(keep[i] and keep[300 + i] for i in range(30))
    assert pair_both_kept <= 2
    # near copies with a DIFFERENT label are not duplicates
    y3 = y2.copy(); y3[300:] = (y3[300:] + 1) % 3
    keep3, info3 = semdedup(E2, y3, threshold=0.98, max_remove_frac=0.5, seed=0)
    assert info3["n_removed"] <= 3


def test_semdedup_cap():
    rng = np.random.default_rng(0)
    E = rng.normal(size=(100, 8)).astype(np.float32)
    E = np.concatenate([E, E + 1e-4])
    y = np.zeros(200, dtype=int)
    keep, info = semdedup(E, y, threshold=0.9, max_remove_frac=0.1, seed=0)
    assert info["capped"] and info["n_removed"] == 20


# ----------------------------------------------------------------------------- stage 3
def _noisy_blobs(n=1500, flip=0.1, seed=0):
    X, y = make_classification(n_samples=n, n_features=10, n_informative=8, n_redundant=0, n_classes=3,
                               n_clusters_per_class=1, class_sep=2.5, flip_y=0.0, random_state=seed)
    rng = np.random.default_rng(seed)
    flipped = rng.random(n) < flip
    y_noisy = y.copy()
    y_noisy[flipped] = (y[flipped] + rng.integers(1, 3, flipped.sum())) % 3
    return X.astype(np.float32), y_noisy, flipped


def test_label_noise_detection_precision_recall():
    X, y, flipped = _noisy_blobs()
    P = oof_probabilities(X, y, 3, lambda s: make_oof_model("image", s), folds=5, seed=0)
    mask, info, _ = find_label_issues(P, y, max_remove_frac=0.3, max_remove_frac_per_class=0.5)
    tp = (mask & flipped).sum()
    assert tp / max(mask.sum(), 1) > 0.7       # precision
    assert tp / flipped.sum() > 0.6            # recall


def test_label_noise_respects_caps():
    X, y, flipped = _noisy_blobs(flip=0.3)
    P = oof_probabilities(X, y, 3, lambda s: make_oof_model("image", s), folds=5, seed=0)
    mask, info, _ = find_label_issues(P, y, max_remove_frac=0.05, max_remove_frac_per_class=0.5)
    assert mask.sum() <= int(0.05 * len(y))
    mask2, _, _ = find_label_issues(P, y, max_remove_frac=0.9, max_remove_frac_per_class=0.1)
    for c in range(3):
        assert (mask2 & (y == c)).sum() <= int(0.1 * (y == c).sum())


# ----------------------------------------------------------------------------- stage 4
@pytest.mark.parametrize("proxy,modality", [("hgb", "tabular"), ("mlp", "image")])
def test_difficulty_flags_flipped_labels(proxy, modality):
    X, y, flipped = _noisy_blobs(n=1200, flip=0.1)
    ds = score_difficulty(X, y, 3, modality=modality, proxy=proxy, epochs=20, seeds=2, seed=0, metric="datamap")
    assert ds.difficulty[flipped].mean() > ds.difficulty[~flipped].mean() + 0.1
    for arr in (ds.el2n, ds.loss, ds.forget, ds.confidence, ds.variability, ds.difficulty):
        assert arr.shape == (len(y),) and np.isfinite(arr).all()
    assert ds.el2n[flipped].mean() > ds.el2n[~flipped].mean()


def test_difficulty_metrics_available():
    X, y, _ = _noisy_blobs(n=600)
    for m in ("combined", "datamap", "el2n", "loss", "forget"):
        ds = score_difficulty(X, y, 3, modality="tabular", epochs=8, seeds=1, metric=m)
        assert ds.metric == m and np.isfinite(ds.difficulty).all()


# ----------------------------------------------------------------------------- stage 5
def test_selectors_exact_budget_unique_in_range():
    rng = np.random.default_rng(0)
    n = 500
    score = rng.random(n)
    geom = rng.normal(size=(n, 6))
    for budget in (1, 50, 250, 499, 500):
        for sel in (select_random(n, budget, np.random.default_rng(1)),
                    select_hardest(score, budget),
                    select_ccs(score, budget, np.random.default_rng(1), beta=0.1, n_strata=20),
                    select_d2(score, geom, budget, k=8)):
            assert len(sel) == budget, (budget, len(sel))
            assert len(np.unique(sel)) == budget
            assert sel.min() >= 0 and sel.max() < n


def test_ccs_covers_score_range_while_hardest_does_not():
    rng = np.random.default_rng(0)
    score = rng.random(1000)
    b = 100
    hard = select_hardest(score, b)
    ccs = select_ccs(score, b, np.random.default_rng(0), beta=0.0, n_strata=20)
    assert score[hard].min() > 0.85            # hardest-only is stuck at the top
    assert score[ccs].min() < 0.2 and score[ccs].max() > 0.8   # CCS spans the range


def test_ccs_beta_drops_hardest():
    score = np.linspace(0, 1, 1000)
    sel = select_ccs(score, 100, np.random.default_rng(0), beta=0.2, n_strata=10)
    assert score[sel].max() <= np.sort(score)[-200] + 1e-9


def test_d2_deterministic_and_spreads_out():
    rng = np.random.default_rng(0)
    # two tight clusters, identical difficulty inside them: D2 should not take all points from one cluster
    a = rng.normal(0, 0.05, (100, 2)); b = rng.normal(5, 0.05, (100, 2))
    geom = np.concatenate([a, b]); score = np.ones(200)
    s1 = select_d2(score, geom, 20, k=10); s2 = select_d2(score, geom, 20, k=10)
    assert (s1 == s2).all()
    n_a = (s1 < 100).sum()
    assert 5 <= n_a <= 15


# ----------------------------------------------------------------------------- stage 6
def test_quotas_proportional_sum_and_bounds():
    orig = {0: 700, 1: 200, 2: 100}
    q = allocate_quotas(orig, orig, 400, "proportional")
    assert sum(q.values()) == 400
    assert q == {0: 280, 1: 80, 2: 40}


def test_quotas_balanced_cap_keeps_minority_whole():
    orig = {0: 9000, 1: 500, 2: 500}
    q = allocate_quotas(orig, orig, 3000, "balanced_cap")
    assert sum(q.values()) == 3000
    assert q[1] == 500 and q[2] == 500 and q[0] == 2000


def test_quotas_redistribute_when_class_short():
    orig = {0: 500, 1: 500}
    avail = {0: 500, 1: 50}          # class 1 lost most rows during cleaning
    q = allocate_quotas(orig, avail, 400, "proportional")
    assert sum(q.values()) == 400 and q[1] == 50 and q[0] == 350


def test_quotas_never_exceed_available_and_min_per_class():
    orig = {0: 1000, 1: 3}
    q = allocate_quotas(orig, orig, 10, "proportional", min_per_class=1)
    assert sum(q.values()) == 10 and q[1] >= 1
    q = allocate_quotas(orig, orig, 5000, "proportional")
    assert q == {0: 1000, 1: 3}


def test_resolve_mode_auto():
    assert resolve_mode("auto", {0: 900, 1: 100}, 5.0) == "balanced_cap"
    assert resolve_mode("auto", {0: 600, 1: 400}, 5.0) == "proportional"


def test_select_per_class_obeys_quotas():
    rng = np.random.default_rng(0)
    y = np.array([0] * 300 + [1] * 100)
    d = rng.random(400); g = rng.normal(size=(400, 4))
    cfg = PruneConfig()
    for name in ("random", "el2n", "ccs", "d2"):
        sel = select_per_class(name, y, {0: 60, 1: 20}, d, d, g, cfg, np.random.default_rng(0))
        assert (y[sel] == 0).sum() == 60 and (y[sel] == 1).sum() == 20 and len(np.unique(sel)) == 80


# ----------------------------------------------------------------------------- energy
def test_energy_meter_and_break_even():
    m = EnergyMeter(10.0)
    with m.stage("a"):
        sum(i * i for i in range(200000))
    assert m.stages["a"].cpu_s > 0 and m.joules() == pytest.approx(m.stages["a"].cpu_s * 10.0)
    assert break_even_runs(100, 50, 20) == pytest.approx(100 / 30)
    assert break_even_runs(100, 20, 50) == float("inf")


# ----------------------------------------------------------------------------- codecarbon backend
def test_codecarbon_backend_with_fake_module(monkeypatch):
    import sys, types

    class FakeData:  # mimics codecarbon's final_emissions_data
        energy_consumed = 0.002

    class FakeTracker:
        def __init__(self, **kw): self.kw = kw
        def start(self): pass
        def stop(self):
            self.final_emissions_data = FakeData()
            return 0.001

    fake = types.ModuleType("codecarbon"); fake.OfflineEmissionsTracker = FakeTracker
    monkeypatch.setitem(sys.modules, "codecarbon", fake)
    m = EnergyMeter(10.0, backend="codecarbon", grid_kg_per_kwh=0.5, price_per_kwh=10.0)
    assert m.backend_used == "codecarbon"
    with m.stage("a"):
        pass
    assert m.kwh() == pytest.approx(0.002) and m.co2_kg() == pytest.approx(0.001) and m.cost() == pytest.approx(0.02)
    assert m.summary()["backend_used"] == "codecarbon"


def test_codecarbon_missing_falls_back_to_estimate(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "codecarbon", None)     # makes `import codecarbon` raise ImportError
    m = EnergyMeter(10.0, backend="codecarbon")
    assert m.backend_used == "estimate" and "unavailable" in m.backend_note
    with m.stage("a"):
        sum(i for i in range(100000))
    assert m.kwh() > 0 and m.co2_kg() == pytest.approx(m.kwh() * 0.7)


def test_savings_card_math():
    from dataprune.energy import savings_card
    c = savings_card(overhead_kwh=1.0, full_train_kwh=5.0, pruned_train_kwh=2.0, grid_kg_per_kwh=0.5,
                     price_per_kwh=10.0, runs=4)
    assert c["per_run_saving_kwh"] == 3.0 and c["break_even_runs"] == pytest.approx(1 / 3)
    assert c["net_saving_kwh"] == pytest.approx(11.0) and c["net_saving_co2_kg"] == pytest.approx(5.5)
    assert savings_card(1.0, 2.0, 3.0)["break_even_runs"] is None


def test_pipeline_reports_co2_and_cost():
    X, y = make_classification(n_samples=400, n_features=6, random_state=0)
    r = prune(pd.DataFrame(X), y, PruneConfig(keep_ratio=0.5, selector="random", grid_kg_co2_per_kwh=0.4))
    e = r.energy
    assert e["backend_used"] == "estimate" and e["total_co2_kg_est"] == pytest.approx(e["total_kwh_est"] * 0.4)
    assert "total_cost_est" in e

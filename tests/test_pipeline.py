import json

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_digits, make_classification

from dataprune import PruneConfig, prune, run_experiment


def _tab(n=2500, seed=0, weights=None):
    X, y = make_classification(n_samples=n, n_features=10, n_informative=6, weights=weights, flip_y=0.0,
                               random_state=seed)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(10)]), y


def test_tabular_target_size_and_class_proportions():
    df, y = _tab(weights=[0.7, 0.3])
    r = prune(df, y, PruneConfig(keep_ratio=0.4, selector="ccs"))
    k = r.keep_indices
    assert len(k) == len(np.unique(k))
    assert abs(len(k) - round(0.4 * len(y))) <= 0           # exact target reached
    kept_share = np.bincount(y[k]) / len(k)
    full_share = np.bincount(y) / len(y)
    assert np.abs(kept_share - full_share).max() < 0.01     # constant-proportion rule
    assert r.selector_used == "ccs" and r.race_info is None


@pytest.mark.parametrize("sel", ["random", "el2n", "ccs", "d2", "auto"])
def test_all_selectors_run_end_to_end(sel):
    df, y = _tab(n=1200)
    r = prune(df, y, PruneConfig(keep_ratio=0.3, selector=sel, proxy_seeds=1, race_repeats=1))
    assert len(r.keep_indices) == round(0.3 * 1200)
    assert r.selector_used in ("random", "el2n", "ccs", "d2")
    if sel == "auto":
        assert r.race_table is not None and set(r.race_table["selector"]) == {"el2n", "ccs", "d2", "random"}


def test_exact_duplicates_removed_and_accounted_for():
    df, y = _tab(n=1000)
    dup_idx = np.arange(100)
    df2 = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)
    y2 = np.concatenate([y, y[dup_idx]])
    r = prune(df2, y2, PruneConfig(keep_ratio=0.5, selector="random"))
    assert len(r.removed_by_stage["1_hash"]) == 100
    assert set(r.removed_by_stage["1_hash"]) == set(range(1000, 1100))   # later copies are the ones dropped
    assert len(r.keep_indices) == 550                                      # target is relative to ORIGINAL size
    # every original row is either kept or removed by exactly one stage
    all_removed = np.concatenate(list(r.removed_by_stage.values()))
    assert len(np.unique(all_removed)) == len(all_removed)
    assert len(all_removed) + len(r.keep_indices) == len(y2)
    assert not (set(all_removed) & set(r.keep_indices))


def test_label_noise_stage_removes_mostly_flipped_rows():
    X, y = make_classification(n_samples=2000, n_features=10, n_informative=8, n_redundant=0, n_classes=2,
                               class_sep=2.5, flip_y=0.0, random_state=3)
    rng = np.random.default_rng(3)
    flip = rng.random(len(y)) < 0.08
    yn = np.where(flip, 1 - y, y)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(10)])
    r = prune(df, yn, PruneConfig(keep_ratio=0.5, selector="random"))
    rem = r.removed_by_stage["3_label_noise"]
    assert len(rem) > 50
    assert flip[rem].mean() > 0.7


def test_imbalanced_auto_quota_keeps_minority():
    df, y = _tab(n=4000, weights=[0.95, 0.05])
    r = prune(df, y, PruneConfig(keep_ratio=0.3, selector="random", do_label_noise=False))
    assert r.quota_mode == "balanced_cap"
    minority_total = int((y == 1).sum())
    kept_minority = int((y[r.keep_indices] == 1).sum())
    assert kept_minority >= 0.95 * minority_total
    assert len(r.keep_indices) == round(0.3 * len(y))


def test_target_larger_than_cleaned_set_warns():
    df, y = _tab(n=500)
    df2 = pd.concat([df] * 3, ignore_index=True)       # 1500 rows, only 500 unique
    y2 = np.tile(y, 3)
    r = prune(df2, y2, PruneConfig(keep_ratio=0.9, selector="random", do_label_noise=False))
    assert len(r.keep_indices) == 500
    assert any("larger than" in w for w in r.warnings)
    assert r.selector_used == "keep_all"


def test_images_digits_with_planted_duplicates():
    d = load_digits()
    imgs, y = d.images.copy(), d.target.copy()
    rng = np.random.default_rng(0)
    near = imgs[:60] + rng.normal(scale=0.01, size=imgs[:60].shape)
    imgs2 = np.concatenate([imgs, imgs[60:120], near])       # 60 exact + 60 near copies
    y2 = np.concatenate([y, y[60:120], y[:60]])
    r = prune(imgs2, y2, PruneConfig(keep_ratio=0.4, selector="random", do_label_noise=False))
    assert len(r.removed_by_stage["1_hash"]) >= 60
    assert len(r.removed_by_stage["2_semdedup"]) >= 55        # ~one member of each of the 60 near-copy pairs
    removed = set(r.removed_by_stage["2_semdedup"])
    n0 = len(imgs)
    both_survive = 0
    for i in range(60):
        a, b = i, n0 + 60 + i                                 # original and its near copy
        both_survive += (a not in removed) and (b not in removed)
    assert both_survive <= 5
    assert r.modality == "image"
    assert len(r.keep_indices) == round(0.4 * len(y2))


def test_json_report_roundtrip(tmp_path):
    df, y = _tab(n=800)
    r = prune(df, y, PruneConfig(keep_ratio=0.5, selector="auto", race_repeats=1, proxy_seeds=1))
    p = tmp_path / "rep.json"
    r.save_json(str(p))
    data = json.loads(p.read_text())
    assert data["n_kept"] == len(r.keep_indices) and "race" in data and "energy" in data


def test_invalid_config_and_inputs():
    df, y = _tab(n=200)
    with pytest.raises(ValueError):
        prune(df, y, PruneConfig(keep_ratio=0.0))
    with pytest.raises(ValueError):
        prune(df, y, PruneConfig(keep_ratio=1.5))
    with pytest.raises(ValueError):
        prune(df, y[:-1], PruneConfig())
    with pytest.raises(ValueError):
        PruneConfig(selector="nope").validate()


def test_mixed_type_csv_like_frame():
    rng = np.random.default_rng(0)
    n = 1500
    df = pd.DataFrame({
        "age": rng.integers(18, 80, n).astype(float),
        "job": rng.choice(["a", "b", "c", None], n),
        "city": rng.choice([f"c{i}" for i in range(60)], n),     # high-cardinality
        "flag": rng.random(n) > 0.5,
        "income": rng.normal(50, 10, n),
    })
    df.loc[rng.random(n) < 0.05, "income"] = np.nan
    y = ((df["age"] > 45) ^ (df["job"] == "a")).astype(int).to_numpy()
    r = prune(df, y, PruneConfig(keep_ratio=0.5, selector="ccs"))
    assert len(r.keep_indices) == 750


def test_run_experiment_proof_no_test_leakage_and_report():
    df, y = _tab(n=1800, weights=[0.6, 0.4])
    out = run_experiment(df, y, PruneConfig(keep_ratio=0.4, selector="auto", race_repeats=1, proxy_seeds=1),
                         eval_seeds=(0, 1))
    res, table, verdict = out["result"], out["table"], out["verdict"]
    assert set(table["method"]) == {"full", "pruned", "random_stratified", "random_uniform"}
    n_train = int(table.set_index("method").loc["full", "n_train"])
    assert n_train == int(round(0.8 * 1800))
    assert int(table.set_index("method").loc["pruned", "n_train"]) == len(res.keep_indices)
    assert int(table.set_index("method").loc["random_uniform", "n_train"]) == len(res.keep_indices)
    assert 0.5 < verdict["pruned_retained_vs_full"] <= 1.2
    assert isinstance(verdict["fidelity_pass"], bool)
    assert "break_even_training_runs" in verdict
    # pruning only ever saw training rows: indices are within the training split
    assert res.keep_indices.max() < n_train


def test_run_experiment_images():
    d = load_digits()
    out = run_experiment(d.images, d.target, PruneConfig(keep_ratio=0.4, selector="ccs", proxy_seeds=1),
                         eval_seeds=(0,))
    assert out["verdict"]["pruned_retained_vs_full"] > 0.8

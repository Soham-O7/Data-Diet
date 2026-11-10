"""Messy real-world inputs: the pipeline should either work or fail with a clear message."""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from dataprune import PruneConfig, prune, run_experiment
from dataprune.cli import main


def _base(n=500, f=6, seed=0, classes=2):
    X, y = make_classification(n_samples=n, n_features=f, n_informative=min(f, 4), n_classes=classes,
                               n_clusters_per_class=1, random_state=seed)
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(f)]), y


def test_missing_values_and_strings():
    X, y = _base()
    X["cat"] = np.where(np.arange(len(X)) % 3 == 0, "a", "b")
    X["hi"] = [f"id{i % 120}" for i in range(len(X))]          # high-cardinality categorical
    X.iloc[::5, 0] = np.nan
    X.loc[::11, "cat"] = None
    res = prune(X, y, PruneConfig(keep_ratio=0.5))
    assert len(res.keep_indices) == 250


def test_many_columns():
    X, y = _base(n=400, f=300)
    res = prune(X, y, PruneConfig(keep_ratio=0.4, selector="ccs"))
    assert len(res.keep_indices) == 160


def test_single_class_target_is_handled():
    X, _ = _base(n=100)
    y = np.zeros(100, dtype=int)
    res = prune(X, y, PruneConfig(keep_ratio=0.5))
    assert len(res.keep_indices) == 50


@pytest.mark.parametrize("n", [4, 6, 12, 30])
def test_very_small_files(n):
    X, y = _base(n=max(n, 20))
    X, y = X.iloc[:n], y[:n]
    res = prune(X, y, PruneConfig(keep_ratio=0.5))
    assert 1 <= len(res.keep_indices) <= n


def test_too_small_raises_clear_error():
    X, y = _base(n=20)
    with pytest.raises(ValueError, match="at least 4"):
        prune(X.iloc[:3], y[:3])


def test_string_labels_and_rare_class():
    X, y = _base(n=400)
    lab = np.where(y == 0, "no", "yes").astype(object)
    lab[:2] = "rare"
    res = prune(X, lab, PruneConfig(keep_ratio=0.5))
    kept_rare = (lab[res.keep_indices] == "rare").sum()
    assert kept_rare >= 1


def test_all_duplicates():
    X, y = _base(n=50)
    X = pd.concat([X.iloc[:1]] * 60, ignore_index=True)
    y = np.zeros(60, dtype=int)
    res = prune(X, y, PruneConfig(keep_ratio=0.5))
    assert len(res.keep_indices) >= 1


def test_constant_feature_columns():
    X, y = _base()
    X["const"] = 1.0
    res = prune(X, y, PruneConfig(keep_ratio=0.5))
    assert len(res.keep_indices) == 250


def test_multiclass_experiment_runs():
    X, y = _base(n=900, f=8, classes=4)
    out = run_experiment(X, y, PruneConfig(keep_ratio=0.5), eval_seeds=(0,))
    assert "pruned" in set(out["table"]["method"])


def test_cli_missing_target_values_message(tmp_path):
    X, y = _base(n=100)
    df = X.assign(label=y.astype(float))
    df.loc[0, "label"] = np.nan
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing values"):
        main(["run", "--csv", str(p), "--target", "label", "--keep", "0.5", "--out", str(tmp_path / "o")])

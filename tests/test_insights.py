import json
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from dataprune import (ImpactTracker, PruneConfig, class_warnings, compare_training, health_score, prune,
                       recommend_keep_ratio, removal_export, run_experiment)
from dataprune.cli import main
from dataprune.impact import measure_command


def _df(n=800, seed=0, flip=0.0, weights=None):
    X, y = make_classification(n_samples=n, n_features=8, n_informative=5, weights=weights, random_state=seed)
    if flip:
        r = np.random.default_rng(seed)
        f = r.random(n) < flip
        y = np.where(f, 1 - y, y)
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(8)]), y


def test_removal_export_accounts_for_every_row():
    X, y = _df(flip=0.08)
    X = pd.concat([X, X.iloc[:40]], ignore_index=True)
    y = np.concatenate([y, y[:40]])
    res = prune(X, y, PruneConfig(keep_ratio=0.5, selector="ccs"))
    ex = removal_export(res, X)
    assert len(ex) == len(y) - len(res.keep_indices)
    assert set(ex["original_index"]).isdisjoint(set(res.keep_indices))
    assert set(ex["stage"]) >= {"1_hash", "5_selection"}
    assert ex.loc[ex.stage == "1_hash", "score"].isna().all()
    assert ex.loc[ex.stage == "5_selection", "score"].notna().all()
    if (ex.stage == "3_label_noise").any():
        assert (ex.loc[ex.stage == "3_label_noise", "score"] < 0).all()
    assert "x0" in ex.columns


def test_class_warnings_flags_squeezed_minority_and_is_quiet_otherwise():
    X, y = _df(n=1000, weights=[0.95, 0.05])
    quiet = prune(X, y, PruneConfig(keep_ratio=0.5, quota_mode="balanced_cap"))
    assert class_warnings(quiet) == []
    res = prune(X, y, PruneConfig(keep_ratio=0.5))
    res.keep_indices = res.keep_indices[res.y_int[res.keep_indices] == 0]    # force minority to vanish
    w = class_warnings(res)
    assert any("NO rows left" in m for m in w)


def test_health_score_orders_dirty_below_clean():
    X, y = _df()
    clean = health_score(X, y)["score"]
    dirty_X = pd.concat([X, X.iloc[:200]], ignore_index=True)
    dirty_X.iloc[::7, 0] = np.nan
    dirty_y = np.concatenate([y, y[:200]])
    h = health_score(dirty_X, dirty_y)
    assert h["duplicate_rows"] >= 100 and h["missing_cells"] > 0
    assert 0 <= h["score"] < clean <= 100


def test_health_score_detects_label_conflicts():
    X = pd.DataFrame({"a": [1, 1, 2, 3], "b": [0, 0, 1, 1]})
    assert health_score(X, np.array([0, 1, 0, 1]))["label_conflict_groups"] == 1


def test_recommend_keep_ratio_returns_sweep_and_stops_on_failure():
    X, y = _df(n=1200)
    out = recommend_keep_ratio(X, y, PruneConfig(seed=0), ratios=(0.8, 0.5, 0.02), tolerance=0.03,
                               eval_seeds=(0,))
    sw = out["sweep"]
    assert list(sw["keep_ratio"])[0] == 0.8
    assert sw["within_tolerance"].iloc[:-1].all() or not sw["within_tolerance"].iloc[-1]
    if out["recommended_keep_ratio"] is not None:
        assert out["recommended_keep_ratio"] in set(sw["keep_ratio"][sw["within_tolerance"]])


def test_impact_tracker_totals(tmp_path):
    t = ImpactTracker(str(tmp_path / "log.json"))
    t.log("a", overhead_kwh=1e-5, saved_kwh_per_run=4e-6, runs=10)
    t.log("b", overhead_kwh=2e-5, saved_kwh_per_run=1e-6, runs=5)
    tot = t.totals()
    assert tot["n_logged"] == 2
    assert tot["net_kwh"] == pytest.approx((4e-5 - 1e-5) + (5e-6 - 2e-5))
    assert json.loads((tmp_path / "log.json").read_text())[1]["label"] == "b"


def test_impact_tracker_from_experiment(tmp_path):
    X, y = _df(n=600)
    out = run_experiment(X, y, PruneConfig(keep_ratio=0.5), eval_seeds=(0,))
    rec = ImpactTracker(str(tmp_path / "l.json")).log_experiment("run1", out, runs=3)
    assert rec["overhead_kwh"] > 0 and "net_cost" in rec


def test_measure_and_compare_commands():
    busy = [sys.executable, "-c", "sum(i*i for i in range(6_000_000))"]
    quick = [sys.executable, "-c", "pass"]
    m = measure_command(busy)
    assert m["returncode"] == 0 and m["cpu_s"] > 0 and m["kwh_est"] > 0
    cmp = compare_training(busy, quick)
    assert cmp["full"]["cpu_s"] > cmp["pruned"]["cpu_s"]
    assert cmp["savings_card"]["per_run_saving_kwh"] > 0
    with pytest.raises(RuntimeError):
        compare_training([sys.executable, "-c", "raise SystemExit(3)"], quick)


def test_cli_writes_removed_rows_and_warnings(tmp_path):
    X, y = _df(n=600, flip=0.05)
    df = X.assign(label=y)
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    out = tmp_path / "o"
    assert main(["run", "--csv", str(p), "--target", "label", "--keep", "0.5", "--selector", "random",
                 "--no-proof", "--out", str(out)]) == 0
    rm = pd.read_csv(out / "removed_rows.csv")
    kept = np.loadtxt(out / "kept_indices.txt", dtype=int)
    assert len(rm) + len(kept) == len(df)
    rep = json.loads((out / "report.json").read_text())
    assert "class_warnings" in rep and "health" in rep


def test_cli_recommend(tmp_path, capsys):
    X, y = _df(n=800)
    p = tmp_path / "d.csv"
    X.assign(label=y).to_csv(p, index=False)
    assert main(["recommend", "--csv", str(p), "--target", "label", "--tolerance", "0.05"]) == 0
    assert "recommended keep ratio" in capsys.readouterr().out
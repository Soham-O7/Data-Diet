import json

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from dataprune.cli import main


def test_cli_csv_roundtrip(tmp_path):
    X, y = make_classification(n_samples=600, n_features=6, n_informative=4, random_state=0)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(6)])
    df["id"] = range(len(df))
    df["label"] = y
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    out = tmp_path / "out"
    rc = main(["run", "--csv", str(p), "--target", "label", "--drop", "id", "--keep", "0.5",
               "--selector", "ccs", "--out", str(out)])
    assert rc == 0
    pruned = pd.read_csv(out / "pruned.csv")
    assert len(pruned) == 300 and list(pruned.columns) == list(df.columns)
    rep = json.loads((out / "report.json").read_text())
    assert rep["n_kept"] == 300 and "proof" in rep
    idx = np.loadtxt(out / "kept_indices.txt", dtype=int)
    assert len(idx) == 300 and len(set(idx)) == 300

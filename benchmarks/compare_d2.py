"""Compare D2 variants (own 'simplified' vs 'paper'-faithful) with stages 1-3 OFF, same features and test set.
Usage: python benchmarks/compare_d2.py [--datasets ...] [--keeps 0.2 0.4 0.6] [--seeds 3] [--gamma-r 0.9]"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import OFFLINE, load  # noqa: E402

from dataprune import PruneConfig, prune  # noqa: E402
from dataprune.embedders import get_embedder  # noqa: E402
from dataprune.models import make_eval_model  # noqa: E402
from dataprune.preprocess import TabularPreprocessor  # noqa: E402
from dataprune.verify import _fit_eval  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=OFFLINE)
    ap.add_argument("--keeps", nargs="+", type=float, default=[0.2, 0.4, 0.6])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--gamma-r", type=float, default=0.9)
    ap.add_argument("--embedder", default="pixel")
    ap.add_argument("--out", default="benchmark_out/d2_comparison.csv")
    a = ap.parse_args()
    rows = []
    for name in a.datasets:
        d = load(name, seed=0)
        mod = d["modality"]
        Xtr, ytr, Xte, yte = d["X_train"], np.asarray(d["y_train"]), d["X_test"], np.asarray(d["y_test"])
        cl = np.unique(ytr); lut = {c: i for i, c in enumerate(cl)}
        yi = np.array([lut[c] for c in ytr]); yti = np.array([lut[c] for c in yte])
        ref = TabularPreprocessor().fit(Xtr) if mod == "tabular" else get_embedder(a.embedder).fit(Xtr)
        Ztr, Zte = ref.transform(Xtr), ref.transform(Xte)
        emb = get_embedder(a.embedder) if mod == "image" else None
        full = np.mean([_fit_eval(make_eval_model(mod, s), Ztr, yi, Zte, yti, len(cl), 15.0)["acc"] for s in range(a.seeds)])
        for keep in a.keeps:
            for mode in ("simplified", "paper"):
                accs = []
                for s in range(a.seeds):
                    cfg = PruneConfig(keep_ratio=keep, seed=s, selector="d2", d2_mode=mode, d2_paper_gamma_r=a.gamma_r,
                                      do_hash=False, do_semdedup=False, do_label_noise=False)
                    idx = prune(Xtr, ytr, cfg, modality=mod, embedder=emb).keep_indices
                    accs.append(_fit_eval(make_eval_model(mod, s), Ztr[idx], yi[idx], Zte, yti, len(cl), 15.0)["acc"])
                rows.append(dict(dataset=name, keep=keep, d2_mode=mode, retained_mean=np.mean(accs) / full,
                                 retained_sd=np.std(accs) / full))
                print(f"{name:<24} keep {keep:.0%} {mode:<10} retained {np.mean(accs)/full:.1%} ± {np.std(accs)/full:.1%}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(df.pivot_table(index=["dataset", "keep"], columns="d2_mode", values="retained_mean").round(4).to_string())


if __name__ == "__main__":
    main()

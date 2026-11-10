"""Item 3: re-tune the SemDeDup cosine threshold for a given embedder (e.g. DINOv2).

For each threshold: how many rows stage 2 removes, and the test accuracy of a model trained on what is left
(vs training on everything). Pick the LOWEST threshold (most removal) whose accuracy stays within tolerance.

  python benchmarks/tune_semdedup.py --dataset mnist --embedder dinov2 --max-rows 10000
  python benchmarks/tune_semdedup.py --images my_folder/ --embedder dinov2
  python benchmarks/tune_semdedup.py --dataset digits_corrupted            # works offline with the pixel embedder
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataprune.embedders import get_embedder  # noqa: E402
from dataprune.models import make_eval_model  # noqa: E402
from dataprune.semdedup import semdedup  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", help="name from benchmarks/datasets.py (image datasets)")
    src.add_argument("--images", help="folder with one sub-folder per class")
    ap.add_argument("--embedder", default="pixel", choices=["pixel", "dinov2"])
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[0.90, 0.93, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995])
    ap.add_argument("--tolerance", type=float, default=0.01, help="max accuracy loss vs no dedup")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="benchmark_out/semdedup_threshold.csv")
    a = ap.parse_args()

    if a.dataset:
        from datasets import load
        d = load(a.dataset, seed=a.seed, max_rows=a.max_rows)
        if d["modality"] != "image":
            ap.error("pick an image dataset")
        Xtr, ytr, Xte, yte = d["X_train"], np.asarray(d["y_train"]), d["X_test"], np.asarray(d["y_test"])
    else:
        from dataprune.io import load_image_folder
        imgs, labels, _ = load_image_folder(a.images)
        idx = np.arange(len(labels))
        tr, te = train_test_split(idx, test_size=0.25, stratify=labels, random_state=a.seed)
        Xtr, ytr, Xte, yte = imgs[tr], labels[tr], imgs[te], labels[te]

    cl = np.unique(ytr); lut = {c: i for i, c in enumerate(cl)}
    y = np.array([lut[c] for c in ytr]); yt = np.array([lut[c] for c in yte])
    emb = get_embedder(a.embedder)
    Z = emb.fit_transform(Xtr); Zt = emb.transform(Xte)
    base = make_eval_model("image", a.seed).fit(Z, y).score(Zt, yt)
    rows = [dict(threshold=None, n_kept=len(y), removed_frac=0.0, acc=base, loss_vs_full=0.0)]
    for t in a.thresholds:
        keep, info = semdedup(Z, y, threshold=t, max_remove_frac=1.0, seed=a.seed)
        acc = make_eval_model("image", a.seed).fit(Z[keep], y[keep]).score(Zt, yt)
        rows.append(dict(threshold=t, n_kept=int(keep.sum()), removed_frac=1 - keep.mean(), acc=acc,
                         loss_vs_full=base - acc))
    df = pd.DataFrame(rows)
    ok = df[(df.threshold.notna()) & (df.loss_vs_full <= a.tolerance)]
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if len(ok):
        best = ok.sort_values("threshold").iloc[0]
        print(f"\nsuggested semdedup_threshold = {best.threshold} (removes {best.removed_frac:.1%}, "
              f"accuracy loss {best.loss_vs_full:.4f} <= {a.tolerance})")
    else:
        print("\nno threshold stayed within tolerance; raise --thresholds toward 1.0 or loosen --tolerance")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    print("note: max_remove_frac is lifted to 1.0 here so the sweep shows the raw effect; the pipeline caps stage 2 at 25%.")


if __name__ == "__main__":
    main()

"""Command line:  python -m dataprune run --csv data.csv --target label --keep 0.4 --out out/
                  python -m dataprune run --images folder/ --keep 0.4 --out out/ [--embedder dinov2] [--copy-images]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

from .config import PruneConfig
from .embedders import get_embedder
from .experiment import run_experiment
from .io import load_csv, load_image_folder
from .pipeline import prune, _jsonable


def _print_summary(res):
    s = res.summary()
    print(f"\nkept {s['n_kept']} of {s['n_original']} rows ({s['kept_fraction']:.1%}; target {s['target_keep_ratio']:.0%})")
    print(f"selector: {s['selector_used']}" + ("  (chosen automatically by selector race)" if s["selector_chosen_automatically"] else ""))
    print(f"quota mode: {s['quota_mode']}   quotas: {s['quotas']}")
    for st in s["stages"]:
        print(f"  {st['name']:<24} {st['n_in']:>8} -> {st['n_out']:<8} ({st['removed']:>6} removed, {st['seconds']:.2f}s)")
    e = s["energy"]
    print(f"pruning overhead: {e['total_cpu_s']:.1f} CPU-s ~ {e['total_kwh_est']:.2e} kWh (ESTIMATE @ {e['watts_per_core']} W/core)")
    for w in s["warnings"]:
        print("WARNING:", w)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dataprune")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="prune a dataset")
    src = r.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv")
    src.add_argument("--images", help="folder with one sub-folder per class")
    r.add_argument("--target", help="target column (CSV)")
    r.add_argument("--drop", nargs="*", default=[], help="CSV columns to ignore (ids etc.)")
    r.add_argument("--keep", type=float, required=True, help="fraction of rows to keep, e.g. 0.4")
    r.add_argument("--selector", default="auto", choices=["auto", "random", "el2n", "ccs", "d2"])
    r.add_argument("--quota-mode", default="auto", choices=["auto", "proportional", "balanced_cap"])
    r.add_argument("--embedder", default="pixel", choices=["pixel", "dinov2"])
    r.add_argument("--img-size", type=int, nargs=2, metavar=("H", "W"))
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--watts-per-core", type=float, default=15.0)
    r.add_argument("--no-proof", action="store_true", help="skip the full-vs-pruned-vs-random proof run")
    r.add_argument("--copy-images", action="store_true")
    r.add_argument("--out", default="prune_out")
    args = ap.parse_args(argv)

    cfg = PruneConfig(keep_ratio=args.keep, selector=args.selector, quota_mode=args.quota_mode, seed=args.seed,
                      cpu_watts_per_core=args.watts_per_core)
    os.makedirs(args.out, exist_ok=True)

    if args.csv:
        if not args.target:
            ap.error("--target is required with --csv")
        X, y = load_csv(args.csv, args.target, args.drop)
        res = prune(X, y.to_numpy(), cfg)
        full = pd.read_csv(args.csv)
        full.iloc[res.keep_indices].to_csv(os.path.join(args.out, "pruned.csv"), index=False)
        modality, Xp, yp, paths = "tabular", X, y.to_numpy(), None
    else:
        size = tuple(args.img_size) if args.img_size else None
        imgs, labels, paths = load_image_folder(args.images, size=size)
        emb = get_embedder(args.embedder)
        res = prune(imgs, labels, cfg, embedder=emb)
        with open(os.path.join(args.out, "kept_paths.txt"), "w") as f:
            f.write("\n".join(paths[i] for i in res.keep_indices))
        if args.copy_images:
            for i in res.keep_indices:
                dst = os.path.join(args.out, "pruned_images", labels[i])
                os.makedirs(dst, exist_ok=True)
                shutil.copy(paths[i], dst)
        modality, Xp, yp = "image", imgs, labels

    np.savetxt(os.path.join(args.out, "kept_indices.txt"), res.keep_indices, fmt="%d")
    summary = res.summary()
    _print_summary(res)

    if not args.no_proof and len(yp) >= 200:
        print("\nrunning proof (hold out 20%, prune the rest, train full vs pruned vs random) ...")
        out = run_experiment(Xp, yp, cfg, modality=modality,
                             embedder=get_embedder(args.embedder) if modality == "image" else None)
        print(out["table"].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        v = out["verdict"]
        print(f"\npruned retains {v['pruned_retained_vs_full']:.1%} of full accuracy "
              f"(random-stratified: {v['random_stratified_retained_vs_full']:.1%}); "
              f"worst per-class recall drop {v['worst_per_class_recall_drop_vs_full']:.3f}; "
              f"fidelity {'PASS' if v['fidelity_pass'] else 'FAIL'} "
              f"(thresholds {v['thresholds']})")
        summary["proof"] = {"table": out["table"].to_dict(orient="records"), "verdict": v}
        out["raw"].to_csv(os.path.join(args.out, "proof_raw.csv"), index=False)
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(summary, f, indent=2, default=_jsonable)
    print(f"\nwrote {args.out}/ (report.json, kept_indices.txt, " + ("pruned.csv" if args.csv else "kept_paths.txt") + ")")
    return 0


if __name__ == "__main__":
    sys.exit(main())

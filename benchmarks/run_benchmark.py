"""Benchmark: random vs EL2N-only vs CCS vs D2 vs full mix, at several keep ratios.

Method definitions
  random   : per-class-quota random selection with stages 1-3 OFF
             (class quotas are the project's signature rule, so every method shares them).
  el2n/ccs/d2 : stages 1-3 OFF, difficulty scoring + that selector + class quotas ("selector-only" baselines)
  clean_random / clean_el2n : stages 1-3 ON (hash, SemDeDup for images, confident learning) + random / EL2N selection
             (separates what CLEANING contributes from what the SELECTOR contributes)
  full_mix : all stages (hash, SemDeDup for images, confident learning, difficulty) + selector race

Fairness: every method is evaluated on the SAME feature matrix (fit once on the full, corrupted training split) and
the SAME clean held-out test set. Each (method, keep ratio) is repeated for `--seeds` pruning seeds, each followed by
a model fit with that seed; 'full' is fit once per seed on all training rows (corruption included).
Energy numbers are CPU-seconds x assumed watts/core (an ESTIMATE).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import NEEDS_DOWNLOAD, OFFLINE, load  # noqa: E402

from dataprune import PruneConfig, prune  # noqa: E402
from dataprune.embedders import CachedEmbedder, get_embedder  # noqa: E402
from dataprune.energy import break_even_runs  # noqa: E402
from dataprune.models import make_eval_model  # noqa: E402
from dataprune.preprocess import TabularPreprocessor  # noqa: E402
from dataprune.verify import _fit_eval  # noqa: E402

METHODS = ["random", "el2n", "ccs", "d2", "clean_random", "clean_el2n", "full_mix"]


EXTRA: dict = {}      # filled from CLI flags (semdedup threshold, energy backend, grid, price, d2 mode)


def method_config(method: str, keep: float, seed: int, watts: float) -> PruneConfig:
    base = dict(keep_ratio=keep, seed=seed, cpu_watts_per_core=watts, **EXTRA)
    if method == "random":
        return PruneConfig(selector="random", do_hash=False, do_semdedup=False, do_label_noise=False, **base)
    if method in ("el2n", "ccs", "d2"):
        return PruneConfig(selector=method, do_hash=False, do_semdedup=False, do_label_noise=False, **base)
    if method == "clean_random":      # stages 1-3 (cleaning) + random selection: isolates what cleaning alone buys
        return PruneConfig(selector="random", **base)
    if method == "clean_el2n":        # stages 1-3 + EL2N-only: the fairer EL2N baseline on corrupted data
        return PruneConfig(selector="el2n", **base)
    if method == "full_mix":
        return PruneConfig(selector="auto", **base)
    raise ValueError(method)


def run_dataset(name, keeps, seeds, watts, max_rows, embedder_name):
    d = load(name, seed=0, max_rows=max_rows)
    modality = d["modality"]
    Xtr, ytr, Xte, yte = d["X_train"], np.asarray(d["y_train"]), d["X_test"], np.asarray(d["y_test"])
    classes = np.unique(ytr)
    lut = {c: i for i, c in enumerate(classes)}
    y_int = np.array([lut[c] for c in ytr]); yte_int = np.array([lut[c] for c in yte])
    C = len(classes)

    # one shared featurisation for evaluation
    emb_cpu_s = 0.0
    if modality == "tabular":
        ref = TabularPreprocessor().fit(Xtr)
        emb = None
    elif embedder_name == "pixel":
        ref = get_embedder("pixel").fit(Xtr)
        emb = get_embedder("pixel")
    else:                                   # DINOv2 is stateless -> embed every image ONCE and reuse it everywhere
        emb = CachedEmbedder(get_embedder(embedder_name))
        ref = emb
    Ztr = ref.transform(Xtr)
    if isinstance(emb, CachedEmbedder):
        emb_cpu_s = emb.cpu_s               # one-time cost of embedding the train set; charged to every pruning run
        print(f"  embedded {emb.n_embedded} train images once: {emb.wall_s:.0f}s wall, {emb_cpu_s:.0f} CPU-s", flush=True)
    Zte = ref.transform(Xte)
    print(f"\n=== {name}: train {len(ytr)} rows, test {len(yte)} rows, {C} classes ({d['notes']}) ===", flush=True)

    rows = []
    full_runs = [_fit_eval(make_eval_model(modality, s), Ztr, y_int, Zte, yte_int, C, watts) for s in range(seeds)]
    full_acc = float(np.mean([r["acc"] for r in full_runs]))
    full_kwh = float(np.mean([r["fit_kwh_est"] for r in full_runs]))
    full_cpu = float(np.mean([r["fit_cpu_s"] for r in full_runs]))
    rows.append(dict(dataset=name, keep=1.0, method="full", seed=-1, n_train=len(ytr), acc=full_acc,
                     bal_acc=float(np.mean([r["bal_acc"] for r in full_runs])), retained=1.0,
                     prune_cpu_s=0.0, prune_kwh=0.0, train_cpu_s=full_cpu, train_kwh=full_kwh,
                     break_even_runs=np.nan, selector="-"))
    print(f"  full: acc {full_acc:.4f}  (fit {full_cpu:.2f} CPU-s)", flush=True)

    for keep in keeps:
        for method in METHODS:
            for s in range(seeds):
                cfg = method_config(method, keep, s, watts)
                t0 = time.perf_counter()
                res = prune(Xtr, ytr, cfg, modality=modality, embedder=emb)
                wall = time.perf_counter() - t0
                idx = res.keep_indices
                ev = _fit_eval(make_eval_model(modality, s), Ztr[idx], y_int[idx], Zte, yte_int, C, watts)
                over_kwh = res.energy["total_kwh_est"] + emb_cpu_s * watts / 3.6e6     # + one-time embedding cost
                be = break_even_runs(over_kwh * 3.6e6, full_kwh * 3.6e6, ev["fit_kwh_est"] * 3.6e6)
                rows.append(dict(dataset=name, keep=keep, method=method, seed=s, n_train=len(idx), acc=ev["acc"],
                                 bal_acc=ev["bal_acc"], retained=ev["acc"] / full_acc,
                                 prune_cpu_s=res.energy["total_cpu_s"] + emb_cpu_s, prune_kwh=over_kwh,
                                 train_cpu_s=ev["fit_cpu_s"], train_kwh=ev["fit_kwh_est"],
                                 break_even_runs=be if np.isfinite(be) else np.nan, selector=res.selector_used))
            sub = [r for r in rows if r["keep"] == keep and r["method"] == method]
            print(f"  keep {keep:.0%} {method:<9} acc {np.mean([r['acc'] for r in sub]):.4f} "
                  f"retained {np.mean([r['retained'] for r in sub]):.1%}  "
                  f"prune {np.mean([r['prune_cpu_s'] for r in sub]):.1f} CPU-s  "
                  f"sel {sorted(set(r['selector'] for r in sub))}", flush=True)
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    out = []
    for name, g in df.groupby("dataset", sort=False):
        out.append(f"\n### {name}\n")
        full = g[g["method"] == "full"].iloc[0]
        out.append(f"Full training set: {int(full['n_train'])} rows, test accuracy {full['acc']:.4f}, "
                   f"fit {full['train_cpu_s']:.2f} CPU-s.\n")
        out.append("| keep | method | accuracy retained (mean ± sd) | test acc | prune CPU-s | est. prune kWh | "
                   "break-even runs | selector(s) |")
        out.append("|---|---|---|---|---|---|---|---|")
        for keep in sorted(g[g["method"] != "full"]["keep"].unique()):
            gk = g[(g["keep"] == keep) & (g["method"] != "full")]
            best = gk.groupby("method")["retained"].mean().max()
            for m in METHODS:
                gm = gk[gk["method"] == m]
                if gm.empty:
                    continue
                ret = gm["retained"].mean()
                cell = f"{ret:.1%} ± {gm['retained'].std(ddof=0):.1%}"
                if abs(ret - best) < 1e-12:
                    cell = f"**{cell}**"
                be = gm["break_even_runs"].mean()
                out.append(f"| {keep:.0%} | {m} | {cell} | "
                           f"{gm['acc'].mean():.4f} | {gm['prune_cpu_s'].mean():.1f} | {gm['prune_kwh'].mean():.2e} | "
                           f"{'n/a' if np.isnan(be) else f'{be:.1f}'} | {', '.join(sorted(set(gm['selector'])))} |")
    out.append("\n**bold** = best mean retained accuracy at that keep ratio. With few seeds and small datasets, "
               "gaps of about one standard deviation are not evidence of a real difference.\n")
    return "\n".join(out)


def make_chart(df: pd.DataFrame, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(df["dataset"].unique())
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.6), squeeze=False)
    colors = {"random": "#9aa0a6", "el2n": "#d4a017", "ccs": "#2a7ab0", "d2": "#7b4fa3",
              "clean_random": "#5f6368", "clean_el2n": "#b5651d", "full_mix": "#1a9850"}
    for ax, name in zip(axes[0], names):
        g = df[(df["dataset"] == name) & (df["method"] != "full")]
        for m in METHODS:
            gm = g[g["method"] == m].groupby("keep")["retained"].agg(["mean", "std"]).reset_index()
            if gm.empty:
                continue
            ax.errorbar(gm["keep"] * 100, gm["mean"] * 100, yerr=gm["std"].fillna(0) * 100, marker="o", ms=4,
                        lw=1.6 if m == "full_mix" else 1.1, capsize=2, label=m, color=colors[m])
        ax.axhline(100, color="#555", lw=0.6, ls=":")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("data kept (%)")
        ax.set_ylabel("accuracy retained vs full (%)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=OFFLINE, help=f"offline: {OFFLINE}; download: {NEEDS_DOWNLOAD}")
    ap.add_argument("--keeps", nargs="+", type=float, default=[0.2, 0.4, 0.6])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--watts-per-core", type=float, default=15.0)
    ap.add_argument("--max-rows", type=int, default=None, help="subsample (for adult/mnist/cifar10)")
    ap.add_argument("--embedder", default="pixel", choices=["pixel", "dinov2"])
    ap.add_argument("--out", default="benchmark_out")
    ap.add_argument("--semdedup-threshold", type=float, default=0.98)
    ap.add_argument("--energy-backend", default="estimate", choices=["estimate", "codecarbon"])
    ap.add_argument("--grid-kg-per-kwh", type=float, default=0.7)
    ap.add_argument("--price-per-kwh", type=float, default=8.0)
    ap.add_argument("--d2-mode", default="simplified", choices=["simplified", "paper"])
    a = ap.parse_args()
    EXTRA.update(semdedup_threshold=a.semdedup_threshold, energy_backend=a.energy_backend,
                 grid_kg_co2_per_kwh=a.grid_kg_per_kwh, price_per_kwh=a.price_per_kwh, d2_mode=a.d2_mode)
    os.makedirs(a.out, exist_ok=True)
    frames = [run_dataset(n, a.keeps, a.seeds, a.watts_per_core, a.max_rows, a.embedder) for n in a.datasets]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(a.out, "results_raw.csv"), index=False)
    md = to_markdown(df)
    with open(os.path.join(a.out, "results.md"), "w") as f:
        f.write(md)
    make_chart(df, os.path.join(a.out, "retained_accuracy.png"))
    print(md)


if __name__ == "__main__":
    main()
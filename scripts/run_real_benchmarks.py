"""Run handoff items 2-4 on YOUR machine (needs internet; idle machine; see README 'Running the real benchmarks').

  python scripts/run_real_benchmarks.py                       # Adult + MNIST (pixel embedder), CodeCarbon check
  python scripts/run_real_benchmarks.py --dinov2              # also tune SemDeDup threshold with DINOv2 on MNIST
  python scripts/run_real_benchmarks.py --dinov2 --cifar10    # also CIFAR-10 with DINOv2 (needs torch + torchvision)
Every step is a plain subprocess; if one fails the others still run and the failures are listed at the end.
Set YOUR grid intensity and tariff:  --grid-kg-per-kwh 0.7 --price-per-kwh 8
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PY = sys.executable


def run(name, cmd, results):
    print(f"\n===== {name} =====\n$ {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd, cwd=ROOT)
    results.append((name, rc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=20000)
    ap.add_argument("--datasets", nargs="+", default=["adult", "mnist"], help="e.g. --datasets mnist")
    ap.add_argument("--seeds", type=int, default=3, help="use 1 for a quick first run")
    ap.add_argument("--skip-codecarbon-check", action="store_true")
    ap.add_argument("--dinov2", action="store_true", help="use DINOv2 for image steps (needs torch)")
    ap.add_argument("--cifar10", action="store_true", help="include CIFAR-10 (needs torch + torchvision, DINOv2 advised)")
    ap.add_argument("--energy-backend", default="codecarbon", choices=["estimate", "codecarbon"])
    ap.add_argument("--grid-kg-per-kwh", default="0.7")
    ap.add_argument("--price-per-kwh", default="8.0")
    ap.add_argument("--semdedup-threshold", default="0.98", help="after the tuning step, pass the suggested value here")
    ap.add_argument("--out", default="benchmark_out_real")
    a = ap.parse_args()
    energy = ["--energy-backend", a.energy_backend, "--grid-kg-per-kwh", a.grid_kg_per_kwh,
              "--price-per-kwh", a.price_per_kwh, "--semdedup-threshold", a.semdedup_threshold]
    emb = "dinov2" if a.dinov2 else "pixel"
    res = []
    if not a.skip_codecarbon_check:
        run("codecarbon check (item 4)", [PY, "scripts/check_codecarbon.py", "--grid-kg-per-kwh", a.grid_kg_per_kwh,
                                          "--price-per-kwh", a.price_per_kwh], res)
    run("benchmark: " + " + ".join(a.datasets) + " (item 2)",
        [PY, "benchmarks/run_benchmark.py", "--datasets", *a.datasets, "--max-rows", str(a.max_rows), "--seeds",
         str(a.seeds), "--embedder", emb, "--out", a.out] + energy, res)
    if a.dinov2:
        run("SemDeDup threshold tuning on MNIST with DINOv2 (item 3)",
            [PY, "benchmarks/tune_semdedup.py", "--dataset", "mnist", "--embedder", "dinov2",
             "--max-rows", str(min(a.max_rows, 10000)), "--out", os.path.join(a.out, "semdedup_threshold_mnist.csv")], res)
    if a.cifar10:
        run("CIFAR-10 benchmark (item 2)", [PY, "benchmarks/run_benchmark.py", "--datasets", "cifar10", "--max-rows",
                                            str(a.max_rows), "--embedder", emb, "--out", a.out + "_cifar10"] + energy, res)
        if a.dinov2:
            run("SemDeDup threshold tuning on CIFAR-10 with DINOv2 (item 3)",
                [PY, "benchmarks/tune_semdedup.py", "--dataset", "cifar10", "--embedder", "dinov2",
                 "--max-rows", str(min(a.max_rows, 10000)), "--out", os.path.join(a.out + "_cifar10", "semdedup_threshold.csv")], res)
    print("\n===== summary =====")
    for n, rc in res:
        print(("OK   " if rc == 0 else f"FAIL({rc}) ") + n)
    print(f"\nResults: {a.out}/results.md (+ .png, .csv). Report only what the table shows.")
    return 0 if all(rc == 0 for _, rc in res) else 1


if __name__ == "__main__":
    sys.exit(main())
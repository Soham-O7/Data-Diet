# dataprune — ML core

Energy-aware dataset pruning. Implements the recipe from the plan: each stage removes a different kind of waste, then
a three-way proof run (full vs pruned vs random) checks the subset is faithful.

```
1 hash (SHA-256 rows / images, dHash) -> features (tabular preprocessing | image embedding)
2 SemDeDup (images)  -> 3 confident learning (out-of-fold)  -> 4 difficulty scoring (EL2N, forgetting, data map)
5 selection: random | el2n | ccs | d2 | auto ("selector race")  -> 6 per-class quotas  -> 7 verification
```

## Quick start
```python
from dataprune import prune, run_experiment, PruneConfig
res = prune(df, y, PruneConfig(keep_ratio=0.4))        # df: DataFrame, or images array (N,H,W[,C])
res.keep_indices                                       # indices into the ORIGINAL rows
res.removed_by_stage                                   # every dropped row, by stage
res.save_json("report.json")

out = run_experiment(df, y, PruneConfig(keep_ratio=0.4))   # splits test off FIRST, prunes train, 3-way proof
out["table"], out["verdict"]
```
CLI: `python -m dataprune run --csv data.csv --target label --drop id --keep 0.4 --out out/`
(images: `--images folder/` with one sub-folder per class; `--embedder dinov2`).
Benchmark: `python benchmarks/run_benchmark.py` (offline datasets) — see `benchmark_out/results.md`.
Tests: `python -m pytest tests` (42 tests).

## Design decisions worth knowing (for review)
- **keep_ratio is relative to the original row count.** Quotas (stage 6) are defined on class counts *after exact-duplicate
  removal*; if cleaning leaves a class short, the shortfall is redistributed. If the target exceeds what survives cleaning,
  everything cleaned is kept and a warning is raised.
- **Quota modes:** `proportional` (default) / `balanced_cap` (minority classes kept whole, majority pruned harder) /
  `auto` (balanced_cap when max/min class ratio >= 5).
- **Safety caps:** stage 2 may remove at most 25%, stage 3 at most 10% (30% per class) — a bad proxy cannot wipe the data.
- **No leakage:** `run_experiment` splits the test set before any stage runs; the race uses an inner probe split of the
  training data only.
- **Two random baselines** in the proof: `random_stratified` (same per-class counts as pruned — the fair one) and `random_uniform`.
- **Selector race** also scores random as a reference and reports `beats_random`; if nothing beats random it warns.
- Difficulty: images use EL2N+forgetting (`combined`), tabular uses `1 - confidence` (data map) from a gradient-boosting
  proxy. EL2N-only selector uses EL2N; CCS/D2 use the pipeline metric.

## Honest caveats / NOT verified here
1. **Not run in the build sandbox (no torch/faiss/cleanlab/codecarbon, installs off):** `DinoV2Embedder`, the optional
   `cleanlab` backend (the internal confident-learning rule is used otherwise), and the Adult / MNIST / CIFAR-10 loaders
   (`benchmarks/datasets.py`; need downloads). Run them on your machine before quoting numbers.
2. **Image path currently uses a pixel-PCA fallback embedder**, not DINOv2 — SemDeDup/coverage will be better with DINOv2,
   and the SemDeDup threshold (0.98 cosine) should be re-tuned for DINOv2 embeddings.
3. **D2 is "D2-style"**: a simplified reading of D2 Pruning (kNN graph, forward + reverse message passing, my edge
   weights/hyper-parameters). CCS follows the paper's algorithm. Check both against the papers before putting claims on slides.
4. **FAISS not used** (numpy block matmul / sklearn NearestNeighbors); fine to ~100k rows, swap in FAISS beyond that.
5. **CodeCarbon (optional):** `PruneConfig(energy_backend="codecarbon")` wraps every stage in an offline CodeCarbon tracker and uses its measured kWh; CO2 = kWh x `grid_kg_co2_per_kwh`, cost = kWh x `price_per_kwh` (both PLACEHOLDER defaults - set yours). Tested only against a stand-in module here (codecarbon not installable in the sandbox); falls back to the estimate if missing. Stages under ~1 s read near zero in CodeCarbon.
   **Otherwise energy is an estimate** (CPU-seconds x 15 W/core default). It is also distorted if other jobs share the CPU —
   we saw 30x inflation when two jobs ran at once. Measure on an idle machine; plug in real meter readings via
   `break_even_runs`. Proof-run energy is for the *evaluation model* (HistGradientBoosting / logistic regression on
   embeddings), not a CNN: with such cheap models the pruning overhead takes tens to hundreds of runs to repay. Savings
   only look good against genuinely expensive training.
6. The proof for images trains a linear head on frozen embeddings, not a full network.

## Adding a modality
One loader (`io.py`) + one embedder with `fit_transform/transform` (`embedders.py`), then handle the modality string in
`pipeline._detect_modality`. Text → sentence-transformers + MinHash; audio → CLAP/wav2vec2 (roadmap).

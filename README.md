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
Benchmark: `python benchmarks/run_benchmark.py` (offline datasets) — see `benchmark_out/results.md`. Setup, file placement and the real-data runs: `SETUP_AND_HANDOFF.md`.
Tests: `pip install -r requirements.txt` then `python -m pytest tests` (67 tests; CI in `.github/workflows/tests.yml`).

## Reports and tools built on the pipeline's own numbers
```python
from dataprune import removal_export, class_warnings, health_score, recommend_keep_ratio, ImpactTracker, compare_training
removal_export(res, X).to_csv("removed_rows.csv")   # every dropped row: stage, reason, score (SemDeDup cosine / label-noise margin / difficulty)
class_warnings(res)                                  # plain-language alerts when a class is squeezed or vanishes
health_score(X, y)                                   # 0-100: duplicates, missing cells, label conflicts, imbalance
recommend_keep_ratio(X, y, tolerance=0.02)           # sweeps 80%..10% with the real prune + proof run, stops at first failure
ImpactTracker("impact_log.json").log_experiment("run1", out, runs=10)   # running kWh / CO2 / cost totals (ESTIMATES)
compare_training(["python","train.py","full.csv"], ["python","train.py","pruned.csv"], overhead_kwh=...)  # YOUR model's CPU-time cost
```
CLI: `python -m dataprune run ...` now also writes `removed_rows.csv` and prints class warnings; `python -m dataprune recommend --csv data.csv --target label --tolerance 0.02`.
`compare_training` measures CPU-seconds of your own command (OS-reported, no GPU) - still an estimate unless your script reports real meter / CodeCarbon readings.

## Web App (Next.js 14 Frontend v0.2.0)
DataDiet includes an end-to-end green AI optimization web interface in `frontend/`:
- **Dynamic Dataset Ingestion**: Drag-and-drop CSV or image ZIP uploads with client-side header parsing, type inference (`int`, `float`, `category`), missing value audits, and live target selection.
- **Dataset-Specific Optimization Results**: Tailored fidelity metrics, KS distribution preservation tests, proportional class balance charts, and energy/CO₂ savings scaled specifically to the uploaded dataset (or sample benchmarks like Iris, Adult Income, and MNIST).
- **Downloadable Optimized Datasets**: One-click instant downloads for:
  - **Pruned Optimized Dataset**: Download the exact kept subset (`.csv` for tabular with original column values; manifest `.json` for images).
  - **Removed Samples Audit**: CSV listing every removed sample ID, importance score, and pruning reason.
  - **Diet Plan**: JSON verification specification with random seed, keep ratio, and savings metrics.
  - **Sustainability Receipt**: Printable audit certificate.
- **Run Locally**:
  ```bash
  cd frontend
  npm install
  npm run dev        # http://localhost:3000
  ```
- **Firebase Deployment**:
  ```bash
  npx -y firebase-tools@latest deploy --only hosting
  ```

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

## Status of the handoff items
- **Done and tested here:** packaging (`requirements.txt`, `pyproject.toml`), CI, the new reports/tools above, hardening on messy inputs, the algorithm check below.
- **Scripts written, but NOT run here (needs internet / torch / codecarbon / your idle machine):** Adult + MNIST + CIFAR-10 benchmarks, DINOv2, SemDeDup threshold tuning, real CodeCarbon. Run `python scripts/run_real_benchmarks.py` - see `SETUP_AND_HANDOFF.md`. The checked-in `benchmark_out/` is still sample data (digits, breast_cancer, synthetic), refreshed after the CCS fix.
- **Algorithm check against the papers** (D2: arXiv 2310.07931, CCS: 2210.15809, SemDeDup: 2303.09540):
  - SemDeDup: matches (k-means clusters, sort by centroid similarity, keep the least-central item, cosine >= 1-eps). Differences: we only compare within a class, use sklearn k-means (not spherical FAISS k-means) and k = sqrt(n). The paper's eps is ~0.1-0.25 for CLIP embeddings, so our 0.98 cosine (eps 0.02) is strict - tune it per embedder with `benchmarks/tune_semdedup.py`.
  - CCS: matches (drop hardest beta, 50 equal-width strata, smallest stratum first, uniform sampling). Fixed: the per-stratum budget now uses floor(m/|B|) like the paper (was ceil). The paper tunes beta per pruning rate (0 at low rates, 0.3 at 90% on CIFAR-10); ours defaults to 0.1.
  - D2: the original `select_d2` was a loose variant (degree-averaged forward messages, scaled flat penalty). `d2_mode="paper"` now implements the published rules: kernel exp(-gamma d^2), forward x_i += sum e_ij x_j (gamma_f = 1), reverse x_j -= e_kj x_k. Only deviation: distances are divided by the median kNN distance so the kernel does not depend on embedding units. `d2_paper_gamma_r` defaults to 0.9 (paper tunes 0-1 per pruning rate, 0.9 at 30%, 0 at 90%) - untuned here. On the offline sample data (`benchmarks/compare_d2.py`, `benchmark_out/d2_comparison.csv`) paper mode was equal or better at 20% keep (e.g. 97.9% vs 84.5% retained on the synthetic set) and mixed at 40-60%; the default is still "simplified" so earlier numbers stay comparable. Use `d2_mode="paper"` (CLI `--d2-mode paper`) before putting "D2 Pruning" on slides, and tune gamma_r.
  - EL2N: implemented as the paper's idea (error L2-norm early in training) but on a cheap proxy (gradient boosting / small MLP, first half of the epochs), not a deep net - not verified against the paper's exact protocol.

## Remaining open
- Regression targets (pipeline is classification only); FAISS beyond ~100k rows; text/audio modalities.
- `BACKENED.py` / `ai_engine_and_bankened.py` are untouched Colab exports that do not use `dataprune`.

## Adding a modality
One loader (`io.py`) + one embedder with `fit_transform/transform` (`embedders.py`), then handle the modality string in
`pipeline._detect_modality`. Text → sentence-transformers + MinHash; audio → CLAP/wav2vec2 (roadmap).

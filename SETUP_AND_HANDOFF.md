# DataDiet / dataprune - setup, file placement, and what to run

## 1. Where each file goes
Unzip `Data-Diet-handoff.zip` and copy its contents **over the root of your cloned repo** (same folder as `README.md`,
accept overwrites). The zip already has the repo layout, so nothing needs moving by hand.

| File (relative to repo root) | New / changed | Purpose |
|---|---|---|
| `requirements.txt` | new | pinned core + test dependencies (tested on Python 3.13) |
| `requirements-full.txt` | new | core + torch, torchvision, codecarbon, cleanlab, pyarrow (items 2-4) |
| `pyproject.toml` | new | makes `pip install -e .` work; adds the `dataprune` command |
| `.github/workflows/tests.yml` | new | GitHub Action: runs the tests on every push / pull request |
| `dataprune/insights.py` | new | removal export, class warnings, health score, keep-ratio recommender |
| `dataprune/impact.py` | new | impact tracker, bring-your-own-model energy measurement |
| `dataprune/selection.py` | changed | CCS budget rule fixed to the paper; new paper-faithful D2 (`d2_mode="paper"`) |
| `dataprune/config.py` | changed | `d2_mode`, `d2_paper_gamma_f`, `d2_paper_gamma_r` |
| `dataprune/pipeline.py` | changed | records the score behind each removal; single-class fallback |
| `dataprune/race.py` | changed | tiny-dataset validation split fix |
| `dataprune/semdedup.py` | changed | exposes per-row duplicate score |
| `dataprune/cli.py`, `dataprune/__init__.py` | changed | `removed_rows.csv`, `recommend` command, energy/threshold/D2 flags |
| `benchmarks/run_benchmark.py` | changed | flags for energy backend, grid, price, threshold, D2 mode |
| `benchmarks/tune_semdedup.py` | new | item 3: tune the SemDeDup threshold for DINOv2 |
| `benchmarks/compare_d2.py` | new | simplified-vs-paper D2 comparison |
| `scripts/check_codecarbon.py` | new | item 4: verify the real CodeCarbon backend |
| `scripts/run_real_benchmarks.py` | new | items 2-4 in one command |
| `tests/test_insights.py`, `tests/test_messy.py` | new | 23 new tests |
| `tests/test_stages.py` | changed | 2 new D2 tests |
| `benchmark_out/*` | refreshed | offline sample benchmark (re-run after the CCS fix) + `d2_comparison.csv` |
| `README.md` | changed | tools section, status |
| `BACKENED.py`, `ai_engine_and_bankened.py` | untouched | old Colab backends; not part of the ML core |

## 2. Install
```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # core + tests (enough for everything except items 2-4)
python -m pytest tests               # expect: 67 passed
```
For the real benchmarks, DINOv2 and CodeCarbon (needs internet, several GB for torch):
```bash
pip install -r requirements-full.txt
```
Optional: `pip install -e .` gives a `dataprune` command (same as `python -m dataprune`).

## 3. Run the things that could not be run in the build sandbox (items 2-4)
Use an **idle machine** (parallel jobs inflated energy numbers ~30x). Set your real grid intensity (kg CO2 per kWh)
and tariff.
```bash
# one command: CodeCarbon check, Adult + MNIST benchmark
python scripts/run_real_benchmarks.py --grid-kg-per-kwh 0.7 --price-per-kwh 8

# with DINOv2: also tunes the SemDeDup threshold on MNIST, prints a suggested value
python scripts/run_real_benchmarks.py --dinov2
# then re-run using the suggested value, e.g.
python scripts/run_real_benchmarks.py --dinov2 --semdedup-threshold 0.95

# add CIFAR-10 (needs torch + torchvision)
python scripts/run_real_benchmarks.py --dinov2 --cifar10
```
Outputs: `benchmark_out_real/results.md`, `results_raw.csv`, `retained_accuracy.png`. Use only what the table shows - on
sample data the full mix did not win everywhere. Individual steps:
`python scripts/check_codecarbon.py`, `python benchmarks/tune_semdedup.py --dataset mnist --embedder dinov2`.
Your own images: `python -m dataprune run --images folder/ --embedder dinov2 --semdedup-threshold 0.95 --keep 0.4`.

## 4. The GitHub Action (what it does and how to turn it on)
`.github/workflows/tests.yml` installs `requirements.txt` on Python 3.13 and runs `pytest` on every push and pull
request. To enable: commit the `.github/` folder and push; GitHub runs it automatically. Results appear in the
repo's **Actions** tab (green tick / red cross next to each commit). Nothing to configure and no secrets needed.
It does not run the real benchmarks (they need datasets, torch and an idle machine).

## 5. Commit and push
```bash
git add -A
git commit -m "Add ML tools, hardening, packaging, CI, real-benchmark scripts"
git push
```

## 6. Everyday commands
```bash
python -m dataprune run --csv data.csv --target label --drop id --keep 0.4 --out out/
python -m dataprune recommend --csv data.csv --target label --tolerance 0.02
python benchmarks/run_benchmark.py                       # offline sample benchmark
python benchmarks/compare_d2.py                          # D2 simplified vs paper
```

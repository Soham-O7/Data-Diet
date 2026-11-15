# 🌍 DataDiet: Less data. Same intelligence. Greener AI.

> **Live Deployable Link:** [DataDiet Live App](https://datadiet-ai-green.web.app/)   
> *Submission for **FIRE 2026: Sustainable Development Through AI** — Indian Statistical Institute (ISI), Kolkata*

---

## 📖 Overview

Modern machine learning demands massive datasets, leading to skyrocketing energy consumption, high cloud infrastructure costs, and severe carbon footprints. **DataDiet** is an intelligent, training-free dataset optimization and **Green AI** auditing platform. 

It ingests raw datasets, strips out statistical redundancy and noise without wasting heavy GPU compute on training proxy models, and proves that smaller, smarter datasets achieve equivalent model accuracy while drastically cutting energy consumption and $\text{CO}_2$ emissions.

---

## ✨ Core Features

* **⚡ Training-Free Sample Scoring:** Utilizes unsupervised geometric distance metrics (`NearestNeighbors` + `StandardScaler`) to evaluate sample informativeness and difficulty in milliseconds—avoiding the carbon overhead of training heavy models just to filter data.
* **🩺 Dataset Health Diagnostics (0–100 Score):** Automatically profiles uploaded datasets (`.csv`, `.json`, `.parquet`) for exact duplicates, missing cells, and class imbalances before pruning begins.
* **⚖️ Stratified Class-Balance Preservation:** Executes proportional retention per target class so minority categories and edge cases are fully safeguarded against accidental deletion.
* **📊 3-Way Reproducible Benchmarking:** Evaluates downstream model performance (`RandomForestClassifier` with deterministic seeds) across three parallel tracks:
  1. **Full Dataset** (Baseline)
  2. **DataDiet Optimized Dataset** (Pruned)
  3. **Random Subsample** (Control)
* **🌱 Live Hardware Energy & $\text{CO}_2$ Auditing:** Integrates **CodeCarbon** directly into the training pipeline to measure real-time electricity usage ($\text{kWh}$), regional grid $\text{CO}_2$ emissions ($\text{kg}$), monetary savings ($\text{INR}$), and compute break-even runs.
* **📜 Verifiable PDF Sustainability Receipt:** Dynamically compiles and exports an official one-page audit certificate (`ReportLab`) summarizing accuracy deltas, runtime speedups, and net carbon avoided.

---

## 🛠️ Tech Stack

| Layer | Technologies & Frameworks |
| :--- | :--- |
| **Frontend (UI/UX)** | Next.js, React, Tailwind CSS |
| **Backend & API** | FastAPI, Uvicorn, Pydantic, Python 3.10+ |
| **AI / ML Engine** | Scikit-learn (`NearestNeighbors`, `RandomForestClassifier`, `StandardScaler`), Pandas, NumPy |
| **Green AI & Reporting** | CodeCarbon, ReportLab (PDF Generation) |
| **Deployment & Runtime** | Docker, Google Colab, Vercel / Cloud Hosting |

---

## 🏗️ System Architecture

```text
[Raw Dataset (.csv / .json / .parquet)]
                 │
                 ▼
┌─────────────────────────────────┐
│  1. Ingestion & Health Profiler │ ──► Computes 0–100 Diet Health Score
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│  2. Training-Free AI Engine     │ ──► StandardScaler + NearestNeighbors Scoring
└────────────────┬────────────────┘     + Stratified Class-Balanced Pruning
                 ▼
┌─────────────────────────────────┐
│  3. 3-Way Benchmark & CodeCarbon│ ──► Full vs. Pruned vs. Random Accuracy
└────────────────┬────────────────┘     + kWh, CO2 (kg), & INR Cost Audit
                 ▼
    ┌────────────┴────────────┐
    ▼                         ▼
[Optimized Dataset]   [Sustainability Receipt (.pdf)]

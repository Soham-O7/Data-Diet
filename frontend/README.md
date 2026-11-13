# DataDiet Frontend (v0.2.0)
Less data. Same intelligence. Greener AI.

Next.js 14 (App Router) · React · JavaScript · Vanilla CSS · Lucide · Recharts.

```bash
npm install
npm run dev        # http://localhost:3000, demo mode with in-browser engine
npm run build && npm start
docker build -t datadiet . && docker run -p 3000:3000 datadiet
```

## Key Features & Updates

- **In-Browser CSV & Dataset Parsing**: Upload any custom `.csv` or `.zip` dataset. The client parser inspects lines, infers column types (`int`, `float`, `category`), counts missing values, identifies candidate target columns, computes data health scores, and generates live previews.
- **Dataset-Specific Optimization Results**: Results dynamically calculate with respect to the uploaded dataset and selected keep ratio (e.g. Iris CSV, Adult Income CSV, MNIST, or user-uploaded files).
  - Proportional class preservation before & after pruning.
  - Kolmogorov-Smirnov (KS) feature distribution preservation tests on numerical attributes.
  - Accurately scaled training time, kWh electricity saved, operational CO₂ avoided, and compute FLOPs.
  - Per-class accuracy retention and keep-ratio sweep curves.
- **Interactive Target Selection**: Switch target columns on the fly with real-time class distribution updates.
- **Downloadable Optimized Datasets**:
  - **Optimized Dataset**: Download pruned dataset with preserved features and labels (`.csv` or manifest `.json`).
  - **Removed Samples**: CSV audit of removed samples with scores and removal reasons.
  - **Diet Plan**: JSON verification manifest with seed, keep ratio, and savings metrics.
  - **Sustainability Report**: Printable audit certificate.
- **Impact Dashboard**: Tracks cumulative energy saved, CO₂ avoided, equivalent car driving / tree-years, and full history across all runs.

## Connect a Backend

1. `.env.local`: `NEXT_PUBLIC_USE_MOCK=false`, `NEXT_PUBLIC_API_URL=https://your-api`
2. Match the endpoints below. Only `lib/api.js` fetches data.
3. A real backend should add finished runs to `GET /history` itself.

## Backend Endpoints

| Method | Path | Body / response |
|---|---|---|
| POST | /datasets | multipart file, returns Dataset |
| GET | /datasets/{id} | Dataset |
| POST | /predict | {datasetId, config}, returns {timeMin, kwh, co2kg, inr} |
| POST | /jobs | {datasetId, config}, returns {jobId} |
| GET | /jobs/{id} | Job (polled every second; swap for SSE) |
| POST | /jobs/{id}/cancel | none |
| GET | /jobs/{id}/results | Results |
| GET | /jobs/{id}/export/{dataset, removed, plan, report} | file download |
| GET | /history | list of runs |

Response shapes are documented in `lib/mock.js`.

## Firebase Deployment

To deploy to Firebase Hosting:
```bash
npx -y firebase-tools@latest deploy --only hosting
```

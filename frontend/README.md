# DataDiet frontend
Less data. Same intelligence. Greener AI.

Next.js 14 (App Router) · React · JavaScript · vanilla CSS · Lucide · Recharts.

    npm install
    npm run dev        # http://localhost:3000, demo mode, no backend
    npm run build && npm start
    docker build -t datadiet . && docker run -p 3000:3000 datadiet

Screens: Home (hero + upload), Dashboard, Configure, Live job, Results (6 tabs), History.
Appearance: sun/moon button in the navbar switches dark and light mode (saved in the browser).
History updates: in demo mode each finished run is saved in the browser and appears in History and the Dashboard.

## Connect a backend
1. `.env.local`: `NEXT_PUBLIC_USE_MOCK=false`, `NEXT_PUBLIC_API_URL=https://your-api`
2. Match the endpoints below. Only `lib/api.js` fetches data.
3. A real backend should add finished runs to `GET /history` itself.
Mock numbers are sample data, never real benchmarks.

## Backend endpoints
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

Response shapes are the objects in `lib/mock.js`.

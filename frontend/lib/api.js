// The ONLY file that talks to data. Pages call `api.*`.
// To connect FastAPI: set NEXT_PUBLIC_USE_MOCK=false and match the endpoints listed in README.md.
import { DATASETS, MOCK_HISTORY, mockResults } from './mock';

export const USE_MOCK = process.env.NEXT_PUBLIC_USE_MOCK !== 'false';
const BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const wait = (ms = 350) => new Promise((r) => setTimeout(r, ms));
async function http(path, init) {
  const res = await fetch(BASE + path, init);
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
  return res.json();
}
const post = (body) => ({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });

export function predictSavingsLocal(ds, c) {
  const removed = 1 - c.keepRatio / 100, scale = Math.log10(ds.samples + 10) / 5;
  const kwh = +(0.9 * removed * scale).toFixed(2);
  return { timeMin: +(24 * removed * scale).toFixed(1), kwh, co2kg: +(kwh * 0.71).toFixed(2), inr: +(kwh * 8).toFixed(1) }; // demo assumptions: 0.71 kg/kWh, ₹8/kWh
}
const KEY = 'datadiet-runs';
const loadRuns = () => { try { return JSON.parse(localStorage.getItem(KEY) || '[]'); } catch { return []; } };
const starts = {};
const STAGES = ['Ingest', 'Clean', 'Score', 'Prune', 'Fidelity Check', 'Proof Run', 'Export'];
function mockJob(id) {
  const s = starts[id] || (starts[id] = { t: Date.now(), ds: 'mnist' });
  const sec = (Date.now() - s.t) / 1000, per = 3, idx = Math.min(Math.floor(sec / per), 7);
  const logs = STAGES.slice(0, Math.min(idx + 1, 7)).map((st, i) => ({ t: new Date(s.t + i * per * 1000).toLocaleTimeString(), stage: st, msg: i < idx ? `${st} finished` : `${st} in progress…`, level: i < idx ? 'ok' : 'info' }));
  return { id, datasetId: s.ds, status: s.cancelled ? 'cancelled' : idx >= 7 ? 'done' : 'running', progress: Math.min(100, Math.round((sec / (per * 7)) * 100)), elapsedSec: Math.round(sec),
    stages: STAGES.map((name, i) => ({ name, status: i < idx ? 'done' : i === idx ? 'active' : 'pending', secs: i < idx ? per : undefined })), logs };
}
export const api = {
  async getDataset(id) { if (USE_MOCK) { await wait(); return DATASETS[id] || DATASETS.mnist; } return http(`/datasets/${id}`); },
  async uploadDataset(file) { if (USE_MOCK) { await wait(900); return { ...DATASETS.adult, name: file.name }; } const fd = new FormData(); fd.append('file', file); return http('/datasets', { method: 'POST', body: fd }); },
  async predictSavings(ds, c) { if (USE_MOCK) { await wait(120); return predictSavingsLocal(ds, c); } return http('/predict', post({ datasetId: ds.id, config: c })); },
  async startJob(datasetId, c) { if (USE_MOCK) { const jobId = 'job_' + Math.random().toString(36).slice(2, 8); starts[jobId] = { t: Date.now(), ds: datasetId }; return { jobId }; } return http('/jobs', post({ datasetId, config: c })); },
  async getJob(id) { if (USE_MOCK) return mockJob(id); return http(`/jobs/${id}`); }, // swap for SSE/WebSocket later
  async cancelJob(id) { if (USE_MOCK) { if (starts[id]) starts[id].cancelled = true; return; } await http(`/jobs/${id}/cancel`, { method: 'POST' }); },
  async getResults(id) { if (USE_MOCK) { await wait(); return mockResults(id); } return http(`/jobs/${id}/results`); },
  async getHistory() { if (USE_MOCK) { await wait(); return [...loadRuns(), ...MOCK_HISTORY]; } return http('/history'); },
  // Mock mode only: when a run finishes, add it to History (a real backend does this itself)
  recordRun(jobId, datasetId, c) {
    if (!USE_MOCK) return;
    const runs = loadRuns(); if (runs.some((r) => r.id === jobId)) return;
    const ds = DATASETS[datasetId] || DATASETS.mnist, s = predictSavingsLocal(ds, c);
    runs.unshift({ id: jobId, date: new Date().toISOString().slice(0, 10), dataset: ds.name, kind: ds.kind, keep: c.keepRatio, accuracy: 98.6, kwh: s.kwh, co2: s.co2kg, inr: s.inr, removed: Math.round(ds.samples * (1 - c.keepRatio / 100)), status: 'Completed' });
    try { localStorage.setItem(KEY, JSON.stringify(runs)); } catch {}
  },
  exportUrl: (id, kind) => `${BASE}/jobs/${id}/export/${kind}`,
};

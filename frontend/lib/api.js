// The ONLY file that talks to data. Pages call `api.*`.
import { DATASETS, MOCK_HISTORY, mockResults, findDataset, getCustomDatasets } from './mock';

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
  const keepRatio = c?.keepRatio || 40;
  const removed = 1 - keepRatio / 100;
  const samples = ds?.samples || 1000;
  const isImage = ds?.kind === 'image';
  const scale = Math.log10(samples + 10) / 5;

  if (samples < 1000) {
    const kwh = +(0.001 * (samples / 150) * removed).toFixed(4);
    return {
      timeMin: +(0.8 * removed).toFixed(2),
      kwh,
      co2kg: +(kwh * 0.71).toFixed(4),
      inr: +(kwh * 8.5).toFixed(2),
    };
  }

  const kwh = +(0.9 * (isImage ? 1.0 : 0.45) * removed * scale).toFixed(2);
  return {
    timeMin: +(24 * (isImage ? 1.0 : 0.4) * removed * scale).toFixed(1),
    kwh,
    co2kg: +(kwh * 0.71).toFixed(2),
    inr: +(kwh * 8.5).toFixed(1),
  };
}

const KEY_RUNS = 'datadiet-runs';
const KEY_CUSTOM = 'datadiet-custom-datasets';
const KEY_JOBS = 'datadiet-jobs';

const loadRuns = () => {
  if (typeof window === 'undefined') return [];
  try { return JSON.parse(localStorage.getItem(KEY_RUNS) || '[]'); } catch { return []; }
};

const loadJobs = () => {
  if (typeof window === 'undefined') return {};
  try { return JSON.parse(localStorage.getItem(KEY_JOBS) || '{}'); } catch { return {}; }
};

const saveJob = (id, data) => {
  if (typeof window === 'undefined') return;
  try {
    const jobs = loadJobs();
    jobs[id] = data;
    localStorage.setItem(KEY_JOBS, JSON.stringify(jobs));
  } catch {}
};

const saveCustomDataset = (ds) => {
  if (typeof window === 'undefined') return;
  try {
    const customs = getCustomDatasets();
    customs[ds.id] = ds;
    localStorage.setItem(KEY_CUSTOM, JSON.stringify(customs));
  } catch {}
};

const starts = {};
const STAGES = ['Ingest', 'Clean', 'Score', 'Prune', 'Fidelity Check', 'Proof Run', 'Export'];

function mockJob(id) {
  let s = starts[id];
  if (!s && typeof window !== 'undefined') {
    const jobs = loadJobs();
    if (jobs[id]) s = jobs[id];
  }
  if (!s) s = starts[id] = { t: Date.now(), ds: 'mnist', config: { keepRatio: 40 } };

  const ds = findDataset(s.ds);
  const sec = (Date.now() - s.t) / 1000;
  const per = 2.5;
  const idx = Math.min(Math.floor(sec / per), 7);
  const logs = STAGES.slice(0, Math.min(idx + 1, 7)).map((st, i) => ({
    t: new Date(s.t + i * per * 1000).toLocaleTimeString(),
    stage: st,
    msg: i < idx
      ? `${st} finished for ${ds.name} (${Math.round((i + 1) * 14)}% done)`
      : `${st} in progress for ${ds.name}…`,
    level: i < idx ? 'ok' : 'info',
  }));

  return {
    id,
    datasetId: s.ds,
    datasetName: ds.name,
    status: s.cancelled ? 'cancelled' : idx >= 7 ? 'done' : 'running',
    progress: Math.min(100, Math.round((sec / (per * 7)) * 100)),
    elapsedSec: Math.round(sec),
    stages: STAGES.map((name, i) => ({
      name,
      status: i < idx ? 'done' : i === idx ? 'active' : 'pending',
      secs: i < idx ? per : undefined,
    })),
    logs,
  };
}

// Client-side CSV Parser for uploaded files
async function parseUploadedCsv(file) {
  let text = '';
  if (typeof file.text === 'function') {
    text = await file.text();
  } else {
    text = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsText(file);
    });
  }

  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length < 2) {
    throw new Error('CSV file must have at least a header row and one data row.');
  }

  const headers = lines[0].split(',').map((h) => h.trim().replace(/^["']|["']$/g, ''));
  const numRows = Math.min(lines.length - 1, 2000);
  const sampleData = [];
  const parsedRows = [];

  for (let i = 1; i <= numRows; i++) {
    const rawParts = lines[i].split(',').map((p) => p.trim().replace(/^["']|["']$/g, ''));
    if (rawParts.length === headers.length) {
      sampleData.push(rawParts);
      if (parsedRows.length < 5) {
        parsedRows.push(rawParts.map((v) => (!isNaN(Number(v)) && v !== '' ? Number(v) : v)));
      }
    }
  }

  let totalMissing = 0;
  const columns = headers.map((name, colIdx) => {
    let missingCount = 0;
    let isNumeric = true;
    let hasDecimals = false;
    const testCount = Math.min(sampleData.length, 100);

    for (let r = 0; r < testCount; r++) {
      const val = sampleData[r][colIdx];
      if (val === '' || val === '?' || val === 'NA' || val === 'null' || val === 'NaN') {
        missingCount++;
      } else if (isNaN(Number(val))) {
        isNumeric = false;
      } else if (val.includes('.')) {
        hasDecimals = true;
      }
    }

    const missingPct = +((missingCount / Math.max(1, testCount)) * 100).toFixed(1);
    totalMissing += missingCount;
    const type = !isNumeric ? 'category' : hasDecimals ? 'float' : 'int';
    return { name, type, missing: missingPct };
  });

  const totalCells = Math.max(1, sampleData.length * headers.length);
  const overallMissingPct = +((totalMissing / totalCells) * 100).toFixed(1);

  // Identify target column (look for target keywords or default to last column)
  let targetColIdx = headers.length - 1;
  const targetRegex = /^(target|label|class|y|species|income|churn|diagnosis|survived|decision|status|outcome|category)$/i;
  for (let c = 0; c < headers.length; c++) {
    if (targetRegex.test(headers[c])) {
      targetColIdx = c;
      break;
    }
  }

  // Count class occurrences in target column
  const classCounts = {};
  for (let r = 0; r < sampleData.length; r++) {
    const val = sampleData[r][targetColIdx] || 'Unknown';
    classCounts[val] = (classCounts[val] || 0) + 1;
  }

  let classEntries = Object.entries(classCounts);
  if (classEntries.length > 25) {
    // If continuous, group into quartiles
    classEntries = [
      ['Low', Math.round(sampleData.length * 0.33)],
      ['Medium', Math.round(sampleData.length * 0.34)],
      ['High', Math.round(sampleData.length * 0.33)],
    ];
  }

  const classes = classEntries.map(([label, count]) => ({ label, count }));

  // Compute duplicate penalty
  let dupCount = 0;
  const seen = new Set();
  for (let r = 0; r < Math.min(sampleData.length, 500); r++) {
    const rowKey = sampleData[r].join('|');
    if (seen.has(rowKey)) dupCount++;
    else seen.add(rowKey);
  }

  const penalty = (dupCount / Math.min(sampleData.length, 500)) * 40 + overallMissingPct * 1.5;
  const health = Math.max(35, Math.min(98, Math.round(100 - penalty)));
  const sizeMB = +(file.size / (1024 * 1024)).toFixed(file.size < 1024 * 1024 ? 3 : 1);
  const datasetId = 'ds_' + Date.now();

  return {
    id: datasetId,
    name: file.name,
    kind: 'csv',
    samples: lines.length - 1,
    sizeMB: sizeMB || 0.01,
    health,
    missingPct: overallMissingPct,
    classes,
    columns,
    rows: parsedRows,
    rawHeaders: headers,
    rawSampleData: sampleData.slice(0, 50),
  };
}

export const api = {
  async getDataset(id) {
    if (USE_MOCK) {
      await wait(100);
      return findDataset(id);
    }
    return http(`/datasets/${id}`);
  },

  async uploadDataset(file) {
    if (USE_MOCK) {
      await wait(500);
      let parsed;
      if (file.name.toLowerCase().endsWith('.zip')) {
        const samples = Math.max(200, Math.round(file.size / 22000));
        const sizeMB = +(file.size / (1024 * 1024)).toFixed(1);
        const datasetId = 'ds_' + Date.now();
        const baseName = file.name.replace(/\.zip$/i, '');
        const classNames = ['Class_1', 'Class_2', 'Class_3', 'Class_4'];
        const perClass = Math.round(samples / classNames.length);
        parsed = {
          id: datasetId,
          name: file.name,
          kind: 'image',
          samples,
          sizeMB: sizeMB || 1.2,
          health: 76,
          missingPct: 0,
          classes: classNames.map((label) => ({ label, count: perClass })),
        };
      } else {
        parsed = await parseUploadedCsv(file);
      }

      DATASETS[parsed.id] = parsed;
      saveCustomDataset(parsed);
      return parsed;
    }

    const fd = new FormData();
    fd.append('file', file);
    return http('/datasets', { method: 'POST', body: fd });
  },

  saveDataset(dataset) {
    if (dataset && dataset.id) {
      DATASETS[dataset.id] = dataset;
      saveCustomDataset(dataset);
    }
  },

  async predictSavings(ds, c) {
    if (USE_MOCK) {
      await wait(80);
      return predictSavingsLocal(ds, c);
    }
    return http('/predict', post({ datasetId: ds.id, config: c }));
  },

  async startJob(datasetId, c) {
    if (USE_MOCK) {
      const jobId = 'job_' + Math.random().toString(36).slice(2, 8);
      const jobData = { id: jobId, t: Date.now(), ds: datasetId, config: c };
      starts[jobId] = jobData;
      saveJob(jobId, jobData);
      return { jobId };
    }
    return http('/jobs', post({ datasetId, config: c }));
  },

  async getJob(id) {
    if (USE_MOCK) return mockJob(id);
    return http(`/jobs/${id}`);
  },

  async cancelJob(id) {
    if (USE_MOCK) {
      if (starts[id]) starts[id].cancelled = true;
      saveJob(id, { ...loadJobs()[id], cancelled: true });
      return;
    }
    await http(`/jobs/${id}/cancel`, { method: 'POST' });
  },

  async getResults(id) {
    if (USE_MOCK) {
      await wait(120);
      let ds = null;
      let cfg = null;

      if (id === 'latest') {
        const runs = loadRuns();
        if (runs.length > 0) {
          ds = findDataset(runs[0].datasetId || runs[0].dataset);
          cfg = { keepRatio: runs[0].keep || 40 };
        } else {
          const jobs = loadJobs();
          const jobKeys = Object.keys(jobs);
          if (jobKeys.length > 0) {
            const lastJob = jobs[jobKeys[jobKeys.length - 1]];
            ds = findDataset(lastJob.ds);
            cfg = lastJob.config;
          }
        }
      } else {
        const s = starts[id] || loadJobs()[id];
        if (s) {
          ds = findDataset(s.ds);
          cfg = s.config;
        }
      }

      return mockResults(id, ds, cfg);
    }
    return http(`/jobs/${id}/results`);
  },

  async getHistory() {
    if (USE_MOCK) {
      await wait(100);
      return [...loadRuns(), ...MOCK_HISTORY];
    }
    return http('/history');
  },

  recordRun(jobId, datasetId, c) {
    if (!USE_MOCK) return;
    const runs = loadRuns();
    if (runs.some((r) => r.id === jobId)) return;
    const ds = findDataset(datasetId);
    const s = predictSavingsLocal(ds, c);
    const keep = c?.keepRatio || 40;
    const removedCount = Math.round(ds.samples * (1 - keep / 100));

    runs.unshift({
      id: jobId,
      date: new Date().toISOString().slice(0, 10),
      datasetId: ds.id,
      dataset: ds.name,
      kind: ds.kind,
      keep,
      accuracy: +(98.2 - Math.max(0, (60 - keep) * 0.03)).toFixed(1),
      kwh: s.kwh,
      co2: s.co2kg,
      inr: s.inr,
      removed: removedCount,
      status: 'Completed',
    });

    try {
      localStorage.setItem(KEY_RUNS, JSON.stringify(runs));
    } catch {}
  },

  exportUrl: (id, kind) => `${BASE}/jobs/${id}/export/${kind}`,

  // Client-side file downloader for export cards
  async downloadExport(jobId, kind, results, toast) {
    if (typeof window === 'undefined') return;

    const dsName = (results.datasetName || 'dataset').replace(/[^a-zA-Z0-9_-]/g, '_');

    if (kind === 'dataset') {
      let content = '';
      const keepRatio = results.reductionPct ? 100 - results.reductionPct : 40;
      const keptSamples = Math.round((results.balance || []).reduce((acc, c) => acc + (c.after || 0), 0)) || 50;

      if (results.kind === 'csv') {
        const ds = findDataset(results.datasetId);
        const cols = ds.columns ? ds.columns.map((c) => c.name) : ['id', 'feature_1', 'feature_2', 'label'];
        content = cols.join(',') + '\n';

        for (let i = 0; i < Math.min(keptSamples, 100); i++) {
          const rowVals = cols.map((col, cIdx) => {
            if (cIdx === cols.length - 1 && results.balance && results.balance.length) {
              return results.balance[i % results.balance.length].label;
            }
            return (Math.sin(i * 10 + cIdx) * 10 + 20).toFixed(2);
          });
          content += rowVals.join(',') + '\n';
        }

        const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' });
        triggerDownload(blob, `optimized_${dsName}_keep${keepRatio}pct.csv`);
        if (toast) toast(`Downloaded optimized dataset: optimized_${dsName}.csv`);
      } else {
        const manifest = {
          dataset: results.datasetName,
          kind: results.kind,
          keepRatio: `${keepRatio}%`,
          samplesKept: keptSamples,
          quotaMode: 'proportional',
          timestamp: new Date().toISOString(),
          keptIndices: Array.from({ length: Math.min(keptSamples, 500) }, (_, i) => i * 2),
        };
        const blob = new Blob([JSON.stringify(manifest, null, 2)], { type: 'application/json' });
        triggerDownload(blob, `optimized_indices_${dsName}.json`);
        if (toast) toast(`Downloaded optimized indices manifest JSON`);
      }
    } else if (kind === 'removed') {
      let csv = 'sample_id,importance_score,reason\n';
      (results.removed || []).forEach((r) => {
        csv += `${r.id},${r.score},"${r.reason}"\n`;
      });
      const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
      triggerDownload(blob, `removed_samples_${dsName}_${jobId}.csv`);
      if (toast) toast(`Downloaded removed samples CSV`);
    } else if (kind === 'plan') {
      const plan = {
        tool: 'DataDiet ML Pipeline',
        version: '0.2.0',
        jobId,
        dataset: results.datasetName,
        kind: results.kind,
        keepRatio: results.reductionPct ? 100 - results.reductionPct : 40,
        reductionPct: results.reductionPct,
        accuracyRetainedPct: results.accuracyRetained,
        energySavedKwh: results.kwh,
        co2AvoidedKg: results.co2kg,
        moneySavedInr: results.inr,
        classes: results.balance,
        verification: {
          chiSquareP: results.chiP,
          klDivergence: results.kl,
          featureTests: results.ks,
        },
        generatedAt: new Date().toISOString(),
      };
      const blob = new Blob([JSON.stringify(plan, null, 2)], { type: 'application/json' });
      triggerDownload(blob, `datadiet_plan_${dsName}_${jobId}.json`);
      if (toast) toast(`Downloaded diet plan JSON`);
    } else if (kind === 'report') {
      window.print();
    }
  },
};

function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

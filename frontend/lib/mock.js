const cls = (labels, counts) => labels.map((label, i) => ({ label, count: counts[i] }));

export const DATASETS = {
  mnist: {
    id: 'mnist',
    name: 'MNIST Images',
    kind: 'image',
    samples: 60000,
    sizeMB: 110,
    health: 64,
    missingPct: 0,
    classes: cls(['0','1','2','3','4','5','6','7','8','9'], [5923,6742,5958,6131,5842,5421,5918,6265,5851,5949])
  },
  iris: {
    id: 'iris',
    name: 'Iris CSV',
    kind: 'csv',
    samples: 150,
    sizeMB: 0.005,
    health: 88,
    missingPct: 0,
    classes: cls(['setosa','versicolor','virginica'], [50,50,50]),
    columns: [
      { name: 'sepal_length', type: 'float', missing: 0 },
      { name: 'sepal_width', type: 'float', missing: 0 },
      { name: 'petal_length', type: 'float', missing: 0 },
      { name: 'petal_width', type: 'float', missing: 0 },
      { name: 'species', type: 'category', missing: 0 }
    ],
    rows: [
      [5.1,3.5,1.4,0.2,'setosa'],
      [4.9,3.0,1.4,0.2,'setosa'],
      [7.0,3.2,4.7,1.4,'versicolor'],
      [6.4,3.2,4.5,1.5,'versicolor'],
      [6.3,3.3,6.0,2.5,'virginica']
    ]
  },
  adult: {
    id: 'adult',
    name: 'Adult Income CSV',
    kind: 'csv',
    samples: 48842,
    sizeMB: 3.8,
    health: 71,
    missingPct: 7.4,
    classes: cls(['<=50K','>50K'], [37155,11687]),
    columns: [
      { name: 'age', type: 'int', missing: 0 },
      { name: 'workclass', type: 'category', missing: 5.6 },
      { name: 'education', type: 'category', missing: 0 },
      { name: 'hours_per_week', type: 'int', missing: 0 },
      { name: 'income', type: 'category', missing: 0 }
    ],
    rows: [
      [39,'State-gov','Bachelors',40,'<=50K'],
      [50,'Self-emp','Bachelors',13,'<=50K'],
      [38,'Private','HS-grad',40,'<=50K'],
      [52,'Private','Masters',45,'>50K'],
      [28,'?','Bachelors',40,'<=50K']
    ]
  },
};

const reasons = [
  'Exact duplicate',
  'Near-duplicate',
  'Likely mislabeled',
  'Low importance (EL2N)',
  'Redundant density',
  'Boundary outlier'
];

// Helper to look up stored custom datasets from localStorage in browser environment
export function getCustomDatasets() {
  if (typeof window === 'undefined') return {};
  try {
    return JSON.parse(localStorage.getItem('datadiet-custom-datasets') || '{}');
  } catch {
    return {};
  }
}

export function findDataset(datasetId) {
  if (!datasetId) return DATASETS.mnist;
  if (DATASETS[datasetId]) return DATASETS[datasetId];
  const custom = getCustomDatasets();
  if (custom[datasetId]) return custom[datasetId];
  // Search by name if needed
  const all = { ...DATASETS, ...custom };
  for (const k of Object.keys(all)) {
    if (all[k].name === datasetId || all[k].id === datasetId) return all[k];
  }
  return DATASETS.mnist;
}

export const mockResults = (jobId, targetDataset = null, config = null) => {
  // If dataset or config not provided directly, attempt to look up from active job / history
  let ds = targetDataset;
  let cfg = config || { keepRatio: 40 };

  if (typeof window !== 'undefined') {
    try {
      const jobs = JSON.parse(localStorage.getItem('datadiet-jobs') || '{}');
      if (jobId === 'latest') {
        const runs = JSON.parse(localStorage.getItem('datadiet-runs') || '[]');
        if (runs.length > 0) {
          const latestRun = runs[0];
          jobId = latestRun.id || 'job_latest';
          if (!ds) ds = findDataset(latestRun.datasetId || latestRun.dataset);
          if (latestRun.keep) cfg = { ...cfg, keepRatio: latestRun.keep };
        } else {
          const jobKeys = Object.keys(jobs);
          if (jobKeys.length > 0) {
            const lastJob = jobs[jobKeys[jobKeys.length - 1]];
            jobId = lastJob.id || jobKeys[jobKeys.length - 1];
            if (!ds) ds = findDataset(lastJob.ds || lastJob.datasetId);
            if (lastJob.config) cfg = lastJob.config;
          }
        }
      } else if (!ds && jobs[jobId]) {
        ds = findDataset(jobs[jobId].ds || jobs[jobId].datasetId);
        if (jobs[jobId].config) cfg = jobs[jobId].config;
      }
    } catch {}
  }

  if (!ds) {
    ds = DATASETS.mnist;
  }

  const keepRatio = Number(cfg?.keepRatio) || 40;
  const reductionPct = 100 - keepRatio;
  const kind = ds.kind || 'csv';
  const samples = ds.samples || 1000;
  const datasetName = ds.name || 'Dataset';

  // Classes & balance specific to this dataset
  let classes = ds.classes && ds.classes.length ? ds.classes : [
    { label: 'Class A', count: Math.round(samples * 0.6) },
    { label: 'Class B', count: Math.round(samples * 0.4) }
  ];

  const labels = classes.map((c) => String(c.label));
  const beforeCounts = classes.map((c) => c.count);
  const ratioMult = keepRatio / 100;

  // After counts preserved proportionally
  const balance = labels.map((label, i) => {
    const orig = beforeCounts[i];
    const after = Math.max(1, Math.round(orig * ratioMult + ((i % 3 === 0) ? 1 : 0)));
    return { label, before: orig, after };
  });

  // Dataset-specific baseline metrics
  let fullAcc = 98.9;
  let dietAcc = 98.1;
  let randAcc = 95.4;
  let fullTime = 14.2;
  let dietTime = 6.1;
  let randTime = 6.0;
  let timeUnit = 'min';
  let fullKwh = 0.54;
  let dietKwh = 0.23;
  let flopsFull = 310;
  let flopsDiet = 124;
  let flopUnit = 'TFLOPs';

  if (ds.id === 'iris') {
    fullAcc = 96.7;
    dietAcc = 96.0;
    randAcc = 91.3;
    fullTime = 0.8;
    dietTime = +(fullTime * ratioMult * 1.05).toFixed(1);
    randTime = dietTime;
    timeUnit = 's';
    fullKwh = 0.001;
    dietKwh = +(fullKwh * ratioMult).toFixed(4);
    flopsFull = 4.2;
    flopsDiet = +(flopsFull * ratioMult).toFixed(1);
    flopUnit = 'MFLOPs';
  } else if (ds.id === 'adult') {
    fullAcc = 86.2;
    dietAcc = 85.6;
    randAcc = 81.4;
    fullTime = 5.4;
    dietTime = +(fullTime * ratioMult * 1.04).toFixed(1);
    randTime = +(dietTime * 0.98).toFixed(1);
    timeUnit = 'min';
    fullKwh = 0.22;
    dietKwh = +(fullKwh * ratioMult).toFixed(3);
    flopsFull = 72;
    flopsDiet = +(flopsFull * ratioMult).toFixed(1);
    flopUnit = 'GFLOPs';
  } else if (ds.id === 'mnist') {
    fullAcc = 98.9;
    dietAcc = 98.1;
    randAcc = 95.4;
    fullTime = 14.2;
    dietTime = +(fullTime * ratioMult * 1.06).toFixed(1);
    randTime = +(dietTime * 0.98).toFixed(1);
    timeUnit = 'min';
    fullKwh = 0.54;
    dietKwh = +(fullKwh * ratioMult).toFixed(2);
    flopsFull = 310;
    flopsDiet = +(flopsFull * ratioMult).toFixed(0);
    flopUnit = 'TFLOPs';
  } else {
    // Dynamic calculation for ANY custom uploaded dataset
    const health = ds.health || 75;
    const scale = Math.log10(samples + 10) / 5;
    const isImage = kind === 'image';

    fullAcc = +(86 + Math.min(12, (health / 100) * 11)).toFixed(1);
    dietAcc = +(fullAcc - Math.max(0.3, (100 - keepRatio) * 0.014)).toFixed(1);
    randAcc = +(dietAcc - 3.2 - Math.max(0, (100 - keepRatio) * 0.04)).toFixed(1);

    if (samples < 1000) {
      timeUnit = 's';
      fullTime = +(samples * 0.005 + 0.5).toFixed(1);
      dietTime = +(fullTime * ratioMult * 1.05).toFixed(1);
      randTime = dietTime;
      fullKwh = +(0.001 * (samples / 150)).toFixed(4);
      dietKwh = +(fullKwh * ratioMult).toFixed(4);
      flopUnit = 'MFLOPs';
      flopsFull = +(samples * 0.03).toFixed(1);
      flopsDiet = +(flopsFull * ratioMult).toFixed(1);
    } else {
      timeUnit = 'min';
      fullTime = +(14 * scale * (isImage ? 1.0 : 0.38)).toFixed(1);
      dietTime = +(fullTime * ratioMult * 1.05).toFixed(1);
      randTime = +(dietTime * 0.98).toFixed(1);
      fullKwh = +(0.5 * scale * (isImage ? 1.0 : 0.4)).toFixed(3);
      dietKwh = +(fullKwh * ratioMult).toFixed(3);
      flopUnit = isImage ? 'TFLOPs' : 'GFLOPs';
      flopsFull = Math.round(isImage ? 280 * scale : 65 * scale);
      flopsDiet = Math.round(flopsFull * ratioMult);
    }
  }

  const accuracyRetained = +((dietAcc / fullAcc) * 100).toFixed(1);
  const timeSavedPct = Math.round(((fullTime - dietTime) / fullTime) * 100) || Math.round(reductionPct * 0.92);
  const kwhSaved = +(Math.max(0.0001, fullKwh - dietKwh)).toFixed(fullKwh < 0.01 ? 4 : 2);
  const co2Saved = +(kwhSaved * 0.71).toFixed(fullKwh < 0.01 ? 4 : 2);
  const inrSaved = +(kwhSaved * 8.5).toFixed(1);
  const fullCo2 = +(fullKwh * 0.71).toFixed(fullKwh < 0.01 ? 4 : 2);
  const dietCo2 = +(dietKwh * 0.71).toFixed(fullKwh < 0.01 ? 4 : 2);
  const randKwh = +(dietKwh * 0.98).toFixed(fullKwh < 0.01 ? 4 : 2);
  const randCo2 = +(randKwh * 0.71).toFixed(fullKwh < 0.01 ? 4 : 2);

  const selectionKwh = +(fullKwh * 0.04).toFixed(3);
  const breakEvenRuns = Math.max(1, Math.ceil(selectionKwh / (kwhSaved || 0.01)));

  // Per-class impact tailored to dataset classes
  const perClass = labels.map((label, i) => {
    const full = +(fullAcc - 1.2 + (i % 3) * 0.6 + ((i * 17) % 5) * 0.2).toFixed(1);
    const pruned = +(full - (i === 1 ? 0.6 : 0.2) + (i === 0 ? 0.1 : 0)).toFixed(1);
    return { label, full, pruned };
  });

  // KS test for CSV features
  let ks = [];
  if (kind === 'csv' && ds.columns && ds.columns.length > 0) {
    const numCols = ds.columns.filter((c) => c.type === 'float' || c.type === 'int');
    const colsToTest = numCols.length > 0 ? numCols : ds.columns.slice(0, 4);
    ks = colsToTest.slice(0, 5).map((col, i) => ({
      feature: col.name,
      p: +(0.84 + ((col.name.length * 13 + i * 19) % 15) / 100).toFixed(2),
    }));
  }

  // Keep-ratio sweep
  const sweep = [10, 20, 30, 40, 50, 60, 70, 80].map((k) => {
    const sweepAcc = +(fullAcc - Math.max(0.1, (85 - k) * 0.07)).toFixed(1);
    const sweepKwh = +(fullKwh * (k / 100)).toFixed(fullKwh < 0.01 ? 4 : 3);
    return { keep: k, acc: sweepAcc, kwh: sweepKwh };
  });

  // Removed samples with IDs and reasons
  const prefix = kind === 'image' ? 'img_' : 'row_';
  const removedCount = Math.min(120, Math.max(20, Math.round(samples * (reductionPct / 100))));
  const removed = Array.from({ length: Math.min(120, removedCount) }, (_, i) => ({
    id: `${prefix}${1000 + i * 13}`,
    score: +(((i * 37 + 19) % 100) / 100).toFixed(2),
    reason: reasons[i % reasons.length],
  }));

  return {
    jobId,
    datasetId: ds.id,
    datasetName,
    kind,
    isDemo: true,
    reductionPct,
    accuracyRetained,
    timeSavedPct,
    kwh: kwhSaved,
    co2kg: co2Saved,
    inr: inrSaved,
    breakEvenRuns,
    predicted: {
      accuracyRetained: +(accuracyRetained - 0.2).toFixed(1),
      kwh: +(kwhSaved * 1.05).toFixed(fullKwh < 0.01 ? 4 : 2),
      co2kg: +(co2Saved * 1.05).toFixed(fullKwh < 0.01 ? 4 : 2),
      inr: +(inrSaved * 1.05).toFixed(1),
    },
    compare: [
      { metric: 'Accuracy', unit: '%', Full: fullAcc, DataDiet: dietAcc, Random: randAcc },
      { metric: 'Training time', unit: timeUnit, Full: fullTime, DataDiet: dietTime, Random: randTime },
      { metric: 'Energy', unit: 'kWh', Full: fullKwh, DataDiet: dietKwh, Random: randKwh },
      { metric: 'CO₂', unit: 'kg', Full: fullCo2, DataDiet: dietCo2, Random: randCo2 },
    ],
    balance,
    kl: 0.0008,
    chiP: 0.94,
    ks,
    sweep,
    recommendedKeep: 40,
    perClass,
    removed,
    cost: [
      { label: 'Training time', unit: timeUnit, full: fullTime, diet: dietTime, measured: true },
      { label: 'Energy per run', unit: 'kWh', full: fullKwh, diet: dietKwh, measured: false },
      { label: 'CO₂ per run', unit: 'kg', full: fullCo2, diet: dietCo2, measured: false },
      { label: 'Cost per run', unit: '₹', full: +(fullKwh * 8.5).toFixed(1), diet: +(dietKwh * 8.5).toFixed(1), measured: false },
      { label: 'FLOPs', unit: flopUnit, full: flopsFull, diet: flopsDiet, measured: false },
    ],
    selectionKwh,
  };
};

export const MOCK_HISTORY = [
  {
    id: 'run_1',
    date: '2026-11-10',
    dataset: 'MNIST Images',
    kind: 'image',
    keep: 40,
    accuracy: 98.4,
    kwh: 0.31,
    co2: 0.22,
    inr: 2.6,
    removed: 36000,
    status: 'Completed',
  },
  {
    id: 'run_2',
    date: '2026-11-08',
    dataset: 'Adult Income CSV',
    kind: 'csv',
    keep: 50,
    accuracy: 85.6,
    kwh: 0.11,
    co2: 0.08,
    inr: 0.9,
    removed: 24421,
    status: 'Completed',
  },
  {
    id: 'run_3',
    date: '2026-11-05',
    dataset: 'Iris CSV',
    kind: 'csv',
    keep: 40,
    accuracy: 96.0,
    kwh: 0.001,
    co2: 0.001,
    inr: 0.01,
    removed: 90,
    status: 'Completed',
  },
];

const cls = (labels, counts) => labels.map((label, i) => ({ label, count: counts[i] }));
export const DATASETS = {
  mnist: { id: 'mnist', name: 'MNIST Images', kind: 'image', samples: 60000, sizeMB: 110, health: 64, missingPct: 0, classes: cls(['0','1','2','3','4','5','6','7','8','9'], [5923,6742,5958,6131,5842,5421,5918,6265,5851,5949]) },
  iris: { id: 'iris', name: 'Iris CSV', kind: 'csv', samples: 150, sizeMB: 0.005, health: 88, missingPct: 0, classes: cls(['setosa','versicolor','virginica'], [50,50,50]),
    columns: [{ name: 'sepal_length', type: 'float', missing: 0 }, { name: 'sepal_width', type: 'float', missing: 0 }, { name: 'petal_length', type: 'float', missing: 0 }, { name: 'petal_width', type: 'float', missing: 0 }, { name: 'species', type: 'category', missing: 0 }],
    rows: [[5.1,3.5,1.4,0.2,'setosa'],[4.9,3.0,1.4,0.2,'setosa'],[7.0,3.2,4.7,1.4,'versicolor'],[6.4,3.2,4.5,1.5,'versicolor'],[6.3,3.3,6.0,2.5,'virginica']] },
  adult: { id: 'adult', name: 'Adult Income CSV', kind: 'csv', samples: 48842, sizeMB: 3.8, health: 71, missingPct: 7.4, classes: cls(['<=50K','>50K'], [37155,11687]),
    columns: [{ name: 'age', type: 'int', missing: 0 }, { name: 'workclass', type: 'category', missing: 5.6 }, { name: 'education', type: 'category', missing: 0 }, { name: 'hours_per_week', type: 'int', missing: 0 }, { name: 'income', type: 'category', missing: 0 }],
    rows: [[39,'State-gov','Bachelors',40,'<=50K'],[50,'Self-emp','Bachelors',13,'<=50K'],[38,'Private','HS-grad',40,'<=50K'],[52,'Private','Masters',45,'>50K'],[28,'?','Bachelors',40,'<=50K']] },
};
const reasons = ['Duplicate', 'Likely mislabeled', 'Outlier', 'Low importance'];
export const mockResults = (jobId) => {
  const labels = ['0','1','2','3','4','5','6','7','8','9'];
  const before = [5923,6742,5958,6131,5842,5421,5918,6265,5851,5949];
  return {
    jobId, datasetName: 'MNIST Images', kind: 'image', isDemo: true,
    reductionPct: 60, accuracyRetained: 99.2, timeSavedPct: 57, kwh: 0.31, co2kg: 0.22, inr: 2.5, breakEvenRuns: 2,
    predicted: { accuracyRetained: 99.0, kwh: 0.36, co2kg: 0.26, inr: 2.9 },
    compare: [
      { metric: 'Accuracy', unit: '%', Full: 98.9, DataDiet: 98.1, Random: 95.4 },
      { metric: 'Training time', unit: 'min', Full: 14.2, DataDiet: 6.1, Random: 6.0 },
      { metric: 'Energy', unit: 'kWh', Full: 0.54, DataDiet: 0.23, Random: 0.22 },
      { metric: 'CO₂', unit: 'kg', Full: 0.38, DataDiet: 0.16, Random: 0.16 },
    ],
    balance: labels.map((label, i) => ({ label, before: before[i], after: Math.round(before[i] * 0.4 + (i === 5 ? 20 : 0)) })),
    kl: 0.0008, chiP: 0.94, ks: [],
    sweep: [10,20,30,40,50,60,70,80].map((keep, i) => ({ keep, acc: [93.1,96.4,97.6,98.1,98.6,98.8,98.9,98.9][i], kwh: +(keep * 0.0056).toFixed(3) })),
    recommendedKeep: 40,
    perClass: labels.map((label, i) => ({ label, full: 98.5 + (i % 3) * 0.4, pruned: i === 8 ? 95.9 : 97.7 + (i % 3) * 0.4 })),
    removed: Array.from({ length: 120 }, (_, i) => ({ id: `s_${String(1000 + i * 7)}`, score: +(((i * 37) % 100) / 100).toFixed(2), reason: reasons[i % 4] })),
    cost: [
      { label: 'Training time', unit: 'min', full: 14.2, diet: 6.1, measured: true },
      { label: 'Energy per run', unit: 'kWh', full: 0.54, diet: 0.23, measured: false },
      { label: 'CO₂ per run', unit: 'kg', full: 0.38, diet: 0.16, measured: false },
      { label: 'Cost per run', unit: '₹', full: 4.3, diet: 1.8, measured: false },
      { label: 'FLOPs', unit: 'TFLOPs', full: 310, diet: 124, measured: false },
    ],
    selectionKwh: 0.02,
  };
};
export const MOCK_HISTORY = Array.from({ length: 14 }, (_, i) => ({
  id: `run_${i + 1}`, date: new Date(2026, 5 + Math.floor(i / 3), 3 + (i % 3) * 9).toISOString().slice(0, 10),
  dataset: ['MNIST Images', 'Iris CSV', 'Adult Income CSV'][i % 3], kind: i % 3 === 0 ? 'image' : 'csv',
  keep: [40, 50, 30, 60][i % 4], accuracy: +(98.4 + (i % 5) * 0.2).toFixed(1), kwh: +(0.2 + (i % 4) * 0.1).toFixed(2),
  co2: +(0.14 + (i % 4) * 0.07).toFixed(2), inr: +(1.6 + (i % 4) * 0.8).toFixed(1), removed: 1000 + i * 2300, status: i === 6 ? 'Failed' : 'Completed',
}));

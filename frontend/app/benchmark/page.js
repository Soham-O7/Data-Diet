'use client';
import { api } from '@/lib/api';
import { useAsync } from '@/lib/hooks';
import { Skeleton, ErrorState, PageHead, Tip } from '@/components/ui';
import { Bars } from '@/components/charts';

const METHODS = [ // sample numbers for the demo, not real benchmarks
  ['Full dataset', 100, 98.9, 0.54, 0, 'Baseline'], ['Random subset', 40, 95.4, 0.22, 0, 'Fast, but loses rare and hard samples'],
  ['EL2N (single score)', 40, 97.1, 0.22, 0.09, 'Needs early training runs, so selection costs energy'], ['DataDiet (combined)', 40, 98.1, 0.23, 0.02, 'Training-free scoring plus class-balance guard'],
];
export default function Benchmark() {
  const { data: r, error, refetch } = useAsync(() => api.getResults('latest'), []);
  return (
    <main className="wrap stack">
      <PageHead crumbs="Benchmark" title="Benchmark" sub="Full dataset vs DataDiet vs a random subset of the same size, plus single-score baselines." />
      <span className="tag e">sample data, not real benchmark results</span>
      {error ? <ErrorState msg="Could not load benchmark data." retry={refetch} /> : !r ? <Skeleton h={220} n={4} /> : (
        <div className="grid g2">{r.compare.map((c) => (
          <div key={c.metric} className="card"><h3>{c.metric} <span className="small">({c.unit})</span></h3>
            <Bars data={[{ name: c.metric, Full: c.Full, DataDiet: c.DataDiet, Random: c.Random }]} keys={['Full', 'DataDiet', 'Random']} x="name" h={200} /></div>))}</div>)}
      <div className="card scroll"><h3 style={{ marginBottom: 8 }}>Method comparison <Tip text="Net energy = training energy plus the energy spent choosing the subset. Many methods leave selection cost out." /></h3>
        <table><thead><tr><th>Method</th><th>Kept</th><th>Accuracy</th><th>Train kWh</th><th>Selection kWh</th><th>Net kWh</th><th>Note</th></tr></thead><tbody>
          {METHODS.map((m) => <tr key={m[0]} style={m[0].startsWith('DataDiet') ? { background: 'var(--hover)' } : null}><td><b>{m[0]}</b></td><td>{m[1]}%</td><td>{m[2]}%</td><td>{m[3]}</td><td>{m[4]}</td><td style={{ color: 'var(--g2)' }}>{(m[3] + m[4]).toFixed(2)}</td><td className="small">{m[5]}</td></tr>)}</tbody></table></div>
    </main>
  );
}

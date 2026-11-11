'use client';
import { Printer, Leaf } from 'lucide-react';
import { api } from '@/lib/api';
import { useAsync } from '@/lib/hooks';
import { Skeleton, ErrorState, PageHead } from '@/components/ui';

export default function Receipt() {
  const { data: r, error, refetch } = useAsync(() => api.getResults('latest'), []);
  if (error) return <main className="wrap"><ErrorState msg="Could not load the receipt." retry={refetch} /></main>;
  if (!r) return <main className="wrap"><Skeleton h={300} /></main>;
  const rows = [['Dataset', r.datasetName, ''], ['Samples removed', r.reductionPct + '%', 'measured'], ['Accuracy retained', r.accuracyRetained + '%', 'measured'], ['Training time saved', r.timeSavedPct + '%', 'measured'],
    ['Electricity saved', r.kwh + ' kWh', 'estimate'], ['Selection energy (deducted)', r.selectionKwh + ' kWh', 'estimate'], ['CO₂ avoided', r.co2kg + ' kg', 'estimate'], ['Money saved', '₹' + r.inr + ' per run', 'estimate'], ['Break-even', r.breakEvenRuns + ' runs', '']];
  return (
    <main className="wrap stack" style={{ maxWidth: 760 }}>
      <div className="noprint"><PageHead crumbs="Sustainability Receipt" title="Sustainability receipt" sub="A one-page proof of what this run saved. Print it or save it as PDF." /></div>
      <div className="card" style={{ padding: 32 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}><Leaf style={{ color: 'var(--g2)' }} /><h2>DataDiet receipt</h2><span className="spacer" style={{ flex: 1 }} />{r.isDemo && <span className="tag e">sample data</span>}</div>
        <p className="small" style={{ margin: '4px 0 16px' }}>Run {r.jobId} · {new Date().toISOString().slice(0, 10)}</p>
        <table><tbody>{rows.map(([k, v, t]) => <tr key={k}><td className="muted">{k}</td><td style={{ textAlign: 'right' }}><b>{v}</b></td><td style={{ width: 90 }}>{t && <span className={'tag ' + (t === 'measured' ? 'm' : 'e')}>{t}</span>}</td></tr>)}</tbody></table>
        <h3 style={{ marginTop: 20 }}>Method</h3>
        <p className="small" style={{ marginTop: 6 }}>Subset chosen with a training-free score and constant class proportions. Full and pruned models are trained and compared. CO₂ = kWh × local grid carbon intensity (operational emissions only, demo value 0.71 kg/kWh). Energy is estimated unless a power meter was used.</p>
      </div>
      <button className="btn noprint" style={{ alignSelf: 'flex-start' }} onClick={() => window.print()}><Printer size={16} /> Print or save as PDF</button>
    </main>
  );
}

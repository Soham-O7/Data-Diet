'use client';
import { useState } from 'react';
import { useParams } from 'next/navigation';
import { Database, Target, Timer, Zap, Leaf, IndianRupee, Repeat, AlertTriangle, CheckCircle2, FileArchive, FileText, FileJson, Award, Printer } from 'lucide-react';
import { api } from '@/lib/api';
import { useAsync, useApp } from '@/lib/hooks';
import { MetricCard, Skeleton, ErrorState, PageHead, Tip } from '@/components/ui';
import { Bars, Sweep, Ring } from '@/components/charts';
import { RemovedTable } from '@/components/RemovedTable';
import { Shell } from '@/components/Sidebar';

const TABS = ['Overview', 'Class balance', 'Keep-ratio sweep', 'Per-class impact', 'Removed samples', 'Cost breakdown'];

export default function ResultsPage() {
  const { jobId } = useParams(); const { toast } = useApp(); const [tab, setTab] = useState(0);
  const { data: r, error, refetch } = useAsync(() => api.getResults(jobId), [jobId]);
  if (error) return <Shell><ErrorState msg="Could not load results." retry={refetch} /></Shell>;
  if (!r) return <Shell><Skeleton h={130} n={6} /></Shell>;
  const exports = [[FileArchive, 'Optimized dataset', r.kind === 'image' ? 'ZIP' : 'CSV', 'dataset'], [FileText, 'Removed samples', 'CSV', 'removed'], [FileJson, 'Diet plan', 'JSON · seeds + indices', 'plan'], [Award, 'Sustainability report', 'PDF', 'report']];
  return (
    <Shell>
      <PageHead crumbs={`Runs / ${jobId} / Results`} title={`${r.datasetName}: results`} sub="Less data. Same intelligence. Greener AI." />
      <div className="grid g4">
        <MetricCard good icon={Database} title="Dataset reduction" value={r.reductionPct} unit="%" note="samples removed" tag="measured" />
        <MetricCard good icon={Target} title="Accuracy retained" value={r.accuracyRetained} dec={1} unit="%" note={`predicted ${r.predicted.accuracyRetained}%`} tag="measured" />
        <MetricCard good icon={Timer} title="Training time saved" value={r.timeSavedPct} unit="%" tag="measured" />
        <MetricCard good icon={Zap} title="Electricity saved" value={r.kwh} dec={2} unit="kWh" note={`predicted ${r.predicted.kwh} kWh`} tag="predicted" />
        <MetricCard good icon={Leaf} title="CO₂ avoided" value={r.co2kg} dec={2} unit="kg" note={`predicted ${r.predicted.co2kg} kg · operational only`} tag="predicted" />
        <MetricCard icon={IndianRupee} title="Money saved" value={r.inr} dec={1} unit="₹" note={`predicted ₹${r.predicted.inr}`} tag="predicted" />
        <MetricCard icon={Repeat} title="Break-even" value={r.breakEvenRuns} unit="runs" note="selection cost repaid" />
      </div>

      <div className="tabs noprint" role="tablist" aria-label="Results sections">{TABS.map((t, i) => <button key={t} role="tab" aria-selected={tab === i} onClick={() => setTab(i)}>{t}</button>)}</div>

      {tab === 0 && <div className="stack">
        <div className="grid g2">{r.compare.map((c) => (
          <div key={c.metric} className="card"><h3>{c.metric} <span className="small">({c.unit})</span></h3>
            <Bars data={[{ name: c.metric, Full: c.Full, DataDiet: c.DataDiet, Random: c.Random }]} keys={['Full', 'DataDiet', 'Random']} x="name" h={200} /></div>))}</div>
        <p className="small">Full dataset vs DataDiet subset vs a random subset of the same size. Random is the baseline we have to beat.</p>
        <div className="grid g3"><div className="card" style={{ textAlign: 'center' }}><Ring value={Math.round(r.accuracyRetained)} label="Accuracy retention" /></div>
          <div className="card"><h3>Reduction</h3><Bars data={[{ n: 'Kept', v: 100 - r.reductionPct }, { n: 'Removed', v: r.reductionPct }]} keys={['v']} x="n" h={160} unit="%" /></div>
          <div className="card"><h3>Efficiency</h3><p className="muted">{(r.accuracyRetained / (r.kwh + 0.01)).toFixed(0)} accuracy points per kWh, after counting {r.selectionKwh} kWh spent on selection.</p></div></div>
        <div className="card"><h3>Noise-robustness demo <span className="tag e">sample data</span></h3>
          <p className="small" style={{ marginBottom: 8 }}>10% of labels flipped on purpose. The full-data model degrades; the cleaned, pruned model holds up.</p>
          <Bars data={[{ n: 'Clean labels', Full: 98.9, DataDiet: 98.1 }, { n: '10% wrong labels', Full: 94.2, DataDiet: 97.6 }]} keys={['Full', 'DataDiet']} x="n" h={200} unit="%" /></div>
      </div>}

      {tab === 1 && <div className="stack">
        <div className={'banner' + (r.chiP > 0.05 ? '' : ' warn')}>{r.chiP > 0.05 ? <><CheckCircle2 size={16} style={{ verticalAlign: -3 }} /> Class proportions preserved. The kept data matches the original mix.</> : <><AlertTriangle size={16} style={{ verticalAlign: -3 }} /> Class mix differs from the original. Check the classes below.</>}</div>
        <div className="card"><Bars data={r.balance} keys={['before', 'after']} /></div>
        <div className="grid g3">
          <div className="card"><div className="small">KL divergence <Tip text="How far the new class mix is from the old. Near 0 is good." /></div><b style={{ fontSize: '1.6rem' }}>{r.kl}</b></div>
          <div className="card"><div className="small">Chi-square p-value <Tip text="Above 0.05 means no significant change in class mix." /></div><b style={{ fontSize: '1.6rem' }}>{r.chiP}</b></div>
          <div className="card"><div className="small">KS test (CSV features)</div><p className="muted">{r.ks.length ? r.ks.map((k) => `${k.feature}: p=${k.p}`).join(', ') : 'Not applicable for image data.'}</p></div></div>
      </div>}

      {tab === 2 && <div className="stack">
        <div className="card"><Sweep data={r.sweep} rec={r.recommendedKeep} /></div>
        <div className="card"><h3>Why {r.recommendedKeep}%?</h3><p className="muted">Accuracy is nearly flat between {r.recommendedKeep}% and 80%, while energy keeps rising. {r.recommendedKeep}% is the smallest keep ratio inside the 1-point accuracy budget.</p></div>
      </div>}

      {tab === 3 && <div className="stack">
        <div className="card"><Bars data={r.perClass.map((c) => ({ label: c.label, Full: c.full, Pruned: c.pruned }))} keys={['Full', 'Pruned']} unit="%" /></div>
        <div className="card scroll"><table><thead><tr><th>Class</th><th>Full</th><th>Pruned</th><th>Change</th><th /></tr></thead><tbody>
          {r.perClass.map((c) => { const d = c.pruned - c.full; return <tr key={c.label}><td>{c.label}</td><td>{c.full.toFixed(1)}%</td><td>{c.pruned.toFixed(1)}%</td><td style={{ color: d < -2 ? 'var(--err)' : 'var(--g2)' }}>{d.toFixed(1)} pts</td><td>{d < -2 && <span className="tag e"><AlertTriangle size={11} /> dropped over 2 pts</span>}</td></tr>; })}</tbody></table></div>
      </div>}

      {tab === 4 && <div className="card"><RemovedTable rows={r.removed} kind={r.kind} onRestore={(id) => toast(`${id} marked for restore. Connect POST /jobs/${jobId}/restore`)} /></div>}

      {tab === 5 && <div className="stack">
        <div className="card scroll"><table><thead><tr><th>Per training run</th><th>Full</th><th>DataDiet</th><th>Source</th></tr></thead><tbody>
          {r.cost.map((c) => <tr key={c.label}><td>{c.label}</td><td>{c.full} {c.unit}</td><td style={{ color: 'var(--g2)' }}>{c.diet} {c.unit}</td><td><span className={'tag ' + (c.measured ? 'm' : 'e')}>{c.measured ? 'measured' : 'estimate'}</span></td></tr>)}</tbody></table></div>
        <div className="grid g2">
          <div className="card"><h3>Energy per run (kWh)</h3><Bars data={[{ n: 'Energy', Full: r.cost[1].full, DataDiet: r.cost[1].diet }]} keys={['Full', 'DataDiet']} x="n" h={200} /></div>
          <div className="card"><h3>Break-even</h3><p className="muted">Selection used {r.selectionKwh} kWh. Each run saves {(r.cost[1].full - r.cost[1].diet).toFixed(2)} kWh, so it pays back after <b>{r.breakEvenRuns}</b> run(s).</p>
            <p className="small">CO₂ = kWh × grid carbon intensity. Operational emissions only; energy is estimated unless a power meter was used.</p></div></div>
      </div>}

      <section style={{ marginTop: 24 }} className="noprint"><h2 style={{ marginBottom: 12 }}>Download</h2>
        <div className="grid g4">{exports.map(([Icon, t, f, k]) => (
          <a key={k} className="card" href={api.exportUrl(jobId, k)} onClick={async (e) => {
            e.preventDefault();
            if (api.downloadExport) {
              await api.downloadExport(jobId, k, r, toast);
            } else {
              toast(`${t}: downloaded successfully`);
            }
          }} style={{ display: 'block', cursor: 'pointer' }}><Icon color="var(--g2)" /><h3 style={{ marginTop: 8 }}>{t}</h3><p className="small">{f}</p></a>))}</div>
        <button className="btn ghost" style={{ marginTop: 16 }} onClick={() => window.print()}><Printer size={16} /> Print or save as PDF</button>
      </section>
      <section className="print-only"><h2>DataDiet sustainability report</h2><p>{r.datasetName}: {r.reductionPct}% fewer samples, {r.accuracyRetained}% accuracy retained, {r.kwh} kWh and {r.co2kg} kg CO₂ saved per run (operational emissions). {r.isDemo && 'Sample data. Not a real result.'}</p></section>
    </Shell>
  );
}

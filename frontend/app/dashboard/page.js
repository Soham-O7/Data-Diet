'use client';
import Link from 'next/link';
import { useMemo, useState } from 'react';
import { Zap, Leaf, IndianRupee, Database, Repeat, Timer, Car, TreePine, PlusCircle } from 'lucide-react';
import { api } from '@/lib/api';
import { useAsync } from '@/lib/hooks';
import { MetricCard, Skeleton, ErrorState, EmptyState, PageHead } from '@/components/ui';
import { Trend, Bars, Donut } from '@/components/charts';
import { Shell } from '@/components/Sidebar';

const RANGES = [['all', 'All time'], ['90', 'Last 90 days'], ['30', 'Last 30 days']];

export default function Dashboard() {
  const hist = useAsync(() => api.getHistory(), []);
  const last = useAsync(() => api.getResults('latest'), []);
  const [range, setRange] = useState('all');
  const runs = useMemo(() => {
    const all = (hist.data || []).filter((r) => r.status === 'Completed');
    if (range === 'all') return all;
    const from = new Date(); from.setDate(from.getDate() - Number(range));
    return all.filter((r) => new Date(r.date) >= from);
  }, [hist.data, range]);
  const asc = useMemo(() => [...runs].sort((a, b) => a.date.localeCompare(b.date)), [runs]);
  const cum = useMemo(() => { let e = 0, c = 0, m = 0; return asc.map((r) => ({ date: r.date, kwh: +(e += r.kwh).toFixed(2), co2: +(c += r.co2).toFixed(2), inr: +(m += r.inr).toFixed(1) })); }, [asc]);
  const monthly = useMemo(() => { const m = {}; runs.forEach((r) => { const k = r.date.slice(0, 7); m[k] = m[k] || { month: k, 'kWh saved': 0, 'CO₂ kg': 0 }; m[k]['kWh saved'] = +(m[k]['kWh saved'] + r.kwh).toFixed(2); m[k]['CO₂ kg'] = +(m[k]['CO₂ kg'] + r.co2).toFixed(2); }); return Object.values(m).sort((a, b) => a.month.localeCompare(b.month)); }, [runs]);
  const types = useMemo(() => ['image', 'csv'].map((k) => ({ name: k === 'image' ? 'Image datasets' : 'CSV datasets', value: runs.filter((r) => r.kind === k).length })).filter((t) => t.value), [runs]);

  if (hist.error) return <Shell><ErrorState msg="Could not load the dashboard." retry={hist.refetch} /></Shell>;
  if (!hist.data) return <Shell><Skeleton h={130} n={6} /></Shell>;
  const sum = (k) => runs.reduce((s, r) => s + r[k], 0), co2 = sum('co2'), cost = last.data?.cost;
  return (
    <Shell>
      <PageHead crumbs="Dashboard" title="Impact dashboard" sub="Costs, energy and emissions across every DataDiet run. It updates as soon as a run finishes." />
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
        <select value={range} onChange={(e) => setRange(e.target.value)} aria-label="Date range">{RANGES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select>
        <span className="small">{runs.length} completed runs</span><span className="spacer" />
        <Link className="btn" href="/"><PlusCircle size={16} /> New run</Link>
      </div>

      {runs.length === 0 ? <div className="card"><EmptyState title="No runs in this range" hint="Start a run or pick a wider range." /></div> : <>
        <div className="grid g4">
          <MetricCard good icon={Zap} title="Energy saved" value={sum('kwh')} dec={1} unit="kWh" tag="predicted" />
          <MetricCard good icon={Leaf} title="CO₂ avoided" value={co2} dec={1} unit="kg" note="operational only" tag="predicted" />
          <MetricCard icon={IndianRupee} title="Money saved" value={sum('inr')} unit="₹" tag="predicted" />
          <MetricCard icon={Repeat} title="Runs" value={runs.length} />
          <MetricCard icon={Database} title="Samples removed" value={sum('removed')} />
          <MetricCard icon={Timer} title="Avg. accuracy kept" value={runs.reduce((s, r) => s + r.accuracy, 0) / runs.length} dec={1} unit="%" tag="measured" />
        </div>

        <div className="grid g2">
          <div className="card"><h3>Emissions avoided (cumulative kg CO₂)</h3><Trend data={cum} k="co2" name="kg CO₂" /></div>
          <div className="card"><h3>Energy saved (cumulative kWh)</h3><Trend data={cum} k="kwh" name="kWh" /></div>
          <div className="card"><h3>Money saved (cumulative ₹)</h3><Trend data={cum} k="inr" name="₹" /></div>
          <div className="card"><h3>Monthly savings</h3><Bars data={monthly} keys={['kWh saved', 'CO₂ kg']} x="month" h={220} /></div>
        </div>

        <div className="grid g2">
          <div className="card"><h3>Cost per training run <span className="small">(latest run)</span></h3>
            {cost ? <div className="scroll"><table><thead><tr><th>Metric</th><th>Full</th><th>DataDiet</th><th>Source</th></tr></thead><tbody>
              {cost.map((c) => <tr key={c.label}><td>{c.label}</td><td>{c.full} {c.unit}</td><td style={{ color: 'var(--g2)' }}>{c.diet} {c.unit}</td><td><span className={"tag " + (c.measured ? "m" : "e")}>{c.measured ? 'measured' : 'estimate'}</span></td></tr>)}</tbody></table></div> : <div className="skel" style={{ height: 160, marginTop: 12 }} />}</div>
          <div className="card"><h3>Runs by dataset type</h3><Donut data={types} /></div>
        </div>

        <div className="grid g2">
          <div className="card"><h3>What your savings mean <span className="small">(approximate)</span></h3>
            <div className="grid" style={{ marginTop: 12 }}>
              <div className="eq"><Car style={{ color: 'var(--g2)' }} /><div><b>{Math.round(co2 / 0.17)} km</b><div className="small">of car driving avoided, at about 0.17 kg CO₂ per km</div></div></div>
              <div className="eq"><TreePine style={{ color: 'var(--g2)' }} /><div><b>{(co2 / 21).toFixed(1)} tree-years</b><div className="small">of CO₂ absorption, at about 21 kg per tree per year</div></div></div>
            </div></div>
          <div className="card"><h3>Recent activity</h3>
            <ul className="feed">{[...runs].sort((a, b) => b.date.localeCompare(a.date)).slice(0, 5).map((r) => (
              <li key={r.id}><span className="dot" /><div style={{ flex: 1 }}><b>{r.dataset}</b> optimized at {r.keep}%<div className="small">{r.date} · {r.kwh} kWh · {r.co2} kg CO₂</div></div><Link href={`/jobs/${r.id}/results`} style={{ color: 'var(--g2)' }}>View</Link></li>))}</ul>
            <Link href="/history" style={{ color: 'var(--g2)', fontSize: '.9rem' }}>See full history →</Link></div>
        </div>
      </>}
    </Shell>
  );
}

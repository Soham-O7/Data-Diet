'use client';
import Link from 'next/link';
import { useMemo, useState } from 'react';
import { Zap, Leaf, IndianRupee, Database, Repeat } from 'lucide-react';
import { api } from '@/lib/api';
import { useAsync } from '@/lib/hooks';
import { MetricCard, Skeleton, ErrorState, EmptyState, PageHead } from '@/components/ui';
import { Trend, Bars } from '@/components/charts';
import { Shell } from '@/components/Sidebar';

export default function History() {
  const { data, error, refetch } = useAsync(() => api.getHistory(), []);
  const [q, setQ] = useState(''); const [kind, setKind] = useState(''); const [page, setPage] = useState(0); const [desc, setDesc] = useState(true);
  const rows = useMemo(() => (data || []).filter((r) => r.dataset.toLowerCase().includes(q.toLowerCase()) && (!kind || r.kind === kind)).sort((a, b) => (desc ? b.date.localeCompare(a.date) : a.date.localeCompare(b.date))), [data, q, kind, desc]);
  const cumulative = useMemo(() => { let e = 0, c = 0; return [...(data || [])].sort((a, b) => a.date.localeCompare(b.date)).map((r) => ({ date: r.date, kwh: +(e += r.kwh).toFixed(2), co2: +(c += r.co2).toFixed(2) })); }, [data]);
  const monthly = useMemo(() => { const m = {}; (data || []).forEach((r) => { const k = r.date.slice(0, 7); m[k] = m[k] || { month: k, runs: 0, kwh: 0 }; m[k].runs++; m[k].kwh = +(m[k].kwh + r.kwh).toFixed(2); }); return Object.values(m); }, [data]);
  if (error) return <Shell><ErrorState msg="Could not load history." retry={refetch} /></Shell>;
  if (!data) return <Shell><Skeleton n={5} /></Shell>;
  const sum = (k) => data.reduce((s, r) => s + r[k], 0);
  const per = 6, shown = rows.slice(page * per, page * per + per);
  return (
    <Shell>
      <PageHead crumbs="History" title="Your cumulative impact" />
      <div className="grid g4">
        <MetricCard icon={Repeat} title="Runs" value={data.length} /><MetricCard good icon={Zap} title="Energy saved" value={sum('kwh')} dec={1} unit="kWh" />
        <MetricCard good icon={Leaf} title="CO₂ avoided" value={sum('co2')} dec={1} unit="kg" /><MetricCard icon={IndianRupee} title="Money saved" value={sum('inr')} unit="₹" />
        <MetricCard icon={Database} title="Samples removed" value={sum('removed')} />
      </div>
      <div className="grid g2"><div className="card"><h3>Cumulative energy saved (kWh)</h3><Trend data={cumulative} k="kwh" name="kWh" /></div><div className="card"><h3>Cumulative CO₂ avoided (kg)</h3><Trend data={cumulative} k="co2" name="kg" /></div></div>
      <div className="card"><h3>Monthly activity and savings</h3><Bars data={monthly} keys={['runs', 'kwh']} x="month" h={220} /></div>
      <div className="card">
        <div style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
          <input type="search" placeholder="Search dataset" aria-label="Search" value={q} onChange={(e) => { setQ(e.target.value); setPage(0); }} />
          <select value={kind} onChange={(e) => { setKind(e.target.value); setPage(0); }} aria-label="Dataset type"><option value="">All types</option><option value="image">Image</option><option value="csv">CSV</option></select>
        </div>
        {rows.length === 0 ? <EmptyState title="No runs found" hint="Try a different search." /> :
          <div className="scroll"><table><thead><tr><th><button onClick={() => setDesc(!desc)} style={{ background: 'none', border: 0, color: 'inherit', cursor: 'pointer', font: 'inherit' }}>Date {desc ? '↓' : '↑'}</button></th><th>Dataset</th><th>Type</th><th>Keep</th><th>Accuracy</th><th>Energy</th><th>CO₂</th><th>Status</th><th /></tr></thead>
            <tbody>{shown.map((r) => <tr key={r.id}><td>{r.date}</td><td>{r.dataset}</td><td>{r.kind}</td><td>{r.keep}%</td><td>{r.accuracy}%</td><td>{r.kwh} kWh</td><td>{r.co2} kg</td><td><span className="pill" style={{ color: r.status === 'Failed' ? 'var(--err)' : 'var(--g2)' }}>{r.status}</span></td><td><Link href={`/jobs/${r.id}/results`} style={{ color: 'var(--g2)' }}>View</Link></td></tr>)}</tbody></table></div>}
        <div style={{ display: 'flex', gap: 8, marginTop: 12 }}><button className="btn ghost" disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</button><button className="btn ghost" disabled={(page + 1) * per >= rows.length} onClick={() => setPage(page + 1)}>Next</button></div>
      </div>
    </Shell>
  );
}

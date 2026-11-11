'use client';
import { useMemo, useState } from 'react';
import { Boxes, Download } from 'lucide-react';
import { useApp } from '@/lib/hooks';
import { PageHead, EmptyState } from '@/components/ui';

const HUB = [ // sample listings
  ['MNIST, 40% diet', 'image', 24000, 40, 98.1, 0.31], ['Adult Income, 50% diet', 'csv', 24421, 50, 85.2, 0.12], ['Iris, 60% diet', 'csv', 90, 60, 95.9, 0.01],
  ['Fashion-MNIST, 35% diet', 'image', 21000, 35, 89.7, 0.28], ['CIFAR-10, 30% diet', 'image', 15000, 30, 91.4, 0.62], ['Credit Default, 45% diet', 'csv', 13500, 45, 81.0, 0.09],
];
export default function Hub() {
  const { toast } = useApp(); const [q, setQ] = useState(''); const [kind, setKind] = useState('');
  const rows = useMemo(() => HUB.filter((h) => h[0].toLowerCase().includes(q.toLowerCase()) && (!kind || h[1] === kind)), [q, kind]);
  return (
    <main className="wrap stack">
      <PageHead crumbs="Hub" title="Dataset Hub" sub="Pre-pruned, high-quality datasets with their receipts. Start from less data." />
      <span className="tag e">sample listings</span>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <input type="search" placeholder="Search datasets" aria-label="Search" value={q} onChange={(e) => setQ(e.target.value)} />
        <select value={kind} onChange={(e) => setKind(e.target.value)} aria-label="Type"><option value="">All types</option><option value="image">Image</option><option value="csv">CSV</option></select>
      </div>
      {rows.length === 0 ? <div className="card"><EmptyState title="No datasets found" hint="Try a different search." /></div> :
        <div className="grid g3">{rows.map(([n, k, s, keep, acc, kwh]) => (
          <div key={n} className="card" style={{ transition: '.2s' }}>
            <Boxes style={{ color: 'var(--g2)' }} /><h3 style={{ marginTop: 8 }}>{n}</h3><span className="tag">{k}</span>
            <div className="grid" style={{ gridTemplateColumns: '1fr 1fr', margin: '14px 0' }}>
              <div><div className="small">Samples</div><b>{s.toLocaleString()}</b></div><div><div className="small">Kept</div><b>{keep}%</b></div>
              <div><div className="small">Accuracy</div><b>{acc}%</b></div><div><div className="small">Energy saved</div><b style={{ color: 'var(--g2)' }}>{kwh} kWh</b></div></div>
            <button className="btn ghost" style={{ width: '100%' }} onClick={() => toast(`${n}: connect your backend to download`)}><Download size={15} /> Get dataset</button>
          </div>))}</div>}
    </main>
  );
}

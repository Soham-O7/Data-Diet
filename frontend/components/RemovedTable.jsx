'use client';
import { useMemo, useState } from 'react';
import { ArrowUpDown, Undo2 } from 'lucide-react';
import { EmptyState } from './ui';
const REASONS = ['Duplicate', 'Likely mislabeled', 'Outlier', 'Low importance'];
const PER = 8;

export function RemovedTable({ rows, kind, onRestore }) {
  const [q, setQ] = useState(''); const [reason, setReason] = useState(''); const [sort, setSort] = useState({ key: null, dir: 1 });
  const [page, setPage] = useState(0); const [showPreview, setShowPreview] = useState(true);
  const data = useMemo(() => {
    let d = rows.filter((r) => (!reason || r.reason === reason) && r.id.toLowerCase().includes(q.toLowerCase()));
    if (sort.key) d = [...d].sort((a, b) => (a[sort.key] > b[sort.key] ? 1 : -1) * sort.dir);
    return d;
  }, [rows, q, reason, sort]);
  const pages = Math.max(1, Math.ceil(data.length / PER)), shown = data.slice(page * PER, page * PER + PER);
  const th = (key, label) => <th><button onClick={() => setSort({ key, dir: sort.key === key ? -sort.dir : 1 })} style={{ background: 'none', border: 0, color: 'inherit', cursor: 'pointer', font: 'inherit' }}>{label} <ArrowUpDown size={12} /></button></th>;
  return (<div>
    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 12 }}>
      <input type="search" placeholder="Search sample ID" aria-label="Search" value={q} onChange={(e) => { setQ(e.target.value); setPage(0); }} />
      <select value={reason} onChange={(e) => { setReason(e.target.value); setPage(0); }} aria-label="Filter by reason"><option value="">All reasons</option>{REASONS.map((r) => <option key={r}>{r}</option>)}</select>
      <label className="small" style={{ alignSelf: 'center' }}><input type="checkbox" checked={showPreview} onChange={(e) => setShowPreview(e.target.checked)} /> Show preview column</label>
    </div>
    {shown.length === 0 ? <EmptyState title="No samples match" hint="Clear the search or pick another reason." /> :
      <div className="scroll"><table><thead><tr>{showPreview && <th>{kind === 'image' ? 'Thumbnail' : 'Sample row'}</th>}{th('id', 'Sample ID')}{th('score', 'Importance')}{th('reason', 'Reason')}<th /></tr></thead>
        <tbody>{shown.map((r) => <tr key={r.id}>
          {showPreview && <td>{kind === 'image' ? <div className="tile" style={{ width: 40 }}>#</div> : <span className="small">row {r.id.slice(2)}: …</span>}</td>}
          <td>{r.id}</td><td>{r.score.toFixed(2)}</td><td><span className="pill">{r.reason}</span></td>
          <td><button className="btn ghost" style={{ padding: '4px 10px' }} onClick={() => onRestore && onRestore(r.id)}><Undo2 size={13} /> Restore</button></td></tr>)}</tbody></table></div>}
    <div style={{ display: 'flex', gap: 8, marginTop: 12, alignItems: 'center' }}>
      <button className="btn ghost" disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</button>
      <button className="btn ghost" disabled={page + 1 >= pages} onClick={() => setPage(page + 1)}>Next</button>
      <span className="small">Page {page + 1} of {pages} · {data.length} samples</span></div>
  </div>);
}

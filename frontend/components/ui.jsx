'use client';
import { useEffect, useRef, useState } from 'react';
import { Info, AlertTriangle, Inbox } from 'lucide-react';

export function Counter({ to, dec = 0 }) {
  const [v, setV] = useState(0); const last = useRef(0);
  useEffect(() => { let f; const s = performance.now(), from = last.current;
    const run = (t) => { const p = Math.min((t - s) / 600, 1); last.current = from + (to - from) * p; setV(last.current); if (p < 1) f = requestAnimationFrame(run); };
    f = requestAnimationFrame(run); return () => cancelAnimationFrame(f); }, [to]);
  return <>{v.toFixed(dec)}</>;
}
export const Tip = ({ text }) => <span className="tipw" tabIndex={0} data-tip={text} role="note" aria-label={text}><Info size={14} /></span>;
export function MetricCard({ icon: Icon, title, value, unit, note, dec = 0, tag, good }) {
  return (<div className={'card metric' + (good ? ' good' : '')}>
    <div className="t"><Icon size={16} color="var(--g2)" />{title}{tag && <span className={'tag ' + (tag === 'measured' ? 'm' : 'e')}>{tag}</span>}</div>
    <div><span className="v"><Counter to={value} dec={dec} /></span><span className="u">{unit}</span></div>
    {note && <div className="small" style={{ marginTop: 4 }}>{note}</div>}
  </div>);
}
export const ProgressBar = ({ value }) => <div className="bar" role="progressbar" aria-valuenow={value} aria-valuemin={0} aria-valuemax={100}><i style={{ width: value + '%' }} /></div>;
export const Skeleton = ({ h = 120, n = 1 }) => <div className="grid g3">{Array.from({ length: n }, (_, i) => <div key={i} className="skel" style={{ height: h }} />)}</div>;
export const EmptyState = ({ title, hint }) => <div className="empty"><Inbox size={36} /><h3 style={{ marginTop: 8 }}>{title}</h3><p className="small">{hint}</p></div>;
export const ErrorState = ({ msg, retry }) => <div className="banner warn"><AlertTriangle size={16} style={{ verticalAlign: -3 }} /> {msg} {retry && <button className="btn ghost" onClick={retry} style={{ marginLeft: 12, padding: '4px 12px' }}>Try again</button>}</div>;
export const PageHead = ({ crumbs, title, sub }) => <div style={{ margin: '12px 0 24px' }}><div className="crumb">{crumbs}</div><h2 style={{ fontSize: '1.9rem' }}>{title}</h2>{sub && <p className="muted">{sub}</p>}</div>;

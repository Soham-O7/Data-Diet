'use client';
import Link from 'next/link';
import { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { Check, X, Loader2 } from 'lucide-react';
import { api } from '@/lib/api';
import { useApp } from '@/lib/hooks';
import { ProgressBar, PageHead, ErrorState, Skeleton } from '@/components/ui';
import { Shell } from '@/components/Sidebar';

export default function JobPage() {
  const { jobId } = useParams(); const router = useRouter(); const { config, toast } = useApp();
  const [j, setJ] = useState(null); const [err, setErr] = useState(false);
  useEffect(() => {
    let live = true, timer;
    const tick = async () => { try { const d = await api.getJob(jobId); if (!live) return; setJ(d); setErr(false); if (d.status === 'done') { api.recordRun(jobId, d.datasetId, config); toast('Run saved to History'); }
      if (d.status === 'running') timer = setTimeout(tick, 1000); } catch { if (live) setErr(true); } };
    tick(); return () => { live = false; clearTimeout(timer); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);
  const reload = async () => { setErr(false); setJ(await api.getJob(jobId)); };
  if (err) return <Shell><ErrorState msg="Lost contact with this job." retry={reload} /></Shell>;
  if (!j) return <Shell><Skeleton h={160} /></Shell>;
  const retry = async () => { const r = await api.startJob(j.datasetId, config); router.push(`/jobs/${r.jobId}`); };
  const cancel = async () => { await api.cancelJob(jobId); toast('Run cancelled'); setJ(await api.getJob(jobId)); };
  return (
    <Shell>
      <PageHead crumbs={`Runs / ${jobId}`} title={j.status === 'done' ? 'Run complete' : j.status === 'cancelled' ? 'Run cancelled' : 'Running…'} />
      <div className="card"><div className="stages">{j.stages.map((s) => (
        <div key={s.name} className={'stage ' + s.status}><div className="d">{s.status === 'done' ? <Check size={16} /> : s.status === 'failed' ? <X size={16} /> : s.status === 'active' ? <Loader2 size={16} className="spin" /> : null}</div>{s.name}<div className="small">{s.secs ? s.secs + 's' : ''}</div></div>))}</div></div>
      <div className="card"><div style={{ display: 'flex', justifyContent: 'space-between' }}><b>{j.progress}% complete</b><span className="small">Elapsed {j.elapsedSec}s</span></div><div style={{ marginTop: 8 }}><ProgressBar value={j.progress} /></div></div>
      <div className="term" aria-live="polite">{j.logs.map((l, i) => <div key={i}><span className="small">{l.t}</span> <span className="info">[{l.stage}]</span> <span className={l.level}>{l.level === 'ok' ? '✓' : '•'} {l.msg}</span></div>)}</div>
      <div style={{ display: 'flex', gap: 12 }}>
        {j.status === 'running' && <button className="btn ghost" onClick={cancel}>Cancel run</button>}
        {(j.status === 'failed' || j.status === 'cancelled') && <button className="btn" onClick={retry}>Retry run</button>}
        {j.status === 'done' && <Link className="btn" href={`/jobs/${jobId}/results`}>See results</Link>}
      </div>
    </Shell>
  );
}

'use client';
import { useRouter, useParams } from 'next/navigation';
import { useState } from 'react';
import { Clock, Zap, Leaf, IndianRupee, Copy, Files, Tag, Activity } from 'lucide-react';
import { api } from '@/lib/api';
import { useAsync, useApp } from '@/lib/hooks';
import { MetricCard, Tip, Skeleton, ErrorState, PageHead } from '@/components/ui';
import { Shell } from '@/components/Sidebar';

const OPTS = [['exactDup', 'Exact duplicates', Copy, 'Removes byte-identical rows or images.'], ['nearDup', 'Near duplicates', Files, 'Removes samples that are almost the same (hash, then MinHash/LSH).'], ['mislabeled', 'Possibly mislabeled', Tag, 'Flags samples whose label disagrees with their nearest neighbours.'], ['outliers', 'CSV outliers', Activity, 'Flags rows far outside the normal range. CSV only.']];

export default function ConfigureClient() {
  const { datasetId } = useParams(); const router = useRouter(); const { config: c, setConfig } = useApp();
  const [starting, setStarting] = useState(false);
  const ds = useAsync(() => api.getDataset(datasetId), [datasetId]);
  const sv = useAsync(() => (ds.data ? api.predictSavings(ds.data, c) : Promise.resolve(null)), [ds.data, c.keepRatio]);
  const run = async () => { setStarting(true); const r = await api.startJob(datasetId, c); router.push(`/jobs/${r.jobId}`); };
  if (ds.error) return <Shell><ErrorState msg="Could not load this dataset." retry={ds.refetch} /></Shell>;
  if (!ds.data) return <Shell><Skeleton h={200} n={3} /></Shell>;
  const keep = Math.round(ds.data.samples * c.keepRatio / 100), s = sv.data;
  return (
    <Shell>
      <PageHead crumbs={`New run / ${ds.data.name} / Configure`} title="Configure your diet" sub="Pick how much data to keep. Class proportions stay constant, so rare classes are not lost." />
      <div className="cfg">
        <div className="stack">
          <div className="card">
            <h3>Keep ratio <Tip text="Share of samples kept. Each class is sampled in proportion, so a class with 10% of the data still has 10% after pruning." /></h3>
            <p style={{ fontSize: '2.4rem', fontWeight: 700 }}>{c.keepRatio}%</p>
            <input type="range" className="range" min={10} max={90} step={5} value={c.keepRatio} disabled={c.autoRecommend} aria-label="Keep ratio" style={{ '--p': ((c.keepRatio - 10) / 80) * 100 + '%' }} onChange={(e) => setConfig({ keepRatio: Number(e.target.value) })} />
            <div className="grid g3" style={{ marginTop: 16 }}>
              <div><div className="small">Original</div><b>{ds.data.samples.toLocaleString()}</b></div>
              <div><div className="small">Kept</div><b style={{ color: 'var(--g2)' }}>{keep.toLocaleString()}</b></div>
              <div><div className="small">Removed</div><b>{(ds.data.samples - keep).toLocaleString()}</b></div>
            </div>
            <label className="chk" style={{ marginTop: 16 }}><input type="checkbox" checked={c.autoRecommend} onChange={(e) => setConfig({ autoRecommend: e.target.checked, keepRatio: e.target.checked ? 40 : c.keepRatio })} />
              <span><b>Recommend a ratio for me</b><br /><span className="small">We sweep 10–80% and pick the point where accuracy starts to fall.</span></span></label>
          </div>
          <div className="card"><h3>Cleaning</h3>
            <div className="grid g2" style={{ marginTop: 12 }}>{OPTS.map(([k, t, Icon, d]) => (
              <label key={k} className={'chk' + (c[k] ? ' on' : '')}><input type="checkbox" checked={c[k]} onChange={(e) => setConfig({ [k]: e.target.checked })} /><span><Icon size={14} /> <b>{t}</b><br /><span className="small">{d}</span></span></label>))}</div>
          </div>
          <div className="card grid g2">
            <div><h3>Model</h3><select value={c.model} onChange={(e) => setConfig({ model: e.target.value })} style={{ marginTop: 8 }}><option value="builtin">Built-in model</option><option value="custom">My own training script</option></select></div>
            <div><h3>Random seed</h3><input type="number" value={c.seed} onChange={(e) => setConfig({ seed: Number(e.target.value) })} style={{ marginTop: 8 }} /></div>
          </div>
        </div>
        <div className="stack">
          <h3 className="muted">Predicted savings per training run <span className="tag e">predicted</span></h3>
          <div className="grid g2">
            <MetricCard icon={Clock} title="Time saved" value={s ? s.timeMin : 0} dec={1} unit="min" note="vs. training on all data" />
            <MetricCard icon={Zap} title="Energy saved" value={s ? s.kwh : 0} dec={2} unit="kWh" good />
            <MetricCard icon={Leaf} title="CO₂ avoided" value={s ? s.co2kg : 0} dec={2} unit="kg" good note="kWh × grid intensity" />
            <MetricCard icon={IndianRupee} title="Money saved" value={s ? s.inr : 0} dec={1} unit="₹" />
          </div>
          <p className="small">Estimates from dataset size and keep ratio. Measured numbers appear after your run.</p>
          <button className="btn" style={{ width: '100%' }} disabled={starting} onClick={run}>{starting ? 'Starting…' : 'Run DataDiet'}</button>
        </div>
      </div>
    </Shell>
  );
}

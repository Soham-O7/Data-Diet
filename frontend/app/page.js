'use client';
import Link from 'next/link';
import { useRef, useState } from 'react';
import { UploadCloud, Database, Layers, HeartPulse, Gauge, Cpu, Terminal } from 'lucide-react';
import { api } from '@/lib/api';
import { MetricCard, Tip, ErrorState } from '@/components/ui';
import { Bars, Ring } from '@/components/charts';
import { HeroArt } from '@/components/HeroArt';

const SAMPLES = [['mnist', 'MNIST Images'], ['iris', 'Iris CSV'], ['adult', 'Adult Income CSV']];
const MAX_MB = 500;

export default function Home() {
  const [ds, setDs] = useState(null); const [pct, setPct] = useState(0); const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(''); const [target, setTarget] = useState(''); const [over, setOver] = useState(false);
  const input = useRef(null);

  const upload = async (file) => {
    if (!file) return;
    if (!/\.(csv|zip)$/i.test(file.name) || file.size > MAX_MB * 1e6) { setErr(`Use one .csv or .zip file under ${MAX_MB} MB.`); return; }
    setErr(''); setBusy(true); setPct(10);
    const t = setInterval(() => setPct((p) => Math.min(p + 12, 90)), 150);
    try { setDs(await api.uploadDataset(file)); } catch (e) { setErr(e.message); } finally { clearInterval(t); setPct(100); setBusy(false); }
  };
  const sample = async (id) => { setErr(''); setDs(await api.getDataset(id)); setTimeout(() => document.getElementById('profile')?.scrollIntoView({ behavior: 'smooth' }), 400); };

  return (
    <main className="wrap">
      <section className="hero">
        <div>
          <h1>Make AI training lighter. Make the planet greener.</h1>
          <p className="muted" style={{ margin: '16px 0', maxWidth: 520 }}>Optimize your datasets, keep model performance, and cut the compute, electricity and carbon behind every training run.</p>
          <div className="flow" aria-label="Workflow"><span>Clean</span><span>Prune</span><span>Prove</span></div>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
            <a className="btn" href="#upload">Optimize your dataset</a>
            <button className="btn ghost" onClick={() => sample('mnist')}>Explore demo</button>
            <Link className="btn ghost" href="/dashboard">Open dashboard</Link>
          </div>
        </div>
        <HeroArt />
      </section>

      <section id="upload" className="card">
        <div className={'drop' + (over ? ' on' : '')} role="button" tabIndex={0} onClick={() => input.current.click()} onKeyDown={(e) => e.key === 'Enter' && input.current.click()}
          onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)} onDrop={(e) => { e.preventDefault(); setOver(false); upload(e.dataTransfer.files[0]); }}>
          <input ref={input} type="file" accept=".csv,.zip" hidden aria-label="Upload dataset" onChange={(e) => upload(e.target.files[0])} />
          <UploadCloud size={36} color="var(--g2)" />
          <h2 style={{ marginTop: 8 }}>{over ? 'Drop it here' : 'Drop a CSV or a ZIP of images'}</h2>
          <p className="small">or click to browse · up to {MAX_MB} MB</p>
        </div>
        {busy && <div style={{ marginTop: 16 }}><div className="bar"><i style={{ width: pct + '%' }} /></div><p className="small">Uploading… {pct}%</p></div>}
        {err && <div style={{ marginTop: 16 }}><ErrorState msg={err} /></div>}
        <div style={{ display: 'flex', gap: 8, marginTop: 16, flexWrap: 'wrap', alignItems: 'center' }}>
          <span className="small">No file handy? Try a sample:</span>
          {SAMPLES.map(([id, n]) => <button key={id} className="btn ghost" style={{ padding: '6px 14px' }} onClick={() => sample(id)}>{n}</button>)}
        </div>
      </section>

      {ds && (
        <section id="profile" className="stack" style={{ marginTop: 32 }}>
          <h2>{ds.name}</h2>
          <div className="grid g4">
            <MetricCard icon={Database} title="Total samples" value={ds.samples} />
            <MetricCard icon={Layers} title="Classes" value={ds.classes.length} />
            <MetricCard icon={HeartPulse} title="Dataset size" value={ds.sizeMB} dec={ds.sizeMB < 10 ? 2 : 0} unit="MB" />
            <div className="card" style={{ display: 'flex', alignItems: 'center', gap: 12 }}><Ring value={ds.health} label="Health score" /><Tip text="Higher is better. We estimate how much is redundant (near-copies) and noisy (labels that disagree with their neighbours)." /></div>
          </div>
          <div className="grid g2">
            <div className="card"><h3>Class distribution</h3><Bars data={ds.classes} keys={['count']} /></div>
            <div className="card"><h3>Preview</h3>
              {ds.kind === 'image'
                ? <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fill,minmax(56px,1fr))', marginTop: 12 }}>{Array.from({ length: 18 }, (_, i) => <div key={i} className="tile" title="Placeholder tile">{i % 10}</div>)}</div>
                : <>
                  <div style={{ margin: '12px 0' }}><label className="small">Target column </label>
                    <select value={target || ds.columns[ds.columns.length - 1].name} onChange={(e) => setTarget(e.target.value)}>{ds.columns.map((c) => <option key={c.name}>{c.name}</option>)}</select>
                    <span className="small"> · {ds.missingPct}% missing values</span></div>
                  <div className="scroll"><table><thead><tr>{ds.columns.map((c) => <th key={c.name}>{c.name}<br /><span className="tag">{c.type}</span>{c.missing > 0 && <span className="tag e"> {c.missing}% missing</span>}</th>)}</tr></thead>
                    <tbody>{ds.rows.map((r, i) => <tr key={i}>{r.map((v, j) => <td key={j}>{v}</td>)}</tr>)}</tbody></table></div></>}
            </div>
          </div>
          <Link className="btn" href={`/configure/${ds.id}`}><Gauge size={16} /> Configure diet</Link>
        </section>
      )}

      <section className="grid g3" style={{ marginTop: 40 }}>
        <div className="card hl"><Gauge style={{ color: 'var(--g2)' }} /><h3 style={{ marginTop: 8 }}>See everything on the dashboard</h3><p className="small">Costs, energy, emissions and every past run in one place. <Link href="/dashboard" style={{ color: 'var(--g2)' }}>Open it</Link>.</p></div>
        <div className="card"><Cpu color="var(--g2)" /><h3 style={{ marginTop: 8 }}>Bring your own model</h3><p className="small">Plug in a training script and costs are calibrated on your hardware. Select it on the configure screen.</p></div>
        <div className="card"><Terminal color="var(--g2)" /><h3 style={{ marginTop: 8 }}>Or use the CLI</h3><p className="small"><code>pip install datadiet</code> then <code>datadiet prune data/ --keep 0.4</code>. Coming with the backend.</p></div>
      </section>
    </main>
  );
}

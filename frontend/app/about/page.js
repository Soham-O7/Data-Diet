import { PageHead } from '@/components/ui';
const STEPS = [['Clean', 'Remove exact and near duplicates, likely mislabeled samples and outliers.'], ['Prune', 'Keep the smallest subset that stays inside your accuracy budget, with class proportions held constant.'], ['Prove', 'Train on full and pruned data, compare accuracy, and report time, kWh and CO₂ for both runs.']];
export default function About() {
  return (
    <main className="wrap stack">
      <PageHead crumbs="About" title="About DataDiet" sub="Less data. Same intelligence. Greener AI." />
      <div className="card hl"><p>DataDiet helps students, startups and small ML teams train on less data without losing accuracy. We score every sample for redundancy, difficulty and label noise with a cheap, training-free method, so choosing the subset does not burn the energy it is meant to save.</p></div>
      <div className="grid g3">{STEPS.map(([t, d]) => <div key={t} className="card"><h3>{t}</h3><p className="muted" style={{ marginTop: 6 }}>{d}</p></div>)}</div>
      <div className="grid g2">
        <div className="card"><h3>What makes it different</h3><p className="muted" style={{ marginTop: 6 }}>Most data tools decide where data is stored or how clean it is. DataDiet decides how much data a model needs, and shows the proof: accuracy, training time, kWh and CO₂, net of the selection cost.</p></div>
        <div className="card"><h3>Honest limits</h3><p className="muted" style={{ marginTop: 6 }}>CO₂ here is operational only, from kWh × grid intensity. Energy is an estimate unless measured with a power meter. Demo screens use sample data, never real benchmarks.</p></div>
      </div>
    </main>
  );
}

'use client';
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ReferenceDot, ResponsiveContainer, Tooltip, XAxis, YAxis, Area, AreaChart, Pie, PieChart, Cell } from 'recharts';

const COL = ['var(--g)', 'var(--info)', 'var(--warn)', 'var(--mint)'];
const tip = { contentStyle: { background: 'var(--elev)', border: '1px solid var(--line)', borderRadius: 8, color: 'var(--tx)' }, cursor: { fill: 'var(--hover)' } };
const ax = { stroke: 'var(--mut)', fontSize: 12, tickLine: false };
export function Bars({ data, keys, x = 'label', h = 260, unit }) {
  return (<ResponsiveContainer width="100%" height={h}><BarChart data={data}><CartesianGrid stroke="var(--line)" vertical={false} /><XAxis dataKey={x} {...ax} /><YAxis {...ax} unit={unit} width={48} /><Tooltip {...tip} /><Legend />
    {keys.map((k, i) => <Bar key={k} dataKey={k} fill={COL[i]} radius={[4, 4, 0, 0]} />)}</BarChart></ResponsiveContainer>);
}
export function Sweep({ data, rec }) {
  const p = data.find((d) => d.keep === rec);
  return (<ResponsiveContainer width="100%" height={300}><LineChart data={data}><CartesianGrid stroke="var(--line)" /><XAxis dataKey="keep" {...ax} unit="%" label={{ value: 'Keep ratio', position: 'insideBottom', offset: -2, fill: 'var(--mut)' }} />
    <YAxis yAxisId="a" {...ax} unit="%" domain={['auto', 100]} width={48} /><YAxis yAxisId="e" orientation="right" {...ax} unit=" kWh" width={64} /><Tooltip {...tip} /><Legend verticalAlign="top" />
    <Line yAxisId="a" dataKey="acc" name="Accuracy (%)" stroke="var(--g)" strokeWidth={2.5} dot={false} /><Line yAxisId="e" dataKey="kwh" name="Energy (kWh)" stroke="var(--warn)" strokeWidth={2} dot={false} />
    {p && <ReferenceDot yAxisId="a" x={p.keep} y={p.acc} r={8} fill="var(--g2)" stroke="#fff" label={{ value: 'Recommended', fill: 'var(--mint)', position: 'top' }} />}</LineChart></ResponsiveContainer>);
}
export function Trend({ data, k, name }) {
  return (<ResponsiveContainer width="100%" height={220}><AreaChart data={data}><CartesianGrid stroke="var(--line)" vertical={false} /><XAxis dataKey="date" {...ax} /><YAxis {...ax} width={48} /><Tooltip {...tip} />
    <Area dataKey={k} name={name} stroke="var(--g)" fill="var(--fillsoft)" strokeWidth={2} /></AreaChart></ResponsiveContainer>);
}
export function Ring({ value, label }) {
  const r = 52, c = 2 * Math.PI * r, color = value > 80 ? 'var(--g)' : value > 60 ? 'var(--warn)' : 'var(--err)';
  return (<div style={{ textAlign: 'center' }}><svg width="130" height="130" viewBox="0 0 130 130" role="img" aria-label={`${label} ${value} out of 100`}>
    <circle cx="65" cy="65" r={r} stroke="var(--track)" strokeWidth="10" fill="none" /><circle cx="65" cy="65" r={r} stroke={color} strokeWidth="10" fill="none" strokeLinecap="round" strokeDasharray={c} strokeDashoffset={c * (1 - value / 100)} transform="rotate(-90 65 65)" style={{ transition: 'stroke-dashoffset .8s' }} />
    <text x="65" y="72" textAnchor="middle" fill="var(--tx)" fontSize="28" fontWeight="700">{value}</text></svg><div className="small">{label}</div></div>);
}
export function Donut({ data, h = 220 }) {
  const cols = ['var(--g)', 'var(--info)', 'var(--warn)', 'var(--mint)'];
  return (<ResponsiveContainer width="100%" height={h}><PieChart><Tooltip {...tip} /><Legend />
    <Pie data={data} dataKey="value" nameKey="name" innerRadius="58%" outerRadius="85%" paddingAngle={3} stroke="none">{data.map((d, i) => <Cell key={d.name} fill={cols[i % cols.length]} />)}</Pie></PieChart></ResponsiveContainer>);
}

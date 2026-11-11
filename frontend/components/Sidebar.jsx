'use client';
import Link from 'next/link';
import { useState } from 'react';
import { usePathname } from 'next/navigation';
import { PlusCircle, LayoutDashboard, History, PanelLeft } from 'lucide-react';
const items = [['/', 'New run', PlusCircle], ['/dashboard', 'Dashboard', LayoutDashboard], ['/history', 'History', History]];
export function Sidebar() {
  const path = usePathname(); const [min, setMin] = useState(false);
  return (<aside className={'side' + (min ? ' min' : '')} aria-label="Sidebar">
    <button className="btn ghost" style={{ marginBottom: 8, padding: 8 }} onClick={() => setMin(!min)} aria-label="Collapse sidebar"><PanelLeft size={16} /></button>
    {items.map(([h, l, I]) => <Link key={h} href={h} className={path === h ? 'on' : ''} title={l}><I size={16} /><span>{l}</span></Link>)}
  </aside>);
}
export const Shell = ({ children }) => <div className="shell"><Sidebar /><main className="wrap stack" style={{ maxWidth: 1240 }}>{children}</main></div>;

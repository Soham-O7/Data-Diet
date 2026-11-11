'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState } from 'react';
import { Home, LayoutDashboard, History, BarChart3, Boxes, ReceiptText, Info, Menu, X, Moon, Sun, UserCircle } from 'lucide-react';
import { USE_MOCK } from '@/lib/api';
import { useApp } from '@/lib/hooks';
const items = [['/', 'Home', Home], ['/dashboard', 'Dashboard', LayoutDashboard], ['/history', 'History', History], ['/benchmark', 'Benchmark', BarChart3], ['/hub', 'Hub', Boxes], ['/receipt', 'Sustainability Receipt', ReceiptText], ['/about', 'About', Info]];
export function Navbar() {
  const path = usePathname(); const [open, setOpen] = useState(false); const { theme, toggleTheme } = useApp();
  const on = (h) => (h === '/' ? path === '/' : path.startsWith(h));
  return (
    <header className="nav"><div className="nav-in">
      <Link href="/" className="brand" aria-label="DataDiet home">
        <span className="logo-badge"><img src="/logo.png" alt="" width="30" height="30" /></span>
        <span className="brand-txt">DataDiet<small>Less data. Greener AI.</small></span>
      </Link>
      <span className="spacer" />
      <nav className={'pillnav' + (open ? ' open' : '')} aria-label="Main" onClick={() => setOpen(false)}>
        {items.map(([h, l, Icon]) => <Link key={h} href={h} className={on(h) ? 'on' : ''} aria-current={on(h) ? 'page' : undefined}><Icon size={16} /><span>{l}</span></Link>)}
      </nav>
      <span className="spacer" />
      {USE_MOCK && <span className="demo">DEMO MODE · SAMPLE DATA</span>}
      <button className="theme-btn" onClick={toggleTheme} aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`} title="Appearance">{theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}</button>
      <UserCircle aria-label="Profile" color="var(--tx2)" />
      <button className="burger2 theme-btn" onClick={() => setOpen(!open)} aria-label="Toggle menu" aria-expanded={open}>{open ? <X size={18} /> : <Menu size={18} />}</button>
    </div></header>
  );
}

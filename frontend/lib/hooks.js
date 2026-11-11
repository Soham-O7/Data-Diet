'use client';
import { createContext, useCallback, useContext, useEffect, useState } from 'react';


// Tiny fetch hook (replaces a data-fetching library). Keeps old data while refetching.
export function useAsync(fn, deps) {
  const [s, setS] = useState({ data: null, error: null, loading: true });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const run = useCallback(() => { let live = true; setS((p) => ({ ...p, loading: true, error: null })); fn().then((d) => live && setS({ data: d, error: null, loading: false })).catch((e) => live && setS({ data: null, error: e, loading: false })); return () => { live = false; }; }, deps);
  useEffect(() => run(), [run]);
  return { ...s, refetch: run };
}

// App-wide state with plain React context (replaces a state library)
export const defaultConfig = { keepRatio: 40, exactDup: true, nearDup: true, mislabeled: true, outliers: false, autoRecommend: false, model: 'builtin', seed: 3 };
const Ctx = createContext(null);
export function AppProvider({ children }) {
  const [config, setC] = useState(defaultConfig);
  const [toasts, setT] = useState([]);
  const [theme, setThemeState] = useState('dark');
  useEffect(() => { const t = document.documentElement.dataset.theme; if (t === 'light' || t === 'dark') setThemeState(t); }, []);
  const toggleTheme = () => { const n = theme === 'dark' ? 'light' : 'dark'; setThemeState(n); document.documentElement.dataset.theme = n; try { localStorage.setItem('datadiet-theme', n); } catch {} };
  const setConfig = (p) => setC((c) => ({ ...c, ...p }));
  const toast = (msg) => { const id = Date.now() + Math.random(); setT((t) => [...t, { id, msg }]); setTimeout(() => setT((t) => t.filter((x) => x.id !== id)), 3500); };
  return (<Ctx.Provider value={{ config, setConfig, toast, theme, toggleTheme }}>{children}
    <div className="toasts" aria-live="polite">{toasts.map((t) => <div key={t.id} className="toast">{t.msg}</div>)}</div></Ctx.Provider>);
}
export const useApp = () => useContext(Ctx);

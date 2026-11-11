import { Inter } from 'next/font/google';
import './globals.css';
import { AppProvider } from '@/lib/hooks';
import { Navbar } from '@/components/Navbar';
const inter = Inter({ subsets: ['latin'] });
export const metadata = { title: 'DataDiet: Less data. Same intelligence. Greener AI.' };
// Sets the saved theme before first paint so there is no flash
const themeScript = `try{var t=localStorage.getItem('datadiet-theme');if(!t&&window.matchMedia('(prefers-color-scheme: light)').matches)t='light';document.documentElement.dataset.theme=t||'dark'}catch(e){document.documentElement.dataset.theme='dark'}`;
export default function RootLayout({ children }) {
  return (
    <html lang="en" data-theme="dark" suppressHydrationWarning>
      <head><script dangerouslySetInnerHTML={{ __html: themeScript }} /></head>
      <body className={inter.className}><AppProvider><Navbar />{children}</AppProvider></body>
    </html>
  );
}

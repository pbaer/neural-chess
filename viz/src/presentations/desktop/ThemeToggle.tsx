// ThemeToggle — flips the site between light and dark by setting
// document.documentElement.dataset.theme (the CSS variables in styles.css react)
// and persisting the choice. The initial theme is applied by an inline script in
// index.html before first paint; this just reads/updates it. NOTE: the heatmap
// color SCALES (colormap.ts) are deliberately theme-independent and never change.

import { useState } from 'react';

type Theme = 'light' | 'dark';

function readTheme(): Theme {
  return document.documentElement.dataset.theme === 'light' ? 'light' : 'dark';
}

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(readTheme);

  const toggle = () => {
    const next: Theme = theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem('nc-theme', next);
    } catch {
      /* private mode / storage disabled — fine, just won't persist */
    }
    setTheme(next);
  };

  const next = theme === 'dark' ? 'light' : 'dark';
  return (
    <button
      className="btn theme-toggle"
      onClick={toggle}
      aria-label={`Switch to ${next} theme`}
      title={`Switch to ${next} theme`}
    >
      {theme === 'dark' ? '☀ Light' : '☾ Dark'}
    </button>
  );
}

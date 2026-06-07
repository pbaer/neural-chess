// Entry point — mounts the desktop presentation. A future mobile/lesson-embed
// presentation would swap only this import; the entire src/core is reused.

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './presentations/desktop/App.tsx';
import './presentations/desktop/styles.css';

const rootEl = document.getElementById('root');
if (!rootEl) throw new Error('Missing #root element.');

createRoot(rootEl).render(
  <StrictMode>
    <App />
  </StrictMode>,
);

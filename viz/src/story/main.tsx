// Entry point for the "story" page (story.html). A standalone narrative page,
// separate from the play/inspector app; reuses the same theme variables.

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { Story } from './Story.tsx';
import './story.css';

const rootEl = document.getElementById('root');
if (!rootEl) throw new Error('Missing #root element.');

createRoot(rootEl).render(
  <StrictMode>
    <Story />
  </StrictMode>,
);

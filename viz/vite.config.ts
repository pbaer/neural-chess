import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Static SPA. `base` is set for GitHub Pages project sites (set DEPLOY_BASE=/ for
// Netlify / root deploys). The inference Web Worker is bundled as an ES module
// (Comlink + the engine kernel), and `import.meta.env.BASE_URL` is used to locate
// the model capsule so lazy weight fetches respect the deploy base.
export default defineConfig({
  base: process.env.DEPLOY_BASE ?? '/neural-chess/',
  plugins: [react()],
  worker: {
    format: 'es',
  },
  build: {
    target: 'es2022',
    sourcemap: true,
  },
});

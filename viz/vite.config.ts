import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Static multi-page site. `base` is set for GitHub Pages project sites (set
// DEPLOY_BASE=/ for Netlify / root deploys). The inference Web Worker is bundled
// as an ES module (Comlink + the engine kernel), and `import.meta.env.BASE_URL`
// locates the model capsule so lazy weight fetches respect the deploy base.
// Two pages: index.html (play + inspector) and story.html (the project story).
export default defineConfig({
  base: process.env.DEPLOY_BASE ?? '/neural-chess/',
  plugins: [react()],
  worker: {
    format: 'es',
  },
  build: {
    target: 'es2022',
    sourcemap: true,
    rollupOptions: {
      input: {
        main: fileURLToPath(new URL('./index.html', import.meta.url)),
        story: fileURLToPath(new URL('./story.html', import.meta.url)),
      },
    },
  },
});

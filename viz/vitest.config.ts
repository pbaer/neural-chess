import { defineConfig } from 'vitest/config';

// Two flavours of test share this config:
//   - Logic/parity tests run in pure Node (fast). They read golden.bin from disk
//     and exercise the TS engine; no DOM needed.
//   - Component/UX tests (files named *.dom.test.tsx) opt into jsdom via
//     environmentMatchGlobs so React Testing Library can render and drive real
//     user interactions. tests/setup.ts adds the jest-dom matchers.
export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    environmentMatchGlobs: [['**/*.dom.test.{ts,tsx}', 'jsdom']],
    include: ['tests/**/*.test.{ts,tsx}', 'src/**/*.test.{ts,tsx}'],
    setupFiles: ['./tests/setup.ts'],
    testTimeout: 30000,
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html'],
      // App source only; entrypoints, generated capsule loaders, and test files
      // aren't unit-testable in isolation and would only dilute the signal.
      include: ['src/**/*.{ts,tsx}'],
      exclude: ['src/**/*.test.{ts,tsx}', 'src/**/main.tsx', 'src/**/*.d.ts'],
    },
  },
});

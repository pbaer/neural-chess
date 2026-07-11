import { defineConfig } from 'vitest/config';

// Two flavours of test share this config, split into vitest projects (the
// vitest-4 replacement for environmentMatchGlobs, which v4 removed):
//   - Logic/parity tests run in pure Node (fast). They read golden.bin from disk
//     and exercise the TS engine; no DOM needed.
//   - Component/UX tests (files named *.dom.test.{ts,tsx}) run in jsdom so React
//     Testing Library can render and drive real user interactions. The files
//     also carry a `// @vitest-environment jsdom` pragma, which makes each one
//     self-contained (the pragma wins over the project environment anyway).
// tests/setup.ts adds the jest-dom matchers in both projects.
const shared = {
  globals: true,
  setupFiles: ['./tests/setup.ts'],
  testTimeout: 30000,
} as const;

export default defineConfig({
  test: {
    projects: [
      {
        test: {
          ...shared,
          name: 'node',
          environment: 'node',
          include: ['tests/**/*.test.{ts,tsx}', 'src/**/*.test.{ts,tsx}'],
          exclude: ['**/*.dom.test.{ts,tsx}', '**/node_modules/**'],
        },
      },
      {
        test: {
          ...shared,
          name: 'dom',
          environment: 'jsdom',
          include: ['tests/**/*.dom.test.{ts,tsx}', 'src/**/*.dom.test.{ts,tsx}'],
        },
      },
    ],
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

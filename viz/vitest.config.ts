import { defineConfig } from 'vitest/config';

// Parity tests are pure Node (they read golden.bin from disk and run the TS
// engine). No DOM needed for M1.
export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    include: ['tests/**/*.test.ts', 'src/**/*.test.ts'],
    testTimeout: 30000,
  },
});

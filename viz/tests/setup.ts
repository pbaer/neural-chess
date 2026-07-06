// Vitest global setup. Adds jest-dom matchers (toBeInTheDocument, toBeDisabled,
// …) for the jsdom component tests. React Testing Library auto-registers its
// afterEach cleanup when Vitest globals are enabled (test.globals in
// vite.config.ts), so no manual cleanup is needed here.
//
// This file runs for every test file (Node and jsdom); the jest-dom import is a
// harmless no-op for the Node-environment logic tests.
import '@testing-library/jest-dom/vitest';

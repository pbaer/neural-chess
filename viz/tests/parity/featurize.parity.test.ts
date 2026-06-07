// Featurize parity: TS featurize() vs the PyTorch golden planes, in BOTH modes
// ('float' = inference.py path; 'trained' = int8-truncated training storage),
// for every FEN. Exercises piece planes, castling, en-passant, counters,
// repetition, and the black-to-move vertical-mirror rotation.

import { describe, it, expect } from 'vitest';
import { featurize } from '../../src/core/engine/index.ts';
import { loadGolden, buildBoardState, maxAbsDiff } from './fixtures.ts';

const TOL = 1e-6; // both sides compute identical float arithmetic

const { golden, tensor } = loadGolden();

describe('featurize parity (TS vs PyTorch)', () => {
  for (const c of golden.cases) {
    describe(c.name, () => {
      const board = buildBoardState(c.realBoard);

      it("mode 'float' matches golden planes", () => {
        const ts = featurize(board, 'float');
        const g = tensor(c.tensors.planes);
        expect(ts.length).toBe(g.length);
        const d = maxAbsDiff(ts, g);
        expect(d, `float planes max-abs ${d}`).toBeLessThan(TOL);
      });

      it("mode 'trained' matches int8-truncated golden planes", () => {
        const ts = featurize(board, 'trained');
        const g = tensor(c.tensors.planesTrained);
        expect(ts.length).toBe(g.length);
        const d = maxAbsDiff(ts, g);
        expect(d, `trained planes max-abs ${d}`).toBeLessThan(TOL);
      });
    });
  }
});

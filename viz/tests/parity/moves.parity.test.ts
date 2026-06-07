// Move-encoding parity: encodeMove / decodeMove / legalMask / rotateSquare vs
// the PyTorch golden (src/v2/moves.py). Also exercises the exact-erf used by GELU.

import { describe, it, expect } from 'vitest';
import {
  encodeMove,
  decodeMove,
  legalMask,
  rotateSquare,
  NUM_MOVES,
  type Move,
} from '../../src/core/engine/index.ts';
import { erf } from '../../src/core/engine/erf.ts';
import { loadGolden, buildBoardState } from './fixtures.ts';

const { golden } = loadGolden();

describe('rotateSquare == chess.square_mirror (sq ^ 56)', () => {
  it('matches python reference for all 64 squares', () => {
    for (let sq = 0; sq < 64; sq++) {
      expect(rotateSquare(sq)).toBe(golden.rotateSquareRef[sq]);
    }
  });
});

describe('encode / decode parity', () => {
  for (const c of golden.cases) {
    describe(c.name, () => {
      const board = buildBoardState(c.rotatedBoard);

      it('encodeMove matches golden index for every legal move', () => {
        for (const lm of c.legalMoves) {
          const mv: Move = { from: lm.from, to: lm.to, promotion: lm.prom || null };
          expect(encodeMove(mv), `encode ${lm.from}->${lm.to} prom${lm.prom}`).toBe(lm.index);
        }
      });

      it('decodeMove round-trips to golden (from/to/promotion)', () => {
        for (const lm of c.legalMoves) {
          const dec = decodeMove(lm.index, board);
          expect(dec.from, `decFrom @${lm.index}`).toBe(lm.decFrom);
          expect(dec.to, `decTo @${lm.index}`).toBe(lm.decTo);
          expect(dec.promotion ?? 0, `decProm @${lm.index}`).toBe(lm.decProm);
        }
      });

      it('legalMask matches golden legal indices', () => {
        const moves: Move[] = c.legalMoves.map((lm) => ({
          from: lm.from,
          to: lm.to,
          promotion: lm.prom || null,
        }));
        const mask = legalMask(moves);
        const expected = new Uint8Array(NUM_MOVES);
        for (const idx of c.legalIndices) expected[idx] = 1;
        let diff = 0;
        for (let i = 0; i < NUM_MOVES; i++) if (mask[i] !== expected[i]) diff++;
        expect(diff, `${diff} mask cells differ`).toBe(0);
      });

      it('decode of bestLegalIndex matches golden decoded move', () => {
        const dec = decodeMove(c.bestLegalIndex, board);
        expect(dec.from).toBe(c.decodedBest.from);
        expect(dec.to).toBe(c.decodedBest.to);
        expect(dec.promotion ?? 0).toBe(c.decodedBest.prom);
      });
    });
  }
});

describe('exact-erf accuracy (vs python math.erf)', () => {
  it('max abs error over dense grid < 1e-9', () => {
    let m = 0;
    for (let i = 0; i < golden.erfGrid.x.length; i++) {
      const d = Math.abs(erf(golden.erfGrid.x[i]) - golden.erfGrid.erf[i]);
      if (d > m) m = d;
    }
    expect(m, `erf max abs err ${m}`).toBeLessThan(1e-9);
  });
});

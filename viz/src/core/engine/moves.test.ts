// Policy aggregation tests — the legal-masked softmax and the per-from-square
// "piece-to-move" probability (SUM over the 73 move-types) used by the policy-head
// board shading in the Model Inspector.

import { describe, it, expect } from 'vitest';
import { legalPolicySoftmax, pieceToMoveProbs, NUM_MOVE_TYPES } from './moves.ts';

describe('legalPolicySoftmax', () => {
  it('softmaxes over only the legal indices and sums to 1', () => {
    const logits = new Float32Array(4672);
    // Three legal moves with distinct logits; everything else illegal.
    const a = 0 * NUM_MOVE_TYPES + 0; // from a1
    const b = 0 * NUM_MOVE_TYPES + 1; // from a1 (different move-type)
    const c = 8 * NUM_MOVE_TYPES + 5; // from a2
    logits[a] = 1;
    logits[b] = 1;
    logits[c] = 1;
    // An illegal index with a HUGE logit must be ignored entirely.
    logits[100] = 100;
    const probs = legalPolicySoftmax(logits, [a, b, c]);
    const sum = [...probs.values()].reduce((s, p) => s + p, 0);
    expect(sum).toBeCloseTo(1, 12);
    // Equal logits → equal share among the three legal moves.
    expect(probs.get(a)).toBeCloseTo(1 / 3, 12);
    expect(probs.get(c)).toBeCloseTo(1 / 3, 12);
    expect(probs.has(100)).toBe(false);
  });

  it('empty legal set returns an empty map', () => {
    expect(legalPolicySoftmax(new Float32Array(4672), []).size).toBe(0);
  });
});

describe('pieceToMoveProbs', () => {
  it('SUMS (not averages) move-type probability per from-square', () => {
    // Two moves from square 0, one from square 8.
    const probs = new Map<number, number>([
      [0 * NUM_MOVE_TYPES + 0, 0.5],
      [0 * NUM_MOVE_TYPES + 3, 0.2],
      [8 * NUM_MOVE_TYPES + 7, 0.3],
    ]);
    const ptm = pieceToMoveProbs(probs);
    expect(ptm).toHaveLength(64);
    // Square 0 = SUM of its two move-types (0.7), NOT the average (0.35).
    expect(ptm[0]).toBeCloseTo(0.7, 6); // Float32 accumulation
    expect(ptm[8]).toBeCloseTo(0.3, 6);
    // Untouched squares are 0; the whole board sums to the total probability mass.
    expect(ptm[1]).toBe(0);
    expect([...ptm].reduce((s, p) => s + p, 0)).toBeCloseTo(1, 6);
  });

  it('composes with legalPolicySoftmax to a normalized board', () => {
    const logits = new Float32Array(4672);
    const idxs = [0, 1, NUM_MOVE_TYPES + 4, NUM_MOVE_TYPES + 9, 2 * NUM_MOVE_TYPES];
    idxs.forEach((i, k) => (logits[i] = k * 0.3));
    const ptm = pieceToMoveProbs(legalPolicySoftmax(logits, idxs));
    expect([...ptm].reduce((s, p) => s + p, 0)).toBeCloseTo(1, 6);
    // Square 1 holds two moves; its mass = their combined probability.
    expect(ptm[1]).toBeGreaterThan(ptm[0]);
  });
});

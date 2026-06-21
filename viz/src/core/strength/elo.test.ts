import { describe, it, expect } from 'vitest';
import { estimateElo, ONESHOT_ELO, MCTS_ANCHORS } from './elo.ts';

describe('estimateElo', () => {
  it('returns the one-shot base at variety 0 when MCTS is off (sims ignored)', () => {
    expect(estimateElo(false, 100, 0)).toEqual(ONESHOT_ELO);
    expect(estimateElo(false, 300, 0)).toEqual(ONESHOT_ELO);
  });

  it('returns MCTS anchor values exactly at anchor sim-counts (variety 0)', () => {
    for (const a of MCTS_ANCHORS) {
      expect(estimateElo(true, a.sims, 0)).toEqual({ elo: a.elo, lo: a.lo, hi: a.hi });
    }
  });

  it('clamps below/above the anchor range to the nearest anchor', () => {
    const first = MCTS_ANCHORS[0];
    const last = MCTS_ANCHORS[MCTS_ANCHORS.length - 1];
    expect(estimateElo(true, 1, 0)).toEqual({ elo: first.elo, lo: first.lo, hi: first.hi });
    expect(estimateElo(true, 99999, 0)).toEqual({ elo: last.elo, lo: last.lo, hi: last.hi });
  });

  it('interpolates the base curve monotonically between anchors (variety 0)', () => {
    let prev = -Infinity;
    for (let s = 10; s <= 300; s += 10) {
      const e = estimateElo(true, s, 0).elo;
      expect(e).toBeGreaterThanOrEqual(prev - 1);
      prev = e;
    }
  });

  it('variety lowers (or holds) the estimate, monotonically', () => {
    for (const mcts of [false, true]) {
      let prev = Infinity;
      for (let v = 0; v <= 1.0001; v += 0.1) {
        const e = estimateElo(mcts, 100, v).elo;
        expect(e).toBeLessThanOrEqual(prev + 1); // non-increasing in variety
        prev = e;
      }
      // variety 0 is the strongest.
      expect(estimateElo(mcts, 100, 0).elo).toBeGreaterThanOrEqual(estimateElo(mcts, 100, 1).elo);
    }
  });

  it('variety shifts the whole interval by the same amount (spread preserved)', () => {
    const a = estimateElo(true, 100, 0);
    const b = estimateElo(true, 100, 1);
    expect(a.hi - a.lo).toBe(b.hi - b.lo);
    expect(a.elo - b.elo).toBeGreaterThan(0);
  });

  it('always reports lo <= elo <= hi', () => {
    for (const v of [0, 0.5, 1]) {
      for (let s = 10; s <= 300; s += 10) {
        const { elo, lo, hi } = estimateElo(true, s, v);
        expect(lo).toBeLessThanOrEqual(elo);
        expect(elo).toBeLessThanOrEqual(hi);
      }
    }
  });
});

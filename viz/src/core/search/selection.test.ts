// Tests pinning the value-adaptive, bounded move-selection behavior:
//   - losing → near-argmax (sharpens to the best move),
//   - winning → more spread (samples alternatives more often),
//   - NEVER plays outside the reasonable set (the hard floor),
//   - S=0 is fully deterministic,
//   - the value→temperature curve is monotonic in V.

import { describe, it, expect } from 'vitest';
import {
  selectMoveIndex,
  reasonableSetIndices,
  valueAdaptiveTemperature,
  DEFAULT_SELECTION_CONFIG,
  type SelectionCandidate,
} from './selection.ts';

/** The value modulation m(V) ∈ [0,1] recovered from T = S·maxTemp·m(V) at S=1. */
function moduleM(value: number): number {
  return valueAdaptiveTemperature(value, 1) / DEFAULT_SELECTION_CONFIG.maxTemp;
}

/** Mulberry32 — a small deterministic RNG so sampling tests are reproducible. */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Fraction of N samples that picked index 0 (the top move) under value V. */
function topShare(cands: SelectionCandidate[], value: number, variety: number, n = 4000): number {
  const r = rng(12345);
  let top = 0;
  for (let i = 0; i < n; i++) if (selectMoveIndex(cands, { value, variety, rng: r }) === 0) top++;
  return top / n;
}

describe('valueAdaptiveTemperature', () => {
  it('is exactly 0 when variety is 0 (fully deterministic, regardless of V)', () => {
    expect(valueAdaptiveTemperature(0, 0)).toBe(0); // S=0
    expect(valueAdaptiveTemperature(1, 0)).toBe(0); // S=0 even while winning
    expect(valueAdaptiveTemperature(-1, 0)).toBe(0); // S=0 even while losing
  });

  it('is effectively 0 (below the deterministic threshold) when fully losing', () => {
    expect(valueAdaptiveTemperature(-1, 1)).toBeLessThan(1e-3); // sharpen to best move
  });

  it('increases monotonically with the value V (losing → winning)', () => {
    const S = 1;
    const samples = [-1, -0.5, 0, 0.5, 1].map((v) => valueAdaptiveTemperature(v, S));
    for (let i = 1; i < samples.length; i++) {
      expect(samples[i]).toBeGreaterThan(samples[i - 1]);
    }
  });

  it('increases monotonically with the variety slider S', () => {
    expect(valueAdaptiveTemperature(0, 0.25)).toBeLessThan(valueAdaptiveTemperature(0, 0.75));
  });

  // The steep sigmoid m(V)=sigmoid(k·V): essentially saturated by |V|=0.5.
  describe('value modulation m(V) — steep sigmoid saturated by |V|=0.5', () => {
    it('m(−0.5) ≲ 0.02 — losing by half ⇒ near-strongest (T≈0)', () => {
      expect(moduleM(-0.5)).toBeLessThanOrEqual(0.02);
    });

    it('m(+0.5) ≳ 0.98 — winning by half ⇒ near-loosest (full variety)', () => {
      expect(moduleM(0.5)).toBeGreaterThanOrEqual(0.98);
    });

    it('m(0) ≈ 0.5 — moderate variety at an even position', () => {
      expect(moduleM(0)).toBeCloseTo(0.5, 5);
    });

    it('is monotonically increasing in V across the full range', () => {
      const vs = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1];
      const ms = vs.map(moduleM);
      for (let i = 1; i < ms.length; i++) expect(ms[i]).toBeGreaterThan(ms[i - 1]);
    });
  });
});

describe('reasonableSetIndices (the hard floor)', () => {
  const cands: SelectionCandidate[] = [
    { weight: 0.5 }, // top
    { weight: 0.3 },
    { weight: 0.12 },
    { weight: 0.05 }, // below 12% of top (0.06) → excluded
    { weight: 0.02 }, // far below → excluded
    { weight: 0.01 },
  ];

  it('keeps only moves at/above the relative-weight floor, capped at top-K', () => {
    const set = reasonableSetIndices(cands);
    expect(set).toContain(0);
    expect(set).toContain(1);
    expect(set).toContain(2);
    expect(set).not.toContain(3);
    expect(set).not.toContain(4);
    expect(set.length).toBeLessThanOrEqual(5);
  });

  it('drops MCTS moves whose Q is far below the best (losing alternatives)', () => {
    const mcts: SelectionCandidate[] = [
      { weight: 40, q: 0.2 }, // top-visited, good
      { weight: 35, q: 0.18 }, // close Q → kept
      { weight: 30, q: -0.5 }, // search judged losing → dropped
    ];
    const set = reasonableSetIndices(mcts);
    expect(set).toContain(0);
    expect(set).toContain(1);
    expect(set).not.toContain(2);
  });

  it('always includes the top move even if its own Q dips below the margin', () => {
    const mcts: SelectionCandidate[] = [
      { weight: 100, q: -0.3 }, // most-visited but low Q
      { weight: 10, q: 0.4 },
    ];
    expect(reasonableSetIndices(mcts)).toContain(0);
  });
});

describe('selectMoveIndex — value-adaptive variety', () => {
  // A clear top move with a couple of reasonable alternatives.
  const cands: SelectionCandidate[] = [
    { weight: 0.45 },
    { weight: 0.3 },
    { weight: 0.18 },
    { weight: 0.04 }, // below floor — never reachable
    { weight: 0.03 },
  ];

  it('S=0 is fully deterministic → always the top move', () => {
    const r = rng(7);
    for (let i = 0; i < 200; i++) {
      expect(selectMoveIndex(cands, { value: 0.5, variety: 0, rng: r })).toBe(0);
    }
  });

  it('losing → near-argmax; winning → more spread (P(top) drops)', () => {
    const losing = topShare(cands, -0.9, 1);
    const even = topShare(cands, 0, 1);
    const winning = topShare(cands, 0.9, 1);
    // Sharpest when losing, most spread when winning.
    expect(losing).toBeGreaterThan(even);
    expect(even).toBeGreaterThan(winning);
    // Losing essentially always plays the best move.
    expect(losing).toBeGreaterThan(0.95);
    // Winning still picks the top move sometimes, but clearly explores.
    expect(winning).toBeLessThan(0.8);
  });

  it('NEVER plays outside the reasonable set, even when fully winning at max variety', () => {
    const r = rng(999);
    for (let i = 0; i < 5000; i++) {
      const idx = selectMoveIndex(cands, { value: 1, variety: 1, rng: r });
      expect(idx).toBeLessThanOrEqual(2); // indices 3,4 are below the floor
    }
  });

  it('a single reasonable move is always chosen (degenerate distribution)', () => {
    const one: SelectionCandidate[] = [{ weight: 1 }, { weight: 0.001 }];
    expect(selectMoveIndex(one, { value: 1, variety: 1, rng: rng(1) })).toBe(0);
  });
});

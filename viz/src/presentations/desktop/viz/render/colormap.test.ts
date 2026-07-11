// colormap tests — pure color math (no DOM).

import { describe, it, expect } from 'vitest';
import { diverging, sequential, rgbCss, rangeOf, autoMode, niceSeq, colorScaleOf } from './colormap.ts';

describe('diverging (two-slope over [lo, hi])', () => {
  it('matches the sequential zero color at value 0', () => {
    expect(diverging(0, -1, 1)).toEqual(sequential(0, 0, 1));
  });

  it('greens toward hi, reddens toward lo', () => {
    const pos = diverging(1, -1, 1);
    const neg = diverging(-1, -1, 1);
    expect(pos[1]).toBeGreaterThan(pos[0]); // green channel dominates (positive)
    expect(pos[1]).toBeGreaterThan(pos[2]);
    expect(neg[0]).toBeGreaterThan(neg[1]); // red channel dominates (negative)
    expect(neg[0]).toBeGreaterThan(neg[2]);
  });

  it('scales each side independently to its own extent', () => {
    expect(diverging(2, -1, 2)).toEqual(diverging(1, -1, 1)); // both = full green (at hi)
    expect(diverging(1, -1, 2)).not.toEqual(diverging(2, -1, 2)); // v=1 only halfway up
    expect(diverging(-1, -1, 2)).toEqual(diverging(-1, -1, 1)); // both = full red (at lo)
  });

  it('clamps past the range and degrades to zero when a side has no extent', () => {
    expect(diverging(5, -1, 1)).toEqual(diverging(1, -1, 1)); // clamps to hi
    expect(diverging(3, 0, 0)).toEqual(sequential(0, 0, 1)); // no positive extent → zero
  });
});

describe('sequential', () => {
  it('is monotonic-ish from low to high and clamps', () => {
    const lo = sequential(0, 0, 1);
    const hi = sequential(1, 0, 1);
    expect(hi[0] + hi[1]).toBeGreaterThan(lo[0] + lo[1]); // brighter at the top
    expect(sequential(2, 0, 1)).toEqual(sequential(1, 0, 1));
    expect(sequential(-1, 0, 1)).toEqual(sequential(0, 0, 1));
  });
});

describe('reductions + helpers', () => {
  it('rangeOf', () => {
    expect(rangeOf(new Float32Array([-3, 1, 2]))).toEqual([-3, 2]);
    expect(rangeOf(new Float32Array([]))).toEqual([0, 0]);
  });

  it('autoMode picks diverging only when values go negative', () => {
    expect(autoMode(new Float32Array([0, 0.5, 1]))).toBe('sequential');
    expect(autoMode(new Float32Array([-0.1, 0.5]))).toBe('diverging');
  });

  it('niceSeq snaps near-0/1 ends and expands a constant field', () => {
    expect(niceSeq(0.002, 0.5)).toEqual([0, 0.5]); // tiny min snaps to 0
    expect(niceSeq(0, 0.998)).toEqual([0, 1]); // near-1 max snaps to 1
    expect(niceSeq(0.25, 0.75)).toEqual([0.25, 0.75]); // genuine range kept as-is
    const [lo, hi] = niceSeq(1, 1); // constant non-zero field → usable span from 0
    expect(lo).toBe(0);
    expect(hi).toBe(1);
  });

  it('colorScaleOf spans the actual data range (float32-exact values)', () => {
    expect(colorScaleOf(new Float32Array([0, 0.25, 0.5]))).toEqual({ mode: 'sequential', range: [0, 0.5] });
    expect(colorScaleOf(new Float32Array([-0.25, 0.5]))).toEqual({ mode: 'diverging', range: [-0.25, 0.5] });
  });

  it('rgbCss formats', () => {
    expect(rgbCss([1, 2, 3])).toBe('rgb(1,2,3)');
  });
});

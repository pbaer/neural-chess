// Colormaps for tensor heatmaps — pure functions, no DOM. Two families, BOTH
// parameterised by the data's actual [lo, hi] range (never a fixed 0..1):
//   diverging  — signed values (weights, scores, residuals): a two-slope scale
//                red(lo) → ZERO(0) → green(hi). The negative and positive sides
//                are scaled independently so the bar spans exactly [lo, hi] while
//                0 always lands on the neutral ZERO color.
//   sequential — non-negative values (probabilities, magnitudes): ZERO(lo) →
//                bright ramp(hi).
// Both families share the SAME color at value 0 (the `ZERO` control point), so a
// zero reads identically across the two scales.
// Returned as [r,g,b] in 0..255; rgbCss() formats for canvas/SVG fills.

export type RGB = [number, number, number];

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function lerpRGB(a: RGB, b: RGB, t: number): RGB {
  return [Math.round(lerp(a[0], b[0], t)), Math.round(lerp(a[1], b[1], t)), Math.round(lerp(a[2], b[2], t))];
}

// The shared value-0 color: the low end of the sequential ramp AND the midpoint
// of the diverging scale, so a zero looks the same on either. Kept a mid-dark
// slate-blue (not near-black) so chess pieces drawn over a heatmap stay legible.
const ZERO: RGB = [52, 62, 84]; // slate-blue

// Diverging endpoints: a warm red (negative), the shared slate-blue zero, a clear
// green (positive).
const NEG: RGB = [200, 64, 56]; // red
const MID: RGB = ZERO;
const POS: RGB = [70, 160, 75]; // green

/**
 * Two-slope diverging color over the actual range [lo, hi] (lo ≤ 0 ≤ hi in the
 * usual case). v=0 → ZERO; v=hi → green; v=lo → red; each side scaled on its own
 * so the scale covers exactly the data found. Values beyond [lo, hi] clamp.
 */
export function diverging(v: number, lo: number, hi: number): RGB {
  if (v >= 0) return lerpRGB(MID, POS, hi > 0 ? clamp01(v / hi) : 0);
  return lerpRGB(MID, NEG, lo < 0 ? clamp01(v / lo) : 0);
}

// Sequential ramp control points (dark navy → teal → yellow), low→high.
const SEQ: RGB[] = [
  ZERO,
  [38, 78, 120],
  [40, 140, 150],
  [120, 196, 120],
  [240, 220, 90],
];

/**
 * Map a value in [min,max] to a sequential color (clamped). Good for
 * probabilities (min=0,max=1) or any non-negative magnitude.
 */
export function sequential(v: number, min: number, max: number): RGB {
  const span = max - min;
  const t = span > 0 ? clamp01((v - min) / span) : 0;
  const scaled = t * (SEQ.length - 1);
  const i = Math.min(SEQ.length - 2, Math.floor(scaled));
  return lerpRGB(SEQ[i], SEQ[i + 1], scaled - i);
}

export function rgbCss(c: RGB): string {
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

/** Symmetric extent of |v| over an array (for a diverging scale). */
export function maxAbsOf(data: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < data.length; i++) {
    const a = Math.abs(data[i]);
    if (a > m) m = a;
  }
  return m;
}

/** [min,max] of an array. */
export function rangeOf(data: ArrayLike<number>): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < data.length; i++) {
    const x = data[i];
    if (x < lo) lo = x;
    if (x > hi) hi = x;
  }
  if (lo === Infinity) return [0, 0];
  return [lo, hi];
}

/** Pick a sensible default colormap mode for a tensor by its value range. */
export function autoMode(data: ArrayLike<number>): 'diverging' | 'sequential' {
  const [lo] = rangeOf(data);
  return lo < -1e-9 ? 'diverging' : 'sequential';
}

/**
 * Tidy a sequential [min, max] for display: cover the actual data, but snap an
 * endpoint to 0 or 1 when it's essentially there (clean legends), and give a
 * constant field a usable span (anchored at 0) instead of a degenerate one.
 */
export function niceSeq(min: number, max: number): [number, number] {
  let lo = min;
  let hi = max;
  if (Math.abs(lo) < 5e-3) lo = 0;
  if (Math.abs(hi - 1) < 5e-3) hi = 1;
  if (hi - lo < 1e-9) {
    lo = Math.min(0, lo);
    if (hi <= lo) hi = lo + 1e-9;
  }
  return [lo, hi];
}

/** mode + actual [lo, hi] color range for a tensor (diverging exact, sequential tidied). */
export function colorScaleOf(data: ArrayLike<number>): { mode: 'diverging' | 'sequential'; range: [number, number] } {
  const [min, max] = rangeOf(data);
  if (min < -1e-9) return { mode: 'diverging', range: [min, max] };
  return { mode: 'sequential', range: niceSeq(min, max) };
}

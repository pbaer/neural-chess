// High-accuracy error function for the exact-erf GELU (NOT the tanh approx).
//
// Implementation: the Chebyshev approximation to erfc from Numerical Recipes
// (3rd ed., `erfccheb`), accurate to better than ~1e-12 across the whole real
// line — far inside the engine's 1e-4 per-op parity budget. We verify it in
// erf.parity against Python's C-library `math.erf` over a dense grid.
//
// GELU uses erf(x/sqrt(2)); the worst-case argument magnitudes after LayerNorm
// + a linear are modest, so accuracy here is never the binding constraint.

const COF = [
  -1.3026537197817094, 6.4196979235649026e-1, 1.9476473204185836e-2,
  -9.561514786808631e-3, -9.46595344482036e-4, 3.66839497852761e-4,
  4.2523324806907e-5, -2.0278578112534e-5, -1.624290004647e-6,
  1.303655835580e-6, 1.5626441722e-8, -8.5238095915e-8,
  6.529054439e-9, 5.059343495e-9, -9.91364156e-10,
  -2.27365122e-10, 9.6467911e-11, 2.394038e-12,
  -6.886027e-12, 8.94487e-13, 3.13092e-13,
  -1.12708e-13, 3.81e-16, 7.106e-15,
];

/** Complementary error function for z >= 0, via Chebyshev fit. */
function erfccheb(z: number): number {
  let d = 0;
  let dd = 0;
  const t = 2 / (2 + z);
  const ty = 4 * t - 2;
  for (let j = COF.length - 1; j > 0; j--) {
    const tmp = d;
    d = ty * d - dd + COF[j];
    dd = tmp;
  }
  return t * Math.exp(-z * z + 0.5 * (COF[0] + ty * d) - dd);
}

/** erfc(x) for all real x. */
export function erfc(x: number): number {
  return x >= 0 ? erfccheb(x) : 2 - erfccheb(-x);
}

/** erf(x) for all real x. */
export function erf(x: number): number {
  return x >= 0 ? 1 - erfccheb(x) : erfccheb(-x) - 1;
}

// Stockfish-anchored playing-strength estimate for the hero model, as a function
// of the user's settings: one-shot vs MCTS, the MCTS max-sims slider, AND the
// Move-variety slider. Numbers are empirical: the GPU model (the browser model's
// twin) played ~326 games vs Stockfish 18 (UCI_LimitStrength at calibrated Elo
// rungs), with move selection going through the SAME value-adaptive selection
// logic the browser uses, so "variety" here means exactly what it means in the
// app. Its Elo was fit by maximum likelihood per config (eval/v3/elo_calibrate.py
// -> eval/v3/elo_calibration.json).
//
// Model: estimate = base(mode, sims) - varietyPenalty(mode, variety).
//   - base: variety=0 (strongest play). MCTS interpolates in log(sims) between
//     anchor sim-counts, where strength-vs-sims is close to linear.
//   - varietyPenalty: Elo given up by loosening play; 0 at variety=0, growing
//     toward variety=1. The value-adaptive selection only loosens when winning
//     and never leaves the model's reasonable set, so the penalty is bounded —
//     and much smaller in MCTS (where a Q-margin floor also gates the set) than
//     in one-shot.
//
// Notes on the measured data baked in below:
//   - MCTS at only 10 sims (~1310) is actually WEAKER than one-shot (~1572):
//     shallow noisy search underperforms the raw policy. Strength then climbs to
//     ~2474 at 300 sims. (This is a real effect — the CIs separate.)
//   - The 150/200 base points inverted within noise (overlapping CIs); pooled to
//     a flat ~2300 so the displayed Elo never dips as you add simulations.
//   - The MCTS variety penalty is small and noisy (per-config CIs ~±150); modeled
//     as a smoothed monotone curve rather than the raw per-config values.
//
// The number is anchored to Stockfish 18's own Elo scale, which only roughly
// tracks FIDE / online ratings (and its UCI_Elo floor is 1320).

export interface EloEstimate {
  /** Point estimate (Stockfish-18-anchored Elo). */
  elo: number;
  /** 95% confidence interval (not currently shown in the UI, but kept faithful). */
  lo: number;
  hi: number;
}

interface Anchor extends EloEstimate {
  sims: number;
}

interface PenaltyPoint {
  /** Move-variety setting in [0,1]. */
  v: number;
  /** Elo given up vs variety=0 at this setting. */
  p: number;
}

// ── Base strength curve (variety=0), measured vs Stockfish 18. ──
export const ONESHOT_ELO: EloEstimate = { elo: 1572, lo: 1446, hi: 1698 };

export const MCTS_ANCHORS: Anchor[] = [
  { sims: 10, elo: 1310, lo: 1166, hi: 1444 },
  { sims: 25, elo: 1692, lo: 1566, hi: 1822 },
  { sims: 50, elo: 1912, lo: 1774, hi: 2056 },
  { sims: 100, elo: 2132, lo: 1986, hi: 2294 },
  { sims: 150, elo: 2300, lo: 2170, hi: 2470 }, // 150/200 pooled (inverted within noise)
  { sims: 200, elo: 2300, lo: 2150, hi: 2460 },
  { sims: 300, elo: 2474, lo: 2312, hi: 2650 },
];

// ── Variety penalty (Elo below variety=0), interpolated in `v`. p(0)=0. ──
// One-shot: measured -180 @ v0.5, -304 @ v1.0 (large — no Q-filter on its set).
export const ONESHOT_PENALTY: PenaltyPoint[] = [
  { v: 0, p: 0 }, { v: 0.5, p: 180 }, { v: 1.0, p: 304 },
];
// MCTS: small + noisy (mcts-100: -106/+50; mcts-300: +126/+250, wide CIs).
// Smoothed monotone, weighting the tighter measurements; far below one-shot.
export const MCTS_PENALTY: PenaltyPoint[] = [
  { v: 0, p: 0 }, { v: 0.5, p: 30 }, { v: 1.0, p: 100 },
];

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

/** Linear-interpolate the variety penalty (Elo) at setting `variety`. */
function penaltyAt(curve: PenaltyPoint[], variety: number): number {
  const v = Math.min(Math.max(variety, 0), 1);
  if (v <= curve[0].v) return curve[0].p;
  for (let i = 0; i < curve.length - 1; i++) {
    if (v <= curve[i + 1].v) {
      const a = curve[i];
      const b = curve[i + 1];
      const t = (v - a.v) / (b.v - a.v);
      return lerp(a.p, b.p, t);
    }
  }
  return curve[curve.length - 1].p;
}

/** Base (variety=0) Elo for the given mode + sims. */
function baseElo(mctsEnabled: boolean, sims: number): EloEstimate {
  if (!mctsEnabled) return ONESHOT_ELO;
  const a = MCTS_ANCHORS;
  if (sims <= a[0].sims) return strip(a[0]);
  if (sims >= a[a.length - 1].sims) return strip(a[a.length - 1]);
  for (let i = 0; i < a.length - 1; i++) {
    const lo = a[i];
    const hi = a[i + 1];
    if (sims <= hi.sims) {
      const t = (Math.log(sims) - Math.log(lo.sims)) / (Math.log(hi.sims) - Math.log(lo.sims));
      return {
        elo: Math.round(lerp(lo.elo, hi.elo, t)),
        lo: Math.round(lerp(lo.lo, hi.lo, t)),
        hi: Math.round(lerp(lo.hi, hi.hi, t)),
      };
    }
  }
  return strip(a[a.length - 1]);
}

/**
 * Estimated Stockfish-anchored Elo for the given settings.
 * `sims` is ignored when `mctsEnabled` is false. The variety penalty shifts the
 * whole interval down (loosening play lowers strength but keeps the same spread).
 */
export function estimateElo(mctsEnabled: boolean, sims: number, variety: number): EloEstimate {
  const base = baseElo(mctsEnabled, sims);
  const pen = Math.round(penaltyAt(mctsEnabled ? MCTS_PENALTY : ONESHOT_PENALTY, variety));
  return { elo: base.elo - pen, lo: base.lo - pen, hi: base.hi - pen };
}

function strip(a: Anchor): EloEstimate {
  return { elo: a.elo, lo: a.lo, hi: a.hi };
}

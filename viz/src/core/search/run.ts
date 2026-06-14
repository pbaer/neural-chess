// Search driver — runs the stepwise MCTS up to a MAXIMUM simulation budget, with
// an early cutoff once a single move is clearly best (to keep play snappy), then
// applies the shared value-adaptive selection to pick which searched move to play.
// Emits throttled progress snapshots and cooperatively yields so the host (the
// inference worker) stays responsive and can be cancelled. No wall-clock budget:
// the sim cap + cutoff govern the search, for predictability.
//
// Presentation-agnostic: time source, yield, progress, and cancel are injected.

import { Chess } from 'chess.js';
import { MCTS, type Evaluator, type MCTSConfig } from './mcts.ts';
import { selectMoveIndex } from './selection.ts';
import type { SearchOptions, SearchResult, SearchSnapshot } from './types.ts';

/** Always run at least this many sims before any early cutoff can trigger. */
export const MIN_SIMS = 10;
/** Default early-cutoff visit-fraction threshold (top move's share of visits). */
export const DEFAULT_CUTOFF_THRESHOLD = 0.7;

export interface RunSearchHooks {
  /** Throttled progress callback (live snapshots while searching). */
  onProgress?: (snap: SearchSnapshot) => void;
  /** Polled between simulations; returning true ends the search early. */
  shouldCancel?: () => boolean;
  /** Monotonic clock in ms (defaults to performance.now / Date.now). */
  now?: () => number;
  /** Cooperative yield to the host event loop (defaults to a macrotask). */
  yieldToHost?: () => Promise<void>;
}

const EMIT_MS = 90; // progress snapshot cadence (~11fps live updates)
const YIELD_MS = 60; // event-loop yield cadence (keeps cancel latency low, low overhead)

function defaultNow(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

function defaultYield(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * Run an MCTS search from `fen` and return the chosen move + final snapshot.
 * The model is consulted only through `evaluator` (priors P + value V).
 */
export async function runSearch(
  fen: string,
  evaluator: Evaluator,
  options: SearchOptions,
  hooks: RunSearchHooks = {},
): Promise<SearchResult> {
  const now = hooks.now ?? defaultNow;
  const yieldToHost = hooks.yieldToHost ?? defaultYield;
  const config: MCTSConfig = { cPuct: options.cPuct, initialRepCounts: options.repCounts };
  if (options.seed !== undefined) {
    config.rng = mulberry32(options.seed);
  }

  const chess = new Chess(fen);
  const mcts = new MCTS(chess, evaluator, config);
  mcts.init();

  const start = now();
  let lastEmit = start;
  let lastYield = start;
  const maxSims = Math.max(1, options.sims | 0);
  const minSims = Math.min(MIN_SIMS, maxSims);
  const cutoff = clamp(options.cutoffThreshold ?? DEFAULT_CUTOFF_THRESHOLD, 0.5, 1);

  // No legal moves (shouldn't happen — caller gates on game-over) → bail.
  if (mcts.root.children && mcts.root.children.size > 0) {
    for (let i = 0; i < maxSims; i++) {
      mcts.simulate();
      const simsDone = i + 1;
      const t = now();
      if (hooks.shouldCancel?.()) break;

      // Early cutoff: once we've run the minimum, stop if a single move is clearly
      // best — either its visit lead is already unbeatable with the sims remaining,
      // or its share of visits has crossed the threshold. Keeps obvious recaptures
      // snappy while leaving close positions to the full search (then Part-1 variety).
      if (simsDone >= minSims && simsDone < maxSims) {
        const { topN, secondN } = mcts.topTwoChildVisits();
        const remaining = maxSims - simsDone;
        if (topN - secondN > remaining) break; // 2nd can no longer overtake
        if (topN / simsDone >= cutoff) break; // dominant move
      }

      if (hooks.onProgress && t - lastEmit >= EMIT_MS) {
        hooks.onProgress(mcts.snapshot(maxSims, t - start, true));
        lastEmit = t;
      }
      if (t - lastYield >= YIELD_MS) {
        await yieldToHost();
        lastYield = now();
      }
    }
  }

  const elapsed = now() - start;
  const snapshot = mcts.snapshot(maxSims, elapsed, false);
  // Value-adaptive final selection over the searched root moves: D = visit counts,
  // V = backed-up root Q, plus each child's Q so a losing alternative can't be
  // sampled. variety=0 ⇒ the most-visited (strongest) move, deterministically.
  const rows = mcts.rootChildren();
  const rng = config.rng ?? Math.random;
  const pick = selectMoveIndex(
    rows.map((r) => ({ weight: r.n, q: r.q })),
    { value: snapshot.rootEval, variety: options.variety ?? 0, rng },
  );
  const chosen = pick >= 0 ? rows[pick] : null;
  hooks.onProgress?.(snapshot);
  return {
    move: chosen ? { from: chosen.from, to: chosen.to, promotion: chosen.promotion, uci: chosen.uci, san: chosen.san, fromIdx: chosen.fromIdx, toIdx: chosen.toIdx } : null,
    value: snapshot.rootEval,
    prior: chosen?.p ?? 0,
    snapshot,
  };
}

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

/** Mulberry32 — deterministic PRNG (kept local to avoid leaking into the tree API). */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

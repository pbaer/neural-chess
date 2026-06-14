// Search driver — runs the stepwise MCTS to completion under a sims + wall-clock
// budget, emitting throttled progress snapshots and cooperatively yielding so the
// host (the inference worker) stays responsive and can be cancelled.
//
// Presentation-agnostic: time source, yield, progress, and cancel are injected.

import { Chess } from 'chess.js';
import { MCTS, type Evaluator, type MCTSConfig } from './mcts.ts';
import type { SearchOptions, SearchResult, SearchSnapshot } from './types.ts';

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
  const sims = Math.max(1, options.sims | 0);
  const timeMs = Math.max(1, options.timeMs | 0);

  // No legal moves (shouldn't happen — caller gates on game-over) → bail.
  if (mcts.root.children && mcts.root.children.size > 0) {
    for (let i = 0; i < sims; i++) {
      mcts.simulate();
      const t = now();
      if (t - start >= timeMs) break;
      if (hooks.shouldCancel?.()) break;
      if (hooks.onProgress && t - lastEmit >= EMIT_MS) {
        hooks.onProgress(mcts.snapshot(sims, t - start, true));
        lastEmit = t;
      }
      if (t - lastYield >= YIELD_MS) {
        await yieldToHost();
        lastYield = now();
      }
    }
  }

  const elapsed = now() - start;
  const snapshot = mcts.snapshot(sims, elapsed, false);
  const best = mcts.bestMove(options.temperature ?? 0);
  hooks.onProgress?.(snapshot);
  return {
    move: best?.move ?? null,
    value: snapshot.rootEval,
    prior: best?.prior ?? 0,
    snapshot,
  };
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

// Search (MCTS) shared types — presentation-agnostic.
//
// These describe the configuration, the live/final visualization snapshot, and
// the final move the search picks. They cross the worker boundary (Comlink) and
// drive the desktop search visualization, so they are plain data only.

import type { PromotionPiece } from '../game/gameStore.ts';

/** A fully-specified move in chess.js coordinates (+ engine square indices for drawing). */
export interface MoveLite {
  /** Algebraic from-square, e.g. 'g1'. */
  from: string;
  /** Algebraic to-square, e.g. 'f3'. */
  to: string;
  promotion?: PromotionPiece;
  /** UCI / long-algebraic, e.g. 'g1f3' (+ promotion letter). */
  uci: string;
  san: string;
  /** python-chess square index (a1=0..h8=63) for board rendering. */
  fromIdx: number;
  toIdx: number;
}

/** Per-root-move search statistics — the row data for the PUCT stats table. */
export interface RootChildStat extends MoveLite {
  /** Visit count N(s,a). */
  n: number;
  /** Mean action value Q(s,a) from the ROOT mover's perspective, in [-1,1]. */
  q: number;
  /** Policy prior P(s,a) (legal-masked, renormalized). */
  p: number;
  /** PUCT score = q + c_puct·P·sqrt(ΣN_b)/(1+N) at the current visit counts. */
  puct: number;
}

/** A live (or final) snapshot of the search, for visualization. */
export interface SearchSnapshot {
  /** Simulations completed so far (== root visit count). */
  simsDone: number;
  /** Simulation budget (for a progress indicator). */
  totalSims: number;
  /** Wall-clock elapsed, ms. */
  elapsedMs: number;
  /** Backed-up root evaluation (root.Q), side-to-move perspective, [-1,1]. */
  rootEval: number;
  /** UCI of the current most-visited root move (the would-be choice), or null. */
  bestUci: string | null;
  /** Principal variation (most-visited path) as a move list. */
  pv: Array<{ uci: string; san: string }>;
  /** Root child stats, sorted by visit count desc (capped to a readable count). */
  children: RootChildStat[];
  /** True while the search is still running. */
  running: boolean;
}

/** Configuration for a single search. */
export interface SearchOptions {
  /** MAXIMUM number of simulations (the search may stop earlier — see early cutoff). */
  sims: number;
  /** PUCT exploration constant (project default 1.5). */
  cPuct: number;
  /**
   * Move-variety setting S in [0,1] for the FINAL, value-adaptive move selection
   * over the searched root moves. 0 = always the most-visited (strongest) move.
   * The search itself is unchanged; this only governs which searched move is played.
   */
  variety: number;
  /**
   * Early-cutoff threshold in (0,1]: stop once the top move's visit fraction
   * reaches this (default 0.7). The search also stops once the visit lead is
   * unbeatable. At least MIN_SIMS simulations always run first.
   */
  cutoffThreshold?: number;
  /** Optional RNG seed (deterministic selection sampling / tie-breaks). */
  seed?: number;
  /**
   * Position repetition counts (repKey -> count) carried from the real game so
   * featurization's repetition plane and threefold-draw detection are faithful.
   */
  repCounts?: Record<string, number>;
}

/** The result of a completed (or cancelled) search. */
export interface SearchResult {
  /** The chosen move, or null if the position had no legal move. */
  move: MoveLite | null;
  /** Backed-up root evaluation, side-to-move perspective, [-1,1]. */
  value: number;
  /** Policy prior of the chosen move (for the move readout). */
  prior: number;
  /** Final visualization snapshot. */
  snapshot: SearchSnapshot;
}

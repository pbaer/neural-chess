// AlphaZero-style PUCT Monte Carlo Tree Search — a faithful TypeScript mirror of
// src/mcts.py (the project's canonical engine).
//
// PRINCIPLE-3 CARVE-OUT: the search uses ONLY the model's own policy priors P
// and value estimate V. No hand-coded chess heuristics (no material, piece-square
// tables, mobility, pruning, opening book) ever enter the tree. The only chess
// "knowledge" is the rules of the game (legal moves + terminal detection), which
// are mechanics, not strategy. The tree just lets the model think harder with its
// own judgment.
//
// Design (mirrors mcts.py):
//  - Sequential MCTS: one model eval per simulation at the expanded leaf.
//  - No tree reuse across moves (rebuilt each move) — simple + correct.
//  - Final move = most-visited root child (temperature 0), or sampled by
//    N^(1/temp) for temperature > 0.
//  - Make/unmake on a single chess.js board (instead of copying) so chess.js's
//    own move stack is available and repetition counts stay exact.

import { Chess, type Move as VerboseMove } from 'chess.js';
import { algToIdx } from '../game/chessAdapter.ts';
import type { MoveLite, RootChildStat, SearchSnapshot } from './types.ts';
import type { PromotionPiece } from '../game/gameStore.ts';

/** Leaf evaluation supplied by the model (priors over legal moves + value). */
export interface LeafEvaluation {
  /** Legal-masked, renormalized priors over this node's legal moves. */
  priors: Array<{ move: VerboseMove; prior: number }>;
  /** Value in [-1,1] from the side-to-move's perspective. */
  value: number;
}

/** Context the evaluator may consult (e.g. featurization's repetition plane). */
export interface SearchContext {
  isRepetition(count: number): boolean;
}

/** Evaluate a leaf board → (priors, value). Implemented over the model (worker). */
export type Evaluator = (chess: Chess, ctx: SearchContext) => LeafEvaluation;

/**
 * PUCT score for a child: Q(s,a) + c_puct·P(s,a)·sqrt(ΣN_b)/(1+N(s,a)).
 * `q` and `parentN` are pre-computed by the caller (q already from the PARENT
 * mover's perspective, i.e. negated child mean value). Exported for testing.
 */
export function puctScore(q: number, p: number, parentN: number, childN: number, cPuct: number): number {
  return q + cPuct * p * Math.sqrt(parentN) / (1 + childN);
}

class Node {
  prior: number;
  /** Move that reaches this node from its parent (null at the root). */
  move: VerboseMove | null;
  children: Map<string, Node> | null = null; // keyed by uci/lan; null until expanded
  N = 0; // visit count
  W = 0; // total action value, from THIS node's mover's perspective
  isExpanded = false;
  terminalValue: number | null = null; // set if this node is game-over

  constructor(prior: number, move: VerboseMove | null) {
    this.prior = prior;
    this.move = move;
  }

  get Q(): number {
    return this.N > 0 ? this.W / this.N : 0;
  }
}

/** Sample a Gamma(alpha,1) variate (Marsaglia–Tsang) — for Dirichlet root noise. */
function sampleGamma(alpha: number, rng: () => number): number {
  if (alpha < 1) {
    // Boost: Gamma(a) = Gamma(a+1) * U^(1/a).
    return sampleGamma(alpha + 1, rng) * Math.pow(rng() || 1e-12, 1 / alpha);
  }
  const d = alpha - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x: number;
    let v: number;
    do {
      // Box–Muller normal.
      const u1 = rng() || 1e-12;
      const u2 = rng();
      x = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = rng() || 1e-12;
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

export interface MCTSConfig {
  cPuct?: number;
  dirichletFrac?: number;
  dirichletAlpha?: number;
  /** Repetition counts (repKey -> count) carried from the real game. */
  initialRepCounts?: Record<string, number>;
  rng?: () => number;
}

/**
 * Stepwise PUCT MCTS. The caller drives it one simulation at a time (so the
 * worker can enforce a wall-clock budget, emit progress, and yield), then reads
 * out the visit statistics / PV / best move.
 *
 * The supplied chess.js board is OWNED by the search: each simulate() pushes
 * moves down the selected path and pops them all back, leaving the board exactly
 * at the root. Do not mutate it externally while a search is in progress.
 */
export class MCTS {
  readonly root = new Node(1.0, null);
  rootValue = 0;
  private readonly chess: Chess;
  private readonly evaluate: Evaluator;
  private readonly cPuct: number;
  private readonly dirichletFrac: number;
  private readonly dirichletAlpha: number;
  private readonly rng: () => number;
  private readonly repCounts: Map<string, number>;

  constructor(chess: Chess, evaluator: Evaluator, config: MCTSConfig = {}) {
    this.chess = chess;
    this.evaluate = evaluator;
    this.cPuct = config.cPuct ?? 1.5;
    this.dirichletFrac = config.dirichletFrac ?? 0;
    this.dirichletAlpha = config.dirichletAlpha ?? 0.3;
    this.rng = config.rng ?? Math.random;
    this.repCounts = new Map(Object.entries(config.initialRepCounts ?? {}));
  }

  /** Position key (FEN piece/side/castling/ep fields) — matches the game store. */
  private repKey(): string {
    return this.chess.fen().split(' ').slice(0, 4).join(' ');
  }

  private isRepetition(count: number): boolean {
    return (this.repCounts.get(this.repKey()) ?? 0) >= count;
  }

  /** Side-to-move value if game over (rules only, no heuristics), else null. */
  private terminalValue(): number | null {
    if (this.chess.isCheckmate()) return -1; // side to move was mated → lost
    if (this.chess.isStalemate()) return 0;
    if (this.chess.isInsufficientMaterial()) return 0;
    // Threefold (and fivefold) via the carried repetition counts.
    if (this.isRepetition(3)) return 0;
    // Fifty-move / seventy-five-move and any other chess.js-recognized draw.
    if (this.chess.isDraw()) return 0;
    return null;
  }

  /** Expand a leaf: attach children from priors, or mark terminal. Returns its value. */
  private expand(node: Node): number {
    const tv = this.terminalValue();
    if (tv !== null) {
      node.terminalValue = tv;
      node.isExpanded = true;
      return tv;
    }
    const { priors, value } = this.evaluate(this.chess, { isRepetition: (n) => this.isRepetition(n) });
    const children = new Map<string, Node>();
    for (const { move, prior } of priors) children.set(move.lan, new Node(prior, move));
    node.children = children;
    node.isExpanded = true;
    return value;
  }

  /** PUCT: pick the child maximizing q + c·P·sqrt(ΣN)/(1+N). Returns [uci, child]. */
  private puctSelect(node: Node): [string, Node] {
    let bestScore = -Infinity;
    let best: [string, Node] | null = null;
    for (const [uci, child] of node.children!) {
      // child.Q is from the child's mover's view; negate for THIS node's mover.
      const q = child.N > 0 ? -child.Q : 0;
      const score = puctScore(q, child.prior, node.N, child.N, this.cPuct);
      if (score > bestScore) {
        bestScore = score;
        best = [uci, child];
      }
    }
    return best!;
  }

  /** Expand the root and (optionally) mix in Dirichlet exploration noise. */
  init(): void {
    this.rootValue = this.expand(this.root);
    if (this.dirichletFrac > 0 && this.root.children && this.root.children.size > 0) {
      const kids = [...this.root.children.values()];
      const gammas = kids.map(() => sampleGamma(this.dirichletAlpha, this.rng));
      const total = gammas.reduce((s, g) => s + g, 0) || 1;
      kids.forEach((c, i) => {
        const noise = gammas[i] / total;
        c.prior = (1 - this.dirichletFrac) * c.prior + this.dirichletFrac * noise;
      });
    }
  }

  /** Run exactly one simulation (selection → expansion/eval → backup), make/unmake. */
  simulate(): void {
    let node = this.root;
    const path: Node[] = [node];
    let pushed = 0;

    // Selection: descend until an unexpanded / terminal / childless node.
    while (node.isExpanded && node.terminalValue === null && node.children && node.children.size > 0) {
      const [, child] = this.puctSelect(node);
      const mv = child.move!;
      this.chess.move({ from: mv.from, to: mv.to, promotion: mv.promotion });
      this.bumpRep(+1);
      pushed++;
      node = child;
      path.push(node);
    }

    // Expansion + evaluation (unless already terminal).
    const leafValue = node.terminalValue !== null ? node.terminalValue : this.expand(node);

    // Backup: flip sign each ply back up the path.
    let v = leafValue;
    for (let i = path.length - 1; i >= 0; i--) {
      path[i].N += 1;
      path[i].W += v;
      v = -v;
    }

    // Unmake back to the root.
    for (let i = 0; i < pushed; i++) {
      this.bumpRep(-1);
      this.chess.undo();
    }
  }

  private bumpRep(delta: number): void {
    const k = this.repKey();
    const next = (this.repCounts.get(k) ?? 0) + delta;
    if (next <= 0) this.repCounts.delete(k);
    else this.repCounts.set(k, next);
  }

  /** Simulations completed so far (== root visit count). */
  get simCount(): number {
    return this.root.N;
  }

  /** Top two root-child visit counts (single pass) — for the early-cutoff test. */
  topTwoChildVisits(): { topN: number; secondN: number } {
    let topN = 0;
    let secondN = 0;
    if (this.root.children) {
      for (const c of this.root.children.values()) {
        if (c.N > topN) {
          secondN = topN;
          topN = c.N;
        } else if (c.N > secondN) {
          secondN = c.N;
        }
      }
    }
    return { topN, secondN };
  }

  /** Root child stats sorted by visit count (desc), for the PUCT table / arrows. */
  rootChildren(cap = Infinity): RootChildStat[] {
    if (!this.root.children) return [];
    const rows: RootChildStat[] = [];
    for (const child of this.root.children.values()) {
      const mv = child.move!;
      const q = child.N > 0 ? -child.Q : 0; // root mover's perspective
      const p = child.prior;
      const puct = puctScore(q, p, this.root.N, child.N, this.cPuct);
      rows.push({ ...moveLite(mv), n: child.N, q, p, puct });
    }
    rows.sort((a, b) => b.n - a.n || b.p - a.p);
    return Number.isFinite(cap) ? rows.slice(0, cap) : rows;
  }

  /** Principal variation: the most-visited path from the root. */
  pv(maxLen = 12): Array<{ uci: string; san: string }> {
    const line: Array<{ uci: string; san: string }> = [];
    let node: Node | null = this.root;
    while (node && node.children && node.children.size > 0 && line.length < maxLen) {
      let best: Node | null = null;
      for (const c of node.children.values()) {
        if (!best || c.N > best.N || (c.N === best.N && c.prior > best.prior)) best = c;
      }
      if (!best || best.N === 0 || !best.move) break;
      line.push({ uci: best.move.lan, san: best.move.san });
      node = best;
    }
    return line;
  }

  /** The chosen move + its prior. temperature 0 = most-visited; >0 samples by N^(1/T). */
  bestMove(temperature = 0): { move: MoveLite; prior: number } | null {
    if (!this.root.children || this.root.children.size === 0) return null;
    const kids = [...this.root.children.values()];
    let chosen: Node;
    if (temperature > 0.01) {
      const weights = kids.map((c) => Math.pow(c.N, 1 / temperature));
      const total = weights.reduce((s, w) => s + w, 0);
      let r = this.rng() * (total || 1);
      chosen = kids[kids.length - 1];
      for (let i = 0; i < kids.length; i++) {
        r -= weights[i];
        if (r <= 0) {
          chosen = kids[i];
          break;
        }
      }
    } else {
      chosen = kids[0];
      for (const c of kids) if (c.N > chosen.N || (c.N === chosen.N && c.prior > chosen.prior)) chosen = c;
    }
    return { move: moveLite(chosen.move!), prior: chosen.prior };
  }

  /** Build a full visualization snapshot at the current search state. */
  snapshot(totalSims: number, elapsedMs: number, running: boolean, childCap = 12): SearchSnapshot {
    const children = this.rootChildren(childCap);
    return {
      simsDone: this.simCount,
      totalSims,
      elapsedMs,
      rootEval: this.root.Q,
      bestUci: children.length > 0 ? children[0].uci : null,
      pv: this.pv(),
      children,
      running,
    };
  }
}

/** chess.js verbose move → MoveLite (adds engine square indices + uci). */
function moveLite(mv: VerboseMove): MoveLite {
  return {
    from: mv.from,
    to: mv.to,
    promotion: mv.promotion as PromotionPiece | undefined,
    uci: mv.lan,
    san: mv.san,
    fromIdx: algToIdx(mv.from),
    toIdx: algToIdx(mv.to),
  };
}

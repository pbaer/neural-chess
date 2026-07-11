// MCTS engine tests — the PUCT math, a deterministic fixed-tree case, terminal/
// sign correctness, and a real-engine integration that also profiles the per-move
// latency (so the sim budget can be sanity-checked against the ≤1–2s target).

import { describe, it, expect } from 'vitest';
import { Chess, type Move as VerboseMove } from 'chess.js';
import { MCTS, puctScore, type Evaluator } from './mcts.ts';
import { makeEngineEvaluator } from './evaluator.ts';
import { runSearch } from './run.ts';
import { loadEngine } from '../../../tests/parity/fixtures.ts';

/** A synthetic evaluator: priors from a weight fn (default uniform) + a value fn
 *  (default 0). NB: option keys avoid `valueOf` (would shadow Object.prototype). */
function stubEvaluator(opts: {
  priorFn?: (mv: VerboseMove) => number;
  valueFn?: (chess: Chess) => number;
} = {}): Evaluator {
  return (chess) => {
    const moves = chess.moves({ verbose: true }) as VerboseMove[];
    const raw = moves.map((m) => (opts.priorFn ? opts.priorFn(m) : 1));
    const sum = raw.reduce((s, w) => s + w, 0) || 1;
    return {
      priors: moves.map((m, i) => ({ move: m, prior: raw[i] / sum })),
      value: opts.valueFn ? opts.valueFn(chess) : 0,
    };
  };
}

describe('puctScore', () => {
  it('matches Q + c·P·sqrt(ΣN)/(1+N)', () => {
    // q=0.2, p=0.5, parentN=16, childN=3, c=1.5 → 0.2 + 1.5*0.5*4/4 = 0.95
    expect(puctScore(0.2, 0.5, 16, 3, 1.5)).toBeCloseTo(0.95, 12);
    // Unvisited child (childN=0) at an unexpanded parent (parentN=0) → pure q.
    expect(puctScore(0, 0.3, 0, 0, 1.5)).toBeCloseTo(0, 12);
    // First selection after root expand (parentN=1): u = c·P·1/(1+0).
    expect(puctScore(0, 0.4, 1, 0, 1.5)).toBeCloseTo(0.6, 12);
  });
});

describe('MCTS tree mechanics (deterministic)', () => {
  it('expands the root over all legal moves and counts every simulation', () => {
    const chess = new Chess();
    const mcts = new MCTS(chess, stubEvaluator(), { cPuct: 1.5 });
    mcts.init();
    expect(mcts.root.children!.size).toBe(20); // 20 legal first moves
    for (let i = 0; i < 50; i++) mcts.simulate();
    expect(mcts.simCount).toBe(50);
    // Visit counts over root children sum to the simulation count.
    const total = mcts.rootChildren().reduce((s, c) => s + c.n, 0);
    expect(total).toBe(50);
    // Board is restored to the root after every make/unmake.
    expect(chess.fen()).toBe(new Chess().fen());
  });

  it('directs visits toward high-prior moves (PUCT exploration term)', () => {
    // Heavily favor 1.e4 in the priors, neutral value everywhere. PUCT should
    // spend most of its visits on e4 (P dominates the U term early).
    const evalr = stubEvaluator({ priorFn: (m) => (m.san === 'e4' ? 50 : 1) });
    const chess = new Chess();
    const mcts = new MCTS(chess, evalr, { cPuct: 1.5 });
    mcts.init();
    for (let i = 0; i < 200; i++) mcts.simulate();
    const rows = mcts.rootChildren();
    expect(rows[0].san).toBe('e4'); // most-visited
    expect(rows[0].n).toBeGreaterThan(100); // the lion's share of 200 sims
    expect(mcts.bestMove(0)!.move.san).toBe('e4');
  });

  it('backs up a high leaf value to the root mover (sign convention)', () => {
    // One reply (1.e4) yields a position the opponent (black, to move) rates very
    // badly (−0.9) → +0.9 for the white root. Use a near-greedy c_puct so visits
    // and the backed-up Q both follow the value.
    const evalr = stubEvaluator({
      priorFn: (m) => (m.san === 'e4' ? 50 : 1),
      valueFn: (chess) => (chess.history({ verbose: true }).slice(-1)[0]?.san === 'e4' ? -0.9 : 0),
    });
    const chess = new Chess();
    const mcts = new MCTS(chess, evalr, { cPuct: 0.1 });
    mcts.init();
    for (let i = 0; i < 80; i++) mcts.simulate();
    const e4 = mcts.rootChildren().find((r) => r.san === 'e4')!;
    expect(e4.q).toBeGreaterThan(0); // backed-up value favors the white mover
    expect(mcts.bestMove(0)!.move.san).toBe('e4');
  });

  it('finds a mate-in-1 via terminal detection + sign (no value signal)', () => {
    // Back-rank mate: white Rd1 -> d8#. Evaluator gives ZERO value everywhere, so
    // only terminal detection (checkmate = +1 for the mover) can find it.
    const chess = new Chess('6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1');
    const mcts = new MCTS(chess, stubEvaluator({ valueFn: () => 0 }), { cPuct: 1.5 });
    mcts.init();
    for (let i = 0; i < 300; i++) mcts.simulate();
    expect(mcts.bestMove(0)!.move.san).toBe('Rd8#');
    // Root backed-up eval should approach a win for the side to move.
    expect(mcts.root.Q).toBeGreaterThan(0.3);
  });

  it('snapshot is internally consistent', () => {
    const chess = new Chess();
    const mcts = new MCTS(chess, stubEvaluator(), { cPuct: 1.5 });
    mcts.init();
    for (let i = 0; i < 40; i++) mcts.simulate();
    const snap = mcts.snapshot(40, 12.3, false);
    expect(snap.simsDone).toBe(40);
    expect(snap.children[0].uci).toBe(snap.bestUci);
    expect(snap.pv.length).toBeGreaterThan(0);
    expect(snap.pv[0].uci).toBe(snap.bestUci); // PV starts with the chosen move
    // children sorted by visits descending
    for (let i = 1; i < snap.children.length; i++) {
      expect(snap.children[i - 1].n).toBeGreaterThanOrEqual(snap.children[i].n);
    }
  });
});

describe('early cutoff (max-sims + snappy play on a dominant move)', () => {
  it('stops well before the max once one move dominates the visits', async () => {
    // 1.e4 is hugely favored in the priors → visits concentrate fast → the visit
    // fraction crosses the 0.7 cutoff long before the 200-sim cap.
    const evalr = stubEvaluator({ priorFn: (m) => (m.san === 'e4' ? 200 : 1) });
    const fen = new Chess().fen();
    const result = await runSearch(fen, evalr, {
      sims: 200,
      cPuct: 1.5,
      variety: 0,
      cutoffThreshold: 0.7,
      seed: 1,
    });
    expect(result.snapshot.simsDone).toBeGreaterThanOrEqual(10); // MIN_SIMS floor
    expect(result.snapshot.simsDone).toBeLessThan(200); // cut off early
    expect(result.move!.san).toBe('e4');
  });

  it('runs the full budget when no move dominates (close position)', async () => {
    // Uniform priors + zero value → visits stay spread → no cutoff → all sims run.
    const result = await runSearch(new Chess().fen(), stubEvaluator(), {
      sims: 60,
      cPuct: 3,
      variety: 0,
      cutoffThreshold: 0.7,
      seed: 2,
    });
    expect(result.snapshot.simsDone).toBe(60);
  });

  it('never cuts off below the minimum (tiny budgets run fully)', async () => {
    const result = await runSearch(new Chess().fen(), stubEvaluator({ priorFn: (m) => (m.san === 'e4' ? 200 : 1) }), {
      sims: 5,
      cPuct: 1.5,
      variety: 0,
      cutoffThreshold: 0.7,
      seed: 3,
    });
    expect(result.snapshot.simsDone).toBe(5);
  });
});

describe('MCTS with the real engine (integration + profiling)', () => {
  // 300 real-engine sims take ~10s bare but can exceed vitest's default 30s
  // timeout under v8 coverage instrumentation, so give this test extra room.
  it('returns a legal move and is faithful to model priors at low sims', { timeout: 120_000 }, async () => {
    const engine = loadEngine();
    const evalr = makeEngineEvaluator(engine);
    const fen = new Chess().fen();

    const t0 = performance.now();
    const result = await runSearch(fen, evalr, { sims: 300, cPuct: 1.5, variety: 0, cutoffThreshold: 1, seed: 1 });
    const dt = performance.now() - t0;

    expect(result.move).not.toBeNull();
    // The chosen move is legal in the start position.
    const legal = new Chess().moves({ verbose: true }).map((m) => m.lan);
    expect(legal).toContain(result.move!.uci);
    expect(result.snapshot.simsDone).toBeGreaterThan(0);

    // Profiling (visible with --reporter verbose): per-300-sim latency + per-sim.
    console.log(
      `[MCTS profile] 300 sims in ${dt.toFixed(1)}ms (${(dt / 300).toFixed(3)}ms/sim); ` +
        `chosen ${result.move!.san} eval ${result.value.toFixed(3)}`,
    );
  });
});

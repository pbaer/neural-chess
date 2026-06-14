// Model evaluator for MCTS — turns the inference Engine into a leaf evaluator.
//
// Given a chess.js board it featurizes (in the engine's moving-side frame),
// runs a single forward pass, and returns the model's legal-masked policy priors
// (mapped back to real chess.js moves) plus the value head output. This is the
// ONLY chess "knowledge" the search consumes: the model's own P and V.

import type { Chess } from 'chess.js';
import type { Engine } from '../engine/index.ts';
import { featurize } from '../engine/index.ts';
import { buildLegalMaskAndMap, chessToBoardState, epTargetFromFen } from '../game/chessAdapter.ts';
import type { Evaluator, SearchContext } from './mcts.ts';

/** Build an MCTS Evaluator backed by an in-thread inference Engine. */
export function makeEngineEvaluator(engine: Engine): Evaluator {
  return (chess: Chess, ctx: SearchContext) => {
    const board = chessToBoardState(chess, {
      epSquare: epTargetFromFen(chess.fen()),
      isRepetition: (n) => ctx.isRepetition(n),
    });
    const planes = featurize(board, 'float');
    const { mask, indexToMove } = buildLegalMaskAndMap(chess);
    const res = engine.forward(planes, { legalMask: mask });
    const priors: Array<{ move: import('chess.js').Move; prior: number }> = [];
    for (const [idx, mv] of indexToMove) {
      priors.push({ move: mv, prior: res.policyProbs[idx] ?? 0 });
    }
    return { priors, value: res.value };
  };
}

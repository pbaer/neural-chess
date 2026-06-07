// Play-loop smoke test (pure Node, no Web Worker): exercises the M2 game wiring
// directly against the engine kernel —
//   chess.js position → featurize('float') → buildLegalMaskAndMap → forward
//     → bestLegalIndex → indexToMove → a LEGAL chess.js move.
// Also cross-checks that the index→move map agrees with the explicit
// decodeMove + sq^56 un-rotation path for a black-to-move position.

import { describe, it, expect } from 'vitest';
import { Chess } from 'chess.js';
import { featurize, decodeMove, rotateSquare } from '../../src/core/engine/index.ts';
import type { BoardState, PieceType } from '../../src/core/engine/index.ts';
import { algToIdx, buildLegalMaskAndMap, chessToBoardState } from '../../src/core/game/chessAdapter.ts';
import { loadEngine } from '../parity/fixtures.ts';

const engine = loadEngine();

function boardStateOf(chess: Chess): BoardState {
  return chessToBoardState(chess, { epSquare: null, isRepetition: () => false });
}

/** Pick the model's move for the side to move, returning the applied chess.js SAN. */
function modelReply(chess: Chess) {
  const planes = featurize(boardStateOf(chess), 'float');
  const { mask, indexToMove } = buildLegalMaskAndMap(chess);
  const res = engine.forward(planes, { legalMask: mask });
  const mv = indexToMove.get(res.bestLegalIndex);
  return { res, mv };
}

describe('play loop: model picks a legal move (white to move, startpos)', () => {
  const chess = new Chess();
  const { res, mv } = modelReply(chess);

  it('bestLegalIndex maps to a legal chess.js move', () => {
    expect(mv, 'index not in legal map').toBeDefined();
    const applied = chess.move({ from: mv!.from, to: mv!.to, promotion: mv!.promotion ?? 'q' });
    expect(applied.san).toBeTruthy();
  });

  it('value is a finite number in [-1,1]', () => {
    expect(Number.isFinite(res.value)).toBe(true);
    expect(res.value).toBeGreaterThanOrEqual(-1);
    expect(res.value).toBeLessThanOrEqual(1);
  });
});

describe('play loop: black to move (un-rotation)', () => {
  // 1.e4 → black to move.
  const chess = new Chess();
  chess.move('e4');
  expect(chess.turn()).toBe('b');

  const { res, mv } = modelReply(chess);

  it('produces a legal black move', () => {
    expect(mv).toBeDefined();
    const legalUcis = new Set(chess.moves({ verbose: true }).map((m) => m.from + m.to + (m.promotion ?? '')));
    expect(legalUcis.has(mv!.from + mv!.to + (mv!.promotion ?? ''))).toBe(true);
  });

  it('index→move map agrees with decodeMove + sq^56 un-rotation', () => {
    // Build a rotated board for decodeMove's promotion disambiguation: in the
    // engine frame, pieceTypeAt(s) is the real piece at s^56.
    const real = new Map<number, PieceType>();
    for (const row of chess.board()) {
      for (const cell of row) if (cell) real.set(algToIdx(cell.square), cell.type as PieceType);
    }
    const rotated: BoardState = {
      ...boardStateOf(chess),
      pieceTypeAt: (sq) => real.get(rotateSquare(sq)) ?? null,
    };
    const dec = decodeMove(res.bestLegalIndex, rotated); // engine frame
    const realFrom = rotateSquare(dec.from);
    const realTo = rotateSquare(dec.to);
    expect(realFrom).toBe(algToIdx(mv!.from));
    expect(realTo).toBe(algToIdx(mv!.to));
  });
});

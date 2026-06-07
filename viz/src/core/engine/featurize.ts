// Featurize a BoardState into the 21-plane (21·8·8) model input.
// Faithful port of src/v2/featurize.py (_featurize_white_to_move /
// _featurize_black_to_move) over the BoardState adapter.
//
// Output layout: Float32Array(21*64), plane-major, position = h*8 + w (so
// index = plane*64 + h*8 + w). Black-to-move applies a VERTICAL mirror
// (h = 7-rank, file unchanged; castling/ep mirrored) — the moving side always
// sits in planes 0-5 at the bottom. This is a coordinate change, not chess
// knowledge.
//
// int8-truncation toggle:
//   'float'   = matches src/v3/inference.py (no truncation) — canonical for play.
//   'trained' = mirrors the int8 storage the model trained on (Math.trunc toward
//               zero == int8 cast for the in-range values here): kills plane-17
//               fractions, 0.5→0 on plane-20, fullmove/100 → {0,1,2}.

import type { BoardState, Color, PieceType } from './boardState.ts';

export const NUM_PLANES = 21;
export type FeaturizeMode = 'float' | 'trained';

// P,R,N,B,Q,K → 0..5 (matches _WHITE_PIECE_PLANE order in featurize.py).
const PLANE_OF: Record<PieceType, number> = { p: 0, r: 1, n: 2, b: 3, q: 4, k: 5 };

function idx(plane: number, h: number, w: number): number {
  return plane * 64 + h * 8 + w;
}

function featurizeWhiteToMove(b: BoardState): Float32Array {
  const x = new Float32Array(NUM_PLANES * 64);
  // Pieces: white = own (0-5), black = opp (6-11).
  for (const pc of b.pieces()) {
    const h = pc.square >> 3;
    const w = pc.square & 7;
    const base = pc.color === 'w' ? 0 : 6;
    x[idx(base + PLANE_OF[pc.type], h, w)] = 1;
  }
  // Castling 12-15: own(white) K/Q, opp(black) K/Q.
  if (b.hasKingsideCastlingRights('w')) fill(x, 12);
  if (b.hasQueensideCastlingRights('w')) fill(x, 13);
  if (b.hasKingsideCastlingRights('b')) fill(x, 14);
  if (b.hasQueensideCastlingRights('b')) fill(x, 15);
  // EP target (16).
  if (b.epSquare != null) x[idx(16, b.epSquare >> 3, b.epSquare & 7)] = 1;
  // Counters / side / repetition.
  fillVal(x, 17, Math.min(b.halfmoveClock / 100, 1.0));
  fillVal(x, 18, Math.min(b.fullmoveNumber / 100, 2.0));
  fillVal(x, 19, 1.0);
  fillVal(x, 20, repetitionValue(b));
  return x;
}

function featurizeBlackToMove(b: BoardState): Float32Array {
  const x = new Float32Array(NUM_PLANES * 64);
  // Pieces in the rotated frame: black = own (0-5), white = opp (6-11); rank flipped.
  for (const pc of b.pieces()) {
    const h = 7 - (pc.square >> 3);
    const w = pc.square & 7; // file unchanged
    const base = pc.color === 'b' ? 0 : 6;
    x[idx(base + PLANE_OF[pc.type], h, w)] = 1;
  }
  // Castling: black rights → own, white rights → opp.
  if (b.hasKingsideCastlingRights('b')) fill(x, 12);
  if (b.hasQueensideCastlingRights('b')) fill(x, 13);
  if (b.hasKingsideCastlingRights('w')) fill(x, 14);
  if (b.hasQueensideCastlingRights('w')) fill(x, 15);
  // EP target: vertical mirror.
  if (b.epSquare != null) x[idx(16, 7 - (b.epSquare >> 3), b.epSquare & 7)] = 1;
  fillVal(x, 17, Math.min(b.halfmoveClock / 100, 1.0));
  fillVal(x, 18, Math.min(b.fullmoveNumber / 100, 2.0));
  fillVal(x, 19, 1.0);
  fillVal(x, 20, repetitionValue(b));
  return x;
}

function repetitionValue(b: BoardState): number {
  if (b.isRepetition(3)) return 1.0;
  if (b.isRepetition(2)) return 0.5;
  return 0.0;
}

function fill(x: Float32Array, plane: number): void {
  x.fill(1.0, plane * 64, plane * 64 + 64);
}
function fillVal(x: Float32Array, plane: number, v: number): void {
  if (v !== 0) x.fill(v, plane * 64, plane * 64 + 64);
}

/**
 * Encode a BoardState into the (21·8·8) plane tensor, applying rotation for
 * black-to-move. `mode` defaults to 'float' (matches the deployed engine).
 */
export function featurize(board: BoardState, mode: FeaturizeMode = 'float'): Float32Array {
  const x = board.turn === 'w' ? featurizeWhiteToMove(board) : featurizeBlackToMove(board);
  if (mode === 'trained') applyInt8Truncation(x);
  return x;
}

/**
 * Mirror the int8 storage the model trained on: truncate toward zero (== int8
 * cast for all in-range plane values). Mutates and returns `x`.
 */
export function applyInt8Truncation(x: Float32Array): Float32Array {
  for (let i = 0; i < x.length; i++) x[i] = Math.trunc(x[i]);
  return x;
}

export type { Color };

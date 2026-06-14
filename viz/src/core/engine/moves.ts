// Move encoding — faithful port of src/v2/moves.py (AlphaZero-style 8×8×73).
//
// Flat index = from_square*73 + move_type, from_square python-chess (a1=0..h8=63).
// 73 move types: 56 sliding (8 dir × 7 dist) + 8 knight + 9 underpromotion.
// Queen-promotions are encoded as the regular sliding move (NOT in the
// underpromotion block).
//
// rotateSquare(sq) = sq ^ 56 == chess.square_mirror == VERTICAL mirror (flip
// rank, keep file). Used to un-rotate model moves when black was to move. (NOT
// 63-sq, which would flip both axes — that is wrong; the engine is authoritative.)

import { PIECE_TYPE_INT, type BoardState } from './boardState.ts';

export const NUM_MOVE_TYPES = 73;
export const NUM_MOVES = 64 * NUM_MOVE_TYPES; // 4672

// (dh,dw) per direction 0..7: N,NE,E,SE,S,SW,W,NW. h=rank, w=file.
const SLIDE_DIRS: Array<[number, number]> = [
  [+1, 0],
  [+1, +1],
  [0, +1],
  [-1, +1],
  [-1, 0],
  [-1, -1],
  [0, -1],
  [+1, -1],
];
const NUM_SLIDE_TYPES = 56;

const KNIGHT_DIRS: Array<[number, number]> = [
  [+2, +1],
  [+2, -1],
  [-2, +1],
  [-2, -1],
  [+1, +2],
  [+1, -2],
  [-1, +2],
  [-1, -2],
];
const NUM_KNIGHT_TYPES = 8;

const UNDERPROM_DIRS = [-1, 0, +1]; // df: left, forward, right
// python-chess piece ints: KNIGHT=2, BISHOP=3, ROOK=4
const UNDERPROM_PIECES = [2, 3, 4];

const QUEEN = 5; // python-chess QUEEN

export interface Move {
  from: number;
  to: number;
  /** python-chess promotion piece int (2=N,3=B,4=R,5=Q) or null/undefined. */
  promotion?: number | null;
}

// (from,to,prom) -> move_type. prom key: 0 = none, else piece int.
const ENCODE = new Map<number, number>();
// decode[from*73+mt] = {to, prom(-1 if none/queen-slide)}
const DECODE_TO = new Int32Array(NUM_MOVES).fill(-1);
const DECODE_PROM = new Int32Array(NUM_MOVES).fill(-1);

function encKey(from: number, to: number, prom: number): number {
  // from,to in [0,64), prom in [0,6]
  return (from * 64 + to) * 8 + prom;
}

function rankOf(sq: number): number {
  return sq >> 3;
}
function fileOf(sq: number): number {
  return sq & 7;
}

(function buildTables() {
  // Sliding moves (+ queen-promotion sharing the slot)
  for (let from = 0; from < 64; from++) {
    const fh = rankOf(from);
    const fw = fileOf(from);
    for (let d = 0; d < SLIDE_DIRS.length; d++) {
      const [dh, dw] = SLIDE_DIRS[d];
      for (let dist = 1; dist < 8; dist++) {
        const th = fh + dh * dist;
        const tw = fw + dw * dist;
        if (th < 0 || th >= 8 || tw < 0 || tw >= 8) continue;
        const to = th * 8 + tw;
        const mt = d * 7 + (dist - 1);
        ENCODE.set(encKey(from, to, 0), mt);
        ENCODE.set(encKey(from, to, QUEEN), mt);
        DECODE_TO[from * 73 + mt] = to;
        DECODE_PROM[from * 73 + mt] = -1;
      }
    }
  }
  // Knight moves
  for (let from = 0; from < 64; from++) {
    const fh = rankOf(from);
    const fw = fileOf(from);
    for (let kI = 0; kI < KNIGHT_DIRS.length; kI++) {
      const [dh, dw] = KNIGHT_DIRS[kI];
      const th = fh + dh;
      const tw = fw + dw;
      if (th < 0 || th >= 8 || tw < 0 || tw >= 8) continue;
      const to = th * 8 + tw;
      const mt = NUM_SLIDE_TYPES + kI;
      ENCODE.set(encKey(from, to, 0), mt);
      DECODE_TO[from * 73 + mt] = to;
      DECODE_PROM[from * 73 + mt] = -1;
    }
  }
  // Underpromotions (N,B,R only)
  for (let from = 0; from < 64; from++) {
    const fh = rankOf(from);
    const fw = fileOf(from);
    for (const dh of [+1, -1]) {
      const th = fh + dh;
      if (th !== 0 && th !== 7) continue;
      for (let dfI = 0; dfI < UNDERPROM_DIRS.length; dfI++) {
        const tw = fw + UNDERPROM_DIRS[dfI];
        if (tw < 0 || tw >= 8) continue;
        const to = th * 8 + tw;
        for (let pI = 0; pI < UNDERPROM_PIECES.length; pI++) {
          const piece = UNDERPROM_PIECES[pI];
          const mt = NUM_SLIDE_TYPES + NUM_KNIGHT_TYPES + dfI * 3 + pI;
          ENCODE.set(encKey(from, to, piece), mt);
          DECODE_TO[from * 73 + mt] = to;
          DECODE_PROM[from * 73 + mt] = piece;
        }
      }
    }
  }
})();

/** Encode a move to its flat index in [0,4672), or -1 if unencodable. */
export function encodeMove(move: Move): number {
  if (move.from == null || move.to == null) return -1;
  const prom = move.promotion ?? 0;
  const mt = ENCODE.get(encKey(move.from, move.to, prom));
  if (mt === undefined) return -1;
  return move.from * NUM_MOVE_TYPES + mt;
}

/**
 * Decode a flat index back to a Move on the given board. The board disambiguates
 * queen-promotion vs a plain sliding move for a pawn reaching the last rank.
 */
export function decodeMove(flatIdx: number, board: BoardState): Move {
  const from = Math.floor(flatIdx / NUM_MOVE_TYPES);
  const mt = flatIdx % NUM_MOVE_TYPES;
  const to = DECODE_TO[from * 73 + mt];
  if (to < 0) return { from, to: from, promotion: null }; // unmapped slot
  let promotion: number | null = DECODE_PROM[from * 73 + mt] < 0 ? null : DECODE_PROM[from * 73 + mt];
  if (promotion === null) {
    const pt = board.pieceTypeAt(from);
    if (pt === 'p') {
      const toRank = rankOf(to);
      if (toRank === 0 || toRank === 7) promotion = QUEEN;
    }
  }
  return { from, to, promotion };
}

/** Build a (4672,) legal mask from a list of legal moves (encoding each). */
export function legalMask(legalMoves: Move[]): Uint8Array {
  const mask = new Uint8Array(NUM_MOVES);
  for (const mv of legalMoves) {
    const idx = encodeMove(mv);
    if (idx >= 0) mask[idx] = 1;
  }
  return mask;
}

/**
 * Softmax over ONLY the legal flat indices → Map(flatIdx → probability), summing
 * to 1. Mathematically identical to a full-domain softmax then mask+renormalize
 * over the legal set (the engine's policyProbs), but computed straight from the
 * legal indices so it can be derived from just the logits + a legal-move list.
 */
export function legalPolicySoftmax(logits: ArrayLike<number>, legalIndices: number[]): Map<number, number> {
  const out = new Map<number, number>();
  if (legalIndices.length === 0) return out;
  let max = -Infinity;
  for (const i of legalIndices) if (logits[i] > max) max = logits[i];
  const weights: number[] = [];
  let total = 0;
  for (const i of legalIndices) {
    const e = Math.exp(logits[i] - max);
    weights.push(e);
    total += e;
  }
  total = total || 1;
  legalIndices.forEach((i, k) => out.set(i, weights[k] / total));
  return out;
}

/**
 * P(the model moves the piece on each FROM-square) — for each from-square, the
 * SUM of the legal-masked, renormalized policy probability over the 73 move-types
 * originating from that square. Returns a 64-array indexed by from-square
 * (a1=0 .. h8=63), in whatever frame the supplied indices are in (engine frame
 * for the policy head). SUM, not average: a square's total move probability.
 */
export function pieceToMoveProbs(probsByIndex: Map<number, number>): Float32Array {
  const out = new Float32Array(64);
  for (const [idx, p] of probsByIndex) out[Math.floor(idx / NUM_MOVE_TYPES)] += p;
  return out;
}

/** Vertical-mirror a square (flip rank, keep file): chess.square_mirror. */
export function rotateSquare(sq: number): number {
  return sq ^ 56;
}

export { PIECE_TYPE_INT };

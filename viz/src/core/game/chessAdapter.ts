// chess.js ↔ engine adapter (presentation-agnostic).
//
// Bridges chess.js (the rules/legality source) to the engine's chess-free
// BoardState contract, and translates between chess.js's algebraic squares and
// the engine's python-chess square indices (a1=0 .. h8=63, idx = rank*8 + file).
//
// The engine plays in a "moving-side-at-bottom" frame: for black-to-move,
// featurize() vertically mirrors the board, so the policy output is in the
// rotated frame. We therefore build the legal mask by rotating each chess.js
// legal move into the engine frame (sq ^ 56) before encoding, and remember the
// ORIGINAL chess.js move per policy index — so mapping the model's chosen index
// back to a real move is the exact inverse rotation, done robustly.

import { Chess, type Move as ChessMove } from 'chess.js';
import { encodeMove, NUM_MOVES } from '../engine/index.ts';
import type { BoardState, Color, PieceInfo, PieceType } from '../engine/index.ts';

/** Algebraic square ('e4') → python-chess index (0..63). */
export function algToIdx(sq: string): number {
  return (sq.charCodeAt(1) - 49) * 8 + (sq.charCodeAt(0) - 97);
}

/** python-chess index (0..63) → algebraic square ('e4'). */
export function idxToAlg(idx: number): string {
  return String.fromCharCode(97 + (idx & 7)) + String.fromCharCode(49 + (idx >> 3));
}

/** chess.js promotion letter → python-chess piece int (n=2,b=3,r=4,q=5). */
const PROM_INT: Record<string, number> = { n: 2, b: 3, r: 4, q: 5 };

/**
 * Build a BoardState (the engine's featurize/decode contract) from a chess.js
 * position. `epSquare` is the python-chess en-passant target in REAL coords
 * (set after any double pawn push, matching python-chess semantics); featurize
 * applies the rotation. `isRepetition` is supplied by the game store, which
 * tracks position counts.
 */
export function chessToBoardState(
  chess: Chess,
  ctx: { epSquare: number | null; isRepetition: (count: number) => boolean },
): BoardState {
  const fenParts = chess.fen().split(' ');
  const castling = fenParts[2];
  const halfmove = parseInt(fenParts[4], 10) || 0;
  const fullmove = parseInt(fenParts[5], 10) || 1;
  const turn = chess.turn() as Color;

  const pieces: PieceInfo[] = [];
  const bySquare = new Map<number, PieceType>();
  for (const row of chess.board()) {
    for (const cell of row) {
      if (!cell) continue;
      const idx = algToIdx(cell.square);
      pieces.push({ square: idx, color: cell.color as Color, type: cell.type as PieceType });
      bySquare.set(idx, cell.type as PieceType);
    }
  }

  return {
    turn,
    pieces: () => pieces,
    hasKingsideCastlingRights: (c) => castling.includes(c === 'w' ? 'K' : 'k'),
    hasQueensideCastlingRights: (c) => castling.includes(c === 'w' ? 'Q' : 'q'),
    epSquare: ctx.epSquare,
    halfmoveClock: halfmove,
    fullmoveNumber: fullmove,
    isRepetition: ctx.isRepetition,
    pieceTypeAt: (sq) => bySquare.get(sq) ?? null,
  };
}

/**
 * Build the (4672) legal mask AND a map from each occupied policy index back to
 * the originating chess.js move. For black-to-move, squares are rotated into the
 * engine frame (sq ^ 56) before encoding. The returned map lets the play loop
 * convert the model's chosen index straight back to a real, fully-specified
 * chess.js move (correct promotion included) — the robust inverse of rotation.
 */
export function buildLegalMaskAndMap(chess: Chess): {
  mask: Uint8Array;
  indexToMove: Map<number, ChessMove>;
} {
  const black = chess.turn() === 'b';
  const mask = new Uint8Array(NUM_MOVES);
  const indexToMove = new Map<number, ChessMove>();
  for (const mv of chess.moves({ verbose: true })) {
    let from = algToIdx(mv.from);
    let to = algToIdx(mv.to);
    if (black) {
      from ^= 56;
      to ^= 56;
    }
    const promotion = mv.promotion ? PROM_INT[mv.promotion] : null;
    const idx = encodeMove({ from, to, promotion });
    if (idx >= 0) {
      mask[idx] = 1;
      indexToMove.set(idx, mv);
    }
  }
  return { mask, indexToMove };
}

/** En-passant target (real-coords python-chess index) after a move, or null. */
export function epTargetAfterMove(mv: ChessMove): number | null {
  // chess.js flag 'b' == big pawn move (two-square advance); the skipped square
  // is the midpoint. python-chess sets ep_square after ANY double push.
  if (!mv.flags.includes('b')) return null;
  return (algToIdx(mv.from) + algToIdx(mv.to)) / 2;
}

/** En-passant target parsed from a FEN's ep field, or null. */
export function epTargetFromFen(fen: string): number | null {
  const field = fen.split(' ')[3];
  return !field || field === '-' ? null : algToIdx(field);
}

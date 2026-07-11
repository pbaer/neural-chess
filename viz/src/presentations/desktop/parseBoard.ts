// FEN → board-cell parsing shared by the board-drawing components (the play
// board, StageDetail's plane/board glyphs, and the AttentionBoard).

import type { Color, PieceType } from '../../core/index.ts';

export interface BoardCell {
  color: Color;
  type: PieceType;
}

/** Parse a FEN placement field into a 64-array indexed by python-chess square. */
export function parseBoard(fen: string): (BoardCell | null)[] {
  const arr: (BoardCell | null)[] = new Array(64).fill(null);
  const rows = (fen.split(' ')[0] ?? '').split('/'); // rows[0] = rank 8
  for (let r = 0; r < 8; r++) {
    const rankIdx = 7 - r;
    let file = 0;
    for (const ch of rows[r] ?? '') {
      if (ch >= '1' && ch <= '8') {
        file += ch.charCodeAt(0) - 48;
      } else {
        arr[rankIdx * 8 + file] = { color: ch === ch.toUpperCase() ? 'w' : 'b', type: ch.toLowerCase() as PieceType };
        file++;
      }
    }
  }
  return arr;
}

// BoardState — the chess-board contract that featurize() and moves decode()
// consume. The engine is chess.js-free: the game layer (M2, chess.js wrapper)
// implements this; parity tests construct a plain object from golden data.
//
// Square indexing is python-chess: a1=0 .. h8=63, square = rank*8 + file,
// rank = square>>3, file = square&7.

export type Color = 'w' | 'b';

/** Piece type letters, lowercase (python-chess / SAN convention). */
export type PieceType = 'p' | 'n' | 'b' | 'r' | 'q' | 'k';

export interface PieceInfo {
  /** python-chess square index 0..63. */
  square: number;
  color: Color;
  type: PieceType;
}

export interface BoardState {
  /** Side to move. */
  turn: Color;
  /** All pieces on the board. */
  pieces(): PieceInfo[];
  hasKingsideCastlingRights(color: Color): boolean;
  hasQueensideCastlingRights(color: Color): boolean;
  /** En-passant target square (python-chess index) or null. */
  epSquare: number | null;
  halfmoveClock: number;
  fullmoveNumber: number;
  /**
   * True if the current position has occurred at least `count` times in the
   * game (mirrors python-chess Board.is_repetition). featurize uses 3 then 2.
   */
  isRepetition(count: number): boolean;
  /** Piece type at a square, or null. Needed by decodeMove for promotions. */
  pieceTypeAt(square: number): PieceType | null;
}


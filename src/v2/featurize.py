# -*- coding: utf-8 -*-
"""v2 board featurization: python-chess Board -> (21, 8, 8) float32 input.

21 planes, all derived from the *observable game state* (positions and the
move record so far) — no computed chess heuristics. See
memory/project-principles.md Principle 2.

**Rotation convention**: when it is black to move, we mirror the board
(via python-chess's board.mirror(), which both spatially flips and swaps
colors) before featurizing. This means the moving side always occupies
planes 0-5 ("own pieces") and the bottom rank (h=0 = rank 1 from moving
side's perspective). The model effectively always sees "white to move."

This is a coordinate-frame transformation, NOT a chess-theory injection
(per Principle 2 — coordinate choices are inductive bias / spatial
preprocessing, not "human chess knowledge fed as supervision"). It does
materially help the model: without rotation, the model has to learn pawn
directionality and own/opp distinction across both colors — empirically
that's a significant capacity cost (epoch 9 of T1a without rotation
produced 100%/0% win-rate split between white and black side).

Plane layout (with rotation applied — moving side always in planes 0-5):
   0-5   own pieces by type (P, R, N, B, Q, K)
   6-11  opponent pieces by type (same order)
   12    castling: own kingside    (constant-valued plane: all 1s if right exists)
   13    castling: own queenside
   14    castling: opponent kingside
   15    castling: opponent queenside
   16    en passant target square  (single 1 at the EP target in rotated frame)
   17    halfmove clock / 100      (constant-valued plane, capped at 1.0)
   18    fullmove number / 100     (constant-valued plane, capped at 2.0)
   19    side to move              (constant 1.0 — always white in the rotated
                                    frame; retained for input-shape stability
                                    with earlier no-rotation v2 design, model
                                    can ignore it)
   20    threefold-repetition count: 0.0, 0.5, or 1.0  (have we seen this
         position 0 / 1 / 2+ times before in this game)

Coordinate convention in the rotated frame: spatial (h, w) maps to
python-chess square = h * 8 + w. Move-encoding and decoding stay in this
rotated frame; callers un-rotate at the boundary by applying 63 - sq
to from_square and to_square.
"""
import chess
import numpy as np

NUM_PLANES = 21

_WHITE_PIECE_PLANE = {
    chess.PAWN: 0, chess.ROOK: 1, chess.KNIGHT: 2,
    chess.BISHOP: 3, chess.QUEEN: 4, chess.KING: 5,
}
_BLACK_PIECE_PLANE = {p: 6 + i for p, i in _WHITE_PIECE_PLANE.items()}


def _featurize_white_to_move(board: chess.Board) -> np.ndarray:
    """Featurize a board where white is to move. Planes 0-5 = white = moving side.

    Callers that have a black-to-move board should call .mirror() on it first
    (which swaps colors AND flips spatially), then pass the result here.
    """
    x = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    # Piece planes (0-11): planes 0-5 = white (moving side), 6-11 = black (opponent)
    for square, piece in board.piece_map().items():
        h = chess.square_rank(square)
        w = chess.square_file(square)
        plane = (_WHITE_PIECE_PLANE if piece.color == chess.WHITE
                 else _BLACK_PIECE_PLANE)[piece.piece_type]
        x[plane, h, w] = 1.0

    # Castling rights (12-15): white = own, black = opp (in the post-mirror frame)
    if board.has_kingside_castling_rights(chess.WHITE):
        x[12, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        x[13, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        x[14, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        x[15, :, :] = 1.0

    # En passant target (16) — in current frame; if board was mirrored upstream,
    # ep_square is already in the mirrored coordinate system.
    if board.ep_square is not None:
        eh = chess.square_rank(board.ep_square)
        ew = chess.square_file(board.ep_square)
        x[16, eh, ew] = 1.0

    # Counters (17, 18) — normalized & clipped
    x[17, :, :] = min(board.halfmove_clock / 100.0, 1.0)
    x[18, :, :] = min(board.fullmove_number / 100.0, 2.0)

    # Side to move (19) — always 1.0 in the rotated frame (moving side = white)
    x[19, :, :] = 1.0

    # Repetition count (20)
    if board.is_repetition(3):
        x[20, :, :] = 1.0
    elif board.is_repetition(2):
        x[20, :, :] = 0.5

    return x


def _featurize_black_to_move(board: chess.Board) -> np.ndarray:
    """Featurize a black-to-move board in the rotated frame (moving side at bottom).

    Equivalent to _featurize_white_to_move(board.mirror()) but avoids the
    expensive board.mirror() copy — instead we read each piece directly
    and place it in the mirrored coordinates. Important because shard
    generation calls this on ~7.5M positions.

    Mirror is **vertical only** (rank flips, file stays). This matches
    python-chess's board.mirror() and the natural chess convention:
    black's d-file queens stay on the d-file from the moving-side view,
    kingside stays kingside.
    """
    x = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    # Piece planes: in the rotated frame, BLACK becomes "own" (planes 0-5)
    # and WHITE becomes "opponent" (planes 6-11). Spatially, rank is flipped.
    for square, piece in board.piece_map().items():
        h = 7 - chess.square_rank(square)
        w = chess.square_file(square)           # file UNCHANGED
        if piece.color == chess.BLACK:
            plane = _WHITE_PIECE_PLANE[piece.piece_type]      # own slot
        else:
            plane = _BLACK_PIECE_PLANE[piece.piece_type]      # opp slot
        x[plane, h, w] = 1.0

    # Castling rights: BLACK rights become "own", WHITE rights become "opp"
    if board.has_kingside_castling_rights(chess.BLACK):
        x[12, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        x[13, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.WHITE):
        x[14, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        x[15, :, :] = 1.0

    # En passant target: vertical mirror
    if board.ep_square is not None:
        eh = 7 - chess.square_rank(board.ep_square)
        ew = chess.square_file(board.ep_square)
        x[16, eh, ew] = 1.0

    x[17, :, :] = min(board.halfmove_clock / 100.0, 1.0)
    x[18, :, :] = min(board.fullmove_number / 100.0, 2.0)
    x[19, :, :] = 1.0  # side to move always 1.0 in rotated frame

    if board.is_repetition(3):
        x[20, :, :] = 1.0
    elif board.is_repetition(2):
        x[20, :, :] = 0.5

    return x


def featurize(board: chess.Board) -> np.ndarray:
    """Encode a chess.Board into (21, 8, 8) float32 model input.

    Applies rotation for black-to-move so the moving side is always at the
    bottom (planes 0-5 = own pieces). Callers must un-rotate any predicted
    move via rotate_square() if the input board had board.turn == chess.BLACK.
    """
    if board.turn == chess.WHITE:
        return _featurize_white_to_move(board)
    return _featurize_black_to_move(board)


def rotate_square(sq: int) -> int:
    """Vertical-mirror a square: flip rank, keep file.

    Matches python-chess board.mirror() / chess.square_mirror(). Used to
    convert moves between the actual board and the rotated frame the model
    sees when black is to move.
    """
    return chess.square_mirror(sq)

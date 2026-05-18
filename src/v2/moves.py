# -*- coding: utf-8 -*-
"""v2 move encoding: AlphaZero-style 8x8x73 = 4672 move space.

For each of the 64 from-squares, 73 possible "move types":
  - 56 sliding moves: 8 directions x 7 distances (1..7)
    direction order: N=0, NE=1, E=2, SE=3, S=4, SW=5, W=6, NW=7
    queen-promotions are encoded as the regular sliding move to last rank
  - 8 knight moves: 8 (dr, df) offsets in a fixed order
  - 9 underpromotions: 3 directions (left-diagonal, forward, right-diagonal)
    x 3 promotion pieces (knight, bishop, rook). Queen-promos use the
    regular sliding move (above), not this block.

Flat index convention: idx = from_square * 73 + move_type
where from_square is the python-chess square index (a1=0..h8=63).

So the model's output tensor (B, 73, 8, 8) — channel = move_type,
spatial = from_square (h*8 + w == from_square in python-chess).
When flattened it gives flat[from_sq * 73 + move_type].
"""
import chess
import numpy as np

# Sliding directions: (dh, dw) per direction index 0..7
# h = rank (0..7), w = file (0..7); +h is "up the board for white"
_SLIDE_DIRS = [
    (+1,  0),  # 0: N
    (+1, +1),  # 1: NE
    ( 0, +1),  # 2: E
    (-1, +1),  # 3: SE
    (-1,  0),  # 4: S
    (-1, -1),  # 5: SW
    ( 0, -1),  # 6: W
    (+1, -1),  # 7: NW
]
NUM_SLIDE_TYPES = 8 * 7  # 56

# Knight offsets in a fixed canonical order
_KNIGHT_DIRS = [
    (+2, +1), (+2, -1), (-2, +1), (-2, -1),
    (+1, +2), (+1, -2), (-1, +2), (-1, -2),
]
NUM_KNIGHT_TYPES = 8

# Underpromotion: (dh_relative, dw_relative, promotion_piece)
# For white promoting to rank 8: dh = +1; dw in {-1, 0, +1} (capture-left,
# forward, capture-right). Promotion piece is N (knight), B (bishop), or R (rook).
# Queen-promotions use the regular sliding move (sliding N, distance 1).
# For black, dh = -1, but the move-type is still indexed by (relative direction)
# because the move-type space is fixed at 73 regardless of color.
_UNDERPROM_DIRS = [-1, 0, +1]   # df offset (left, forward, right)
_UNDERPROM_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
NUM_UNDERPROM_TYPES = 9

NUM_MOVE_TYPES = NUM_SLIDE_TYPES + NUM_KNIGHT_TYPES + NUM_UNDERPROM_TYPES  # 73
NUM_MOVES = 64 * NUM_MOVE_TYPES  # 4672


# ---- Build lookup tables once (encoder + decoder) ----

def _build_encode_table():
    """Returns dict: (from_sq, to_sq, promotion_or_None) -> move_type_index"""
    table = {}

    # Sliding moves
    for from_sq in range(64):
        fh, fw = chess.square_rank(from_sq), chess.square_file(from_sq)
        for d_idx, (dh, dw) in enumerate(_SLIDE_DIRS):
            for dist in range(1, 8):
                th = fh + dh * dist
                tw = fw + dw * dist
                if 0 <= th < 8 and 0 <= tw < 8:
                    to_sq = th * 8 + tw
                    move_type = d_idx * 7 + (dist - 1)
                    table[(from_sq, to_sq, None)] = move_type
                    # Queen-promotion uses the sliding move encoding too
                    # (only matters when from on rank 7 white / rank 1 black)
                    table[(from_sq, to_sq, chess.QUEEN)] = move_type

    # Knight moves
    for from_sq in range(64):
        fh, fw = chess.square_rank(from_sq), chess.square_file(from_sq)
        for k_idx, (dh, dw) in enumerate(_KNIGHT_DIRS):
            th = fh + dh
            tw = fw + dw
            if 0 <= th < 8 and 0 <= tw < 8:
                to_sq = th * 8 + tw
                move_type = NUM_SLIDE_TYPES + k_idx
                table[(from_sq, to_sq, None)] = move_type

    # Underpromotions (N, B, R only — Q uses sliding)
    for from_sq in range(64):
        fh, fw = chess.square_rank(from_sq), chess.square_file(from_sq)
        # Determine promotion direction by source rank: white promotes from
        # rank 7 (move +1), black promotes from rank 1 (move -1)
        for dh in (+1, -1):
            th = fh + dh
            if th != 0 and th != 7:
                continue  # underpromotion only into last-rank
            for df_idx, df in enumerate(_UNDERPROM_DIRS):
                tw = fw + df
                if not (0 <= tw < 8):
                    continue
                to_sq = th * 8 + tw
                for p_idx, piece in enumerate(_UNDERPROM_PIECES):
                    move_type = (NUM_SLIDE_TYPES + NUM_KNIGHT_TYPES
                                 + df_idx * 3 + p_idx)
                    # Same (from, to, piece) keyed by promotion piece
                    table[(from_sq, to_sq, piece)] = move_type

    return table


_ENCODE_TABLE = _build_encode_table()


def _build_decode_table():
    """Returns array of shape (64, 73, 3) -> (from_sq, to_sq, promotion_or_-1)
    where promotion is -1 if not a promotion, else the chess piece type int.
    Used to convert (from_sq, move_type) back to a chess.Move quickly.
    """
    table = np.full((64, NUM_MOVE_TYPES, 3), -1, dtype=np.int8)
    for (from_sq, to_sq, prom), mt in _ENCODE_TABLE.items():
        # Sliding & queen-promotion share a slot; the inverse for that slot
        # records the no-promotion-needed sliding move. Promotion is set
        # separately when the from-rank is pre-promotion.
        # For underpromotions specifically the slot is unambiguous.
        # We'll just take the last writer; underpromotion entries are written
        # after sliding so they overwrite cleanly for those move_type slots
        # but those move_types don't overlap with sliding anyway.
        table[from_sq, mt, 0] = from_sq
        table[from_sq, mt, 1] = to_sq
        table[from_sq, mt, 2] = -1 if prom is None or prom == chess.QUEEN else prom
    return table


_DECODE_TABLE = _build_decode_table()


def encode_move(move: chess.Move) -> int:
    """Convert a chess.Move to its flat index in [0, 4672).

    Returns -1 if the move can't be encoded (shouldn't happen for legal moves
    but defensive — e.g., null moves).
    """
    if move is None or move.from_square is None or move.to_square is None:
        return -1
    key = (move.from_square, move.to_square, move.promotion)
    mt = _ENCODE_TABLE.get(key, -1)
    if mt < 0:
        return -1
    return move.from_square * NUM_MOVE_TYPES + mt


def decode_move(flat_idx: int, board: chess.Board) -> chess.Move:
    """Convert a flat move index back to a chess.Move on the given board.

    The board is needed to:
      - Disambiguate queen-promotion vs regular sliding move for pawns
        reaching the last rank (the move_type slot is the same).
      - Always returns a chess.Move object; legality is the caller's problem.
    """
    from_sq = flat_idx // NUM_MOVE_TYPES
    mt = flat_idx % NUM_MOVE_TYPES
    to_sq = int(_DECODE_TABLE[from_sq, mt, 1])
    prom = int(_DECODE_TABLE[from_sq, mt, 2])
    if to_sq < 0:
        # Unmapped slot (shouldn't be hit for indexes from a softmax)
        return chess.Move.null()

    promotion = None if prom < 0 else prom

    # If this is a sliding move and the moving piece is a pawn reaching the
    # last rank, it MUST be a promotion. Default to queen (the underpromotion
    # paths use distinct move_type slots so we won't be here for those).
    if promotion is None:
        piece = board.piece_type_at(from_sq)
        if piece == chess.PAWN:
            to_rank = chess.square_rank(to_sq)
            if to_rank == 0 or to_rank == 7:
                promotion = chess.QUEEN

    return chess.Move(from_sq, to_sq, promotion=promotion)


def legal_mask(board: chess.Board) -> np.ndarray:
    """Return a (4672,) bool array marking which flat-indices are legal moves.

    Iterates board.legal_moves and encodes each. Used by the policy head's
    legal-move masking step (analogous to v1, but over the larger move space).
    """
    mask = np.zeros(NUM_MOVES, dtype=bool)
    for mv in board.legal_moves:
        idx = encode_move(mv)
        if idx >= 0:
            mask[idx] = True
    return mask

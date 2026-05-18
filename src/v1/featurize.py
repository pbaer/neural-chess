# -*- coding: utf-8 -*-
"""v1 board featurization: FEN -> 6-plane sign-encoded -> 12-plane binary.

Shared between v1 training (ChessDataset reads 6-plane NPZ, expands per-batch)
and v1 inference (featurize_board_for_model goes straight from FEN to 12-plane).
"""
import numpy as np

# Shape constants for the parsed NPZ format.
_FEATURES = 6 * 8 * 8
_OUTPUTS = 64 * 64

PAWN_PLANE = 0
ROOK_PLANE = 1
KNIGHT_PLANE = 2
BISHOP_PLANE = 3
QUEEN_PLANE = 4
KING_PLANE = 5

WHITE = True
BLACK = False


def featurize_board(board_fen, rotate=False):
    board_array = np.zeros((6, 8, 8), dtype='int8')
    f = -1 # file (column)
    r = 0 # rank (row)
    for c in board_fen:
        f += 1
        if c == 'P' or c == 'p':
            board_array[PAWN_PLANE, r, f] = 1 if c == 'P' else -1
        elif c == 'R' or c == 'r':
            board_array[ROOK_PLANE, r, f] = 1 if c == 'R' else -1
        elif c == 'N' or c == 'n':
            board_array[KNIGHT_PLANE, r, f] = 1 if c == 'N' else -1
        elif c == 'B' or c == 'b':
            board_array[BISHOP_PLANE, r, f] = 1 if c == 'B' else -1
        elif c == 'Q' or c == 'q':
            board_array[QUEEN_PLANE, r, f] = 1 if c == 'Q' else -1
        elif c == 'K' or c == 'k':
            board_array[KING_PLANE, r, f] = 1 if c == 'K' else -1
        elif c == '/':
            assert f == 8
            f = -1
            r += 1
        elif c == ' ':
            break
        else: # a number indicating 1 or more blank squares
            f += int(c) - 1
    # TODO: add parsing for castling availability
    if rotate:
        for p in range(6): # all planes
            for i in range(4): # first half of the ranks
                for j in range(8): # all files
                    temp = board_array[p, i, j]
                    board_array[p, i, j] = -board_array[p, 7-i, 7-j]
                    board_array[p, 7-i, 7-j] = -temp
    return board_array.reshape((1, _FEATURES))


def expand_planes(x_6):
    """Convert a sign-encoded 6-plane board to a binary 12-plane representation.

    Input:  (6, 8, 8) array with values in {-1, 0, +1}
    Output: (12, 8, 8) float32 — planes 0-5 are the moving side's pieces
            (val > 0), planes 6-11 are the opponent's pieces (val < 0), in
            the same piece-type order (P, R, N, B, Q, K).

    The two-plane-per-piece-type layout gives the convolutions cleaner
    features than the single-plane sign encoding: a filter that fires on
    "my pawn here" doesn't have to also learn that magnitude encodes
    color, and the gradient signal for the two cases is independent.
    """
    x_12 = np.zeros((12, 8, 8), dtype=np.float32)
    x_12[:6] = (x_6 > 0)
    x_12[6:] = (x_6 < 0)
    return x_12


def featurize_board_for_model(board_fen, rotate=False):
    """Produce model-ready (12, 8, 8) float32 input from a FEN string.

    Combines featurize_board (FEN -> 6-plane sign-encoded) with expand_planes
    (-> 12-plane own/opponent binary). Used by v1 inference code paths.
    Training uses expand_planes directly on the cached 6-plane NPZ data.
    """
    return expand_planes(featurize_board(board_fen, rotate=rotate).reshape(6, 8, 8))

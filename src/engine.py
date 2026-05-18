# -*- coding: utf-8 -*-
"""Architecture-invariant Stockfish/UCI engine helpers."""
import os

import chess
import chess.engine


def create_engine(path='bin/stockfish.exe', depth=0, skill_level=0):
    """Open a UCI engine via python-chess and configure depth/skill."""
    # asyncio's subprocess on Windows refuses a bare relative path like
    # "bin/stockfish.exe" — resolve to an absolute path so it works whether
    # the caller passed "bin/stockfish.exe", "./bin/stockfish.exe", or an
    # already-absolute path.
    if not os.path.isabs(path) and os.path.exists(path):
        path = os.path.abspath(path)
    engine = chess.engine.SimpleEngine.popen_uci(path)
    engine.configure({"Skill Level": skill_level})
    engine._depth = depth  # stash for use in play_engine
    return engine


def generate_engine_move(engine, board):
    depth = getattr(engine, '_depth', 1)
    result = engine.play(board, chess.engine.Limit(depth=depth))
    return result.move

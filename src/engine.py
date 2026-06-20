# -*- coding: utf-8 -*-
"""Architecture-invariant Stockfish/UCI engine helpers."""
import glob
import os
import re

import chess
import chess.engine

# Versioned Stockfish binaries live in bin/ as `stockfish-vNN.exe` (NN = the
# Stockfish major version). Only the small classical `stockfish-v08.exe` is
# committed; newer builds (e.g. the 100 MB+ NNUE `stockfish-v18.exe`) are kept
# LOCAL ONLY (gitignored, too big for GitHub's 100 MB file limit). The resolver
# below auto-prefers the highest local version, so evals use the strongest
# Stockfish you have on disk while clones still work off the committed v08.
_BIN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'bin')


def resolve_stockfish(path=None):
    """Return an absolute path to a Stockfish binary.

    An explicit `path` that exists wins (caller override). Otherwise pick the
    highest-versioned `bin/stockfish-vNN.exe` present, then fall back to a
    legacy `bin/stockfish.exe`. Raises if none is found.
    """
    if path and os.path.exists(path):
        return os.path.abspath(path)
    versioned = []
    for p in glob.glob(os.path.join(_BIN_DIR, 'stockfish-v*.exe')):
        m = re.search(r'stockfish-v(\d+)\.exe$', os.path.basename(p))
        if m:
            versioned.append((int(m.group(1)), p))
    if versioned:
        return os.path.abspath(max(versioned)[1])
    legacy = os.path.join(_BIN_DIR, 'stockfish.exe')
    if os.path.exists(legacy):
        return os.path.abspath(legacy)
    raise FileNotFoundError(
        f"No Stockfish binary found (looked for {path!r}, bin/stockfish-vNN.exe, bin/stockfish.exe)")


def create_engine(path=None, depth=0, skill_level=0):
    """Open a UCI engine via python-chess and configure depth/skill.

    `path=None` (or a path that doesn't exist) auto-resolves to the newest
    local Stockfish — see resolve_stockfish().
    """
    path = resolve_stockfish(path)
    engine = chess.engine.SimpleEngine.popen_uci(path)
    engine.configure({"Skill Level": skill_level})
    engine._depth = depth  # stash for use in play_engine
    return engine


def generate_engine_move(engine, board):
    depth = getattr(engine, '_depth', 1)
    result = engine.play(board, chess.engine.Limit(depth=depth))
    return result.move

# -*- coding: utf-8 -*-
"""Sample opening sequences under the realistic eval config to study what the
model actually plays in the opening phase.

Plays N games against Stockfish at a configurable difficulty with temperature
0.5 and decay 0.05/ply on the model's side. For each game records the first
`n_plies` plies in SAN, then prints the most common sequences for analysis.

Usage:
    python eval/v1/opening_diversity.py [epoch] [n_games] [n_plies] [depth] [skill]
Defaults: epoch=7, n_games=1000, n_plies=8 (4 moves per side), depth=5, skill=10.
"""
import os
import sys
import time
from collections import Counter

# Allow running from any cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import chess  # noqa: E402

from src.engine import create_engine, generate_engine_move  # noqa: E402
from src.inference_api import load_policy_engine  # noqa: E402
from src.stats import compute_temperature, init_stats  # noqa: E402

CHECKPOINT_DIR = os.path.join(_REPO_ROOT, 'model', 'v1', 'checkpoints')
STOCKFISH_PATH = os.path.join(_REPO_ROOT, 'bin', 'stockfish.exe')

EPOCH = int(sys.argv[1]) if len(sys.argv) > 1 else 7
N_GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
N_PLIES = int(sys.argv[3]) if len(sys.argv) > 3 else 8
SF_DEPTH = int(sys.argv[4]) if len(sys.argv) > 4 else 5
SF_SKILL = int(sys.argv[5]) if len(sys.argv) > 5 else 10
COLOR_ARG = sys.argv[6].lower() if len(sys.argv) > 6 else 'white'
MODEL_COLOR = chess.WHITE if COLOR_ARG.startswith('w') else chess.BLACK
TEMP = 0.5
DECAY = 0.05


def play_one_opening(policy_engine, engine):
    board = chess.Board()
    san_moves = []
    for _ in range(N_PLIES):
        if board.is_game_over():
            break
        if board.turn == MODEL_COLOR:
            temp = compute_temperature(TEMP, DECAY, board.ply())
            stats = init_stats()
            mv = policy_engine.generate_move(board, stats, temperature=temp)
        else:
            mv = generate_engine_move(engine, board)
        san_moves.append(board.san(mv))
        board.push(mv)
    return tuple(san_moves)


def format_sequence(seq):
    """Format SAN list as '1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. O-O Nf6' style."""
    parts = []
    for i, mv in enumerate(seq):
        if i % 2 == 0:
            parts.append(f'{i//2 + 1}. {mv}')
        else:
            parts.append(mv)
    return ' '.join(parts)


def main():
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'model_e{EPOCH:04d}.pt')
    print(f'Loading {checkpoint_path}...', flush=True)
    policy_engine = load_policy_engine(checkpoint_path)
    engine = create_engine(path=STOCKFISH_PATH, depth=SF_DEPTH, skill_level=SF_SKILL)
    print(f'Stockfish: depth={SF_DEPTH}, skill={SF_SKILL}', flush=True)
    print(f'Sampling {N_GAMES} games × {N_PLIES} plies '
          f'(model={"white" if MODEL_COLOR else "black"}, '
          f't={TEMP}, decay={DECAY})\n', flush=True)

    openings = Counter()
    t0 = time.time()
    try:
        for i in range(N_GAMES):
            openings[play_one_opening(policy_engine, engine)] += 1
            if (i + 1) % 100 == 0:
                print(f'  {i+1}/{N_GAMES} done in {time.time()-t0:.1f}s', flush=True)
    finally:
        engine.quit()

    print(f'\n{N_GAMES} games in {time.time()-t0:.1f}s')
    print(f'Unique {N_PLIES}-ply sequences: {len(openings)}\n')

    print(f'Top 40 sequences (by count):')
    for seq, count in openings.most_common(40):
        pct = 100 * count / N_GAMES
        print(f'  {count:>4}  {pct:>5.1f}%   {format_sequence(seq)}')

    # Model's first move is at index 0 if white, index 1 if black.
    model_first_idx = 0 if MODEL_COLOR == chess.WHITE else 1
    label = "1st" if MODEL_COLOR == chess.WHITE else "2nd-ply (1st model)"
    print(f'\nModel\'s {label}-move distribution:')
    first_moves = Counter()
    for seq, count in openings.items():
        if len(seq) > model_first_idx:
            first_moves[seq[model_first_idx]] += count
    for mv, count in first_moves.most_common():
        print(f'  {count:>4}  {100*count/N_GAMES:>5.1f}%   {mv}')

    # When playing black, also show distribution of model's response *grouped*
    # by the white move it's responding to.
    if MODEL_COLOR == chess.BLACK and N_PLIES >= 2:
        print('\nModel\'s response (ply 1) grouped by white\'s 1st move:')
        by_white = {}
        for seq, count in openings.items():
            if len(seq) >= 2:
                w_move = seq[0]
                b_move = seq[1]
                by_white.setdefault(w_move, Counter())[b_move] += count
        for w_move in sorted(by_white, key=lambda m: -sum(by_white[m].values())):
            total = sum(by_white[w_move].values())
            print(f'  after 1.{w_move} ({total} games):')
            for b_move, count in by_white[w_move].most_common():
                print(f'      {count:>4}  {100*count/total:>5.1f}%   {b_move}')


if __name__ == '__main__':
    main()

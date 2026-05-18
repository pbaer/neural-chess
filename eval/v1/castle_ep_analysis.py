# -*- coding: utf-8 -*-
"""Measure castling and en passant rates for the model under realistic eval
conditions. 100 games as white and 100 as black against hard Stockfish.

Reports:
- castle rate (model + Stockfish), kingside vs queenside split
- en passant capture count per side
- average game length so we can sanity-check that low castling isn't just
  short games where neither side had time to castle
"""
import os
import sys
import time

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
N_GAMES_PER_COLOR = int(sys.argv[2]) if len(sys.argv) > 2 else 100
SF_DEPTH = int(sys.argv[3]) if len(sys.argv) > 3 else 5
SF_SKILL = int(sys.argv[4]) if len(sys.argv) > 4 else 10
TEMP = 0.5
DECAY = 0.05


def play_game(policy_engine, engine, model_color):
    board = chess.Board()
    info = {
        'model_castle': None,  # 'K' / 'Q' / None
        'model_ep': 0,
        'sf_castle': None,
        'sf_ep': 0,
        'plies': 0,
        'result': None,
    }
    while not board.is_game_over() and info['plies'] < 300:
        if board.turn == model_color:
            temp = compute_temperature(TEMP, DECAY, board.ply())
            stats = init_stats()
            mv = policy_engine.generate_move(board, stats, temperature=temp)
            if info['model_castle'] is None:
                if board.is_kingside_castling(mv):
                    info['model_castle'] = 'K'
                elif board.is_queenside_castling(mv):
                    info['model_castle'] = 'Q'
            if board.is_en_passant(mv):
                info['model_ep'] += 1
        else:
            mv = generate_engine_move(engine, board)
            if info['sf_castle'] is None:
                if board.is_kingside_castling(mv):
                    info['sf_castle'] = 'K'
                elif board.is_queenside_castling(mv):
                    info['sf_castle'] = 'Q'
            if board.is_en_passant(mv):
                info['sf_ep'] += 1
        board.push(mv)
        info['plies'] += 1
    info['result'] = board.result() if board.is_game_over() else '*'
    return info


def summarize(label, games):
    n = len(games)
    if n == 0:
        return
    mc = sum(1 for r in games if r['model_castle'])
    mk = sum(1 for r in games if r['model_castle'] == 'K')
    mq = sum(1 for r in games if r['model_castle'] == 'Q')
    mep_games = sum(1 for r in games if r['model_ep'] > 0)
    mep_total = sum(r['model_ep'] for r in games)

    sc = sum(1 for r in games if r['sf_castle'])
    sk = sum(1 for r in games if r['sf_castle'] == 'K')
    sq = sum(1 for r in games if r['sf_castle'] == 'Q')
    sep_games = sum(1 for r in games if r['sf_ep'] > 0)
    sep_total = sum(r['sf_ep'] for r in games)

    avg_plies = sum(r['plies'] for r in games) / n
    results = {'1-0': 0, '0-1': 0, '1/2-1/2': 0, '*': 0}
    for r in games:
        results[r['result']] = results.get(r['result'], 0) + 1

    print(f'\nModel as {label.upper()} ({n} games, avg {avg_plies:.0f} plies):')
    print(f'  Model castled: {mc}/{n} = {100*mc/n:.0f}%   '
          f'(O-O: {mk}={100*mk/n:.0f}%, O-O-O: {mq}={100*mq/n:.0f}%)')
    print(f'  Model en passant: {mep_games}/{n} games ({100*mep_games/n:.1f}%, '
          f'{mep_total} total captures)')
    print(f'  Stockfish castled: {sc}/{n} = {100*sc/n:.0f}%   '
          f'(O-O: {sk}={100*sk/n:.0f}%, O-O-O: {sq}={100*sq/n:.0f}%)')
    print(f'  Stockfish en passant: {sep_games}/{n} games ({100*sep_games/n:.1f}%, '
          f'{sep_total} total captures)')
    print(f'  Results: {results}')


def main():
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'model_e{EPOCH:04d}.pt')
    print(f'Loading {checkpoint_path}...', flush=True)
    policy_engine = load_policy_engine(checkpoint_path)
    engine = create_engine(path=STOCKFISH_PATH, depth=SF_DEPTH, skill_level=SF_SKILL)
    print(f'Stockfish: depth={SF_DEPTH}, skill={SF_SKILL}, t={TEMP}, decay={DECAY}\n',
          flush=True)

    by_color = {'white': [], 'black': []}
    t0 = time.time()
    try:
        for color, label in [(chess.WHITE, 'white'), (chess.BLACK, 'black')]:
            for i in range(N_GAMES_PER_COLOR):
                r = play_game(policy_engine, engine, color)
                by_color[label].append(r)
                if (i + 1) % 25 == 0:
                    print(f'  model {label}: {i+1}/{N_GAMES_PER_COLOR} '
                          f'({time.time()-t0:.1f}s)', flush=True)
    finally:
        engine.quit()

    summarize('white', by_color['white'])
    summarize('black', by_color['black'])


if __name__ == '__main__':
    main()

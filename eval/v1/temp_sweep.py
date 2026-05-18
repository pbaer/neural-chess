# -*- coding: utf-8 -*-
"""Sweep (temperature, decay) configs on the latest checkpoint to find a balance
between opening diversity and win-rate cost vs greedy.

Tests each config on easy (d=1, s=0) and med (d=3, s=5) Stockfish at 50 games
per setting (white side), plus an opening-diversity probe (100 samples of the
first move from the starting position).

Usage: python eval/v1/temp_sweep.py [epoch_number]
"""
import os
import re
import subprocess
import sys
import time
from collections import Counter

# Allow running from any cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import chess  # noqa: E402

from src.inference_api import load_policy_engine  # noqa: E402
from src.stats import init_stats  # noqa: E402

PYTHON = os.path.join(_REPO_ROOT, '.venv', 'Scripts', 'python.exe')
PLAY_SCRIPT = os.path.join(_REPO_ROOT, 'play.py')
CHECKPOINT_DIR = os.path.join(_REPO_ROOT, 'model', 'v1', 'checkpoints')
EPOCH = int(sys.argv[1]) if len(sys.argv) > 1 else 3

CONFIGS = [
    # (label, temp_start, temp_decay)
    ('greedy',    0.0,  0.0),
    ('t0.2 d0.05', 0.2, 0.05),
    ('t0.3 d0.05', 0.3, 0.05),
    ('t0.5 d0.05', 0.5, 0.05),  # CLI default
    ('t0.5 d0.10', 0.5, 0.10),
    ('t1.0 d0.05', 1.0, 0.05),
]


def first_move_diversity(policy_engine, temp_start, n=100):
    """Sample n first-moves from the starting position; return (unique_count, top3)."""
    moves = []
    for _ in range(n):
        board = chess.Board()
        stats = init_stats()
        mv = policy_engine.generate_move(board, stats, temperature=temp_start)
        moves.append(mv.uci())
    counter = Counter(moves)
    top3 = counter.most_common(3)
    return len(counter), top3


def run_subprocess_eval(epoch, temp, decay, depth, skill, color, games):
    checkpoint = os.path.join(CHECKPOINT_DIR, f'model_e{epoch:04d}.pt')
    cmd = [
        PYTHON, PLAY_SCRIPT, checkpoint, 'engine',
        '-n', str(games), '-d', str(depth), '-s', str(skill),
        '--color', color, '-t', str(temp), '--temp-decay', str(decay),
    ]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                       cwd=_REPO_ROOT)
    elapsed = time.time() - t0
    m = re.search(r'(\d+) games \(([\d.]+)% won, ([\d.]+)% draw, ([\d.]+)% lost\)',
                  r.stdout)
    if not m:
        return None
    return {
        'won': float(m.group(2)),
        'draw': float(m.group(3)),
        'lost': float(m.group(4)),
        'elapsed_s': elapsed,
    }


def run_eval_both_colors(epoch, temp, decay, depth, skill, games=50):
    """50/50 white/black split."""
    half = games // 2
    other = games - half
    w = run_subprocess_eval(epoch, temp, decay, depth, skill, 'white', half)
    b = run_subprocess_eval(epoch, temp, decay, depth, skill, 'black', other)
    if w is None or b is None:
        return None
    won = (w['won'] / 100) * half + (b['won'] / 100) * other
    draw = (w['draw'] / 100) * half + (b['draw'] / 100) * other
    lost = (w['lost'] / 100) * half + (b['lost'] / 100) * other
    return {
        'won': 100 * won / games,
        'draw': 100 * draw / games,
        'lost': 100 * lost / games,
        'elapsed_s': w['elapsed_s'] + b['elapsed_s'],
    }


def main():
    checkpoint = os.path.join(CHECKPOINT_DIR, f'model_e{EPOCH:04d}.pt')
    print(f'Loading {checkpoint}...', flush=True)
    policy_engine = load_policy_engine(checkpoint)

    print(f'\n{"config":<12} {"unique":<7} {"top-3 first moves":<40} '
          f'{"easy (d=1, s=0)":<24} {"med (d=3, s=5)":<24} {"time":<7}')
    print('-' * 116)

    for label, temp, decay in CONFIGS:
        unique, top3 = first_move_diversity(policy_engine, temp, n=100)
        top3_str = ' '.join(f'{mv}:{ct}' for mv, ct in top3)

        easy = run_eval_both_colors(EPOCH, temp, decay, 1, 0, 50)
        med = run_eval_both_colors(EPOCH, temp, decay, 3, 5, 50)

        if easy and med:
            print(f'{label:<12} {unique:<7} {top3_str:<40} '
                  f'W:{easy["won"]:.0f} D:{easy["draw"]:.0f} L:{easy["lost"]:.0f}'
                  f'{"":<14}'
                  f'W:{med["won"]:.0f} D:{med["draw"]:.0f} L:{med["lost"]:.0f}'
                  f'{"":<14}'
                  f'{easy["elapsed_s"]+med["elapsed_s"]:.0f}s',
                  flush=True)
        else:
            print(f'{label:<12} {unique:<7} {top3_str:<40} ERROR', flush=True)


if __name__ == '__main__':
    main()

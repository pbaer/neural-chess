# -*- coding: utf-8 -*-
"""How do our models fare against a uniform-random-move opponent?

The interesting question isn't whether we win (we should dominate) but whether
weak models fail to *convert* — i.e. reach a winning position but draw by the
50-move rule / threefold repetition because they can't force mate. That tests
finishing/endgame ability, not opening/middlegame strength.

RandomPolicyEngine: matches the user's spec — pick a piece uniformly among
pieces that have at least one legal move, then pick uniformly among that
piece's legal moves. (Under check, python-chess's legal_moves is already
restricted to check-resolving moves, so the sample space is correct.)
"""
import glob
import os
import random
import sys

import chess

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch
from src.inference_api import load_policy_engine
from src.game_loop import play_models
from src.random_engine import RandomPolicyEngine


def run(model_path, games=50, max_plies=400, seed=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_policy_engine(model_path, device=device)
    rnd = RandomPolicyEngine(seed=seed)

    half = games // 2
    other = games - half
    # Model as white
    w = play_models(model, rnd, limit=half, a_color=chess.WHITE,
                    temperature=0.0, max_plies=max_plies)
    # Model as black
    b = play_models(model, rnd, limit=other, a_color=chess.BLACK,
                    temperature=0.0, max_plies=max_plies)

    wins = w['results']['1-0'] + b['results']['0-1']
    draws = w['results']['1/2-1/2'] + b['results']['1/2-1/2']
    losses = w['results']['0-1'] + b['results']['1-0']
    total = wins + draws + losses
    avg_plies = (w['turns'] + b['turns']) / max(total, 1)
    return {
        'won': 100 * wins / total, 'draw': 100 * draws / total,
        'lost': 100 * losses / total, 'wins': wins, 'draws': draws,
        'losses': losses, 'avg_plies': avg_plies,
    }


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    targets = [
        ('v1-best', 'model/v1/checkpoints/model_e0009.pt'),
        ('v2-2M', 'model/v2/v2-2M/model_e0009.pt'),
        ('v2-3M', 'model/v2/v2-3M/model_e0009.pt'),
        ('v2-5M-b', 'model/v2/v2-5M-b/model_e0009.pt'),
        ('v2-8M-b', 'model/v2/v2-8M-b/model_e0009.pt'),
        ('v2-12M', 'model/v2/v2-12M/model_e0007.pt'),
        ('v2-14M', 'model/v2/v2-14M/model_e0007.pt'),
        ('v2-19M', 'model/v2/v2-19M/model_e0009.pt'),
        ('v2-37M', None),  # resolve latest below
    ]
    print(f'Models vs uniform-random opponent ({games} games each, 50/50 colors, greedy)\n')
    print(f'{"model":12s}  {"W/D/L":>14s}  {"win%":>6s}  {"draw%":>6s}  {"avg plies/game":>14s}')
    for name, path in targets:
        if path is None:
            pts = sorted(glob.glob(os.path.join(_REPO, 'model', 'v2', name, '*.pt')))
            if not pts:
                continue
            path = pts[-1]
        if not os.path.exists(os.path.join(_REPO, path)) and not os.path.exists(path):
            print(f'{name:12s}  (checkpoint not found: {path})')
            continue
        try:
            r = run(path, games=games)
            wdl = f"{r['wins']}/{r['draws']}/{r['losses']}"
            print(f'{name:12s}  {wdl:>14s}  {r["won"]:>5.1f}%  {r["draw"]:>5.1f}%  {r["avg_plies"]:>14.1f}')
        except Exception as e:
            print(f'{name:12s}  ERROR: {str(e)[:50]}')


if __name__ == '__main__':
    main()

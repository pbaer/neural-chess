# -*- coding: utf-8 -*-
"""Sweep MCTS settings on a v2 checkpoint to find the strongest play within a
latency budget. Compares raw single-shot policy vs PUCT MCTS at various sim
counts / c_puct, against the SF ladder and head-to-head vs raw.

Latency budget: SF-ultra measured ~440ms/move max. Sequential MCTS is ~6ms/sim
uncontended, so ~70 sims fits the budget. (Running alongside training inflates
wall-clock ~2x, but we use fixed sim counts so the *settings* stay reproducible.)

Usage:
    python eval/v2/mcts_sweep.py --ckpt model/v2/v2-19M/model_e0009.pt
"""
import argparse
import csv
import os
import sys
import time

import chess

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch
from src.inference_api import load_policy_engine
from src.engine import create_engine
from src.game_loop import play_engine, play_models
from src.mcts import MCTSEngine
from src.stats import model_record

SF_TIERS = {
    'sf_med':    (3, 5),
    'sf_hard':   (5, 10),
    'sf_magnus': (8, 20),
    'sf_ultra':  (16, 20),
}


def _per_move_ms(stats):
    """Avg MCTS-side ms per move from a play_engine/play_models stats dict."""
    model_moves = max(stats['turns'] // 2, 1)
    return stats['model_minutes_elapsed'] * 60_000 / model_moves


def eval_vs_sf(mcts_engine, tier, games):
    depth, skill = SF_TIERS[tier]
    eng = create_engine(path='bin/stockfish.exe', depth=depth, skill_level=skill)
    try:
        half = games // 2
        other = games - half
        w = play_engine(mcts_engine, eng, limit=half, model_color=chess.WHITE,
                        temperature=0.0, temp_decay=0.0)
        b = play_engine(mcts_engine, eng, limit=other, model_color=chess.BLACK,
                        temperature=0.0, temp_decay=0.0)
    finally:
        eng.quit()
    ww, wd, wl = model_record(w, chess.WHITE)
    bw, bd, bl = model_record(b, chess.BLACK)
    wins, draws, losses = ww + bw, wd + bd, wl + bl
    total = wins + draws + losses
    ms = _per_move_ms({'turns': w['turns'] + b['turns'],
                       'model_minutes_elapsed': w['model_minutes_elapsed'] + b['model_minutes_elapsed']})
    return {'won': 100*wins/total, 'draw': 100*draws/total, 'lost': 100*losses/total,
            'wins': wins, 'draws': draws, 'losses': losses, 'ms_per_move': ms}


def eval_vs_raw(mcts_engine, raw_engine, games):
    half = games // 2
    other = games - half
    w = play_models(mcts_engine, raw_engine, limit=half, a_color=chess.WHITE, temperature=0.0)
    b = play_models(mcts_engine, raw_engine, limit=other, a_color=chess.BLACK, temperature=0.0)
    wins = w['results']['1-0'] + b['results']['0-1']
    draws = w['results']['1/2-1/2'] + b['results']['1/2-1/2']
    losses = w['results']['0-1'] + b['results']['1-0']
    total = wins + draws + losses
    ms = _per_move_ms({'turns': w['turns'] + b['turns'],
                       'model_minutes_elapsed': w['model_minutes_elapsed'] + b['model_minutes_elapsed']})
    return {'won': 100*wins/total, 'draw': 100*draws/total, 'lost': 100*losses/total,
            'wins': wins, 'draws': draws, 'losses': losses, 'ms_per_move': ms}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='model/v2/v2-19M/model_e0009.pt')
    ap.add_argument('--csv', default='eval/v2/eval_mcts_sweep.csv')
    ap.add_argument('--sf-games', type=int, default=40)
    ap.add_argument('--raw-games', type=int, default=40)
    # sweep grid
    ap.add_argument('--sims', type=int, nargs='+', default=[25, 50, 70, 100])
    ap.add_argument('--cpuct', type=float, nargs='+', default=[1.5])
    ap.add_argument('--tiers', nargs='+', default=['sf_hard', 'sf_ultra'])
    ap.add_argument('--skip-raw', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'MCTS sweep on {args.ckpt}\n')
    raw = load_policy_engine(args.ckpt, device=device)

    write_header = not os.path.exists(args.csv)
    f = open(args.csv, 'a', newline='')
    wr = csv.writer(f)
    if write_header:
        wr.writerow(['ckpt', 'mode', 'sims', 'cpuct', 'opponent', 'games',
                     'won_pct', 'draw_pct', 'lost_pct', 'wdl', 'ms_per_move'])

    def log(mode, sims, cpuct, opp, games, r):
        wdl = f"{r['wins']}/{r['draws']}/{r['losses']}"
        print(f'  {mode:5s} sims={sims or "-":>4} cpuct={cpuct or "-":>3}  {opp:9s}  '
              f'W:{r["won"]:.0f}% D:{r["draw"]:.0f}% L:{r["lost"]:.0f}%  '
              f'({wdl})  {r["ms_per_move"]:.0f}ms/move', flush=True)
        wr.writerow([os.path.basename(os.path.dirname(args.ckpt)), mode, sims or '', cpuct or '',
                     opp, games, f"{r['won']:.1f}", f"{r['draw']:.1f}", f"{r['lost']:.1f}",
                     wdl, f"{r['ms_per_move']:.0f}"])
        f.flush()

    for cpuct in args.cpuct:
        for sims in args.sims:
            print(f'--- MCTS sims={sims} c_puct={cpuct} ---', flush=True)
            mcts = MCTSEngine(raw, c_puct=cpuct, n_simulations=sims)
            if not args.skip_raw:
                r = eval_vs_raw(mcts, raw, args.raw_games)
                log('mcts', sims, cpuct, 'raw_self', args.raw_games, r)
            for tier in args.tiers:
                r = eval_vs_sf(mcts, tier, args.sf_games)
                log('mcts', sims, cpuct, tier, args.sf_games, r)

    f.close()
    print('\nSWEEP DONE', flush=True)


if __name__ == '__main__':
    main()

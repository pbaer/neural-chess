# -*- coding: utf-8 -*-
"""v2 per-epoch eval: run greedy vs Stockfish on the gating ladder, AND
greedy vs the strongest v1 checkpoint (head-to-head).

The Stockfish ladder mirrors v1's eval_one.py exactly (easy/med/hard/magnus,
gated by ≤10% loss at the prior tier). The new v1-h2h section runs N games
50/50 white/black against the configured v1 checkpoint.

Output CSV is parallel to v1's eval_results.csv schema but with one extra
'opponent' column to distinguish 'sf_easy', 'sf_med', 'sf_hard', 'sf_magnus',
and 'v1_e0009' (or whatever v1 best is configured to).

Usage:
    python eval/v2/eval_v2.py <epoch> --save-dir <v2 model dir> --eval-csv <out csv>
"""
import argparse
import csv
import datetime
import os
import re
import subprocess
import sys
import time

# Allow direct invocation
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


PYTHON = os.path.abspath(os.path.join(_REPO_ROOT, '.venv/Scripts/python.exe'))
DEFAULT_V1_BEST = 'model/v1/checkpoints/model_e0009.pt'   # the v1 best checkpoint per training_summary.md
GAMES = 100
MAGNUS_GAMES = 10
GATE_PCT = 10.0


def parse_play_output(stdout: str):
    moves = re.search(r'(\d+) model moves \(([\d.]+)% first-pick illegal\)', stdout)
    games = re.search(
        r'(\d+) games \(([\d.]+)% won, ([\d.]+)% draw, ([\d.]+)% lost\)', stdout)
    if games is None:
        return None
    return {
        'moves': int(moves.group(1)) if moves else 0,
        'illegal_pct': float(moves.group(2)) if moves else 0.0,
        'won': float(games.group(2)),
        'draw': float(games.group(3)),
        'lost': float(games.group(4)),
    }


def _run_play_one_color(model_path: str, depth: int, skill: int, games: int,
                        color: str, temp: float = 0.0, decay: float = 0.0):
    """Run play.py engine subcommand for one color, return parsed stats."""
    cmd = [
        PYTHON, 'play.py', model_path, 'engine',
        '-n', str(games), '-d', str(depth), '-s', str(skill),
        '--color', color, '-t', str(temp), '--temp-decay', str(decay),
    ]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, cwd=_REPO_ROOT)
    elapsed = time.time() - t0
    stats = parse_play_output(r.stdout)
    if stats is None:
        return {'error': r.stderr[:500] or r.stdout[-500:], 'elapsed_s': elapsed}
    stats['elapsed_s'] = elapsed
    return stats


def run_sf_eval(model_path: str, depth: int, skill: int, games: int):
    """50/50 white/black split vs Stockfish at given depth/skill."""
    half = games // 2
    other = games - half
    w = _run_play_one_color(model_path, depth, skill, half, 'white')
    if 'error' in w:
        return w
    b = _run_play_one_color(model_path, depth, skill, other, 'black')
    if 'error' in b:
        return b
    won = (w['won']/100)*half + (b['won']/100)*other
    draw = (w['draw']/100)*half + (b['draw']/100)*other
    lost = (w['lost']/100)*half + (b['lost']/100)*other
    moves = w['moves'] + b['moves']
    illegal_count = (w['illegal_pct']/100)*w['moves'] + (b['illegal_pct']/100)*b['moves']
    return {
        'moves': moves,
        'illegal_pct': 100 * illegal_count / max(moves, 1),
        'won': 100 * won / games,
        'draw': 100 * draw / games,
        'lost': 100 * lost / games,
        'white_won': w['won'],
        'black_won': b['won'],
        'elapsed_s': w['elapsed_s'] + b['elapsed_s'],
    }


def run_h2h_eval(v2_path: str, opponent_path: str, games: int = 100, temp: float = 0.0):
    """v2 head-to-head against another PolicyEngine checkpoint (e.g. v1 best).
    50/50 color split. Stats are from v2's perspective.
    """
    # Use a small Python script to drive play_models (avoids subprocess overhead
    # per game). Import once, play both colors.
    import chess
    from src.inference_api import load_policy_engine
    from src.game_loop import play_models

    e_v2 = load_policy_engine(v2_path)
    e_opp = load_policy_engine(opponent_path)

    half = games // 2
    other = games - half

    t0 = time.time()
    w = play_models(e_v2, e_opp, limit=half, a_color=chess.WHITE,
                    temperature=temp, temp_decay=0.0)
    b = play_models(e_v2, e_opp, limit=other, a_color=chess.BLACK,
                    temperature=temp, temp_decay=0.0)
    elapsed = time.time() - t0

    # Aggregate from v2's perspective
    # play_models returns 'results' dict keyed '1-0'/'0-1'/'1/2-1/2' from white's view
    # When v2 was white (w), '1-0' = v2 win
    # When v2 was black (b), '0-1' = v2 win
    v2_wins = w['results']['1-0'] + b['results']['0-1']
    v2_draws = w['results']['1/2-1/2'] + b['results']['1/2-1/2']
    v2_losses = w['results']['0-1'] + b['results']['1-0']
    total = v2_wins + v2_draws + v2_losses

    # White / black split (v2's win-rate as each color)
    white_won = 100 * w['results']['1-0'] / max(half, 1)
    black_won = 100 * b['results']['0-1'] / max(other, 1)

    return {
        'moves': w['legal_moves'] + b['legal_moves'],
        'illegal_pct': 100 * (w['illegal_moves'] + b['illegal_moves'])
                       / max(w['legal_moves'] + b['legal_moves'], 1),
        'won': 100 * v2_wins / max(total, 1),
        'draw': 100 * v2_draws / max(total, 1),
        'lost': 100 * v2_losses / max(total, 1),
        'white_won': white_won,
        'black_won': black_won,
        'elapsed_s': elapsed,
    }


def run_random_eval(v2_path: str, games: int = 100):
    """v2 vs a uniform-random-move opponent. 50/50 color split.
    The headline metrics are win% (want 100%) and avg plies/game (finishing
    speed — lower is better). Draws here are conversion failures."""
    import chess
    from src.inference_api import load_policy_engine
    from src.game_loop import play_models
    from src.random_engine import RandomPolicyEngine

    e_v2 = load_policy_engine(v2_path)
    rnd = RandomPolicyEngine(seed=0)

    half = games // 2
    other = games - half

    t0 = time.time()
    w = play_models(e_v2, rnd, limit=half, a_color=chess.WHITE,
                    temperature=0.0, temp_decay=0.0, max_plies=400)
    b = play_models(e_v2, rnd, limit=other, a_color=chess.BLACK,
                    temperature=0.0, temp_decay=0.0, max_plies=400)
    elapsed = time.time() - t0

    v2_wins = w['results']['1-0'] + b['results']['0-1']
    v2_draws = w['results']['1/2-1/2'] + b['results']['1/2-1/2']
    v2_losses = w['results']['0-1'] + b['results']['1-0']
    total = v2_wins + v2_draws + v2_losses
    total_plies = w['turns'] + b['turns']

    return {
        'moves': total_plies,   # total plies; avg plies/game = moves / games
        'illegal_pct': 0.0,
        'won': 100 * v2_wins / max(total, 1),
        'draw': 100 * v2_draws / max(total, 1),
        'lost': 100 * v2_losses / max(total, 1),
        'white_won': 100 * w['results']['1-0'] / max(half, 1),
        'black_won': 100 * b['results']['0-1'] / max(other, 1),
        'elapsed_s': elapsed,
    }


def _most_recent_lost(csv_path: str, opponent: str):
    """Most recent lost_pct for the given opponent label, or None."""
    if not os.path.exists(csv_path):
        return None
    last = None
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('opponent') == opponent and row.get('lost_pct'):
                try:
                    last = float(row['lost_pct'])
                except ValueError:
                    pass
    return last


SF_LADDER = [
    # (opponent label, depth, skill, games, gate-from)
    ('sf_easy',   1,  0,  GAMES,        None),
    ('sf_med',    3,  5,  GAMES,        'sf_easy'),
    ('sf_hard',   5, 10,  GAMES,        'sf_med'),
    ('sf_magnus', 8, 20, MAGNUS_GAMES, 'sf_hard'),    # redefined 2026-05-17: was d=15 s=20
    ('sf_ultra', 16, 20, MAGNUS_GAMES, 'sf_magnus'),  # added 2026-05-17
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('epoch', type=int, help='epoch number to evaluate')
    parser.add_argument('--save-dir', default='model/v2',
                        help='v2 checkpoint directory')
    parser.add_argument('--save-name', default='model',
                        help='v2 checkpoint name prefix')
    parser.add_argument('--eval-csv', default='eval/v2/eval_results.csv',
                        help='output CSV')
    parser.add_argument('--v1-best', default=DEFAULT_V1_BEST,
                        help='path to v1 best checkpoint for head-to-head')
    parser.add_argument('--h2h-games', type=int, default=100,
                        help='games for v1 head-to-head')
    parser.add_argument('--skip-h2h', action='store_true')
    parser.add_argument('--skip-sf', action='store_true')
    parser.add_argument('--skip-random', action='store_true',
                        help='Skip the uniform-random-mover baseline')
    parser.add_argument('--random-games', type=int, default=100,
                        help='games for the random-mover baseline')
    parser.add_argument('--no-gate', action='store_true',
                        help='Run all SF tiers regardless of prior loss-rate gating')
    args = parser.parse_args()

    v2_path = os.path.join(args.save_dir, f'{args.save_name}_e{args.epoch:04d}.pt')
    if not os.path.exists(v2_path):
        print(f"ERROR: v2 checkpoint not found: {v2_path}")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.eval_csv) or '.', exist_ok=True)
    write_header = not os.path.exists(args.eval_csv)
    with open(args.eval_csv, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['epoch', 'timestamp', 'opponent', 'games',
                        'won_pct', 'draw_pct', 'lost_pct',
                        'white_won_pct', 'black_won_pct',
                        'illegal_pct', 'moves', 'elapsed_s'])

        # Stockfish ladder
        if not args.skip_sf:
            for opp, depth, skill, games, gate in SF_LADDER:
                if gate is not None and not args.no_gate:
                    g_loss = _most_recent_lost(args.eval_csv, gate)
                    if g_loss is None or g_loss > GATE_PCT:
                        desc = (f'>{GATE_PCT:.0f}% ({g_loss:.0f}%)'
                                if g_loss is not None else 'no prior data')
                        print(f'SKIP {opp}: gate {gate!r} {desc}', flush=True)
                        continue
                ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f'EVAL {ts} epoch={args.epoch} opp={opp} '
                      f'depth={depth} skill={skill} games={games}', flush=True)
                r = run_sf_eval(v2_path, depth, skill, games)
                if 'error' in r:
                    print(f'  ERROR: {r["error"][:200]}', flush=True)
                    w.writerow([args.epoch, ts, opp, games,
                                '', '', '', '', '', '', '', f'{r["elapsed_s"]:.1f}'])
                else:
                    print(f'  W:{r["won"]:.0f}% D:{r["draw"]:.0f}% L:{r["lost"]:.0f}% '
                          f'(W:{r["white_won"]:.0f}% / B:{r["black_won"]:.0f}%) '
                          f'illegal:{r["illegal_pct"]:.1f}% '
                          f'({r["elapsed_s"]:.1f}s)', flush=True)
                    w.writerow([args.epoch, ts, opp, games,
                                r['won'], r['draw'], r['lost'],
                                r['white_won'], r['black_won'],
                                r['illegal_pct'], r['moves'],
                                f'{r["elapsed_s"]:.1f}'])
                f.flush()

        # Random-mover baseline (want 100% wins + low avg plies/game)
        if not args.skip_random:
            ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f'EVAL {ts} epoch={args.epoch} opp=random '
                  f'(uniform random, {args.random_games} games)', flush=True)
            r = run_random_eval(v2_path, args.random_games)
            if 'error' in r:
                print(f'  ERROR: {r["error"][:200]}', flush=True)
                w.writerow([args.epoch, ts, 'random', args.random_games,
                            '', '', '', '', '', '', '', f'{r["elapsed_s"]:.1f}'])
            else:
                avg_plies = r['moves'] / max(args.random_games, 1)
                print(f'  W:{r["won"]:.0f}% D:{r["draw"]:.0f}% L:{r["lost"]:.0f}% '
                      f'avg {avg_plies:.1f} plies/game '
                      f'({r["elapsed_s"]:.1f}s)', flush=True)
                w.writerow([args.epoch, ts, 'random', args.random_games,
                            r['won'], r['draw'], r['lost'],
                            r['white_won'], r['black_won'],
                            r['illegal_pct'], r['moves'],
                            f'{r["elapsed_s"]:.1f}'])
            f.flush()

        # v1 head-to-head
        if not args.skip_h2h and os.path.exists(args.v1_best):
            v1_label = 'v1_' + os.path.splitext(os.path.basename(args.v1_best))[0]
            ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f'EVAL {ts} epoch={args.epoch} opp={v1_label} '
                  f'(v1 best, {args.h2h_games} games)', flush=True)
            r = run_h2h_eval(v2_path, args.v1_best, args.h2h_games)
            if 'error' in r:
                print(f'  ERROR: {r["error"][:200]}', flush=True)
                w.writerow([args.epoch, ts, v1_label, args.h2h_games,
                            '', '', '', '', '', '', '', f'{r["elapsed_s"]:.1f}'])
            else:
                print(f'  W:{r["won"]:.0f}% D:{r["draw"]:.0f}% L:{r["lost"]:.0f}% '
                      f'(W:{r["white_won"]:.0f}% / B:{r["black_won"]:.0f}%) '
                      f'illegal:{r["illegal_pct"]:.1f}% '
                      f'({r["elapsed_s"]:.1f}s)', flush=True)
                w.writerow([args.epoch, ts, v1_label, args.h2h_games,
                            r['won'], r['draw'], r['lost'],
                            r['white_won'], r['black_won'],
                            r['illegal_pct'], r['moves'],
                            f'{r["elapsed_s"]:.1f}'])
            f.flush()

    print(f'EVAL_DONE epoch={args.epoch}', flush=True)


if __name__ == '__main__':
    main()

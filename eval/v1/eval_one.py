# -*- coding: utf-8 -*-
"""Evaluate one trained v1 checkpoint against Stockfish at a few difficulty settings.

Called by the long-running training session as each epoch is saved. Appends a
row per (epoch, setting) to eval_results.csv (sitting alongside this script in
eval/v1/). The difficulty schedule ramps with epoch so early epochs don't burn
time playing hopeless games against hard Stockfish.

Usage: python eval/v1/eval_one.py <epoch_number>
"""
import csv
import datetime
import os
import re
import subprocess
import sys
import time

# Resolve repo-relative paths so this can be invoked from any cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
EVAL_CSV = os.path.join(_HERE, 'eval_results.csv')
PYTHON = os.path.join(_REPO_ROOT, '.venv', 'Scripts', 'python.exe')
PLAY_SCRIPT = os.path.join(_REPO_ROOT, 'play.py')
CHECKPOINT_DIR = os.path.join(_REPO_ROOT, 'model', 'v1', 'checkpoints')


def parse_play_output(stdout):
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


def _run_one_color(checkpoint_path, depth, skill, games, color, temp, decay):
    cmd = [
        PYTHON, PLAY_SCRIPT, checkpoint_path, 'engine',
        '-n', str(games), '-d', str(depth), '-s', str(skill),
        '--color', color,
        '-t', str(temp), '--temp-decay', str(decay),
    ]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                       cwd=_REPO_ROOT)
    elapsed = time.time() - t0
    stats = parse_play_output(r.stdout)
    if stats is None:
        return {'error': r.stderr[:500] or r.stdout[-500:], 'elapsed_s': elapsed}
    stats['elapsed_s'] = elapsed
    stats['games_run'] = games
    return stats


def run_eval(checkpoint_path, depth, skill, games, temp, decay):
    """Play `games` total, split evenly between model-as-white and model-as-black.

    Returns aggregate stats over both colors. Stockfish has slight color
    asymmetries (white moves first) and the trained model may too, so 50/50
    averaging gives a less biased estimate of overall strength than playing
    one color only.
    """
    half = games // 2
    other = games - half  # handles odd `games` correctly

    white = _run_one_color(checkpoint_path, depth, skill, half, 'white', temp, decay)
    if 'error' in white:
        return white
    black = _run_one_color(checkpoint_path, depth, skill, other, 'black', temp, decay)
    if 'error' in black:
        return black

    # Weight the two color runs by their game counts and combine.
    total_games = half + other
    won = (white['won'] / 100) * half + (black['won'] / 100) * other
    draw = (white['draw'] / 100) * half + (black['draw'] / 100) * other
    lost = (white['lost'] / 100) * half + (black['lost'] / 100) * other
    moves = white['moves'] + black['moves']
    illegal_count = ((white['illegal_pct'] / 100) * white['moves']
                     + (black['illegal_pct'] / 100) * black['moves'])

    return {
        'moves': moves,
        'illegal_pct': 100 * illegal_count / max(moves, 1),
        'won': 100 * won / total_games,
        'draw': 100 * draw / total_games,
        'lost': 100 * lost / total_games,
        'elapsed_s': white['elapsed_s'] + black['elapsed_s'],
        'white_won': white['won'],
        'black_won': black['won'],
    }


def _most_recent_lose_pct(label, mode):
    """Return the most recent lost_pct from EVAL_CSV for (label, mode), or None."""
    if not os.path.exists(EVAL_CSV):
        return None
    last = None
    with open(EVAL_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get('label') == label and row.get('mode') == mode
                    and row.get('lost_pct')):
                try:
                    last = float(row['lost_pct'])
                except ValueError:
                    pass
    return last


# (mode_name, temp_start, temp_decay) — 'greedy' measures top-1 strength;
# 'realistic' matches the play.py / Arena defaults so the eval reflects how
# the model would play in actual interactive games.
MODES = [
    ('greedy',    0.0, 0.0),
    ('realistic', 0.5, 0.05),
]

# Each ladder rung gates on the immediately-easier rung in the *same mode*
# having a recent ≤10% loss rate, so each mode advances independently.
# "magnus" is realistic-only: at Stockfish skill=20 the opponent is
# essentially deterministic, so greedy+magnus would mostly play one or two
# repeated lines and give us no measurement diversity. Temperature on our
# side is what makes it possible to sample distinct games at that ceiling.
#
# "magnus" itself is an approximation: Stockfish 8 (the bundled binary)
# predates UCI_LimitStrength/UCI_Elo, so we can't target 2830 Elo directly.
# Skill 20 + depth 15 gives a strong-but-not-superhuman ceiling — a rough
# surrogate for Magnus-level strength, not a calibrated match.
GAMES = 100
LADDER = [
    # (label, depth, skill, games, gate_label, modes_running_this_rung)
    ('easy',   1,  0,  GAMES, None,   ['greedy', 'realistic']),
    ('med',    3,  5,  GAMES, 'easy', ['greedy', 'realistic']),
    ('hard',   5, 10,  GAMES, 'med',  ['greedy', 'realistic']),
    ('magnus', 15, 20, 10,    'hard', ['realistic']),
]
GATE_PCT = 10.0


def main(epoch):
    write_header = not os.path.exists(EVAL_CSV)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'model_e{epoch:04d}.pt')
    with open(EVAL_CSV, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['epoch', 'timestamp', 'mode', 'label', 'depth', 'skill',
                        'won_pct', 'draw_pct', 'lost_pct',
                        'white_won_pct', 'black_won_pct',
                        'illegal_pct', 'moves', 'elapsed_s'])

        # Iterate modes as the outer loop so each mode's ladder advances
        # together — easy/med/hard within a mode get logged consecutively.
        for mode, temp, decay in MODES:
            for label, depth, skill, games, gate, modes_for_rung in LADDER:
                if mode not in modes_for_rung:
                    continue
                if gate is not None:
                    gate_loss = _most_recent_lose_pct(gate, mode)
                    if gate_loss is None or gate_loss > GATE_PCT:
                        desc = (f'>{GATE_PCT:.0f}% loss ({gate_loss:.0f}%)'
                                if gate_loss is not None else 'no prior data')
                        print(f'SKIP {mode}/{label}: gate {gate!r} has {desc}',
                              flush=True)
                        continue

                ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f'EVAL {ts} epoch={epoch} mode={mode} setting={label} '
                      f'depth={depth} skill={skill} games={games} '
                      f't={temp} decay={decay}', flush=True)
                r = run_eval(checkpoint_path, depth, skill,
                             games=games, temp=temp, decay=decay)
                if 'error' in r:
                    print(f'  ERROR: {r["error"][:200]}', flush=True)
                    w.writerow([epoch, ts, mode, label, depth, skill,
                                '', '', '', '', '', '', '',
                                f'{r["elapsed_s"]:.1f}'])
                else:
                    print(f'  W:{r["won"]:.0f}% D:{r["draw"]:.0f}% L:{r["lost"]:.0f}% '
                          f'(W:{r["white_won"]:.0f}% / B:{r["black_won"]:.0f}%) '
                          f'illegal:{r["illegal_pct"]:.1f}% '
                          f'({r["elapsed_s"]:.1f}s, {games} games)', flush=True)
                    w.writerow([epoch, ts, mode, label, depth, skill,
                                r['won'], r['draw'], r['lost'],
                                r['white_won'], r['black_won'],
                                r['illegal_pct'], r['moves'],
                                f'{r["elapsed_s"]:.1f}'])
                f.flush()
    print(f'EVAL_DONE epoch={epoch}', flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]))

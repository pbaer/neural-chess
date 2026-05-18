# -*- coding: utf-8 -*-
"""Evaluate all v1 checkpoints against Stockfish across temperature settings."""
import glob
import logging
import os
import re
import sys
import time

# Allow running as `python eval/v1/evaluate.py` from any cwd.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import chess  # noqa: E402
import torch  # noqa: E402

# Suppress noisy python-chess warnings about null moves
logging.getLogger('chess.engine').setLevel(logging.ERROR)

from src.engine import create_engine  # noqa: E402
from src.game_loop import play_engine  # noqa: E402
from src.inference_api import load_policy_engine  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STOCKFISH_PATH = os.path.join(_REPO_ROOT, 'bin', 'stockfish.exe')
STOCKFISH_DEPTH = 5
STOCKFISH_SKILL = 10
GAMES_PER_SETTING = 100
MODEL_DIR = os.path.join(_REPO_ROOT, 'model', 'v1', 'checkpoints')
MODEL_PREFIX = 'model'

TEMP_SETTINGS = [
    ('greedy',   0.0, 0.0),
    ('subtle',   0.3, 0.03),
    ('moderate', 0.5, 0.05),
    ('default',  1.0, 0.05),
    ('sharp',    1.0, 0.10),
]

# ---------------------------------------------------------------------------
# Discover checkpoints
# ---------------------------------------------------------------------------

def find_checkpoints(model_dir=MODEL_DIR, prefix=MODEL_PREFIX):
    pattern = os.path.join(model_dir, f'{prefix}_e*.pt')
    matches = glob.glob(pattern)
    epoch_re = re.compile(re.escape(prefix) + r'_e(\d+)\.pt$')
    results = []
    for path in matches:
        m = epoch_re.search(os.path.basename(path))
        if m:
            results.append((int(m.group(1)), path))
    return sorted(results)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(max_models=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    checkpoints = find_checkpoints()
    if not checkpoints:
        print(f"No checkpoints found in {MODEL_DIR}")
        return

    if max_models is not None and len(checkpoints) > max_models:
        # Pick evenly spaced checkpoints including first and last
        if max_models == 1:
            checkpoints = [checkpoints[-1]]
        else:
            indices = [round(i * (len(checkpoints) - 1) / (max_models - 1))
                       for i in range(max_models)]
            checkpoints = [checkpoints[i] for i in indices]

    print(f"Evaluating {len(checkpoints)} checkpoint(s): "
          + ", ".join(os.path.basename(p) for _, p in checkpoints))
    print(f"Stockfish: depth={STOCKFISH_DEPTH}, skill={STOCKFISH_SKILL}")
    print(f"Games per setting: {GAMES_PER_SETTING}")
    print(f"Temperature settings: {len(TEMP_SETTINGS)}")
    total_games = len(checkpoints) * len(TEMP_SETTINGS) * GAMES_PER_SETTING
    print(f"Total games: {total_games}")
    print()

    def start_engine():
        return create_engine(path=STOCKFISH_PATH, depth=STOCKFISH_DEPTH,
                             skill_level=STOCKFISH_SKILL)

    engine = start_engine()
    all_results = []

    try:
        for ckpt_idx, (epoch, path) in enumerate(checkpoints):
            model_name = os.path.basename(path).replace('.pt', '')
            policy_engine = load_policy_engine(path, device=device)
            print(f"{'='*70}")
            print(f"Model: {model_name} ({ckpt_idx+1}/{len(checkpoints)})")
            print(f"{'='*70}")

            for setting_idx, (label, temp, decay) in enumerate(TEMP_SETTINGS):
                t0 = time.time()
                progress = (ckpt_idx * len(TEMP_SETTINGS) + setting_idx)
                progress_total = len(checkpoints) * len(TEMP_SETTINGS)
                print(f"\n  [{progress+1}/{progress_total}] "
                      f"t={temp}, decay={decay} ({label})  ", end='', flush=True)

                try:
                    stats = play_engine(policy_engine, engine, GAMES_PER_SETTING,
                                        model_color=chess.WHITE,
                                        verbose=False,
                                        temperature=temp, temp_decay=decay)
                except Exception as e:
                    print(f"\n  ERROR: {e} — restarting engine and retrying")
                    try:
                        engine.quit()
                    except Exception:
                        pass
                    engine = start_engine()
                    stats = play_engine(policy_engine, engine, GAMES_PER_SETTING,
                                        model_color=chess.WHITE,
                                        verbose=False,
                                        temperature=temp, temp_decay=decay)

                wins = stats['results']['1-0']
                draws = stats['results']['1/2-1/2']
                losses = stats['results']['0-1']
                games = stats['games']
                elapsed = time.time() - t0
                illegal_pct = 100 * stats['illegal_moves'] / max(1, stats['legal_moves'])

                print(f"  W:{wins} D:{draws} L:{losses}  "
                      f"({100*wins/games:.0f}%/{100*draws/games:.0f}%/{100*losses/games:.0f}%)  "
                      f"illegal:{illegal_pct:.1f}%  {elapsed:.0f}s")

                all_results.append({
                    'model': model_name,
                    'epoch': epoch,
                    'label': label,
                    'temp': temp,
                    'decay': decay,
                    'wins': wins,
                    'draws': draws,
                    'losses': losses,
                    'games': games,
                    'illegal_pct': illegal_pct,
                    'seconds': elapsed,
                })

            del policy_engine
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    finally:
        try:
            engine.quit()
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Stockfish: depth={STOCKFISH_DEPTH}, skill={STOCKFISH_SKILL}, "
          f"{GAMES_PER_SETTING} games per setting\n")

    # Table header
    header = f"{'Model':<16} {'Setting':<10} {'Temp':>4} {'Decay':>5}  " \
             f"{'W':>3} {'D':>3} {'L':>3}  {'Win%':>5} {'Draw%':>5} {'Loss%':>5}  {'Illegal%':>8}"
    print(header)
    print('-' * len(header))

    for r in all_results:
        g = r['games']
        print(f"{r['model']:<16} {r['label']:<10} {r['temp']:>4.1f} {r['decay']:>5.2f}  "
              f"{r['wins']:>3} {r['draws']:>3} {r['losses']:>3}  "
              f"{100*r['wins']/g:>5.1f} {100*r['draws']/g:>5.1f} {100*r['losses']/g:>5.1f}  "
              f"{r['illegal_pct']:>7.1f}%")

    # Best overall
    print(f"\n{'='*70}")
    print("BEST SETTINGS (by win rate)")
    print(f"{'='*70}")
    sorted_results = sorted(all_results, key=lambda r: (r['wins'], r['draws']), reverse=True)
    for i, r in enumerate(sorted_results[:5]):
        g = r['games']
        print(f"  {i+1}. {r['model']} / {r['label']} (t={r['temp']}, d={r['decay']})  "
              f"W:{r['wins']} D:{r['draws']} L:{r['losses']}  "
              f"({100*r['wins']/g:.0f}%/{100*r['draws']/g:.0f}%/{100*r['losses']/g:.0f}%)")

    # Per-model best
    print(f"\n{'='*70}")
    print("BEST SETTING PER MODEL")
    print(f"{'='*70}")
    for epoch, path in checkpoints:
        model_name = os.path.basename(path).replace('.pt', '')
        model_results = [r for r in all_results if r['model'] == model_name]
        best = max(model_results, key=lambda r: (r['wins'], r['draws']))
        g = best['games']
        print(f"  {model_name}: {best['label']} (t={best['temp']}, d={best['decay']})  "
              f"W:{best['wins']} D:{best['draws']} L:{best['losses']}  "
              f"({100*best['wins']/g:.0f}%/{100*best['draws']/g:.0f}%/{100*best['losses']/g:.0f}%)")

    # Per-setting best
    print(f"\n{'='*70}")
    print("BEST MODEL PER SETTING")
    print(f"{'='*70}")
    for label, temp, decay in TEMP_SETTINGS:
        setting_results = [r for r in all_results if r['label'] == label]
        best = max(setting_results, key=lambda r: (r['wins'], r['draws']))
        g = best['games']
        print(f"  {label:<10} (t={temp}, d={decay}): {best['model']}  "
              f"W:{best['wins']} D:{best['draws']} L:{best['losses']}  "
              f"({100*best['wins']/g:.0f}%/{100*best['draws']/g:.0f}%/{100*best['losses']/g:.0f}%)")

    # Training progression
    print(f"\n{'='*70}")
    print("TRAINING PROGRESSION (greedy setting)")
    print(f"{'='*70}")
    for epoch, path in checkpoints:
        model_name = os.path.basename(path).replace('.pt', '')
        greedy = [r for r in all_results if r['model'] == model_name and r['label'] == 'greedy']
        if greedy:
            r = greedy[0]
            g = r['games']
            print(f"  Epoch {epoch}: W:{r['wins']} D:{r['draws']} L:{r['losses']}  "
                  f"({100*r['wins']/g:.0f}%/{100*r['draws']/g:.0f}%/{100*r['losses']/g:.0f}%)  "
                  f"illegal:{r['illegal_pct']:.1f}%")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate v1 model checkpoints vs Stockfish')
    parser.add_argument('--quick', action='store_true',
                        help='Quick sanity test: 1 model, all temp settings, 5 games each')
    parser.add_argument('--games', type=int, default=None,
                        help='Override games per setting')
    parser.add_argument('--models', type=int, default=None,
                        help='Max number of models to evaluate (picks evenly spaced epochs)')
    parser.add_argument('--depth', type=int, default=None, help='Stockfish depth')
    parser.add_argument('--skill', type=int, default=None, help='Stockfish skill')
    args = parser.parse_args()

    if args.quick:
        GAMES_PER_SETTING = 5
        if args.depth is None:
            STOCKFISH_DEPTH = 1
        if args.skill is None:
            STOCKFISH_SKILL = 0
    if args.games is not None:
        GAMES_PER_SETTING = args.games
    if args.depth is not None:
        STOCKFISH_DEPTH = args.depth
    if args.skill is not None:
        STOCKFISH_SKILL = args.skill

    # If --models is specified or --quick, limit the number of checkpoints
    max_models = args.models
    if args.quick and max_models is None:
        max_models = 1

    main(max_models=max_models)

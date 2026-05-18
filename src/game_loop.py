# -*- coding: utf-8 -*-
"""Architecture-invariant model-vs-engine game loop.

Takes a PolicyEngine for the model side and a UCI engine handle for the
opponent. The PolicyEngine abstraction is what makes this version-agnostic
— v1/v2/... all plug in the same way.
"""
import sys
import time

import chess

from src.engine import generate_engine_move
from src.stats import compute_temperature, init_stats, model_record


def play_models(engine_a, engine_b, limit=10, a_color=chess.WHITE,
                verbose=False, temperature=0.0, temp_decay=0.0,
                max_plies=500):
    """Play `engine_a` vs `engine_b` for `limit` games. engine_a plays
    `a_color` (WHITE or BLACK). Both are PolicyEngines (same interface).

    Returns stats dict with results from engine_a's perspective:
        wins / draws / losses, total plies, per-engine timing, etc.

    Used for v2-vs-v1 head-to-head evaluations. Both sides are subject to
    the temperature schedule — symmetric behavior.

    `max_plies` caps any single game to avoid infinite loops if the two
    models get stuck in a repetition that neither claims.
    """
    stats = init_stats()
    start_time = time.time()
    a_time = 0.0
    b_time = 0.0

    for game_num in range(1, limit + 1):
        board = chess.Board()
        plies = 0
        while not board.is_game_over() and plies < max_plies:
            current_engine = engine_a if board.turn == a_color else engine_b
            t = compute_temperature(temperature, temp_decay, board.ply())
            t0 = time.time()
            move = current_engine.generate_move(board, stats, temperature=t)
            dt = time.time() - t0
            if board.turn == a_color:
                a_time += dt
            else:
                b_time += dt
            board.push(move)
            plies += 1
            stats['turns'] += 1

        if board.is_game_over():
            result = board.result()
        else:
            # Hit max_plies — declare a draw
            result = '1/2-1/2'

        stats['games'] += 1
        stats['results'][result] += 1

        if verbose:
            w, d, l = model_record(stats, a_color)
            if result == '1/2-1/2':
                label = 'DRAW'
            elif (result == '1-0') == (a_color == chess.WHITE):
                label = 'A WINS'
            else:
                label = 'B WINS'
            print(f"  Game {game_num}/{limit}: {label} in {board.fullmove_number} moves "
                  f"(A:{w} D:{d} B:{l})")
        else:
            if result == '1/2-1/2':
                sys.stdout.write('-')
            elif (result == '1-0') == (a_color == chess.WHITE):
                sys.stdout.write('A')
            else:
                sys.stdout.write('B')
            sys.stdout.flush()

    stats['minutes_elapsed'] = (time.time() - start_time) / 60
    stats['model_minutes_elapsed'] = a_time / 60        # engine_a's time
    stats['engine_minutes_elapsed'] = b_time / 60        # engine_b's time (named for symmetry)
    if not verbose:
        print()
    return stats


def play_engine(policy_engine, engine, limit=10000, model_color=chess.WHITE,
                verbose=False, temperature=0.5, temp_decay=0.05):
    """Play model vs engine for `limit` games. Model plays `model_color`."""
    stats = init_stats()
    start_time = time.time()
    model_turn_time = 0
    engine_turn_time = 0

    for game_num in range(1, limit + 1):
        board = chess.Board()

        while True:
            if board.is_game_over():
                result = board.result()
                stats['games'] += 1
                stats['results'][result] += 1
                if verbose:
                    w, d, l = model_record(stats, model_color)
                    if result == '1/2-1/2':
                        label = 'DRAW'
                    elif (result == '1-0') == (model_color == chess.WHITE):
                        label = 'WIN'
                    else:
                        label = 'LOSS'
                    print(f"  Game {game_num}/{limit}: {label} in {board.fullmove_number} moves "
                          f"(W:{w} D:{d} L:{l})")
                else:
                    if result == '1/2-1/2':
                        sys.stdout.write('-')
                    elif (result == '1-0') == (model_color == chess.WHITE):
                        sys.stdout.write('X')
                    else:
                        sys.stdout.write('.')
                    sys.stdout.flush()
                break

            turn_start = time.time()
            if board.turn == model_color:
                temp = compute_temperature(temperature, temp_decay, board.ply())
                move = policy_engine.generate_move(board, stats, temperature=temp)
                model_turn_time += time.time() - turn_start
            else:
                move = generate_engine_move(engine, board)
                engine_turn_time += time.time() - turn_start

            board.push(move)
            stats['turns'] += 1

    stats['minutes_elapsed'] += (time.time() - start_time) / 60
    stats['model_minutes_elapsed'] += model_turn_time / 60
    stats['engine_minutes_elapsed'] += engine_turn_time / 60
    if not verbose:
        print()
    return stats

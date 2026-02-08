# -*- coding: utf-8 -*-
import argparse
import chess
import chess.engine
import chess.pgn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import torch

from model import load_model
from parse import featurize_board


# ---------------------------------------------------------------------------
# Stockfish helper (uses python-chess engine protocol)
# ---------------------------------------------------------------------------

def create_engine(path='bin/stockfish.exe', depth=0, skill_level=0):
    """Open a UCI engine via python-chess and configure depth/skill."""
    engine = chess.engine.SimpleEngine.popen_uci(path)
    engine.configure({"Skill Level": skill_level})
    engine._depth = depth  # stash for use in play_engine
    return engine


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def init_stats():
    return {
        'legal_moves': 0,
        'illegal_moves': 0,
        'turns': 0,
        'games': 0,
        'results': {'1-0': 0, '0-1': 0, '1/2-1/2': 0},
        'minutes_elapsed': 0,
        'model_minutes_elapsed': 0,
        'engine_minutes_elapsed': 0,
        'en_passant_captures': 0,
        'castles': 0,
    }


def merge_stats(stats_list):
    merged = init_stats()
    for s in stats_list:
        merged['legal_moves'] += s['legal_moves']
        merged['illegal_moves'] += s['illegal_moves']
        merged['turns'] += s['turns']
        merged['games'] += s['games']
        for k in merged['results']:
            merged['results'][k] += s['results'][k]
        merged['minutes_elapsed'] = max(merged['minutes_elapsed'], s['minutes_elapsed'])
        merged['en_passant_captures'] += s['en_passant_captures']
        merged['castles'] += s['castles']
    return merged


def print_intragame_stats(stats, prefix=''):
    moves = stats['legal_moves'] + stats['illegal_moves']
    if moves == 0:
        return
    print(prefix + "%d move attempts (%.2f%% illegal)" % (moves, 100 * stats['illegal_moves'] / moves))
    if stats['games'] > 0:
        print(prefix + "%d en passant captures (%.2f per game)" % (stats['en_passant_captures'], stats['en_passant_captures'] / stats['games']))
        print(prefix + "%d castles (%.2f per game)" % (stats['castles'], stats['castles'] / stats['games']))


def print_stats(stats, prefix=''):
    games = stats['games']
    if games == 0:
        return
    mins = stats['minutes_elapsed']
    if mins > 0:
        print(prefix + "%.1f minutes (%.2f seconds per game)" %
              (mins, 60 * mins / games))
    print_intragame_stats(stats, prefix)
    print(prefix + "%d turns (%.1f per game)" % (stats['turns'], stats['turns'] / games))
    print(prefix + "%d games (%.2f%% won, %.2f%% draw, %.2f%% lost)" %
          (games,
           100 * stats['results']['1-0'] / games,
           100 * stats['results']['1/2-1/2'] / games,
           100 * stats['results']['0-1'] / games))


def init_stats_logfile(filename):
    with open(filename, 'w') as f:
        f.write("model\tgames\tillegal_move_pct\twon_pct\tdraw_pct\tlost_pct\n")


def print_stats_to_logfile(model_filename, stats, filename):
    moves = stats['legal_moves'] + stats['illegal_moves']
    games = stats['games']
    with open(filename, 'a') as f:
        f.write("%s\t%d\t%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\n" %
                (model_filename, games,
                 100 * stats['illegal_moves'] / moves,
                 100 * stats['results']['1-0'] / games,
                 100 * stats['results']['1/2-1/2'] / games,
                 100 * stats['results']['0-1'] / games))


# ---------------------------------------------------------------------------
# Move generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_model_move(model, board, stats, device=None, show_output=False):
    """Generate a move using the neural network. Supports both colors."""
    if device is None:
        device = next(model.parameters()).device

    is_white = board.turn  # True = white, False = black

    # Featurize — rotate if black's turn so the model always sees "white to move"
    features = featurize_board(board.fen(), rotate=(not is_white))
    x = torch.from_numpy(features.astype(np.float32)).reshape(1, 6, 8, 8).to(device)
    logits = model(x)
    y = torch.softmax(logits, dim=1).cpu().numpy().reshape(64, 64)

    if show_output:
        plt.imshow(y, cmap='Greys')

    iterations = 0
    max_iterations = 64 * 64

    while iterations < max_iterations:
        from_square, to_square = np.unravel_index(y.argmax(), y.shape)

        # Un-rotate if we were playing as black
        if not is_white:
            from_square = 63 - from_square
            to_square = 63 - to_square

        move = chess.Move(from_square, to_square)

        # Pawn promotion: rank 7 for white, rank 0 for black
        piece = board.piece_type_at(from_square)
        if piece == chess.PAWN:
            target_rank = chess.square_rank(to_square)
            if (is_white and target_rank == 7) or (not is_white and target_rank == 0):
                move.promotion = chess.QUEEN

        if board.is_legal(move):
            stats['legal_moves'] += 1
            if board.is_en_passant(move):
                stats['en_passant_captures'] += 1
            if board.is_castling(move):
                stats['castles'] += 1
            break
        else:
            stats['illegal_moves'] += 1

        # Zero out this prediction — but use the rotated coordinates for indexing y
        if not is_white:
            y[63 - from_square, 63 - to_square] = 0
        else:
            y[from_square, to_square] = 0
        iterations += 1

    return move


def generate_engine_move(engine, board):
    depth = getattr(engine, '_depth', 1)
    result = engine.play(board, chess.engine.Limit(depth=depth))
    return result.move


# ---------------------------------------------------------------------------
# Game loops
# ---------------------------------------------------------------------------

def play_interactive(model, board, device=None, move_uci=None):
    if move_uci is not None:
        board.push_uci(move_uci)
    board.push(generate_model_move(model, board, init_stats(), device=device, show_output=True))
    return board


def play_engine(model, engine, limit=10000, model_color=chess.WHITE, device=None,
                verbose=False):
    """Play model vs engine for `limit` games. Model plays `model_color`."""
    stats = init_stats()
    start_time = time.time()
    model_turn_time = 0
    engine_turn_time = 0

    for game_num in range(1, limit + 1):
        board = chess.Board()
        game_node = chess.pgn.Game()
        game_node.headers['White'] = 'Neural-Chess' if model_color == chess.WHITE else 'Stockfish'
        game_node.headers['Black'] = 'Stockfish' if model_color == chess.WHITE else 'Neural-Chess'

        while True:
            if board.is_game_over():
                result = board.result()
                stats['games'] += 1
                stats['results'][result] += 1
                if verbose:
                    w = stats['results']['1-0']
                    d = stats['results']['1/2-1/2']
                    l = stats['results']['0-1']
                    label = {'1-0': 'WIN', '0-1': 'LOSS', '1/2-1/2': 'DRAW'}[result]
                    print(f"  Game {game_num}/{limit}: {label} in {board.fullmove_number} moves "
                          f"(W:{w} D:{d} L:{l})")
                else:
                    if result == '1-0':
                        sys.stdout.write('X')
                    elif result == '1/2-1/2':
                        sys.stdout.write('-')
                    else:
                        sys.stdout.write('.')
                    sys.stdout.flush()
                break

            turn_start = time.time()
            if board.turn == model_color:
                move = generate_model_move(model, board, stats, device=device)
                model_turn_time += time.time() - turn_start
            else:
                move = generate_engine_move(engine, board)
                engine_turn_time += time.time() - turn_start

            board.push(move)
            game_node = game_node.add_main_variation(move)
            stats['turns'] += 1

    stats['minutes_elapsed'] += (time.time() - start_time) / 60
    stats['model_minutes_elapsed'] += model_turn_time / 60
    stats['engine_minutes_elapsed'] += engine_turn_time / 60
    if not verbose:
        print()
    return stats


def play_engine_forever(model_filename_root, last_model_filename=None, device=None):
    engine = create_engine(depth=0, skill_level=0)
    try:
        while not os.path.isfile('.stopplay'):
            model_filename = None
            play_next = (last_model_filename is None)
            for filename in sorted(os.listdir('model')):
                if not filename.startswith(model_filename_root) or not filename.endswith('.pt'):
                    continue
                if play_next:
                    model_filename = filename
                    break
                if filename == last_model_filename:
                    play_next = True
            if model_filename is None:
                time.sleep(5)
                continue
            model = load_model('model/' + model_filename, device=device)
            print("\nPlaying %s..." % model_filename)
            stats = play_engine(model, engine, 1000, device=device)
            print()
            print_stats(stats)
            print_stats_to_logfile(model_filename, stats, '.playstats.txt')
            last_model_filename = model_filename
        if os.path.isfile('.stopplay'):
            os.remove('.stopplay')
    finally:
        engine.quit()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Neural-Chess: play against Stockfish or interactively')
    parser.add_argument('model', help='Model name (e.g. model_e0002) or path to .pt file')
    sub = parser.add_subparsers(dest='mode', required=True)

    # Engine mode
    eng = sub.add_parser('engine', help='Play against a UCI engine (Stockfish)')
    eng.add_argument('-n', '--games', type=int, default=10, help='Number of games (default: 10)')
    eng.add_argument('-d', '--depth', type=int, default=0, help='Stockfish search depth (default: 0)')
    eng.add_argument('-s', '--skill', type=int, default=0, help='Stockfish skill level 0-20 (default: 0)')
    eng.add_argument('--color', choices=['white', 'black'], default='white',
                     help='Color the model plays (default: white)')
    eng.add_argument('--stockfish', default='bin/stockfish.exe', help='Path to Stockfish executable')

    # Interactive mode
    inter = sub.add_parser('interactive', help='Play interactively (you vs the model)')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    model = load_model(args.model, device=device)
    print(f"Loaded {args.model}")

    if args.mode == 'engine':
        model_color = chess.WHITE if args.color == 'white' else chess.BLACK
        print(f"Playing {args.games} games vs Stockfish (depth={args.depth}, skill={args.skill})")
        print(f"Model plays {'white' if model_color == chess.WHITE else 'black'}")
        print()

        engine = create_engine(path=args.stockfish, depth=args.depth, skill_level=args.skill)
        try:
            stats = play_engine(model, engine, args.games, model_color=model_color,
                                device=device, verbose=True)
        finally:
            engine.quit()

        print()
        print_stats(stats)

    elif args.mode == 'interactive':
        board = chess.Board()
        print()
        print(board)
        print()
        while not board.is_game_over():
            if board.turn:
                # Human plays white
                move_uci = input("Your move (UCI, e.g. e2e4): ").strip()
                if move_uci in ('quit', 'exit', 'q'):
                    break
                try:
                    board.push_uci(move_uci)
                except ValueError:
                    print(f"Illegal move: {move_uci}")
                    continue
            else:
                # Model plays black
                stats = init_stats()
                stats['games'] = 1
                move = generate_model_move(model, board, stats, device=device)
                print(f"Neural-Chess plays: {move.uci()}")
                board.push(move)
            print()
            print(board)
            print()
        if board.is_game_over():
            print(f"Game over: {board.result()}")


if __name__ == '__main__':
    main()

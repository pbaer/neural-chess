# -*- coding: utf-8 -*-
"""Thin dispatcher: parses CLI, loads a PolicyEngine, runs game/interactive loop.

All architecture-specific logic lives behind src.inference_api.load_policy_engine —
adding v2 later means dropping a V2PolicyEngine into src/v2/inference.py; this
file doesn't change.
"""
import argparse

import chess
import torch

from src.engine import create_engine
from src.game_loop import play_engine
from src.inference_api import load_policy_engine
from src.stats import compute_temperature, init_stats, print_stats


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
    eng.add_argument('--stockfish', default=None,
                     help='Path to Stockfish executable (default: newest local bin/stockfish-vNN.exe)')
    eng.add_argument('-t', '--temperature', type=float, default=0.5,
                     help='Starting temperature for move sampling (0 = greedy, default: 0.5)')
    eng.add_argument('--temp-decay', type=float, default=0.05,
                     help='Exponential decay rate per ply (default: 0.05)')

    # Interactive mode
    inter = sub.add_parser('interactive', help='Play interactively (you vs the model)')
    inter.add_argument('-t', '--temperature', type=float, default=0.5,
                       help='Starting temperature for move sampling (0 = greedy, default: 0.5)')
    inter.add_argument('--temp-decay', type=float, default=0.05,
                       help='Exponential decay rate per ply (default: 0.05)')

    # MCTS options (apply to both subcommands). When --mcts is set, the loaded
    # PolicyEngine is wrapped in an MCTSEngine (AlphaZero-style PUCT). Only
    # works for v2 checkpoints (need a value head).
    for sub_p in (eng, inter):
        sub_p.add_argument('--mcts', action='store_true',
                           help='Use PUCT MCTS instead of raw single-shot policy')
        sub_p.add_argument('--mcts-sims', type=int, default=70,
                           help='MCTS simulations per move (default 70 ~= SF-ultra latency)')
        sub_p.add_argument('--mcts-cpuct', type=float, default=1.5,
                           help='PUCT exploration constant')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    policy_engine = load_policy_engine(args.model, device=device)
    print(f"Loaded {args.model}")

    if getattr(args, 'mcts', False):
        from src.mcts import MCTSEngine
        policy_engine = MCTSEngine(policy_engine, c_puct=args.mcts_cpuct,
                                   n_simulations=args.mcts_sims)
        print(f"MCTS enabled: {args.mcts_sims} sims/move, c_puct={args.mcts_cpuct}")

    if args.mode == 'engine':
        model_color = chess.WHITE if args.color == 'white' else chess.BLACK
        print(f"Playing {args.games} games vs Stockfish (depth={args.depth}, skill={args.skill})")
        print(f"Model plays {'white' if model_color == chess.WHITE else 'black'}")
        if args.temperature > 0:
            print(f"Temperature: {args.temperature} (decay: {args.temp_decay}/ply)")
        print()

        engine = create_engine(path=args.stockfish, depth=args.depth, skill_level=args.skill)
        try:
            stats = play_engine(policy_engine, engine, args.games, model_color=model_color,
                                verbose=True,
                                temperature=args.temperature,
                                temp_decay=args.temp_decay)
        finally:
            engine.quit()

        print()
        print_stats(stats, model_color=model_color)

    elif args.mode == 'interactive':
        board = chess.Board()
        if args.temperature > 0:
            print(f"Temperature: {args.temperature} (decay: {args.temp_decay}/ply)")
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
                temp = compute_temperature(args.temperature, args.temp_decay, board.ply())
                move = policy_engine.generate_move(board, stats, temperature=temp)
                print(f"Neural-Chess plays: {move.uci()}")
                board.push(move)
            print()
            print(board)
            print()
        if board.is_game_over():
            print(f"Game over: {board.result()}")


if __name__ == '__main__':
    main()

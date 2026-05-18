# -*- coding: utf-8 -*-
"""Architecture-invariant UCI protocol handler.

Drives a PolicyEngine through the UCI request/response loop. The protocol
layer doesn't care which model version is in use — it just calls
policy_engine.generate_move() when the GUI asks for a best move.
"""
import sys

import chess

from src.stats import init_stats, print_intragame_stats


def run_uci(policy_engine):
    """UCI protocol loop. Reads stdin, writes stdout, until 'quit'."""
    board = None
    stats = init_stats()
    stats['games'] = 1

    while True:
        try:
            s = input().strip()
        except EOFError:
            break

        if s == 'uci':
            print('id name Neural-Chess')
            print('id author pbaer')
            print('uciok')
            sys.stdout.flush()

        elif s == 'isready':
            print('readyok')
            sys.stdout.flush()

        elif s == 'ucinewgame':
            board = None
            stats = init_stats()
            stats['games'] = 1

        elif s.startswith('position'):
            parts = s.split()
            # position startpos [moves ...]
            # position fen <fen> [moves ...]
            if 'startpos' in parts:
                board = chess.Board()
                moves_idx = parts.index('startpos') + 1
            elif 'fen' in parts:
                fen_idx = parts.index('fen') + 1
                # FEN is 6 space-separated fields
                fen_parts = parts[fen_idx:fen_idx + 6]
                board = chess.Board(' '.join(fen_parts))
                moves_idx = fen_idx + 6
            else:
                continue

            if len(parts) > moves_idx and parts[moves_idx] == 'moves':
                for move_str in parts[moves_idx + 1:]:
                    board.push_uci(move_str)

        elif s.startswith('go'):
            if board is not None:
                move = policy_engine.generate_move(board, stats)
                print_intragame_stats(stats, 'info string ')
                print('bestmove ' + move.uci())
                sys.stdout.flush()

        elif s == 'quit':
            break

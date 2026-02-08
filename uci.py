# -*- coding: utf-8 -*-
import chess
import sys
import torch

from model import load_model
from play import generate_model_move, init_stats, print_intragame_stats


def run_uci(model_filename):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_filename is None:
        model_filename = 'model'
    model = load_model(model_filename, device=device)

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
                move = generate_model_move(model, board, stats, device=device)
                print_intragame_stats(stats, 'info string ')
                print('bestmove ' + move.uci())
                sys.stdout.flush()

        elif s == 'quit':
            break


if __name__ == '__main__':
    run_uci(sys.argv[1] if len(sys.argv) >= 2 else None)

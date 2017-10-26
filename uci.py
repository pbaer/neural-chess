# -*- coding: utf-8 -*-
import chess
import sys
from model import load_model
from play import generate_model_move
from play import init_stats
from play import print_intragame_stats

def run_uci(model_filename):
    if model_filename == None:
        model_filename = 'modelB3'
    board = None
    model = load_model(model_filename)
    stats = init_stats()
    stats['games'] = 1
    while True:
        s = input()
        if s == 'isready':
            print('readyok')
        if s.startswith('position startpos'):
            board = chess.Board()
        if s.startswith('position startpos moves'):
            for move in s.split()[3:]:
                board.push_uci(move)
        if s.startswith('go'):
            move = generate_model_move(model, board, stats)
            print_intragame_stats(stats, 'info ')
            print('bestmove ' + move.uci())
        if s == 'quit':
            break;

if __name__ == '__main__':
    run_uci(sys.argv[1] if len(sys.argv) >= 2 else None)

# -*- coding: utf-8 -*-
import chess.pgn
import numpy as np
import os
from train import TrainingSet
#import pdb

PAWN_PLANE = 0
ROOK_PLANE = 1
KNIGHT_PLANE = 2
BISHOP_PLANE = 3
QUEEN_PLANE = 4
KING_PLANE = 5

WHITE = True
BLACK = False

def parse_training_set():
    training_set = TrainingSet(4000000) # more than we need for any single .PGN
    for filename in os.listdir('data'):
        if not filename.endswith('.PGN'):
            continue
        print("Parsing %s..." % (filename))
        parse_games(training_set, filename)
        training_set.save_to_file(filename[:-4] + '.npz')
        print("Parsed %s (%d training rows)" % (filename, training_set.rows))
        training_set.reset()

def parse_games(training_set, filename):
    pgn = open('data/' + filename)
    while True:
        game = chess.pgn.read_game(pgn)
        if game == None:
            break
        if game.headers['Result'] == '1/2-1/2': # Don't train on draws
            continue
        if parse_game(training_set, game, train_player=(game.headers['Result'] == '1-0')) == False:
            break

def parse_game(training_set, game, train_player):
    #pdb.set_trace()
    node = game
    while True:
        board_fen = node.board().fen()
        player_turn = node.board().turn
        if node.is_end():
            break
        node = node.variations[0]
        if train_player != player_turn: 
            continue # We only care about moves made by the winner of each game
        x, y = parse_turn(board_fen, node.move, rotate=(train_player == BLACK))
        if training_set.add_row(x ,y) == False:
            return False
    return True

def parse_turn(board_fen, move, rotate):
    x = featurize_board(board_fen, rotate)
    y = outputize_move(move, rotate)
    return x, y

def featurize_board(board_fen, rotate):
    board_array = np.zeros((6, 8, 8), dtype='int8')
    f = -1 # file (column)
    r = 0 # rank (row)
    for c in board_fen:
        f += 1
        if c == 'P' or c == 'p':
            board_array[PAWN_PLANE, r, f] = 1 if c == 'P' else -1
        elif c == 'R' or c == 'r':
            board_array[ROOK_PLANE, r, f] = 1 if c == 'R' else -1
        elif c == 'N' or c == 'n':
            board_array[KNIGHT_PLANE, r, f] = 1 if c == 'N' else -1
        elif c == 'B' or c == 'b':
            board_array[BISHOP_PLANE, r, f] = 1 if c == 'B' else -1
        elif c == 'Q' or c == 'q':
            board_array[QUEEN_PLANE, r, f] = 1 if c == 'Q' else -1
        elif c == 'K' or c == 'k':
            board_array[KING_PLANE, r, f] = 1 if c == 'K' else -1
        elif c == '/':
            assert f == 8
            f = -1
            r += 1
        elif c == ' ':
            break
        else: # a number indicating 1 or more blank squares
            f += int(c) - 1
    # TODO: add parsing for castling availability
    if rotate:
        for p in range(6): # all planes
            for i in range(4): # first half of the ranks
                for j in range(8): # all files
                    temp = board_array[p, i, j]
                    board_array[p, i, j] = -board_array[p, 7-i, 7-j]
                    board_array[p, 7-i, 7-j] = -temp
    return board_array.reshape((1, TrainingSet.FEATURES))

def outputize_move(move, rotate):
    move_array = np.zeros((64, 64), dtype='int8')
    if (rotate):
        move_array[63-move.from_square, 63-move.to_square] = 1
    else:
        move_array[move.from_square, move.to_square] = 1
    return move_array.reshape((1, TrainingSet.OUTPUTS))        
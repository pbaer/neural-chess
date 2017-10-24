# -*- coding: utf-8 -*-
import chess.pgn
import numpy as np
import os
from train import TrainingSet

PAWN_PLANE = 0
ROOK_PLANE = 1
KNIGHT_PLANE = 2
BISHOP_PLANE = 3
QUEEN_PLANE = 4
KING_PLANE = 5

def parse_training_set(training_set, limit):
    for filename in os.listdir('data'):
        if not filename.endswith('.PGN'):
            continue
        limit = parse_games(training_set, 'data/' + filename, limit)
    print("%d rows in training set" % (train_rows))

def parse_games(training_set, filename, limit):
    global train_rows
    pgn = open(filename)
    games = 0
    turns = 0
    while limit != 0 and not training_set.is_full():
        game = chess.pgn.read_game(pgn)
        if game == None:
            break
        if game.headers['Result'] != '1-0': # We only care about games that White won for now
            continue
        turns += parse_game(training_set, game)
        games += 1
        if games % 100 == 0:
            print("%d games parsed in %s (%d training rows so far)" % (games, filename, train_rows))
        limit -= 1
        #pdb.set_trace()
    print("%s: %d turns in %d games parsed (white moves & wins only)" % (filename, turns, games))
    return limit

def parse_game(training_set, game):
    node = game
    turns = 0
    while True:
        board_fen = node.board().fen()
        white_turn = node.board().turn
        if node.is_end():
            break
        node = node.variations[0]
        if white_turn: # We only care about white (winning-game) moves for now
            x, y = parse_turn(board_fen, node.move)
            if training_set.add_row(x ,y) == False:
                break
            turns += 1
    return turns

def parse_turn(board_fen, move):
    x = featurize_board(board_fen)
    y = outputize_move(move)
    return x, y

def featurize_board(board_fen):
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
    return board_array.reshape((1, TrainingSet.FEATURES))

def outputize_move(move):
    move_array = np.zeros((64, 64), dtype='int8')
    move_array[move.from_square, move.to_square] = 1
    return move_array.reshape((1, TrainingSet.OUTPUTS))

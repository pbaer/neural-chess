# -*- coding: utf-8 -*-
"""v1 PGN parser: walks PGN files and writes 6-plane / one-hot NPZ archives.

The on-disk format here is what src/v1/dataset.py consumes. The TrainingSet
class is the legacy in-memory accumulator (kept because the parse flow uses
it). All v1-specific.
"""
import os

import chess.pgn
import numpy as np

from src.v1.featurize import (
    _FEATURES,
    _OUTPUTS,
    BLACK,
    featurize_board,
)


class TrainingSet():
    FEATURES = _FEATURES
    OUTPUTS = _OUTPUTS

    def __init__(self, max_rows):
        self.X = np.zeros((max_rows, self.FEATURES), dtype='int8')
        self.Y = np.zeros((max_rows, self.OUTPUTS), dtype='int8')
        self.rows = 0
        self.max_rows = max_rows

    def reset(self):
        self.rows = 0

    def get(self):
        return self.X[0:self.rows, :], self.Y[0:self.rows, :]

    def is_full(self):
        return self.rows == self.max_rows

    def add_from_file(self, filename):
        data = np.load(filename)
        return self.add_from_data(data)

    def add_from_data(self, data):
        data_rows = data['meta'][0]
        if (self.rows + data_rows > self.max_rows):
            return False
        data_X = data['X']
        data_Y = data['Y']
        self.X[self.rows:(self.rows + data_rows), :] = data_X[0:data_rows, :]
        self.Y[self.rows:(self.rows + data_rows), :] = data_Y[0:data_rows, :]
        self.rows += data_rows
        return True

    def add_from_folder(self, foldername, printonly=False):
        total_rows = 0
        for filename in os.listdir(foldername):
            if not filename.endswith('.npz'):
                continue
            data = np.load(foldername + '/' + filename)
            data_rows = data['meta'][0]
            print("%d rows in %s" % (data_rows, filename))
            total_rows += data_rows
            if printonly:
                continue
            if not self.add_from_data(data):
                total_rows -= data_rows
                print("Training set full, not adding this file.")
                break
        print("%d total rows (%.2fGB expanded)" % (total_rows, (float(total_rows) * (self.FEATURES + self.OUTPUTS)) / (1024 * 1024 * 1024)))

    def add_row(self, x, y):
        if self.is_full():
            return False
        self.X[self.rows] = x
        self.Y[self.rows] = y
        self.rows += 1
        return True

    def save_to_file(self, filename):
        meta = np.ndarray((1), dtype=int)
        meta[0] = self.rows
        np.savez_compressed('data/v1/' + filename, X=self.X[0:self.rows, :], Y=self.Y[0:self.rows, :], meta=meta)


def outputize_move(move, rotate):
    move_array = np.zeros((64, 64), dtype='int8')
    if (rotate):
        move_array[63-move.from_square, 63-move.to_square] = 1
    else:
        move_array[move.from_square, move.to_square] = 1
    return move_array.reshape((1, _OUTPUTS))


def parse_turn(board_fen, move, rotate):
    x = featurize_board(board_fen, rotate)
    y = outputize_move(move, rotate)
    return x, y


def parse_game(training_set, game, train_player):
    node = game
    while True:
        board = node.board()
        board_fen = board.fen()
        player_turn = board.turn
        if node.is_end():
            break
        node = node.variations[0]
        if train_player != player_turn:
            continue # We only care about moves made by the winner of each game
        x, y = parse_turn(board_fen, node.move, rotate=(train_player == BLACK))
        if training_set.add_row(x, y) == False:
            return False
    return True


def parse_games(training_set, filename, src_dir='data/v1'):
    with open(src_dir + '/' + filename) as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            if game.headers['Result'] == '1/2-1/2': # Don't train on draws
                continue
            if parse_game(training_set, game, train_player=(game.headers['Result'] == '1-0')) == False:
                break


def parse_training_set(src_dir='data/v1'):
    training_set = TrainingSet(4000000) # more than we need for any single .PGN
    for filename in os.listdir(src_dir):
        if not filename.endswith('.PGN'):
            continue
        print("Parsing %s..." % (filename))
        parse_games(training_set, filename, src_dir=src_dir)
        training_set.save_to_file(filename[:-4] + '.npz')
        print("Parsed %s (%d training rows)" % (filename, training_set.rows))
        training_set.reset()


if __name__ == '__main__':
    parse_training_set()

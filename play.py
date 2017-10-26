# -*- coding: utf-8 -*-
import chess
import matplotlib.pyplot as plt
import numpy as np
#import random
import time
from parse import featurize_board

#CONFIDENCE_BIAS = 5

def init_stats():
    stats = {}
    stats['legal_moves'] = 0
    stats['illegal_moves'] = 0
    stats['turns'] = 0
    stats['games'] = 0
    stats['results'] = {}
    stats['results']['1-0'] = 0
    stats['results']['0-1'] = 0
    stats['results']['1/2-1/2'] = 0
    stats['end_states'] = {}
    stats['minutes_elapsed'] = 0
    stats['won_games'] = []
    stats['draw_games'] = []
    stats['en_passant_captures'] = 0
    stats['castles'] = 0
    return stats

def print_intragame_stats(stats, prefix=''):
    moves = stats['legal_moves'] + stats['illegal_moves']
    print(prefix + ("%d move attempts (%.2f%% illegal)" % (moves, 100 * stats['illegal_moves']/moves)))
    print(prefix + ("%d en passant captures (%.2f per game)" % (stats['en_passant_captures'], stats['en_passant_captures']/stats['games'])))
    print(prefix + ("%d castles (%.2f per game)" % (stats['castles'], stats['castles']/stats['games'])))

def print_stats(stats, prefix=''):
    games = stats['games']
    print(prefix + ("%.1f minutes (%.2f seconds per game)" % (stats['minutes_elapsed'], 60 * stats['minutes_elapsed']/games)))
    print_intragame_stats(stats, prefix)
    print(prefix + ("%d turns (%.1f per game)" % (stats['turns'], stats['turns']/games)))
    print(prefix + ("%d games (%.2f%% won, %.2f%% draw, %.2f%% lost, %.2f%% unique end states)" %
         (games,
          100 * stats['results']['1-0']/games,
          100 * stats['results']['1/2-1/2']/games,
          100 * stats['results']['0-1']/games,
          100 * len(stats['end_states'])/games)))

def generate_model_move(model, board, stats, show_output=False):
    assert board.turn # Must be White's turn
    y = model.predict(featurize_board(board.fen())).reshape((64, 64))
    if show_output:
        plt.imshow(y, cmap='Greys')
    #remaining_confidence = 1
    iterations = 0
    max_iterations = 64 * 64 # Prevent infinite loops (possible due to randomization)
    #first_legal_move = None
    while iterations < max_iterations:
        from_square, to_square = np.unravel_index(y.argmax(), y.shape) # Get best remaining move
        move = chess.Move(from_square, to_square)
        if board.piece_type_at(from_square) == chess.PAWN and chess.square_rank(to_square) == 7:
            move.promotion = chess.QUEEN # Always promote to queen for now
        if board.is_legal(move):
            stats['legal_moves'] += 1
            if board.is_en_passant(move):
                stats['en_passant_captures'] += 1
            if board.is_castling(move):
                stats['castles'] += 1
            break
            #if first_legal_move == None:
            #    first_legal_move = move
            #relative_move_confidence = (y[from_square, to_square] + CONFIDENCE_BIAS)/(remaining_confidence + CONFIDENCE_BIAS) # The model's prediction that this move is the best move, given the remaining choices (with a bias to ensure it's never too small)
            #print(str(iterations) + ": " + str(y[from_square, to_square]) + " (relative: " + str(relative_move_confidence) + ")")
            #if random.uniform(0, 1) < relative_move_confidence: # Randomization filter
            #    break
        else:
            stats['illegal_moves'] += 1
        #remaining_confidence -= y[from_square, to_square]
        y[from_square, to_square] = 0
        iterations += 1
    #if not board.is_legal(move):
    #    move = first_legal_move
    return move

def generate_engine_move(engine, board):
    engine.setfenposition(board.fen())
    return chess.Move.from_uci(engine.bestmove()['move']) # TODO deal with pawn promotion

def play_interactive(model, board, move_uci=None):
    if not move_uci == None:
        assert board.turn == False # Must be Black's turn
        board.push_uci(move_uci)
    board.push(generate_model_move(model, board, init_stats(), show_output=True))
    return board

def play_engine(model, engine, limit=1):
    stats = init_stats()
    start_time = time.time()
    while limit > 0:
        board = chess.Board()
        engine.newgame()
        game_node = chess.pgn.Game()
        game_node.headers['White'] = 'Neural-Chess'
        game_node.headers['Black'] = 'Stockfish'
        while True:
            if board.is_game_over():
                stats['games'] += 1
                stats['results'][board.result()] += 1
                end_state = board.fen().split()[0] # Only consider the board itself, not castling rights, turn count etc.
                if end_state in stats['end_states']:
                    stats['end_states'][end_state] += 1
                else:
                    stats['end_states'][end_state] = 1
                if board.result() == '1-0':
                    stats['won_games'].append(str(game_node.root()))
                if board.result() == '1/2-1/2':
                    stats['draw_games'].append(str(game_node.root()))
                print("Game %d: %s (%d wins, %d draws)" %
                      (stats['games'],
                       board.result(),
                       stats['results']['1-0'],
                       stats['results']['1/2-1/2']))
                break
            if board.turn:
                move = generate_model_move(model, board, stats)
            else:
                move = generate_engine_move(engine, board)
            board.push(move)
            game_node = game_node.add_main_variation(move)
            stats['turns'] += 1
        limit -= 1
    stats['minutes_elapsed'] += (time.time() - start_time)/60
    print()
    print_stats(stats)

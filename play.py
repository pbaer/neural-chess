# -*- coding: utf-8 -*-

import chess
import matplotlib.pyplot as plt
import numpy as np
#import random
import time

#CONFIDENCE_BIAS = 5

def init_play_stats():
    global g_stats
    g_stats = {}
    g_stats['legal_moves'] = 0
    g_stats['illegal_moves'] = 0
    g_stats['turns'] = 0
    g_stats['games'] = 0
    g_stats['results'] = {}
    g_stats['results']['1-0'] = 0
    g_stats['results']['0-1'] = 0
    g_stats['results']['1/2-1/2'] = 0
    g_stats['end_states'] = {}
    g_stats['minutes_elapsed'] = 0
    g_stats['won_games'] = []
    g_stats['draw_games'] = []
    g_stats['en_passant_captures'] = 0
    g_stats['castles'] = 0

def generate_model_move(model, board, show_output=False):
    global g_stats
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
            g_stats['legal_moves'] += 1
            if board.is_en_passant(move):
                g_stats['en_passant_captures'] += 1
            if board.is_castling(move):
                g_stats['castles'] += 1
            break
            #if first_legal_move == None:
            #    first_legal_move = move
            #relative_move_confidence = (y[from_square, to_square] + CONFIDENCE_BIAS)/(remaining_confidence + CONFIDENCE_BIAS) # The model's prediction that this move is the best move, given the remaining choices (with a bias to ensure it's never too small)
            #print(str(iterations) + ": " + str(y[from_square, to_square]) + " (relative: " + str(relative_move_confidence) + ")")
            #if random.uniform(0, 1) < relative_move_confidence: # Randomization filter
            #    break
        else:
            g_stats['illegal_moves'] += 1
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
    board.push(generate_model_move(model, board, show_output=True))
    return board

def play_engine(model, engine, limit=1):
    global g_stats
    start_time = time.time()
    while limit > 0:
        board = chess.Board()
        engine.newgame()
        game_node = chess.pgn.Game()
        game_node.headers['White'] = 'Neural Network Model'
        game_node.headers['Black'] = 'Stockfish'
        while True:
            if board.is_game_over():
                g_stats['games'] += 1
                g_stats['results'][board.result()] += 1
                end_state = board.fen().split()[0] # Only consider the board itself, not castling rights, turn count etc.
                if end_state in g_stats['end_states']:
                    g_stats['end_states'][end_state] += 1
                else:
                    g_stats['end_states'][end_state] = 1
                if board.result() == '1-0':
                    g_stats['won_games'].append(str(game_node.root()))
                if board.result() == '1/2-1/2':
                    g_stats['draw_games'].append(str(game_node.root()))
                print("Game %d: %s (%d wins, %d draws)" %
                      (g_stats['games'],
                       board.result(),
                       g_stats['results']['1-0'],
                       g_stats['results']['1/2-1/2']))
                break
            if board.turn:
                move = generate_model_move(model, board)
            else:
                move = generate_engine_move(engine, board)
            board.push(move)
            game_node = game_node.add_main_variation(move)
            g_stats['turns'] += 1
        limit -= 1
    g_stats['minutes_elapsed'] += (time.time() - start_time)/60
    print()
    print_stats()

def print_stats():
    global g_stats
    moves = g_stats['legal_moves'] + g_stats['illegal_moves']
    games = g_stats['games']
    print("%.1f minutes (%.2f seconds per game)" % (g_stats['minutes_elapsed'], 60 * g_stats['minutes_elapsed']/games))
    print("%d move attempts (%.2f%% illegal)" %
         (moves,
          100 * g_stats['illegal_moves']/moves))
    print("%d turns (%.1f per game)" % (g_stats['turns'], g_stats['turns']/games))
    print("%d games (%.2f%% won, %.2f%% draw, %.2f%% lost, %.2f%% unique end states)" %
         (games,
          100 * g_stats['results']['1-0']/games,
          100 * g_stats['results']['1/2-1/2']/games,
          100 * g_stats['results']['0-1']/games,
          100 * len(g_stats['end_states'])/games))

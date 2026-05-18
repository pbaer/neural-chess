# -*- coding: utf-8 -*-
"""Architecture-invariant game statistics + temperature schedule."""
import math

import chess


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


def print_intragame_stats(stats, prefix=''):
    moves = stats['legal_moves']
    if moves == 0:
        return
    print(prefix + "%d model moves (%.2f%% first-pick illegal)" %
          (moves, 100 * stats['illegal_moves'] / moves))
    if stats['games'] > 0:
        print(prefix + "%d en passant captures (%.2f per game)" % (stats['en_passant_captures'], stats['en_passant_captures'] / stats['games']))
        print(prefix + "%d castles (%.2f per game)" % (stats['castles'], stats['castles'] / stats['games']))


def model_record(stats, model_color=chess.WHITE):
    """Return (wins, draws, losses) from the model's perspective."""
    r = stats['results']
    if model_color == chess.WHITE:
        return r['1-0'], r['1/2-1/2'], r['0-1']
    return r['0-1'], r['1/2-1/2'], r['1-0']


def print_stats(stats, prefix='', model_color=chess.WHITE):
    games = stats['games']
    if games == 0:
        return
    mins = stats['minutes_elapsed']
    if mins > 0:
        print(prefix + "%.1f minutes (%.2f seconds per game)" %
              (mins, 60 * mins / games))
    print_intragame_stats(stats, prefix)
    print(prefix + "%d turns (%.1f per game)" % (stats['turns'], stats['turns'] / games))
    wins, draws, losses = model_record(stats, model_color)
    print(prefix + "%d games (%.2f%% won, %.2f%% draw, %.2f%% lost)" %
          (games, 100 * wins / games, 100 * draws / games, 100 * losses / games))


def compute_temperature(temp_start, temp_decay, ply):
    """Exponentially decaying temperature: temp_start * exp(-temp_decay * ply)."""
    if temp_start <= 0:
        return 0.0
    return temp_start * math.exp(-temp_decay * ply)

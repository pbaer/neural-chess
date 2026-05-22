# -*- coding: utf-8 -*-
"""Uniform-random-move opponent implementing the PolicyEngine interface.

A baseline opponent for measuring conversion ability: a trained model should
beat this 100% of the time and ideally mate quickly. Draws against it are
conversion failures (50-move rule, threefold, or stalemate) — a useful proxy
for finishing/endgame technique that the SF ladder doesn't isolate.
"""
import random


class RandomPolicyEngine:
    """Two-step uniform sampling: pick a piece uniformly among pieces that have
    at least one legal move, then pick uniformly among that piece's legal moves.
    Under check, board.legal_moves is already restricted to check-resolving
    moves, so the sample space is correct automatically.

    Implements the same generate_move(board, stats, temperature) signature as
    PolicyEngine so it slots into game_loop.play_models().
    """

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def generate_move(self, board, stats, temperature: float = 0.0):
        by_piece = {}
        for m in board.legal_moves:
            by_piece.setdefault(m.from_square, []).append(m)
        piece_sq = self.rng.choice(list(by_piece.keys()))
        move = self.rng.choice(by_piece[piece_sq])
        if stats is not None:
            stats['legal_moves'] = stats.get('legal_moves', 0) + 1
        return move

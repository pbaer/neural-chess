# -*- coding: utf-8 -*-
"""AlphaZero-style PUCT Monte Carlo Tree Search.

Principle-compliant per memory/project-principles.md (MCTS carve-out): the
search uses ONLY the model's policy priors P and value estimate V. No
hand-coded chess heuristics enter the tree — the only chess "knowledge" is the
rules of the game (legal moves, terminal detection), which are mechanics, not
strategy.

The engine wraps any object exposing:
    evaluate(board) -> (priors: dict[chess.Move, float], value: float in [-1,1])
where value is from the side-to-move's perspective. V2PolicyEngine implements
this.

Design notes:
- Sequential MCTS: one model eval per simulation at the expanded leaf.
- No tree reuse across moves (rebuilt each move) — simple + correct first.
- Final move selection: argmax visit count (temperature 0) or sample by
  N^(1/temp) for temperature > 0.
- Optional Dirichlet noise on root priors for exploration (off by default;
  competitive eval wants deterministic strongest play).
"""
import math
import random

import chess
import numpy as np


class _Node:
    __slots__ = ('prior', 'children', 'N', 'W', 'is_expanded', 'terminal_value')

    def __init__(self, prior: float):
        self.prior = prior          # P(a) from parent's policy head
        self.children = {}          # move -> _Node
        self.N = 0                  # visit count
        self.W = 0.0                # total action-value (from this node's mover's view)
        self.is_expanded = False
        self.terminal_value = None  # set if this node is a game-over state

    @property
    def Q(self):
        return self.W / self.N if self.N > 0 else 0.0


class MCTS:
    def __init__(self, engine, c_puct: float = 1.5, n_simulations: int = 100,
                 dirichlet_alpha: float = 0.3, dirichlet_frac: float = 0.0,
                 seed: int = 0):
        """
        Args:
            engine: object with evaluate(board) -> (priors dict, value float)
            c_puct: exploration constant in the PUCT formula
            n_simulations: fixed number of simulations per move
            dirichlet_alpha: Dirichlet noise concentration (root only)
            dirichlet_frac: fraction of root prior that is noise (0 = none)
            seed: RNG seed for noise + tie-breaking
        """
        self.engine = engine
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_frac = dirichlet_frac
        self.rng = np.random.default_rng(seed)

    def _terminal_value(self, board: chess.Board):
        """Value from the side-to-move's perspective if the game is over, else None.
        Uses only the rules of the game (no chess heuristics)."""
        if board.is_checkmate():
            # Side to move has been mated -> they lost.
            return -1.0
        if (board.is_stalemate() or board.is_insufficient_material()
                or board.is_seventyfive_moves() or board.is_fivefold_repetition()):
            return 0.0
        # Claimable draws (threefold / fifty-move) — treat as draw in search.
        if board.can_claim_threefold_repetition() or board.can_claim_fifty_moves():
            return 0.0
        return None

    def _expand(self, node: _Node, board: chess.Board):
        """Evaluate a leaf and attach children. Returns the leaf value from
        the side-to-move's perspective."""
        tv = self._terminal_value(board)
        if tv is not None:
            node.terminal_value = tv
            node.is_expanded = True
            return tv
        priors, value = self.engine.evaluate(board)
        for move, p in priors.items():
            node.children[move] = _Node(p)
        node.is_expanded = True
        return value

    def _puct_select(self, node: _Node):
        """Pick the child maximizing PUCT. Returns (move, child_node)."""
        sqrt_total = math.sqrt(node.N)  # sum of child visits == node.N after first expand
        best_score = -float('inf')
        best = None
        for move, child in node.children.items():
            # Q is from the child's mover's perspective; from THIS node's mover's
            # perspective the child's value is negated.
            q = -child.Q if child.N > 0 else 0.0
            u = self.c_puct * child.prior * sqrt_total / (1 + child.N)
            score = q + u
            if score > best_score:
                best_score = score
                best = (move, child)
        return best

    def run(self, board: chess.Board):
        """Run n_simulations from `board` and return (best_move, root, info)."""
        root = _Node(prior=1.0)
        root_value = self._expand(root, board)

        # Optional Dirichlet noise on root priors (exploration).
        if self.dirichlet_frac > 0 and root.children:
            moves = list(root.children.keys())
            noise = self.rng.dirichlet([self.dirichlet_alpha] * len(moves))
            for m, n in zip(moves, noise):
                c = root.children[m]
                c.prior = (1 - self.dirichlet_frac) * c.prior + self.dirichlet_frac * n

        for _ in range(self.n_simulations):
            node = root
            search_board = board.copy()
            path = [node]

            # Selection: descend until we reach an unexpanded node.
            while node.is_expanded and node.terminal_value is None and node.children:
                move, child = self._puct_select(node)
                search_board.push(move)
                node = child
                path.append(node)

            # Expansion + evaluation (unless terminal).
            if node.terminal_value is not None:
                leaf_value = node.terminal_value
            else:
                leaf_value = self._expand(node, search_board)

            # Backup: leaf_value is from the perspective of the side to move at
            # the leaf. Walk back up flipping sign each ply.
            v = leaf_value
            for n in reversed(path):
                n.N += 1
                n.W += v
                v = -v

        # Final move selection by visit count.
        info = {
            'root_value': root_value,
            'visits': {m: c.N for m, c in root.children.items()},
            'q': {m: -c.Q for m, c in root.children.items()},
            'n_sims': self.n_simulations,
        }
        if not root.children:
            return None, root, info
        best_move = max(root.children.items(), key=lambda kv: kv[1].N)[0]
        return best_move, root, info

    def select_move(self, board: chess.Board, temperature: float = 0.0):
        best_move, root, info = self.run(board)
        if best_move is None:
            legal = list(board.legal_moves)
            return random.choice(legal) if legal else None
        if temperature and temperature > 0.01:
            moves = list(info['visits'].keys())
            counts = np.array([info['visits'][m] for m in moves], dtype=np.float64)
            weights = counts ** (1.0 / temperature)
            weights /= weights.sum()
            return moves[int(self.rng.choice(len(moves), p=weights))]
        return best_move


class MCTSEngine:
    """Drop-in PolicyEngine-compatible wrapper around MCTS so it works anywhere
    a PolicyEngine is expected (play.py, game_loop.play_models, eval)."""

    def __init__(self, policy_engine, c_puct: float = 1.5, n_simulations: int = 100,
                 dirichlet_frac: float = 0.0, seed: int = 0):
        self.policy_engine = policy_engine
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.dirichlet_frac = dirichlet_frac
        self.seed = seed

    def generate_move(self, board: chess.Board, stats, temperature: float = 0.0) -> chess.Move:
        mcts = MCTS(self.policy_engine, c_puct=self.c_puct,
                    n_simulations=self.n_simulations,
                    dirichlet_frac=self.dirichlet_frac, seed=self.seed)
        move = mcts.select_move(board, temperature=temperature)
        if stats is not None:
            stats['legal_moves'] = stats.get('legal_moves', 0) + 1
            if move is not None and board.is_en_passant(move):
                stats['en_passant_captures'] = stats.get('en_passant_captures', 0) + 1
            if move is not None and board.is_castling(move):
                stats['castles'] = stats.get('castles', 0) + 1
        return move

# -*- coding: utf-8 -*-
"""v3 inference: wraps ChessModelV3 as a PolicyEngine.

The featurization (21-plane + rotation), move encoding (8×8×73), legal masking
and decode are identical to v2 — only the model architecture differs — so this
reuses src.v2.featurize / src.v2.moves and mirrors V2PolicyEngine's logic.
"""
from dataclasses import asdict

import chess
import numpy as np
import torch

from src.inference_api import PolicyEngine
from src.v2.featurize import featurize, rotate_square
from src.v2.moves import NUM_MOVES, decode_move, legal_mask
from src.v3.model import ChessConfigV3, ChessModelV3


def load_v3_model(filename, device=None, config: ChessConfigV3 = None):
    """Load a v3 .pt checkpoint (dict with 'model' + 'config' + arch='v3')."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load(filename, map_location=device, weights_only=False)
    state = data['model'] if isinstance(data, dict) and 'model' in data else data
    if config is None:
        if isinstance(data, dict) and 'config' in data:
            allowed = {f for f in ChessConfigV3.__dataclass_fields__}
            config = ChessConfigV3(**{k: v for k, v in data['config'].items() if k in allowed})
        else:
            config = ChessConfigV3()
    model = ChessModelV3(config)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def save_v3_checkpoint(path, model, optimizer=None, scheduler=None, epoch=None,
                       config: ChessConfigV3 = None):
    blob = {
        'model': (model._orig_mod if hasattr(model, '_orig_mod') else model).state_dict(),
        'epoch': epoch if epoch is not None else 0,
        'arch': 'v3',
    }
    if config is not None:
        blob['config'] = asdict(config)
    if optimizer is not None:
        blob['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        blob['scheduler'] = scheduler.state_dict()
    torch.save(blob, path)


class V3PolicyEngine(PolicyEngine):
    """PolicyEngine for the v3 attention architecture."""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    @classmethod
    def from_checkpoint(cls, path, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_v3_model(path, device=device)
        return cls(model, device)

    @torch.no_grad()
    def evaluate(self, board: chess.Board):
        """One forward pass -> (priors dict {actual Move: prob}, value float).
        Used by MCTS. Mirrors V2PolicyEngine.evaluate."""
        is_white = board.turn
        x = featurize(board)
        rotated_board = board if is_white else board.mirror()
        x = torch.from_numpy(x).unsqueeze(0).to(self.device)
        logits, value = self.model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float64).reshape(NUM_MOVES)
        v = float(value.squeeze().item())
        mask = legal_mask(rotated_board)
        probs = np.where(mask, probs, 0.0)
        total = probs.sum()
        priors = {}
        for idx in np.flatnonzero(mask):
            rm = decode_move(int(idx), rotated_board)
            move = rm if is_white else chess.Move(
                rotate_square(rm.from_square), rotate_square(rm.to_square), promotion=rm.promotion)
            if move in board.legal_moves:
                priors[move] = probs[idx] / total if total > 0 else 1.0 / len(np.flatnonzero(mask))
        if not priors:
            legal = list(board.legal_moves)
            priors = {m: 1.0 / len(legal) for m in legal}
        return priors, v

    @torch.no_grad()
    def generate_move(self, board: chess.Board, stats, temperature: float = 0.0) -> chess.Move:
        is_white = board.turn
        x = featurize(board)
        rotated_board = board if is_white else board.mirror()
        x = torch.from_numpy(x).unsqueeze(0).to(self.device)
        logits, value = self.model(x)
        if temperature > 0.01:
            probs = torch.softmax(logits / temperature, dim=1)
        else:
            probs = torch.softmax(logits, dim=1)
        probs = probs.cpu().numpy().astype(np.float64).reshape(NUM_MOVES)
        mask = legal_mask(rotated_board)
        if not mask[int(probs.argmax())]:
            stats['illegal_moves'] += 1
        probs = np.where(mask, probs, 0.0)
        total = probs.sum()
        if total > 0:
            probs /= total
            idx = int(np.random.choice(NUM_MOVES, p=probs)) if temperature > 0.01 else int(probs.argmax())
        else:
            idx = int(np.random.choice(np.flatnonzero(mask)))
        rotated_move = decode_move(idx, rotated_board)
        if is_white:
            move = rotated_move
        else:
            move = chess.Move(rotate_square(rotated_move.from_square),
                              rotate_square(rotated_move.to_square),
                              promotion=rotated_move.promotion)
        if move not in board.legal_moves:
            legal = list(board.legal_moves)
            if legal:
                move = legal[np.random.randint(len(legal))]
        stats['legal_moves'] += 1
        if board.is_en_passant(move):
            stats['en_passant_captures'] += 1
        if board.is_castling(move):
            stats['castles'] += 1
        return move

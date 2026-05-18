# -*- coding: utf-8 -*-
"""v2 inference: wraps ChessModelV2 as a PolicyEngine.

Knows about the 21-plane input encoding, the 8x8x73 = 4672 output, legal-mask
sampling over the larger move space, and the no-rotation-trick convention
(side-to-move is an explicit input plane).

Per memory/project-principles.md: temperature sampling stays for variety;
hard greedy is the default at eval time.
"""
import chess
import numpy as np
import torch

from src.inference_api import PolicyEngine
from src.v2.featurize import featurize, rotate_square
from src.v2.model import ChessConfigV2, ChessModelV2
from src.v2.moves import NUM_MOVES, NUM_MOVE_TYPES, decode_move, legal_mask


def load_v2_model(filename, device=None, config: ChessConfigV2 = None):
    """Load a v2 .pt checkpoint.

    Accepts the standard dict format: {'model': state_dict, 'optimizer': ...,
    'scheduler': ..., 'epoch': ..., 'config': {...}}. If 'config' is present
    in the checkpoint we reconstruct ChessModelV2 with it; otherwise we use
    the passed-in config or default T0a.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = torch.load(filename, map_location=device, weights_only=False)
    state = data['model'] if isinstance(data, dict) and 'model' in data else data

    if config is None:
        if isinstance(data, dict) and 'config' in data:
            cfg_dict = data['config']
            # Filter to only fields ChessConfigV2 knows
            allowed = {f for f in ChessConfigV2.__dataclass_fields__}
            cfg = ChessConfigV2(**{k: v for k, v in cfg_dict.items() if k in allowed})
        else:
            cfg = ChessConfigV2()
    else:
        cfg = config

    model = ChessModelV2(cfg)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def save_v2_checkpoint(path, model, optimizer=None, scheduler=None, epoch=None,
                       config: ChessConfigV2 = None):
    """Save a v2 training checkpoint with full state."""
    from dataclasses import asdict
    blob = {
        'model': (model._orig_mod if hasattr(model, '_orig_mod') else model).state_dict(),
        'epoch': epoch if epoch is not None else 0,
        'arch': 'v2',
    }
    if config is not None:
        blob['config'] = asdict(config)
    if optimizer is not None:
        blob['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        blob['scheduler'] = scheduler.state_dict()
    torch.save(blob, path)


class V2PolicyEngine(PolicyEngine):
    """PolicyEngine implementation for the v2 architecture (T0a baseline)."""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    @classmethod
    def from_checkpoint(cls, path, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_v2_model(path, device=device)
        return cls(model, device)

    @torch.no_grad()
    def generate_move(self, board: chess.Board, stats, temperature: float = 0.0) -> chess.Move:
        """Generate a legal move using the v2 model.

        Pattern: rotate board if black to move (so the model always sees
        "white to move"), forward pass -> softmax -> legal mask in rotated
        frame -> renormalize -> sample/argmax -> un-rotate the predicted
        from/to back to actual coordinates.
        """
        is_white = board.turn
        # Featurize handles rotation internally; we also need a rotated
        # board for legal mask + decode_move calls below.
        x = featurize(board)
        rotated_board = board if is_white else board.mirror()
        x = torch.from_numpy(x).unsqueeze(0).to(self.device)

        logits, value = self.model(x)
        # logits: (1, 4672), value: (1, 1) in rotated frame (always "side-to-move")

        if temperature > 0.01:
            probs = torch.softmax(logits / temperature, dim=1)
        else:
            probs = torch.softmax(logits, dim=1)
        probs = probs.cpu().numpy().astype(np.float64).reshape(NUM_MOVES)

        # Legal mask in rotated frame
        mask = legal_mask(rotated_board)
        if not mask[int(probs.argmax())]:
            stats['illegal_moves'] += 1

        probs = np.where(mask, probs, 0.0)
        total = probs.sum()
        if total > 0:
            probs /= total
            if temperature > 0.01:
                idx = int(np.random.choice(NUM_MOVES, p=probs))
            else:
                idx = int(probs.argmax())
        else:
            idx = int(np.random.choice(np.flatnonzero(mask)))

        # Decode in rotated frame
        rotated_move = decode_move(idx, rotated_board)

        # Un-rotate the move back to actual board coords if needed
        if is_white:
            move = rotated_move
        else:
            move = chess.Move(
                rotate_square(rotated_move.from_square),
                rotate_square(rotated_move.to_square),
                promotion=rotated_move.promotion,
            )

        # Safety net: if decode produced something not actually legal,
        # fall back to a random legal move
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


def load_v2_engine(path, device=None):
    """Convenience: load a v2 checkpoint and wrap as a V2PolicyEngine."""
    return V2PolicyEngine.from_checkpoint(path, device=device)

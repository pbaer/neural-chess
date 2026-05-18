# -*- coding: utf-8 -*-
"""v1 inference: wraps ChessModel as a PolicyEngine.

Knows about the 12-plane input encoding, the rotate-for-black trick, the
4096-logit output reshape, and pawn-promotion handling — none of which
should leak into the generic game loop / UCI driver.
"""
import chess
import numpy as np
import torch

from src.inference_api import PolicyEngine
from src.v1.featurize import featurize_board_for_model
from src.v1.model import ChessModel


def load_v1_model(filename, device=None):
    """Load a v1 .pt checkpoint into a raw nn.Module.

    Accepts both file formats:
      - new: dict with {'model': state_dict, 'optimizer': ..., 'scheduler': ..., 'epoch': ...}
      - legacy: a plain state_dict (weights only)

    Args:
        filename: path to .pt file, or a bare name resolved as model/<name>.pt
        device: torch device (auto-detected if None)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not filename.endswith('.pt'):
        filename = 'model/' + filename + '.pt'

    data = torch.load(filename, map_location=device, weights_only=True)
    state = data['model'] if isinstance(data, dict) and 'model' in data else data

    model = ChessModel()
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def save_v1_model(model, filename):
    """Save v1 model state dict (weights only) as a .pt file into model/ directory.

    Kept for ad-hoc/inference use. Training uses save_checkpoint() instead,
    which writes a richer dict that also restores optimizer + scheduler state on
    resume — load_v1_model() accepts both formats.
    """
    path = 'model/' + filename + '.pt'
    torch.save(model.state_dict(), path)


class V1PolicyEngine(PolicyEngine):
    """PolicyEngine implementation for the v1 residual-CNN architecture."""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    @classmethod
    def from_checkpoint(cls, path, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = load_v1_model(path, device=device)
        return cls(model, device)

    @torch.no_grad()
    def generate_move(self, board, stats, temperature=0.0):
        """Generate a legal move using the neural network. Supports both colors.

        The model's logits are softmax'd, masked to legal moves, renormalized,
        and then either sampled from (temperature > 0.01) or argmax'd (greedy).
        Masking guarantees the first pick is always legal, so there is no retry
        loop — and avoids a numerical pitfall where a sharp distribution (low
        temperature) underflows to a delta function on an illegal argmax.

        Args:
            temperature: > 0.01 to sample from softmax(logits / temperature);
                otherwise picks the highest-probability legal move.

        stats['legal_moves'] counts moves played. stats['illegal_moves'] counts
        moves where the model's *unmasked top-1* was illegal — a model-quality
        metric, not a wasted attempt count.
        """
        device = self.device
        is_white = board.turn  # True = white, False = black

        # Featurize — rotate if black's turn so the model always sees "white to move".
        # Output is (12, 8, 8) float32 (6 own + 6 opponent binary planes).
        features = featurize_board_for_model(board.fen(), rotate=(not is_white))
        x = torch.from_numpy(features).unsqueeze(0).to(device)  # (1, 12, 8, 8)
        logits = self.model(x)

        # Legal-move mask in the model's coordinate frame (rotated for black)
        legal_mask = np.zeros((64, 64), dtype=bool)
        for mv in board.legal_moves:
            fr, to = mv.from_square, mv.to_square
            if not is_white:
                fr, to = 63 - fr, 63 - to
            legal_mask[fr, to] = True

        # Distribution. FP64 keeps renormalization sum close to 1 even when a
        # low temperature pushes most entries through FP32 underflow.
        if temperature > 0.01:
            probs = torch.softmax(logits / temperature, dim=1)
        else:
            probs = torch.softmax(logits, dim=1)
        probs = probs.cpu().numpy().astype(np.float64).reshape(64, 64)

        # Model-quality metric: was the unmasked top-1 illegal?
        if not legal_mask[np.unravel_index(probs.argmax(), probs.shape)]:
            stats['illegal_moves'] += 1

        # Mask, renormalize, then sample or argmax
        probs = np.where(legal_mask, probs, 0.0)
        total = probs.sum()
        if total > 0:
            probs /= total
            if temperature > 0.01:
                idx = np.random.choice(probs.size, p=probs.flatten())
            else:
                idx = probs.argmax()
        else:
            # All legal moves underflowed to 0 — pick uniformly among legal moves
            idx = np.random.choice(np.flatnonzero(legal_mask.ravel()))

        from_square, to_square = np.unravel_index(idx, probs.shape)

        # Un-rotate if we were playing as black
        if not is_white:
            from_square = 63 - from_square
            to_square = 63 - to_square

        from_square, to_square = int(from_square), int(to_square)
        move = chess.Move(from_square, to_square)

        # Pawn promotion: rank 7 for white, rank 0 for black
        piece = board.piece_type_at(from_square)
        if piece == chess.PAWN:
            target_rank = chess.square_rank(to_square)
            if (is_white and target_rank == 7) or (not is_white and target_rank == 0):
                move.promotion = chess.QUEEN

        stats['legal_moves'] += 1
        if board.is_en_passant(move):
            stats['en_passant_captures'] += 1
        if board.is_castling(move):
            stats['castles'] += 1

        return move


def load_v1_engine(path, device=None):
    """Convenience: load a checkpoint and wrap as a V1PolicyEngine."""
    return V1PolicyEngine.from_checkpoint(path, device=device)

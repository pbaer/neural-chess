# -*- coding: utf-8 -*-
"""PolicyEngine abstraction + version-detecting factory.

This decouples the generic game-loop / UCI / eval code from any single
architecture. Each model version implements PolicyEngine in its own
src/<version>/inference.py module. load_policy_engine() inspects a
checkpoint and dispatches to the right loader.
"""
from abc import ABC, abstractmethod
from typing import Optional

import torch


class PolicyEngine(ABC):
    """Abstract interface for a chess move-generating model.

    Implementations encapsulate everything version-specific: input
    featurization, output decoding, illegal-move masking, promotion
    handling, color rotation tricks, etc. Callers just pass in a board
    and a stats dict and get back a chess.Move.
    """

    @abstractmethod
    def generate_move(self, board, stats, temperature: float = 0.0):
        """Pick a legal move for the side to move on ``board``.

        Args:
            board: chess.Board to move on.
            stats: dict mutated to record per-move counters (legal_moves,
                illegal_moves, en_passant_captures, castles).
            temperature: > 0.01 to sample from softmax(logits / T);
                otherwise greedy argmax.

        Returns:
            chess.Move guaranteed to be legal in ``board``.
        """


def _detect_version(path: str, device: Optional[torch.device] = None) -> str:
    """Inspect a checkpoint to determine which architecture version saved it.

    Detection strategy:
      1) If the file is a dict and has 'arch' key, trust it.
      2) Otherwise inspect state_dict shapes:
         - v1: input_conv.weight has 12 input channels
         - v2: input_conv.weight has 21 input channels AND policy_conv2 outputs 73
    """
    if device is None:
        device = torch.device('cpu')
    data = torch.load(path, map_location=device, weights_only=False)

    # Explicit arch tag in checkpoint dict (v2 sets this; v1 doesn't)
    if isinstance(data, dict) and 'arch' in data:
        return data['arch']

    state = data['model'] if isinstance(data, dict) and 'model' in data else data

    # Shape-based detection
    if 'input_conv.weight' in state:
        in_channels = state['input_conv.weight'].shape[1]
        if in_channels == 12:
            return 'v1'
        if in_channels == 21:
            return 'v2'

    raise ValueError(
        f"Could not detect model version from checkpoint {path!r}. "
        f"Top-level keys: {list(state.keys())[:6]}..."
    )


def load_policy_engine(path: str, device: Optional[torch.device] = None) -> PolicyEngine:
    """Load a checkpoint and return the matching PolicyEngine.

    Args:
        path: path to a .pt checkpoint (or a bare name resolved per-version).
        device: torch device; auto-detected if None.

    Returns:
        A concrete PolicyEngine for the architecture that produced the file.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Resolve bare-name shortcut: "foo" -> "model/foo.pt" (legacy behavior
    # preserved from v1 load_model so existing CLI invocations still work).
    resolved = path if path.endswith('.pt') else 'model/' + path + '.pt'

    version = _detect_version(resolved, device=device)
    if version == 'v1':
        from src.v1.inference import V1PolicyEngine
        return V1PolicyEngine.from_checkpoint(resolved, device=device)
    if version == 'v2':
        from src.v2.inference import V2PolicyEngine
        return V2PolicyEngine.from_checkpoint(resolved, device=device)
    raise ValueError(f"Unknown model version: {version!r}")

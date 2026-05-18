# -*- coding: utf-8 -*-
"""v1 neural-chess: 12-plane residual CNN with conv policy head.

Architecture-specific modules:
    model       -- ChessModel class
    featurize   -- FEN -> 6-plane / 12-plane board encoders
    parse_pgn   -- PGN -> NPZ data pipeline (legacy)
    dataset     -- ChessDataset for the v1 NPZ format
    inference   -- V1PolicyEngine (implements src.inference_api.PolicyEngine)
    train       -- training loop and checkpoint I/O
"""

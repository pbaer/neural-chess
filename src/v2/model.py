# -*- coding: utf-8 -*-
"""v2 model architecture — config-driven so the same code supports the whole
scale ladder (T0 smoke through T3 full).

T0a (current default): plain ResNet encoder + policy head + value head.
No lookahead block yet — that's T0b. Built so adding the lookahead is a
local change to forward() without touching the encoder, heads, or training
loop.

Per memory/project-principles.md:
  - Input is the board state only (21 planes — see featurize.py)
  - Outputs are policy (8x8x73=4672 logits) and value (P(win) scalar in [-1,+1])
  - No auxiliary heads predicting computed chess features
"""
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from src.v2.featurize import NUM_PLANES
from src.v2.moves import NUM_MOVES, NUM_MOVE_TYPES


@dataclass
class ChessConfigV2:
    """All architecture size knobs. Same class supports every ladder tier."""
    # Encoder
    encoder_blocks: int = 6
    encoder_channels: int = 128
    # Policy head intermediate channels
    policy_channels: int = 32
    # Value head intermediate width
    value_channels: int = 1       # 1x1 conv compresses to this many planes
    value_hidden: int = 64        # MLP hidden width

    # Future-move auxiliary heads (v3 POC-1): N training-only policy heads that
    # predict the moves actually played at the next N plies of the same game.
    # 0 = none (inference-time models). Principle-2-clean: targets are observed
    # future moves, not computed features. Discarded at inference.
    future_move_heads: int = 0

    # Lookahead block (not used in T0a; placeholder for T0b+)
    # K=0 means "no lookahead" — pure encoder->heads
    lookahead_K: int = 0
    lookahead_depth: int = 0
    dynamics_dim: int = 0
    aggregator_heads: int = 4
    aggregator_layers: int = 2

    # Input
    input_planes: int = NUM_PLANES
    # Output
    output_size: int = NUM_MOVES


def _default_t0a_config() -> ChessConfigV2:
    """T0a smoke-test default: 6 blocks * 128 ch, no lookahead."""
    return ChessConfigV2()


class ResidualBlock(nn.Module):
    """Two-conv residual block. Same as v1 but parameterized by channel count."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class _MoveHead(nn.Module):
    """A policy-shaped head: 1x1 conv compression -> BN -> ReLU -> 1x1 conv to
    73 move-types -> reshape to flat (B, 4672). Used for the future-move
    auxiliary heads (the MAIN policy head stays inline in ChessModelV2 to
    preserve its state_dict key names for warm-start loading)."""

    def __init__(self, channels: int, mid: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, mid, 1, bias=False)
        self.bn = nn.BatchNorm2d(mid)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid, NUM_MOVE_TYPES, 1)

    def forward(self, out):
        p = self.relu(self.bn(self.conv1(out)))
        p = self.conv2(p)  # (B, 73, 8, 8)
        return p.permute(0, 2, 3, 1).reshape(p.size(0), NUM_MOVES)


class ChessModelV2(nn.Module):
    """v2 model. Config-driven; same class for every ladder tier.

    Inputs:  (B, 21, 8, 8) float32 — see featurize.py
    Outputs:
       policy_logits: (B, 4672) — raw logits, flat[from_sq * 73 + move_type]
       value:         (B, 1)    — tanh-scaled win probability in [-1, +1]
                                  from the moving player's perspective
    """

    def __init__(self, config: Optional[ChessConfigV2] = None):
        super().__init__()
        if config is None:
            config = _default_t0a_config()
        self.config = config

        C = config.encoder_channels
        # Input stem
        self.input_conv = nn.Conv2d(config.input_planes, C, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(C)
        self.relu = nn.ReLU(inplace=True)

        # Residual tower
        self.tower = nn.Sequential(
            *[ResidualBlock(C) for _ in range(config.encoder_blocks)]
        )

        # Lookahead block (T0a: disabled by K=0; T0b+ enables via config)
        self.has_lookahead = config.lookahead_K > 0
        if self.has_lookahead:
            # Import here to avoid circular dep (lookahead imports ResidualBlock from this file)
            from src.v2.lookahead import LookaheadBlock
            self.lookahead = LookaheadBlock(
                channels=C,
                K=config.lookahead_K,
                depth=config.lookahead_depth,
                proposer_channels=config.policy_channels,
                aggregator_heads=config.aggregator_heads,
                aggregator_layers=config.aggregator_layers,
            )

        # Policy head: per-square 73-channel output, reshape to flat 4672.
        # AlphaZero-style. Spatial dim is the from-square (h*8+w == python-chess sq).
        self.policy_conv1 = nn.Conv2d(C, config.policy_channels, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(config.policy_channels)
        self.policy_conv2 = nn.Conv2d(config.policy_channels, NUM_MOVE_TYPES, 1)

        # Value head: 1x1 conv -> flatten -> MLP -> tanh
        self.value_conv = nn.Conv2d(C, config.value_channels, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(config.value_channels)
        self.value_fc1 = nn.Linear(config.value_channels * 64, config.value_hidden)
        self.value_fc2 = nn.Linear(config.value_hidden, 1)

        # Future-move auxiliary heads (training-only; unused at inference).
        # New module names => not present in older checkpoints => init fresh
        # on warm-start (strict=False) without disturbing the main heads.
        self.future_heads = nn.ModuleList(
            [_MoveHead(C, config.policy_channels)
             for _ in range(config.future_move_heads)]
        )

    def forward(self, x, return_future: bool = False):
        # x: (B, 21, 8, 8)
        out = self.relu(self.input_bn(self.input_conv(x)))
        out = self.tower(out)  # (B, C, 8, 8)

        if self.has_lookahead:
            out = self.lookahead(out)

        # Policy head
        p = self.relu(self.policy_bn(self.policy_conv1(out)))
        p = self.policy_conv2(p)  # (B, 73, 8, 8)
        # Reshape so that flat[from_sq * 73 + move_type] matches the encoder.
        # In PyTorch's default (B, C, H, W) layout, permute to (B, H, W, C)
        # gives flat order [h, w, c]. We want from_sq = h*8 + w, then move_type.
        p = p.permute(0, 2, 3, 1).reshape(p.size(0), NUM_MOVES)

        # Value head
        v = self.relu(self.value_bn(self.value_conv(out)))
        v = v.flatten(1)  # (B, value_channels * 64)
        v = self.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))  # (B, 1)

        # Inference path is unchanged: returns (policy, value). Aux heads only
        # computed when explicitly requested (training) and present.
        if return_future and len(self.future_heads) > 0:
            future = [head(out) for head in self.future_heads]
            return p, v, future
        return p, v


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

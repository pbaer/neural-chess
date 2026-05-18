# -*- coding: utf-8 -*-
"""v1 chess model: 10-block residual CNN with conv policy head.

Architecture-specific: 12-plane input, 256-channel tower, 4096-logit output.
Load/save wrappers live in src/v1/inference.py and src/v1/train.py.
"""
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
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
        out = self.relu(out + residual)
        return out


class ChessModel(nn.Module):
    """Residual CNN for chess move prediction (AlphaZero/Leela-style).

    Input:  (B, 12, 8, 8) — 6 own-piece + 6 opponent-piece binary planes
    Output: (B, 4096)     — raw logits over 64×64 (from, to) square pairs,
                            flat[from * 64 + to] in python-chess indexing
    """

    def __init__(self, num_blocks=10, channels=256, policy_channels=32, input_planes=12):
        super().__init__()

        # Input convolution
        self.input_conv = nn.Conv2d(input_planes, channels, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        # Residual tower
        self.residual_tower = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )

        # Policy head — AlphaZero-style per-square move planes.
        # The 1×1 compression preserves spatial structure (which the dense
        # head was throwing away); the second 1×1 projects to 64 "to-square"
        # logits per "from-square" location. Output is (B, 64, 8, 8): far
        # fewer parameters than the old Linear(2048, 4096), and the spatial
        # inductive bias is kept all the way through to the output.
        self.policy_conv1 = nn.Conv2d(channels, policy_channels, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_channels)
        self.policy_conv2 = nn.Conv2d(policy_channels, 64, 1)

    def forward(self, x):
        # x: (B, 12, 8, 8)
        out = self.relu(self.input_bn(self.input_conv(x)))
        out = self.residual_tower(out)

        p = self.relu(self.policy_bn(self.policy_conv1(out)))
        p = self.policy_conv2(p)  # (B, 64_to, 8_h_from, 8_w_from)

        # Match the (B, 4096) label convention: flat[from_pc * 64 + to_pc]
        # where from_pc / to_pc are python-chess square indices.
        # The featurization places FEN row 0 (rank 8) at spatial h=0, but
        # python-chess square 0 (a1) corresponds to rank 1 — so we flip h
        # to align spatial (h, w) with python-chess square = h * 8 + w.
        p = p.flip(2)
        p = p.permute(0, 2, 3, 1).reshape(p.size(0), 4096)
        return p


def create_model():
    return ChessModel()

# -*- coding: utf-8 -*-
import torch
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

    Input:  (B, 6, 8, 8)  — 6 piece-type planes on an 8×8 board
    Output: (B, 4096)      — raw logits over 64×64 from-to square pairs
    """

    def __init__(self, num_blocks=10, channels=128, policy_channels=32):
        super().__init__()

        # Input convolution
        self.input_conv = nn.Conv2d(6, channels, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        # Residual tower
        self.residual_tower = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(channels, policy_channels, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_channels)
        self.policy_fc = nn.Linear(policy_channels * 8 * 8, 4096)

    def forward(self, x):
        # x: (B, 6, 8, 8)
        out = self.relu(self.input_bn(self.input_conv(x)))
        out = self.residual_tower(out)

        # Policy head
        p = self.relu(self.policy_bn(self.policy_conv(out)))
        p = p.flatten(1)  # (B, policy_channels * 64)
        p = self.policy_fc(p)  # (B, 4096)
        return p


def create_model():
    return ChessModel()


def load_model(filename, device=None):
    """Load a saved .pt model file.

    Args:
        filename: path to .pt file, or a bare name resolved as model/<name>.pt
        device: torch device (auto-detected if None)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not filename.endswith('.pt'):
        filename = 'model/' + filename + '.pt'

    model = ChessModel()
    model.load_state_dict(torch.load(filename, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def save_model(model, filename):
    """Save model state dict as a .pt file into model/ directory."""
    path = 'model/' + filename + '.pt'
    torch.save(model.state_dict(), path)

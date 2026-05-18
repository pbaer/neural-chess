# -*- coding: utf-8 -*-
"""Differentiable internal lookahead block for v2 (T0b+).

Single-forward-pass "thinking ahead" — no external search loop. Given the
encoder's root latent state s_0, generates K candidate moves, applies a
learned dynamics network to produce K next-position latents, optionally
recurses for D plies, then aggregates all leaf latents via attention back
into a refined root representation.

The whole thing is differentiable end-to-end. Per
memory/project-principles.md Principle 3, this is "internal computation
within one forward pass" — distinct from MCTS which is an external loop.

Design notes:
  - The proposer (which picks top-K candidates) is a small dedicated head
    inside this block, not the model's main policy head. Cleaner gradients
    and avoids accidentally training the main head with the lookahead
    objective.
  - Top-K is non-differentiable through the selection itself, but gradients
    flow normally through the leaf latents and the aggregation. The proposer
    learns by getting better at picking candidates whose downstream rollouts
    improve the final policy/value losses.
  - Action encoding: 2 binary planes (from-square mark, to-square mark).
    Minimal — the dynamics network learns to interpret these as a move
    being played. Doesn't need to know the move type explicitly because
    the geometric (from, to) pattern conveys most of the action info.
  - Aggregation: spatial-mean-pool each leaf to a (channels,) token, run
    multi-head attention over (root_token + leaf_tokens), use the root
    token's output as a refinement that's broadcast back over spatial dims
    and added to s_0. The residual structure means the model can ignore
    the lookahead (set refinement to zero) early in training if it isn't
    helpful yet, and rely on it more as the dynamics network learns.
"""
import torch
import torch.nn as nn

from src.v2.moves import NUM_MOVES, NUM_MOVE_TYPES, _DECODE_TABLE
from src.v2.model import ResidualBlock


class LookaheadBlock(nn.Module):
    """Differentiable internal lookahead with attention aggregation.

    Args:
        channels: latent state channel count (matches encoder)
        K: branching factor (candidates considered at each ply)
        depth: number of plies to roll out
        proposer_channels: intermediate width for the candidate-proposer head
        dynamics_layers: residual blocks inside the dynamics network
        aggregator_heads: attention heads for the leaf aggregator
        aggregator_layers: transformer layers for the aggregator
    """

    def __init__(self, channels: int, K: int = 2, depth: int = 1,
                 proposer_channels: int = 32,
                 dynamics_layers: int = 3,
                 aggregator_heads: int = 4,
                 aggregator_layers: int = 2):
        super().__init__()
        assert depth >= 1, "depth must be >= 1"
        assert K >= 1, "K must be >= 1"
        self.channels = channels
        self.K = K
        self.depth = depth
        self.n_leaves = K ** depth

        # Candidate proposer: 1x1 conv -> BN+ReLU -> 1x1 conv -> 73 channels.
        # Same shape as the main policy head but a separate copy.
        self.proposer_conv1 = nn.Conv2d(channels, proposer_channels, 1, bias=False)
        self.proposer_bn = nn.BatchNorm2d(proposer_channels)
        self.proposer_conv2 = nn.Conv2d(proposer_channels, NUM_MOVE_TYPES, 1)

        # Dynamics network: latent state + action planes -> next latent state.
        # Action is encoded as 2 binary planes (from-square mark, to-square mark).
        self.dynamics_in_conv = nn.Conv2d(channels + 2, channels, 3, padding=1, bias=False)
        self.dynamics_in_bn = nn.BatchNorm2d(channels)
        self.dynamics_blocks = nn.ModuleList(
            [ResidualBlock(channels) for _ in range(dynamics_layers)]
        )

        # Aggregator: multi-head attention over (root + all leaf) tokens.
        # Each token is a spatial-mean-pool of a latent state -> (channels,).
        agg_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=aggregator_heads,
            dim_feedforward=channels * 2,
            dropout=0.0,
            batch_first=True,
        )
        self.aggregator = nn.TransformerEncoder(agg_layer, num_layers=aggregator_layers)

        # Move-decode table as a buffer (no gradient, just lookup).
        # _DECODE_TABLE[from_sq, move_type] = (from_sq, to_sq, promotion-or--1).
        # We only need column 1 (to_sq) for dynamics action encoding.
        self.register_buffer(
            'to_sq_table',
            torch.from_numpy(_DECODE_TABLE[:, :, 1].copy()).long(),  # (64, 73)
            persistent=False,
        )

        self.relu = nn.ReLU(inplace=True)

    def _propose(self, s: torch.Tensor) -> torch.Tensor:
        """Compute candidate-move logits at latent state s.
        Returns: (B, NUM_MOVES) logits.
        """
        p = self.relu(self.proposer_bn(self.proposer_conv1(s)))
        p = self.proposer_conv2(p)  # (B, 73, 8, 8)
        return p.permute(0, 2, 3, 1).reshape(s.size(0), NUM_MOVES)

    def _encode_action(self, from_sq: torch.Tensor, to_sq: torch.Tensor) -> torch.Tensor:
        """Build 2-plane action encoding for a batch.
        from_sq, to_sq: (B,) long tensors
        Returns: (B, 2, 8, 8)
        """
        B = from_sq.shape[0]
        device = from_sq.device
        planes = torch.zeros(B, 2, 64, device=device, dtype=torch.float32)
        # Defensive: clamp to_sq to valid range (decode_table may have -1 for unmapped slots,
        # but top-K candidates should hit valid slots)
        to_sq_safe = to_sq.clamp(min=0, max=63)
        batch_idx = torch.arange(B, device=device)
        planes[batch_idx, 0, from_sq] = 1.0
        planes[batch_idx, 1, to_sq_safe] = 1.0
        return planes.view(B, 2, 8, 8)

    def _step_dynamics(self, s: torch.Tensor, from_sq: torch.Tensor,
                       to_sq: torch.Tensor) -> torch.Tensor:
        """Apply learned dynamics: (state, action) -> next_state."""
        action = self._encode_action(from_sq, to_sq)  # (B, 2, 8, 8)
        x = torch.cat([s, action], dim=1)  # (B, channels+2, 8, 8)
        out = self.relu(self.dynamics_in_bn(self.dynamics_in_conv(x)))
        for block in self.dynamics_blocks:
            out = block(out)
        return out

    def _rollout(self, s: torch.Tensor, remaining_depth: int) -> list:
        """Recursive rollout. Returns list of leaf latent tensors.
        At the bottom (remaining_depth == 0), returns [s].
        """
        if remaining_depth == 0:
            return [s]
        # Propose K candidates at this state
        logits = self._propose(s)               # (B, 4672)
        topk = logits.topk(self.K, dim=-1).indices  # (B, K) — selected move indices

        leaves = []
        for k in range(self.K):
            move_idx = topk[:, k]              # (B,)
            from_sq = (move_idx // NUM_MOVE_TYPES).long()
            move_type = (move_idx % NUM_MOVE_TYPES).long()
            # Lookup to_sq via the precomputed table
            to_sq = self.to_sq_table[from_sq, move_type]   # (B,)
            s_next = self._step_dynamics(s, from_sq, to_sq)
            leaves.extend(self._rollout(s_next, remaining_depth - 1))
        return leaves

    def forward(self, s_root: torch.Tensor) -> torch.Tensor:
        """s_root: (B, channels, 8, 8) encoder output.
        Returns: (B, channels, 8, 8) refined latent (additive residual update).
        """
        # Roll out the search tree
        leaves = self._rollout(s_root, self.depth)
        # leaves: list of n_leaves tensors, each (B, channels, 8, 8)

        # Pool each leaf and the root to (B, channels) tokens
        pooled_leaves = [leaf.mean(dim=[2, 3]) for leaf in leaves]
        root_pooled = s_root.mean(dim=[2, 3])  # (B, channels)

        # Stack: (B, 1 + n_leaves, channels)
        tokens = torch.stack([root_pooled] + pooled_leaves, dim=1)

        # Multi-head attention over tokens
        attended = self.aggregator(tokens)  # (B, 1 + n_leaves, channels)

        # Take the root token's output as the refinement vector
        refinement = attended[:, 0, :]  # (B, channels)

        # Broadcast over spatial dims and add to the encoder output
        # (Additive residual update: the lookahead's job is to refine, not replace.)
        return s_root + refinement.unsqueeze(-1).unsqueeze(-1)

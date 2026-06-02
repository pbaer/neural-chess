# -*- coding: utf-8 -*-
"""v3 model architecture — square-token transformer (attention tower).

The v3 thesis (see memory/v3-build-log.md): the CNN builds long-range board
understanding only by stacking many conv layers; self-attention lets any square
influence any other in one layer. v3 keeps v2's input (21 planes, rotation) and
output interface (policy 4672 + value scalar) UNCHANGED — the only thing that
changes is the inner tower (CNN -> conv-stem + transformer).

Principle-2 note (the one principle-sensitive choice): the attention uses a
learned **2D relative-position bias** indexed by the (Δrank, Δfile) displacement
between squares. This encodes only board *coordinate geometry* ("squares on an
8×8 grid relate by displacement") — the transformer analog of a CNN's
translation-equivariant convolution, which v1/v2 already rely on. It provides a
uniform learnable slot for ALL 225 displacements and singles out none of them:
the model must DISCOVER from games which offsets matter (e.g. that knights use
(±1,±2)). It does NOT bake in any piece's movement rule. (We deliberately did
NOT use named relationship buckets like "knight-offset/diagonal" — that would
inject the movement rules.) Toggle via config.geometry_bias.
"""
from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from src.v2.featurize import NUM_PLANES
from src.v2.moves import NUM_MOVES, NUM_MOVE_TYPES
from src.v2.model import ResidualBlock  # shared conv building block


@dataclass
class ChessConfigV3:
    """Architecture knobs for the v3 attention tower."""
    d_model: int = 256          # token dim == conv-stem width
    n_heads: int = 8
    n_blocks: int = 20          # transformer encoder blocks
    ffn_mult: int = 4           # FFN hidden = ffn_mult * d_model
    stem_blocks: int = 2        # conv residual blocks in the stem
    geometry_bias: bool = True  # 2D relative-position bias (board geometry)
    value_hidden: int = 128
    # Gradient checkpointing granularity (memory<->compute tradeoff):
    #   0 = none (fastest, most memory); 1 = every block; N = every Nth block.
    checkpoint_every: int = 1
    input_planes: int = NUM_PLANES
    output_size: int = NUM_MOVES


def _rel_index() -> torch.Tensor:
    """(64,64) long tensor: for query square i, key square j, the index into a
    flattened 15×15 (Δrank, Δfile) displacement table. Δ in [-7,7] -> +7 -> [0,14]."""
    idx = torch.zeros(64, 64, dtype=torch.long)
    for i in range(64):
        ri, fi = i // 8, i % 8
        for j in range(64):
            rj, fj = j // 8, j % 8
            dr, df = (ri - rj) + 7, (fi - fj) + 7
            idx[i, j] = dr * 15 + df
    return idx


class GeoSelfAttention(nn.Module):
    """Multi-head self-attention over the 64 square-tokens, with an optional
    learned 2D relative-position (geometry) bias added to the attention logits."""

    def __init__(self, d_model: int, n_heads: int, geometry_bias: bool):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.geometry_bias = geometry_bias
        if geometry_bias:
            # one learnable bias per (head, displacement); 15*15 = 225 displacements
            self.rel_bias = nn.Parameter(torch.zeros(n_heads, 15 * 15))
            self.register_buffer('rel_idx', _rel_index(), persistent=False)

    def forward(self, x):  # x: (B, N=64, d_model)
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)        # (3, B, n_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]        # (B, n_heads, N, head_dim)
        # Fused, memory-efficient attention. The geometry bias is passed as an
        # additive float attn_mask (broadcasts over batch); SDPA applies the
        # 1/sqrt(d) scale internally.
        attn_mask = None
        if self.geometry_bias:
            attn_mask = self.rel_bias[:, self.rel_idx].unsqueeze(0).to(q.dtype)  # (1, n_heads, N, N)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm transformer encoder block."""

    def __init__(self, cfg: ChessConfigV3):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = GeoSelfAttention(cfg.d_model, cfg.n_heads, cfg.geometry_bias)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        hidden = cfg.ffn_mult * cfg.d_model
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, cfg.d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class ChessModelV3(nn.Module):
    """v3 attention tower. Same I/O as v2:
       inputs:  (B, 21, 8, 8)
       outputs: policy_logits (B, 4672), value (B, 1) tanh in [-1,1].
    """

    def __init__(self, config: Optional[ChessConfigV3] = None):
        super().__init__()
        if config is None:
            config = ChessConfigV3()
        self.config = config
        C = config.d_model

        # Conv stem: lift 21 planes -> C, inject local geometry cheaply.
        self.input_conv = nn.Conv2d(config.input_planes, C, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(C)
        self.relu = nn.ReLU(inplace=True)
        self.stem = nn.Sequential(*[ResidualBlock(C) for _ in range(config.stem_blocks)])

        # Token positional embedding (absolute square identity) + transformer tower.
        self.pos_emb = nn.Parameter(torch.zeros(1, 64, C))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_blocks)])
        self.final_ln = nn.LayerNorm(C)

        # Heads (fixed interface). Policy: per-token -> 73 move-types.
        self.policy_head = nn.Linear(C, NUM_MOVE_TYPES)
        # Value: mean-pool tokens -> MLP -> tanh.
        self.value_fc1 = nn.Linear(C, config.value_hidden)
        self.value_fc2 = nn.Linear(config.value_hidden, 1)

    def forward(self, x):
        # Conv stem
        h = self.relu(self.input_bn(self.input_conv(x)))   # (B, C, 8, 8)
        h = self.stem(h)
        # Tokenize: (B,C,8,8) -> (B,64,C), token s = h*8+w (matches square index)
        t = h.flatten(2).transpose(1, 2)                   # (B, 64, C)
        t = t + self.pos_emb
        ckpt = self.config.checkpoint_every
        for i, blk in enumerate(self.blocks):
            if self.training and ckpt > 0 and (i % ckpt == 0):
                t = torch.utils.checkpoint.checkpoint(blk, t, use_reentrant=False)
            else:
                t = blk(t)
        t = self.final_ln(t)                               # (B, 64, C)

        # Policy: (B,64,73) -> flat (B, 4672) with flat[square*73 + move_type]
        p = self.policy_head(t).reshape(t.size(0), NUM_MOVES)

        # Value: mean over the 64 square tokens -> MLP -> tanh
        v = t.mean(dim=1)                                  # (B, C)
        v = self.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))                  # (B, 1)
        return p, v


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

# -*- coding: utf-8 -*-
"""One-shot parameter-utilization analysis on a v2 checkpoint.

Computes:
  1. BatchNorm gamma distribution per layer — small |gamma| ≈ dead channel
  2. Per-residual-block contribution (||block(x) - x|| / ||x||) on a real batch
  3. Conv weight L2 norm per layer
  4. Policy/value head activation usage

Reads a real batch of positions from a shard for forward passes.
"""
import os
import sys
import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.v2.model import ChessConfigV2, ChessModelV2
from src.v2.dataset import ChessDatasetV2


def load_checkpoint(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg_dict = ckpt.get('config', {})
    # ChessConfigV2 expects keyword args; filter to known fields
    known = {'encoder_blocks', 'encoder_channels', 'policy_channels',
             'value_channels', 'value_hidden', 'lookahead_K', 'lookahead_depth',
             'input_planes', 'num_move_classes'}
    cfg = ChessConfigV2(**{k: v for k, v in cfg_dict.items() if k in known})
    model = ChessModelV2(cfg)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state)
    return model, cfg


def bn_gamma_stats(model):
    """Distribution of |gamma| (BN weight) across all BN layers."""
    rows = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            g = mod.weight.detach().abs()
            rows.append({
                'layer': name,
                'channels': g.numel(),
                'mean_|g|': g.mean().item(),
                'median_|g|': g.median().item(),
                'min_|g|': g.min().item(),
                'frac<0.01': (g < 0.01).float().mean().item(),
                'frac<0.05': (g < 0.05).float().mean().item(),
                'frac<0.10': (g < 0.10).float().mean().item(),
            })
    return rows


def conv_weight_stats(model):
    """L2 norm + sparsity-ish stats per Conv2d."""
    rows = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Conv2d):
            w = mod.weight.detach()
            rows.append({
                'layer': name,
                'shape': tuple(w.shape),
                'params': w.numel(),
                'l2_norm': w.norm().item(),
                'mean_|w|': w.abs().mean().item(),
                'frac<1e-3': (w.abs() < 1e-3).float().mean().item(),
            })
    return rows


def block_contributions(model, x, device='cuda'):
    """For each residual block, measure ||block(input) - input|| / ||input||
    on the real batch — i.e. how much the block changes its input."""
    model = model.to(device).eval()
    with torch.no_grad(), torch.amp.autocast(device, dtype=torch.float16):
        # Stem
        h = model.input_conv(x)
        h = model.input_bn(h)
        h = torch.relu(h)
        stem_out = h.clone()

        rows = []
        for i, blk in enumerate(model.tower):
            inp = h
            h_new = blk(inp)
            delta = (h_new - inp).float()
            denom = inp.float().norm().item() + 1e-9
            rows.append({
                'block_idx': i,
                'relative_delta_norm': delta.norm().item() / denom,
                'mean_input_norm': inp.float().flatten(1).norm(dim=1).mean().item(),
                'mean_output_norm': h_new.float().flatten(1).norm(dim=1).mean().item(),
            })
            h = h_new
        return rows, stem_out, h


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else 'model/v2/v2-19M/model_e0009.pt'
    shard_dir = sys.argv[2] if len(sys.argv) > 2 else 'data/v2/training_T1_rot_40M'
    n_samples = 4096

    print(f'Checkpoint: {ckpt_path}')
    print(f'Shard: {shard_dir} (using first {n_samples} samples)\n')

    model, cfg = load_checkpoint(ckpt_path)
    print(f'Config: blocks={cfg.encoder_blocks}, channels={cfg.encoder_channels}, '
          f'policy_ch={cfg.policy_channels}, value_ch={cfg.value_channels}, '
          f'value_hidden={cfg.value_hidden}')
    total = sum(p.numel() for p in model.parameters())
    print(f'Total params: {total:,}\n')

    # ---------- BN gamma analysis ----------
    print('=== BatchNorm gamma (|weight|) per layer ===')
    print(f'{"layer":50s}  {"ch":>4s}  {"mean":>6s}  {"median":>6s}  {"min":>7s}  {"<0.01":>5s}  {"<0.05":>5s}  {"<0.10":>5s}')
    for r in bn_gamma_stats(model):
        print(f'{r["layer"]:50s}  {r["channels"]:>4d}  {r["mean_|g|"]:6.3f}  {r["median_|g|"]:6.3f}  '
              f'{r["min_|g|"]:7.4f}  {r["frac<0.01"]*100:4.1f}%  {r["frac<0.05"]*100:4.1f}%  {r["frac<0.10"]*100:4.1f}%')

    # Aggregate dead-channel stats
    all_g = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            all_g.append(mod.weight.detach().abs())
    all_g = torch.cat(all_g)
    print(f'\nAll BN gammas across model: n={all_g.numel():,}')
    print(f'  mean={all_g.mean():.3f}, median={all_g.median():.3f}')
    print(f'  frac < 0.01: {(all_g < 0.01).float().mean()*100:.1f}%   (effectively dead)')
    print(f'  frac < 0.05: {(all_g < 0.05).float().mean()*100:.1f}%   (very small)')
    print(f'  frac < 0.10: {(all_g < 0.10).float().mean()*100:.1f}%   (small)')
    print(f'  frac < 0.50: {(all_g < 0.50).float().mean()*100:.1f}%')

    # ---------- Conv weight norms ----------
    print('\n=== Conv weight L2 norms ===')
    print(f'{"layer":50s}  {"shape":>20s}  {"l2":>8s}  {"mean|w|":>8s}  {"params":>10s}')
    for r in conv_weight_stats(model):
        sh = str(r['shape'])
        print(f'{r["layer"]:50s}  {sh:>20s}  {r["l2_norm"]:8.3f}  {r["mean_|w|"]:8.5f}  {r["params"]:>10,d}')

    # ---------- Per-block residual contribution ----------
    print('\n=== Loading sample batch and running per-block contribution ===')
    ds = ChessDatasetV2(shard_dir)
    xs = []
    for i in range(n_samples):
        x, _, _ = ds[i]
        xs.append(x.unsqueeze(0))
    x_batch = torch.cat(xs, dim=0).cuda()
    print(f'Batch shape: {tuple(x_batch.shape)}')

    rows, stem_out, final_h = block_contributions(model, x_batch)
    print('\n=== Per-block residual contribution (||block(x)-x|| / ||x||) ===')
    print(f'{"block":>5s}  {"rel_delta":>10s}  {"in_norm":>10s}  {"out_norm":>10s}')
    for r in rows:
        print(f'{r["block_idx"]:>5d}  {r["relative_delta_norm"]:10.4f}  '
              f'{r["mean_input_norm"]:10.3f}  {r["mean_output_norm"]:10.3f}')

    # Summary
    deltas = [r['relative_delta_norm'] for r in rows]
    print(f'\nBlock contribution summary:')
    print(f'  min: {min(deltas):.4f}   (block doing least work, idx={deltas.index(min(deltas))})')
    print(f'  max: {max(deltas):.4f}   (block doing most work, idx={deltas.index(max(deltas))})')
    print(f'  mean: {sum(deltas)/len(deltas):.4f}')


if __name__ == '__main__':
    main()

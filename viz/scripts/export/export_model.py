# -*- coding: utf-8 -*-
"""Export a checkpoint to a versioned, self-describing **Model Capsule** — the only
thing the web tool loads (it never reads a .pt). The capsule is architecture-
version-NEUTRAL: the architecture is described by a `graph` of typed stages, so the
tool renders whatever it finds.

Output (viz/public/weights/<model_id>/):
  capsule.json   manifest: capsule_version, arch (display-only), graph[], tensors[] index, sha256
  weights.bin    every tensor concatenated as little-endian float32 (BatchNorm folded)
  config.json    raw checkpoint config (reference)

Usage (from repo root, with the venv python):
  .venv/Scripts/python.exe viz/scripts/export/export_model.py \
      --ckpt model/v3/v3.1-nano/v3.1-nano_e0015.pt \
      --out-dir viz/public/weights/v3.1-nano --model-id v3.1-nano --arch v3.1
"""
import argparse
import datetime
import hashlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fold_bn import fold_conv_bn

CAPSULE_VERSION = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--model-id', required=True)
    ap.add_argument('--arch', default=None, help='display-only tag; default = ckpt arch')
    args = ap.parse_args()

    d = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    sd = {k: v.cpu().numpy() for k, v in d['model'].items()}
    cfg = dict(d.get('config', {}))
    arch = args.arch or d.get('arch', 'v3')

    # ---- resolve architecture from tensor shapes (the source of truth) ----
    C = int(sd['input_conv.weight'].shape[0])
    in_planes = int(sd['input_conv.weight'].shape[1])
    stem_kernel = int(sd['input_conv.weight'].shape[-1])
    n_blocks = 1 + max(int(k.split('.')[1]) for k in sd if k.startswith('blocks.'))
    stem_blocks = (1 + max(int(k.split('.')[1]) for k in sd if k.startswith('stem.'))) \
        if any(k.startswith('stem.') for k in sd) else 0
    geometry_bias = 'blocks.0.attn.rel_bias' in sd
    n_heads = int(cfg.get('n_heads', 8))
    head_dim = C // n_heads
    value_hidden = int(sd['value_fc1.weight'].shape[0])
    ffn_hidden = int(sd['blocks.0.ffn.0.weight'].shape[0])
    ffn_mult = ffn_hidden // C

    # ---- fold BatchNorm into the conv(s) ----
    folded = {}
    W, b = fold_conv_bn(sd['input_conv.weight'], sd['input_bn.weight'], sd['input_bn.bias'],
                        sd['input_bn.running_mean'], sd['input_bn.running_var'])
    folded['stem.embed.weight'], folded['stem.embed.bias'] = W, b
    for i in range(stem_blocks):
        for c in (1, 2):
            W2, b2 = fold_conv_bn(sd[f'stem.{i}.conv{c}.weight'], sd[f'stem.{i}.bn{c}.weight'],
                                  sd[f'stem.{i}.bn{c}.bias'], sd[f'stem.{i}.bn{c}.running_mean'],
                                  sd[f'stem.{i}.bn{c}.running_var'])
            folded[f'stem.block{i}.conv{c}.weight'], folded[f'stem.block{i}.conv{c}.bias'] = W2, b2

    # ---- ordered tensor list (folded conv tensors + pass-through fp32) ----
    tensors = []
    def add(name, arr): tensors.append((name, np.ascontiguousarray(arr, dtype=np.float32)))
    add('stem.embed.weight', folded['stem.embed.weight'])
    add('stem.embed.bias', folded['stem.embed.bias'])
    for i in range(stem_blocks):
        for c in (1, 2):
            add(f'stem.block{i}.conv{c}.weight', folded[f'stem.block{i}.conv{c}.weight'])
            add(f'stem.block{i}.conv{c}.bias', folded[f'stem.block{i}.conv{c}.bias'])
    add('pos_emb', sd['pos_emb'])
    for i in range(n_blocks):
        p = f'blocks.{i}.'
        for nm in ['ln1.weight', 'ln1.bias', 'attn.qkv.weight', 'attn.qkv.bias',
                   'attn.proj.weight', 'attn.proj.bias']:
            add(p + nm, sd[p + nm])
        if geometry_bias:
            add(p + 'attn.rel_bias', sd[p + 'attn.rel_bias'])
        for nm in ['ln2.weight', 'ln2.bias', 'ffn.0.weight', 'ffn.0.bias',
                   'ffn.2.weight', 'ffn.2.bias']:
            add(p + nm, sd[p + nm])
    for nm in ['final_ln.weight', 'final_ln.bias', 'policy_head.weight', 'policy_head.bias',
               'value_fc1.weight', 'value_fc1.bias', 'value_fc2.weight', 'value_fc2.bias']:
        add(nm, sd[nm])

    # ---- write weights.bin + build the tensor index ----
    os.makedirs(args.out_dir, exist_ok=True)
    index, blob, off = [], [], 0
    for name, arr in tensors:
        flat = arr.reshape(-1).astype('<f4')
        index.append({'name': name, 'shape': list(arr.shape), 'dtype': 'f32',
                      'offset': off, 'length': int(flat.size)})
        blob.append(flat.tobytes()); off += int(flat.size)
    raw = b''.join(blob)
    with open(os.path.join(args.out_dir, 'weights.bin'), 'wb') as f:
        f.write(raw)
    sha = hashlib.sha256(raw).hexdigest()

    # ---- self-describing architecture graph (typed stages) ----
    g = [{'id': 'io.planes', 'kind': 'input_planes',
          'dims': {'planes': in_planes, 'h': 8, 'w': 8}, 'weights': [], 'reads': []}]
    if stem_kernel == 1 and stem_blocks == 0:
        g.append({'id': 'stem.embed', 'kind': 'embed',
                  'dims': {'in': in_planes, 'out': C, 'kernel': 1},
                  'weights': ['stem.embed.weight', 'stem.embed.bias'], 'reads': ['io.planes']})
        last = 'stem.embed'
    else:
        g.append({'id': 'stem.conv', 'kind': 'stem_conv',
                  'dims': {'in': in_planes, 'out': C, 'kernel': stem_kernel},
                  'weights': ['stem.embed.weight', 'stem.embed.bias'], 'reads': ['io.planes']})
        last = 'stem.conv'
        for i in range(stem_blocks):
            g.append({'id': f'stem.block{i}', 'kind': 'stem_block', 'dims': {'c': C},
                      'weights': [f'stem.block{i}.conv{c}.{w}' for c in (1, 2) for w in ('weight', 'bias')],
                      'reads': [last]})
            last = f'stem.block{i}'
    g.append({'id': 'tokenize', 'kind': 'tokenize', 'dims': {'tokens': 64, 'd': C},
              'weights': ['pos_emb'], 'reads': [last]})
    last = 'tokenize'
    for i in range(n_blocks):
        p = f'blocks.{i}.'
        w = [p + nm for nm in ['ln1.weight', 'ln1.bias', 'attn.qkv.weight', 'attn.qkv.bias',
                               'attn.proj.weight', 'attn.proj.bias']]
        if geometry_bias:
            w.append(p + 'attn.rel_bias')
        w += [p + nm for nm in ['ln2.weight', 'ln2.bias', 'ffn.0.weight', 'ffn.0.bias',
                                'ffn.2.weight', 'ffn.2.bias']]
        g.append({'id': f'block.{i}', 'kind': 'block',
                  'dims': {'d': C, 'heads': n_heads, 'head_dim': head_dim,
                           'ffn': ffn_hidden, 'geometry_bias': geometry_bias},
                  'weights': w, 'reads': [last]})
        last = f'block.{i}'
    g.append({'id': 'final_ln', 'kind': 'layernorm', 'dims': {'d': C},
              'weights': ['final_ln.weight', 'final_ln.bias'], 'reads': [last]})
    g.append({'id': 'head.policy', 'kind': 'policy_head',
              'dims': {'in': C, 'move_types': 73, 'moves': 4672},
              'weights': ['policy_head.weight', 'policy_head.bias'], 'reads': ['final_ln']})
    g.append({'id': 'head.value', 'kind': 'value_head', 'dims': {'in': C, 'hidden': value_hidden},
              'weights': ['value_fc1.weight', 'value_fc1.bias', 'value_fc2.weight', 'value_fc2.bias'],
              'reads': ['final_ln']})

    nominal = int(sum(v.size for k, v in sd.items()
                      if not k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))))
    manifest = {
        'capsule_version': CAPSULE_VERSION, 'arch': arch, 'model_id': args.model_id,
        'created': datetime.datetime.now().isoformat(timespec='seconds'),
        'param_count': nominal, 'stored_floats': int(off), 'folded_bn': True,
        'config': {'d_model': C, 'n_heads': n_heads, 'n_blocks': n_blocks, 'ffn_mult': ffn_mult,
                   'stem_kernel': stem_kernel, 'stem_blocks': stem_blocks,
                   'value_hidden': value_hidden, 'geometry_bias': geometry_bias,
                   'input_planes': in_planes},
        'weights_file': 'weights.bin', 'weights_bytes': len(raw), 'weights_sha256': sha,
        'tensors': index, 'graph': g,
    }
    with open(os.path.join(args.out_dir, 'capsule.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f"Model Capsule -> {args.out_dir}")
    print(f"  arch={arch} id={args.model_id} | nominal params={nominal:,} | stored floats={off:,} "
          f"(BN-fold {off - nominal:+d}) | weights.bin {len(raw):,} B | sha {sha[:12]}")
    print(f"  {len(index)} tensors, {len(g)} graph stages "
          f"(d{C} h{n_heads} b{n_blocks} ffn{ffn_mult} stem_k{stem_kernel}/{stem_blocks} geo={geometry_bias})")


if __name__ == '__main__':
    main()

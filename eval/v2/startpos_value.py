# -*- coding: utf-8 -*-
"""What value does each v2 model assign to the initial chess position?

The value head outputs a scalar in [-1, +1] = expected game outcome from the
side-to-move's perspective. At the start, white is to move, so the output is
the model's learned estimate of white's first-move advantage.

Also computes the empirical white score from the training corpus index
(games_index.parquet) as the ground-truth target the value head is fit to.
"""
import glob
import os
import sys

import chess
import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.v2.model import ChessConfigV2, ChessModelV2
from src.v2.featurize import featurize


def load_model(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    known = {'encoder_blocks', 'encoder_channels', 'policy_channels',
             'value_channels', 'value_hidden', 'lookahead_K', 'lookahead_depth',
             'input_planes', 'num_move_classes'}
    cfg = ChessConfigV2(**{k: v for k, v in ckpt.get('config', {}).items() if k in known})
    m = ChessModelV2(cfg)
    m.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    m.eval()
    return m


def start_value(model):
    board = chess.Board()
    x = featurize(board)  # (21, 8, 8) float32, white to move (no rotation)
    xt = torch.from_numpy(np.asarray(x)).unsqueeze(0).float()
    with torch.no_grad():
        _, v = model(xt)
    return float(v.squeeze().item())


def empirical_white_score():
    """From games_index.parquet, compute white W/D/L and expected value."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None
    path = os.path.join(_REPO, 'data', 'v2', 'games_index.parquet')
    if not os.path.exists(path):
        return None
    t = pq.read_table(path, columns=None)
    cols = t.column_names
    # find the result column
    result_col = None
    for cand in ('result', 'Result', 'outcome'):
        if cand in cols:
            result_col = cand
            break
    if result_col is None:
        return {'columns': cols}  # report schema so we can adapt
    results = t.column(result_col).to_pylist()
    w = sum(1 for r in results if r in ('1-0', 1.0, '1'))
    l = sum(1 for r in results if r in ('0-1', 0.0, '0'))
    d = sum(1 for r in results if r in ('1/2-1/2', 0.5, '1/2'))
    n = w + l + d
    if n == 0:
        return {'columns': cols, 'sample_values': results[:10]}
    exp_val = (w - l) / n  # white's expected value in [-1, +1]
    return {'n': n, 'white_win': w/n, 'draw': d/n, 'black_win': l/n,
            'white_score_pct': 100*(w + 0.5*d)/n, 'expected_value': exp_val}


def main():
    print('=== Empirical white advantage in training corpus ===')
    emp = empirical_white_score()
    if emp is None:
        print('  (games_index.parquet not found or pyarrow missing)')
    elif 'expected_value' not in emp:
        print(f'  Could not find result column. Schema: {emp}')
    else:
        print(f"  Games: {emp['n']:,}")
        print(f"  White win: {emp['white_win']*100:.1f}%  Draw: {emp['draw']*100:.1f}%  Black win: {emp['black_win']*100:.1f}%")
        print(f"  White score: {emp['white_score_pct']:.1f}%  (draws = 0.5)")
        print(f"  => Expected value from white's perspective: {emp['expected_value']:+.4f}")
        print(f"     (this is the target the value head is fit toward for the start position)")

    print('\n=== Model value-head predictions for the starting position ===')
    print(f'{"model":18s}  {"ckpt":16s}  {"start value":>12s}  {"as score%":>10s}')

    # Order models by approximate param count
    order = ['v2-2M', 'v2-3M', 'v2-5M-a', 'v2-5M-b', 'v2-8M-a', 'v2-8M-b',
             'v2-12M', 'v2-14M', 'v2-19M', 'v2-37M']
    for name in order:
        d = os.path.join(_REPO, 'model', 'v2', name)
        if not os.path.isdir(d):
            continue
        pts = sorted(glob.glob(os.path.join(d, '*.pt')))
        if not pts:
            continue
        last = pts[-1]
        try:
            m = load_model(last)
            v = start_value(m)
            score = (v + 1) / 2 * 100  # convert [-1,1] -> [0,100]% white score
            print(f'{name:18s}  {os.path.basename(last):16s}  {v:>+12.4f}  {score:>9.1f}%')
        except Exception as e:
            print(f'{name:18s}  {os.path.basename(last):16s}  ERROR: {str(e)[:40]}')

    # Per-epoch trajectory for v2-37M (in progress)
    print('\n=== v2-37M value-head trajectory across epochs ===')
    print(f'{"epoch":>5s}  {"start value":>12s}  {"as score%":>10s}')
    d = os.path.join(_REPO, 'model', 'v2', 'v2-37M')
    for pt in sorted(glob.glob(os.path.join(d, '*.pt'))):
        ep = os.path.basename(pt).replace('model_e', '').replace('.pt', '')
        try:
            m = load_model(pt)
            v = start_value(m)
            score = (v + 1) / 2 * 100
            print(f'{ep:>5s}  {v:>+12.4f}  {score:>9.1f}%')
        except Exception as e:
            print(f'{ep:>5s}  ERROR: {str(e)[:40]}')


if __name__ == '__main__':
    main()

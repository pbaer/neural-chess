# -*- coding: utf-8 -*-
"""Pretty-print and rank the aggregated-data sweep metrics.

Reads eval/v3/agg_metrics.csv (written by src/v3/train_agg.py) and shows, for the
final epoch of each run, the held-out metrics with deltas vs the 'base' run.
Lower rmse/CE = better; higher top1/top3 = better.
"""
import csv
import os
import sys

CSV = sys.argv[1] if len(sys.argv) > 1 else 'eval/v3/agg_metrics.csv'
rows = list(csv.DictReader(open(CSV)))
if not rows:
    print("no rows"); sys.exit()

# keep the last epoch per save_name
last = {}
for r in rows:
    last[r['save_name']] = r  # rows appended in order; last wins

def f(r, k):
    try:
        return float(r[k])
    except (KeyError, ValueError):
        return float('nan')

cols = [('top1', 'u_top1', +1), ('top3', 'u_top3', +1), ('pol_ce', 'u_CE', -1),
        ('w_top1', 'w_top1', +1), ('w_pol_ce', 'w_CE', -1),
        ('val_rmse_one', 'rmse1', -1), ('val_rmse_avg', 'rmseA', -1),
        ('val_rmse_avg_hi', 'rmseHi', -1), ('w_rmse_avg', 'w_rmseA', -1)]

base = last.get('base')
order = sorted(last.values(), key=lambda r: -f(r, 'w_top1'))
hdr = f"{'run':10} {'val':4} {'pol':11} {'tau':4} " + ' '.join(f"{c[1]:>8}" for c in cols)
print(hdr)
print('-' * len(hdr))
for r in order:
    line = (f"{r['save_name']:10} {r['value_mode']:4} {r['policy_mode']:11} "
            f"{r['tau']:>4} ")
    cells = []
    for key, name, sign in cols:
        v = f(r, key)
        s = f"{v:8.4f}"
        if base is not None and r is not base:
            d = v - f(base, key)
            mark = '+' if d * sign > 0 else ('=' if abs(d) < 1e-9 else '-')
            s = f"{v:7.4f}{mark}"
        cells.append(s)
    print(line + ' '.join(cells))
print("\n(+ = better than base on that metric; sign-aware. ranked by w_top1.)")

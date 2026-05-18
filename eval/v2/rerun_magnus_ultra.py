# -*- coding: utf-8 -*-
"""One-shot: rerun magnus (d=8 s=20) + ultra (d=16 s=20) for a fixed set of
checkpoints. Writes results into the matching per-model eval CSVs."""
import csv
import datetime
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from eval.v2.eval_v2 import run_sf_eval

GAMES = 10
TARGETS = [
    # (label, ckpt path, csv path, epoch)
    ('T2-FAT-e7', 'model/v2/T2_FAT/model_e0007.pt', 'eval/v2/eval_T2_FAT.csv', 7),
    ('v2f-e0',    'model/v2/T_FINAL/model_e0000.pt', 'eval/v2/eval_T_FINAL.csv', 0),
    ('v2f-e1',    'model/v2/T_FINAL/model_e0001.pt', 'eval/v2/eval_T_FINAL.csv', 1),
    ('v2f-e2',    'model/v2/T_FINAL/model_e0002.pt', 'eval/v2/eval_T_FINAL.csv', 2),
]
OPPONENTS = [
    ('sf_magnus', 8, 20),
    ('sf_ultra', 16, 20),
]


def append_row(csv_path, row):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['epoch', 'timestamp', 'opponent', 'games',
                        'won_pct', 'draw_pct', 'lost_pct',
                        'white_won_pct', 'black_won_pct',
                        'illegal_pct', 'moves', 'elapsed_s'])
        w.writerow(row)


def main():
    for label, ckpt, csv_path, epoch in TARGETS:
        for opp, depth, skill in OPPONENTS:
            ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f'EVAL {ts} {label} ({ckpt}) opp={opp} depth={depth} skill={skill} games={GAMES}',
                  flush=True)
            r = run_sf_eval(ckpt, depth, skill, GAMES)
            if 'error' in r:
                print(f'  ERROR: {r["error"][:200]}', flush=True)
                append_row(csv_path, [epoch, ts, opp, GAMES, '', '', '', '', '', '', '',
                                      f'{r["elapsed_s"]:.1f}'])
            else:
                print(f'  W:{r["won"]:.0f}% D:{r["draw"]:.0f}% L:{r["lost"]:.0f}% '
                      f'(W:{r["white_won"]:.0f}% / B:{r["black_won"]:.0f}%) '
                      f'illegal:{r["illegal_pct"]:.1f}% '
                      f'({r["elapsed_s"]:.1f}s)', flush=True)
                append_row(csv_path, [epoch, ts, opp, GAMES,
                                      r['won'], r['draw'], r['lost'],
                                      r['white_won'], r['black_won'],
                                      r['illegal_pct'], r['moves'],
                                      f'{r["elapsed_s"]:.1f}'])
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()

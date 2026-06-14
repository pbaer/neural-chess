# -*- coding: utf-8 -*-
"""Distillation campaign: train 116k v3.1 students distilled from the v3-37M
teacher (targets precomputed by src/v3/teacher_label.py), and A/B each vs the
browser model v3.1-eq. Architecture + recipe are IDENTICAL to v3.1-eq — the only
change is adding the teacher's soft policy/value to the supervision (alpha mix,
temperature T). Goal: a CLEARLY stronger drop-in (h2h > 0.5 and SF ladder up).

Each config: train (16ep) -> h2h vs v3.1-eq (t0.3 + t0.5) -> SF ladder (easy/med/
hard, 120g). Run from repo root AFTER data/v2/agg_100M_teacher exists.
"""
import argparse
import csv
import os
import re
import subprocess
import time

PY = os.path.abspath('.venv/Scripts/python.exe')
PACKED = 'data/v2/agg_100M_packed'
TEACHER = 'data/v2/agg_100M_teacher'
EQ = 'model/v3/v3.1/v3.1-eq/v3.1-eq_e0015.pt'
LOG = 'logs/distill.log'
LOGDIR = 'eval/v3/logs'
EPOCH_SIZE = 8_000_000
EPOCHS = 16
# v3.1-eq SF baseline, same-session (eval/v3/eq_sf.csv): easy 70.0 / med 45.8 / hard 22.5
RUNS = [
    # iter1: pure-teacher a1.0/T2 WON (h2h 0.527, SF +6.7pp); mixes a0.5/0.7 hurt.
    # iter2: softer T HURT (T3 -> h2h 0.38), so the optimum is T<=2 / sharper.
    # iter2b (this run): probe SHARPER T + a slight human component.
    # (done & logged: D-a05-t2, D-a07-t3, D-a10-t2[best so far], D-a10-t3)
    dict(name='D-a10-t1', alpha=1.0, temp=1.0),
    dict(name='D-a10-t15', alpha=1.0, temp=1.5),
    dict(name='D-a09-t2', alpha=0.9, temp=2.0),
]


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def train(run):
    name, d = run['name'], f"model/v3/distill/{run['name']}"
    final = os.path.join(d, f"{name}_e{EPOCHS-1:04d}.pt")
    if os.path.isfile(final):
        log(f"SKIP train {name}"); return final
    cmd = [PY, '-u', '-m', 'src.v3.train_agg_fast', '--save-dir', d, '--save-name', name,
           '--packed-dir', PACKED, '--distill-dir', TEACHER,
           '--distill-alpha', str(run['alpha']), '--distill-temp', str(run['temp']),
           '--tau', '0.5', '--value-loss-weight', '1.0', '--epoch-size', str(EPOCH_SIZE),
           '--epochs', str(EPOCHS), '--lr-horizon', str(EPOCHS), '--lr', '1e-3',
           '--warmup-steps', '300', '--batch-size', '1024', '--seed', '0',
           '--save-every-steps', '2000',
           '--d-model', '32', '--n-heads', '4', '--n-blocks', '8', '--value-hidden', '64',
           '--stem-kernel', '1', '--stem-blocks', '0', '--ffn-mult', '4',
           '--metrics-csv', f'eval/v3/{name}_metrics.csv']
    log(f"TRAIN {name} (alpha={run['alpha']} temp={run['temp']})")
    with open(f"{LOGDIR}/{name}.log", 'a') as lf:
        try:
            subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=3 * 3600)
        except subprocess.TimeoutExpired:
            log(f"{name} TIMEOUT")
    if os.path.isfile(final):
        log(f"DONE {name}"); return final
    log(f"{name} FAILED"); return None


def h2h(ckpt_a, ckpt_b, games, temp):
    try:
        out = subprocess.run([PY, 'eval/v3/agg_h2h.py', ckpt_a, ckpt_b, str(games), str(temp)],
                             capture_output=True, text=True, timeout=3600).stdout
    except subprocess.TimeoutExpired:
        return None
    m = re.search(r'A score: ([\d.]+)', out)
    return float(m.group(1)) if m else None


def sf(run):
    name, d = run['name'], f"model/v3/distill/{run['name']}"
    out = f"eval/v3/distill_{name}_sf.csv"
    if not os.path.isfile(out):
        try:
            subprocess.run([PY, 'eval/v2/eval_v2.py', str(EPOCHS - 1), '--save-dir', d,
                            '--save-name', name, '--eval-csv', out, '--tiers',
                            'sf_easy,sf_med,sf_hard', '--sf-games', '120', '--skip-h2h',
                            '--skip-random', '--no-gate'], timeout=3600)
        except subprocess.TimeoutExpired:
            log(f"SF {name} TIMEOUT")
    res = {}
    try:
        for r in csv.DictReader(open(out)):
            res[r['opponent']] = float(r['won_pct'])
    except OSError:
        pass
    return res


def main():
    argparse.ArgumentParser().parse_args()
    os.makedirs('model/v3/distill', exist_ok=True)
    os.makedirs(LOGDIR, exist_ok=True)
    log("=== DISTILL CAMPAIGN START (teacher=v3-37M, student=v3.1 116k) ===")
    log("baseline v3.1-eq SF: easy 70.0/med 45.8/hard 22.5 | best so far D-a10-t2: h2h 0.527 SF 76.7/52.5/22.5")
    for run in RUNS:
        ck = train(run)
        if not ck:
            continue
        s03, s05 = h2h(ck, EQ, 400, 0.3), h2h(ck, EQ, 300, 0.5)
        r = sf(run)
        log(f"RESULT {run['name']}: h2h_vs_eq t0.3={s03} t0.5={s05} | "
            f"SF easy {r.get('sf_easy')} med {r.get('sf_med')} hard {r.get('sf_hard')}")
    log("=== DISTILL CAMPAIGN DONE ===")


if __name__ == '__main__':
    main()

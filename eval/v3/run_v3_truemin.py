# -*- coding: utf-8 -*-
"""True-minimum scale pass on the CONFIRMED v3.1 architecture.

The v3.2 ablations proved v3.1 is tight: geometry bias + absolute pos_emb + 4 heads
+ DISTINCT blocks are all load-bearing, and weight-sharing is fatal. So the size
floor must come from scaling DOWN the only free knobs — d_model and n_blocks —
never from removing structure or sharing weights. Goal: the smallest v3.1-shaped
net that still beats sf_easy >= 50%.

Reference: v3.1-eq d32/h4/b8/ffn4/vh64 (no conv) = 116,266p, sf_easy 77%.
Ladder: probe the WIDTH cliff (d24/d20/d16/d12 at b8 — width is the binding
constraint at the bottom) then DEPTH at the cliff width (d16 @ b8/b5/b3). 4 heads
kept throughout (R8: heads matter more than head_dim). Run from repo root; refine
around the 50% crossing after this coarse pass.
"""
import argparse, csv, os, re, subprocess, time

PY = os.path.abspath('.venv/Scripts/python.exe')
LOGDIR = 'eval/v3/logs'
LOG = 'logs/v3_truemin.log'
PACKED = 'data/v2/agg_100M_packed'
EPOCH_SIZE = 8_000_000
SAVE_ROOT = 'model/v3/truemin'
REF = 'v3.1-eq d32/h4/b8 = 116,266p, sf_easy 77.0 / sf_med 43.3 / top1 0.466'


def cfg(d, b, h=4):
    return ['--d-model', str(d), '--n-heads', str(h), '--n-blocks', str(b),
            '--value-hidden', '64', '--stem-kernel', '1', '--stem-blocks', '0',
            '--ffn-mult', '4']


RUNS = [
    dict(name='T1-d24b8', args=cfg(24, 8), epochs=16),
    dict(name='T2-d20b8', args=cfg(20, 8), epochs=16),
    dict(name='T3-d16b8', args=cfg(16, 8), epochs=16),
    dict(name='T4-d16b5', args=cfg(16, 5), epochs=16),
    dict(name='T5-d16b3', args=cfg(16, 3), epochs=16),
    dict(name='T6-d12b8', args=cfg(12, 8), epochs=16),
]


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def final_top1(name):
    last = ''
    try:
        for ln in open(f"{LOGDIR}/{name}.log", errors='replace'):
            for p in ln.replace('\r', '\n').split('\n'):
                if 'EVAL e' in p and 'top1' in p:
                    last = p
    except OSError:
        return None
    m = re.search(r'top1 ([\d.]+)', last)
    return float(m.group(1)) if m else None


def params_of(name):
    try:
        for ln in open(f"{LOGDIR}/{name}.log", errors='replace'):
            if 'params=' in ln:
                m = re.search(r'params=([\d,]+)', ln)
                if m:
                    return m.group(1)
    except OSError:
        pass
    return '?'


def train(run):
    name, d = run['name'], f"{SAVE_ROOT}/{run['name']}"
    final = os.path.join(d, f"{name}_e{run['epochs']-1:04d}.pt")
    rlog = f"{LOGDIR}/{name}.log"
    if os.path.isfile(final):
        log(f"SKIP train {name}"); return True
    cmd = ([PY, '-u', '-m', 'src.v3.train_agg_fast', '--save-dir', d, '--save-name', name,
            '--packed-dir', PACKED, '--tau', '0.5', '--value-loss-weight', '1.0',
            '--epoch-size', str(EPOCH_SIZE), '--epochs', str(run['epochs']),
            '--lr-horizon', str(run['epochs']), '--lr', '1e-3', '--warmup-steps', '300',
            '--batch-size', '1024', '--seed', '0', '--save-every-steps', '2000',
            '--metrics-csv', f'eval/v3/{name}_metrics.csv'] + run['args'])
    log(f"TRAIN {name}: {' '.join(run['args'])}")
    for attempt in range(1, 4):
        with open(rlog, 'a') as lf:
            lf.write(f"\n===== attempt {attempt} {time.strftime('%H:%M:%S')} =====\n")
            try:
                subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd='.', timeout=3 * 3600)
            except subprocess.TimeoutExpired:
                log(f"{name} TIMEOUT")
        if os.path.isfile(final):
            log(f"DONE {name} ({params_of(name)}p, top1={final_top1(name)})"); return True
        try:
            if 'nan' in open(rlog, errors='replace').read()[-4000:].lower():
                log(f"{name}: nan -> quarantine"); return False
        except OSError:
            pass
    log(f"{name} FAILED"); return False


def sf_eval(run):
    name, d, ep = run['name'], f"{SAVE_ROOT}/{run['name']}", run['epochs'] - 1
    out = f"eval/v3/truemin_{name}_sf.csv"
    if not os.path.isfile(out):
        cmd = [PY, 'eval/v2/eval_v2.py', str(ep), '--save-dir', d, '--save-name', name,
               '--eval-csv', out, '--tiers', 'sf_easy,sf_med', '--sf-games', '120',
               '--skip-h2h', '--skip-random']
        log(f"SF-EVAL {name}")
        with open(f"{LOGDIR}/{name}_sfeval.log", 'a') as lf:
            try:
                subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd='.', timeout=3600)
            except subprocess.TimeoutExpired:
                log(f"SF eval {name} TIMEOUT")
    res = {}
    try:
        for row in csv.DictReader(open(out)):
            res[row['opponent']] = (float(row['won_pct']), float(row['lost_pct']))
    except OSError:
        pass
    return res


def main():
    argparse.ArgumentParser().parse_args()
    os.makedirs(SAVE_ROOT, exist_ok=True); os.makedirs(LOGDIR, exist_ok=True)
    log("=== V3 TRUE-MIN scale pass START ===")
    log(f"ref = {REF}")
    for run in RUNS:
        if not train(run):
            log(f"{run['name']} failed; skip eval"); continue
        res = sf_eval(run)
        e, m = res.get('sf_easy', (None, None)), res.get('sf_med', (None, None))
        clears = '' if e[0] is None else ('  <<< CLEARS 50%' if e[0] >= 50 else '  --- below 50%')
        log(f"RESULT {run['name']} ({params_of(run['name'])}p): sf_easy won {e[0]}% lost {e[1]}% "
            f"| sf_med won {m[0]}% | top1 {final_top1(run['name'])}{clears}")
    log("=== V3 TRUE-MIN scale pass DONE ===")


if __name__ == '__main__':
    main()

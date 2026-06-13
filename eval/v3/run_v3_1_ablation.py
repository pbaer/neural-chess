# -*- coding: utf-8 -*-
"""v3.1 conv-stem ablation: is the conv stem doing anything attention can't?

v3.1 = pure square-token transformer: the conv stem (3x3 input conv + 2 residual
conv blocks) is replaced by a 1x1 per-square embed (stem_kernel=1, stem_blocks=0)
-> ZERO spatial convolution; attention + the geometry bias must do ALL spatial work.

Two controlled runs vs the v3-nano-tau baseline (conv, d32x8, 158,762 params, sf_easy 58% won):
  - v3.1-eq  (d32x8,  no conv -> 116,266p): equal-config -> isolates conv's contribution
             (does strength survive, and do we get a free shrink below 159k?)
  - v3.1-pm  (d32x11, no conv -> 157,078p): param-matched (conv's freed budget reinvested as
             3 more attention blocks) -> the decisive "is conv the best use of these params?" test
Identical recipe to the nano search (16ep, 8M, tau0.5, vlw1.0, lr1e-3, seed0).
Run from repo root.
"""
import argparse, csv, os, subprocess, time

PY = os.path.abspath('.venv/Scripts/python.exe')
LOGDIR = 'eval/v3/logs'
ABL_LOG = 'logs/v3_1_ablation.log'
PACKED = 'data/v2/agg_100M_packed'
EPOCH_SIZE = 8_000_000
BASELINE = 'v3-nano-tau (conv d32x8, 158,762p): sf_easy 58.3% won / 16.7% lost, top1 0.4535'

RUNS = [
    dict(name='v3.1-eq', d=32, blocks=8,  heads=4, vh=64, stem_kernel=1, stem_blocks=0, params=116266, epochs=16),
    dict(name='v3.1-pm', d=32, blocks=11, heads=4, vh=64, stem_kernel=1, stem_blocks=0, params=157078, epochs=16),
]


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(ABL_LOG, 'a') as f:
        f.write(line + '\n')


def final_top1(name):
    import re
    last = ''
    try:
        for ln in open(f"{LOGDIR}/{name}.log", errors='replace'):
            for piece in ln.replace('\r', '\n').split('\n'):
                if 'EVAL e' in piece and 'top1' in piece:
                    last = piece
    except OSError:
        return None
    m = re.search(r'top1 ([\d.]+)', last)
    return float(m.group(1)) if m else None


def train(run):
    name, d = run['name'], f"model/v3/v3.1/{run['name']}"
    final = os.path.join(d, f"{name}_e{run['epochs']-1:04d}.pt")
    rlog = f"{LOGDIR}/{name}.log"
    if os.path.isfile(final):
        log(f"SKIP train {name} (final ckpt exists)"); return True
    cmd = [PY, '-u', '-m', 'src.v3.train_agg_fast',
           '--save-dir', d, '--save-name', name, '--packed-dir', PACKED,
           '--d-model', str(run['d']), '--n-heads', str(run['heads']),
           '--n-blocks', str(run['blocks']), '--value-hidden', str(run['vh']),
           '--stem-kernel', str(run['stem_kernel']), '--stem-blocks', str(run['stem_blocks']),
           '--tau', '0.5', '--value-loss-weight', '1.0',
           '--epoch-size', str(EPOCH_SIZE), '--epochs', str(run['epochs']),
           '--lr-horizon', str(run['epochs']), '--lr', '1e-3', '--warmup-steps', '300',
           '--batch-size', '1024', '--seed', '0', '--save-every-steps', '2000',
           '--metrics-csv', f'eval/v3/{name}_metrics.csv']
    log(f"TRAIN {name} (d{run['d']} b{run['blocks']} k{run['stem_kernel']} stem{run['stem_blocks']} = {run['params']:,}p, {run['epochs']}ep)")
    for attempt in range(1, 4):
        with open(rlog, 'a') as lf:
            lf.write(f"\n===== attempt {attempt} {time.strftime('%H:%M:%S')} =====\n")
            try:
                subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd='.', timeout=3 * 3600)
            except subprocess.TimeoutExpired:
                log(f"{name} attempt {attempt} TIMEOUT")
        if os.path.isfile(final):
            log(f"DONE train {name} (attempt {attempt}); top1={final_top1(name)}"); return True
        try:
            if 'nan' in open(rlog, errors='replace').read()[-4000:].lower():
                log(f"{name}: nan -> quarantine"); return False
        except OSError:
            pass
    log(f"{name} FAILED"); return False


def sf_eval(run):
    name, d, ep = run['name'], f"model/v3/v3.1/{run['name']}", run['epochs'] - 1
    out = f"eval/v3/v3_1_{name}_sf.csv"
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
    ap = argparse.ArgumentParser(); ap.parse_args()
    os.makedirs('model/v3/v3.1', exist_ok=True); os.makedirs(LOGDIR, exist_ok=True)
    log("=== V3.1 CONV-STEM ABLATION START ===")
    log(f"baseline = {BASELINE}")
    for run in RUNS:
        if not train(run):
            log(f"{run['name']} failed; skipping eval"); continue
        res = sf_eval(run)
        easy, med = res.get('sf_easy', (None, None)), res.get('sf_med', (None, None))
        log(f"RESULT {run['name']} ({run['params']:,}p, NO conv stem): sf_easy won {easy[0]}% lost {easy[1]}% "
            f"| sf_med won {med[0]}% | top1 {final_top1(run['name'])}")
    log("=== V3.1 ABLATION DONE ===  (compare to baseline above; v3.1-pm vs v3-nano-tau is the equal-param verdict)")


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""v3.2 architecture exploration — BATCH 1 (free-flag, no model.py changes).

Hunts for more dead weight / cheap strength-per-param tweaks in the v3.1 pure-
transformer core (v3.1-eq = d32/h4/b8/vh64, no conv stem; sf_easy 77% / med 43%).
The blocks are 94% of params, so the decisive bets are equal-param reallocation
(FFN<->depth) and the principle-critical geometry-bias test. Front-loaded so an
early cut still yields the FFN + principle verdicts.

  R1 FFN-SLIM   ffn_mult 4->2 (83k, -29%)      : is FFNx4 needed? cheap shrink test
  R2 FFN->DEPTH ffn_mult 2 + n_blocks 11 (~111k): equal-param FFN-vs-depth (decisive)
  R3 GEO-OFF    geometry_bias off (109k)        : does the (Δr,Δf) prior earn its keep? (Principle-2)
  R8 H2         n_heads 4->2 (~113k)            : wider heads, cheaper geo bias
  R7 WIDE       d48 / n_blocks 4 (~128k)        : width<->depth triangulation

Code-requiring runs (R4 pos-emb toggle, R5 LEAN cleanup combo, R6 weight-sharing)
are BATCH 2 after the small model.py additions. Identical recipe to v3.1-eq.
Run from repo root.
"""
import argparse, csv, os, subprocess, time

PY = os.path.abspath('.venv/Scripts/python.exe')
LOGDIR = 'eval/v3/logs'
LOG = 'logs/v3_2_explore.log'
PACKED = 'data/v2/agg_100M_packed'
EPOCH_SIZE = 8_000_000
BASELINE = 'v3.1-eq (d32/h4/b8/vh64, NO conv): sf_easy 77.0% / med 43.3%, top1 0.466, 116,266p'

BASE = ['--d-model', '32', '--n-heads', '4', '--n-blocks', '8', '--value-hidden', '64',
        '--stem-kernel', '1', '--stem-blocks', '0', '--ffn-mult', '4']
def variant(**kw):
    a = list(BASE)
    for k, v in kw.items():
        flag = '--' + k.replace('_', '-')
        if v is True:
            a.append(flag)
        else:
            # replace existing value for this flag, else append
            if flag in a:
                a[a.index(flag) + 1] = str(v)
            else:
                a += [flag, str(v)]
    return a

RUNS = [
    dict(name='R1-ffn2',     args=variant(ffn_mult=2),                       epochs=16),
    dict(name='R2-ffn2-b11', args=variant(ffn_mult=2, n_blocks=11),          epochs=16),
    dict(name='R3-geo-off',  args=variant(no_geometry_bias=True),            epochs=16),
    dict(name='R8-h2',       args=variant(n_heads=2),                        epochs=16),
    dict(name='R7-d48-b4',   args=variant(d_model=48, n_blocks=4),           epochs=16),
]


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def final_top1(name):
    import re
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
                import re
                m = re.search(r'params=([\d,]+)', ln)
                if m:
                    return m.group(1)
    except OSError:
        pass
    return '?'


def train(run):
    name, d = run['name'], f"model/v3/v3.2/{run['name']}"
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
    name, d, ep = run['name'], f"model/v3/v3.2/{run['name']}", run['epochs'] - 1
    out = f"eval/v3/v3_2_{name}_sf.csv"
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
    os.makedirs('model/v3/v3.2', exist_ok=True); os.makedirs(LOGDIR, exist_ok=True)
    log("=== V3.2 EXPLORE (batch 1: free-flag) START ===")
    log(f"baseline = {BASELINE}")
    for run in RUNS:
        if not train(run):
            log(f"{run['name']} failed; skip eval"); continue
        res = sf_eval(run)
        e, m = res.get('sf_easy', (None, None)), res.get('sf_med', (None, None))
        log(f"RESULT {run['name']} ({params_of(run['name'])}p): sf_easy won {e[0]}% lost {e[1]}% "
            f"| sf_med won {m[0]}% | top1 {final_top1(run['name'])}  [vs v3.1-eq 77.0/43.3]")
    log("=== V3.2 EXPLORE batch 1 DONE ===")


if __name__ == '__main__':
    main()

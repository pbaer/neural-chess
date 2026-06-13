# -*- coding: utf-8 -*-
"""v3-nano-tau size search: how SMALL can the model get while still beating
Stockfish-easiest >=50% on a single forward pass, keeping ALL key v3 architecture
elements (conv stem, geometry-bias attention, FFN, policy+value heads)?

Only scale knobs shrink: d_model, n_blocks, n_heads, value_hidden. Recipe is what
the micro campaign taught: tau (avg value + soft policy + tau0.5) + LONG training
(length was the dominant lever) + vlw 1.0 (lowering it didn't help). Same seed/corpus.

Trains the frontier candidates (descending size), evals each vs sf_easy (+sf_med
for context), and reports the smallest that clears sf_easy >=50% won.
Run from repo root after the micro campaign frees the GPU.
"""
import argparse
import csv
import os
import subprocess
import time

PY = os.path.abspath('.venv/Scripts/python.exe')
LOGDIR = 'eval/v3/logs'
NANO_LOG = 'logs/nano_search.log'
PACKED = 'data/v2/agg_100M_packed'
SUMMARY = 'eval/v3/nano_summary.csv'

# Frontier candidates (descending params). All: n_heads=2, value_hidden=32,
# stem_blocks=2, ffn_mult=4 (faithful). params precomputed (tmp/nano_params.py).
RUNS = [
    # UPWARD re-bracket: d16x4 (30k) won only 4% vs sf_easy (strength cliff).
    # Crossover to >=50% is between 30k and micro's 695k. Ascending ladder + a
    # narrow-but-deep depth-rescue probe. Smallest that clears 50% = v3-nano-tau.
    dict(name='nano-d16x10', d=16, blocks=10, heads=2, vh=32, params=52574,  epochs=16),
    dict(name='nano-d24x8',  d=24, blocks=8,  heads=2, vh=64, params=91978,  epochs=16),
    dict(name='nano-d32x8',  d=32, blocks=8,  heads=4, vh=64, params=158762, epochs=16),
    dict(name='nano-d48x8',  d=48, blocks=8,  heads=4, vh=64, params=335818, epochs=16),
]
EPOCH_SIZE = 8_000_000
START = time.time()


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(NANO_LOG, 'a') as f:
        f.write(line + '\n')


def wait_campaign_done(timeout_min=120):
    """Don't start until the micro campaign has freed the GPU."""
    cl = 'logs/campaign.log'
    t0 = time.time()
    while time.time() - t0 < timeout_min * 60:
        try:
            if 'CAMPAIGN DONE' in open(cl, errors='replace').read():
                time.sleep(20)  # grace: let it fully exit
                log("micro campaign done -> GPU free, starting nano search."); return
        except OSError:
            pass
        log("waiting for micro campaign to finish (GPU busy) ...")
        time.sleep(60)
    log("WARN: campaign-done wait timed out; proceeding.")


def final_top1(name):
    f = f"{LOGDIR}/{name}.log"
    try:
        import re
        last = ''
        for ln in open(f, errors='replace'):
            for piece in ln.replace('\r', '\n').split('\n'):
                if 'EVAL e' in piece and 'top1' in piece:
                    last = piece
        m = re.search(r'top1 ([\d.]+)', last)
        return float(m.group(1)) if m else None
    except OSError:
        return None


def train(run):
    name = run['name']; d = f"model/v3/nano/{name}"
    final = os.path.join(d, f"{name}_e{run['epochs']-1:04d}.pt")
    rlog = f"{LOGDIR}/{name}.log"
    if os.path.isfile(final):
        log(f"SKIP train {name} (final ckpt exists)"); return True
    cmd = [PY, '-u', '-m', 'src.v3.train_agg_fast',
           '--save-dir', d, '--save-name', name, '--packed-dir', PACKED,
           '--d-model', str(run['d']), '--n-heads', str(run['heads']),
           '--n-blocks', str(run['blocks']), '--value-hidden', str(run['vh']),
           '--tau', '0.5', '--value-loss-weight', '1.0',
           '--epoch-size', str(EPOCH_SIZE), '--epochs', str(run['epochs']),
           '--lr-horizon', str(run['epochs']), '--lr', '1e-3', '--warmup-steps', '300',
           '--batch-size', '1024', '--seed', '0', '--save-every-steps', '2000',
           '--metrics-csv', f'eval/v3/{name}_metrics.csv']
    log(f"TRAIN {name} (d{run['d']} b{run['blocks']} h{run['heads']} = {run['params']:,} params, {run['epochs']}ep)")
    for attempt in range(1, 4):
        with open(rlog, 'a') as lf:
            lf.write(f"\n===== attempt {attempt} {time.strftime('%H:%M:%S')} =====\n")
            try:
                subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd='.', timeout=3 * 3600)
            except subprocess.TimeoutExpired:
                log(f"{name} attempt {attempt} TIMEOUT")
        if os.path.isfile(final):
            log(f"DONE train {name} (attempt {attempt}); final top1={final_top1(name)}"); return True
        try:
            if 'nan' in open(rlog, errors='replace').read()[-4000:].lower():
                log(f"{name}: nan -> quarantine"); return False
        except OSError:
            pass
    log(f"{name} FAILED"); return False


def sf_eval(run):
    name = run['name']; d = f"model/v3/nano/{name}"; ep = run['epochs'] - 1
    out = f"eval/v3/nano_{name}_sf.csv"
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
    # parse won% per tier
    res = {}
    try:
        for row in csv.DictReader(open(out)):
            res[row['opponent']] = (float(row['won_pct']), float(row['lost_pct']))
    except OSError:
        pass
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-wait', action='store_true')
    args = ap.parse_args()
    os.makedirs(f"model/v3/nano", exist_ok=True)
    os.makedirs(LOGDIR, exist_ok=True)
    log("=== NANO SEARCH START ===")
    if not args.skip_wait:
        wait_campaign_done()

    rows = []
    for run in RUNS:
        ok = train(run)
        if not ok:
            log(f"{run['name']} training failed; skipping eval"); continue
        res = sf_eval(run)
        easy = res.get('sf_easy', (None, None))
        med = res.get('sf_med', (None, None))
        t1 = final_top1(run['name'])
        clears = easy[0] is not None and easy[0] >= 50.0
        log(f"RESULT {run['name']} ({run['params']:,}p): sf_easy won {easy[0]}% lost {easy[1]}% "
            f"| sf_med won {med[0]}% | top1 {t1} | clears50={clears}")
        rows.append(dict(name=run['name'], params=run['params'], d=run['d'], blocks=run['blocks'],
                         top1=t1, sf_easy_won=easy[0], sf_easy_lost=easy[1], sf_med_won=med[0],
                         clears_50=clears))

    if rows:
        with open(SUMMARY, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        passing = [r for r in rows if r['clears_50']]
        if passing:
            best = min(passing, key=lambda r: r['params'])
            log(f"=== SMALLEST CLEARING sf_easy>=50%: {best['name']} ({best['params']:,} params, "
                f"sf_easy {best['sf_easy_won']}% won) ===")
        else:
            log("=== NO candidate cleared sf_easy>=50% at the tested sizes (need bigger) ===")
    log("=== NANO SEARCH DONE ===")


if __name__ == '__main__':
    main()

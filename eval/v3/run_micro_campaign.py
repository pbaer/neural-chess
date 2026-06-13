# -*- coding: utf-8 -*-
"""Unattended overnight campaign: train v3-micro-tau training-TWEAK variants
(no architecture changes), then comprehensively eval (SF ladder + round-robin
head-to-head). Idempotent, sequential (single GPU), time-gated, crash-tolerant.

Design (from the campaign-design workflow):
- Levers ranked: training LENGTH (top) > value_loss_weight DOWN (sharper policy
  -> stronger single-pass play) > stacked combo > peak-LR / LR-schedule shape.
- OFAT vs the running baseline M0 (e12, vlw1.0, lr1e-3, cosine), highest-confidence
  runs first so a time-cut still leaves the best models trained.
- All runs share seed 0 + the same packed corpus => OFAT deltas are signal.
- Eval is single-pass greedy (our optimization target): SF ladder via eval_v2.py;
  round-robin h2h via agg_h2h.py at temp 0.3, anchored by the v3-18M-tau yardstick.

This script only EXECUTES + collects raw CSV/logs; final synthesis is done by hand.
Run from repo root:  .venv/Scripts/python.exe eval/v3/run_micro_campaign.py --train-deadline-min N
"""
import argparse
import os
import subprocess
import sys
import time

PY = os.path.abspath('.venv/Scripts/python.exe')
LOGDIR = 'eval/v3/logs'
CAMP_LOG = 'logs/campaign.log'
PACKED = 'data/v2/agg_100M_packed'
YARD = 'model/v3/v3-18M-tau/v3-18M-tau_e0009.pt'
M0 = dict(name='M0-base', dir='model/v3/v3-micro-tau', save='v3-micro-tau', final_ep=11)

# Training sweep, priority order (highest-confidence first). Each: extra args vs the
# common base (tau0.5, epoch-size10M, bs1024, seed0, save-every-steps2000).
RUNS = [
    dict(name='M1-len20',  epochs=20, args=['--epochs', '20', '--lr-horizon', '20']),
    dict(name='M2-vlw050', epochs=12, args=['--epochs', '12', '--lr-horizon', '12', '--value-loss-weight', '0.5']),
    dict(name='M4-combo',  epochs=16, args=['--epochs', '16', '--lr-horizon', '16', '--value-loss-weight', '0.35', '--weight-decay', '0']),
    dict(name='M3-vlw025', epochs=12, args=['--epochs', '12', '--lr-horizon', '12', '--value-loss-weight', '0.25']),
    dict(name='M5-lr15',   epochs=12, args=['--epochs', '12', '--lr-horizon', '12', '--lr', '1.5e-3', '--warmup-steps', '600']),
    dict(name='M6-trunc',  epochs=20, args=['--epochs', '20', '--lr-horizon', '28']),
]

START = time.time()


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(CAMP_LOG, 'a') as f:
        f.write(line + '\n')


def min_left(deadline_min):
    return deadline_min - (time.time() - START) / 60.0


def ckpt_path(d, save, ep):
    return os.path.join(d, f"{save}_e{ep:04d}.pt")


def wait_for_m0(timeout_min=90):
    final = ckpt_path(M0['dir'], M0['save'], M0['final_ep'])
    if os.path.isfile(final):
        log("M0 baseline already complete."); return True
    log(f"Waiting for M0 baseline to finish ({final}) ...")
    t0 = time.time()
    while time.time() - t0 < timeout_min * 60:
        if os.path.isfile(final):
            time.sleep(25)  # grace: let M0's process exit + free its 26GB RAM / GPU first
            log("M0 baseline complete -> starting sweep."); return True
        time.sleep(30)
    log("WARN: M0 wait timed out; proceeding anyway."); return os.path.isfile(final)


def train_one(run, deadline_min):
    name, epochs = run['name'], run['epochs']
    d = f"model/v3/micro/{name}"
    final = ckpt_path(d, name, epochs - 1)
    rlog = f"{LOGDIR}/{name}.log"
    if os.path.isfile(final):
        log(f"SKIP {name} (final ckpt exists)"); return True
    left = min_left(deadline_min)
    if left < 75:
        log(f"TIME-GATE: {left:.0f} min left < 75 -> skip {name} (+rest of training)"); return None
    cmd = [PY, '-u', '-m', 'src.v3.train_agg_fast',
           '--save-dir', d, '--save-name', name, '--packed-dir', PACKED,
           '--tau', '0.5', '--epoch-size', '10000000', '--batch-size', '1024',
           '--seed', '0', '--save-every-steps', '2000',
           '--metrics-csv', f'eval/v3/{name}_metrics.csv'] + run['args']
    log(f"TRAIN {name}: {' '.join(run['args'])}  [{left:.0f} min to deadline]")
    for attempt in range(1, 4):
        with open(rlog, 'a') as lf:
            lf.write(f"\n===== attempt {attempt} {time.strftime('%H:%M:%S')} =====\n")
            try:
                subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd='.', timeout=3 * 3600)
            except subprocess.TimeoutExpired:
                log(f"{name} attempt {attempt} TIMEOUT (3h)")
        if os.path.isfile(final):
            log(f"DONE {name} (attempt {attempt})"); return True
        try:
            tail = open(rlog, encoding='utf-8', errors='replace').read()[-4000:].lower()
            if 'nan' in tail or 'traceback' in tail:
                log(f"{name}: nan/traceback detected -> quarantine, stop retrying"); return False
        except OSError:
            pass
        log(f"{name} attempt {attempt} ended w/o final ckpt -> retry (auto-resume)")
    log(f"{name} FAILED after 3 attempts"); return False


def sf_eval(name, d, save, ep):
    out_csv = f"eval/v3/campaign_{name}_sf.csv"
    if os.path.isfile(out_csv):
        log(f"SKIP SF eval {name} (csv exists)"); return
    rlog = f"{LOGDIR}/{name}_sfeval.log"
    cmd = [PY, 'eval/v2/eval_v2.py', str(ep), '--save-dir', d, '--save-name', save,
           '--eval-csv', out_csv, '--tiers', 'sf_easy,sf_med,sf_hard,sf_magnus',
           '--sf-games', '120', '--magnus-games', '40', '--skip-h2h', '--skip-random']
    log(f"SF-EVAL {name} (epoch {ep})")
    with open(rlog, 'a') as lf:
        try:
            subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd='.', timeout=2 * 3600)
        except subprocess.TimeoutExpired:
            log(f"SF eval {name} TIMEOUT")


def h2h(ca, cb, games=100, temp=0.3):
    cmd = [PY, 'eval/v3/agg_h2h.py', ca, cb, str(games), str(temp)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd='.', timeout=3600)
        out = r.stdout
    except subprocess.TimeoutExpired:
        return None
    score, wdl = '', ',,'
    for ln in out.splitlines():
        if ln.startswith('A score:'):
            score = ln.split(':')[1].split('(')[0].strip()
        if ln.startswith('A:'):
            import re
            m = re.search(r'W\s*([\d.]+)%\s*D\s*([\d.]+)%\s*L\s*([\d.]+)%', ln)
            if m:
                wdl = f"{m.group(1)},{m.group(2)},{m.group(3)}"
    return score, wdl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-deadline-min', type=int, default=480,
                    help='minutes from start after which NO new training launches (reserve rest for eval)')
    ap.add_argument('--skip-wait-m0', action='store_true')
    args = ap.parse_args()
    os.makedirs(LOGDIR, exist_ok=True)
    os.makedirs('model/v3/micro', exist_ok=True)
    log(f"=== CAMPAIGN START === train-deadline {args.train_deadline_min} min")

    # Phase 0: wait for the baseline M0 to free the GPU.
    if not args.skip_wait_m0:
        wait_for_m0()

    # Phase 1: training sweep (priority order, time-gated, idempotent).
    for run in RUNS:
        res = train_one(run, args.train_deadline_min)
        if res is None:   # time-gate hit -> stop launching training, go eval
            break

    # Phase 2: discover completed models.
    models = []
    if os.path.isfile(ckpt_path(M0['dir'], M0['save'], M0['final_ep'])):
        models.append((M0['name'], ckpt_path(M0['dir'], M0['save'], M0['final_ep']),
                       M0['dir'], M0['save'], M0['final_ep']))
    for run in RUNS:
        d = f"model/v3/micro/{run['name']}"; ep = run['epochs'] - 1
        fp = ckpt_path(d, run['name'], ep)
        if os.path.isfile(fp):
            models.append((run['name'], fp, d, run['name'], ep))
    log(f"Completed models for eval: {[m[0] for m in models]}")

    # Phase 3: SF ladder per model.
    log("=== SF LADDER EVAL ===")
    for name, fp, d, save, ep in models:
        sf_eval(name, d, save, ep)

    # Phase 4: round-robin head-to-head (+ yardstick), append per pairing.
    log("=== ROUND-ROBIN H2H ===")
    nodes = [(n, fp) for (n, fp, *_rest) in models]
    if os.path.isfile(YARD):
        nodes.append(('v3-18M-tau', YARD))
    rr_csv = 'eval/v3/campaign_rr.csv'
    done = set()
    if os.path.isfile(rr_csv):
        for ln in open(rr_csv):
            p = ln.split(',')
            if len(p) >= 2:
                done.add((p[0], p[1]))
    else:
        with open(rr_csv, 'w') as f:
            f.write("A,B,A_score,A_W,A_D,A_L\n")
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            na, ca = nodes[i]; nb, cb = nodes[j]
            if (na, nb) in done:
                continue
            left = min_left(args.train_deadline_min + 75)  # absolute deadline ~= train-deadline + reserved
            log(f"H2H {na} vs {nb} [{left:.0f} min to hard deadline]")
            res = h2h(ca, cb, games=100, temp=0.3)
            if res is None:
                log(f"H2H {na} vs {nb} timed out/failed"); continue
            score, wdl = res
            with open(rr_csv, 'a') as f:
                f.write(f"{na},{nb},{score},{wdl}\n")
            log(f"  {na} score {score} vs {nb}")
    log("=== CAMPAIGN DONE ===")


if __name__ == '__main__':
    main()

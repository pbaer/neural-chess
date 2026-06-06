# -*- coding: utf-8 -*-
"""Driver for the position-aggregated data-strategy hill-climb.

Runs the fixed small-v3 arch on one aggregated corpus, varying ONLY the three
data knobs (value-mode, policy-mode, tau). Every run shares identical
hyperparameters so the data treatment is the only independent variable.

Stages:
  stage1  isolate the label levers at tau=1 (production distribution):
            base (one/hard_sample), av (avg/hard_sample),
            sp (one/soft),          avsp (avg/soft)
  stage2  tau sweep on the winning label combo (set via --label-combo)

Resumable: skips a treatment whose save_name already has the target epoch ckpt.
Usage:
  python eval/v3/agg_sweep.py stage1 --agg-dir data/v2/agg_8M
  python eval/v3/agg_sweep.py stage2 --agg-dir data/v2/agg_8M --value-mode avg --policy-mode soft
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PY = os.path.join(ROOT, '.venv', 'Scripts', 'python.exe')

# Fixed hyperparameters shared by EVERY run (only data knobs vary).
FIXED = dict(epoch_size=1_500_000, epochs=2, lr_horizon=8, batch_size=1024,
             lr=1e-3, weight_decay=1e-4, value_loss_weight=1.0,
             num_workers=12, seed=0)


def run_one(label, agg_dir, value_mode, policy_mode, tau, metrics_csv, save_root,
            **over):
    fixed = dict(FIXED, **over)
    save_dir = os.path.join(save_root, label)
    target_ckpt = os.path.join(save_dir, f"model_e{fixed['epochs']-1:04d}.pt")
    if os.path.exists(target_ckpt):
        print(f"SKIP {label} (exists {target_ckpt})", flush=True)
        return
    os.makedirs('eval/v3/logs', exist_ok=True)
    log = f"eval/v3/logs/agg_{label}.log"
    cmd = [PY, '-m', 'src.v3.train_agg',
           '--agg-dir', agg_dir, '--save-dir', save_dir, '--save-name', label,
           '--value-mode', value_mode, '--policy-mode', policy_mode,
           '--tau', str(tau), '--metrics-csv', metrics_csv,
           '--epoch-size', str(fixed['epoch_size']), '--epochs', str(fixed['epochs']),
           '--lr-horizon', str(fixed['lr_horizon']), '--batch-size', str(fixed['batch_size']),
           '--lr', str(fixed['lr']), '--weight-decay', str(fixed['weight_decay']),
           '--value-loss-weight', str(fixed['value_loss_weight']),
           '--num-workers', str(fixed['num_workers']), '--seed', str(fixed['seed'])]
    print(f"\n=== RUN {label}: value={value_mode} policy={policy_mode} tau={tau} ===",
          flush=True)
    t0 = time.time()
    with open(log, 'w') as lf:
        r = subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT)
    print(f"  {label} rc={r.returncode} in {time.time()-t0:.0f}s -> {log}", flush=True)
    if r.returncode != 0:
        print(f"  !! {label} FAILED; see {log}", flush=True)
        tail = open(log).read()[-1500:]
        print(tail, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['stage1', 'stage2'])
    ap.add_argument('--agg-dir', required=True)
    ap.add_argument('--metrics-csv', default='eval/v3/agg_metrics.csv')
    ap.add_argument('--save-root', default='model/v3/agg')
    ap.add_argument('--value-mode', default='avg', help='[stage2] winning value mode')
    ap.add_argument('--policy-mode', default='soft', help='[stage2] winning policy mode')
    ap.add_argument('--taus', default='0.0,0.3,0.5,0.7')
    ap.add_argument('--vlw', type=float, default=None,
                    help='[stage2] override value-loss-weight for all tau runs')
    args = ap.parse_args()
    over = {} if args.vlw is None else {'value_loss_weight': args.vlw}

    if args.stage == 'stage1':
        plan = [
            ('base', 'one', 'hard_sample', 1.0),
            ('av',   'avg', 'hard_sample', 1.0),
            ('sp',   'one', 'soft',        1.0),
            ('avsp', 'avg', 'soft',        1.0),
        ]
    else:
        vm, pm = args.value_mode, args.policy_mode
        vtag = 'v%g' % args.vlw if args.vlw is not None else ''
        plan = [(f"tau{t}{vtag}".replace('.', ''), vm, pm, float(t))
                for t in args.taus.split(',')]
    for label, vm, pm, tau in plan:
        run_one(label, args.agg_dir, vm, pm, tau, args.metrics_csv, args.save_root,
                **over)
    print("\nSWEEP STAGE DONE", flush=True)


if __name__ == '__main__':
    main()

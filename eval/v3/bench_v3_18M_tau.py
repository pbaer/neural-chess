# -*- coding: utf-8 -*-
"""Comprehensive, high-signal benchmark for v3-18M-tau:
  - Stockfish ladder easy->ultra with elevated game counts
  - Large-N pure-policy head-to-heads vs v3-37M and v3-18M (two temperatures)
  - MCTS-50 head-to-head vs v3-37M (where the averaged-value win should pay off)
Parses every result, computes Elo + 95% CI for the head-to-heads, prints a
summary table and writes eval/v3/bench_v3_18M_tau_summary.txt.

The central question: does the recipe (signal density) let an 18M model match or
beat the 2x-larger v3-37M (capacity)? And vs v3-18M (equal params) it isolates
the recipe's pure contribution.
"""
import math
import os
import re
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PY = os.path.join(ROOT, '.venv', 'Scripts', 'python.exe')

TAU = 'model/v3/v3-18M-tau/v3-18M-tau_e0009.pt'
V37 = 'model/v3/v3-37M/model_e0009.pt'
V18 = 'model/v3/v3-18M/model_e0008.pt'
SUMMARY = 'eval/v3/bench_v3_18M_tau_summary.txt'

lines = []
def out(s=''):
    print(s, flush=True)
    lines.append(s)


def run(cmd, timeout=7200):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    return r.stdout + '\n' + r.stderr


def elo_ci(w, d, l):
    n = w + d + l
    s = (w + 0.5 * d) / n
    # std error of the score (per-game variance from W/D/L outcomes {1,.5,0})
    m1 = (w * 1.0 + d * 0.25 + l * 0.0) / n          # E[x^2]
    var = max(m1 - s * s, 1e-9) / n
    se = math.sqrt(var)
    def to_elo(x):
        x = min(max(x, 1e-6), 1 - 1e-6)
        return 400.0 * math.log10(x / (1 - x))
    return s, to_elo(s), to_elo(s - 1.96 * se), to_elo(s + 1.96 * se)


def h2h(label, a, b, games, temp, mcts_sims=0):
    out(f"\n===== H2H {label}: {games}g temp{temp}"
        + (f" MCTS-{mcts_sims}" if mcts_sims else " (policy)") + " =====")
    if mcts_sims:
        cmd = [PY, 'eval/v3/agg_h2h_mcts.py', a, b, str(games), str(mcts_sims), '1.5', str(temp)]
    else:
        cmd = [PY, 'eval/v3/agg_h2h.py', a, b, str(games), str(temp)]
    o = run(cmd)
    m = re.search(r'\((\d+)/(\d+)/(\d+)\)', o)
    if not m:
        out("  PARSE FAIL:\n" + o[-800:]); return None
    w, d, l = int(m.group(1)), int(m.group(2)), int(m.group(3))
    s, e, lo, hi = elo_ci(w, d, l)
    out(f"  W/D/L = {w}/{d}/{l}   score {s:.3f}   Elo {e:+.0f}  [{lo:+.0f}, {hi:+.0f}] (95%)")
    return (label, w, d, l, s, e, lo, hi)


def main():
    for p in (TAU, V37, V18):
        if not os.path.exists(os.path.join(ROOT, p)):
            out(f"MISSING checkpoint: {p}");
    t0 = time.time()

    # --- Stockfish ladder (elevated counts) ---
    out("===== STOCKFISH LADDER (v3-18M-tau, e9) =====")
    sfcsv = 'eval/v3/v3-18M-tau_bench_sf.csv'
    if os.path.exists(os.path.join(ROOT, sfcsv)):
        os.remove(os.path.join(ROOT, sfcsv))
    o = run([PY, 'eval/v2/eval_v2.py', '9',
             '--save-dir', 'model/v3/v3-18M-tau', '--save-name', 'v3-18M-tau',
             '--eval-csv', sfcsv, '--tiers', 'sf_easy,sf_med,sf_hard,sf_magnus,sf_ultra',
             '--sf-games', '300', '--magnus-games', '50',
             '--skip-h2h', '--skip-random'], timeout=14400)
    for ln in o.splitlines():
        if re.search(r'W:|EVAL .*opp=|ERROR', ln):
            out("  " + ln.strip())

    # --- pure-policy head-to-heads (large N) ---
    results = []
    results.append(h2h('vs v3-37M (policy, t0.5)', TAU, V37, 800, 0.5))
    results.append(h2h('vs v3-37M (policy, t0.3)', TAU, V37, 400, 0.3))
    results.append(h2h('vs v3-18M (policy, t0.5)', TAU, V18, 800, 0.5))

    # --- MCTS head-to-head (value payoff) ---
    results.append(h2h('vs v3-37M (MCTS-50, t0.5)', TAU, V37, 80, 0.5, mcts_sims=50))

    # --- summary ---
    out("\n" + "=" * 64)
    out("SUMMARY  (v3-18M-tau = A; Elo > 0 means tau is stronger)")
    out("=" * 64)
    for r in results:
        if r:
            out(f"  {r[0]:30} score {r[4]:.3f}  Elo {r[5]:+.0f} [{r[6]:+.0f},{r[7]:+.0f}]")
    out(f"\ntotal bench time {(time.time()-t0)/60:.0f} min")
    with open(os.path.join(ROOT, SUMMARY), 'w') as f:
        f.write('\n'.join(lines))
    out(f"\nwrote {SUMMARY}")
    out("BENCH_DONE")


if __name__ == '__main__':
    main()

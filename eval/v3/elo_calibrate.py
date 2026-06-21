# -*- coding: utf-8 -*-
"""Stockfish-anchored Elo of the browser model (D-a10-t2, GPU) as a function of
play settings: one-shot vs MCTS, the MCTS max-sims slider, AND the Move-variety
slider.

Move selection goes through a faithful Python port of the browser's
selection.ts (value-adaptive, bounded "reasonable set"), so variety here means
exactly what it means in the app. variety=0 is the deterministic top move (the
model's strongest play) and gives the base strength curve; variety>0 loosens
play when winning (bounded), giving a measurable Elo penalty.

We measure: (1) the base curve at variety=0 across one-shot + MCTS sim anchors,
and (2) the variety penalty at a couple of representative configs and variety
levels. estimateElo in the app combines base(curve) - penalty(variety).

Elo is fit by maximum likelihood vs Stockfish 18 at calibrated UCI_Elo rungs
(logistic model, draw = half-point); 95% CI = profile-likelihood interval.
Results write to JSON after every config (deadline/crash-safe). ASCII output.

Run from repo root:  python eval/v3/elo_calibrate.py [budget_seconds]
"""
import collections
import json
import math
import os
import sys
import time

import chess
import chess.engine
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.inference_api import load_policy_engine
from src.mcts import MCTS

CKPT = 'model/v3/distill/D-a10-t2/D-a10-t2_e0015.pt'
SF_PATH = 'bin/stockfish-v18.exe'
SF_MOVE_TIME = 0.10
SF_ELO_MIN, SF_ELO_MAX = 1320, 3190
C_PUCT = 1.5
OUT = 'eval/v3/elo_calibration.json'
MAX_PLIES = 300

# Selection config — must mirror viz/src/core/search/selection.ts DEFAULT.
SEL = dict(topK=5, relFloor=0.12, qMargin=0.10, maxTemp=1.5, steepness=9.2)

# Base strength curve: variety=0 across one-shot + MCTS sim anchors.
BASE_ANCHORS = [
    ('one-shot', None, 30), ('mcts-10', 10, 30), ('mcts-25', 25, 30),
    ('mcts-50', 50, 28), ('mcts-100', 100, 24), ('mcts-150', 150, 22),
    ('mcts-200', 200, 20), ('mcts-300', 300, 18),
]
# Variety penalty: (label, sims, games) measured at variety levels below.
VARIETY_CONFIGS = [('one-shot', None, 24), ('mcts-100', 100, 22), ('mcts-300', 300, 16)]
VARIETY_LEVELS = [0.5, 1.0]

BUDGET_S = float(sys.argv[1]) if len(sys.argv) > 1 else 150 * 60
DEADLINE = time.time() + BUDGET_S


def log(m):
    print(m, flush=True)


# ---------- faithful port of selection.ts ----------
def _value_adaptive_temp(value, variety):
    S = min(max(variety, 0.0), 1.0)
    V = min(max(value, -1.0), 1.0)
    if S <= 0:
        return 0.0
    m = 1.0 / (1.0 + math.exp(-SEL['steepness'] * V))
    return S * SEL['maxTemp'] * m


def _reasonable_set(weights, qs):
    n = len(weights)
    if n == 0:
        return []
    top = int(np.argmax(weights))
    top_w = weights[top]
    have_q = qs is not None
    best_q = max(qs) if have_q else None
    order = sorted(range(n), key=lambda i: -weights[i])
    out = []
    for i in order:
        if len(out) >= SEL['topK']:
            break
        if top_w > 0 and weights[i] < SEL['relFloor'] * top_w:
            continue
        if have_q and qs[i] < best_q - SEL['qMargin']:
            continue
        out.append(i)
    if top not in out:
        out.insert(0, top)
    return out


def _select_index(weights, qs, value, variety, rng):
    s = _reasonable_set(weights, qs)
    if not s:
        return -1
    top = max(s, key=lambda i: weights[i])
    if len(s) == 1:
        return top
    T = _value_adaptive_temp(value, variety)
    if T <= 1e-3:
        return top
    # Normalize by the max weight before exponentiating: relative magnitudes are
    # all that matter for sampling, and this keeps every term in [0,1] so a small
    # T can't overflow (visit counts ** (1/T) otherwise blows past float range).
    mx = max(weights[i] for i in s)
    if mx <= 0:
        return top
    w = [(max(weights[i], 0.0) / mx) ** (1.0 / T) for i in s]
    tot = sum(w)
    if not (tot > 0):
        return top
    r = rng.random() * tot
    for k, idx in enumerate(s):
        r -= w[k]
        if r <= 0:
            return idx
    return s[-1]


class VarietySelector:
    """Browser-faithful move generator at a fixed variety, for one-shot or MCTS."""

    def __init__(self, base, variety, sims=None, rng=None):
        self.base = base
        self.variety = variety
        self.sims = sims
        self.rng = rng or np.random.default_rng(0)

    def generate_move(self, board, stats=None, temperature=0.0):
        if self.sims is None:
            priors, value = self.base.evaluate(board)
            moves = list(priors.keys())
            weights = [priors[m] for m in moves]
            qs = None
        else:
            mcts = MCTS(self.base, c_puct=C_PUCT, n_simulations=self.sims, dirichlet_frac=0.0, seed=0)
            _, _, info = mcts.run(board)
            moves = list(info['visits'].keys())
            if not moves:
                legal = list(board.legal_moves)
                return legal[0] if legal else None
            weights = [float(info['visits'][m]) for m in moves]
            qs = [float(info['q'][m]) for m in moves]
            value = float(info['root_value'])
        if self.sims is None:
            value = float(value)
        idx = _select_index(weights, qs, value, self.variety, self.rng)
        return moves[idx] if idx >= 0 else moves[0]


# ---------- Elo machinery ----------
def expected(opp, r):
    return 1.0 / (1.0 + 10.0 ** ((opp - r) / 400.0))


def mle_elo(rungs):
    grid = np.arange(800, 3400, 2.0)
    ll = np.zeros_like(grid)
    for i, r in enumerate(grid):
        s = 0.0
        for opp, w, d, l in rungs:
            e = min(max(expected(opp, r), 1e-9), 1 - 1e-9)
            s += w * math.log(e) + l * math.log(1 - e) + d * (math.log(e) + math.log(1 - e)) * 0.5
        ll[i] = s
    k = int(np.argmax(ll))
    within = grid[ll >= ll[k] - 1.92]
    return float(grid[k]), float(within.min()), float(within.max())


def implied(opp, w, d, l):
    n = w + d + l
    if n == 0:
        return opp
    s = min(max((w + 0.5 * d) / n, 0.5 / n), 1 - 0.5 / n)
    return opp + 400.0 * math.log10(s / (1 - s))


def clamp_rung(e):
    return int(round(min(max(e, SF_ELO_MIN), SF_ELO_MAX)))


def play_game(selector, sf, model_white):
    board = chess.Board()
    mc = chess.WHITE if model_white else chess.BLACK
    while not board.is_game_over() and board.ply() < MAX_PLIES:
        if board.turn == mc:
            mv = selector.generate_move(board)
        else:
            mv = sf.play(board, chess.engine.Limit(time=SF_MOVE_TIME)).move
        if mv is None:
            break
        board.push(mv)
    res = board.result() if board.is_game_over() else '1/2-1/2'
    if res == '1/2-1/2':
        return 0.5
    return 1.0 if ((res == '1-0') == (mc == chess.WHITE)) else 0.0


def play_rung(selector, sf, opp, n):
    sf.configure({'UCI_LimitStrength': True, 'UCI_Elo': int(round(opp))})
    w = d = l = 0
    for g in range(n):
        if time.time() > DEADLINE:
            break
        s = play_game(selector, sf, model_white=(g % 2 == 0))
        if s == 1.0:
            w += 1; ch = 'W'
        elif s == 0.0:
            l += 1; ch = 'L'
        else:
            d += 1; ch = '-'
        sys.stdout.write(ch); sys.stdout.flush()
    print()
    return w, d, l


def calibrate(selector, sf, label, target_games, guess):
    half = max(6, target_games // 2)
    r1 = clamp_rung(guess)
    log(f"[{label}] rung1 SF_Elo={r1} (x{half}):")
    w1, d1, l1 = play_rung(selector, sf, r1, half)
    imp1 = implied(r1, w1, d1, l1)
    if abs(imp1 - r1) < 80:
        r2 = clamp_rung(r1 + (120 if (w1 + 0.5 * d1) >= (w1 + d1 + l1) / 2 else -120))
    else:
        r2 = clamp_rung(imp1)
    if r2 == r1:
        r2 = clamp_rung(r1 + 120)
    rem = max(6, target_games - half)
    log(f"[{label}] rung2 SF_Elo={r2} (x{rem}) [implied ~{imp1:.0f}]:")
    w2, d2, l2 = play_rung(selector, sf, r2, rem)
    rungs = [(r1, w1, d1, l1), (r2, w2, d2, l2)]
    elo, lo, hi = mle_elo(rungs)
    res = dict(label=label, elo=round(elo), lo=round(lo), hi=round(hi),
               rungs=[dict(opp=o, w=w, d=dd, l=l,
                           score=round((w + 0.5 * dd) / max(w + dd + l, 1), 3))
                      for o, w, dd, l in rungs])
    log(f"[{label}] => Elo {res['elo']} (95% CI {res['lo']}-{res['hi']})\n")
    return res, elo


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"device={dev} budget={BUDGET_S/60:.0f}min ckpt={CKPT}")
    base = load_policy_engine(CKPT, device=dev)
    sf = chess.engine.SimpleEngine.popen_uci(os.path.abspath(SF_PATH))
    log(f"opponent: {sf.id.get('name')} (UCI_Elo {SF_ELO_MIN}-{SF_ELO_MAX}, {SF_MOVE_TIME}s/mv)\n")
    rng = np.random.default_rng(12345)
    out = {'meta': {'opponent': sf.id.get('name'), 'sf_move_time': SF_MOVE_TIME,
                    'ckpt': CKPT, 'selection': SEL}, 'base': [], 'variety': []}
    base_elo = {}
    done_base, done_var = set(), set()
    # Resume: reuse anything already measured in a prior (crashed) run.
    if os.path.exists(OUT):
        try:
            prev = json.load(open(OUT))
            out['base'] = prev.get('base', [])
            out['variety'] = prev.get('variety', [])
            for r in out['base']:
                base_elo[r['label']] = r['elo']; done_base.add(r['label'])
            for r in out['variety']:
                done_var.add((r['label'], r['variety']))
            log(f"resume: {len(done_base)} base + {len(done_var)} variety configs already done\n")
        except (OSError, ValueError):
            pass

    log("=== BASE CURVE (variety=0) ===")
    guess = 1500.0
    for label, sims, n in BASE_ANCHORS:
        if label in done_base:
            guess = max(base_elo[label], guess); continue
        if time.time() > DEADLINE:
            log(f"[{label}] SKIPPED (deadline)"); continue
        sel = VarietySelector(base, variety=0.0, sims=sims, rng=rng)
        res, elo = calibrate(sel, sf, label, n, guess)
        res['sims'] = sims; res['variety'] = 0.0
        out['base'].append(res); base_elo[label] = elo
        guess = max(elo, guess)
        json.dump(out, open(OUT, 'w'), indent=2)

    log("=== VARIETY PENALTY ===")
    for label, sims, n in VARIETY_CONFIGS:
        if label not in base_elo:
            continue
        for S in VARIETY_LEVELS:
            if (f"{label}-v{S}", S) in done_var or (label, S) in done_var:
                continue
            if time.time() > DEADLINE:
                log(f"[{label} v{S}] SKIPPED (deadline)"); continue
            sel = VarietySelector(base, variety=S, sims=sims, rng=rng)
            res, elo = calibrate(sel, sf, f"{label}-v{S}", n, base_elo[label] - 60)
            res['sims'] = sims; res['variety'] = S
            res['penalty'] = round(base_elo[label] - elo)
            out['variety'].append(res)
            log(f"[{label} v{S}] penalty vs variety0: {res['penalty']} Elo")
            json.dump(out, open(OUT, 'w'), indent=2)

    sf.quit()
    log("\n========= ELO SUMMARY (Stockfish 18 anchored) =========")
    log("BASE CURVE (variety=0):")
    for r in out['base']:
        log(f"  {r['label']:10s} Elo {r['elo']:5d}  CI {r['lo']}-{r['hi']}")
    log("VARIETY PENALTY (Elo below variety=0):")
    for r in out['variety']:
        log(f"  {r['label']:14s} Elo {r['elo']:5d}  penalty {r['penalty']:+d}")
    log(f"\nwrote {OUT}\nelapsed {(time.time()-(DEADLINE-BUDGET_S))/60:.1f} min")


if __name__ == '__main__':
    main()

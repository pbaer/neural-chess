# -*- coding: utf-8 -*-
"""MCTS head-to-head between two checkpoints: both wrapped in MCTSEngine with the
same sims / c_puct, so the comparison isolates model quality under search. This
is where a better-calibrated VALUE head (the averaged-value recipe's main win)
should pay off, since PUCT consumes the value. Principle-clean (MCTS uses only
the model's P/V — the documented carve-out).

Usage: python eval/v3/agg_h2h_mcts.py <ckptA> <ckptB> [games] [sims] [c_puct] [temp]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import chess
from src.inference_api import load_policy_engine
from src.game_loop import play_models
from src.mcts import MCTSEngine

A = sys.argv[1]
B = sys.argv[2]
games = int(sys.argv[3]) if len(sys.argv) > 3 else 60
sims = int(sys.argv[4]) if len(sys.argv) > 4 else 50
c_puct = float(sys.argv[5]) if len(sys.argv) > 5 else 1.5
temp = float(sys.argv[6]) if len(sys.argv) > 6 else 0.5

print(f"A={A}\nB={B}\ngames={games} sims={sims} c_puct={c_puct} temp={temp}", flush=True)
ea = MCTSEngine(load_policy_engine(A), c_puct=c_puct, n_simulations=sims)
eb = MCTSEngine(load_policy_engine(B), c_puct=c_puct, n_simulations=sims)
half = games // 2
t0 = time.time()
w = play_models(ea, eb, limit=half, a_color=chess.WHITE, temperature=temp, temp_decay=0.05)
b = play_models(ea, eb, limit=games - half, a_color=chess.BLACK, temperature=temp, temp_decay=0.05)
el = time.time() - t0

aw = w['results']['1-0'] + b['results']['0-1']
ad = w['results']['1/2-1/2'] + b['results']['1/2-1/2']
al = w['results']['0-1'] + b['results']['1-0']
n = aw + ad + al
score = (aw + 0.5 * ad) / n
print(f"\n=== MCTS-{sims} {n} games  {el:.0f}s ===")
print(f"A:  W {100*aw/n:.1f}%  D {100*ad/n:.1f}%  L {100*al/n:.1f}%   ({aw}/{ad}/{al})")
print(f"A score: {score:.3f}   (>0.5 = A stronger)")

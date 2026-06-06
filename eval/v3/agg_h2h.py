# -*- coding: utf-8 -*-
"""Generic head-to-head between two checkpoints (any arch). Reports W/D/L and
score from model A's perspective. Used to confirm whether a data-strategy's
held-out edge translates into actual playing strength.

Usage: python eval/v3/agg_h2h.py <ckptA> <ckptB> [games] [temp]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import chess
from src.inference_api import load_policy_engine
from src.game_loop import play_models

A = sys.argv[1]
B = sys.argv[2]
games = int(sys.argv[3]) if len(sys.argv) > 3 else 100
temp = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
decay = 0.05

print(f"A={A}\nB={B}\ngames={games} temp={temp} decay={decay}", flush=True)
ea = load_policy_engine(A)
eb = load_policy_engine(B)
half = games // 2
t0 = time.time()
w = play_models(ea, eb, limit=half, a_color=chess.WHITE, temperature=temp, temp_decay=decay)
b = play_models(ea, eb, limit=games - half, a_color=chess.BLACK, temperature=temp, temp_decay=decay)
el = time.time() - t0

aw = w['results']['1-0'] + b['results']['0-1']
ad = w['results']['1/2-1/2'] + b['results']['1/2-1/2']
al = w['results']['0-1'] + b['results']['1-0']
n = aw + ad + al
score = (aw + 0.5 * ad) / n
print(f"\n=== {n} games  {el:.0f}s ===")
print(f"A:  W {100*aw/n:.1f}%  D {100*ad/n:.1f}%  L {100*al/n:.1f}%   ({aw}/{ad}/{al})")
print(f"A score: {score:.3f}   (>0.5 = A stronger)")

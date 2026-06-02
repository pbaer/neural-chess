# -*- coding: utf-8 -*-
"""Matched-scale head-to-head: v3-18M (attention) vs v2-19M (CNN).

Both ~18-19M params. Plays N games, 50/50 colors, with a mild temperature so
the games vary (model-vs-model at temp 0 would replay one identical game).
Reports W/D/L from v3's perspective.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import chess
from src.inference_api import load_policy_engine
from src.game_loop import play_models

V3 = 'model/v3/v3-37M/model_e0009.pt'   # v3 attention 37M, e9 (final)
V2 = 'model/v2/v2-37M/model_e0007.pt'   # v2 CNN 37M, e7 (final/best)

games = int(sys.argv[1]) if len(sys.argv) > 1 else 100
temp = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
decay = 0.05

print(f"Loading v3-37M(e9) and v2-37M(e7)... (games={games}, temp={temp}, decay={decay})", flush=True)
e3 = load_policy_engine(V3)
e2 = load_policy_engine(V2)

half = games // 2
t0 = time.time()
# v3 = engine_a
w = play_models(e3, e2, limit=half, a_color=chess.WHITE, temperature=temp, temp_decay=decay)
print(f"  ...white half done ({time.time()-t0:.0f}s)", flush=True)
b = play_models(e3, e2, limit=games - half, a_color=chess.BLACK, temperature=temp, temp_decay=decay)
elapsed = time.time() - t0

# Aggregate from v3's perspective. results keyed from WHITE's view of each game.
v3w = w['results']['1-0'] + b['results']['0-1']
v3d = w['results']['1/2-1/2'] + b['results']['1/2-1/2']
v3l = w['results']['0-1'] + b['results']['1-0']
n = v3w + v3d + v3l

print(f"\n=== HEAD-TO-HEAD  v3-37M(e9) vs v2-37M(e7)  |  {n} games  |  {elapsed:.0f}s ===")
print(f"v3-37M:  W {100*v3w/n:.1f}%   D {100*v3d/n:.1f}%   L {100*v3l/n:.1f}%    ({v3w}/{v3d}/{v3l})")
score = (v3w + 0.5*v3d) / n
print(f"v3 score: {score:.3f}  (>0.5 = v3 stronger)")
print(f"  v3-as-white: {w['results']['1-0']}W/{w['results']['1/2-1/2']}D/{w['results']['0-1']}L "
      f"| v3-as-black: {b['results']['0-1']}W/{b['results']['1/2-1/2']}D/{b['results']['1-0']}L")

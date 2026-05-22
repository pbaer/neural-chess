# -*- coding: utf-8 -*-
"""10 games: v2-37M @ 200-sim MCTS vs Stockfish ultra (d=16, s=20).
Reports W/D/L (model perspective) and total thinking time per side.

Robust to Stockfish subprocess crashes (Windows 0xC0000005 access violations
that occasionally hit a long-idle UCI engine): a fresh SF process is spawned
per game, and a crashed game is retried with a new engine.
"""
import os
import sys
import time

import chess
import chess.engine

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch
from src.inference_api import load_policy_engine
from src.engine import create_engine, generate_engine_move
from src.mcts import MCTSEngine

CKPT = 'model/v2/v2-37M/model_e0007.pt'
SIMS = 200
CPUCT = 1.5
GAMES = 10
MAX_PLIES = 400


def play_one_game(mcts, model_color, max_retries=3):
    """Play a single game vs a fresh SF-ultra. Returns
    (result_str, model_seconds, sf_seconds, plies). Retries on engine crash."""
    for attempt in range(max_retries):
        eng = create_engine(path='bin/stockfish.exe', depth=16, skill_level=20)
        board = chess.Board()
        model_s = 0.0
        sf_s = 0.0
        plies = 0
        try:
            while not board.is_game_over() and plies < MAX_PLIES:
                t0 = time.time()
                if board.turn == model_color:
                    stats = {'legal_moves': 0}
                    move = mcts.generate_move(board, stats, temperature=0.0)
                    model_s += time.time() - t0
                else:
                    move = generate_engine_move(eng, board)
                    sf_s += time.time() - t0
                board.push(move)
                plies += 1
            result = board.result() if board.is_game_over() else '1/2-1/2'
            eng.quit()
            return result, model_s, sf_s, plies
        except chess.engine.EngineTerminatedError:
            try:
                eng.close()
            except Exception:
                pass
            print(f'    (SF crashed mid-game, retry {attempt+1}/{max_retries})', flush=True)
            continue
    raise RuntimeError('Stockfish kept crashing; giving up on this game')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raw = load_policy_engine(CKPT, device=device)
    mcts = MCTSEngine(raw, c_puct=CPUCT, n_simulations=SIMS)

    print(f'v2-37M @ {SIMS}-sim MCTS (c_puct={CPUCT}) vs sf_ultra (d=16, s=20), {GAMES} games\n', flush=True)

    wins = draws = losses = 0
    total_model_s = 0.0
    total_sf_s = 0.0
    total_plies = 0

    for g in range(1, GAMES + 1):
        # alternate colors: odd games white, even games black
        model_color = chess.WHITE if g % 2 == 1 else chess.BLACK
        result, model_s, sf_s, plies = play_one_game(mcts, model_color)
        total_model_s += model_s
        total_sf_s += sf_s
        total_plies += plies

        model_won = (result == '1-0') == (model_color == chess.WHITE) and result != '1/2-1/2'
        if result == '1/2-1/2':
            draws += 1; tag = 'DRAW'
        elif model_won:
            wins += 1; tag = 'WIN'
        else:
            losses += 1; tag = 'LOSS'
        color = 'W' if model_color == chess.WHITE else 'B'
        print(f'  Game {g}/{GAMES} ({color}): {tag} in {plies} plies  '
              f'(model {model_s:.1f}s / SF {sf_s:.1f}s)', flush=True)

    model_moves = total_plies // 2
    print(f'\n=== RESULT (model perspective) ===')
    print(f'  W/D/L: {wins}/{draws}/{losses}  '
          f'({100*wins/GAMES:.0f}% / {100*draws/GAMES:.0f}% / {100*losses/GAMES:.0f}%)')
    print(f'  Total plies: {total_plies}  (~{model_moves} moves per side)')
    print(f'\n=== TOTAL THINKING TIME ===')
    print(f'  v2-37M @ {SIMS}-sim MCTS : {total_model_s:8.1f} s  ({total_model_s/60:.2f} min)')
    print(f'  Stockfish ultra (d=16)  : {total_sf_s:8.1f} s  ({total_sf_s/60:.2f} min)')
    print(f'  ratio (model / SF)      : {total_model_s/max(total_sf_s,1e-9):.2f}x')
    if model_moves > 0:
        print(f'\n  per-move avg: MCTS {total_model_s*1000/model_moves:.0f} ms  |  '
              f'SF {total_sf_s*1000/max(model_moves,1):.0f} ms')


if __name__ == '__main__':
    main()

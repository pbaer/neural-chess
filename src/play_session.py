# -*- coding: utf-8 -*-
"""Session-based play harness for one-move-at-a-time games.

Same Player abstraction as the auto-play loop in play.py engine mode, but
state persists between command invocations so an external "agent" (human at
the CLI, or an LLM tool-using AI) can submit moves one at a time.

Players (specs):
    interactive          -- submitted via `move` command on the next invocation
    agent[:LABEL]        -- same as interactive but tags the session metadata
    neural-chess:PATH    -- a v1 or v2 checkpoint (auto-detected via inference_api)
    stockfish[:DEPTH[:SKILL[:PATH]]]
                         -- UCI engine. Defaults: depth=10, skill=20, path=bin/stockfish.exe

Commands:
    start --white WSPEC --black BSPEC [--game-id ID] [--label LABEL]
        Initializes a new session, runs engine players up to first agent turn
        (or to game end if both sides are engines). Prints the resulting state.

    move --game-id ID --uci UCI
        Submits the next move (must be the agent's turn). Then advances any
        consecutive engine turns, stopping at the next agent turn or game end.
        Prints the resulting state.

    show --game-id ID
        Prints current state without changing anything.

    list
        Lists all sessions in tmp/play_sessions/.

State files: tmp/play_sessions/<game_id>.json (gitignored).

This is intended to also serve as the substrate for a future "agent-eval"
mode where the harness pits engines against LLMs over many games. The
shared design with the existing engine-vs-engine path is that both reduce
to "two Players, a Board, and a moves list" -- only the player implementations
differ.
"""
import argparse
import datetime
import json
import os
import sys
import time
import uuid
from typing import Optional

import chess
import torch

from src.inference_api import load_policy_engine

SESSION_DIR = os.path.join('tmp', 'play_sessions')

DEFAULT_STOCKFISH = 'bin/stockfish.exe'


# ---------- Player specs ----------

def parse_player_spec(spec: str) -> dict:
    """Parse a CLI player spec into a normalized dict stored in the session.

    Returns a dict with 'type' and type-specific config keys.
    """
    if spec == 'interactive':
        return {'type': 'interactive', 'label': None}
    if spec == 'agent' or spec.startswith('agent:'):
        label = spec[len('agent:'):] if ':' in spec else None
        return {'type': 'interactive', 'label': label}
    if spec.startswith('neural-chess:'):
        path = spec[len('neural-chess:'):]
        return {'type': 'neural-chess', 'path': path}
    if spec.startswith('stockfish'):
        # stockfish | stockfish:depth | stockfish:depth:skill | stockfish:depth:skill:path
        parts = spec.split(':', 3)
        depth = int(parts[1]) if len(parts) > 1 and parts[1] else 10
        skill = int(parts[2]) if len(parts) > 2 and parts[2] else 20
        path = parts[3] if len(parts) > 3 and parts[3] else DEFAULT_STOCKFISH
        return {'type': 'stockfish', 'depth': depth, 'skill': skill, 'path': path}
    raise ValueError(f"Unknown player spec: {spec!r}")


def player_label(p: dict) -> str:
    """Short human-readable label for the player (used in printouts)."""
    if p['type'] == 'interactive':
        return p.get('label') or 'agent'
    if p['type'] == 'neural-chess':
        return f"neural-chess:{p['path']}"
    if p['type'] == 'stockfish':
        return f"stockfish:d{p['depth']}:s{p['skill']}"
    return p['type']


# ---------- Session I/O ----------

def session_path(game_id: str) -> str:
    return os.path.join(SESSION_DIR, f'{game_id}.json')


def load_session(game_id: str) -> dict:
    path = session_path(game_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No session at {path!r}")
    with open(path) as f:
        return json.load(f)


def save_session(state: dict) -> None:
    os.makedirs(SESSION_DIR, exist_ok=True)
    path = session_path(state['game_id'])
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)


def reconstruct_board(state: dict) -> chess.Board:
    """Replay the recorded moves to get the current board."""
    b = chess.Board()
    for u in state['moves']:
        b.push_uci(u)
    return b


# ---------- Engine drivers ----------

_DEVICE = None


def device() -> torch.device:
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return _DEVICE


def neural_chess_move(board: chess.Board, model_path: str, temperature: float) -> chess.Move:
    """Single-shot inference: load model, generate one move."""
    engine = load_policy_engine(model_path, device=device())
    stats = {'legal_moves': 0, 'illegal_moves': 0, 'en_passant_captures': 0, 'castles': 0}
    return engine.generate_move(board, stats, temperature=temperature)


def stockfish_move(board: chess.Board, depth: int, skill: int, path: str) -> chess.Move:
    """Single-shot stockfish move at given depth/skill."""
    if not os.path.isabs(path) and os.path.exists(path):
        path = os.path.abspath(path)
    engine = chess.engine.SimpleEngine.popen_uci(path)
    try:
        engine.configure({'Skill Level': skill})
        result = engine.play(board, chess.engine.Limit(depth=depth))
        return result.move
    finally:
        engine.quit()


# late-bind chess.engine to avoid mandatory import at top
import chess.engine  # noqa: E402


def advance_until_agent_or_end(state: dict, temperature: float) -> dict:
    """Apply moves for any engine-side players whose turn it is, in sequence,
    until the next 'interactive' player's turn arrives or the game ends.

    Mutates and returns state. Caller is responsible for persisting.
    """
    board = reconstruct_board(state)
    while not board.is_game_over():
        side = state['white'] if board.turn == chess.WHITE else state['black']
        if side['type'] == 'interactive':
            break

        t0 = time.time()
        if side['type'] == 'neural-chess':
            move = neural_chess_move(board, side['path'], temperature=temperature)
        elif side['type'] == 'stockfish':
            move = stockfish_move(board, side['depth'], side['skill'], side['path'])
        else:
            raise ValueError(f"Unknown engine type: {side['type']!r}")
        elapsed = time.time() - t0

        san = board.san(move)
        board.push(move)
        state['moves'].append(move.uci())
        state.setdefault('move_log', []).append({
            'ply': len(state['moves']),
            'by': 'white' if not board.turn else 'black',  # board.turn flipped after push
            'player': player_label(side),
            'uci': move.uci(),
            'san': san,
            'elapsed_s': round(elapsed, 3),
        })

    if board.is_game_over():
        state['result'] = board.result()
        state['termination'] = describe_termination(board)
    return state


def describe_termination(board: chess.Board) -> str:
    """Human-friendly reason the game ended."""
    if board.is_checkmate():
        winner = 'black' if board.turn == chess.WHITE else 'white'
        return f'checkmate (won by {winner})'
    if board.is_stalemate():
        return 'stalemate'
    if board.is_insufficient_material():
        return 'insufficient material'
    if board.is_seventyfive_moves():
        return 'seventy-five move rule'
    if board.is_fivefold_repetition():
        return 'fivefold repetition'
    if board.can_claim_threefold_repetition():
        return 'threefold repetition (claimed)'
    if board.can_claim_fifty_moves():
        return 'fifty-move rule (claimed)'
    if board.is_variant_end():
        return 'variant end'
    return 'unknown'


# ---------- Pretty-printing ----------

def render_board_ascii(board: chess.Board, perspective: str = 'white') -> str:
    """Unicode board view. perspective ∈ {'white', 'black'} flips the view."""
    # python-chess Board.__str__ is rank 8 at top (white perspective)
    raw = str(board)
    if perspective == 'black':
        # Flip vertically
        lines = raw.splitlines()
        lines = list(reversed(lines))
        lines = [line[::-1].replace('. ', ' .')[::-1] for line in lines]  # leave dots alone-ish
        raw = '\n'.join(lines)
    # Add file/rank coords
    ranks = list(range(8, 0, -1)) if perspective == 'white' else list(range(1, 9))
    files = 'a b c d e f g h' if perspective == 'white' else 'h g f e d c b a'
    out = []
    for i, line in enumerate(raw.splitlines()):
        out.append(f"{ranks[i]} {line}")
    out.append(f"  {files}")
    return '\n'.join(out)


def render_state(state: dict, verbose: bool = True) -> str:
    """Build the canonical text printout returned by start/move/show."""
    board = reconstruct_board(state)
    lines = []
    lines.append(f"Game ID:   {state['game_id']}")
    lines.append(f"White:     {player_label(state['white'])}")
    lines.append(f"Black:     {player_label(state['black'])}")
    lines.append(f"Plies:     {len(state['moves'])}")
    lines.append(f"FEN:       {board.fen()}")

    if state.get('result'):
        lines.append(f"Result:    {state['result']}")
        lines.append(f"Reason:    {state.get('termination', '?')}")
    else:
        side = 'white' if board.turn == chess.WHITE else 'black'
        next_player = state['white'] if board.turn == chess.WHITE else state['black']
        lines.append(f"To move:   {side} ({player_label(next_player)})")
        if next_player['type'] == 'interactive':
            legal = sorted(m.uci() for m in board.legal_moves)
            lines.append(f"Legal:     {len(legal)} moves")
            if verbose:
                lines.append("           " + ' '.join(legal))

    perspective = 'white'
    # If only one side is interactive, show from their perspective
    w, b = state['white'], state['black']
    if w['type'] == 'interactive' and b['type'] != 'interactive':
        perspective = 'white'
    elif b['type'] == 'interactive' and w['type'] != 'interactive':
        perspective = 'black'

    lines.append("")
    lines.append(render_board_ascii(board, perspective=perspective))

    # Recent moves
    if verbose and state.get('move_log'):
        recent = state['move_log'][-6:]
        lines.append("")
        lines.append("Recent moves:")
        for m in recent:
            lines.append(f"  ply {m['ply']:>3}  {m['by']:>5}  {m['san']:>7}  ({m['uci']}, {m['player']}, {m['elapsed_s']}s)")

    return '\n'.join(lines)


# ---------- Commands ----------

def cmd_start(args: argparse.Namespace) -> None:
    white = parse_player_spec(args.white)
    black = parse_player_spec(args.black)
    game_id = args.game_id or f"g{int(time.time())}_{uuid.uuid4().hex[:6]}"

    state = {
        'game_id': game_id,
        'white': white,
        'black': black,
        'moves': [],
        'move_log': [],
        'result': None,
        'termination': None,
        'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'metadata': {
            'label': args.label,
            'temperature': args.temperature,
        },
    }

    print(f"Started new session: {game_id}")
    print(f"  White: {player_label(white)}")
    print(f"  Black: {player_label(black)}")
    print()

    state = advance_until_agent_or_end(state, temperature=args.temperature)
    save_session(state)
    print(render_state(state))


def cmd_move(args: argparse.Namespace) -> None:
    state = load_session(args.game_id)
    if state.get('result'):
        print(f"Game already over: {state['result']} ({state.get('termination')})")
        sys.exit(2)

    board = reconstruct_board(state)
    side = state['white'] if board.turn == chess.WHITE else state['black']
    if side['type'] != 'interactive':
        print(f"It's the engine's turn ({player_label(side)}), not the agent's. "
              f"Run `show` to inspect, or this session is mis-aligned.")
        sys.exit(2)

    uci = args.uci.strip()
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        print(f"Bad UCI string: {uci!r}")
        sys.exit(3)
    if move not in board.legal_moves:
        legal = sorted(m.uci() for m in board.legal_moves)
        print(f"Illegal move: {uci}")
        print(f"Legal ({len(legal)}): {' '.join(legal)}")
        sys.exit(3)

    san = board.san(move)
    board.push(move)
    state['moves'].append(uci)
    state.setdefault('move_log', []).append({
        'ply': len(state['moves']),
        'by': 'white' if not board.turn else 'black',  # flipped after push
        'player': player_label(side),
        'uci': uci,
        'san': san,
        'elapsed_s': 0.0,
    })

    if board.is_game_over():
        state['result'] = board.result()
        state['termination'] = describe_termination(board)
    else:
        state = advance_until_agent_or_end(
            state, temperature=state.get('metadata', {}).get('temperature', 0.0))

    save_session(state)
    print(render_state(state))


def cmd_show(args: argparse.Namespace) -> None:
    state = load_session(args.game_id)
    print(render_state(state, verbose=True))


def cmd_list(args: argparse.Namespace) -> None:
    if not os.path.isdir(SESSION_DIR):
        print("(no sessions)")
        return
    files = sorted(os.listdir(SESSION_DIR))
    if not files:
        print("(no sessions)")
        return
    for fn in files:
        if not fn.endswith('.json'):
            continue
        try:
            with open(os.path.join(SESSION_DIR, fn)) as f:
                s = json.load(f)
        except Exception as e:
            print(f"  {fn}  (unreadable: {e})")
            continue
        gid = s['game_id']
        w = player_label(s['white'])
        b = player_label(s['black'])
        plies = len(s.get('moves', []))
        res = s.get('result') or 'in-progress'
        print(f"  {gid}  {w} vs {b}  plies={plies}  {res}")


def main():
    parser = argparse.ArgumentParser(description='Session-based chess play harness')
    sub = parser.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('start', help='Start a new session')
    s.add_argument('--white', required=True, help='Player spec for white')
    s.add_argument('--black', required=True, help='Player spec for black')
    s.add_argument('--game-id', default=None, help='Optional explicit ID; auto-generated otherwise')
    s.add_argument('--label', default=None, help='Optional free-text label stored in metadata')
    s.add_argument('--temperature', type=float, default=0.0,
                   help='Neural-chess sampling temperature (default 0 = greedy)')
    s.set_defaults(func=cmd_start)

    m = sub.add_parser('move', help='Submit the next move from the interactive side')
    m.add_argument('--game-id', required=True)
    m.add_argument('--uci', required=True, help='UCI move string, e.g. e2e4 or e7e8q')
    m.set_defaults(func=cmd_move)

    sh = sub.add_parser('show', help='Print current state of a session')
    sh.add_argument('--game-id', required=True)
    sh.set_defaults(func=cmd_show)

    ls = sub.add_parser('list', help='List all sessions')
    ls.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()

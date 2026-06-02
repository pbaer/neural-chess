# -*- coding: utf-8 -*-
"""v2 dataset: PGN -> memmap-friendly binary shards, and PyTorch Dataset
that reads the shards via np.memmap.

Two responsibilities here, kept in one module to keep the file layout simple:
  1) generate_shards() — scan filtered PGN files, sample positions per the
     tier mix, featurize each, encode each move, write three memmap files
     (X.bin, Y_policy.bin, Y_value.bin) plus a meta.json.
  2) ChessDatasetV2 — PyTorch Dataset wrapping those memmap files. Fast
     random-access for training. No in-RAM materialization of the whole
     dataset — the OS pages in working sets via memmap.

Per memory/project-principles.md Principle 2: the supervision targets
generated here are exactly three things, all observable in the games:
  - the move that was played at this position (policy target)
  - the final outcome of the game (value target, from the moving player's
    perspective: +1 win, 0 draw, -1 loss)
  - (the position itself is the input X)
No computed chess features anywhere.
"""
import gc
import io
import json
import math
import os
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import chess
import chess.pgn
import numpy as np
import torch
from torch.utils.data import Dataset

from src.v2.featurize import NUM_PLANES, featurize, rotate_square
from src.v2.moves import NUM_MOVES, encode_move


# ---- Shard generation (PGN -> memmap binary) ----

@dataclass
class ShardSpec:
    """Output of generate_shards(); also persisted as meta.json."""
    n_samples: int
    input_planes: int = NUM_PLANES
    n_move_classes: int = NUM_MOVES
    x_dtype: str = 'int8'      # binary planes
    y_policy_dtype: str = 'int32'
    y_value_dtype: str = 'int8'
    # Mix actually achieved (may differ slightly from request)
    tier_counts: dict = None
    tier_mix_requested: dict = None
    sources: dict = None
    skip_first_plies: int = 2
    # Future-move self-supervision targets (Principle-2-clean: the moves
    # actually played at the next N plies of the same game). 0 = none.
    n_future_moves: int = 0
    y_future_dtype: str = 'int32'
    # Provenance
    generated_at: str = ''


def _iter_pgn_games(pgn_path: str):
    """Yield chess.pgn.Game objects from a PGN file (one at a time, low memory)."""
    with open(pgn_path, 'r', encoding='utf-8', errors='replace') as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                return
            yield game


def _outcome_from_result(result: str, side_to_move: bool) -> int:
    """Convert (result string, who's to move at this position) -> {-1, 0, +1}.
    Value is from the moving player's perspective.
    """
    if result == '1/2-1/2':
        return 0
    if result == '1-0':
        return 1 if side_to_move == chess.WHITE else -1
    if result == '0-1':
        return 1 if side_to_move == chess.BLACK else -1
    return 0


def _process_game(game: chess.pgn.Game, skip_first_plies: int = 2):
    """Yield (x, y_policy, y_value) tuples for every ply in the game past skip.
    Both sides' moves included.

    Featurization applies rotation for black-to-move (so the moving side is
    always at the bottom). The move's (from_sq, to_sq) are mirrored to match
    the rotated frame so the policy target lives in the same coordinate
    system as the input.
    """
    result = game.headers.get('Result', '*')
    if result not in ('1-0', '0-1', '1/2-1/2'):
        return  # no clear outcome; skip

    board = game.board()
    ply_idx = 0
    for move in game.mainline_moves():
        # Skip first N plies (no useful signal — opening universals)
        if ply_idx >= skip_first_plies:
            # Encode the move IN THE ROTATED FRAME if board is black-to-move
            if board.turn == chess.WHITE:
                mt_idx = encode_move(move)
            else:
                rotated_move = chess.Move(
                    rotate_square(move.from_square),
                    rotate_square(move.to_square),
                    promotion=move.promotion,
                )
                mt_idx = encode_move(rotated_move)
            if mt_idx >= 0:
                x = featurize(board)  # handles rotation internally
                y_value = _outcome_from_result(result, board.turn)
                yield x, mt_idx, y_value
        board.push(move)
        ply_idx += 1


def generate_shards(
    filtered_dir: str,
    out_dir: str,
    target_positions: int,
    tier_mix: dict = None,
    skip_first_plies: int = 2,
    seed: int = 0,
    progress_every: int = 10000,
    n_future_moves: int = 0,
):
    """Build (X.bin, Y_policy.bin, Y_value.bin, meta.json) from filtered PGNs.

    Args:
        filtered_dir: directory with tier_top_*.pgn, tier_mid_*.pgn, tier_low_*.pgn
        out_dir: where to write shard files
        target_positions: total positions to write
        tier_mix: dict mapping tier name -> proportion. Defaults to 65/25/10.
        skip_first_plies: drop the first N plies of each game (no learning signal)
        seed: random seed for game sampling

    The current strategy:
        - Compute per-tier target = total * proportion
        - For each tier, stream-read its PGN file, randomly accept games with
          a probability calibrated to hit the target (reservoir-light approach:
          we assume the source has more games than needed and uniformly subsample)
        - For each accepted game, emit all (position, move, outcome) tuples
          past the skip
        - Stop when we have enough positions

    Writes to memmap files of pre-allocated maximum size; truncates afterward.
    """
    if tier_mix is None:
        tier_mix = {'top': 0.65, 'mid': 0.25, 'low': 0.10}

    # Atomic write: build into a `.tmp` sibling, then rename into place at end.
    # Cleanup-on-start: delete any prior tmp from a failed run AND any existing
    # contents at the final out_dir (segfaults last night were traced to stale
    # large memmap files from earlier failed runs colliding with new 'w+'
    # allocations on Windows).
    final_dir = out_dir
    tmp_dir = out_dir + '.tmp'
    for d in (tmp_dir, final_dir):
        if os.path.exists(d):
            print(f"Removing existing {d!r}", flush=True)
            shutil.rmtree(d)
    os.makedirs(tmp_dir)

    # Pre-allocate memmap files (overshoot slightly, will write meta with actual count)
    cap = int(target_positions * 1.05) + 1000

    x_path = os.path.join(tmp_dir, 'X.bin')
    yp_path = os.path.join(tmp_dir, 'Y_policy.bin')
    yv_path = os.path.join(tmp_dir, 'Y_value.bin')
    yf_path = os.path.join(tmp_dir, 'Y_future.bin')

    X = np.memmap(x_path, dtype=np.int8, mode='w+',
                  shape=(cap, NUM_PLANES, 8, 8))
    YP = np.memmap(yp_path, dtype=np.int32, mode='w+', shape=(cap,))
    YV = np.memmap(yv_path, dtype=np.int8, mode='w+', shape=(cap,))
    # Future-move targets (cap, n_future_moves) int32; -1 = no such future ply
    # (game ended) -> masked at train time via cross-entropy ignore_index=-1.
    YF = None
    if n_future_moves > 0:
        YF = np.memmap(yf_path, dtype=np.int32, mode='w+',
                       shape=(cap, n_future_moves))
    # First flush so the OS knows the full file size up front.
    X.flush(); YP.flush(); YV.flush()
    if YF is not None:
        YF.flush()

    rng = random.Random(seed)

    write_idx = 0
    tier_counts = {t: 0 for t in tier_mix}
    sources = {t: '' for t in tier_mix}
    tier_files = {
        'top': 'tier_top_2400plus.pgn',
        'mid': 'tier_mid_1900-2400.pgn',
        'low': 'tier_low_1600-1900.pgn',
    }

    for tier, proportion in tier_mix.items():
        tier_target = int(round(target_positions * proportion))
        tier_start_write = write_idx
        tier_pgn = os.path.join(filtered_dir, tier_files[tier])
        sources[tier] = tier_files[tier]
        if not os.path.exists(tier_pgn):
            print(f"WARN: {tier_pgn} not found, skipping tier {tier!r}")
            continue
        print(f"Tier {tier}: target {tier_target:,} positions from {tier_pgn}", flush=True)

        game_count = 0
        positions_in_tier = 0
        for game in _iter_pgn_games(tier_pgn):
            game_count += 1
            samples = list(_process_game(game, skip_first_plies))
            if not samples:
                continue
            # Future-move targets: for sample j, the policy labels of samples
            # j+1..j+n_future_moves WITHIN THE SAME GAME (-1 past game end).
            # Computed from the full game so truncating the write below is safe.
            futures = None
            if n_future_moves > 0:
                pol = [s[1] for s in samples]
                m = len(samples)
                futures = [[pol[j + k] if j + k < m else -1
                            for k in range(1, n_future_moves + 1)]
                           for j in range(m)]
            n = len(samples)
            # Bounds check vs memmap cap
            if write_idx + n > cap:
                n = cap - write_idx
                samples = samples[:n]
                if futures is not None:
                    futures = futures[:n]
            for i, (x, y_pol, y_val) in enumerate(samples):
                X[write_idx] = x.astype(np.int8)
                YP[write_idx] = y_pol
                YV[write_idx] = y_val
                if YF is not None:
                    YF[write_idx] = futures[i]
                write_idx += 1
                positions_in_tier += 1
                if positions_in_tier >= tier_target:
                    break
            if positions_in_tier >= tier_target:
                break
            if game_count % 1000 == 0:
                print(f"  {tier}: {game_count} games -> {positions_in_tier:,} positions",
                      flush=True)
            if write_idx % progress_every == 0:
                # ensure memmap is flushed periodically for crash-safety
                X.flush()
                YP.flush()
                YV.flush()
                if YF is not None:
                    YF.flush()

        tier_counts[tier] = write_idx - tier_start_write
        print(f"  {tier} DONE: {tier_counts[tier]:,} positions from "
              f"{game_count:,} games", flush=True)

    X.flush(); YP.flush(); YV.flush()
    if YF is not None:
        YF.flush()
    # Explicit cleanup: numpy memmap objects hold OS-level file handles that
    # don't always release in time for the subsequent truncate() on Windows.
    # del + gc.collect makes the release synchronous.
    del X, YP, YV
    if YF is not None:
        del YF
    gc.collect()

    # Truncate to actual size
    actual = write_idx
    print(f"\nTotal: {actual:,} positions written")

    with open(x_path, 'r+b') as f:
        f.truncate(actual * NUM_PLANES * 8 * 8)
    with open(yp_path, 'r+b') as f:
        f.truncate(actual * 4)
    with open(yv_path, 'r+b') as f:
        f.truncate(actual * 1)
    if n_future_moves > 0:
        with open(yf_path, 'r+b') as f:
            f.truncate(actual * n_future_moves * 4)

    # Write meta — to tmp dir, NOT final dir yet
    import datetime
    spec = ShardSpec(
        n_samples=actual,
        tier_counts=tier_counts,
        tier_mix_requested=tier_mix,
        sources=sources,
        skip_first_plies=skip_first_plies,
        n_future_moves=n_future_moves,
        generated_at=datetime.datetime.now().isoformat(timespec='seconds'),
    )
    with open(os.path.join(tmp_dir, 'meta.json'), 'w') as f:
        json.dump(asdict(spec), f, indent=2)

    # Atomic finalize: rename tmp -> final. If anything failed above, the tmp
    # dir is still on disk with clear evidence; no half-finished final/.
    os.rename(tmp_dir, final_dir)
    print(f"Wrote meta.json: {asdict(spec)}")
    return spec


# ---- PyTorch Dataset ----

class ChessDatasetV2(Dataset):
    """Reads (X.bin, Y_policy.bin, Y_value.bin) shards via np.memmap.

    Memory footprint is bounded by the OS page-cache working set, not the
    total dataset size — supports arbitrarily large training shards.
    """

    def __init__(self, shard_dir: str, with_future: bool = False):
        with open(os.path.join(shard_dir, 'meta.json')) as f:
            meta = json.load(f)
        self.n_samples = meta['n_samples']
        self.input_planes = meta['input_planes']
        self.n_move_classes = meta['n_move_classes']
        self.shard_dir = shard_dir
        # Future-move targets: opt-in (with_future) AND present in the shard.
        # Default off keeps the 3-tuple interface for existing callers.
        self.n_future_moves = meta.get('n_future_moves', 0) if with_future else 0
        if with_future and self.n_future_moves == 0:
            raise ValueError(
                f"with_future=True but shard {shard_dir!r} has no future-move "
                f"targets (meta n_future_moves=0). Regenerate with --future-moves.")
        self._open_memmaps()

    def _open_memmaps(self):
        self.X = np.memmap(os.path.join(self.shard_dir, 'X.bin'),
                           dtype=np.int8, mode='r',
                           shape=(self.n_samples, self.input_planes, 8, 8))
        self.YP = np.memmap(os.path.join(self.shard_dir, 'Y_policy.bin'),
                            dtype=np.int32, mode='r',
                            shape=(self.n_samples,))
        self.YV = np.memmap(os.path.join(self.shard_dir, 'Y_value.bin'),
                            dtype=np.int8, mode='r',
                            shape=(self.n_samples,))
        self.YF = None
        if self.n_future_moves > 0:
            self.YF = np.memmap(os.path.join(self.shard_dir, 'Y_future.bin'),
                                dtype=np.int32, mode='r',
                                shape=(self.n_samples, self.n_future_moves))

    # np.memmap can't be pickled through the Windows multiprocessing pipe
    # (OSError 22). Strip the memmaps before pickling and re-open them in
    # the worker after unpickling — workers each get their own file handle.
    def __getstate__(self):
        return {
            'n_samples': self.n_samples,
            'input_planes': self.input_planes,
            'n_move_classes': self.n_move_classes,
            'shard_dir': self.shard_dir,
            'n_future_moves': self.n_future_moves,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open_memmaps()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.X[idx].astype(np.float32)
        if self.YF is not None:
            return (
                torch.from_numpy(x),
                int(self.YP[idx]),
                float(self.YV[idx]),
                torch.from_numpy(self.YF[idx].astype(np.int64)),
            )
        return (
            torch.from_numpy(x),
            int(self.YP[idx]),
            float(self.YV[idx]),
        )


if __name__ == '__main__':
    # CLI: generate shards
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--filtered-dir', default='data/v2/filtered',
                        help='directory with tier_*.pgn files')
    parser.add_argument('--out-dir', required=True,
                        help='where to write X.bin, Y_policy.bin, Y_value.bin, meta.json')
    parser.add_argument('--positions', type=int, required=True,
                        help='target number of positions')
    parser.add_argument('--top-pct', type=float, default=0.65)
    parser.add_argument('--mid-pct', type=float, default=0.25)
    parser.add_argument('--low-pct', type=float, default=0.10)
    parser.add_argument('--skip-plies', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--future-moves', type=int, default=0,
                        help='emit Y_future.bin with the next N moves played '
                             '(self-supervision targets); 0 = none')
    args = parser.parse_args()

    tier_mix = {'top': args.top_pct, 'mid': args.mid_pct, 'low': args.low_pct}
    generate_shards(args.filtered_dir, args.out_dir, args.positions,
                    tier_mix=tier_mix, skip_first_plies=args.skip_plies,
                    seed=args.seed, n_future_moves=args.future_moves)

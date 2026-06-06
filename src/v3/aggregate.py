# -*- coding: utf-8 -*-
"""Position-aggregated dataset builder (the [[position-aggregated-dataset]] idea).

Streams the same filtered human-game PGNs the v2/v3 shards come from, but instead
of emitting one (position, single-game-move, single-game-outcome) row per ply,
it AGGREGATES by unique position. For every unique position it records:

  - value_avg : mean game outcome over ALL games reaching it (de-noised Q,
                from the side-to-move's perspective, in [-1, 1])
  - value_one : the outcome of ONE uniformly-random instance (reservoir-sampled)
                -- lets us reproduce the current "single-game value" baseline
                from the same artifact, for a clean A/B.
  - a move histogram (the distribution of human moves played from the position)
                -- a soft policy target; preserves what hard-dedup would destroy.
  - count     : number of instances (games x plies) that reached the position
                -- drives count^tau frequency-tempered sampling at train time.

Principle check (memory/project-principles.md): every stored target is still an
*observed* quantity from human games (an average of real outcomes, a histogram of
real moves). Nothing is computed by an engine. Principle-1/2 clean.

Position identity key = python-chess polyglot Zobrist hash, which encodes piece
placement + side-to-move + castling rights + (capturable) en-passant -- exactly
the threefold-repetition notion of "same position". Move counters do NOT enter
the key (so transpositions reached at different move numbers merge); colors do
NOT merge (a white-to-move position and its color-flip are distinct keys).

Featurization, move encoding and the int8 storage truncation are IDENTICAL to
src/v2/dataset.py so a model trained on this artifact sees the same input
distribution as the production frequency-weighted shards -- the only thing that
changes is the labels and the sampling.

Output (memmap artifact, in OUT_DIR):
  X.bin           int8    (U, 21, 8, 8)   representative featurization (1st seen)
  value_avg.bin   float32 (U,)            mean outcome
  value_one.bin   int8    (U,)            one random instance's outcome {-1,0,1}
  count.bin       int32   (U,)            instance count
  hist_ptr.bin    int64   (U+1,)          CSR offsets into hist_*
  hist_moves.bin  int32   (H,)            move indices (rotated frame, 0..4671)
  hist_counts.bin int32   (H,)            per-move counts
  split.bin       int8    (U,)            0 = train, 1 = val (held-out)
  meta.json
"""
import argparse
import gc
import json
import os
import random
import shutil
import time

import chess
import chess.pgn
import chess.polyglot
import numpy as np

from src.v2.featurize import NUM_PLANES, featurize, rotate_square
from src.v2.moves import NUM_MOVES, encode_move


TIER_FILES = {
    'top': 'tier_top_2400plus.pgn',
    'mid': 'tier_mid_1900-2400.pgn',
    'low': 'tier_low_1600-1900.pgn',
}


def _outcome(result: str, side_to_move: bool) -> int:
    if result == '1/2-1/2':
        return 0
    if result == '1-0':
        return 1 if side_to_move == chess.WHITE else -1
    if result == '0-1':
        return 1 if side_to_move == chess.BLACK else -1
    return 0


def _encoded_move(board: chess.Board, move: chess.Move) -> int:
    """Encode a move in the rotated frame (same convention as the model input)."""
    if board.turn == chess.WHITE:
        return encode_move(move)
    rm = chess.Move(rotate_square(move.from_square),
                    rotate_square(move.to_square), promotion=move.promotion)
    return encode_move(rm)


def build(filtered_dir, out_dir, target_instances, tier_mix=None,
          skip_first_plies=2, seed=0, val_mod=20, cap_unique=None,
          progress_every=500000):
    if tier_mix is None:
        tier_mix = {'top': 0.65, 'mid': 0.25, 'low': 0.10}
    if cap_unique is None:
        cap_unique = int(target_instances * 0.98) + 100000
    rng = random.Random(seed)

    tmp_dir = out_dir + '.tmp'
    for d in (tmp_dir, out_dir):
        if os.path.exists(d):
            print(f"Removing existing {d!r}", flush=True)
            shutil.rmtree(d)
    os.makedirs(tmp_dir)

    # Output X memmap (representative featurization, written on discovery).
    X = np.memmap(os.path.join(tmp_dir, 'X.bin'), dtype=np.int8, mode='w+',
                  shape=(cap_unique, NUM_PLANES, 8, 8))
    # Scalar accumulators (preallocated to cap; truncated at end).
    value_sum = np.zeros(cap_unique, dtype=np.float64)
    count = np.zeros(cap_unique, dtype=np.int64)
    value_one = np.zeros(cap_unique, dtype=np.int8)
    # Move histogram: stream (row, move) pairs to a flat int32 array and build
    # the CSR histogram afterward via a vectorized sort+group. This avoids the
    # per-position Python dicts that dominated RAM (~900 B/unique) and capped the
    # corpus size — pairs cost only 8 B/instance, so the full 100M fits in ~10GB.
    cap_pairs = int(target_instances * 1.01) + 10000
    pairs = np.empty((cap_pairs, 2), dtype=np.int32)
    pair_idx = 0

    key_to_row = {}
    n_rows = 0
    n_inst = 0
    n_dropped = 0       # instances we could not add a NEW row for (cap hit)
    t0 = time.time()
    tier_counts = {}

    for tier, proportion in tier_mix.items():
        tier_target = int(round(target_instances * proportion))
        tier_pgn = os.path.join(filtered_dir, TIER_FILES[tier])
        if not os.path.exists(tier_pgn):
            print(f"WARN: {tier_pgn} not found, skipping {tier!r}", flush=True)
            continue
        print(f"Tier {tier}: target {tier_target:,} instances from {tier_pgn}",
              flush=True)
        tier_inst = 0
        n_games = 0
        with open(tier_pgn, 'r', encoding='utf-8', errors='replace') as f:
            while tier_inst < tier_target:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                result = game.headers.get('Result', '*')
                if result not in ('1-0', '0-1', '1/2-1/2'):
                    continue
                n_games += 1
                board = game.board()
                ply = 0
                for move in game.mainline_moves():
                    if ply >= skip_first_plies:
                        mt = _encoded_move(board, move)
                        if mt >= 0:
                            key = chess.polyglot.zobrist_hash(board)
                            val = _outcome(result, board.turn)
                            row = key_to_row.get(key, -1)
                            if row < 0:
                                if n_rows >= cap_unique:
                                    n_dropped += 1
                                else:
                                    row = n_rows
                                    key_to_row[key] = row
                                    X[row] = featurize(board).astype(np.int8)
                                    n_rows += 1
                            if row >= 0:
                                count[row] += 1
                                value_sum[row] += val
                                # reservoir: keep a uniformly random instance's outcome
                                if rng.random() < 1.0 / count[row]:
                                    value_one[row] = val
                                pairs[pair_idx, 0] = row
                                pairs[pair_idx, 1] = mt
                                pair_idx += 1
                                tier_inst += 1
                                n_inst += 1
                                if n_inst % progress_every == 0:
                                    dt = time.time() - t0
                                    print(f"  {n_inst:,} inst | {n_rows:,} uniq "
                                          f"({n_rows/n_inst:.3f}) | {n_inst/dt:,.0f}/s",
                                          flush=True)
                    board.push(move)
                    ply += 1
                    if tier_inst >= tier_target:
                        break
        tier_counts[tier] = tier_inst
        print(f"  {tier} DONE: {tier_inst:,} inst from {n_games:,} games", flush=True)

    U = n_rows
    print(f"\nUnique positions: {U:,} from {n_inst:,} instances "
          f"(ratio {U/max(n_inst,1):.3f}); dropped {n_dropped:,} over cap", flush=True)

    # Build CSR histogram + value_avg + split.
    value_avg = (value_sum[:U] / np.maximum(count[:U], 1)).astype(np.float32)
    cnt = count[:U].astype(np.int32)
    vone = value_one[:U].copy()
    # Vectorized CSR histogram from the (row, move) pair stream: sort by
    # row-major key (row*5000 + move; move < 4672 < 5000), then group runs of
    # equal (row,move) into per-move counts. Rows end up sorted, so hist_ptr is
    # a searchsorted of row boundaries. Every row has >=1 pair (its discovery
    # instance), so no empty rows.
    P = pairs[:pair_idx]
    combined = P[:, 0].astype(np.int64) * 5000 + P[:, 1]
    order = np.argsort(combined)
    cs = combined[order]
    ms = P[order, 1].astype(np.int32)
    rs = P[order, 0].astype(np.int64)
    change = np.empty(cs.shape[0], dtype=bool)
    change[0] = True
    np.not_equal(cs[1:], cs[:-1], out=change[1:])
    starts = np.flatnonzero(change)
    hist_moves = ms[starts]
    hist_counts = np.diff(np.append(starts, cs.shape[0])).astype(np.int32)
    group_rows = rs[starts]
    hist_ptr = np.searchsorted(group_rows, np.arange(U + 1)).astype(np.int64)
    H = int(starts.shape[0])
    del P, combined, order, cs, ms, rs, change, starts, group_rows
    gc.collect()
    # train/val split by key hash (deterministic, position-level holdout).
    # Recompute parity from row order: we stored rows in discovery order, so use
    # a hash of the row's representative key. Simpler: stash keys array.
    keys = np.fromiter(key_to_row.keys(), dtype=np.uint64, count=U)
    rows = np.fromiter(key_to_row.values(), dtype=np.int64, count=U)
    split = np.zeros(U, dtype=np.int8)
    order = np.empty(U, dtype=np.uint64)
    order[rows] = keys
    split[(order % val_mod) == 0] = 1

    # Persist.
    def _save(name, arr):
        arr.tofile(os.path.join(tmp_dir, name))
    # Truncate X to U rows.
    X.flush()
    del X
    gc.collect()
    with open(os.path.join(tmp_dir, 'X.bin'), 'r+b') as f:
        f.truncate(U * NUM_PLANES * 8 * 8)
    _save('value_avg.bin', value_avg)
    _save('value_one.bin', vone)
    _save('count.bin', cnt)
    _save('hist_ptr.bin', hist_ptr)
    _save('hist_moves.bin', hist_moves)
    _save('hist_counts.bin', hist_counts)
    _save('split.bin', split)

    meta = {
        'n_unique': int(U),
        'n_instances': int(n_inst),
        'n_hist_entries': int(H),
        'n_dropped': int(n_dropped),
        'input_planes': NUM_PLANES,
        'n_move_classes': NUM_MOVES,
        'skip_first_plies': skip_first_plies,
        'tier_mix_requested': tier_mix,
        'tier_counts': tier_counts,
        'val_mod': val_mod,
        'n_val': int(split.sum()),
        'seed': seed,
        'unique_ratio': float(U / max(n_inst, 1)),
    }
    with open(os.path.join(tmp_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    os.rename(tmp_dir, out_dir)
    dt = time.time() - t0
    print(f"\nWrote {out_dir} in {dt:.0f}s. meta={json.dumps(meta, indent=2)}")
    return meta


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--filtered-dir', default='data/v2/filtered')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--instances', type=int, required=True,
                    help='target number of position-instances to aggregate')
    ap.add_argument('--top-pct', type=float, default=0.65)
    ap.add_argument('--mid-pct', type=float, default=0.25)
    ap.add_argument('--low-pct', type=float, default=0.10)
    ap.add_argument('--skip-plies', type=int, default=2)
    ap.add_argument('--val-mod', type=int, default=20, help='1/N positions held out')
    ap.add_argument('--cap-unique', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    build(args.filtered_dir, args.out_dir, args.instances,
          tier_mix={'top': args.top_pct, 'mid': args.mid_pct, 'low': args.low_pct},
          skip_first_plies=args.skip_plies, seed=args.seed,
          val_mod=args.val_mod, cap_unique=args.cap_unique)

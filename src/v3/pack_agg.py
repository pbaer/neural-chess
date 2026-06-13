# -*- coding: utf-8 -*-
"""Build a compact, RAM-resident artifact from a position-aggregated corpus so
the fast in-RAM trainer (`src/v3/train_agg_fast.py`) can be GPU-bound instead of
disk-I/O-bound.

WHY: training `v3-micro-tau` on `agg_100M` is I/O-paced (~2000 samp/s) because
each sample is a random ~1.3KB read from the 97GB `X.bin` memmap. The model is
tiny (~0.7M params) and could consume data ~30-80k samp/s. The fix is to hold
the whole training working set in RAM. Raw int8 X is 97GB (> 64GB RAM), but
~20 of the 21 planes are binary {0,1} after the int8 truncation, so bit-packing
shrinks X to ~16GB — the full 72M-unique corpus then fits in RAM with no data or
precision loss.

LOSSLESSNESS: which planes are binary is determined by a FULL scan of X.bin (not
assumed). Binary planes (values subset of {0,1}) are bit-packed (np.packbits,
big-endian); non-binary planes (only plane 18 = fullmove/100 in {0,1,2}) are
kept as raw int8 bytes. The trainer reconstructs (B,21,8,8) float EXACTLY equal
to `X[i].astype(float32)` — verified by a bit-exact round-trip test.

Also precomputes the soft-policy target as a dense top-K (moves+probs) so the
trainer's per-batch policy target is a gather, not per-sample CSR slicing.

Output (`out_dir`):
  Xpacked.bin    uint8   (U, PB)   bit-packed binary planes ++ raw int8 non-binary
  pol_moves.bin  int16   (U, K)    top-K human move indices (rotated frame, 0-padded)
  pol_probs.bin  float16 (U, K)    renormalized top-K human-move probabilities (0-padded)
  value_avg.bin  float32 (U,)      copied from source (denoised mean outcome Q)
  count.bin      int32   (U,)      copied (drives count^tau sampling)
  split.bin      int8    (U,)      copied (0=train, 1=val)
  meta.json      layout: binary_planes / nonbinary_planes / packed_bytes / topk / ...
"""
import argparse
import json
import os
import shutil
import time

import numpy as np


def _open_src(agg_dir):
    with open(os.path.join(agg_dir, 'meta.json')) as f:
        meta = json.load(f)
    U = meta['n_unique']
    P = meta['input_planes']
    H = meta['n_hist_entries']
    mm = lambda n, dt, sh: np.memmap(os.path.join(agg_dir, n), dtype=dt, mode='r', shape=sh)
    src = {
        'meta': meta, 'U': U, 'P': P, 'H': H,
        'X': mm('X.bin', np.int8, (U, P, 8, 8)),
        'value_avg': mm('value_avg.bin', np.float32, (U,)),
        'count': mm('count.bin', np.int32, (U,)),
        'split': mm('split.bin', np.int8, (U,)),
        'hist_ptr': mm('hist_ptr.bin', np.int64, (U + 1,)),
        'hist_moves': mm('hist_moves.bin', np.int32, (H,)),
        'hist_counts': mm('hist_counts.bin', np.int32, (H,)),
    }
    return src


def scan_binary_planes(X, U, P, chunk):
    """Full scan: a plane is 'binary' iff every stored int8 value is in {0,1}."""
    nonbin = np.zeros(P, dtype=bool)
    pmin = np.full(P, 127, dtype=np.int64)
    pmax = np.full(P, -128, dtype=np.int64)
    t0 = time.time()
    for s in range(0, U, chunk):
        e = min(s + chunk, U)
        c = np.asarray(X[s:e]).reshape(e - s, P, 64)
        nonbin |= ((c != 0) & (c != 1)).any(axis=(0, 2))
        pmin = np.minimum(pmin, c.min(axis=(0, 2)))
        pmax = np.maximum(pmax, c.max(axis=(0, 2)))
        print(f"\r  scan {e:,}/{U:,} ({(e)/U*100:.0f}%) {(time.time()-t0):.0f}s",
              end='', flush=True)
    print()
    binary_planes = [p for p in range(P) if not nonbin[p]]
    nonbinary_planes = [p for p in range(P) if nonbin[p]]
    for p in range(P):
        tag = 'BIN ' if not nonbin[p] else 'INT8'
        print(f"  plane {p:2d}: {tag} range [{pmin[p]},{pmax[p]}]")
    # Sanity for the int8 path: featurize values are all >= 0, so uint8 == int8.
    assert pmin.min() >= 0, "negative int8 plane value -> packing assumption broken"
    return binary_planes, nonbinary_planes


def pack_X(X, U, P, binary_planes, nonbinary_planes, out_path, chunk):
    nb, nn = len(binary_planes), len(nonbinary_planes)
    PB = nb * 8 + nn * 64
    bidx = np.array(binary_planes, dtype=np.int64)
    nidx = np.array(nonbinary_planes, dtype=np.int64)
    Xp = np.memmap(out_path, dtype=np.uint8, mode='w+', shape=(U, PB))
    t0 = time.time()
    for s in range(0, U, chunk):
        e = min(s + chunk, U)
        c = np.asarray(X[s:e]).reshape(e - s, P, 64)
        # binary planes -> bits (big-endian); 64 is a multiple of 8 so each plane
        # occupies a clean 8-byte slice -> plane p -> bytes [p*8:(p+1)*8].
        b = c[:, bidx, :].reshape(e - s, nb * 64).astype(np.uint8)
        packed_bin = np.packbits(b, axis=1)                      # (m, nb*8)
        Xp[s:e, :nb * 8] = packed_bin
        if nn:
            # non-binary planes: raw int8 values (>=0) stored as uint8 bytes.
            Xp[s:e, nb * 8:] = c[:, nidx, :].reshape(e - s, nn * 64).astype(np.uint8)
        print(f"\r  pack {e:,}/{U:,} ({e/U*100:.0f}%) {(time.time()-t0):.0f}s",
              end='', flush=True)
    print()
    Xp.flush()
    del Xp
    return PB


def build_policy_targets(src, U, K, out_dir, chunk):
    """Dense top-K (moves, probs) per position from the CSR move histogram.
    Order within a row is irrelevant to the soft-policy loss (a sum over K), so
    rows with <=K distinct moves are scattered directly; rows with >K take the
    top-K by count. Probs renormalized over the kept moves."""
    hist_ptr, hist_moves, hist_counts = src['hist_ptr'], src['hist_moves'], src['hist_counts']
    pm = np.memmap(os.path.join(out_dir, 'pol_moves.bin'), dtype=np.int16, mode='w+', shape=(U, K))
    pp = np.memmap(os.path.join(out_dir, 'pol_probs.bin'), dtype=np.float16, mode='w+', shape=(U, K))
    t0 = time.time()
    n_truncated = 0
    for ps in range(0, U, chunk):
        pe = min(ps + chunk, U)
        Pn = pe - ps
        es, ee = int(hist_ptr[ps]), int(hist_ptr[pe])
        cmoves = np.asarray(hist_moves[es:ee])
        ccounts = np.asarray(hist_counts[es:ee]).astype(np.float32)
        local_ptr = np.asarray(hist_ptr[ps:pe + 1]) - es          # (Pn+1,)
        clen = local_ptr[1:] - local_ptr[:-1]                     # (Pn,)
        dmoves = np.zeros((Pn, K), dtype=np.int16)
        dcounts = np.zeros((Pn, K), dtype=np.float32)
        # vectorized fill for rows with <= K moves
        entry_pos = np.repeat(np.arange(Pn), clen)                # (E,) local row per entry
        entry_rank = np.arange(ee - es) - local_ptr[entry_pos]    # within-row rank
        ok = clen[entry_pos] <= K
        dmoves[entry_pos[ok], entry_rank[ok]] = cmoves[ok].astype(np.int16)
        dcounts[entry_pos[ok], entry_rank[ok]] = ccounts[ok]
        # rows with > K moves: keep top-K by count
        big = np.nonzero(clen > K)[0]
        for j in big:
            a, b = local_ptr[j], local_ptr[j + 1]
            cc = ccounts[a:b]
            top = np.argsort(cc)[::-1][:K]
            dmoves[j, :K] = cmoves[a:b][top].astype(np.int16)
            dcounts[j, :K] = cc[top]
        n_truncated += len(big)
        rowsum = dcounts.sum(axis=1, keepdims=True)
        rowsum[rowsum == 0] = 1.0
        pm[ps:pe] = dmoves
        pp[ps:pe] = (dcounts / rowsum).astype(np.float16)
        print(f"\r  pol {pe:,}/{U:,} ({pe/U*100:.0f}%) trunc>{K}:{n_truncated:,} "
              f"{(time.time()-t0):.0f}s", end='', flush=True)
    print()
    pm.flush(); pp.flush(); del pm, pp
    return n_truncated


def copy_scalars(src, U, out_dir):
    np.asarray(src['value_avg'][:U]).astype(np.float32).tofile(os.path.join(out_dir, 'value_avg.bin'))
    np.asarray(src['count'][:U]).astype(np.int32).tofile(os.path.join(out_dir, 'count.bin'))
    np.asarray(src['split'][:U]).astype(np.int8).tofile(os.path.join(out_dir, 'split.bin'))


def unpack_reference(packed_row, binary_planes, nonbinary_planes, P):
    """CPU reference reconstruction of ONE packed row -> (P,8,8) float32.
    Mirrors what the trainer does on the GPU; used by the round-trip self-test."""
    nb, nn = len(binary_planes), len(nonbinary_planes)
    x = np.zeros((P, 64), dtype=np.float32)
    bits = np.unpackbits(packed_row[:nb * 8]).reshape(nb, 64)     # big-endian
    for k, p in enumerate(binary_planes):
        x[p] = bits[k]
    if nn:
        nbytes = packed_row[nb * 8:].reshape(nn, 64)
        for k, p in enumerate(nonbinary_planes):
            x[p] = nbytes[k].astype(np.float32)
    return x.reshape(P, 8, 8)


def self_test(src, out_dir, meta, n=20000):
    """Bit-exact: unpack(pack(X[i])) == X[i].astype(float32) for a sample."""
    U, P = meta['n_packed'], src['P']
    PB = meta['packed_bytes']
    Xp = np.memmap(os.path.join(out_dir, 'Xpacked.bin'), dtype=np.uint8, mode='r', shape=(U, PB))
    rng = np.random.default_rng(0)
    idx = rng.choice(U, size=min(n, U), replace=False)
    bad = 0
    for i in idx:
        ref = np.asarray(src['X'][int(i)]).astype(np.float32)
        rec = unpack_reference(np.asarray(Xp[int(i)]), meta['binary_planes'],
                               meta['nonbinary_planes'], P)
        if not np.array_equal(ref, rec):
            bad += 1
    print(f"  round-trip: {len(idx)-bad}/{len(idx)} exact" + (" -- FAIL" if bad else " -- OK"))
    return bad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--agg-dir', default='data/v2/agg_100M')
    ap.add_argument('--out-dir', default='data/v2/agg_100M_packed')
    ap.add_argument('--topk', type=int, default=32)
    ap.add_argument('--chunk', type=int, default=2_000_000)
    ap.add_argument('--limit', type=int, default=0, help='process only first N positions (testing)')
    ap.add_argument('--skip-existing', action='store_true', help='skip stages whose output exists')
    args = ap.parse_args()

    src = _open_src(args.agg_dir)
    U = src['U'] if args.limit <= 0 else min(args.limit, src['U'])
    P = src['P']
    print(f"src={args.agg_dir} U={src['U']:,} (processing {U:,}) planes={P} "
          f"H={src['H']:,} topK={args.topk}", flush=True)

    tmp = args.out_dir + '.tmp'
    if not args.skip_existing:
        for d in (tmp, args.out_dir):
            if os.path.exists(d):
                print(f"removing {d!r}"); shutil.rmtree(d)
    os.makedirs(tmp, exist_ok=True)

    print("[1/4] scanning planes for binary-ness ...", flush=True)
    binary_planes, nonbinary_planes = scan_binary_planes(src['X'], U, P, args.chunk)
    nb, nn = len(binary_planes), len(nonbinary_planes)
    PB = nb * 8 + nn * 64
    print(f"  -> {nb} binary, {nn} non-binary; packed_bytes/pos = {PB} "
          f"(raw {P*64}); X: {U*P*64/1e9:.1f}GB -> {U*PB/1e9:.1f}GB", flush=True)

    print("[2/4] bit-packing X ...", flush=True)
    pack_X(src['X'], U, P, binary_planes, nonbinary_planes,
           os.path.join(tmp, 'Xpacked.bin'), args.chunk)

    print("[3/4] building dense top-K policy targets ...", flush=True)
    n_trunc = build_policy_targets(src, U, args.topk, tmp, args.chunk)

    print("[4/4] copying value_avg / count / split ...", flush=True)
    copy_scalars(src, U, tmp)

    meta = {
        'n_packed': U, 'n_unique_src': src['U'], 'input_planes': P,
        'binary_planes': binary_planes, 'nonbinary_planes': nonbinary_planes,
        'packed_bytes': PB, 'n_binary': nb, 'n_nonbinary': nn,
        'topk': args.topk, 'n_truncated_gtK': int(n_trunc),
        'val_mod': src['meta'].get('val_mod'), 'source_agg': args.agg_dir,
    }
    with open(os.path.join(tmp, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print("verifying bit-exact round-trip ...", flush=True)
    ok = self_test(src, tmp, meta)
    if not ok:
        raise SystemExit("ROUND-TRIP FAILED -- not finalizing output")

    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.rename(tmp, args.out_dir)
    print(f"DONE -> {args.out_dir}", flush=True)


if __name__ == '__main__':
    main()

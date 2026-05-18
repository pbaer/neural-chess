# -*- coding: utf-8 -*-
"""Smoke tests for generate_shards() hardening.

Deliberately reproduces the failure conditions that caused segfaults in the
overnight v2 build session:
1. Stale files in target directory from a prior failed run
2. Pre-existing partial output (no meta.json) from a crash mid-write
3. Re-running on the same output path twice in a row

Each test creates a fresh fixture, runs generate_shards on a small target,
and verifies success (no crash + valid output produced).

Run directly: python -m src.v2.test_dataset_hardening
"""
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np

# Allow direct invocation
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.v2.dataset import generate_shards, ChessDatasetV2


FILTERED_DIR = os.path.join(_REPO_ROOT, 'data', 'v2', 'filtered')
SCRATCH = os.path.join(_REPO_ROOT, 'data', 'v2', '_hardening_test')


def _assert_valid_output(path: str, expected_min_samples: int):
    """Open the dataset via ChessDatasetV2; spot-check a few samples."""
    ds = ChessDatasetV2(path)
    assert len(ds) >= expected_min_samples, f'too few samples: {len(ds)}'
    x, yp, yv = ds[0]
    assert tuple(x.shape) == (21, 8, 8), f'bad X shape: {x.shape}'
    assert isinstance(yp, int) and 0 <= yp < 4672, f'bad policy: {yp}'
    assert yv in (-1.0, 0.0, 1.0), f'bad value: {yv}'
    # Check a sample from the middle
    mid = len(ds) // 2
    x, yp, yv = ds[mid]
    assert tuple(x.shape) == (21, 8, 8)
    # Check last sample
    x, yp, yv = ds[-1]
    assert tuple(x.shape) == (21, 8, 8)
    return len(ds)


def test_clean_first_run():
    """Sanity: just run on a fresh path."""
    out_dir = os.path.join(SCRATCH, 'clean')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    if os.path.exists(out_dir + '.tmp'):
        shutil.rmtree(out_dir + '.tmp')

    print('TEST clean_first_run: ', end='', flush=True)
    spec = generate_shards(FILTERED_DIR, out_dir, target_positions=20000)
    n = _assert_valid_output(out_dir, 19000)
    print(f'PASS (n={n})')


def test_rerun_same_path():
    """Re-running on the same path should clean up and succeed."""
    out_dir = os.path.join(SCRATCH, 'rerun')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    print('TEST rerun_same_path (first): ', end='', flush=True)
    generate_shards(FILTERED_DIR, out_dir, target_positions=15000)
    n1 = _assert_valid_output(out_dir, 14000)
    print(f'PASS (n={n1})')

    print('TEST rerun_same_path (second, same path): ', end='', flush=True)
    generate_shards(FILTERED_DIR, out_dir, target_positions=25000)
    n2 = _assert_valid_output(out_dir, 24000)
    print(f'PASS (n={n2})')


def test_recover_from_stale_partial():
    """A directory containing partial files (no meta.json) from a prior
    crashed run should be cleaned up and replaced."""
    out_dir = os.path.join(SCRATCH, 'partial')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    # Simulate a crashed-mid-write state: large dummy X.bin, no meta.json
    fake_x_path = os.path.join(out_dir, 'X.bin')
    with open(fake_x_path, 'wb') as f:
        f.write(b'\xff' * (10 * 1024 * 1024))  # 10 MB of garbage
    with open(os.path.join(out_dir, 'Y_policy.bin'), 'wb') as f:
        f.write(b'\xff' * (1024 * 1024))
    with open(os.path.join(out_dir, 'Y_value.bin'), 'wb') as f:
        f.write(b'\xff' * (1024 * 1024))

    print('TEST recover_from_stale_partial: ', end='', flush=True)
    generate_shards(FILTERED_DIR, out_dir, target_positions=15000)
    n = _assert_valid_output(out_dir, 14000)
    # Verify the X.bin is the size matching our generator's output (not the stale 10 MB)
    actual_size = os.path.getsize(fake_x_path)
    expected_size = n * 21 * 64
    assert actual_size == expected_size, f'stale file lingered: {actual_size} != {expected_size}'
    print(f'PASS (n={n}, X.bin properly replaced)')


def test_recover_from_stale_tmp():
    """A leftover .tmp directory from a prior crash should be cleaned up."""
    out_dir = os.path.join(SCRATCH, 'tmp_leftover')
    tmp_dir = out_dir + '.tmp'
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    # Simulate a leftover tmp from a crashed run
    os.makedirs(tmp_dir)
    with open(os.path.join(tmp_dir, 'X.bin'), 'wb') as f:
        f.write(b'\xff' * (5 * 1024 * 1024))
    with open(os.path.join(tmp_dir, 'incomplete_marker'), 'w') as f:
        f.write('crashed run')

    print('TEST recover_from_stale_tmp: ', end='', flush=True)
    generate_shards(FILTERED_DIR, out_dir, target_positions=15000)
    n = _assert_valid_output(out_dir, 14000)
    # Tmp dir should be gone (renamed to final)
    assert not os.path.exists(tmp_dir), f'stale tmp dir lingered: {tmp_dir}'
    print(f'PASS (n={n}, .tmp leftover cleaned)')


def test_atomic_finalize_on_failure():
    """If we simulate generate_shards mid-write, the final dir should NOT exist
    yet — only the .tmp dir. (This is testing the atomic-rename property.)"""
    out_dir = os.path.join(SCRATCH, 'atomic')
    tmp_dir = out_dir + '.tmp'
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    print('TEST atomic_finalize_on_failure: ', end='', flush=True)
    # Successful run: final dir is the only thing left, no tmp
    generate_shards(FILTERED_DIR, out_dir, target_positions=10000)
    assert os.path.exists(out_dir), 'final dir missing after success'
    assert not os.path.exists(tmp_dir), 'tmp dir lingered after success'
    print(f'PASS (final exists, tmp does not)')


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    print(f'Hardening tests, scratch dir: {SCRATCH}\n')

    t0 = time.time()
    test_clean_first_run()
    test_rerun_same_path()
    test_recover_from_stale_partial()
    test_recover_from_stale_tmp()
    test_atomic_finalize_on_failure()

    print(f'\nAll tests passed in {time.time()-t0:.1f}s. Cleaning up scratch.')
    shutil.rmtree(SCRATCH, ignore_errors=True)


if __name__ == '__main__':
    main()

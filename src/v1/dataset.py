# -*- coding: utf-8 -*-
"""v1 ChessDataset: loads the 6-plane sign-encoded NPZ corpus into RAM."""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.v1.featurize import expand_planes


class ChessDataset(Dataset):
    """Loads all NPZ files from a folder into memory.

    X is stored as int8 (B, 6, 8, 8) in sign-encoded form (matches the
    on-disk NPZ layout) and expanded to (12, 8, 8) float32 own/opponent
    binary planes lazily in __getitem__. Storing int8 keeps the in-RAM
    footprint to ~7.95 GiB across all 22M samples (vs ~31.8 GiB if we
    cast to float32 6-plane up front, or ~63.6 GiB if we expanded to
    12-plane float32 up front).

    Y is stored as int64 class indices (B,) via argmax of the one-hot labels.
    """

    def __init__(self, folder='data/v1'):
        # Two-pass loading to avoid doubling peak RAM.
        # Pass 1: count total rows from metadata.
        files = []
        total = 0
        for filename in sorted(os.listdir(folder)):
            if not filename.endswith('.npz'):
                continue
            path = folder + '/' + filename
            meta = np.load(path, mmap_mode='r')['meta']
            n = int(meta[0])
            files.append((path, filename, n))
            total += n
            print(f"{n:>9,} rows in {filename}")

        # Pass 2: pre-allocate as int8 and fill in-place.
        self.X = np.empty((total, 6, 8, 8), dtype=np.int8)
        self.Y = np.empty((total,), dtype=np.int64)
        offset = 0
        for path, filename, n in files:
            data = np.load(path)
            self.X[offset:offset + n] = data['X'][:n].reshape(-1, 6, 8, 8)
            self.Y[offset:offset + n] = data['Y'][:n].argmax(axis=1)
            offset += n
            del data

        print(f"{len(self):>9,} total samples loaded "
              f"({self.X.nbytes / 1024**3:.2f} GB features (int8 6-plane, expanded to "
              f"12-plane float32 per batch) + "
              f"{self.Y.nbytes / 1024**3:.2f} GB labels)")

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return torch.from_numpy(expand_planes(self.X[idx])), self.Y[idx]

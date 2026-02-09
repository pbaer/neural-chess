# -*- coding: utf-8 -*-
import glob
import numpy as np
import os
import platform
import re
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import ChessModel, save_model


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class ChessDataset(Dataset):
    """Loads all NPZ files from a folder into memory.

    X is stored as float32 (B, 6, 8, 8).
    Y is stored as int64 class indices (B,) via argmax of the one-hot labels.
    """

    def __init__(self, folder='data'):
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

        # Pass 2: pre-allocate and fill in-place (peak ≈ 32 GB).
        self.X = np.empty((total, 6, 8, 8), dtype=np.float32)
        self.Y = np.empty((total,), dtype=np.int64)
        offset = 0
        for path, filename, n in files:
            data = np.load(path)
            self.X[offset:offset + n] = data['X'][:n].reshape(-1, 6, 8, 8).astype(np.float32)
            self.Y[offset:offset + n] = data['Y'][:n].argmax(axis=1)
            offset += n
            del data

        print(f"{len(self):>9,} total samples loaded "
              f"({self.X.nbytes / 1024**3:.2f} GB features + "
              f"{self.Y.nbytes / 1024**3:.2f} GB labels)")

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), self.Y[idx]


# ---------------------------------------------------------------------------
# Checkpoint auto-resume
# ---------------------------------------------------------------------------

def _find_latest_checkpoint(save_name='model', model_dir='model'):
    """Find the latest checkpoint matching {save_name}_e{epoch}.pt in model_dir.

    Returns (path, epoch) or (None, 0) if no checkpoint exists.
    """
    pattern = os.path.join(model_dir, f'{save_name}_e*.pt')
    matches = glob.glob(pattern)
    if not matches:
        return None, 0

    epoch_re = re.compile(re.escape(save_name) + r'_e(\d+)\.pt$')
    best_path, best_epoch = None, -1
    for path in matches:
        m = epoch_re.search(os.path.basename(path))
        if m:
            ep = int(m.group(1))
            if ep > best_epoch:
                best_epoch = ep
                best_path = path

    if best_path is None:
        return None, 0
    return best_path, best_epoch


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(data_folder='data', batch_size=4096, lr=1e-3, weight_decay=1e-4,
          save_name='model', start_epoch=0, resume_pt=None, no_resume=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Dataset / DataLoader
    dataset = ChessDataset(data_folder)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        pin_memory=(device.type == 'cuda'), num_workers=0)

    # Model
    model = ChessModel().to(device)

    # Auto-resume: find latest checkpoint unless --no-resume or explicit start_epoch
    if resume_pt is None and not no_resume and start_epoch == 0:
        resume_pt, detected_epoch = _find_latest_checkpoint(save_name)
        if resume_pt:
            start_epoch = detected_epoch + 1  # continue from next epoch

    if resume_pt:
        model.load_state_dict(torch.load(resume_pt, map_location=device, weights_only=True))
        print(f"Resumed from {resume_pt} (next epoch: {start_epoch})")

    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")

    # torch.compile requires Triton, which is not available on Windows
    if platform.system() != 'Windows':
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception:
            print("torch.compile not available, using eager mode")
    else:
        print("Skipping torch.compile (not supported on Windows)")

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.CrossEntropyLoss()

    # Mixed precision
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    os.makedirs('model', exist_ok=True)
    epoch = start_epoch

    print(f"\nTraining with batch_size={batch_size}, lr={lr}, AMP={use_amp}")
    print(f"Batches per epoch: {len(loader)}")
    print("Create a '.stop' file to stop after the current epoch.\n")

    while not os.path.isfile('.stop'):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0
        t0 = time.time()

        num_batches = len(loader)
        for batch_idx, (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(batch_x)
                loss = criterion(logits, batch_y)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * batch_x.size(0)
            epoch_correct += (logits.argmax(1) == batch_y).sum().item()
            epoch_samples += batch_x.size(0)

            if (batch_idx + 1) % 100 == 0 or batch_idx == num_batches - 1:
                pct = 100 * (batch_idx + 1) / num_batches
                avg = epoch_loss / epoch_samples
                acc = epoch_correct / epoch_samples
                print(f"\r  Epoch {epoch} [{batch_idx+1}/{num_batches}] "
                      f"{pct:5.1f}% | loss {avg:.4f} | acc {acc:.4f}", end='', flush=True)
        print()  # newline after progress

        scheduler.step()

        elapsed = time.time() - t0
        avg_loss = epoch_loss / epoch_samples
        accuracy = epoch_correct / epoch_samples

        tag = f"{save_name}_e{epoch:04d}"
        save_model(model, tag)
        print(f"Epoch {epoch:4d} | loss {avg_loss:.4f} | acc {accuracy:.4f} | "
              f"{elapsed:.1f}s | lr {scheduler.get_last_lr()[0]:.2e} | saved {tag}.pt")

        epoch += 1

    if os.path.isfile('.stop'):
        os.remove('.stop')
    print("Training stopped.")


# ---------------------------------------------------------------------------
# Legacy TrainingSet class — used by parse.py (pure numpy, no torch dependency)
# ---------------------------------------------------------------------------

class TrainingSet():
    FEATURES = 6 * 8 * 8
    OUTPUTS = 64 * 64

    def __init__(self, max_rows):
        self.X = np.zeros((max_rows, self.FEATURES), dtype='int8')
        self.Y = np.zeros((max_rows, self.OUTPUTS), dtype='int8')
        self.rows = 0
        self.max_rows = max_rows

    def reset(self):
        self.rows = 0

    def get(self):
        return self.X[0:self.rows, :], self.Y[0:self.rows, :]

    def is_full(self):
        return self.rows == self.max_rows

    def add_from_file(self, filename):
        data = np.load(filename)
        return self.add_from_data(data)

    def add_from_data(self, data):
        data_rows = data['meta'][0]
        if (self.rows + data_rows > self.max_rows):
            return False
        data_X = data['X']
        data_Y = data['Y']
        self.X[self.rows:(self.rows + data_rows), :] = data_X[0:data_rows, :]
        self.Y[self.rows:(self.rows + data_rows), :] = data_Y[0:data_rows, :]
        self.rows += data_rows
        return True

    def add_from_folder(self, foldername, printonly=False):
        total_rows = 0
        for filename in os.listdir(foldername):
            if not filename.endswith('.npz'):
                continue
            data = np.load(foldername + '/' + filename)
            data_rows = data['meta'][0]
            print("%d rows in %s" % (data_rows, filename))
            total_rows += data_rows
            if printonly:
                continue
            if not self.add_from_data(data):
                total_rows -= data_rows
                print("Training set full, not adding this file.")
                break
        print("%d total rows (%.2fGB expanded)" % (total_rows, (float(total_rows) * (self.FEATURES + self.OUTPUTS)) / (1024 * 1024 * 1024)))

    def add_row(self, x, y):
        if self.is_full():
            return False
        self.X[self.rows] = x
        self.Y[self.rows] = y
        self.rows += 1
        return True

    def save_to_file(self, filename):
        meta = np.ndarray((1), dtype=int)
        meta[0] = self.rows
        np.savez_compressed('data/' + filename, X=self.X[0:self.rows, :], Y=self.Y[0:self.rows, :], meta=meta)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train chess model')
    parser.add_argument('--data', default='data', help='data folder (default: data)')
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--save-name', default='model', help='checkpoint name prefix')
    parser.add_argument('--start-epoch', type=int, default=0, help='override start epoch')
    parser.add_argument('--resume', default=None, help='explicit checkpoint path to resume from')
    parser.add_argument('--no-resume', action='store_true', help='start fresh, ignore existing checkpoints')
    args = parser.parse_args()

    train(
        data_folder=args.data,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        save_name=args.save_name,
        start_epoch=args.start_epoch,
        resume_pt=args.resume,
        no_resume=args.no_resume,
    )

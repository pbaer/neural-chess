# -*- coding: utf-8 -*-
"""v1 training loop. Writes checkpoints to model/v1/<save_name>_e<NNNN>.pt."""
import glob
import math
import os
import platform
import re
import sys
import time

# Allow `python src/v1/train.py` from the repo root: ensure the repo root
# is on sys.path so `from src.v1...` imports resolve.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.v1.dataset import ChessDataset  # noqa: E402
from src.v1.model import ChessModel  # noqa: E402


DEFAULT_DATA_DIR = 'data/v1'
DEFAULT_MODEL_DIR = 'model/v1'


# ---------------------------------------------------------------------------
# Checkpoint auto-resume
# ---------------------------------------------------------------------------

def _find_latest_checkpoint(save_name='model', model_dir=DEFAULT_MODEL_DIR):
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
# LR schedule
# ---------------------------------------------------------------------------

def _make_cosine_scheduler(optimizer, t_max):
    """Cosine decay over `t_max` epochs that clamps at the floor afterwards.

    The built-in CosineAnnealingLR keeps evaluating cos() past T_max, so the LR
    rebounds back toward base — surprising for indefinite training. This wraps
    a LambdaLR around min(epoch, t_max) so the LR rests at 0 after t_max.
    """
    def lr_lambda(epoch):
        e = min(epoch, t_max)
        return 0.5 * (1.0 + math.cos(math.pi * e / t_max))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def _unwrap_compiled(model):
    """torch.compile wraps the module; persist the underlying state_dict."""
    return getattr(model, '_orig_mod', model)


def save_checkpoint(tag, model, optimizer, scheduler, epoch, model_dir=DEFAULT_MODEL_DIR):
    path = os.path.join(model_dir, tag + '.pt')
    torch.save({
        'model': _unwrap_compiled(model).state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }, path)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(data_folder=DEFAULT_DATA_DIR, model_dir=DEFAULT_MODEL_DIR,
          batch_size=4096, lr=1e-3, weight_decay=1e-4,
          epochs=50, save_name='model', start_epoch=0, resume_pt=None,
          no_resume=False):
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
        resume_pt, detected_epoch = _find_latest_checkpoint(save_name, model_dir=model_dir)
        if resume_pt:
            start_epoch = detected_epoch + 1  # continue from next epoch

    resume_data = None
    if resume_pt:
        resume_data = torch.load(resume_pt, map_location=device, weights_only=True)
        if isinstance(resume_data, dict) and 'model' in resume_data:
            model.load_state_dict(resume_data['model'])
        else:
            # Legacy checkpoint: weights only. Optimizer/scheduler reset.
            model.load_state_dict(resume_data)
            resume_data = None
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
    scheduler = _make_cosine_scheduler(optimizer, t_max=epochs)
    criterion = nn.CrossEntropyLoss()

    if resume_data is not None:
        try:
            optimizer.load_state_dict(resume_data['optimizer'])
            scheduler.load_state_dict(resume_data['scheduler'])
            print("Restored optimizer + scheduler state")
        except (KeyError, ValueError) as e:
            print(f"Could not restore optimizer/scheduler ({e}); continuing with fresh state")

    # Mixed precision
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    os.makedirs(model_dir, exist_ok=True)
    epoch = start_epoch

    print(f"\nTraining with batch_size={batch_size}, lr={lr}, epochs={epochs}, AMP={use_amp}")
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
        save_checkpoint(tag, model, optimizer, scheduler, epoch, model_dir=model_dir)
        print(f"Epoch {epoch:4d} | loss {avg_loss:.4f} | acc {accuracy:.4f} | "
              f"{elapsed:.1f}s | lr {scheduler.get_last_lr()[0]:.2e} | saved {tag}.pt")

        epoch += 1

    if os.path.isfile('.stop'):
        os.remove('.stop')
    print("Training stopped.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train v1 chess model')
    parser.add_argument('--data', default=DEFAULT_DATA_DIR,
                        help=f'data folder (default: {DEFAULT_DATA_DIR})')
    parser.add_argument('--model-dir', default=DEFAULT_MODEL_DIR,
                        help=f'checkpoint output folder (default: {DEFAULT_MODEL_DIR})')
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50,
                        help='cosine LR schedule horizon — LR floors at 0 after this many epochs (default: 50)')
    parser.add_argument('--save-name', default='model', help='checkpoint name prefix')
    parser.add_argument('--start-epoch', type=int, default=0, help='override start epoch')
    parser.add_argument('--resume', default=None, help='explicit checkpoint path to resume from')
    parser.add_argument('--no-resume', action='store_true', help='start fresh, ignore existing checkpoints')
    args = parser.parse_args()

    train(
        data_folder=args.data,
        model_dir=args.model_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        save_name=args.save_name,
        start_epoch=args.start_epoch,
        resume_pt=args.resume,
        no_resume=args.no_resume,
    )

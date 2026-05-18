# -*- coding: utf-8 -*-
"""v2 training loop. Multi-task: policy cross-entropy + value MSE.

Per memory/project-principles.md:
  - Policy target: the move that was played (from the game)
  - Value target: the moving player's eventual outcome (from the game)
  - No other supervision signals (no computed chess features)

Config-driven via ChessConfigV2. Supports auto-resume, mixed precision,
custom cosine LR schedule. Memmap-backed dataset means RAM doesn't bound
the training-set size.
"""
import argparse
import glob
import math
import os
import platform
import re
import sys
import time
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Allow direct invocation (python src/v2/train.py)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.v2.dataset import ChessDatasetV2
from src.v2.inference import save_v2_checkpoint
from src.v2.model import ChessConfigV2, ChessModelV2, count_params


# ---- LR schedule (same clamped-cosine as v1) ----

def _make_cosine_scheduler(optimizer, t_max):
    def lr_lambda(epoch):
        e = min(epoch, t_max)
        return 0.5 * (1.0 + math.cos(math.pi * e / t_max))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---- Checkpoint discovery ----

def _find_latest_checkpoint(save_dir: str, save_name: str = 'model'):
    pattern = os.path.join(save_dir, f'{save_name}_e*.pt')
    matches = glob.glob(pattern)
    if not matches:
        return None, 0
    epoch_re = re.compile(re.escape(save_name) + r'_e(\d+)\.pt$')
    best_path, best_epoch = None, -1
    for p in matches:
        m = epoch_re.search(os.path.basename(p))
        if m:
            ep = int(m.group(1))
            if ep > best_epoch:
                best_epoch = ep
                best_path = p
    return (best_path, best_epoch) if best_path else (None, 0)


# ---- Training loop ----

def train(
    shard_dir: str,
    save_dir: str,
    config: ChessConfigV2 = None,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 50,
    max_epochs: int = None,
    save_name: str = 'model',
    start_epoch: int = 0,
    resume_pt: str = None,
    no_resume: bool = False,
    value_loss_weight: float = 1.0,
    stop_file: str = '.stop',
    log_every: int = 100,
    keep_last_n: int = 0,
):
    """`epochs` is the cosine LR-schedule horizon. `max_epochs` is the
    actual stopping condition (default: same as epochs). Training also exits
    if the `stop_file` appears.

    `keep_last_n=0` (default) keeps every checkpoint. `keep_last_n>0` keeps
    only the most recent N (the rolling latest), to bound disk usage during
    long runs.
    """
    if max_epochs is None:
        max_epochs = epochs
    """Train v2 model on memmap shards.

    Args:
        shard_dir: directory with X.bin / Y_policy.bin / Y_value.bin / meta.json
        save_dir: directory to write checkpoints into (created if missing)
        config: ChessConfigV2; defaults to T0a (6 blocks x 128 ch, no lookahead)
        value_loss_weight: how heavily the value head's MSE counts relative to
            the policy cross-entropy. 1.0 is balanced; lower if value is unstable.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")

    if config is None:
        config = ChessConfigV2()

    # Dataset
    dataset = ChessDatasetV2(shard_dir)
    print(f"Dataset: {len(dataset):,} samples from {shard_dir}")

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        pin_memory=(device.type == 'cuda'), num_workers=0,
    )

    # Model
    model = ChessModelV2(config).to(device)
    print(f"Model params: {count_params(model):,} (config: blocks={config.encoder_blocks}, "
          f"channels={config.encoder_channels}, lookahead_K={config.lookahead_K})")

    # Auto-resume
    if resume_pt is None and not no_resume and start_epoch == 0:
        resume_pt, detected_epoch = _find_latest_checkpoint(save_dir, save_name)
        if resume_pt:
            start_epoch = detected_epoch + 1

    resume_data = None
    if resume_pt:
        resume_data = torch.load(resume_pt, map_location=device, weights_only=False)
        if isinstance(resume_data, dict) and 'model' in resume_data:
            model.load_state_dict(resume_data['model'])
        else:
            model.load_state_dict(resume_data)
            resume_data = None
        print(f"Resumed from {resume_pt} (next epoch: {start_epoch})")

    # torch.compile not available on Windows
    if platform.system() != 'Windows':
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception:
            print("torch.compile not available, eager mode")
    else:
        print("Skipping torch.compile (not supported on Windows)")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = _make_cosine_scheduler(optimizer, t_max=epochs)
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    if resume_data is not None:
        try:
            optimizer.load_state_dict(resume_data['optimizer'])
            scheduler.load_state_dict(resume_data['scheduler'])
            print("Restored optimizer + scheduler state")
        except (KeyError, ValueError) as e:
            print(f"Could not restore optim/sched ({e}); fresh state")

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    os.makedirs(save_dir, exist_ok=True)
    epoch = start_epoch

    print(f"\nTraining: batch={batch_size}, lr={lr}, epochs={epochs} (LR), "
          f"max_epochs={max_epochs}, value_w={value_loss_weight}, AMP={use_amp}")
    print(f"Batches per epoch: {len(loader)}")
    print(f"Create '{stop_file}' to stop after the current epoch (or hit max_epochs).\n")

    while not os.path.isfile(stop_file) and epoch < max_epochs:
        model.train()
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_policy_correct = 0
        epoch_value_se = 0.0
        epoch_samples = 0
        t0 = time.time()
        num_batches = len(loader)

        for batch_idx, (batch_x, batch_yp, batch_yv) in enumerate(loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_yp = batch_yp.to(device, non_blocking=True)
            batch_yv = batch_yv.to(device, non_blocking=True).float()
            bs = batch_x.size(0)

            with torch.amp.autocast('cuda', enabled=use_amp):
                policy_logits, value_pred = model(batch_x)
                value_pred = value_pred.squeeze(-1)
                policy_loss = policy_loss_fn(policy_logits, batch_yp)
                value_loss = value_loss_fn(value_pred, batch_yv)
                loss = policy_loss + value_loss_weight * value_loss

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_policy_loss += policy_loss.item() * bs
            epoch_value_loss += value_loss.item() * bs
            epoch_policy_correct += (policy_logits.argmax(1) == batch_yp).sum().item()
            epoch_value_se += ((value_pred - batch_yv) ** 2).sum().item()
            epoch_samples += bs

            if (batch_idx + 1) % log_every == 0 or batch_idx == num_batches - 1:
                pct = 100 * (batch_idx + 1) / num_batches
                avg_p = epoch_policy_loss / epoch_samples
                avg_v = epoch_value_loss / epoch_samples
                acc = epoch_policy_correct / epoch_samples
                print(f"\r  Epoch {epoch} [{batch_idx+1}/{num_batches}] "
                      f"{pct:5.1f}% | pol {avg_p:.4f} val {avg_v:.4f} | "
                      f"acc {acc:.4f}", end='', flush=True)
        print()  # newline after progress

        scheduler.step()

        elapsed = time.time() - t0
        avg_policy = epoch_policy_loss / epoch_samples
        avg_value = epoch_value_loss / epoch_samples
        accuracy = epoch_policy_correct / epoch_samples
        value_rmse = math.sqrt(epoch_value_se / epoch_samples)

        tag = f"{save_name}_e{epoch:04d}"
        ckpt_path = os.path.join(save_dir, tag + '.pt')
        save_v2_checkpoint(ckpt_path, model, optimizer=optimizer,
                           scheduler=scheduler, epoch=epoch, config=config)
        print(f"Epoch {epoch:4d} | pol {avg_policy:.4f} val {avg_value:.4f} | "
              f"acc {accuracy:.4f} val_rmse {value_rmse:.3f} | "
              f"{elapsed:.1f}s | lr {scheduler.get_last_lr()[0]:.2e} | saved {tag}.pt")

        # Optional rolling-checkpoint policy: keep only the most recent N.
        if keep_last_n > 0:
            existing = sorted(glob.glob(os.path.join(save_dir, f'{save_name}_e*.pt')))
            for old_path in existing[:-keep_last_n]:
                try:
                    os.remove(old_path)
                except OSError:
                    pass

        epoch += 1

    if os.path.isfile(stop_file):
        os.remove(stop_file)
    print("Training stopped.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train v2 chess model')
    parser.add_argument('--shard-dir', required=True,
                        help='directory with X.bin / Y_policy.bin / Y_value.bin / meta.json')
    parser.add_argument('--save-dir', required=True,
                        help='directory to write checkpoints to')
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=15, help='cosine LR horizon')
    parser.add_argument('--max-epochs', type=int, default=None,
                        help='actually stop after this many epochs (default: same as --epochs)')
    parser.add_argument('--keep-last-n', type=int, default=0,
                        help='rolling-checkpoint: keep only the N most recent .pt files (0 = keep all)')
    parser.add_argument('--value-loss-weight', type=float, default=1.0)
    parser.add_argument('--save-name', default='model')
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--stop-file', default='.stop')
    # Config knobs
    parser.add_argument('--blocks', type=int, default=6, help='encoder ResNet blocks')
    parser.add_argument('--channels', type=int, default=128, help='encoder channel count')
    parser.add_argument('--policy-channels', type=int, default=32)
    parser.add_argument('--value-channels', type=int, default=1,
                        help='value head 1x1-conv compression target (per-square channels). '
                             'Default 1 produces a 64-number flatten; 4 produces 256-number flatten.')
    parser.add_argument('--value-hidden', type=int, default=64)
    parser.add_argument('--lookahead-k', type=int, default=0,
                        help='lookahead branching factor (0 = no lookahead block)')
    parser.add_argument('--lookahead-depth', type=int, default=0,
                        help='lookahead rollout depth (0 = no lookahead)')
    parser.add_argument('--aggregator-heads', type=int, default=4)
    parser.add_argument('--aggregator-layers', type=int, default=2)
    args = parser.parse_args()

    config = ChessConfigV2(
        encoder_blocks=args.blocks,
        encoder_channels=args.channels,
        policy_channels=args.policy_channels,
        value_channels=args.value_channels,
        value_hidden=args.value_hidden,
        lookahead_K=args.lookahead_k,
        lookahead_depth=args.lookahead_depth,
        aggregator_heads=args.aggregator_heads,
        aggregator_layers=args.aggregator_layers,
    )

    train(
        shard_dir=args.shard_dir,
        save_dir=args.save_dir,
        config=config,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        max_epochs=args.max_epochs,
        keep_last_n=args.keep_last_n,
        save_name=args.save_name,
        start_epoch=args.start_epoch,
        resume_pt=args.resume,
        no_resume=args.no_resume,
        value_loss_weight=args.value_loss_weight,
        stop_file=args.stop_file,
    )

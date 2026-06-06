# -*- coding: utf-8 -*-
"""Train the (fixed) v3 attention arch on a position-aggregated artifact, with
knobs to A/B the data-treatment ideas from [[position-aggregated-dataset]]:

  --value-mode  avg | one      averaged Q  vs  one random game's outcome (control)
  --policy-mode soft | hard_argmax | hard_sample
                               soft = KL to the human move histogram;
                               hard_sample = CE to a move drawn ~ histogram each
                                 step (== current single-move CE in expectation);
                               hard_argmax = CE to the single most-common move.
  --tau         FLOAT          sample positions ~ count**tau. 1.0 = today's
                               frequency weighting; 0.0 = uniform over unique
                               positions; ~0.3-0.5 = down-weight openings.
  --epoch-size  INT            samples drawn per epoch (fixes COMPUTE across
                               treatments regardless of tau / unique count).

Architecture is held FIXED (the small v3, d_model=256 x 20 blocks by default) so
the only thing under test is the data. Reports held-out metrics every epoch:
value MSE (vs noisy single outcome AND vs de-noised avg-Q), policy top-1/top-3
accuracy (vs most-common human move), and policy cross-entropy vs the histogram.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.v3.model import ChessConfigV3, ChessModelV3, count_params
from src.v3.inference import save_v3_checkpoint

PAD_K = 48  # max distinct moves kept per position (top-K by count, renormalized)


class ChessDatasetAgg(Dataset):
    """Serves (x, value, pol_moves[K], pol_probs[K]) from an aggregated artifact.

    pol_probs is the human move distribution (normalized over the kept top-K
    moves), zero-padded to K. The training loss derives soft / hard targets from
    it, so a single artifact + one Dataset expresses every policy treatment.
    value comes from value_avg or value_one per `value_mode`.
    """

    def __init__(self, agg_dir, split='train', value_mode='avg'):
        with open(os.path.join(agg_dir, 'meta.json')) as f:
            self.meta = json.load(f)
        self.dir = agg_dir
        self.U = self.meta['n_unique']
        self.planes = self.meta['input_planes']
        self.value_mode = value_mode
        self.split = split
        self.H = self.meta['n_hist_entries']
        self._open()
        self._compute_indices()

    def _compute_indices(self):
        sp = np.fromfile(os.path.join(self.dir, 'split.bin'), dtype=np.int8)
        want = 0 if self.split == 'train' else 1
        self.indices = np.flatnonzero(sp == want).astype(np.int64)

    def _open(self):
        # All arrays memmap'd read-only so the OS page cache is SHARED across
        # DataLoader workers (one physical copy) instead of np.fromfile loading a
        # private ~1.8GB copy per worker (16 workers x 1.8GB would exhaust RAM).
        d, U, P, H = self.dir, self.U, self.planes, self.H
        mm = lambda n, dt, sh: np.memmap(os.path.join(d, n), dtype=dt, mode='r', shape=sh)
        self.X = mm('X.bin', np.int8, (U, P, 8, 8))
        self.value_avg = mm('value_avg.bin', np.float32, (U,))
        self.value_one = mm('value_one.bin', np.int8, (U,))
        self.count = mm('count.bin', np.int32, (U,))
        self.hist_ptr = mm('hist_ptr.bin', np.int64, (U + 1,))
        self.hist_moves = mm('hist_moves.bin', np.int32, (H,))
        self.hist_counts = mm('hist_counts.bin', np.int32, (H,))

    # memmaps + the large index array can't/shouldn't be pickled across Windows
    # workers (a 70M-int64 indices array = 564MB overflows the spawn pipe ->
    # "pickle data was truncated"/Errno22). Strip them; rebuild in each worker.
    def __getstate__(self):
        s = self.__dict__.copy()
        for k in ('X', 'value_avg', 'value_one', 'count', 'hist_ptr',
                  'hist_moves', 'hist_counts', 'indices'):
            s.pop(k, None)
        return s

    def __setstate__(self, s):
        self.__dict__.update(s)
        self._open()
        self._compute_indices()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, j):
        i = int(self.indices[j])
        x = torch.from_numpy(np.ascontiguousarray(self.X[i]))  # int8; cast on GPU
        val = (float(self.value_avg[i]) if self.value_mode == 'avg'
               else float(self.value_one[i]))
        a, b = int(self.hist_ptr[i]), int(self.hist_ptr[i + 1])
        moves = self.hist_moves[a:b]
        counts = self.hist_counts[a:b].astype(np.float32)
        if len(moves) > PAD_K:
            keep = np.argsort(counts)[::-1][:PAD_K]
            moves, counts = moves[keep], counts[keep]
        probs = counts / counts.sum()
        pm = np.zeros(PAD_K, dtype=np.int64)
        pp = np.zeros(PAD_K, dtype=np.float32)
        pm[:len(moves)] = moves
        pp[:len(probs)] = probs
        return x, val, torch.from_numpy(pm), torch.from_numpy(pp)


def policy_loss(logits, pm, pp, mode, gen):
    """logits (B,4672); pm (B,K) move idx; pp (B,K) probs (0-padded).
    Returns scalar policy loss for the requested mode."""
    logp = F.log_softmax(logits, dim=1)
    gathered = logp.gather(1, pm)                      # (B,K) log-prob of each move
    if mode == 'soft':
        # cross-entropy to the histogram distribution: -sum_k p_k log q_k
        return -(pp * gathered).sum(dim=1).mean()
    if mode == 'hard_argmax':
        tgt = pm.gather(1, pp.argmax(dim=1, keepdim=True)).squeeze(1)
        return F.nll_loss(logp, tgt)
    if mode == 'hard_sample':
        # draw one move per row ~ histogram (== single-move CE in expectation)
        col = torch.multinomial(pp + 1e-12, 1, generator=gen)   # (B,1)
        tgt = pm.gather(1, col).squeeze(1)
        return F.nll_loss(logp, tgt)
    raise ValueError(mode)


@torch.no_grad()
def evaluate(model, val_loader, device, count_val):
    model.eval()
    n = 0
    se_one = se_avg = 0.0
    top1 = top3 = 0
    polce = 0.0
    se_avg_hi = 0.0; n_hi = 0
    # count-weighted (deployment-distribution) accumulators
    wsum = 0.0; w_se_avg = 0.0; w_top1 = 0.0; w_top3 = 0.0; w_polce = 0.0
    for x, v_avg, v_one, pm, pp, cnt in val_loader:
        x = x.to(device, non_blocking=True).float()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, value = model(x)
        value = value.squeeze(-1).float().cpu()
        logits = logits.float().cpu()
        bs = x.size(0)
        cw = cnt.float()
        se_one += ((value - v_one) ** 2).sum().item()
        se_avg += ((value - v_avg) ** 2).sum().item()
        hi = cnt >= 5
        if hi.any():
            se_avg_hi += ((value[hi] - v_avg[hi]) ** 2).sum().item(); n_hi += int(hi.sum())
        tgt = pm.gather(1, pp.argmax(dim=1, keepdim=True)).squeeze(1)
        topk = logits.topk(3, dim=1).indices
        c1 = (topk[:, 0] == tgt).float()
        c3 = (topk == tgt.unsqueeze(1)).any(dim=1).float()
        top1 += c1.sum().item()
        top3 += c3.sum().item()
        logp = F.log_softmax(logits, dim=1)
        ce = -(pp * logp.gather(1, pm)).sum(dim=1)
        polce += ce.sum().item()
        # count-weighted
        wsum += cw.sum().item()
        w_se_avg += (cw * (value - v_avg) ** 2).sum().item()
        w_top1 += (cw * c1).sum().item()
        w_top3 += (cw * c3).sum().item()
        w_polce += (cw * ce).sum().item()
        n += bs
    return {
        'val_rmse_one': math.sqrt(se_one / n),
        'val_rmse_avg': math.sqrt(se_avg / n),
        'val_rmse_avg_hi': math.sqrt(se_avg_hi / n_hi) if n_hi else float('nan'),
        'top1': top1 / n,
        'top3': top3 / n,
        'pol_ce': polce / n,
        # count-weighted = positions weighted by how often they actually occur
        'w_rmse_avg': math.sqrt(w_se_avg / wsum),
        'w_top1': w_top1 / wsum,
        'w_top3': w_top3 / wsum,
        'w_pol_ce': w_polce / wsum,
        'n_val': n, 'n_hi': n_hi,
    }


class ValView(Dataset):
    """Val dataset that also returns value_avg, value_one, count for metrics."""
    def __init__(self, agg):
        self.agg = agg
    def __len__(self):
        return len(self.agg.indices)
    def __getitem__(self, j):
        x, _, pm, pp = self.agg[j]
        i = int(self.agg.indices[j])
        return (x, float(self.agg.value_avg[i]), float(self.agg.value_one[i]),
                pm, pp, int(self.agg.count[i]))


class TemperedSampler(Sampler):
    """Sample positions with replacement ~ weights, with NO category-count
    limit. torch's WeightedRandomSampler uses torch.multinomial, which caps at
    2^24 (~16.7M) categories — fine for the 8M probe (6.25M train) but the full
    corpus has ~70M train positions, so we draw via numpy instead (cumsum +
    searchsorted, ~1-2s per epoch for 10M draws over 70M)."""

    def __init__(self, weights_np, num_samples, seed=0):
        p = weights_np.astype(np.float64)
        self.p = p / p.sum()
        self.num_samples = int(num_samples)
        self.epoch = 0
        self.seed = seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        idx = rng.choice(len(self.p), size=self.num_samples, replace=True, p=self.p)
        return iter(idx)

    def __len__(self):
        return self.num_samples


def make_cosine(opt, t_max):
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: 0.5 * (1 + math.cos(math.pi * min(e, t_max) / t_max)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--agg-dir', required=True)
    ap.add_argument('--save-dir', required=True)
    ap.add_argument('--save-name', default='model')
    ap.add_argument('--value-mode', choices=['avg', 'one'], default='avg')
    ap.add_argument('--policy-mode',
                    choices=['soft', 'hard_argmax', 'hard_sample'], default='soft')
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--epoch-size', type=int, default=3_000_000)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--lr-horizon', type=int, default=10,
                    help='cosine LR horizon (keep fixed across probes)')
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--value-loss-weight', type=float, default=1.0)
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--metrics-csv', default=None)
    ap.add_argument('--val-cap', type=int, default=300000,
                    help='cap held-out eval to a strided subset (0 = full val)')
    ap.add_argument('--save-every-steps', type=int, default=0)
    # fixed arch (small v3)
    ap.add_argument('--d-model', type=int, default=256)
    ap.add_argument('--n-heads', type=int, default=8)
    ap.add_argument('--n-blocks', type=int, default=20)
    ap.add_argument('--checkpoint-every', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    train_ds = ChessDatasetAgg(args.agg_dir, 'train', args.value_mode)
    val_ds = ValView(ChessDatasetAgg(args.agg_dir, 'val', args.value_mode))
    # Cap val to a representative strided subset — the full val set (1.4M on the
    # 100M corpus) is cold-cache I/O-bound and wasteful to scan every epoch.
    if args.val_cap > 0 and len(val_ds) > args.val_cap:
        from torch.utils.data import Subset
        sub = np.linspace(0, len(val_ds) - 1, args.val_cap).astype(np.int64)
        val_ds = Subset(val_ds, sub.tolist())
    print(f"agg={args.agg_dir} U={train_ds.meta['n_unique']:,} "
          f"train={len(train_ds):,} val={len(val_ds):,} | "
          f"value={args.value_mode} policy={args.policy_mode} tau={args.tau} "
          f"epoch_size={args.epoch_size:,}", flush=True)

    # count^tau sampler over train positions (numpy-based; no 2^24 cap)
    w = np.power(train_ds.count[train_ds.indices].astype(np.float64), args.tau)
    sampler = TemperedSampler(w, num_samples=args.epoch_size, seed=args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None)
    val_loader = DataLoader(
        val_ds, batch_size=2048, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None)

    cfg = ChessConfigV3(d_model=args.d_model, n_heads=args.n_heads,
                        n_blocks=args.n_blocks, checkpoint_every=args.checkpoint_every)
    model = ChessModelV3(cfg).to(device)
    print(f"params={count_params(model):,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, fused=True)
    sched = make_cosine(opt, args.lr_horizon)
    val_loss_fn = nn.MSELoss()
    gen = torch.Generator(device=device); gen.manual_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    # Auto-resume (crash insurance for long runs on the degraded box). Prefer the
    # highest completed-epoch checkpoint; if model_latest.pt is at/ahead of it
    # (crash mid-epoch), restart that epoch from these weights.
    start_epoch = 0
    import glob, re
    ckpts = glob.glob(os.path.join(args.save_dir, f'{args.save_name}_e*.pt'))
    best_e, best_p = -1, None
    for p in ckpts:
        m = re.search(re.escape(args.save_name) + r'_e(\d+)\.pt$', os.path.basename(p))
        if m and int(m.group(1)) > best_e:
            best_e, best_p = int(m.group(1)), p
    resume_p = best_p
    resume_next = best_e + 1
    latest = os.path.join(args.save_dir, f'{args.save_name}_latest.pt')
    if os.path.isfile(latest):
        try:
            _le = torch.load(latest, map_location='cpu', weights_only=False).get('epoch', -1)
            if _le >= resume_next:
                resume_p, resume_next = latest, _le
        except Exception:
            pass
    if resume_p is not None:
        d = torch.load(resume_p, map_location=device, weights_only=False)
        model.load_state_dict(d['model'])
        try:
            opt.load_state_dict(d['optimizer']); sched.load_state_dict(d['scheduler'])
        except (KeyError, ValueError):
            pass
        start_epoch = resume_next
        print(f"RESUME from {resume_p} -> next epoch {start_epoch}", flush=True)

    rows = []
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        ep_pol = ep_val = 0.0; seen = 0; correct = 0
        nb = len(train_loader)
        for bi, (x, val, pm, pp) in enumerate(train_loader):
            x = x.to(device, non_blocking=True).float()
            val = val.to(device, non_blocking=True).float()
            pm = pm.to(device, non_blocking=True)
            pp = pp.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, value = model(x)
                value = value.squeeze(-1)
                pol = policy_loss(logits, pm, pp, args.policy_mode, gen)
                vloss = val_loss_fn(value, val)
                loss = pol + args.value_loss_weight * vloss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bs = x.size(0)
            ep_pol += pol.item() * bs; ep_val += vloss.item() * bs; seen += bs
            with torch.no_grad():
                tgt = pm.gather(1, pp.argmax(1, keepdim=True)).squeeze(1)
                correct += (logits.argmax(1) == tgt).sum().item()
            if (bi + 1) % 100 == 0 or bi == nb - 1:
                print(f"\r  e{epoch} [{bi+1}/{nb}] pol {ep_pol/seen:.4f} "
                      f"val {ep_val/seen:.4f} acc {correct/seen:.4f}",
                      end='', flush=True)
            if args.save_every_steps and (bi + 1) % args.save_every_steps == 0:
                p = os.path.join(args.save_dir, f'{args.save_name}_latest.pt')
                save_v3_checkpoint(p + '.tmp', model, epoch=epoch, config=cfg)
                os.replace(p + '.tmp', p)
        print()
        sched.step()
        m = evaluate(model, val_loader, device, None)
        dt = time.time() - t0
        print(f"  EVAL e{epoch}: rmse_one {m['val_rmse_one']:.4f} "
              f"rmse_avg {m['val_rmse_avg']:.4f} rmse_avg_hi {m['val_rmse_avg_hi']:.4f} "
              f"| top1 {m['top1']:.4f} top3 {m['top3']:.4f} polCE {m['pol_ce']:.4f} "
              f"| w_rmse_avg {m['w_rmse_avg']:.4f} w_top1 {m['w_top1']:.4f} "
              f"w_polCE {m['w_pol_ce']:.4f} | {dt:.0f}s", flush=True)
        ck = os.path.join(args.save_dir, f"{args.save_name}_e{epoch:04d}.pt")
        save_v3_checkpoint(ck, model, optimizer=opt, scheduler=sched,
                           epoch=epoch, config=cfg)
        row = dict(epoch=epoch, train_pol=ep_pol/seen, train_val=ep_val/seen,
                   train_acc=correct/seen, **m,
                   value_mode=args.value_mode, policy_mode=args.policy_mode,
                   tau=args.tau, save_name=args.save_name, sec=round(dt))
        rows.append(row)
    if args.metrics_csv:
        import csv
        new = not os.path.exists(args.metrics_csv)
        with open(args.metrics_csv, 'a', newline='') as f:
            wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if new:
                wtr.writeheader()
            wtr.writerows(rows)
    print("DONE", flush=True)


if __name__ == '__main__':
    main()

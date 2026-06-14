# -*- coding: utf-8 -*-
"""Fast, GPU-bound trainer for the tau recipe on a BIT-PACKED, RAM-resident
artifact (built by `src/v3/pack_agg.py`).

The standard `train_agg.py` is disk-I/O-paced (~2000 samp/s) because each sample
is a random ~1.3KB read from the 97GB X memmap, leaving the tiny ~0.7M-param
model's GPU ~95% idle. Here the whole working set (bit-packed X ~16GB + dense
top-K policy targets + value/count/split, ~26GB total) lives in RAM, batches are
vectorized gathers, X is bit-UNPACKED on the GPU, and the model is torch.compiled
(reduce-overhead / CUDA graphs) — so training is compute-bound (~30-80k samp/s).

The RECIPE is unchanged from the validated tau run: averaged-value MSE + soft-
policy KL to the human move histogram + count^tau sampling (tau~0.5) + vlw 1.0.
Only the data path and throughput mechanics differ. Checkpoints are byte-identical
in format to train_agg.py (arch='v3'), so eval/play load them unchanged.
"""
import argparse
import glob
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.v3.model import ChessConfigV3, ChessModelV3, count_params
from src.v3.inference import save_v3_checkpoint


# ----------------------------------------------------------------------------- data
class PackedCorpus:
    """Loads the packed artifact fully into RAM and serves GPU-unpacked batches."""

    def __init__(self, packed_dir, device, distill_dir=None):
        with open(os.path.join(packed_dir, 'meta.json')) as f:
            self.meta = json.load(f)
        self.device = device
        U = self.meta['n_packed']
        self.U = U
        self.P = self.meta['input_planes']
        self.PB = self.meta['packed_bytes']
        self.nb = self.meta['n_binary']
        self.nn = self.meta['n_nonbinary']
        self.K = self.meta['topk']
        binp = self.meta['binary_planes']
        nonp = self.meta['nonbinary_planes']

        ld = lambda n, dt: np.fromfile(os.path.join(packed_dir, n), dtype=dt)
        t0 = time.time()
        self.Xp = ld('Xpacked.bin', np.uint8).reshape(U, self.PB)      # ~16GB
        self.pol_moves = ld('pol_moves.bin', np.int16).reshape(U, self.K)
        self.pol_probs = ld('pol_probs.bin', np.float16).reshape(U, self.K)
        self.value_avg = ld('value_avg.bin', np.float32)
        self.count = ld('count.bin', np.int32)
        self.split = ld('split.bin', np.int8)
        gb = (self.Xp.nbytes + self.pol_moves.nbytes + self.pol_probs.nbytes
              + self.value_avg.nbytes + self.count.nbytes + self.split.nbytes) / 1e9
        print(f"loaded packed corpus into RAM: {gb:.1f}GB in {time.time()-t0:.0f}s "
              f"(U={U:,} PB={self.PB} K={self.K})", flush=True)

        self.has_teacher = distill_dir is not None
        if self.has_teacher:
            self.t_moves = np.fromfile(os.path.join(distill_dir, 'teacher_moves.bin'),
                                       np.int16).reshape(U, -1)
            self.t_logits = np.fromfile(os.path.join(distill_dir, 'teacher_logits.bin'),
                                        np.float16).reshape(U, -1)
            self.t_value = np.fromfile(os.path.join(distill_dir, 'teacher_value.bin'), np.float16)
            print(f"loaded teacher targets {self.t_moves.shape} "
                  f"({(self.t_moves.nbytes+self.t_logits.nbytes+self.t_value.nbytes)/1e9:.1f}GB) "
                  f"from {distill_dir}", flush=True)

        # GPU-side constants for unpacking
        self.binary_idx = torch.tensor(binp, dtype=torch.long, device=device)
        self.nonbinary_idx = torch.tensor(nonp, dtype=torch.long, device=device)
        self.shifts = torch.tensor([7, 6, 5, 4, 3, 2, 1, 0], dtype=torch.int32, device=device)

        self.train_idx = np.flatnonzero(self.split == 0).astype(np.int64)
        self.val_idx = np.flatnonzero(self.split == 1).astype(np.int64)

    def unpack(self, packed):
        """(B, PB) uint8 on GPU -> (B, P, 8, 8) float32, == X[i].astype(float32)."""
        B = packed.shape[0]
        bin_bytes = packed[:, :self.nb * 8].to(torch.int32)            # (B, nb*8)
        bits = (bin_bytes.unsqueeze(-1) >> self.shifts) & 1            # (B, nb*8, 8)
        bits = bits.reshape(B, self.nb, 64).to(torch.float32)
        x = torch.zeros(B, self.P, 64, device=packed.device, dtype=torch.float32)
        x[:, self.binary_idx, :] = bits
        if self.nn:
            nbb = packed[:, self.nb * 8:].reshape(B, self.nn, 64).to(torch.float32)
            x[:, self.nonbinary_idx, :] = nbb
        return x.reshape(B, self.P, 8, 8)

    def fetch(self, idx):
        """Gather a batch by global position index -> GPU tensors."""
        packed = torch.from_numpy(self.Xp[idx]).to(self.device, non_blocking=True)
        x = self.unpack(packed)
        val = torch.from_numpy(self.value_avg[idx]).to(self.device, non_blocking=True)
        pm = torch.from_numpy(self.pol_moves[idx].astype(np.int64)).to(self.device, non_blocking=True)
        pp = torch.from_numpy(self.pol_probs[idx].astype(np.float32)).to(self.device, non_blocking=True)
        return x, val, pm, pp

    def fetch_teacher(self, idx):
        """Teacher top-K (move indices, raw logits) + value for distillation."""
        tm = torch.from_numpy(self.t_moves[idx].astype(np.int64)).to(self.device, non_blocking=True)
        tl = torch.from_numpy(self.t_logits[idx].astype(np.float32)).to(self.device, non_blocking=True)
        tv = torch.from_numpy(self.t_value[idx].astype(np.float32)).to(self.device, non_blocking=True)
        return tm, tl, tv


class TemperedDraw:
    """count^tau sampling over train positions via cumsum+searchsorted (no 2^24
    category cap, unlike torch.multinomial)."""

    def __init__(self, count_train, train_idx, tau, seed=0):
        w = np.power(count_train.astype(np.float64), tau)
        self.cdf = np.cumsum(w)
        self.cdf /= self.cdf[-1]
        self.train_idx = train_idx
        self.seed = seed

    def draw(self, n, epoch):
        rng = np.random.default_rng(self.seed + 1000 + epoch)
        u = rng.random(n)
        local = np.searchsorted(self.cdf, u, side='right')
        np.clip(local, 0, len(self.train_idx) - 1, out=local)
        return self.train_idx[local]


# ----------------------------------------------------------------------------- loss
def policy_loss_soft(logits, pm, pp):
    logp = F.log_softmax(logits, dim=1)
    return -(pp * logp.gather(1, pm)).sum(dim=1).mean()


@torch.no_grad()
def evaluate(model, corpus, device, batch=4096, cap=200_000):
    model.eval()
    idx = corpus.val_idx
    if cap and len(idx) > cap:
        idx = idx[np.linspace(0, len(idx) - 1, cap).astype(np.int64)]
    n = 0; se_avg = 0.0; top1 = 0; top3 = 0; polce = 0.0
    for s in range(0, len(idx), batch):
        b = idx[s:s + batch]
        x, val, pm, pp = corpus.fetch(b)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, value = model(x)
        logits = logits.float(); value = value.squeeze(-1).float()
        se_avg += ((value - val) ** 2).sum().item()
        tgt = pm.gather(1, pp.argmax(1, keepdim=True)).squeeze(1)
        topk = logits.topk(3, dim=1).indices
        top1 += (topk[:, 0] == tgt).sum().item()
        top3 += (topk == tgt.unsqueeze(1)).any(1).sum().item()
        polce += -(pp * F.log_softmax(logits, 1).gather(1, pm)).sum(1).sum().item()
        n += len(b)
    return {'val_rmse_avg': math.sqrt(se_avg / n), 'top1': top1 / n,
            'top3': top3 / n, 'pol_ce': polce / n, 'n_val': n}


def lr_factor(step, warmup, horizon):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, horizon - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--packed-dir', default='data/v2/agg_100M_packed')
    ap.add_argument('--save-dir', required=True)
    ap.add_argument('--save-name', default='model')
    ap.add_argument('--tau', type=float, default=0.5)
    ap.add_argument('--value-loss-weight', type=float, default=1.0)
    ap.add_argument('--epoch-size', type=int, default=10_000_000)
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--lr-horizon', type=int, default=12, help='cosine horizon in EPOCHS')
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--warmup-steps', type=int, default=300)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--save-every-steps', type=int, default=0)
    ap.add_argument('--metrics-csv', default=None)
    ap.add_argument('--compile', action='store_true',
                    help='opt-in torch.compile; needs Triton (absent on Windows here) and '
                         'the cudagraphs backend gave ~0% on this tiny model, so default is eager')
    ap.add_argument('--max-steps', type=int, default=0, help='>0: stop early (smoke test)')
    # fixed-arch knobs (v3-micro-tau = d64/h8/b10, stem2/ffn4/vh128 = production defaults)
    ap.add_argument('--d-model', type=int, default=64)
    ap.add_argument('--n-heads', type=int, default=8)
    ap.add_argument('--n-blocks', type=int, default=10)
    ap.add_argument('--value-hidden', type=int, default=128)
    ap.add_argument('--ffn-mult', type=int, default=4)
    ap.add_argument('--stem-blocks', type=int, default=2)
    ap.add_argument('--stem-kernel', type=int, default=3,
                    help='1 + --stem-blocks 0 = pure per-square embed, no conv locality (v3.1)')
    ap.add_argument('--no-geometry-bias', action='store_true',
                    help='disable the learned 2D geometry bias (v3.2 ablation)')
    ap.add_argument('--no-pos-emb', action='store_true',
                    help='disable absolute pos embedding; rely on geometry bias (v3.2 R4)')
    ap.add_argument('--share-blocks', action='store_true',
                    help='reuse ONE transformer block across all n_blocks (v3.2 R6)')
    # distillation from a precomputed teacher (src/v3/teacher_label.py)
    ap.add_argument('--distill-dir', default=None,
                    help='dir with teacher_{moves,logits,value}.bin aligned to the corpus')
    ap.add_argument('--distill-alpha', type=float, default=0.0,
                    help='0=pure human labels, 1=pure teacher; mixes policy+value losses')
    ap.add_argument('--distill-temp', type=float, default=1.0,
                    help='softmax temperature on the teacher top-K logits')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    corpus = PackedCorpus(args.packed_dir, device, distill_dir=args.distill_dir)
    sampler = TemperedDraw(corpus.count[corpus.train_idx], corpus.train_idx, args.tau, args.seed)
    steps_per_epoch = args.epoch_size // args.batch_size
    horizon_steps = args.lr_horizon * steps_per_epoch
    print(f"train={len(corpus.train_idx):,} val={len(corpus.val_idx):,} | "
          f"tau={args.tau} epoch_size={args.epoch_size:,} bs={args.batch_size} "
          f"steps/epoch={steps_per_epoch:,} lr={args.lr} warmup={args.warmup_steps}", flush=True)

    cfg = ChessConfigV3(d_model=args.d_model, n_heads=args.n_heads,
                        n_blocks=args.n_blocks, ffn_mult=args.ffn_mult,
                        stem_blocks=args.stem_blocks, stem_kernel=args.stem_kernel,
                        value_hidden=args.value_hidden,
                        geometry_bias=not args.no_geometry_bias,
                        use_pos_emb=not args.no_pos_emb, share_blocks=args.share_blocks,
                        checkpoint_every=0)
    model = ChessModelV3(cfg).to(device)
    print(f"params={count_params(model):,} (cfg d{cfg.d_model} h{cfg.n_heads} "
          f"b{cfg.n_blocks} stem{cfg.stem_blocks} ffn{cfg.ffn_mult} vh{cfg.value_hidden})",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, fused=True)
    os.makedirs(args.save_dir, exist_ok=True)

    # auto-resume (crash insurance on the degraded box)
    start_epoch = 0
    gstep = 0
    ckpts = glob.glob(os.path.join(args.save_dir, f'{args.save_name}_e*.pt'))
    best_e, best_p = -1, None
    for p in ckpts:
        m = re.search(re.escape(args.save_name) + r'_e(\d+)\.pt$', os.path.basename(p))
        if m and int(m.group(1)) > best_e:
            best_e, best_p = int(m.group(1)), p
    if best_p is not None:
        d = torch.load(best_p, map_location=device, weights_only=False)
        model.load_state_dict(d['model'])
        try:
            opt.load_state_dict(d['optimizer'])
        except (KeyError, ValueError):
            pass
        start_epoch = best_e + 1
        gstep = start_epoch * steps_per_epoch
        print(f"RESUME from {best_p} -> next epoch {start_epoch}", flush=True)

    # Eager by default. torch.compile's inductor backend needs Triton (absent on
    # this Windows box) and the Triton-free cudagraphs backend measured ~0% gain
    # on this d=64 model (its tiny matmuls are memory-bandwidth-bound, not launch-
    # bound), so compiling buys nothing here. The win was the RAM-resident data path.
    train_model = torch.compile(model) if args.compile else model

    rows = []
    for epoch in range(start_epoch, args.epochs):
        model.train()
        ep_idx = sampler.draw(args.epoch_size, epoch)
        t0 = time.time()
        # GPU-side accumulators -> sync (.item()) only at print intervals, so the
        # GPU isn't stalled on a CPU readback every step (this tiny model is
        # otherwise partly sync-bound).
        ep_pol_t = torch.zeros((), device=device)
        ep_val_t = torch.zeros((), device=device)
        correct_t = torch.zeros((), device=device, dtype=torch.long)
        ep_pol = ep_val = 0.0; seen = 0; correct = 0
        for bi in range(steps_per_epoch):
            for g in opt.param_groups:
                g['lr'] = args.lr * lr_factor(gstep, args.warmup_steps, horizon_steps)
            b = ep_idx[bi * args.batch_size:(bi + 1) * args.batch_size]
            x, val, pm, pp = corpus.fetch(b)
            distilling = corpus.has_teacher and args.distill_alpha > 0
            if distilling:
                tm, tl, tv = corpus.fetch_teacher(b)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, value = train_model(x)
                value = value.squeeze(-1)
                pol = policy_loss_soft(logits, pm, pp)
                vloss = F.mse_loss(value, val)
                if distilling:
                    tprob = F.softmax(tl / args.distill_temp, dim=1)
                    d_pol = -(tprob * F.log_softmax(logits, dim=1).gather(1, tm)).sum(1).mean()
                    d_val = F.mse_loss(value, tv)
                    a = args.distill_alpha
                    pol = (1 - a) * pol + a * d_pol
                    vloss = (1 - a) * vloss + a * d_val
                loss = pol + args.value_loss_weight * vloss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            gstep += 1
            bs = x.size(0)
            seen += bs
            with torch.no_grad():
                ep_pol_t += pol.detach() * bs
                ep_val_t += vloss.detach() * bs
                tgt = pm.gather(1, pp.argmax(1, keepdim=True)).squeeze(1)
                correct_t += (logits.argmax(1) == tgt).sum()
            if (bi + 1) % 200 == 0 or bi == steps_per_epoch - 1:
                ep_pol, ep_val, correct = ep_pol_t.item(), ep_val_t.item(), correct_t.item()
                rate = seen / (time.time() - t0)
                print(f"\r  e{epoch} [{bi+1}/{steps_per_epoch}] pol {ep_pol/seen:.4f} "
                      f"val {ep_val/seen:.4f} acc {correct/seen:.4f} lr {opt.param_groups[0]['lr']:.2e} "
                      f"{rate:,.0f} samp/s", end='', flush=True)
            if args.save_every_steps and (bi + 1) % args.save_every_steps == 0:
                p = os.path.join(args.save_dir, f'{args.save_name}_latest.pt')
                save_v3_checkpoint(p + '.tmp', model, epoch=epoch, config=cfg)
                os.replace(p + '.tmp', p)
            if args.max_steps and gstep >= args.max_steps:
                print(f"\n  max-steps {args.max_steps} reached -> stop", flush=True)
                return
        print()
        m = evaluate(model, corpus, device)
        dt = time.time() - t0
        print(f"  EVAL e{epoch}: rmse_avg {m['val_rmse_avg']:.4f} top1 {m['top1']:.4f} "
              f"top3 {m['top3']:.4f} polCE {m['pol_ce']:.4f} | {dt:.0f}s "
              f"({seen/dt:,.0f} samp/s)", flush=True)
        ck = os.path.join(args.save_dir, f"{args.save_name}_e{epoch:04d}.pt")
        save_v3_checkpoint(ck, model, optimizer=opt, epoch=epoch, config=cfg)
        rows.append(dict(epoch=epoch, train_pol=ep_pol/seen, train_val=ep_val/seen,
                         train_acc=correct/seen, **m, tau=args.tau, sec=round(dt)))

    if args.metrics_csv and rows:
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

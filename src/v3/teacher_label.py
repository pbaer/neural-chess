# -*- coding: utf-8 -*-
"""Precompute v3-37M *teacher* targets over the agg_100M positions, for distilling
a 116k v3.1 *student* (the browser drop-in) beyond what human labels can reach.

The 116k student is capacity-saturated under hard/soft human supervision (train==val,
converged). Distillation from a strong teacher's full distribution is the standard
way to push a small student further at fixed params. Principle note: the teacher
(v3-37M) was itself trained ONLY on human games (no engine, no self-play) — the user
granted an explicit carve-out to use its soft outputs as supervision.

For each of the 72M unique positions (agg_100M/X.bin) store the teacher's TOP-32
policy (move indices + raw logits) and its value. Targets align row-for-row with
agg_100M (== the packed corpus order), so the fast trainer can load them directly.
The student applies softmax(logit / T) over the top-32 at train time (any T).

Output (OUT_DIR):
  teacher_moves.bin   int16   (U, 32)   top-32 move indices
  teacher_logits.bin  float16 (U, 32)   their raw logits
  teacher_value.bin   float16 (U,)      teacher value (tanh, side-to-move)
  meta.json
Run from repo root.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from src.v3.inference import load_v3_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher', default='model/v3/v3-37M/model_e0009.pt')
    ap.add_argument('--agg-dir', default='data/v2/agg_100M')
    ap.add_argument('--out-dir', default='data/v2/agg_100M_teacher')
    ap.add_argument('--topk', type=int, default=32)
    ap.add_argument('--batch', type=int, default=8192)
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32 = True
    m = load_v3_model(args.teacher, device=dev)
    m.eval()
    X = np.memmap(os.path.join(args.agg_dir, 'X.bin'), dtype=np.int8, mode='r').reshape(-1, 21, 8, 8)
    U, K = X.shape[0], args.topk
    tmp = args.out_dir + '.tmp'
    prog_path = os.path.join(tmp, 'progress.txt')
    resume_from, mode = 0, 'w+'
    if os.path.isdir(tmp) and os.path.isfile(prog_path):
        try:
            resume_from, mode = int(open(prog_path).read().strip()), 'r+'
            print(f"RESUME from index {resume_from:,}", flush=True)
        except (ValueError, OSError):
            resume_from, mode = 0, 'w+'
    os.makedirs(tmp, exist_ok=True)
    mv = np.memmap(os.path.join(tmp, 'teacher_moves.bin'), dtype=np.int16, mode=mode, shape=(U, K))
    lg = np.memmap(os.path.join(tmp, 'teacher_logits.bin'), dtype=np.float16, mode=mode, shape=(U, K))
    vv = np.memmap(os.path.join(tmp, 'teacher_value.bin'), dtype=np.float16, mode=mode, shape=(U,))
    print(f"teacher={args.teacher} | U={U:,} topk={K} batch={args.batch} dev={dev} start={resume_from:,}", flush=True)
    t0 = time.time()
    with torch.no_grad():
        for bi, i in enumerate(range(resume_from, U, args.batch)):
            j = min(i + args.batch, U)
            xb = torch.from_numpy(np.ascontiguousarray(X[i:j])).float().to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                p, v = m(xb)
            top = torch.topk(p.float(), K, dim=1)
            mv[i:j] = top.indices.to(torch.int16).cpu().numpy()
            lg[i:j] = top.values.to(torch.float16).cpu().numpy()
            vv[i:j] = v.float().squeeze(1).to(torch.float16).cpu().numpy()
            if bi % 50 == 0:
                mv.flush(); lg.flush(); vv.flush()                  # durable before progress
                with open(prog_path, 'w') as f:
                    f.write(str(j))
                r = (j - resume_from) / max(time.time() - t0, 1e-6)
                print(f"  {j:,}/{U:,} ({100*j/U:.1f}%) {r:,.0f}/s eta {(U-j)/max(r,1)/60:.0f}m", flush=True)
    mv.flush(); lg.flush(); vv.flush()
    json.dump({'n': int(U), 'topk': K, 'teacher': args.teacher}, open(os.path.join(tmp, 'meta.json'), 'w'))
    if os.path.isfile(prog_path):
        os.remove(prog_path)
    del mv, lg, vv                       # release Windows memmap handles before rename
    import gc
    gc.collect()
    os.rename(tmp, args.out_dir)
    print(f"DONE {U:,} positions in {(time.time()-t0)/60:.1f}m -> {args.out_dir}", flush=True)


if __name__ == '__main__':
    main()

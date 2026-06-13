# -*- coding: utf-8 -*-
"""v3.2 architecture exploration — BATCH 2 (needs the model.py toggles added 2026-06-06).

Batch 1 (run_v3_2_explore.py) covered the free-flag axes (FFN, geometry bias,
heads, width). Batch 2 tests the two new structural toggles plus the combined
"lean" recipe:

  R4 POS-OFF    use_pos_emb off (114k, -2k)      : is the absolute square embedding
                                                   dead weight given the geometry bias? (Principle-clean)
  R6a SHARE     share one block x8 (21k, -82%)   : do the 8 blocks need DISTINCT weights,
                                                   or is one iterated transform enough? (ALBERT bet)
  R6b SHARE-DEEP share one block x16 (21k)        : spend the freed depth — same params, 2x iterations
  R5 LEAN       <FILLED IN AFTER BATCH 1>         : stack the batch-1 winners into one net

Reuses batch 1's train/eval machinery (same recipe, same packed corpus, same
sf_easy+med 120g eval) so results are directly comparable. Run from repo root.
"""
import argparse, os, time
import sys
sys.path.insert(0, os.path.abspath('.'))
from eval.v3.run_v3_2_explore import (variant, train, sf_eval, params_of,
                                       final_top1, BASELINE, LOGDIR)

LOG = 'logs/v3_2_batch2.log'


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


# R5-LEAN is deliberately left out until batch 1 names the winners; append it
# here (a dict with name/args/epochs) before launching, e.g.
#   dict(name='R5-lean', args=variant(no_pos_emb=True, ffn_mult=2, ...), epochs=16)
RUNS = [
    dict(name='R4-pos-off',     args=variant(no_pos_emb=True),                  epochs=16),
    dict(name='R6a-share',      args=variant(share_blocks=True),                epochs=16),
    dict(name='R6b-share-deep', args=variant(share_blocks=True, n_blocks=16),   epochs=16),
    # R5-LEAN: the synthesis / decisive depth-saturation point. Keep every load-
    # bearing piece (geo bias, pos_emb, 4 heads, DISTINCT blocks) and push the one
    # neutral-or-positive lever — FFN->depth — to its limit at ~equal budget:
    # thinnest FFN (ffn1) + maximal distinct depth (b15) ~= 117.9k. Third point on
    # the baseline(ffn4/b8) -> R2(ffn2/b11) -> R5(ffn1/b15) depth curve.
    dict(name='R5-lean',        args=variant(ffn_mult=1, n_blocks=15),          epochs=16),
]


def main():
    argparse.ArgumentParser().parse_args()
    os.makedirs('model/v3/v3.2', exist_ok=True); os.makedirs(LOGDIR, exist_ok=True)
    log("=== V3.2 EXPLORE (batch 2: structural toggles) START ===")
    log(f"baseline = {BASELINE}")
    for run in RUNS:
        if not train(run):
            log(f"{run['name']} failed; skip eval"); continue
        res = sf_eval(run)
        e, m = res.get('sf_easy', (None, None)), res.get('sf_med', (None, None))
        log(f"RESULT {run['name']} ({params_of(run['name'])}p): sf_easy won {e[0]}% lost {e[1]}% "
            f"| sf_med won {m[0]}% | top1 {final_top1(run['name'])}  [vs v3.1-eq 77.0/43.3]")
    log("=== V3.2 EXPLORE batch 2 DONE ===")


if __name__ == '__main__':
    main()

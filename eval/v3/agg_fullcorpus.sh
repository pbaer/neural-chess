#!/usr/bin/env bash
# 18M full-corpus measurement: recipe (avg/soft/tau0.5) vs production-style
# baseline (one/hard_sample/tau1.0) on the FULL 100M-instance aggregated corpus.
# Tests whether the recipe's advantage holds at production data volume (the
# probe was only 8M instances) before committing to a multi-day 37M run.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
AGG=data/v2/agg_100M
CSV=eval/v3/agg_full_metrics.csv
rm -f "$CSV"
# ~5 epochs x 10M samples = 50M draws/model. save-every-steps for crash insurance
# on the degraded box (atomic model_latest.pt every ~10k steps ~= 50 min).
# ~5 epochs x 8M samples = 40M draws/model. 16 workers to parallelize the
# random SSD reads of the 92GB X memmap (training is I/O-paced ~2000/s, like the
# production 100M shard, since the corpus exceeds RAM). val capped (--val-cap
# default 300k) so cold val scans don't dominate each epoch.
COMMON="--agg-dir $AGG --epoch-size 8000000 --epochs 5 --lr-horizon 5 \
  --batch-size 1024 --lr 1e-3 --num-workers 16 --seed 0 --metrics-csv $CSV \
  --save-every-steps 5000"

echo "### TRAIN full_ctrl (one/hard_sample/tau1.0) ###"
$PY -m src.v3.train_agg $COMMON --save-dir model/v3/agg/full_ctrl --save-name full_ctrl \
  --value-mode one --policy-mode hard_sample --tau 1.0 > eval/v3/logs/agg_full_ctrl.log 2>&1
echo "  full_ctrl rc=$?"

echo "### TRAIN full_recipe (avg/soft/tau0.5) ###"
$PY -m src.v3.train_agg $COMMON --save-dir model/v3/agg/full_recipe --save-name full_recipe \
  --value-mode avg --policy-mode soft --tau 0.5 > eval/v3/logs/agg_full_recipe.log 2>&1
echo "  full_recipe rc=$?"

echo "### SF LADDER (epoch 4 = final) ###"
SFCSV=eval/v3/agg_full_sf.csv; rm -f "$SFCSV"
for m in full_ctrl full_recipe; do
  echo "## SF $m ##"
  $PY eval/v2/eval_v2.py 4 --save-dir model/v3/agg/$m --save-name $m \
    --eval-csv "$SFCSV" --tiers sf_easy,sf_med,sf_hard --skip-h2h --skip-random 2>&1 \
    | grep -E "W:|ERROR"
done

echo "### HEAD-TO-HEAD full_recipe vs full_ctrl (400g) ###"
$PY eval/v3/agg_h2h.py model/v3/agg/full_recipe/full_recipe_e0004.pt \
  model/v3/agg/full_ctrl/full_ctrl_e0004.pt 400 0.5 2>&1 | grep -E "A:|score"
echo "FULLCORPUS_DONE"

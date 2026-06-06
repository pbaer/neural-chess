#!/usr/bin/env bash
# 6-epoch confirmation: recommended recipe vs production-style baseline, same arch.
# Tests whether the data-strategy advantage widens with more training before a big run.
set -e
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
AGG=data/v2/agg_8M
CSV=eval/v3/agg_confirm_metrics.csv
rm -f "$CSV"
COMMON="--agg-dir $AGG --epoch-size 1500000 --epochs 6 --lr-horizon 8 \
  --batch-size 1024 --lr 1e-3 --num-workers 12 --seed 0 --metrics-csv $CSV \
  --save-every-steps 3000"

echo "### TRAIN ctrl6 (one/hard_sample/tau1.0) ###"
$PY -m src.v3.train_agg $COMMON --save-dir model/v3/agg/ctrl6 --save-name ctrl6 \
  --value-mode one --policy-mode hard_sample --tau 1.0 > eval/v3/logs/agg_ctrl6.log 2>&1
echo "  ctrl6 rc=$?"

echo "### TRAIN recipe6 (avg/soft/tau0.5) ###"
$PY -m src.v3.train_agg $COMMON --save-dir model/v3/agg/recipe6 --save-name recipe6 \
  --value-mode avg --policy-mode soft --tau 0.5 > eval/v3/logs/agg_recipe6.log 2>&1
echo "  recipe6 rc=$?"

echo "### SF LADDER ###"
SFCSV=eval/v3/agg_confirm_sf.csv; rm -f "$SFCSV"
for m in ctrl6 recipe6; do
  echo "## SF $m ##"
  $PY eval/v2/eval_v2.py 5 --save-dir model/v3/agg/$m --save-name $m \
    --eval-csv "$SFCSV" --tiers sf_easy,sf_med,sf_hard --skip-h2h --skip-random 2>&1 \
    | grep -E "W:|ERROR"
done

echo "### HEAD-TO-HEAD recipe6 vs ctrl6 (300g) ###"
$PY eval/v3/agg_h2h.py model/v3/agg/recipe6/recipe6_e0005.pt \
  model/v3/agg/ctrl6/ctrl6_e0005.pt 300 0.5 2>&1 | grep -E "A:|score"
echo "CONFIRM_DONE"

#!/usr/bin/env bash
# v3-18M-tau: the position-aggregated recipe (averaged value + soft policy +
# tau=0.5) trained to full epochs on the full corpus, at the SMALL v3 arch
# (18.3M). Purpose: test whether better DATA (signal density) lets an 18M model
# match/beat the current 37M best (v3-37M) -- the capacity-vs-signal question.
# Then benchmark vs v3-37M (capacity test) and v3-18M (equal-param recipe test).
set -e
cd "$(dirname "$0")/../.."
PY=.venv/Scripts/python.exe
AGG=data/v2/agg_100M
DIR=model/v3/v3-18M-tau
NAME=v3-18M-tau
CSV=eval/v3/${NAME}_metrics.csv

echo "### TRAIN $NAME (avg/soft/tau0.5, 18M, 10 epochs x 40M) ###"
$PY -m src.v3.train_agg --agg-dir $AGG --save-dir $DIR --save-name $NAME \
  --value-mode avg --policy-mode soft --tau 0.5 --value-loss-weight 1.0 \
  --epoch-size 40000000 --epochs 10 --lr-horizon 10 --lr 1e-3 --batch-size 1024 \
  --num-workers 16 --seed 0 --val-cap 300000 --metrics-csv $CSV \
  --save-every-steps 5000 --d-model 256 --n-blocks 20 \
  > eval/v3/logs/${NAME}.log 2>&1
echo "  train rc=$?"

FINAL=$DIR/${NAME}_e0009.pt
echo "### SF LADDER ($NAME final) ###"
SFCSV=eval/v3/${NAME}_sf.csv; rm -f "$SFCSV"
$PY eval/v2/eval_v2.py 9 --save-dir $DIR --save-name $NAME \
  --eval-csv "$SFCSV" --tiers sf_easy,sf_med,sf_hard,sf_magnus --skip-h2h --skip-random 2>&1 \
  | grep -E "EVAL|W:|ERROR"

echo "### H2H vs v3-37M (current best; capacity-vs-signal) 200g ###"
$PY eval/v3/agg_h2h.py $FINAL model/v3/v3-37M/model_e0009.pt 200 0.5 2>&1 | grep -E "A:|score"

echo "### H2H vs v3-18M (equal params, old data; isolates recipe) 200g ###"
$PY eval/v3/agg_h2h.py $FINAL model/v3/v3-18M/model_e0008.pt 200 0.5 2>&1 | grep -E "A:|score"

echo "V3_18M_TAU_DONE"

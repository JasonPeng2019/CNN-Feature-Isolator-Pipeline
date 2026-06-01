#!/usr/bin/env bash
# Phase 3 Family I: wait for Q1 winner SAE, then train all 4 adjacent transitions
# in parallel (one per GPU), then run the chained Q1->Q5 eval.
set -e
cd "$(dirname "$0")/.."
source env.sh
P2=runs/phase2
SAE_Q1=runs/winner_saes/Q1/sae.pt

echo "waiting for Q1 winner SAE..."
until [ -f "$SAE_Q1" ]; do sleep 15; done
echo "Q1 SAE ready; launching 4 transitions"

declare -a P=("Q1 Q2 $SAE_Q1 $P2/E1_Q2_f0.05/sae.pt 0"
              "Q2 Q3 $P2/E1_Q2_f0.05/sae.pt $P2/E1_Q3_f0.05/sae.pt 1"
              "Q3 Q4 $P2/E1_Q3_f0.05/sae.pt $P2/E1_Q4_f0.05/sae.pt 2"
              "Q4 Q5 $P2/E1_Q4_f0.05/sae.pt $P2/E1_Q5_f0.05/sae.pt 3")
for p in "${P[@]}"; do set -- $p; src=$1; dst=$2; ss=$3; sd=$4; gpu=$5
  out=runs/phase3/T_${src}_${dst}
  python src/train_transition.py --src $src --dst $dst --sae_src $ss --sae_dst $sd \
     --frac 0.05 --epochs 30 --device cuda:$gpu --out $out > ${out}.log 2>&1 &
done
wait
echo "PHASE3 TRANSITIONS DONE"
ls runs/phase3/T_*/result.json

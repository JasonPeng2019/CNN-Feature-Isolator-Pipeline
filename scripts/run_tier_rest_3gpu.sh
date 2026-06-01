#!/usr/bin/env bash
# 3-GPU continuation (GPU2 dead, Xid 154). Waits for the in-flight R110 backbone,
# then runs remaining phases on GPUs 0,1,3 only. No `set -e`.
cd "$(dirname "$0")/.."
source env.sh
LOG=runs/tier_rest; mkdir -p $LOG
echo "=== wait for R110 backbone ==="
until [ -f runs/backbone_r110_c100/done.json ]; do sleep 20; done
echo "R110 backbone done: $(cat runs/backbone_r110_c100/done.json)"

echo "=== taxonomy (g0, re-run) ==="
python src/taxonomy.py --device cuda:0 --out runs/taxonomy > $LOG/taxonomy.log 2>&1 &
echo "=== cache R20 acts (g1) + R110 acts (g3) ==="
python src/cache_activations.py --ckpt runs/backbone_r20_c10/best.pt --dataset cifar10 --out runs/acts_r20_c10 --device cuda:1 > $LOG/cache_r20.log 2>&1 &
python src/cache_activations.py --ckpt runs/backbone_r110_c100/best.pt --dataset cifar100 --out runs/acts_r110_c100 --device cuda:3 > $LOG/cache_r110.log 2>&1 &
wait

echo "=== R20/C10 recon grid (gpus 0,1,3) ==="
python scripts/run_grid.py --phase r20c10 --out_root runs/r20c10 --gpus 0,1,3 > $LOG/r20c10_grid.out 2>&1
echo "=== multi-seed grid (gpus 0,1,3) ==="
python scripts/run_grid.py --phase seeds --out_root runs/seeds --gpus 0,1,3 > $LOG/seeds_grid.out 2>&1

echo "=== ResNet-110 recon (Field@5%, 5 sections, gpus 0,1,3) ==="
gpus=(0 1 3 0 1)
i=0
for sec in Q1 Q2 Q3 Q4 Q5; do
  python src/train_sae.py --section $sec --sae_type field --mask global --target_frac 0.05 --Kmult 8 --n_blocks 3 --epochs 40 \
     --acts runs/acts_r110_c100 --ckpt runs/backbone_r110_c100/best.pt --dataset cifar100 \
     --device cuda:${gpus[$i]} --out runs/r110/F_$sec > $LOG/r110_$sec.log 2>&1 &
  i=$((i+1)); if [ $i -ge 3 ]; then wait; i=0; fi
done
wait

echo "=== ViT-SAE on Imagenette (g0) ==="
python src/vit_sae.py --device cuda:0 --out runs/vit_sae > $LOG/vit_sae.log 2>&1 || echo "VIT step failed (non-fatal)"
echo "TIER_REST DONE"

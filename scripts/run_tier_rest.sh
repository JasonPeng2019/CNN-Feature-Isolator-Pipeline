#!/usr/bin/env bash
# Orchestrate all remaining tier experiments in dependency-ordered phases.
# Assumes: r56 acts cached, phase2 winner SAEs present, R20/C10 backbone trained.
# NOTE: no `set -e` — a non-critical sub-job failure must not abort the pipeline.
cd "$(dirname "$0")/.."
source env.sh
LOG=runs/tier_rest; mkdir -p $LOG
echo "=== Phase A: R110 backbone (g0) + taxonomy (g1) + audit Q3 (g2) + audit Q5 (g3) ==="
python scripts/train_backbone.py --arch resnet110 --dataset cifar100 --epochs 100 --device cuda:0 --out runs/backbone_r110_c100 > $LOG/r110_backbone.log 2>&1 &
python src/taxonomy.py --device cuda:1 --out runs/taxonomy > $LOG/taxonomy.log 2>&1 &
python src/feature_audit.py --section Q3 --sae runs/phase2/E1_Q3_f0.05/sae.pt --device cuda:2 --out runs/audit > $LOG/audit_Q3.log 2>&1 &
python src/feature_audit.py --section Q5 --sae runs/phase2/E1_Q5_f0.05/sae.pt --device cuda:3 --out runs/audit > $LOG/audit_Q5.log 2>&1 &
wait
echo "=== Phase B: cache R20 acts (g0) + cache R110 acts (g1) ==="
python src/cache_activations.py --ckpt runs/backbone_r20_c10/best.pt --dataset cifar10 --out runs/acts_r20_c10 --device cuda:0 > $LOG/cache_r20.log 2>&1 &
python src/cache_activations.py --ckpt runs/backbone_r110_c100/best.pt --dataset cifar100 --out runs/acts_r110_c100 --device cuda:1 > $LOG/cache_r110.log 2>&1 &
wait
echo "=== Phase C: R20/C10 recon grid (all GPUs) ==="
python scripts/run_grid.py --phase r20c10 --out_root runs/r20c10 --gpus 0,1,2,3 > $LOG/r20c10_grid.out 2>&1
echo "=== Phase S: multi-seed stability grid (r56, Field@5%, seeds 0/1/2) ==="
python scripts/run_grid.py --phase seeds --out_root runs/seeds --gpus 0,1,2,3 > $LOG/seeds_grid.out 2>&1
echo "=== Phase D: ResNet-110/C100 recon (Field SAE @5%, 5 sections) ==="
gpus=(0 1 2 3 0); i=0
for sec in Q1 Q2 Q3 Q4 Q5; do
  python src/train_sae.py --section $sec --sae_type field --mask global --target_frac 0.05 --Kmult 8 --n_blocks 3 --epochs 40 \
     --acts runs/acts_r110_c100 --ckpt runs/backbone_r110_c100/best.pt --dataset cifar100 \
     --device cuda:${gpus[$i]} --out runs/r110/F_$sec > $LOG/r110_$sec.log 2>&1 &
  i=$((i+1)); if [ $i -ge 4 ]; then wait; i=0; fi
done
wait
echo "=== Phase E: ViT-SAE transfer on Imagenette (g0) ==="
python src/vit_sae.py --device cuda:0 --out runs/vit_sae > $LOG/vit_sae.log 2>&1 || echo "VIT step failed (non-fatal)"
echo "TIER_REST DONE"

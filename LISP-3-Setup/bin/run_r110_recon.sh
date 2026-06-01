#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/lib/common.sh"
setup_runtime_env

if [[ ! -d "${RUNS_DIR}/acts_r110_c100" ]]; then
  echo "acts_r110_c100 missing; generating cache first"
  "${LISP_3_SETUP}/bin/cache_acts.sh"
fi

IFS=',' read -r -a GPUS <<< "${GPU_LIST_PRIMARY}"
G0="${GPUS[0]}"
G1="${GPUS[1]:-${GPUS[0]}}"

for sec in Q1 Q2 Q3 Q4 Q5; do
  if [[ "${sec}" == "Q1" || "${sec}" == "Q3" || "${sec}" == "Q5" ]]; then
    gpu="${G0}"
  else
    gpu="${G1}"
  fi
  run_with_summary "r110_${sec}" "${PYTHON_BIN} 'src/train_sae.py' --section '${sec}' --sae_type field --mask global --target_frac 0.05 --Kmult 8 --n_blocks 3 --epochs 40 --acts 'runs/acts_r110_c100' --ckpt 'runs/backbone_r110_c100/best.pt' --dataset cifar100 --device cuda:${gpu} --out 'runs/r110/F_${sec}'"
done

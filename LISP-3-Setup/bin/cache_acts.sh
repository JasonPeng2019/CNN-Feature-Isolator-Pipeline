#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/lib/common.sh"
setup_runtime_env

IFS=',' read -r -a GPUS <<< "${GPU_LIST_PRIMARY}"
G0="${GPUS[0]}"
G1="${GPUS[1]:-${GPUS[0]}}"

if [[ ! -d "${RUNS_DIR}/acts_r20_c10" ]]; then
  run_with_summary "cache_r20_c10" "${PYTHON_BIN} 'src/cache_activations.py' --ckpt 'runs/backbone_r20_c10/best.pt' --dataset cifar10 --out 'runs/acts_r20_c10' --device cuda:${G0}"
else
  echo "SKIP cache_r20_c10: ${RUNS_DIR}/acts_r20_c10 exists"
fi

if [[ ! -d "${RUNS_DIR}/acts_r110_c100" ]]; then
  run_with_summary "cache_r110_c100" "${PYTHON_BIN} 'src/cache_activations.py' --ckpt 'runs/backbone_r110_c100/best.pt' --dataset cifar100 --out 'runs/acts_r110_c100' --device cuda:${G1}"
else
  echo "SKIP cache_r110_c100: ${RUNS_DIR}/acts_r110_c100 exists"
fi

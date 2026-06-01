#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/lib/common.sh"
setup_runtime_env

TARGET="${1:-all}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST_PRIMARY}"
G0="${GPUS[0]}"
G1="${GPUS[1]:-${GPUS[0]}}"

need_r20=false
need_r56=false
need_r110=false

case "${TARGET}" in
  r20)
    need_r20=true
    ;;
  r56)
    need_r56=true
    ;;
  r110)
    need_r110=true
    ;;
  primary)
    need_r20=true
    need_r56=true
    ;;
  all)
    need_r20=true
    need_r56=true
    need_r110=true
    ;;
  *)
    echo "ERROR: unknown target '${TARGET}'. Use one of: r20|r56|r110|primary|all" >&2
    exit 1
    ;;
esac

if [[ "${need_r20}" == "true" ]]; then
  if ! cache_ready "${RUNS_DIR}/acts_r20_c10"; then
    run_with_summary "cache_r20_c10" "${PYTHON_BIN} 'src/cache_activations.py' --ckpt 'runs/backbone_r20_c10/best.pt' --dataset cifar10 --out 'runs/acts_r20_c10' --device cuda:${G0}"
  else
    echo "SKIP cache_r20_c10: cache is complete at ${RUNS_DIR}/acts_r20_c10"
  fi
fi

if [[ "${need_r56}" == "true" ]]; then
  if ! cache_ready "${RUNS_DIR}/acts_r56_c100"; then
    run_with_summary "cache_r56_c100" "${PYTHON_BIN} 'src/cache_activations.py' --ckpt 'runs/backbone_r56_c100/best.pt' --dataset cifar100 --out 'runs/acts_r56_c100' --device cuda:${G0}"
  else
    echo "SKIP cache_r56_c100: cache is complete at ${RUNS_DIR}/acts_r56_c100"
  fi
fi

if [[ "${need_r110}" == "true" ]]; then
  if ! cache_ready "${RUNS_DIR}/acts_r110_c100"; then
    run_with_summary "cache_r110_c100" "${PYTHON_BIN} 'src/cache_activations.py' --ckpt 'runs/backbone_r110_c100/best.pt' --dataset cifar100 --out 'runs/acts_r110_c100' --device cuda:${G1}"
  else
    echo "SKIP cache_r110_c100: cache is complete at ${RUNS_DIR}/acts_r110_c100"
  fi
fi

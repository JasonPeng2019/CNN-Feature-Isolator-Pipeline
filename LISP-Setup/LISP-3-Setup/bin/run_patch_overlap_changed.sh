#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/lib/common_changed.sh"
setup_runtime_env
gpus_csv="${GPU_LIST_PRIMARY}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  gpus_csv="0"
fi
run_with_summary_changed "patch_overlap" "${PYTHON_BIN} 'scripts/run_grid_changed.py' --phase 'patch_overlap' --out_root 'runs/phase1_patch_overlap_changed' --gpus '${gpus_csv}'"

#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/lib/common.sh"
setup_runtime_env

if [[ ! -d "${RUNS_DIR}/acts_r56_c100" ]]; then
  echo "ERROR: runs/acts_r56_c100 missing; cannot run taxonomy" >&2
  exit 1
fi

IFS=',' read -r -a GPUS <<< "${GPU_LIST_PRIMARY}"
G0="${GPUS[0]}"
run_with_summary "taxonomy_q4q5" "${PYTHON_BIN} 'src/taxonomy.py' --device cuda:${G0} --out 'runs/taxonomy'"

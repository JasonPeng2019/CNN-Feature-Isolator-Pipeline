#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/lib/common.sh"
setup_runtime_env

if ! cache_ready "${RUNS_DIR}/acts_r56_c100"; then
  echo "acts_r56_c100 missing; generating cache first"
  "${LISP_3_SETUP}/bin/cache_acts.sh" r56
fi

run_grid_with_resume "seeds" "runs/seeds" "${GPU_LIST_PRIMARY}"

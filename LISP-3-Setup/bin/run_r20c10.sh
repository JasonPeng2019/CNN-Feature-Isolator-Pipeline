#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/lib/common.sh"
setup_runtime_env

if ! cache_ready "${RUNS_DIR}/acts_r20_c10"; then
  echo "acts_r20_c10 missing; generating cache first"
  "${LISP_3_SETUP}/bin/cache_acts.sh" r20
fi

run_grid_with_resume "r20c10" "runs/r20c10" "${GPU_LIST_PRIMARY}"

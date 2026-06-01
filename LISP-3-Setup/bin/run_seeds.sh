#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/lib/common.sh"
setup_runtime_env

run_grid_with_resume "seeds" "runs/seeds" "${GPU_LIST_PRIMARY}"

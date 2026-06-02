#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/lib/common_changed.sh"
setup_runtime_env
gpu="$(gpu_csv_to_first "${GPU_LIST_PRIMARY}")"
run_with_summary_changed "phase3_raw_only" "${PYTHON_BIN} 'src/train_chain_changed.py' --carrier_mode raw --bptt --device 'cuda:${gpu}' --out 'runs/phase3b_raw_only_changed'"

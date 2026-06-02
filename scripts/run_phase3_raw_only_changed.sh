#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source env.sh

python src/train_chain_changed.py \
  --carrier_mode raw \
  --bptt \
  --device "${1:-cuda:0}" \
  --out runs/phase3b_raw_only_changed

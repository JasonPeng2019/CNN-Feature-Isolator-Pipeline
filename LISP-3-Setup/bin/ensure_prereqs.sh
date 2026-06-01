#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/lib/common.sh"
setup_runtime_env

missing=0
for f in \
  "${RUNS_DIR}/backbone_r56_c100/best.pt" \
  "${RUNS_DIR}/backbone_r20_c10/best.pt" \
  "${RUNS_DIR}/backbone_r110_c100/best.pt" \
  "${RUNS_DIR}/phase2/E1_Q2_f0.05/sae.pt" \
  "${RUNS_DIR}/phase2/E1_Q3_f0.05/sae.pt" \
  "${RUNS_DIR}/phase2/E1_Q4_f0.05/sae.pt" \
  "${RUNS_DIR}/phase2/E1_Q5_f0.05/sae.pt"; do
  if [[ ! -e "$f" ]]; then
    echo "MISSING prerequisite: $f"
    missing=1
  else
    echo "OK prerequisite: $f"
  fi
done

if [[ ${missing} -ne 0 ]]; then
  echo "Prerequisites missing; fix before running blocked experiments." >&2
  exit 1
fi

echo "Prerequisite check passed."

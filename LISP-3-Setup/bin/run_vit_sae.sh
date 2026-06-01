#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/lib/common.sh"
setup_runtime_env

if [[ "${VIT_ENABLE}" != "true" ]]; then
  echo "VIT_ENABLE=${VIT_ENABLE}; skipping vit_sae run"
  exit 0
fi

if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import timm
PY
then
  echo "ERROR: timm is missing. Install via: ${PYTHON_BIN} -m pip install timm" >&2
  if [[ "${INSTALL_MISSING_DEPS}" == "true" ]]; then
    run_with_summary "install_timm" "${PYTHON_BIN} -m pip install timm"
  else
    exit 1
  fi
fi

IFS=',' read -r -a GPUS <<< "${GPU_LIST_PRIMARY}"
G0="${GPUS[0]}"
run_with_summary "vit_sae" "${PYTHON_BIN} 'src/vit_sae.py' --device cuda:${G0} --out 'runs/vit_sae'"

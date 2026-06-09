#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/lib/common.sh"
setup_runtime_env

cat <<MSG
Bootstrap suggestions for this cluster:
1) Ensure Python packages:
   ${PYTHON_BIN} -m pip install --upgrade pip
   ${PYTHON_BIN} -m pip install torch torchvision timm numpy matplotlib
2) Verify CUDA visibility:
   nvidia-smi
   ${PYTHON_BIN} - <<'PY'
import torch
print(torch.cuda.is_available(), torch.cuda.device_count(), torch.version.cuda)
PY
3) Optional: copy/create .env from .env.example and set GPU_LIST_PRIMARY/overrides.
MSG

if [[ "${INSTALL_MISSING_DEPS}" == "true" ]]; then
  run_with_summary "bootstrap_pip_install" "${PYTHON_BIN} -m pip install --upgrade pip && ${PYTHON_BIN} -m pip install torch torchvision timm numpy matplotlib"
else
  echo "INSTALL_MISSING_DEPS=false; no packages installed."
fi

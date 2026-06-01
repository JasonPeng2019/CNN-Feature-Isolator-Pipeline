#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/lib/common.sh"
setup_runtime_env

echo "LISP_3_SETUP=${LISP_3_SETUP}"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "RUNS_DIR=${RUNS_DIR}"
echo "GPU_LIST_PRIMARY=${GPU_LIST_PRIMARY}"

echo "\n[GPU / Driver]"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "WARNING: nvidia-smi not found"
fi

echo "\n[Python / Torch]"
"${PYTHON_BIN}" - <<'PY'
import importlib, json
mods=['torch','torchvision','numpy','matplotlib','timm']
out={}
for m in mods:
    try:
        mod=importlib.import_module(m)
        out[m]=getattr(mod,'__version__','?')
    except Exception as e:
        out[m]=f"MISSING ({e.__class__.__name__})"
print(json.dumps(out, indent=2))
try:
    import torch
    print('torch.cuda.is_available =', torch.cuda.is_available())
    print('torch.version.cuda =', torch.version.cuda)
    print('device_count =', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
except Exception as e:
    print('torch cuda probe failed:', e)
PY

echo "\n[cuDNN override]"
if [[ -d "${PROJECT_ROOT}/.cudnn12/nvidia/cudnn/lib" ]]; then
  echo "FOUND local cudnn override at ${PROJECT_ROOT}/.cudnn12/nvidia/cudnn/lib"
else
  echo "MISSING local cudnn override (.cudnn12). This is OK if your cluster torch/cudnn stack is healthy."
fi

echo "\n[Prereq checkpoints]"
for f in \
  "${RUNS_DIR}/backbone_r56_c100/best.pt" \
  "${RUNS_DIR}/backbone_r20_c10/best.pt" \
  "${RUNS_DIR}/backbone_r110_c100/best.pt" \
  "${RUNS_DIR}/phase2/E1_Q5_f0.05/sae.pt" \
  "${RUNS_DIR}/winner_saes/Q1/sae.pt"; do
  if [[ -e "$f" ]]; then echo "OK $f"; else echo "MISSING $f"; fi
done

echo "\n[Blocked-artifact expectations]"
for d in "${RUNS_DIR}/acts_r20_c10" "${RUNS_DIR}/acts_r110_c100" "${RUNS_DIR}/taxonomy" "${RUNS_DIR}/vit_sae"; do
  if [[ -e "$d" ]]; then echo "PRESENT $d"; else echo "MISSING $d"; fi
done

echo "\n[No hardcoded host paths check]"
hits="$(grep -R --line-number --exclude='preflight.sh' "/scratch/\\|/home/" "${LISP_3_SETUP}/bin" "${LISP_3_SETUP}/README.md" "${LISP_3_SETUP}/.env.example" 2>/dev/null || true)"
if [[ -n "${hits}" ]]; then
  echo "${hits}"
  echo "WARNING: potential host-specific path found above"
else
  echo "OK: no host-specific absolute paths detected in setup scripts/docs"
fi

#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/lib/common.sh"
setup_runtime_env
: "${CONTINUE_ON_ERROR:=true}"

run_step() {
  local name="$1"
  local script="$2"
  echo "\n=== STEP: ${name} ==="
  if ! "${script}"; then
    echo "STEP FAILED: ${name}" >&2
    if [[ "${CONTINUE_ON_ERROR}" != "true" ]]; then
      echo "CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR}; stopping." >&2
      exit 1
    fi
  fi
}

run_step "preflight" "${LISP_3_SETUP}/bin/preflight.sh"
run_step "ensure_prereqs" "${LISP_3_SETUP}/bin/ensure_prereqs.sh"
run_step "cache_acts" "${LISP_3_SETUP}/bin/cache_acts.sh"
run_step "r20c10" "${LISP_3_SETUP}/bin/run_r20c10.sh"
run_step "seeds" "${LISP_3_SETUP}/bin/run_seeds.sh"
run_step "taxonomy" "${LISP_3_SETUP}/bin/run_taxonomy.sh"
run_step "r110_recon" "${LISP_3_SETUP}/bin/run_r110_recon.sh"

if [[ "${VIT_ENABLE}" == "true" ]]; then
  run_step "vit_sae" "${LISP_3_SETUP}/bin/run_vit_sae.sh"
else
  echo "\n=== STEP: vit_sae skipped (VIT_ENABLE=${VIT_ENABLE}) ==="
fi

echo "\nAll requested blocked experiment steps finished."
echo "Launcher logs: ${LOG_DIR}"
echo "Project tier logs: ${TIER_LOG_DIR}"

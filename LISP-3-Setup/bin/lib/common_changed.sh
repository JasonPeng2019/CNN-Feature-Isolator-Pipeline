#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

changed_log_file_for() {
  local name="$1"
  local suffix="_changed"
  if [[ "${LOG_TIMESTAMP}" == "true" ]]; then
    suffix="${suffix}_$(date +%Y%m%d_%H%M%S)"
  fi
  echo "${LOG_DIR}/${name}${suffix}.log"
}

changed_summary_file_for() {
  local name="$1"
  echo "${LOG_DIR}/${name}_changed_summary.json"
}

run_with_summary_changed() {
  local name="$1"
  local cmd="$2"
  local logf
  logf="$(changed_log_file_for "${name}")"
  local summaryf
  summaryf="$(changed_summary_file_for "${name}")"
  local start_ts
  start_ts="$(ts_now)"
  echo "[$(ts_now)] ${name}: ${cmd}"
  echo "log: ${logf}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "DRY_RUN=true; command not executed" | tee -a "${logf}"
    write_summary_json "${summaryf}" "${name}" "${start_ts}" "$(ts_now)" 0 "${cmd}"
    return 0
  fi
  set +e
  bash -lc "cd '${PROJECT_ROOT}' && ${cmd}" >"${logf}" 2>&1
  local rc=$?
  set -e
  write_summary_json "${summaryf}" "${name}" "${start_ts}" "$(ts_now)" "${rc}" "${cmd}"
  if [[ ${rc} -ne 0 ]]; then
    echo "[$(ts_now)] ${name} failed rc=${rc}. See ${logf}" >&2
    return ${rc}
  fi
  echo "[$(ts_now)] ${name} completed"
}

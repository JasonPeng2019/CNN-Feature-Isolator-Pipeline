#!/usr/bin/env bash
set -euo pipefail

# Canonical anchor variable required by the setup design.
export LISP_3_SETUP="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PROJECT_ROOT="$(cd "${LISP_3_SETUP}/.." && pwd)"
export RUNS_DIR="${PROJECT_ROOT}/runs"
export SCRIPTS_DIR="${PROJECT_ROOT}/scripts"
export SRC_DIR="${PROJECT_ROOT}/src"
export LOG_DIR="${LISP_3_SETUP}/logs"
export TIER_LOG_DIR="${RUNS_DIR}/tier_rest"

mkdir -p "${LOG_DIR}" "${TIER_LOG_DIR}"

load_env() {
  local env_file="${LISP_3_SETUP}/.env"
  if [[ -f "${env_file}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${env_file}"
    set +a
  fi

  : "${GPU_LIST_PRIMARY:=0,1}"
  : "${PYTHON_BIN:=python}"
  : "${USE_LOCAL_CUDNN_OVERRIDE:=auto}"
  : "${INSTALL_MISSING_DEPS:=false}"
  : "${VIT_ENABLE:=true}"
  : "${RERUN_FAILED_ONLY:=false}"
  : "${LOG_TIMESTAMP:=true}"
  : "${DRY_RUN:=false}"
}

setup_runtime_env() {
  load_env

  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES
  fi

  local cudnn_dir="${PROJECT_ROOT}/.cudnn12/nvidia/cudnn/lib"
  case "${USE_LOCAL_CUDNN_OVERRIDE}" in
    force)
      if [[ -d "${cudnn_dir}" ]]; then
        export LD_LIBRARY_PATH="${cudnn_dir}:${LD_LIBRARY_PATH:-}"
      else
        echo "ERROR: USE_LOCAL_CUDNN_OVERRIDE=force but ${cudnn_dir} missing" >&2
        return 1
      fi
      ;;
    auto)
      if [[ -d "${cudnn_dir}" ]]; then
        export LD_LIBRARY_PATH="${cudnn_dir}:${LD_LIBRARY_PATH:-}"
      fi
      ;;
    off)
      ;;
    *)
      echo "ERROR: USE_LOCAL_CUDNN_OVERRIDE must be auto|force|off" >&2
      return 1
      ;;
  esac
}

ts_now() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log_file_for() {
  local name="$1"
  local suffix=""
  if [[ "${LOG_TIMESTAMP}" == "true" ]]; then
    suffix="_$(date +%Y%m%d_%H%M%S)"
  fi
  echo "${LOG_DIR}/${name}${suffix}.log"
}

summary_file_for() {
  local name="$1"
  echo "${LOG_DIR}/${name}_summary.json"
}

write_summary_json() {
  local summary_file="$1"
  local name="$2"
  local start_ts="$3"
  local end_ts="$4"
  local rc="$5"
  local cmd="$6"

  cat > "${summary_file}" <<JSON
{
  "name": "${name}",
  "start": "${start_ts}",
  "end": "${end_ts}",
  "rc": ${rc},
  "command": $(printf '%s' "${cmd}" | python -c 'import json,sys; print(json.dumps(sys.stdin.read()))')
}
JSON
}

run_with_summary() {
  local name="$1"
  local cmd="$2"
  local logf
  logf="$(log_file_for "${name}")"
  local summaryf
  summaryf="$(summary_file_for "${name}")"
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

require_path() {
  local p="$1"
  if [[ ! -e "${p}" ]]; then
    echo "MISSING: ${p}" >&2
    return 1
  fi
}

cache_ready() {
  local d="$1"
  local q
  local split
  if [[ ! -d "${d}" ]]; then
    return 1
  fi
  if [[ ! -f "${d}/norm_stats.json" ]]; then
    return 1
  fi
  for split in train test; do
    for q in Q1 Q2 Q3 Q4 Q5; do
      if [[ ! -f "${d}/${split}_${q}.pt" ]]; then
        return 1
      fi
    done
    if [[ ! -f "${d}/${split}_logits.pt" ]]; then
      return 1
    fi
    if [[ ! -f "${d}/${split}_labels.pt" ]]; then
      return 1
    fi
  done
  return 0
}

gpu_csv_to_first() {
  local csv="$1"
  echo "${csv%%,*}"
}

# Optionally rerun only failed jobs from run_grid output.
run_grid_with_resume() {
  local phase="$1"
  local out_root_rel="$2"
  local gpus_csv="$3"

  local out_root_abs="${PROJECT_ROOT}/${out_root_rel}"
  local grid_summary="${out_root_abs}/grid_summary.json"

  if [[ "${RERUN_FAILED_ONLY}" != "true" || ! -f "${grid_summary}" ]]; then
    run_with_summary "grid_${phase}" "${PYTHON_BIN} 'scripts/run_grid.py' --phase '${phase}' --out_root '${out_root_rel}' --gpus '${gpus_csv}'"
    return
  fi

  echo "RERUN_FAILED_ONLY=true and ${grid_summary} exists; rerunning failed jobs only"
  local tmp_jobs
  tmp_jobs="$(mktemp)"
  ${PYTHON_BIN} - <<PY > "${tmp_jobs}"
import json
from pathlib import Path
p = Path(${grid_summary@Q})
data = json.loads(p.read_text())
for row in data:
    if not row.get("ok", False):
        print(json.dumps(row["job"], sort_keys=True))
PY

  local -a gpu_arr
  IFS=',' read -r -a gpu_arr <<< "${gpus_csv}"
  local i=0
  local failed_any=0

  while IFS= read -r job_json; do
    [[ -z "${job_json}" ]] && continue
    local gpu="${gpu_arr[$((i % ${#gpu_arr[@]}))]}"
    i=$((i+1))
    local cmd
    cmd="${PYTHON_BIN} - <<'PY'\nimport json, subprocess\njob=json.loads(${job_json@Q})\nargs=[]\nfor k,v in job.items():\n    args += [f'--{k}', str(v)]\ncmd=['${PYTHON_BIN}', 'src/train_sae.py','--device','cuda:${gpu}', *args]\nprint('EXEC', ' '.join(cmd), flush=True)\nsubprocess.check_call(cmd, cwd='${PROJECT_ROOT}')\nPY"
    if ! run_with_summary "rerun_${phase}_$((i))" "${cmd}"; then
      failed_any=1
    fi
  done < "${tmp_jobs}"

  rm -f "${tmp_jobs}"
  if [[ ${failed_any} -ne 0 ]]; then
    return 1
  fi
}

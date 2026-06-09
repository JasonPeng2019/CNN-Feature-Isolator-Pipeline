#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 <queue_session_name> <gpu_id> <wait_for_session> <launcher_path> [more_launchers...]" >&2
  exit 1
fi

QUEUE_SESSION="$1"
GPU_ID="$2"
WAIT_FOR_SESSION="$3"
shift 3
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

if tmux has-session -t "${QUEUE_SESSION}" 2>/dev/null; then
  echo "tmux queue session already exists: ${QUEUE_SESSION}" >&2
  exit 1
fi

launchers=("$@")
printf -v launchers_joined '%q ' "${launchers[@]}"

cmd=$(cat <<'SH'
ROOT_PLACEHOLDER=__ROOT__
GPU_PLACEHOLDER=__GPU__
WAIT_PLACEHOLDER=__WAIT__
LAUNCHERS_PLACEHOLDER=(__LAUNCHERS__)

wait_for_gpu_idle() {
  local gpu_id="$1"
  while true; do
    local pids
    pids="$(nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader,nounits 2>/dev/null | awk -F',' -v target="$gpu_id" '
      BEGIN {
        bus["0"]="00000000:1A:00.0";
        bus["1"]="00000000:68:00.0";
      }
      {
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1);
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2);
        if ($1 == bus[target]) print $2;
      }')"
    if [[ -z "${pids}" ]]; then
      break
    fi
    sleep 30
  done
}

while tmux has-session -t "${WAIT_PLACEHOLDER}" 2>/dev/null; do
  sleep 30
done

wait_for_gpu_idle "${GPU_PLACEHOLDER}"

cd "${ROOT_PLACEHOLDER}"
source "${ROOT_PLACEHOLDER}/env.sh"

for launcher in "${LAUNCHERS_PLACEHOLDER[@]}"; do
  export CUDA_VISIBLE_DEVICES="${GPU_PLACEHOLDER}"
  "${launcher}"
done
SH
)

cmd="${cmd/__ROOT__/${ROOT//\//\\/}}"
cmd="${cmd/__GPU__/${GPU_ID}}"
cmd="${cmd/__WAIT__/${WAIT_FOR_SESSION}}"
cmd="${cmd/__LAUNCHERS__/${launchers_joined}}"

tmux new-session -d -s "${QUEUE_SESSION}" "bash -lc ${cmd@Q}"
echo "started queued tmux session ${QUEUE_SESSION} on gpu ${GPU_ID}, waiting for ${WAIT_FOR_SESSION}"

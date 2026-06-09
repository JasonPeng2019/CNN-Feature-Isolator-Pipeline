#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <session_name> <gpu_id> <launcher_path>" >&2
  exit 1
fi

SESSION_NAME="$1"
GPU_ID="$2"
LAUNCHER_PATH="$3"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}" >&2
  exit 1
fi

cmd="cd '${ROOT}' && export CUDA_VISIBLE_DEVICES='${GPU_ID}' && source '${ROOT}/env.sh' && '${LAUNCHER_PATH}'"
tmux new-session -d -s "${SESSION_NAME}" "bash -lc ${cmd@Q}"
echo "started tmux session ${SESSION_NAME} on gpu ${GPU_ID}"

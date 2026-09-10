#!/usr/bin/env bash
# Execute a static text queue serially and durably record command output, exit
# status, and interruptions. Each command has a separate session so the runner
# can reap its whole process group if the runner receives a termination signal.
set -u -o pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <queue-file> <status-log>" >&2
  exit 2
fi

QUEUE_FILE="$1"
STATUS_LOG="$2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "$QUEUE_FILE" ]] || { echo "queue file not found: $QUEUE_FILE" >&2; exit 2; }
mkdir -p "$(dirname "$STATUS_LOG")"
COMMAND_LOG_DIR="${STATUS_LOG}.commands"
mkdir -p "$COMMAND_LOG_DIR"
cd "$ROOT"
source "$ROOT/env.sh"

child_pid=''
child_log=''
current_command=''
current_line=0
current_active=false

write_status() {
  printf '%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$STATUS_LOG"
}

terminate_active_child() {
  [[ -n "$child_pid" ]] || return 0
  if kill -0 "$child_pid" 2>/dev/null; then
    # `setsid` makes child_pid the process-group leader on intended Linux hosts.
    kill -TERM -- "-$child_pid" 2>/dev/null || kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
}

on_signal() {
  local signal=$1 code=1
  case "$signal" in
    HUP) code=129 ;;
    INT) code=130 ;;
    TERM) code=143 ;;
  esac
  trap - HUP INT TERM
  if [[ "$current_active" == true ]]; then
    write_status "INTERRUPTED signal=$signal line=$current_line child_pid=$child_pid log=$child_log command=$current_command"
    terminate_active_child
    write_status "EXIT=$code interrupted=true line=$current_line command=$current_command"
    current_active=false
  else
    write_status "INTERRUPTED signal=$signal idle=true"
  fi
  exit "$code"
}

on_exit() {
  local code=$?
  if [[ "$current_active" == true ]]; then
    write_status "ABANDONED exit=$code line=$current_line child_pid=$child_pid log=$child_log command=$current_command"
  fi
  write_status "QUEUE_EXIT=$code"
}

trap 'on_signal HUP' HUP
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM
trap on_exit EXIT
write_status "RUNNER_START pid=$$ queue=$QUEUE_FILE command_log_dir=$COMMAND_LOG_DIR"

while IFS= read -r command || [[ -n "$command" ]]; do
  [[ -z "${command//[[:space:]]/}" || "$command" == \#* ]] && continue
  current_line=$((current_line + 1))
  current_command="$command"
  child_log="$COMMAND_LOG_DIR/line-$(printf '%05d' "$current_line").log"
  current_active=true
  write_status "START line=$current_line log=$child_log command=$current_command"
  if command -v setsid >/dev/null 2>&1; then
    setsid bash -lc "$current_command" >"$child_log" 2>&1 &
  else
    bash -lc "$current_command" >"$child_log" 2>&1 &
  fi
  child_pid=$!
  write_status "PROCESS_STARTED line=$current_line child_pid=$child_pid log=$child_log"
  wait "$child_pid"
  status=$?
  write_status "EXIT=$status line=$current_line log=$child_log command=$current_command"
  child_pid=''
  current_active=false
done < "$QUEUE_FILE"

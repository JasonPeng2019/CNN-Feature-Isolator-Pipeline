#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
"${HERE}/tmux_queue_changed.sh" \
  patch_queue_changed \
  "${1:-0}" \
  "${2:-rf_fair_changed}" \
  "${HERE}/run_patch_overlap_changed.sh" \
  "${HERE}/run_patch_depth_changed.sh" \
  "${HERE}/run_patch_disjoint_changed.sh"

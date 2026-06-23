#!/usr/bin/env bash
# Source before any GPU run in this project.
# The global env has nvidia-cudnn-cu13 (9.19, CUDA-13 ABI) shadowing the cu128
# torch wheel -> CUDNN_STATUS_NOT_INITIALIZED. We preload a project-local cu12
# cuDNN (9.10) that matches torch 2.10+cu128. Global env left untouched.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="$HERE/.cudnn12/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
if [ -d "$HERE/.pydeps_min" ]; then
  export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$HERE/.pydeps_min"
fi

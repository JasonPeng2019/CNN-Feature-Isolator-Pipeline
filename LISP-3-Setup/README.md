# LISP-3-Setup

Portable blocked-experiment launcher package for `CNN-SAE`.

This package is designed to run on current or future GPU clusters with minimal edits.
All paths are derived from one anchor variable: `LISP_3_SETUP`.

## Location assumption

This folder must live at:

- `CNN-SAE/LISP-3-Setup`

`LISP_3_SETUP` resolves to this folder at runtime; `PROJECT_ROOT` is resolved as `../`.

## Quick start

```bash
cd CNN-SAE/LISP-3-Setup
cp .env.example .env
# edit .env if needed

# 1) verify environment
bin/preflight.sh

# 2) optional bootstrap hints / installs
bin/bootstrap_env.sh

# 3) run primary blocked grids (2 GPUs)
bin/run_r20c10.sh
bin/run_seeds.sh

# 4) run additional blocked experiments
bin/run_taxonomy.sh
bin/run_r110_recon.sh
bin/run_vit_sae.sh

# or run ordered full pipeline
bin/run_all_blocked.sh
```

## Primary targets

- `run_r20c10.sh`
- `run_seeds.sh`

These are the two heavy blocked grids and default priorities for 2-GPU operation.

## Config (`.env`)

Key variables:

- `GPU_LIST_PRIMARY` (default `0,1`)
- `PYTHON_BIN` (default `python`)
- `USE_LOCAL_CUDNN_OVERRIDE` (`auto|force|off`)
- `INSTALL_MISSING_DEPS` (`true|false`)
- `VIT_ENABLE` (`true|false`)
- `RERUN_FAILED_ONLY` (`true|false`)
- `LOG_TIMESTAMP` (`true|false`)
- `DRY_RUN` (`true|false`)
- `CONTINUE_ON_ERROR` (`true|false`)

## Resume mode

When `RERUN_FAILED_ONLY=true`, `run_r20c10.sh` and `run_seeds.sh` will read existing
`runs/*/grid_summary.json` and rerun only failed jobs by reconstructing job commands.

## Logging

Launcher logs and per-step summary JSON files are written to:

- `LISP-3-Setup/logs`

Project-native logs still go to:

- `runs/tier_rest`

## Notes for this current node

Current preflight should report:

- 2 GPUs available (`TITAN RTX`)
- CUDA usable
- `timm` missing (gates `run_vit_sae.sh` unless installed)
- `.cudnn12` override missing (acceptable if system CUDA stack is healthy)
- `runs/acts_r20_c10` and `runs/acts_r110_c100` missing (created by `cache_acts.sh`)

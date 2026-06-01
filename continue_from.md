# Continue-From Runbook (2-GPU Blocked Experiments)

This file is a handoff for moving to a new server with larger storage and resuming the blocked experiment plan from `CNN-SAE/LISP-3-Setup`.

## Goal on the new node

Run the two primary blocked experiments on 2 GPUs:

1. `r20c10` grid (`scripts/run_grid.py --phase r20c10`)
2. `seeds` grid (`scripts/run_grid.py --phase seeds`)

Then run the remaining blocked wrappers:

1. `taxonomy` (`src/taxonomy.py`)
2. `r110_recon` (Q1..Q5 field/global@5% on r110/c100)
3. `vit_sae` (`src/vit_sae.py`, optional, gated by `timm`)

## What these experiments use

### Primary experiment 1: `r20c10`

- Dataset: `cifar10`
- Activation cache: `runs/acts_r20_c10`
- Backbone checkpoint: `runs/backbone_r20_c10/best.pt`
- Jobs: 15 SAE trainings (`Q1..Q5` x fractions `0.02, 0.05, 0.10`)
- Launcher: `LISP-3-Setup/bin/run_r20c10.sh`

### Primary experiment 2: `seeds`

- Dataset: `cifar100` (default path in `train_sae.py` / `run_grid.py`)
- Activation cache: `runs/acts_r56_c100`
- Backbone checkpoint: `runs/backbone_r56_c100/best.pt`
- Jobs: 15 SAE trainings (`Q1..Q5` x seeds `0,1,2`, field/global@5%)
- Launcher: `LISP-3-Setup/bin/run_seeds.sh`

## Storage planning before launch

Expected cache size per CIFAR cache (`acts_*`) is roughly 5.9 GB because cached tensors include:

- `train_Q1..Q5.pt`, `test_Q1..Q5.pt` (fp16 activations)
- `train_logits.pt`, `test_logits.pt`
- `train_labels.pt`, `test_labels.pt`
- `norm_stats.json`

For just the 2 primary runs, budget for:

- `acts_r20_c10` ~5.9 GB
- `acts_r56_c100` ~5.9 GB
- SAE outputs/logs typically much smaller than caches

If you also run `r110_recon`, add:

- `acts_r110_c100` ~5.9 GB

## What to validate before running anything

Run:

```bash
cd CNN-SAE/LISP-3-Setup
cp .env.example .env
bin/preflight.sh
```

Confirm all of the following:

1. `nvidia-smi` shows your intended GPUs.
2. `torch.cuda.is_available()` is `True`.
3. Required Python modules are present (`torch`, `torchvision`, `numpy`, `matplotlib`; `timm` only required for `vit_sae`).
4. Backbone checkpoints exist:
   - `runs/backbone_r20_c10/best.pt`
   - `runs/backbone_r56_c100/best.pt`
   - `runs/backbone_r110_c100/best.pt` (needed only for `r110_recon`)
5. Winner/prior artifacts expected by downstream wrappers exist:
   - `runs/phase2/E1_Q5_f0.05/sae.pt`
   - `runs/winner_saes/Q1/sae.pt`

## Required script fixes before production run

These are important for predictable behavior on a fresh node.

### Fix 1: avoid unnecessary `r110` cache when running only primaries

Current behavior:

- `bin/run_r20c10.sh` calls `bin/cache_acts.sh` if `acts_r20_c10` is missing.
- `bin/cache_acts.sh` currently creates both:
  - `runs/acts_r20_c10`
  - `runs/acts_r110_c100`

Impact:

- Pulls in extra ~5.9 GB even when you only want `r20c10 + seeds`.

Change needed:

- Split cache builder into targeted modes, for example:
  - `cache_acts.sh --target r20`
  - `cache_acts.sh --target r56`
  - `cache_acts.sh --target r110`
- Update `run_r20c10.sh` to request only `r20`.
- Update `run_r110_recon.sh` to request only `r110`.

### Fix 2: make `run_seeds.sh` robust on clean servers

Current behavior:

- `run_seeds.sh` launches seeds grid directly and assumes `runs/acts_r56_c100` already exists.

Impact:

- On a fresh server, seeds run fails if `acts_r56_c100` was not restored/generated.

Change needed:

- Add pre-check in `run_seeds.sh`:
  - If `runs/acts_r56_c100` missing, generate it via `src/cache_activations.py` with r56/c100 args.

### Fix 3: align `run_all_blocked.sh` with storage-aware policy

Current behavior:

- `run_all_blocked.sh` always invokes `cache_acts.sh` in step 3, which currently builds multi-cache.

Change needed:

- Update orchestration so cache steps are explicit and minimal by experiment order.

## Suggested execution order on new server (2 GPUs)

From `CNN-SAE/LISP-3-Setup`:

```bash
bin/preflight.sh
bin/ensure_prereqs.sh

# primary priority
bin/run_r20c10.sh
bin/run_seeds.sh

# additional blocked wrappers
bin/run_taxonomy.sh
bin/run_r110_recon.sh
bin/run_vit_sae.sh   # optional; requires timm
```

Use `DRY_RUN=true` in `.env` first to print commands without launching.

## Other blocked experiments: what they require

### `taxonomy` (`bin/run_taxonomy.sh`)

- Requires `runs/acts_r56_c100` to exist.
- Uses one GPU (`cuda:<first GPU from GPU_LIST_PRIMARY>`).
- Writes to `runs/taxonomy`.

If missing prerequisite:

- Build `acts_r56_c100` first with `src/cache_activations.py --ckpt runs/backbone_r56_c100/best.pt --dataset cifar100 --out runs/acts_r56_c100`.

### `r110_recon` (`bin/run_r110_recon.sh`)

- Requires:
  - `runs/backbone_r110_c100/best.pt`
  - `runs/acts_r110_c100`
- Trains Q1..Q5 field/global@5% sequentially (GPU assignment alternates across the 2 selected GPUs).
- Writes outputs to `runs/r110/F_Q*`.

If missing prerequisite:

- Build `acts_r110_c100` first (or let the wrapper build it, after Fix 1).

### `vit_sae` (`bin/run_vit_sae.sh`)

- Requires `timm` import.
- Gated by `VIT_ENABLE` in `.env`.
- Uses first GPU in `GPU_LIST_PRIMARY`.
- Writes to `runs/vit_sae`.

If `timm` missing:

- Install manually (`python -m pip install timm`) or set `INSTALL_MISSING_DEPS=true` and re-run wrapper.

## Environment knobs to set in `.env`

Minimum recommended:

```bash
GPU_LIST_PRIMARY="0,1"
PYTHON_BIN="python"
USE_LOCAL_CUDNN_OVERRIDE="auto"
RERUN_FAILED_ONLY="false"
DRY_RUN="false"
CONTINUE_ON_ERROR="true"
VIT_ENABLE="false"   # set true only when timm is installed
```

For restarts:

- Set `RERUN_FAILED_ONLY=true` to rerun only failed grid cells for `r20c10` and `seeds` when prior `grid_summary.json` exists.

## Validation after each run

### After `r20c10` / `seeds`

Check:

1. `runs/<phase>/grid_summary.json` exists.
2. Launcher summary JSON exists in `LISP-3-Setup/logs/*_summary.json`.
3. Per-job result files exist under run folder (`result.json`, `sae.pt`).

### After wrappers

Check:

1. `runs/taxonomy` populated.
2. `runs/r110/F_Q1`..`runs/r110/F_Q5` populated with `result.json` and `sae.pt`.
3. `runs/vit_sae` populated if enabled.

## Known state from the previous node (for context)

Observed preflight state on prior node:

1. 2x TITAN RTX detected.
2. CUDA available.
3. `timm` missing.
4. `.cudnn12` local override missing (not fatal if system stack is healthy).
5. `runs/acts_r20_c10` and `runs/acts_r110_c100` missing.

Treat this as a baseline expectation when reproducing on a new cluster.

# Continue-From-2: Secondary Cluster Plan for Remaining Blocked Experiments

This handoff assumes:

1. Primary node (2 GPUs) is running only `r20c10` and `seeds`.
2. A separate GPU cluster will run the remaining blocked experiments without interfering with the primary node.

## Scope for the secondary cluster

Run only:

1. `taxonomy` (`bin/run_taxonomy.sh`)
2. `r110_recon` (`bin/run_r110_recon.sh`)
3. `vit_sae` (`bin/run_vit_sae.sh`, optional)

Do not run `run_r20c10.sh` or `run_seeds.sh` on the secondary cluster unless you intentionally want duplicate primary results.

## Code baseline required on the secondary cluster

Use a repo copy that includes the launcher fixes now present in `LISP-3-Setup`:

1. `cache_acts.sh` supports targets: `r20|r56|r110|primary|all`.
2. `run_seeds.sh` auto-builds `acts_r56_c100` if missing/incomplete.
3. `run_r20c10.sh` and `run_r110_recon.sh` call targeted cache builds.
4. Cache completeness checks are enforced via `cache_ready` in `bin/lib/common.sh`.
5. `preflight.sh` reports cache state as `READY` / `INCOMPLETE` / `MISSING`.

If your secondary cluster copy is older, sync these files from the updated tree before running.

## Secondary cluster setup steps

From `CNN-SAE/LISP-3-Setup`:

```bash
cp .env.example .env
```

Set `.env` for the secondary cluster:

```bash
GPU_LIST_PRIMARY="0,1"   # or the two GPUs assigned on that cluster
PYTHON_BIN="python"
USE_LOCAL_CUDNN_OVERRIDE="auto"
INSTALL_MISSING_DEPS="false"
VIT_ENABLE="false"       # turn true only after timm is installed
RERUN_FAILED_ONLY="false"
DRY_RUN="false"
CONTINUE_ON_ERROR="true"
```

Run environment checks:

```bash
bin/preflight.sh
bin/ensure_prereqs.sh
```

## Prerequisites by blocked experiment

### A) `taxonomy`

Requires:

1. `runs/backbone_r56_c100/best.pt`
2. Complete `runs/acts_r56_c100` cache

If cache is missing/incomplete:

```bash
bin/cache_acts.sh r56
```

Then run:

```bash
bin/run_taxonomy.sh
```

### B) `r110_recon`

Requires:

1. `runs/backbone_r110_c100/best.pt`
2. Complete `runs/acts_r110_c100` cache

If cache is missing/incomplete:

```bash
bin/cache_acts.sh r110
```

Then run:

```bash
bin/run_r110_recon.sh
```

### C) `vit_sae` (optional)

Requires:

1. Python package `timm`
2. ViT script dependencies in the environment

Install if needed:

```bash
python -m pip install timm
```

Then run:

```bash
bin/run_vit_sae.sh
```

## Recommended execution order on secondary cluster

```bash
bin/preflight.sh
bin/ensure_prereqs.sh
bin/cache_acts.sh r56
bin/run_taxonomy.sh
bin/cache_acts.sh r110
bin/run_r110_recon.sh
# optional
bin/run_vit_sae.sh
```

This avoids generating unneeded caches and keeps storage predictable.

## Running in parallel with the primary node

To avoid result collisions:

1. Use separate physical repo clones per cluster (recommended), or
2. Use distinct `runs` roots via bind-mount/symlink strategy per cluster.

Do not point two clusters at the same writable `runs/` directory simultaneously.

If you need to merge outputs later, copy only finished run folders:

1. `runs/taxonomy`
2. `runs/r110`
3. `runs/vit_sae` (if used)
4. `LISP-3-Setup/logs/*summary.json` for auditability

## What to validate after each blocked run

### Taxonomy validation

Check:

1. `runs/taxonomy` exists and contains output artifacts.
2. Launcher summary JSON exists in `LISP-3-Setup/logs`.
3. Command exit code is `0`.

### r110_recon validation

Check:

1. `runs/r110/F_Q1` .. `runs/r110/F_Q5` each contain `result.json` and `sae.pt`.
2. No section failed silently (scan launcher summaries and logs).

### vit_sae validation

Check:

1. `runs/vit_sae` created with expected result artifacts.
2. `timm` import succeeded in preflight or launcher log.

## Failure handling and resume

1. Grid-style resume (`RERUN_FAILED_ONLY=true`) applies to `run_r20c10.sh` and `run_seeds.sh`.
2. `r110_recon` and `taxonomy` are wrapper runs; rerun failed sections manually by re-invoking the wrapper or direct command.
3. Keep `LOG_TIMESTAMP=true` so repeated attempts remain traceable.

## Minimal command block for the secondary cluster

```bash
cd CNN-SAE/LISP-3-Setup
cp .env.example .env
bin/preflight.sh
bin/ensure_prereqs.sh
bin/cache_acts.sh r56
bin/run_taxonomy.sh
bin/cache_acts.sh r110
bin/run_r110_recon.sh
# optional
bin/run_vit_sae.sh
```

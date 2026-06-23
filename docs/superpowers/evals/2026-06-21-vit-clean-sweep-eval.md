# Vit Clean Sweep Eval

## Source Inputs

- Spec path:
  - `ViT_Proj/CNN-SAE/docs/superpowers/specs/2026-06-21-vit-clean-sweep.md`
- Build path:
  - `ViT_Proj/CNN-SAE/docs/superpowers/builds/2026-06-21-vit-clean-sweep-build.md`
- Code surfaces:
  - `ViT_Proj/CNN-SAE/src/vit_sae.py`
  - `ViT_Proj/CNN-SAE/scripts/run_vit_grid.py`

## Acceptance Traceability

| Acceptance criterion | Check type | Evidence |
| --- | --- | --- |
| Launcher script can execute configurable block/fraction/seed jobs with one output dir per run | smoke | tiny launcher run plus `grid_summary.json` |
| `src/vit_sae.py` supports deterministic seeded runs and smoke-friendly CLI controls | smoke | direct CLI invocation with subset/seed args and resulting JSON metadata |
| A smoke-scale single run completes and writes a valid `vit_result.json` | smoke | parsed JSON under a temporary smoke output dir |
| A smoke-scale launcher invocation over at least two jobs completes and writes a sweep summary artifact | smoke | launcher log and `grid_summary.json` with expected job count |
| The process bundle validates successfully | smoke | `validate_process_bundle.py` output |

## Smoke Tests

1. Launcher CLI import/help
   - Purpose:
     - ensure the new launcher is syntactically valid and exposes expected arguments without starting jobs
   - Command or harness:
     - `python scripts/run_vit_grid.py --help`
   - Expected result:
     - exits `0` and prints usage text
   - Failure meaning:
     - syntax/import/path failure in the launcher

2. Tiny direct ViT SAE run
   - Purpose:
     - verify that the hardened single-run path can execute end-to-end on a tiny subset and write `vit_result.json`
   - Command or harness:
     - run `src/vit_sae.py` with one epoch, tiny train/val subsets, and a scratch output directory
   - Expected result:
     - exits `0`, writes `vit_result.json`, and that JSON contains `pred_agree`, `kl`, `frac_retained`, `model`, `block`, `K`
   - Failure meaning:
     - bug in dataset handling, model loading, token-grid training, or output writing

3. Tiny launcher sweep
   - Purpose:
     - verify multi-job orchestration, output-path generation, and sweep-summary writing
   - Command or harness:
     - run `scripts/run_vit_grid.py` over a 2-job grid with smoke-scale subset sizes
   - Expected result:
     - exits `0`, each child run writes `vit_result.json`, and launcher writes `grid_summary.json`
   - Failure meaning:
     - bug in job expansion, child command construction, GPU assignment, or summary writing

4. Bundle validation
   - Purpose:
     - ensure the spec/build/eval triplet is coherent
   - Command or harness:
     - `python .codex/skills/process-eval/scripts/validate_process_bundle.py --spec ... --build ... --eval ...`
   - Expected result:
     - exits `0`
   - Failure meaning:
     - process docs disagree on slug, structure, or required fields

## Bench Tests

1. Full sweep dry specification
   - Metric:
     - launcher covers requested block/fraction/seed combinations
   - Baseline:
     - recommended grid from the project handoff
   - Threshold:
     - job count matches combinatorics exactly
   - Dataset/input slice:
     - full intended sweep configuration, not executed in this task
   - Command:
     - manual later invocation of `scripts/run_vit_grid.py` with full grid values

2. Resource-envelope check
   - Metric:
     - one run directory per job and bounded one-job-per-GPU launcher behavior
   - Baseline:
     - existing `scripts/run_grid.py` launcher style on the CIFAR side
   - Threshold:
     - no path collisions, one child process per GPU slot
   - Dataset/input slice:
     - smoke grid now, full grid later
   - Command:
     - launcher smoke now, tmux full sweep later

## Leak And Validity Review

- Split/group integrity checks:
  - confirm train subset is drawn only from Imagenette train and val subset only from Imagenette val
- Preprocessing-fit boundaries:
  - confirm token-grid normalization stats are fit on train subset only
- Temporal/causal boundaries:
  - not applicable in the autoregressive sense; this is a static image reconstruction path
- Selection-vs-evaluation separation:
  - smoke runs may confirm run-readiness only; later hyperparameter selection from sweep outputs must be documented explicitly rather than treated as untouched evaluation

## Pre-Run Leak Gate

- [ ] Split boundaries are explicit
- [ ] Fit-on-train-only surfaces are listed
- [ ] Intended evaluation-corpus exceptions are documented
- [ ] No confirmed leak remains unresolved

## Evidence Capture

- Logs to save:
  - direct smoke stdout/stderr
  - launcher smoke stdout/stderr
- Metrics files to save:
  - `vit_result.json` for each smoke run
  - `grid_summary.json` for launcher smoke
- Figures/tables to save:
  - none required for smoke readiness
- Where results should be written:
  - scratch smoke directories under `runs/`

## Automation Status

- Tests to implement now:
  - all smoke tests above
- Tests to defer:
  - full sweep execution
  - comparative summary plots across the full grid
- Manual checks:
  - inspect the resulting JSON files for sensible keys and non-empty metrics

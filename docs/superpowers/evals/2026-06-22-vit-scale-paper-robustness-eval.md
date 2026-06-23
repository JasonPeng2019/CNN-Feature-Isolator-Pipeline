# Vit Scale Paper Robustness Eval

## Source Inputs

- Spec path:
  - `ViT_Proj/CNN-SAE/docs/superpowers/specs/2026-06-22-vit-scale-paper-robustness.md`
- Build path:
  - `ViT_Proj/CNN-SAE/docs/superpowers/builds/2026-06-22-vit-scale-paper-robustness-build.md`
- Code surfaces:
  - `ViT_Proj/CNN-SAE/src/vit_sae.py`
  - `ViT_Proj/CNN-SAE/scripts/run_vit_scale_grid.py`
  - `ViT_Proj/CNN-SAE/docs/CURRENT_PROGRESS_AND_TODO.md`

## Acceptance Traceability

| Acceptance criterion | Check type | Evidence |
| --- | --- | --- |
| `src/vit_sae.py` supports Imagenette and generic ImageFolder-style dataset inputs | smoke | tiny direct run through generic dataset path |
| scale launcher supports multi-model job grids with unique outputs and summary JSON | smoke | tiny launcher run plus `grid_summary.json` |
| smoke-scale runs succeed without executing full large runs | smoke | direct and launcher smoke outputs |
| docs clearly record completed work and next larger runs | manual | updated handoff doc contents |
| scale-up process bundle validates successfully | smoke | `validate_process_bundle.py` output |

## Smoke Tests

1. Direct generic dataset smoke
   - Purpose:
     - verify that `src/vit_sae.py` can use a generic dataset path rather than only the `Imagenette` class
   - Command or harness:
     - run `src/vit_sae.py` with `--dataset imagefolder` pointing at the local Imagenette train/val directories, on tiny subsets
   - Expected result:
     - exits `0` and writes `vit_result.json`
   - Failure meaning:
     - generic dataset loading or split wiring is broken

2. Scale launcher smoke
   - Purpose:
     - verify model-grid orchestration and dataset parameter threading
   - Command or harness:
     - run `scripts/run_vit_scale_grid.py` on a tiny 2-job grid using local Imagenette paths
   - Expected result:
     - exits `0`, child jobs write `vit_result.json`, launcher writes `grid_summary.json`
   - Failure meaning:
     - launcher/grid/path handling is broken

3. Bundle validation
   - Purpose:
     - ensure the scale-up process docs are coherent
   - Command or harness:
     - `validate_process_bundle.py --spec ... --build ... --eval ...`
   - Expected result:
     - exits `0`
   - Failure meaning:
     - doc inconsistency or missing required sections

## Bench Tests

1. Larger-model dry handoff
   - Metric:
     - launcher can express `vit_base_patch16_224` runs with explicit batch-size control
   - Baseline:
     - current `vit_small_patch16_224` launcher support
   - Threshold:
     - generated command surfaces include model and dataset settings without code edits
   - Dataset/input slice:
     - deferred real larger-model run
   - Command:
     - manual later invocation of `scripts/run_vit_scale_grid.py`

2. Larger-dataset dry handoff
   - Metric:
     - direct run and launcher can target full split sizes or large subset sizes
   - Baseline:
     - previous small fixed-subset flow
   - Threshold:
     - `ntrain=-1` or `nval=-1` style full-split usage works by interface
   - Dataset/input slice:
     - deferred real larger dataset root
   - Command:
     - manual later invocation with larger dataset directories

## Leak And Validity Review

- Split/group integrity checks:
  - confirm generic dataset loader keeps train and val directories distinct
- Preprocessing-fit boundaries:
  - confirm train-only token-grid mean/std in generic dataset mode
- Temporal/causal boundaries:
  - not applicable in the autoregressive sense
- Selection-vs-evaluation separation:
  - post-run ranking is allowed for model selection only if documented as such, not as untouched evaluation

## Pre-Run Leak Gate

- [x] Split boundaries are explicit
- [x] Fit-on-train-only surfaces are listed
- [x] Intended evaluation-corpus exceptions are documented
- [x] No confirmed leak remains unresolved

## Evidence Capture

- Logs to save:
  - direct generic smoke log
  - scale-launcher smoke log
- Metrics files to save:
  - direct smoke `vit_result.json`
  - scale-launcher `grid_summary.json`
- Figures/tables to save:
  - none required for run-readiness
- Where results should be written:
  - scratch smoke directories under `runs/`

## Automation Status

- Tests to implement now:
  - all smoke tests above
- Tests to defer:
  - real larger-model and larger-dataset full runs
- Manual checks:
  - inspect updated docs for completed-work and next-run clarity

# Vit Sweep Timm Import Eval

## Source Inputs

- Spec path:
  - `ViT_Proj/CNN-SAE/docs/superpowers/specs/2026-06-21-vit-sweep-timm-import.md`
- Build path:
  - `ViT_Proj/CNN-SAE/docs/superpowers/builds/2026-06-21-vit-sweep-timm-import-build.md`
- Code surfaces:
  - `ViT_Proj/CNN-SAE/env.sh`
  - `ViT_Proj/CNN-SAE/src/vit_sae.py`

## Acceptance Traceability

| Acceptance criterion | Check type | Evidence |
| --- | --- | --- |
| `import timm` succeeds after `source env.sh` | smoke | shell command exit `0` |
| tiny `src/vit_sae.py` smoke run completes successfully | smoke | smoke run output and `vit_result.json` |
| fix does not require switching away from the repo's default `python` command | manual | `which python` plus successful smoke on `python` |

## Smoke Tests

1. Import check after env sourcing
   - Purpose:
     - verify the environment fix exposes `timm`
   - Command or harness:
     - `source env.sh && python -c "import timm; print(timm.__file__)"`
   - Expected result:
     - exits `0` and prints a path under the intended Python search path
   - Failure meaning:
     - environment fix incomplete or path ordering wrong

2. Tiny ViT smoke rerun
   - Purpose:
     - verify the original blocker is removed for the actual run path
   - Command or harness:
     - rerun the tiny `src/vit_sae.py` smoke command from the main sweep work
   - Expected result:
     - exits `0` and writes `vit_result.json`
   - Failure meaning:
     - more dependency or runtime bugs remain after the import fix

## Bench Tests

1. Main-sweep compatibility handoff
   - Metric:
     - same `python` command used by the main sweep can now import `timm`
   - Baseline:
     - pre-fix failure with `ModuleNotFoundError`
   - Threshold:
     - import succeeds consistently in a fresh shell after `source env.sh`
   - Dataset/input slice:
     - none
   - Command:
     - fresh-shell import smoke

2. Environment containment
   - Metric:
     - dependency exposure stays project-local
   - Baseline:
     - current project-local cuDNN handling style
   - Threshold:
     - fix does not require global system package mutation to function
   - Dataset/input slice:
     - none
   - Command:
     - manual inspection of `env.sh` and local dependency path

## Leak And Validity Review

- Split/group integrity checks:
  - unchanged
- Preprocessing-fit boundaries:
  - unchanged
- Temporal/causal boundaries:
  - unchanged
- Selection-vs-evaluation separation:
  - unchanged

## Pre-Run Leak Gate

- [x] Split boundaries are explicit
- [x] Fit-on-train-only surfaces are listed
- [x] Intended evaluation-corpus exceptions are documented
- [x] No confirmed leak remains unresolved

## Evidence Capture

- Logs to save:
  - import check output
  - tiny smoke rerun output
- Metrics files to save:
  - `runs/smoke_vit_single/vit_result.json`
- Figures/tables to save:
  - none
- Where results should be written:
  - normal project run directories

## Automation Status

- Tests to implement now:
  - both smoke tests above
- Tests to defer:
  - none for this bug fix
- Manual checks:
  - inspect that `python` remains the default command used after `source env.sh`

# Vit Clean Sweep Build

## Target

- Spec path:
  - `ViT_Proj/CNN-SAE/docs/superpowers/specs/2026-06-21-vit-clean-sweep.md`
- Goal:
  - make the ViT-small Imagenette transfer path run-ready for a configurable sweep over blocks, retained fractions, and seeds, without executing the full sweep in this task
- Primary repo area:
  - `ViT_Proj/CNN-SAE/src/` and `ViT_Proj/CNN-SAE/scripts/`

## Files And Surfaces

- Files to create:
  - `ViT_Proj/CNN-SAE/scripts/run_vit_grid.py`
- Files to edit:
  - `ViT_Proj/CNN-SAE/src/vit_sae.py`
  - process docs in `docs/superpowers/{specs,builds,evals}/`
- Files to inspect but not edit:
  - `ViT_Proj/CNN-SAE/README.md`
  - `ViT_Proj/CNN-SAE/TODO.md`
  - `ViT_Proj/CNN-SAE/docs/PHASE4_RESULTS.md`
  - `ViT_Proj/CNN-SAE/src/sae.py`
  - `ViT_Proj/CNN-SAE/src/masks.py`
  - `ViT_Proj/CNN-SAE/src/eval.py`
  - `ViT_Proj/CNN-SAE/scripts/run_grid.py`

## Data And Interface Contracts

- Inputs:
  - Imagenette train split for SAE fitting
  - Imagenette val split for downstream self-consistency evaluation
  - CLI-provided block/fraction/seed grids
- Outputs:
  - per-run directories with `vit_result.json`
  - launcher-level `grid_summary.json`
- Schema/interface constraints:
  - `vit_result.json` must remain machine-readable JSON with stable headline metrics
  - launcher jobs must expand deterministically from CLI grid parameters
  - per-run output paths must be unique across block/fraction/seed combinations
- Backward-compatibility expectations:
  - direct invocation of `src/vit_sae.py` with its old defaults should still work
  - existing documented `runs/vit_sae/vit_result.json` shape should not be broken

## Build Phases

1. Harden single-run ViT CLI
   - Purpose:
     - make `src/vit_sae.py` deterministic, subset-friendly, and launchable for both smoke tests and future sweeps
   - Prerequisites:
     - inspect current CLI, result schema, and dataset assumptions
   - Planned edits:
     - add seed handling
     - add configurable train/val subset sizes and batch sizes if missing
     - add optional metadata fields useful for sweep summaries
     - preserve direct single-run behavior
   - Verification checkpoint:
     - a tiny direct smoke run produces `vit_result.json` successfully

2. Build sweep launcher
   - Purpose:
     - create a multi-job orchestration script for block/fraction/seed sweeps that is ready for tmux-backed full execution later
   - Prerequisites:
     - single-run CLI is stable and smoke-tested
   - Planned edits:
     - add `scripts/run_vit_grid.py`
     - implement grid expansion, GPU assignment, logging, and summary writing
     - support cheap smoke-scale overrides from CLI
   - Verification checkpoint:
     - a tiny launcher run over at least two jobs finishes and writes a valid `grid_summary.json`

3. Repair and finalize
   - Purpose:
     - close bugs found by smoke tests and leave the path run-ready
   - Prerequisites:
     - smoke outputs from phases 1 and 2
   - Planned edits:
     - targeted fixes only
     - if bugs appear, write a compact bug-fix process bundle before repairing
   - Verification checkpoint:
     - all planned smoke tests pass and the process bundle validates cleanly

## Commands And Execution Notes

- Setup commands:
  - `cd /jumbo/lisp/f003x5w/ViT_Proj/CNN-SAE`
  - `source env.sh`
- Build/run commands:
  - direct smoke: `python src/vit_sae.py ...`
  - launcher smoke: `python scripts/run_vit_grid.py ...`
  - later full sweep: same launcher with the full block/fraction/seed grid and user-chosen GPUs/tmux
- Data or asset preparation:
  - rely on existing local `data/imagenette2-160`
  - avoid triggering a full-data full-epoch run during this task

## Risk Review

- Leak-sensitive steps:
  - train/val subset selection in `src/vit_sae.py`
  - normalization statistics fit location
- Benchmark-sensitive steps:
  - changing result schema used by existing docs
  - introducing launcher defaults that silently diverge from the intended sweep design
- Likely failure modes:
  - `timm` model/config mismatch
  - output-path collisions across jobs
  - missing seed propagation
  - summary file not written if a child job fails
- Escalation conditions:
  - evidence that validation data is used in training tensors
  - a required interface break to existing result consumers
  - smoke failures that imply deeper architectural conflict rather than a local bug

## Leak Review

- Split boundaries to preserve:
  - fit on Imagenette train subset only
  - evaluate on Imagenette val subset only
- Fit-on-train-only requirements:
  - any token-grid normalization meant to represent the training code path must be computed from the selected train subset
- Intended evaluation-corpus exceptions:
  - validation self-consistency is the intended evaluation corpus for this ViT transfer family
- What would count as a confirmed leak here:
  - code evidence that val examples enter the train subset or train-side fitted normalization statistics

## Ready To Implement

- [x] Inputs are readable
- [x] File touch set is known
- [x] Verification checkpoints are defined
- [x] Eval handoff requirements are listed

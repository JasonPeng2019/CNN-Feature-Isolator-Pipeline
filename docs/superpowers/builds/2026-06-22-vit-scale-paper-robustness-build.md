# Vit Scale Paper Robustness Build

## Target

- Spec path:
  - `ViT_Proj/CNN-SAE/docs/superpowers/specs/2026-06-22-vit-scale-paper-robustness.md`
- Goal:
  - make the ViT path run-ready for larger models and larger Imagenette/ImageNet-style dataset runs without executing the full large runs now
- Primary repo area:
  - `ViT_Proj/CNN-SAE/src/`
  - `ViT_Proj/CNN-SAE/scripts/`
  - `ViT_Proj/CNN-SAE/docs/`

## Files And Surfaces

- Files to create:
  - `ViT_Proj/CNN-SAE/scripts/run_vit_scale_grid.py`
- Files to edit:
  - `ViT_Proj/CNN-SAE/src/vit_sae.py`
  - `ViT_Proj/CNN-SAE/docs/CURRENT_PROGRESS_AND_TODO.md`
- Files to inspect but not edit:
  - `ViT_Proj/CNN-SAE/scripts/run_vit_grid.py`
  - `ViT_Proj/CNN-SAE/TODO.md`
  - `ViT_Proj/CNN-SAE/docs/PHASE4_RESULTS.md`

## Data And Interface Contracts

- Inputs:
  - CLI-selected `timm` model name
  - either Imagenette split selection or ImageFolder train/val directories
  - subset sizes or full-split selection for train/val
- Outputs:
  - per-run directories with `vit_result.json` and `sae.pt`
  - scale-launcher `grid_summary.json`
- Schema/interface constraints:
  - `vit_result.json` must continue to expose the current headline metrics
  - output metadata must include dataset/model identity for larger runs
  - launcher output paths must encode model and dataset settings clearly enough to avoid collisions
- Backward-compatibility expectations:
  - existing `run_vit_grid.py` continues to work
  - direct Imagenette usage remains the default for `src/vit_sae.py`

## Build Phases

1. Extend direct ViT dataset/model interface
   - Purpose:
     - make the single-run path usable for larger dataset roots and future larger models
   - Prerequisites:
     - inspect current Imagenette-only dataset assumptions
   - Planned edits:
     - add dataset-kind and dataset-root/train-dir/val-dir controls
     - support full-split selection, e.g. `-1` meaning all available examples
     - record dataset metadata in result outputs
   - Verification checkpoint:
     - tiny ImageFolder-style run against the local Imagenette directory succeeds

2. Build scale launcher
   - Purpose:
     - orchestrate larger-model and larger-dataset jobs without changing the single-run code manually
   - Prerequisites:
     - direct path supports generic datasets and larger model names
   - Planned edits:
     - add `scripts/run_vit_scale_grid.py`
     - expand over model/block/frac/seed
     - thread dataset parameters through to child jobs
     - write a summary JSON similar to the existing small-scale launcher
   - Verification checkpoint:
     - tiny launcher smoke over at least two jobs succeeds

3. Update docs and validate
   - Purpose:
     - record what is already complete and what larger runs should happen next
   - Prerequisites:
     - code path verified by smoke tests
   - Planned edits:
     - update the handoff doc with the completed sweep and new scale-up status
     - validate the scale-up process bundle
   - Verification checkpoint:
     - process bundle validation passes

## Commands And Execution Notes

- Setup commands:
  - `cd /jumbo/lisp/f003x5w/ViT_Proj/CNN-SAE`
  - `source env.sh`
- Build/run commands:
  - direct generic run: `python src/vit_sae.py ...`
  - scale launcher: `python scripts/run_vit_scale_grid.py ...`
  - later full larger run: same launcher with user-chosen large dataset roots and GPU lists
- Data or asset preparation:
  - local Imagenette can be reused as an ImageFolder-style smoke dataset
  - real larger datasets are expected to be provided later by the user

## Risk Review

- Leak-sensitive steps:
  - mapping generic train/val directories to the correct split boundaries
  - fitting token normalization on train-only subsets
- Benchmark-sensitive steps:
  - larger model defaults may require reduced batch sizes
  - path naming must not mix results from different models/datasets
- Likely failure modes:
  - generic dataset path mismatch
  - pretrained weight download hiccups for larger models
  - launcher output-path collisions
- Escalation conditions:
  - evidence of train/val directory confusion
  - unavoidable breaking change to current small-scale launcher or result schema

## Leak Review

- Split boundaries to preserve:
  - explicit train directory or train split only for fitting
  - explicit val directory or val split only for downstream evaluation
- Fit-on-train-only requirements:
  - token-grid mean/std remain train-only
- Intended evaluation-corpus exceptions:
  - validation self-consistency remains the intended evaluation posture for this ViT family
- What would count as a confirmed leak here:
  - direct code evidence of val images used in train subset or train-fit normalization

## Ready To Implement

- [x] Inputs are readable
- [x] File touch set is known
- [x] Verification checkpoints are defined
- [x] Eval handoff requirements are listed

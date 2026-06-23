# Vit Sweep Timm Import Build

## Target

- Spec path:
  - `ViT_Proj/CNN-SAE/docs/superpowers/specs/2026-06-21-vit-sweep-timm-import.md`
- Goal:
  - repair the missing-`timm` environment bug so the ViT sweep path runs after `source env.sh`
- Primary repo area:
  - `ViT_Proj/CNN-SAE/env.sh`

## Files And Surfaces

- Files to create:
  - repo-local dependency directory if required, e.g. `ViT_Proj/CNN-SAE/.pydeps/`
- Files to edit:
  - `ViT_Proj/CNN-SAE/env.sh`
- Files to inspect but not edit:
  - `ViT_Proj/CNN-SAE/src/vit_sae.py`
  - active Python environment surfaces

## Data And Interface Contracts

- Inputs:
  - shell environment after `source env.sh`
- Outputs:
  - importable `timm` for project Python
- Schema/interface constraints:
  - `env.sh` must remain source-able from bash
- Backward-compatibility expectations:
  - existing commands that source `env.sh` should keep working

## Build Phases

1. Confirm environment surfaces
   - Purpose:
     - verify where `torch`, `torchvision`, and `timm` are or are not available
   - Prerequisites:
     - none
   - Planned edits:
     - none; inspection only
   - Verification checkpoint:
     - choose one Python surface to preserve and one local dependency strategy

2. Patch project environment setup
   - Purpose:
     - expose repo-local Python dependencies after `source env.sh`
   - Prerequisites:
     - confirmed dependency strategy
   - Planned edits:
     - update `env.sh` to prepend a repo-local dependency path
     - install `timm` into that path
   - Verification checkpoint:
     - `source env.sh && python -c "import timm"` succeeds

3. Re-run blocked smoke path
   - Purpose:
     - prove the import fix unblocks the real ViT run path
   - Prerequisites:
     - import test passing
   - Planned edits:
     - none unless additional dependency issues appear
   - Verification checkpoint:
     - tiny `src/vit_sae.py` smoke run exits `0`

## Commands And Execution Notes

- Setup commands:
  - `cd /jumbo/lisp/f003x5w/ViT_Proj/CNN-SAE`
- Build/run commands:
  - `python -m pip install --target .pydeps timm`
  - `source env.sh && python -c "import timm"`
  - rerun the previously failing tiny `src/vit_sae.py` command
- Data or asset preparation:
  - none

## Risk Review

- Leak-sensitive steps:
  - none
- Benchmark-sensitive steps:
  - environment drift could make later tmux runs differ from current interactive runs
- Likely failure modes:
  - missing transitive dependency after installing `timm`
  - `env.sh` path ordering not applied in subshells
- Escalation conditions:
  - import still fails after repo-local exposure
  - system Python surfaces break after patching `env.sh`

## Leak Review

- Split boundaries to preserve:
  - unchanged
- Fit-on-train-only requirements:
  - unchanged
- Intended evaluation-corpus exceptions:
  - unchanged
- What would count as a confirmed leak here:
  - not applicable for this environment-only repair

## Ready To Implement

- [x] Inputs are readable
- [x] File touch set is known
- [x] Verification checkpoints are defined
- [x] Eval handoff requirements are listed

# Vit Sweep Timm Import Spec

## Final Result

Sourcing `ViT_Proj/CNN-SAE/env.sh` makes the ViT sweep code import and run successfully by exposing a project-local `timm` installation without breaking the existing CUDA/cuDNN setup.

## Context

The first smoke test for the clean ViT sweep failed before model execution with `ModuleNotFoundError: No module named 'timm'`. The current `env.sh` only sets `LD_LIBRARY_PATH` for cuDNN compatibility and does not ensure Python dependencies for the ViT path are visible. The system Python already has `torch` and `torchvision`, but not `timm`; the local `.venv` has `torch` but not `torchvision` or `timm`, so switching blindly to the venv would trade one import failure for another.

This bug blocks every ViT smoke test and any later tmux-run sweep. The fix should keep dependency scope local to the project and preserve the existing GPU setup behavior.

## Scope

- In scope:
  - make the ViT path importable after `source env.sh`
  - keep the dependency repair local to the project
  - document the bug/fix/eval path
- Not in scope:
  - rebuilding the entire Python environment
  - changing the ViT experiment logic itself except for import/run-readiness
  - changing CIFAR-side dependencies

## Acceptance Criteria

1. After `source env.sh`, `python -c "import timm"` exits successfully from `ViT_Proj/CNN-SAE`.
2. The previously failing tiny `src/vit_sae.py` smoke command runs past import and completes successfully.
3. The fix does not require activating a different Python interpreter than the one already used by the repo scripts.

## Constraints

- Technical constraints:
  - preserve the existing cuDNN preload behavior in `env.sh`
  - avoid relying on mutable global state outside the project when a repo-local dependency path works
- Data/split/leak constraints:
  - none beyond preserving the existing train/val split behavior; this bug is import/environment scoped
- Environment/runtime constraints:
  - the fix must work in non-interactive tmux shells that simply `source env.sh`

## Leak Posture

- Intended test/eval-corpus uses:
  - unchanged from the main ViT sweep spec
- Disallowed leak surfaces:
  - none introduced by the import fix itself
- Evidence required before declaring a confirmed leak:
  - not applicable here unless the fix accidentally changes data inputs

## Required Artifacts

- Spec path:
  - `ViT_Proj/CNN-SAE/docs/superpowers/specs/2026-06-21-vit-sweep-timm-import.md`
- Build doc path:
  - `ViT_Proj/CNN-SAE/docs/superpowers/builds/2026-06-21-vit-sweep-timm-import-build.md`
- Eval doc path:
  - `ViT_Proj/CNN-SAE/docs/superpowers/evals/2026-06-21-vit-sweep-timm-import-eval.md`
- Code / config / script outputs:
  - updated `env.sh`
  - project-local Python dependency directory if needed

## Inputs And Dependencies

- Existing files to read first:
  - `ViT_Proj/CNN-SAE/env.sh`
  - `ViT_Proj/CNN-SAE/src/vit_sae.py`
- Upstream artifacts required:
  - working system Python with `torch` and `torchvision`
  - network access or another route to obtain `timm`
- External assumptions:
  - adding a repo-local Python path is acceptable for this project

## Implementation Phases

1. Environment diagnosis: confirm which Python surfaces already have `torch`/`torchvision` and where `timm` is missing.
2. Local dependency exposure: install `timm` into a repo-local path and update `env.sh` so the default Python sees it.
3. Regression verification: rerun the previously failing import and smoke command.

## Evaluation Hooks

- Smoke tests the eval skill must create:
  - `import timm` after `source env.sh`
  - rerun of the tiny `src/vit_sae.py` smoke command
- Bench tests the eval skill must create:
  - none beyond confirming the fix is compatible with the main sweep path
- Metrics or qualitative checks:
  - import success
  - smoke command exit code
  - preservation of existing `env.sh` cuDNN behavior

## Risks And Open Questions

- Risks:
  - `timm` may pull additional dependencies that also need local exposure
  - modifying `env.sh` could accidentally shadow an unrelated user Python path
- Open questions:
  - whether to add a pinned requirements file later for the ViT path
- Reversible defaults:
  - prepend a repo-local dependency directory in `env.sh`

## Build Inputs

- Target files:
  - `ViT_Proj/CNN-SAE/env.sh`
- Interfaces or data contracts to preserve:
  - `source env.sh` remains the one-step setup command before GPU runs
- Sequence constraints:
  - verify import success before rerunning the tiny ViT smoke command

## Evaluation Inputs

- Acceptance criteria to verify:
  - all three acceptance criteria above
- Regression surfaces:
  - existing `env.sh` CUDA/cuDNN setup
  - the tiny ViT smoke run from the main sweep work
- Evidence to capture:
  - import command output
  - smoke run output

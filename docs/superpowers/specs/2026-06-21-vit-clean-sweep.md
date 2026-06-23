# Vit Clean Sweep Spec

## Final Result

A run-ready, smoke-tested ViT-small Imagenette sweep pipeline exists that can launch multi-block, multi-sparsity, multi-seed SAE jobs without executing a full sweep in this task, and it produces standardized per-run artifacts plus sweep summaries suitable for later tmux-driven runs.

## Context

The repo already contains a single executed ViT transfer artifact in `runs/vit_sae/vit_result.json`, produced by `src/vit_sae.py`, but it does not yet provide a clean experiment family around that artifact. The current handoff priority is "Clean ViT Sweep": turn the existing one-off path into a reproducible sweep over blocks, sparsity budgets, and seeds, while keeping the code path light enough for smoke testing now and full runs later under tmux.

This work affects the ViT/Imagenette branch of `ViT_Proj/CNN-SAE`, not the legacy CIFAR grid. The goal is to make the ViT branch run-ready, summary-producing, and robust enough that an agent or user can launch broad parallel sweeps on the available GPUs.

## Scope

- In scope:
  - make `src/vit_sae.py` sweep-friendly and smoke-testable
  - add a launcher script for the ViT block/fraction/seed sweep
  - add sweep artifact conventions and summary generation
  - add smoke coverage that exercises the launch path and one tiny ViT run
  - fix bugs found during smoke testing
- Not in scope:
  - running the full sweep
  - implementing the separate feature-visualization artifact family
  - scaling to ViT-B/16 or ResNet-50
  - changing the CIFAR SAE training/evaluation posture in this task
  - producing final paper figures

## Acceptance Criteria

1. A launcher script can generate and execute ViT-small Imagenette sweep jobs over configurable blocks, retained fractions, and seeds, with one output directory per run.
2. `src/vit_sae.py` supports deterministic seeded runs and exposes enough CLI controls to support fast smoke tests and later full sweeps.
3. A smoke-scale single run completes successfully on local hardware and writes a valid `vit_result.json`.
4. A smoke-scale launcher invocation over at least two jobs completes successfully and writes a sweep summary artifact.
5. The full process bundle for this work validates successfully with the spec, build, and eval docs sharing one coherent slug and acceptance story.

## Constraints

- Technical constraints:
  - preserve the existing core ViT token-grid SAE logic unless a bug requires a targeted change
  - keep launcher behavior non-destructive and local to project outputs
  - prefer small, composable scripts over monolithic orchestration
- Data/split/leak constraints:
  - no training examples may enter validation summary metrics except through explicitly configured training subsets
  - validation metrics must be computed on validation examples only
  - smoke tests may use tiny train/val subsets, but train and val subsets must remain distinct
- Environment/runtime constraints:
  - all GPU work must assume `source env.sh`
  - the machine has many GPUs available, but this task must not start the full sweep
  - smoke tests must stay cheap enough to run interactively in this session

## Leak Posture

- Intended test/eval-corpus uses:
  - Imagenette validation may be used as the downstream self-consistency evaluation corpus for the ViT transfer path
  - smoke tests may report validation self-consistency metrics without claiming external benchmark finality
- Disallowed leak surfaces:
  - mixing validation examples into SAE training subsets
  - fitting normalization statistics on validation data when the same stats are intended to represent train-only code paths
  - using sweep summary outputs to choose hyperparameters inside the same automated run without documenting that choice
- Evidence required before declaring a confirmed leak:
  - direct code-path evidence showing validation examples entering training tensors or fitted train-side statistics
  - not just the presence of a validation read path

## Required Artifacts

- Spec path:
  - `ViT_Proj/CNN-SAE/docs/superpowers/specs/2026-06-21-vit-clean-sweep.md`
- Build doc path:
  - `ViT_Proj/CNN-SAE/docs/superpowers/builds/2026-06-21-vit-clean-sweep-build.md`
- Eval doc path:
  - `ViT_Proj/CNN-SAE/docs/superpowers/evals/2026-06-21-vit-clean-sweep-eval.md`
- Code / config / script outputs:
  - updated `src/vit_sae.py`
  - new ViT sweep launcher under `scripts/`
  - sweep summary JSON artifact
  - smoke-test evidence in command logs and output directories

## Inputs And Dependencies

- Existing files to read first:
  - `ViT_Proj/CNN-SAE/README.md`
  - `ViT_Proj/CNN-SAE/TODO.md`
  - `ViT_Proj/CNN-SAE/docs/PHASE4_RESULTS.md`
  - `ViT_Proj/CNN-SAE/docs/CURRENT_PROGRESS_AND_TODO.md`
  - `ViT_Proj/CNN-SAE/src/vit_sae.py`
  - `ViT_Proj/CNN-SAE/src/sae.py`
  - `ViT_Proj/CNN-SAE/src/masks.py`
- Upstream artifacts required:
  - local Imagenette data under `data/imagenette2-160`
  - working `timm`, PyTorch, and CUDA environment from `env.sh`
- External assumptions:
  - `vit_small_patch16_224` pretrained weights are available through `timm`
  - the user will run the full sweep later via tmux or similar

## Implementation Phases

1. ViT path hardening: add seed and smoke-friendly controls to `src/vit_sae.py`, keep result schema stable, and ensure per-run outputs are self-contained.
2. Sweep orchestration: add a configurable launcher that expands block/fraction/seed grids, runs jobs across specified GPUs, and records run/sweep summaries.
3. Verification and repair: run smoke tests, fix failures, and iterate until the small-run path and launcher path are green.

## Evaluation Hooks

- Smoke tests the eval skill must create:
  - CLI help/import sanity for the new launcher
  - one tiny single-run ViT SAE smoke test
  - one tiny two-job launcher smoke test
- Bench tests the eval skill must create:
  - deferred full sweep command template and expected artifact checks
  - resource-envelope expectations for later tmux execution
- Metrics or qualitative checks:
  - `vit_result.json` presence and parseability
  - sweep summary presence and correct job counts
  - deterministic seed field recorded in outputs

## Risks And Open Questions

- Risks:
  - pretrained `timm` model path or Imagenette transform assumptions may break on smoke
  - current `vit_sae.py` may be too rigid for subset-based smoke testing
  - multi-job launcher orchestration may fail on path handling or summary writing
- Open questions:
  - whether to fit normalization stats on train-only token grids or on the exact train subset for each seed/run
  - whether to add richer per-run metadata now or keep schema minimal
- Reversible defaults:
  - default sweep grid should match the handoff recommendation but remain CLI-overridable
  - summary schema can be additive so future runs do not break

## Build Inputs

- Target files:
  - `src/vit_sae.py`
  - `scripts/` launcher surface
  - process docs created by this bundle
- Interfaces or data contracts to preserve:
  - `vit_result.json` remains JSON and keeps headline metric keys already used in docs
  - per-run output directory contains at least the result JSON
- Sequence constraints:
  - harden single-run path before building the launcher
  - validate a tiny direct run before validating the multi-job launcher

## Evaluation Inputs

- Acceptance criteria to verify:
  - all five acceptance criteria above
- Regression surfaces:
  - existing one-off `src/vit_sae.py` behavior
  - output JSON key stability
  - train/val separation on smoke paths
- Evidence to capture:
  - smoke command lines
  - created output directories
  - resulting JSON summaries

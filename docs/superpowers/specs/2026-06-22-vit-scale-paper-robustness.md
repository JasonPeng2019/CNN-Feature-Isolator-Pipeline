# Vit Scale Paper Robustness Spec

## Final Result

A run-ready scale-up pipeline exists for larger ViT models and larger Imagenette/ImageNet-style datasets, with generic dataset loading, multi-model sweep launchers, and documented commands for bigger paper-robustness runs that the user can execute later on GPUs.

## Context

The project now has a completed ViT-small Imagenette sweep with stable signal, so the next paper-robustness step is not more tiny-scale debugging but scaling the same mechanism to larger models and larger evaluation surfaces. The current ViT code path is still too narrow for that purpose: it is hard-wired to the `Imagenette` dataset class, only launches one model at a time, and does not yet expose generic ImageNet-style dataset directories or scale-specific launcher surfaces.

This work prepares the next stage of experiments without executing the full large runs in this task. It should make it possible to run:

- larger ViT models such as `vit_base_patch16_224`
- larger train/val subsets, including full available validation sets
- generic `ImageFolder` datasets laid out like `train/<class>` and `val/<class>`
- multi-model sweeps via a dedicated launcher

The output of this task should be run-ready infrastructure plus docs that explain what is already done and what the next large runs should be.

## Scope

- In scope:
  - extend the ViT SAE entrypoint to support generic dataset loading
  - support larger subset sizes and full-split usage without code edits
  - add a multi-model scale launcher for paper-robustness runs
  - document recommended large-run commands and output paths
  - update project docs with the completed ViT-small sweep and the new scale-up status
- Not in scope:
  - actually executing the full larger-model sweep
  - implementing ResNet-50/ImageNet CNN-side scaling in this task
  - feature visualization implementation
  - post-run CPU analysis of the completed sweep

## Acceptance Criteria

1. `src/vit_sae.py` can run against either the built-in Imagenette path or a generic ImageNet-style `ImageFolder` dataset specified by directories or dataset root.
2. A dedicated scale launcher can generate and run configurable multi-model ViT jobs, including at least `vit_small_patch16_224` and `vit_base_patch16_224`, with unique per-run output directories and a summary JSON.
3. Smoke-scale runs succeed for the generic dataset path and for the scale launcher without requiring full large runs.
4. Project docs clearly record what has already been completed, what scale-up infrastructure now exists, and what larger paper-robustness runs should be executed next.
5. The scale-up process bundle validates successfully.

## Constraints

- Technical constraints:
  - preserve backward compatibility for the existing `vit_sae.py` and `run_vit_grid.py` flow
  - keep the larger-run setup local to the project and compatible with `source env.sh`
  - prefer additive CLI expansion over breaking interface changes
- Data/split/leak constraints:
  - train examples and validation examples must remain separated by dataset split or directory boundary
  - normalization statistics used for SAE fitting must be fit on the selected train subset only
  - any full-val setting must still use val-only data for downstream evaluation
- Environment/runtime constraints:
  - larger-model paths must be smoke-tested cheaply in this task
  - the final intended runs may be heavy and are explicitly deferred to the user

## Leak Posture

- Intended test/eval-corpus uses:
  - Imagenette validation and future ImageNet-style validation directories are the intended downstream self-consistency evaluation corpora for this ViT transfer family
- Disallowed leak surfaces:
  - pulling train and val images from the same directory tree without explicit split separation
  - computing train-side token normalization statistics from validation examples
  - using post-run sweep ranking to claim untouched evaluation without documenting model selection
- Evidence required before declaring a confirmed leak:
  - direct code-path evidence that validation images enter training tensors or train-fit statistics
  - not merely that both splits live under a shared dataset root

## Required Artifacts

- Spec path:
  - `ViT_Proj/CNN-SAE/docs/superpowers/specs/2026-06-22-vit-scale-paper-robustness.md`
- Build doc path:
  - `ViT_Proj/CNN-SAE/docs/superpowers/builds/2026-06-22-vit-scale-paper-robustness-build.md`
- Eval doc path:
  - `ViT_Proj/CNN-SAE/docs/superpowers/evals/2026-06-22-vit-scale-paper-robustness-eval.md`
- Code / config / script outputs:
  - updated `src/vit_sae.py`
  - new scale launcher script under `scripts/`
  - updated handoff/progress docs in `docs/`

## Inputs And Dependencies

- Existing files to read first:
  - `ViT_Proj/CNN-SAE/src/vit_sae.py`
  - `ViT_Proj/CNN-SAE/scripts/run_vit_grid.py`
  - `ViT_Proj/CNN-SAE/docs/CURRENT_PROGRESS_AND_TODO.md`
  - `ViT_Proj/CNN-SAE/TODO.md`
  - `ViT_Proj/CNN-SAE/docs/PHASE4_RESULTS.md`
- Upstream artifacts required:
  - local Imagenette directory under `data/imagenette2-160`
  - optional future ImageFolder-style dataset roots supplied by the user
  - working `timm`, `torch`, `torchvision`, and `env.sh`
- External assumptions:
  - larger ViT models may require smaller batch sizes than ViT-small
  - the user will decide which large dataset roots are available locally when launching the real runs

## Implementation Phases

1. Generic dataset hardening: make `src/vit_sae.py` accept Imagenette and generic ImageFolder splits while preserving the current default behavior.
2. Scale launcher: add a multi-model launcher that expands over model/block/frac/seed and dataset settings for larger runs.
3. Documentation and verification: update handoff docs with completed sweep status and new scale-up commands, then smoke-test and validate the process bundle.

## Evaluation Hooks

- Smoke tests the eval skill must create:
  - generic dataset CLI/help/import sanity
  - tiny direct run through the ImageFolder-style path
  - tiny scale-launcher run across at least two jobs
- Bench tests the eval skill must create:
  - deferred full large-run command templates for `vit_base_patch16_224`
  - artifact checks for future bigger-dataset runs
- Metrics or qualitative checks:
  - `vit_result.json` and launcher summary presence
  - model and dataset metadata recorded in outputs
  - docs updated with new scale-up status

## Risks And Open Questions

- Risks:
  - larger pretrained model downloads may require network access and more disk/cache use
  - generic ImageFolder assumptions may not match every future dataset exactly
  - larger models may force user-side batch-size tuning
- Open questions:
  - which exact larger dataset beyond Imagenette the user will mount locally first
  - whether a future follow-up should add dataset manifests/config files instead of CLI-only control
- Reversible defaults:
  - keep Imagenette as the default dataset path
  - keep `vit_base_patch16_224` as the first larger-model target
  - keep full-scale launches deferred to the user

## Build Inputs

- Target files:
  - `src/vit_sae.py`
  - `scripts/` scale launcher surface
  - `docs/CURRENT_PROGRESS_AND_TODO.md`
- Interfaces or data contracts to preserve:
  - existing `vit_result.json` headline keys
  - existing `run_vit_grid.py` small-scale launcher behavior
  - `source env.sh` as the setup contract
- Sequence constraints:
  - extend the direct ViT entrypoint before building the scale launcher
  - smoke-test the generic dataset path before the launcher path
  - update docs after the code paths are verified

## Evaluation Inputs

- Acceptance criteria to verify:
  - all five acceptance criteria above
- Regression surfaces:
  - current Imagenette single-model flow
  - result JSON schema
  - train/val split handling
- Evidence to capture:
  - smoke command outputs
  - generated JSON artifacts
  - updated doc references

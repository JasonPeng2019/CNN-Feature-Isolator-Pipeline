# Vit Base Results Analysis Spec

## Final Result

A complete larger-model ViT analysis package exists for the finished `vit_base_imagenette_full` sweep, including recovered summary tables, graph-generation scripts, generated figures, and synchronized updates to the phase docs, markdown report, and LaTeX analysis report.

## Context

The project already has a completed and documented ViT-small Imagenette sweep, plus run-ready scale infrastructure for larger models. The next missing step is to analyze the completed larger-model artifact itself: the `vit_base_patch16_224` Imagenette sweep under `runs/vit_base_imagenette_full/`.

This larger-model run is qualitatively different from the earlier ViT-small sweep. Preliminary recovery shows a very strong stable regime at late block `10`, but multiple mid-block settings suffer seed-fragile collapses. That means the repo now needs more than a single "best result" table. It needs analysis code and figures that make visible:

- stable winner cells
- per-seed instability and collapse patterns
- behavior by requested retained fraction
- realized-versus-requested sparsity drift
- comparison of the larger-model result family against the earlier ViT-small story

This task is therefore both a reporting task and an analysis-tooling task.

## Scope

- In scope:
- recover the completed `vit_base_imagenette_full` result table from run artifacts
- create analysis scripts that summarize the sweep by requested block/fraction/seed and emit graph-rich figure suites
- generate all feasible larger-model figures from the recovered sweep
- update `docs/PHASE4_RESULTS.md`, `report/REPORT.md`, and `analysis/report.tex` with the larger-model results and interpretation
- document the new larger-model analysis process in a validated process bundle
- Not in scope:
- running new full GPU experiments
- treating the partially run `phase4_changed` family-comparison branch as a completed result family
- implementing feature-visualization / mech-interp sparse feature renderings in this task
- changing the underlying `vit_base_imagenette_full` experiment artifacts

## Acceptance Criteria

1. A scriptable analysis path exists for `runs/vit_base_imagenette_full/` that writes machine-readable summaries under a dedicated analysis output directory and groups runs by the requested sweep settings recovered from run directory names.
2. A figure suite is generated for the larger-model sweep that covers, at minimum, per-setting heatmaps, block/fraction aggregate trends, ranking/Pareto views, per-seed traces, and requested-vs-realized retained-fraction diagnostics.
3. The completed larger-model results are added coherently to `docs/PHASE4_RESULTS.md`, `report/REPORT.md`, and `analysis/report.tex`, including best stable settings, gap-to-meaningful-max interpretation, and the instability/collapse story.
4. The generated analysis artifacts are reproducible from committed scripts without re-running the original GPU experiments.
5. The spec/build/eval triplet validates successfully with the process-bundle validator.

## Constraints

- Technical constraints:
  - preserve the existing `vit_result.json` files as read-only inputs
  - prefer additive analysis scripts over one-off shell fragments
  - keep new outputs under the established project surfaces: `runs/.../analysis/`, `report/plots/`, `report/figures_pdf/`, and `analysis/figures/`
- Data/split/leak constraints:
  - do not recompute training-time quantities from raw train/val images; use the completed run artifacts as the primary evidence source
  - when summarizing the sweep, distinguish the requested retained fraction from the realized `frac_retained` recorded in each JSON
  - treat Imagenette validation as the intended self-consistency evaluation corpus for the ViT family, consistent with the existing ViT transfer posture
- Environment/runtime constraints:
  - analysis and figure generation should be CPU-friendly
  - LaTeX compilation must remain optional-but-supported through the existing `analysis/report.tex` path
  - no full GPU reruns are required for acceptance in this task

## Leak Posture

- Intended test/eval-corpus uses:
  - the completed `vit_base_imagenette_full` sweep uses Imagenette validation as the intended downstream self-consistency corpus, matching the earlier ViT-small transfer family
- Disallowed leak surfaces:
  - reinterpreting post-hoc seed/model-selection on the validation corpus as untouched external generalization
  - collapsing requested and realized sparsity settings into one axis when the realized fraction drift materially changes interpretation
  - mixing incomplete `phase4_changed` runs into the completed `vit_base_imagenette_full` evidence
- Evidence required before declaring a confirmed leak:
  - direct proof from code or artifacts that validation examples entered fitting/training statistics for the larger-model sweep
  - static suspicion or post-hoc result instability alone is not enough to call a leak

## Required Artifacts

- Spec path:
  - `ViT_Proj/CNN-SAE/docs/superpowers/specs/2026-06-23-vit-base-results-analysis.md`
- Build doc path:
  - `ViT_Proj/CNN-SAE/docs/superpowers/builds/2026-06-23-vit-base-results-analysis-build.md`
- Eval doc path:
  - `ViT_Proj/CNN-SAE/docs/superpowers/evals/2026-06-23-vit-base-results-analysis-eval.md`
- Code / config / script outputs:
  - a larger-model analysis script under `scripts/`
  - summary CSV/JSON outputs under `runs/vit_base_imagenette_full/analysis/`
  - generated PNG/PDF figures under `report/plots/`, `report/figures_pdf/`, and `analysis/figures/`
  - synchronized result updates in `docs/PHASE4_RESULTS.md`, `report/REPORT.md`, and `analysis/report.tex`

## Inputs And Dependencies

- Existing files to read first:
  - `ViT_Proj/CNN-SAE/runs/vit_base_imagenette_full/*/vit_result.json`
  - `ViT_Proj/CNN-SAE/scripts/make_vit_sweep_figures.py`
  - `ViT_Proj/CNN-SAE/docs/PHASE4_RESULTS.md`
  - `ViT_Proj/CNN-SAE/report/REPORT.md`
  - `ViT_Proj/CNN-SAE/analysis/report.tex`
  - `ViT_Proj/CNN-SAE/docs/superpowers/specs/2026-06-22-vit-scale-paper-robustness.md`
- Upstream artifacts required:
  - completed `runs/vit_base_imagenette_full/` sweep outputs
  - existing project plotting/reporting directories
  - working `source env.sh` analysis environment with matplotlib and LaTeX available
- External assumptions:
  - the larger-model sweep is complete enough to analyze as a finished artifact family
  - the requested sweep settings are most reliably recovered from run directory names rather than the realized `frac_retained` field alone

## Implementation Phases

1. Result recovery: parse `vit_base_imagenette_full`, recover requested sweep settings, compute stable-setting and instability summaries, and validate completeness.
2. Analysis tooling and figures: implement scripts that emit tidy summaries and a graph-rich figure suite covering winners, aggregates, per-seed instability, and fraction-drift diagnostics.
3. Reporting integration: update the phase docs and both report surfaces with the recovered results, new figures, and interpretation of the larger-model behavior.

## Evaluation Hooks

- Smoke tests the eval skill must create:
  - parse-and-summarize smoke test on the completed `vit_base_imagenette_full` artifacts
  - figure-generation smoke test that produces all expected outputs without errors
  - LaTeX report compile sanity after the new larger-model figures are referenced
- Bench tests the eval skill must create:
  - artifact-count and schema checks for the generated `analysis/` summary tables
  - regression checks that the expected larger-model figure suite is fully materialized
  - spot checks for best-setting values and collapse counts against hand-recovered baselines
- Metrics or qualitative checks:
  - best stable `pred_agree`, seed std, and gap to max are reproduced exactly
  - aggregate by-block and by-requested-fraction summaries match the recovered runs
  - generated figures visibly encode the collapse/instability pattern, not only the top winners

## Risks And Open Questions

- Risks:
  - unstable seeds in some larger-model settings may tempt over-aggregation that hides the actual story
  - realized `frac_retained` drift can silently mis-bucket settings if grouped incorrectly
  - LaTeX/report updates may drift from markdown docs if not synchronized carefully
- Open questions:
  - whether the instability pattern points to genuine model dynamics, tie-induced masking pathology, or another implementation-specific failure mode
  - whether later analysis should compare `vit_base_imagenette_full` directly against the incomplete `phase4_changed` family-comparison branch
- Reversible defaults:
  - use requested sweep fraction from directory names as the primary grouping key
  - treat `pred_agree = 1.0000` as the meaningful maximum for ViT self-consistency
  - keep the larger-model analysis script additive rather than refactoring the older ViT-small script immediately

## Build Inputs

- Target files:
  - `scripts/` analysis/figure-generation surface for larger-model ViT runs
  - `docs/PHASE4_RESULTS.md`
  - `report/REPORT.md`
  - `analysis/report.tex`
- Interfaces or data contracts to preserve:
  - existing `vit_result.json` headline keys and semantics
  - existing report figure directories and naming conventions
  - existing `make_vit_sweep_figures.py` output style where practical
- Sequence constraints:
  - recover and validate the result table before writing the narrative claims
  - generate figures before wiring them into the report surfaces
  - compile the LaTeX report only after the figure files are in place

## Evaluation Inputs

- Acceptance criteria to verify:
  - all five acceptance criteria above
- Regression surfaces:
  - the existing ViT-small reporting path
  - report figure directories and LaTeX include paths
  - summary-script reproducibility from completed artifacts
- Evidence to capture:
  - generated summary CSV/JSON files
  - generated figure files
  - updated docs/report excerpts
  - validation output from the process-bundle validator

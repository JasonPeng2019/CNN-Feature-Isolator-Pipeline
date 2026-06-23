# Vit Base Results Analysis Eval

## Source Inputs

- Spec path:
  - `ViT_Proj/CNN-SAE/docs/superpowers/specs/2026-06-23-vit-base-results-analysis.md`
- Build path:
  - `ViT_Proj/CNN-SAE/docs/superpowers/builds/2026-06-23-vit-base-results-analysis-build.md`
- Code surfaces:
  - `scripts/make_vit_scale_figures.py`
  - `docs/PHASE4_RESULTS.md`
  - `report/REPORT.md`
  - `analysis/report.tex`

## Acceptance Traceability

| Acceptance criterion | Check type | Evidence |
| --- | --- | --- |
| Analysis path exists for `vit_base_imagenette_full` with grouping by requested sweep settings | smoke + manual | generated summary files under `runs/vit_base_imagenette_full/analysis/` plus spot-checked row counts |
| Larger-model figure suite covers heatmaps, aggregates, ranking/Pareto, seed instability, and fraction drift | smoke + bench | expected figure files in `report/plots/`, `report/figures_pdf/`, and `analysis/figures/` |
| Docs and reports include coherent larger-model results and interpretation | manual | updated excerpts in `docs/PHASE4_RESULTS.md`, `report/REPORT.md`, and `analysis/report.tex` |
| Analysis artifacts are reproducible from committed scripts | smoke | rerunnable `python scripts/make_vit_scale_figures.py ...` command |
| Process bundle validates successfully | smoke | output of `validate_process_bundle.py` |

## Smoke Tests

1. Larger-model summary recovery
   - Purpose:
     - confirm the script can parse all completed `vit_base_imagenette_full` artifacts and recover the requested sweep grid
   - Command or harness:
     - `python scripts/make_vit_scale_figures.py --run-root runs/vit_base_imagenette_full --tag vit_base_imagenette_full`
   - Expected result:
     - exits successfully and writes summary CSV/JSON outputs under `runs/vit_base_imagenette_full/analysis/`
   - Failure meaning:
     - parsing, grouping, or file-output logic is broken

2. Figure materialization
   - Purpose:
     - ensure the full larger-model figure suite is produced from the recovered summary tables
   - Command or harness:
     - same script invocation as above, followed by file existence checks for the expected figure stems
   - Expected result:
     - all expected figure files exist in PNG and PDF form
   - Failure meaning:
     - one or more plot paths, caption sidecars, or render steps are broken

3. Report compile sanity
   - Purpose:
     - verify the LaTeX analysis report still compiles after wiring in the larger-model figures
   - Command or harness:
     - `cd analysis && pdflatex -interaction=nonstopmode report.tex && pdflatex -interaction=nonstopmode report.tex`
   - Expected result:
     - `analysis/report.pdf` is produced without fatal errors
   - Failure meaning:
     - the report references missing files or contains broken LaTeX after integration

## Bench Tests

1. Best-setting reproduction
   - Metric:
     - recovered best stable `pred_agree`, seed std, and mean KL
   - Baseline:
     - hand-recovered best stable cell `block=10`, requested `12%`, mean `pred_agree ≈ 0.9851`, seed std `≈ 0.0012`, mean `KL ≈ 0.0063`
   - Threshold:
     - exact agreement up to normal float formatting tolerance
   - Dataset/input slice:
     - completed `runs/vit_base_imagenette_full/`
   - Command:
     - summary script plus manual spot-check of the emitted summary JSON/CSV

2. Instability visibility
   - Metric:
     - generated outputs make seed collapse / instability visible for the fragile mid-block settings
   - Baseline:
     - hand-recovered collapses in blocks `4`, `6`, and `8` for several requested fractions
   - Threshold:
     - at least one figure and one written summary explicitly encode the instability story
   - Dataset/input slice:
     - completed `runs/vit_base_imagenette_full/`
   - Command:
     - inspect generated collapse/seed/aggregate figures and updated doc text

## Leak And Validity Review

- Split/group integrity checks:
  - this task does not introduce new splitting; it analyzes completed artifacts only
- Preprocessing-fit boundaries:
  - no new preprocessing is fit in this task
- Temporal/causal boundaries:
  - not applicable beyond preserving the original ViT run interpretation
- Selection-vs-evaluation separation:
  - documentation must distinguish post-hoc sweep winner selection from untouched external evaluation claims

## Pre-Run Leak Gate

- [x] Split boundaries are explicit
- [x] Fit-on-train-only surfaces are listed
- [x] Intended evaluation-corpus exceptions are documented
- [x] No confirmed leak remains unresolved

## Evidence Capture

- Logs to save:
  - summary-script stdout when run manually
  - LaTeX compile output
- Metrics files to save:
  - emitted CSV/JSON summaries under `runs/vit_base_imagenette_full/analysis/`
- Figures/tables to save:
  - the larger-model figure suite plus any summary tables added to docs/reports
- Where results should be written:
  - `runs/vit_base_imagenette_full/analysis/`
  - `report/plots/`
  - `report/figures_pdf/`
  - `analysis/figures/`

## Automation Status

- Tests to implement now:
  - summary-script execution
  - file existence checks
  - LaTeX compile sanity
- Tests to defer:
  - deeper causal diagnosis of why some larger-model seeds collapse
- Manual checks:
  - confirm updated narratives match emitted summary values
  - inspect the figures to ensure they genuinely surface the instability story

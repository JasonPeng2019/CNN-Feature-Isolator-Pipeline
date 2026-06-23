# Vit Base Results Analysis Build

## Target

- Spec path:
  - `ViT_Proj/CNN-SAE/docs/superpowers/specs/2026-06-23-vit-base-results-analysis.md`
- Goal:
  - recover, visualize, and report the completed `vit_base_imagenette_full` larger-model sweep in a reproducible way
- Primary repo area:
  - `ViT_Proj/CNN-SAE/runs/vit_base_imagenette_full/`

## Files And Surfaces

- Files to create:
  - `scripts/make_vit_scale_figures.py`
  - `runs/vit_base_imagenette_full/analysis/*`
  - larger-model figure files under `report/plots/`, `report/figures_pdf/`, and `analysis/figures/`
- Files to edit:
  - `docs/PHASE4_RESULTS.md`
  - `report/REPORT.md`
  - `analysis/report.tex`
  - optionally `docs/CURRENT_PROGRESS_AND_TODO.md` if the larger-model artifact status needs synchronization
- Files to inspect but not edit:
  - `runs/vit_base_imagenette_full/*/vit_result.json`
  - `scripts/make_vit_sweep_figures.py`
  - `analysis/report.pdf`
  - `docs/superpowers/specs/2026-06-22-vit-scale-paper-robustness.md`

## Data And Interface Contracts

- Inputs:
  - completed `vit_result.json` files under `runs/vit_base_imagenette_full/`
  - run directory names encoding requested `block`, `frac`, and `seed`
- Outputs:
  - machine-readable larger-model summary tables and JSON
  - a larger-model figure suite
  - updated narrative/reporting surfaces
- Schema/interface constraints:
  - preserve the original `vit_result.json` schema and treat it as read-only
  - store requested sweep settings recovered from the run directory name in derived tables
  - keep figure stems stable and report-friendly
- Backward-compatibility expectations:
  - do not break the existing ViT-small reporting script or figure paths
  - new larger-model analysis should be additive and independently runnable

## Build Phases

1. Recover and validate the larger-model sweep table
   - Purpose:
     - reconstruct the intended `block x requested_frac x seed` grid from the run artifact directory names and confirm completion
   - Prerequisites:
     - spec is finalized
     - `runs/vit_base_imagenette_full/` exists and is readable
   - Planned edits:
     - create or extend a script that parses the sweep, validates uniqueness/completeness, and writes tidy outputs
   - Verification checkpoint:
     - recovered table has `60` unique runs and `60` unique requested triplets
     - best single run and best stable setting match hand inspection

2. Generate the larger-model figure suite
   - Purpose:
     - make every important larger-model result legible, including winner cells, aggregate trends, seed instability, and realized-vs-requested sparsity drift
   - Prerequisites:
     - recovered summary tables from Phase 1
   - Planned edits:
     - implement figure generation in `scripts/make_vit_scale_figures.py`
     - emit PNG/PDF outputs and caption sidecars
   - Verification checkpoint:
     - every expected figure file exists
     - plots clearly separate requested fraction from realized fraction
     - instability/collapse behavior is visible in at least one dedicated figure

3. Integrate the larger-model results into docs and reports
   - Purpose:
     - bring the completed larger-model analysis into the main project narrative
   - Prerequisites:
     - larger-model summary tables and figures exist
   - Planned edits:
     - update `docs/PHASE4_RESULTS.md`
     - update `report/REPORT.md`
     - update `analysis/report.tex` and compile `analysis/report.pdf`
   - Verification checkpoint:
     - all reporting surfaces agree on the main numbers
     - LaTeX report compiles successfully with the new figure includes

## Commands And Execution Notes

- Setup commands:
  - `cd /jumbo/lisp/f003x5w/ViT_Proj/CNN-SAE`
  - `source env.sh`
- Build/run commands:
  - `python scripts/make_vit_scale_figures.py --run-root runs/vit_base_imagenette_full --tag vit_base_imagenette_full`
  - `cd analysis && pdflatex -interaction=nonstopmode report.tex && pdflatex -interaction=nonstopmode report.tex`
- Data or asset preparation:
  - no new dataset download or GPU rerun is required
  - the completed run tree is the analysis input

## Risk Review

- Leak-sensitive steps:
  - interpreting the same validation corpus simultaneously as a sweep-selection surface and as untouched external evaluation
  - grouping by realized `frac_retained` instead of requested sweep fraction
- Benchmark-sensitive steps:
  - none on the original training side, since this task is post-hoc analysis only
  - figure/report summaries are benchmark-sensitive because they determine the narrative framing of the larger-model run
- Likely failure modes:
  - incorrect bucket recovery from run directory names
  - misleading aggregate tables caused by realized-fraction drift
  - report drift where one surface shows the best cell but another hides instability
- Escalation conditions:
  - if artifact recovery shows missing or duplicate requested cells
  - if evidence suggests the run family is too incomplete to analyze as a finished result
  - if the recovered numbers conflict with existing docs in a way that changes the main conclusion

## Leak Review

- Split boundaries to preserve:
  - do not alter or reinterpret the original train/val split used by `vit_base_imagenette_full`
- Fit-on-train-only requirements:
  - none are being newly fit in this task; post-hoc analysis operates on completed metrics only
- Intended evaluation-corpus exceptions:
  - Imagenette validation remains the intended self-consistency evaluation corpus for this ViT family
- What would count as a confirmed leak here:
  - direct proof that validation data entered training-time normalization/fitting for the larger-model sweep
  - not merely that some settings collapse or that the same validation corpus informed model selection

## Ready To Implement

- [x] Inputs are readable
- [x] File touch set is known
- [x] Verification checkpoints are defined
- [x] Eval handoff requirements are listed

# Inventory

This inventory follows the playbook categories and points to the files that drive the current analysis.

## Guide docs
- `docs/superpowers/results_analysis_playbook.md`
- `docs/PHASE1_RESULTS.md`
- `docs/PHASE2_RESULTS.md`
- `docs/PHASE3_RESULTS.md`
- `docs/PHASE3b_RESULTS.md`
- `docs/PHASE3b_changed_RESULTS.md`
- `docs/PHASE4_RESULTS.md`
- `docs/report/REPORT.md`

## Experiment code
- `src/backbone.py`
- `src/cache_activations.py`
- `src/controls.py`
- `src/data.py`
- `src/eval.py`
- `src/eval_chain.py`
- `src/feature_audit.py`
- `src/intervene.py`
- `src/masks.py`
- `src/masks_changed.py`
- `src/sae.py`
- `src/sae_changed.py`
- `src/taxonomy.py`
- `src/train_chain.py`
- `src/train_chain_changed.py`
- `src/train_sae.py`
- `src/train_sae_changed.py`
- `src/train_transition.py`
- `src/transitions.py`
- `src/vit_sae.py`

## Orchestration
- `scripts/__pycache__`
- `scripts/make_plots.py`
- `scripts/make_plots_tiers.py`
- `scripts/make_sae_vs_cnn.py`
- `scripts/run_grid.py`
- `scripts/run_grid_changed.py`
- `scripts/run_phase3.sh`
- `scripts/run_phase3_raw_only_changed.sh`
- `scripts/run_tier_rest.sh`
- `scripts/run_tier_rest_01.sh`
- `scripts/run_tier_rest_3gpu.sh`
- `scripts/train_backbone.py`

## Configs and launcher metadata
- `runs/phase1/grid_summary.json`
- `runs/phase2/grid_summary.json`
- `runs/pareto/grid_summary.json`
- `runs/phase1_patch_overlap_changed/grid_summary.json`
- `runs/phase1_patch_depth_changed/grid_summary.json`
- `runs/phase1_patch_disjoint_changed/grid_summary.json`
- `runs/phase2_rf_fair_changed/grid_summary.json`
- `runs/r20c10/grid_summary.json`
- `runs/seeds/grid_summary.json`
- `LISP-Setup/LISP-3-Setup/logs`

## Results artifacts
- `runs/phase1`
- `runs/phase2`
- `runs/phase3`
- `runs/phase3b`
- `runs/phase3b_bptt`
- `runs/phase3b_raw_only_changed`
- `runs/phase2_rf_fair_changed`
- `runs/phase1_patch_overlap_changed`
- `runs/phase1_patch_depth_changed`
- `runs/phase1_patch_disjoint_changed`
- `runs/pareto`
- `runs/r20c10`
- `runs/seeds`
- `runs/r110`
- `runs/taxonomy`
- `runs/vit_sae`
- `runs/intervene_result.json`
- `runs/audit`
- `runs/baselines_pca.json`
- `logs`

## Notes
- Core CIFAR experiments reuse cached train and test activations from `runs/acts_r56_c100`, `runs/acts_r20_c10`, and `runs/acts_r110_c100`.
- `docs/report/REPORT.md` is useful as historical narrative, but the current analysis treats run artifacts and code as primary evidence.

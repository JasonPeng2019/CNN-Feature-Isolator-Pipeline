# Phase 3b Changed Results

## Purpose

This document covers the later `_changed` follow-up suite that revisits the main Phase 3 / 3b story with three targeted experiments:

1. fair-budget RF masking
2. raw-carrier retraining without re-grounding
3. patchwise SAE retries
   - 3a: overlapping patchwise SAE
   - 3b: depth-aware overlapping patchwise SAE
   - 3c: disjoint patchwise SAE

These are not the same as the Phase 4 cluster reruns. Phase 4 tests scale, robustness, architecture shift, and transfer. This Phase 3b changed suite instead asks whether earlier conclusions were distorted by unfair RF budgeting, rollout mismatch in chain training, or an underspecified patchwise/vector baseline.

## Artifact Status

Present `_changed` artifact families:

- `runs/phase2_rf_fair_changed`
- `runs/phase3b_raw_only_changed`
- `runs/phase1_patch_overlap_changed`
- `runs/phase1_patch_depth_changed`

Missing `_changed` artifact family:

- `runs/phase1_patch_disjoint_changed`

So the changed suite is partly complete:

- RF fairness: complete
- raw-only retraining: complete
- overlap patch sweep: complete
- depth-aware patch sweep: partial
- disjoint patch sweep: no artifact directory present

## Experiment 1: Fair-Budget RF

### Goal

Earlier RF-local results were hard to interpret cleanly because the effective sparse budget was not matched fairly across receptive-field windows. This rerun tests whether RF locality itself was the problem, or whether the old budgeting scheme made RF look worse than it really was.

### Artifact

- `runs/phase2_rf_fair_changed`

### Completion

- `15/15` cells present

### Best results and pattern

The fair-budget RF rerun is competitive across all sections and fractions. Representative examples:

- `RFP_Q1_f0.02`: top-1 `0.7148`
- `RFP_Q1_f0.05`: top-1 `0.7162`
- `RFP_Q2_f0.10`: top-1 `0.7170`
- best overall observed top-1: `RFP_Q5_f0.05 = 0.7173`

### Interpretation

- RF locality is not intrinsically fatal.
- A fair per-window budget removes much of the old degeneracy story.
- The earlier weak RF narrative was partly a budgeting artifact, not only an architectural one.

## Experiment 2: Raw-Carrier Retraining Without Re-grounding

### Goal

This follow-up revisits the strongest earlier Phase 3b claim: that repeated re-grounding is necessary because raw learned chaining collapses under recursive rollout.

The changed run keeps the strongest stabilization tools:

- scheduled sampling
- full-chain BPTT
- gradient clipping
- per-step code and reconstruction losses

but removes re-grounding from the training carrier path itself.

### Artifacts

- main result: `runs/phase3b_raw_only_changed/result.json`
- launcher summary: `LISP-3-Setup/logs/phase3_raw_only_changed_summary.json`

The launcher completed successfully with `rc=0`.

### Final metrics

- `chain_raw_top1 = 0.6821`
- `chain_reground_top1 = 0.6339`
- `hybrid_chain_top1 = 0.7120`

Training trajectory highlights:

- epoch 0 raw chain: `0.0123`
- epoch 0 re-grounded chain: `0.5908`
- final raw chain: `0.6821`
- final re-grounded evaluation of the same trained model: `0.6339`

### Interpretation

- Raw-only chaining no longer collapses to chance.
- Direct rollout-aligned retraining recovers a strong raw chain.
- The final raw chain is still below the hybrid path, but it is no longer qualitatively broken.

This changes the earlier interpretation in an important way. The original failure was not only a geometry problem. A large part of it was also an exposure-bias / training-mismatch problem: the predictor was being asked at test time to consume states it had not been trained to roll through repeatedly.

So the updated conclusion is:

> Re-grounding remains a strong stabilizer, but it is not the only viable one. Raw chaining is fragile under the original setup, yet substantially recoverable under rollout-aligned retraining.

## Experiment 3a: Overlapping Patchwise SAE

### Goal

The original vector-style SAE baseline effectively encoded each spatial location independently with a `1x1` operator. This rerun asks whether patchwise local context restores competitiveness when the sparse coder is allowed to scan overlapping CNN-style neighborhoods instead of isolated points.

### Artifact

- `runs/phase1_patch_overlap_changed`

### Completion

- `45/45` cells present

This matches the full planned sweep:

- sections `Q1..Q5`
- fractions `0.02`, `0.05`, `0.10`
- CNN-filter-based multipliers `1x`, `2x`, `3x`

### Main pattern

Representative results:

- `PO_Q1_m1_f0.02`: top-1 `0.7010`
- `PO_Q1_m1_f0.05`: top-1 `0.7168`
- `PO_Q1_m1_f0.10`: top-1 `0.7176`
- `PO_Q1_m2_f0.05`: top-1 `0.7088`
- `PO_Q1_m3_f0.05`: top-1 `0.7033`
- best observed top-1: `PO_Q5_m3_f0.10 = 0.7188`

### Interpretation

The important qualitative pattern is not just the single best cell. It is that:

- overlapping patch context does help relative to the old pointwise vector story
- the strongest settings are usually the smallest CNN-filter-based patches, especially in early sections
- larger patches do not automatically help and often hurt `Q1`

So the original vector failure was not only “vector bad.” It was more specifically “too little spatial context.” But giving that context does not mean “make the patch as large as possible.”

## Experiment 3b: Depth-Aware Overlapping Patchwise SAE

### Goal

This variant asks whether patch size should shrink with depth so that deeper layers do not saturate under overly large local windows.

### Artifact

- `runs/phase1_patch_depth_changed`

### Completion

- `25` result cells present
- full planned sweep was larger than this

So this run is **partial**, not complete.

### What completed

The present artifact set covers:

- all `Q1` cells across schedules and fractions
- all `Q2` cells across schedules and fractions
- early `Q3` cells
- no complete late-section coverage for the full schedule sweep

Representative completed results:

- `PD_schedule_a_Q1_f0.05`: top-1 `0.7088`
- `PD_schedule_b_Q1_f0.10`: top-1 `0.7176`
- `PD_schedule_a_Q2_f0.05`: top-1 `0.7156`
- `PD_schedule_a_Q3_f0.02`: top-1 `0.7020`

### Interpretation

The early evidence suggests:

- the less aggressive schedule staying near the base CNN filter size performs better than the enlarged early-patch schedule
- in the completed subset, `schedule_b` looks stronger than `schedule_a`

But because the run is partial, this should be treated as provisional rather than final.

## Experiment 3c: Disjoint Patchwise SAE

### Goal

This was meant to test whether disjoint patch tilings might behave differently from overlapping CNN-style scanning, especially when patch sizes are chosen relative to the base CNN filter width.

### Artifact status

- no `runs/phase1_patch_disjoint_changed` directory is present
- no completed result set is available in the repo tree

### Interpretation

This is not a negative result. It is an absent artifact. The only defensible statement is that the disjoint patch experiment does not currently appear to have run to completion in the surviving repo state.

## Combined Interpretation

Taken together, the changed Phase 3b suite updates the earlier story in three ways.

### 1. RF locality was not judged fairly before

The fair-budget RF rerun shows that RF-local masking remains viable when the budget is normalized correctly within each window.

### 2. Raw chain failure was partly a training mismatch problem

The raw-only retraining run shows that repeated manifold projection is helpful, but not uniquely necessary. A strong raw chain can re-emerge when the model is trained directly on rollout conditions.

### 3. The old vector baseline failed partly because it had too little spatial context

The overlapping patchwise rerun shows that local patch context helps, but the best patch is usually the smallest CNN-filter-scaled one, not the biggest one. More spatial context helps up to a point; after that it can wash out the useful local structure.

## Bottom Line

The strongest updated Phase 3b changed conclusions are:

- RF-locality remains viable under fair budgeting.
- Raw-chain collapse is substantially recoverable under rollout-aligned retraining.
- Patchwise/vector-style sparse coding improves when given local overlapping context.
- Larger patches are not automatically better.
- The depth-aware patch story is still incomplete.
- The disjoint patch story is still untested in the surviving artifact tree.

These changed runs do not erase the original Phase 3 / 3b findings. They refine them. The revised lesson is not that the earlier conclusions were wrong, but that they were sometimes too strong or too narrow because they were based on less fair RF accounting, less rollout-aligned chain training, and a weaker patchwise baseline.

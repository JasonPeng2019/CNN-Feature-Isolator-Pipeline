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
- `runs/phase1_patch_disjoint_changed`

So the changed suite is now complete for the intended surviving families:

- RF fairness: complete
- raw-only retraining: complete
- overlap patch sweep: complete
- depth-aware patch sweep: complete
- disjoint patch sweep: complete

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

- `60/60` result cells present

This is a complete sweep over:

- schedules `a, b, c, d`
- sections `Q1..Q5`
- fractions `0.02`, `0.05`, `0.10`

Representative completed results:

- `PD_schedule_a_Q1_f0.05`: top-1 `0.7088`
- `PD_schedule_b_Q1_f0.10`: top-1 `0.7176`
- `PD_schedule_a_Q2_f0.05`: top-1 `0.7156`
- `PD_schedule_a_Q3_f0.02`: top-1 `0.7020`
- `PD_schedule_a_Q4_f0.05`: top-1 `0.7184`
- `PD_schedule_a_Q5_f0.05`: top-1 `0.7183`

### Interpretation

The finished sweep changes the earlier reading:

- schedules `b` and `c` are strongest overall by mean top-1 (`0.7128`), but the gap over `a` is small
- the best completed cells are not exotic: `Q4` and `Q5` at `5%` still hit `0.7184` / `0.7183`
- `Q3` remains the hardest section here too; its best completed depth-aware cell is `PD_schedule_a_Q3_f0.05 = 0.7153`
- schedule `d` is the only clearly weaker family, especially in later sections (`Q4/Q5`)

So the depth-aware story is no longer provisional. The more moderate schedules are viable, but they do **not** fundamentally beat the simpler overlapping-patch baseline; they mainly confirm that avoiding oversized patches early is the right instinct.

## Experiment 3c: Disjoint Patchwise SAE

### Goal

This was meant to test whether disjoint patch tilings might behave differently from overlapping CNN-style scanning, especially when patch sizes are chosen relative to the base CNN filter width.

### Artifact status

- `runs/phase1_patch_disjoint_changed`
- launcher summary: `LISP-3-Setup/logs/patch_disjoint_changed_summary.json` with `rc=0`
- `20/20` result cells present

### Best results and pattern

Best `5%` cells by section:

- `PDJ_Q1_m0.5_f0.05`: `0.7158`
- `PDJ_Q2_m0.5_f0.05`: `0.7049`
- `PDJ_Q3_m0.5_f0.05`: `0.7104`
- `PDJ_Q4_m0.5_f0.05`: `0.7184`
- `PDJ_Q5_m0.5_f0.05`: `0.7183`

The striking pattern is that **the smallest patch family (`m0.5`) wins every section**. Increasing the disjoint patch size is actively harmful:

- `Q1`: `0.7158` at `m0.5` falls to `0.5942`, `0.2472`, `0.1260`
- `Q3`: `0.7104` at `m0.5` falls to `0.6587`, `0.4134`, `0.0726`
- `Q4/Q5`: even deep sections collapse once the disjoint patches get too coarse

### Interpretation

This is a real, completed negative result for coarse disjoint tilings:

- disjoint patches can work when they are very small
- larger disjoint tiles are much worse than overlapping scanning
- the project’s patchwise lesson is therefore not just “local context helps,” but more specifically “**overlap matters, and coarse non-overlapping tilings destroy too much structure**”

## Combined Interpretation

Taken together, the changed Phase 3b suite updates the earlier story in three ways.

### 1. RF locality was not judged fairly before

The fair-budget RF rerun shows that RF-local masking remains viable when the budget is normalized correctly within each window.

### 2. Raw chain failure was partly a training mismatch problem

The raw-only retraining run shows that repeated manifold projection is helpful, but not uniquely necessary. A strong raw chain can re-emerge when the model is trained directly on rollout conditions.

### 3. The old vector baseline failed partly because it had too little spatial context, and overlap matters

The overlapping patchwise rerun shows that local patch context helps, but the best patch is usually the smallest CNN-filter-scaled one, not the biggest one. The disjoint rerun sharpens that further: small patches can work, but **non-overlapping coarse tilings are a bad substitute for overlapping local scanning**.

## Bottom Line

The strongest updated Phase 3b changed conclusions are:

- RF-locality remains viable under fair budgeting.
- Raw-chain collapse is substantially recoverable under rollout-aligned retraining.
- Patchwise/vector-style sparse coding improves when given local overlapping context.
- Depth-aware patch scheduling is complete and broadly consistent with the “small local context” story, but not clearly better than the simpler overlap baseline.
- Disjoint patch coding only works at the smallest patch size; coarse disjoint tilings fail badly.
- Larger patches are not automatically better.

These changed runs do not erase the original Phase 3 / 3b findings. They refine them. The revised lesson is not that the earlier conclusions were wrong, but that they were sometimes too strong or too narrow because they were based on less fair RF accounting, less rollout-aligned chain training, and a weaker patchwise baseline.

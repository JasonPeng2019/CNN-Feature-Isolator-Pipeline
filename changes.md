# Changes Log

## 2026-06-01

### Added: RF-percentage budgeting recommendation

A new recommendation was added to:

- `high-level-2/sparse_signal_sae_experiment_plan_12_experiments.tex`

under the **Mask Type 2: Receptive-Field-Local Masking** section.

### What was added

A new subsection titled:

- **Fair-budget comparison note (recommended addition)**

This note states that comparing:

- Global masking tuned by retained percentage/fraction, and
- RF-local masking tuned only by fixed `K_RF`

can be unfair because they may correspond to very different effective sparsity levels across sections.

It recommends adding an RF-percentage variant:

- retain top `p_RF%` inside each RF window,

or adapting `K_RF` per section to match a target effective retained fraction.

### Why this change matters

This makes Global vs RF comparisons apples-to-apples and addresses the known late-section degeneracy issue (especially around `Q5`), where fixed `K_RF` can imply an extremely tiny effective global retained fraction and artificially crash performance.

## 2026-06-01 (Follow-up To Do)

### Add a separate experiment: raw-chain retraining without re-grounding

Add an explicit ablation that retrains chain transitions on predicted carriers **without** decode->re-encode->mask re-grounding in the training loop, and then evaluates:

- raw chain after retrain-only,
- re-grounded chain after retrain-only (optional secondary check),
- comparison against current phase3b (scheduled sampling + re-grounding).

### Why this needs to be isolated

Current phase3b combines two effects:

1. chain-aware retraining on predicted carriers, and
2. manifold re-grounding between steps.

That means current gains do not isolate whether retraining alone can preserve task integrity in raw chaining.

### Hypothesis this tests

This ablation is intended to test whether task-preserving information is already organized in a sparse feature hierarchy strongly enough that retraining alone can stabilize raw composition, versus the alternative that manifold projection is the dominant required mechanism.

Put differently, re-grounding currently validates consistency of sparse transforms under frozen weights and shows high task preservation when repeatedly projected onto SAE-valid code space. The proposed retrain-only raw-chain ablation would measure the independent contribution of hierarchy-learning in transition models without that projection step.

## 2026-06-01 (New Design Note)

### Add a patch-context variant for the Vector / patchwise SAE

The current implemented `VectorSAE` is not truly using multi-pixel spatial patches. In code it is a shared per-location `1x1` encoder/decoder, so each spatial location is encoded independently with no local spatial neighborhood.

This likely handicaps the vector baseline, especially in early and mid CNN sections where the backbone itself is built from overlapping local filters and where RF-local masking most strongly benefits from local context.

### Recommended new experiment family

Add a new vector-style patch-context SAE variant that keeps shared weights across locations but allows each encoded location to see a local activation patch, for example:

- `3x3` overlapping activation patch
- `5x5` overlapping activation patch
- optional depth-aware schedule, such as larger patches in earlier sections and smaller patches in deeper sections

This should be tested before concluding that the vector / patchwise SAE family is intrinsically weak under RF-local or hierarchy settings.

### Why this change matters

The current comparison is between:

- a FieldSAE that performs explicit local spatial mixing, and
- a VectorSAE that only mixes channels at a single spatial site

That means the baseline is not just "simpler"; it is also operating with far less spatial context than the CNN filters that produced the hidden activations.

Testing overlapping activation patches would isolate whether the key missing ingredient is local patch context, rather than the broader FieldSAE architecture itself.

### Recommendation on disjoint patches

For CNN hidden activations, do **not** start with disjoint image-space patches as the main fix. CNN hidden maps arise from overlapping receptive fields, so forcing disjointness may move the representation farther from the model's natural computation.

For ViTs, the situation is different. Because ViTs already begin from image patches / patch tokens, a disjoint image-patch-style or token-patch-style vector SAE may be a more natural baseline and may work better there than it does for CNN hidden maps.

### Suggested experiment order

1. Keep the current `1x1` VectorSAE as baseline.
2. Add overlapping activation-patch VectorSAE variants (`3x3`, then `5x5`).
3. Optionally test a depth-aware patch schedule.
4. Only after that, test disjoint patch variants as an ablation.
5. If extending to ViTs, explicitly note that disjoint patch/token baselines may be more appropriate there than for CNN section activations.

## 2026-06-01 (Implementation-Ready `_changed` Experiment Track)

### Server/runtime note

The current LISP-3 node (`lisplab3.thayer.dartmouth.edu`) currently exposes:

- `tmux 3.4`
- 2 visible GPUs in `nvidia-smi` (`0`, `1`)

The new `_changed` experiment pipeline is therefore prepared for:

- 2 concurrent GPU jobs, and
- queued follow-on tmux jobs for the remaining experiments

without altering old scripts, old files, or old result directories.

### Experiment 1: fair-budget RF

Purpose:

- make RF-local masking comparable to global masking by matching retained percentage per RF window rather than using only a fixed `k_rf`

Definition:

- SAE family: `FieldSAE`
- mask type: `rf_frac`
- retained fractions per RF window: `{0.02, 0.05, 0.10}`
- sections: `Q1..Q5`
- backbone/cache: current ResNet-56 / CIFAR-100 winner path
- output root: `runs/phase2_rf_fair_changed`

### Experiment 2: retraining-only without re-grounding

Purpose:

- isolate whether retraining on predicted carriers can stabilize raw chaining without requiring decode->re-encode->mask re-grounding in the training carrier path

Definition:

- warm-start from current Phase 3 predictors
- carrier mode: raw predicted code
- keep available stabilizers:
  - scheduled sampling
  - BPTT
  - gradient clipping
  - per-step code loss
  - per-step reconstruction loss
- evaluate:
  - raw chain
  - re-grounded chain
  - hybrid chain
- output root: `runs/phase3b_raw_only_changed`

### Experiment 3a: overlapping patchwise SAE retry

Purpose:

- retry the patchwise/vector SAE family with true local patch context and CNN-style overlapping shifts

Definition:

- SAE family: patchwise vector SAE
- patch mode: overlapping activation patches
- stride: `1`
- patch size rule: `current section CNN kernel size × multiplier`
- note: for the current CNN, the local spatial kernel is usually `3x3`, so the multiplier is what creates meaningful variation
- multiplier sweep: `{1x, 2x, 3x}`
- realized patch sizes:
  - rounded/clipped to valid odd patch sizes
  - minimum patch size `1`
- sections: `Q1..Q5`
- retained fractions: `{0.02, 0.05, 0.10}`
- output root: `runs/phase1_patch_overlap_changed`

### Experiment 3b: depth-aware overlapping patchwise SAE with schedule sweep

Purpose:

- test whether deeper sections should use smaller patch context even when early sections benefit from larger local patches

Definition:

- SAE family: patchwise vector SAE
- patch mode: overlapping activation patches
- stride: `1`
- patch size rule: `current section CNN kernel size × section multiplier`
- schedule sweep:
  - `schedule_a`: `Q1/Q2=2x`, `Q3=1x`, `Q4/Q5=0.5x`
  - `schedule_b`: `Q1/Q2/Q3=1x`, `Q4/Q5=0.5x`
  - `schedule_c`: `Q1/Q2/Q3=1x`, `Q4/Q5=0.25x`
  - `schedule_d`: `Q1/Q2=2x`, `Q3/Q4/Q5=1x`
- realized patch sizes:
  - rounded/clipped to valid odd patch sizes
  - minimum patch size `1`
- sections: `Q1..Q5`
- retained fractions: `{0.02, 0.05, 0.10}`
- output root: `runs/phase1_patch_depth_changed`
- naming requirement:
  - every run name should include schedule id, section, and retained fraction

### Experiment 3c: disjoint patchwise SAE retry

Purpose:

- test non-overlapping patchwise coding as a stricter alternative baseline and compare smaller/equal/larger patch sizes relative to the CNN filter width

Definition:

- SAE family: patchwise vector SAE
- patch mode: disjoint activation-map patches
- stride: `patch_size`
- patch size rule: `current section CNN kernel size × multiplier`
- multiplier sweep: `{0.5x, 1x, 2x, 3x}`
- realized patch sizes:
  - rounded/clipped to valid odd patch sizes
  - minimum patch size `1`
- report each run as:
  - smaller than filter
  - equal to filter
  - larger than filter
- retained fraction: fixed at `0.05`
- sections: `Q1..Q5`
- output root: `runs/phase1_patch_disjoint_changed`

### `_changed` implementation rule

All new experiment code should be added as cloned `_changed` files rather than replacing old files. That includes:

- changed SAE implementation files
- changed mask implementation files
- changed trainers
- changed grid launchers
- changed LISP-3 helper scripts

Old files and old results should remain untouched.

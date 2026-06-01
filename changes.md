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

# Coordinated sparse-recovery redo

```text
Topology: Tier 2 — one implementation owner plus one bounded read-only validity review.
Execution: raw static tmux queues; no runtime subagent or multi-agent harness.
```

## Status and purpose

This document replaces the earlier narrow Level-2 plan.  That plan compared one
historical FieldSAE against one small context-vector modification on a single
CNN setting and ViT-Base block 10.  It is retired for this question.  Its code
and result directories remain historical artifacts and must not be presented as
results of this redo.

The new question is whether better sparse training and the three specified SAE
architectures improve reconstruction and frozen-model behavior at matched,
aggressively sparse codes across both CNN and ViT scales.

The declared matrix is intentionally broad.  There is no candidate-promotion
gate and no selection of one model before ViT-Base: all four model families run
on every declared CNN and ViT condition.

## Frozen interpretation of the requested scope

The requested `CNN small`, `CNN large`, `ViT small`, and `ViT base` settings
mean:

| Family | Frozen backbone/data | Fields evaluated |
|---|---|---|
| CNN small | ResNet-20 / CIFAR-10 | sections Q1, Q2, Q3, Q4, Q5 |
| CNN large | ResNet-110 / CIFAR-100 | sections Q1, Q2, Q3, Q4, Q5 |
| ViT small | `vit_small_patch16_224` / Imagenette | patch-token blocks 2, 4, 6, 8, 10 |
| ViT base | `vit_base_patch16_224` / Imagenette | patch-token blocks 2, 4, 6, 8, 10 |

ResNet-56/CIFAR-100 is not part of this requested small/large matrix.  It may
be used only as a historical reference; adding it requires an explicit plan
revision.  ViT prefix tokens are never encoded into the two-dimensional sparse
field and remain unchanged in frozen-backbone splices.

## Common scientific contract

Every listed condition uses the same new experimental procedure.

### Matched sparse message

For one activation field of shape `(C, H, W)`, define the exact per-image
support count

```text
M = max(1, round(original_node_fraction * C * H * W)).
```

The tested original-node fractions are **8%, 4%, 2%, and 1%**.  They are not
latent-field fractions.  A `K = 4C` model retains 2%, 1%, 0.5%, and 0.25% of
its coefficient field respectively.  The old `K = 8C` Field baseline retains
1%, 0.5%, 0.25%, and 0.125% respectively.  All four models receive the same
integer `M` for a given activation field; their dictionary size and encoded
bit-rate are reported separately rather than hidden.

Use deterministic exact global Top-M per sample in FP32:

```text
indices = topk(standardized_scores, M, sorted=False).indices
hard_mask = zeros_like(standardized_scores).scatter(indices, 1)
```

The support is the saved `indices`, not the count of numerically nonzero
values.  Threshold-based masks are prohibited because ties can exceed `M`.
The selector uses the hard forward mask and the straight-through sigmoid
surrogate from `.brainstorm/trainingSAE.md`.

### Clean data and reproducibility

- CIFAR-10 and CIFAR-100 each use a stratified split of their original training
  data for fit and validation; their official test set is untouched until final
  **fidelity-holdout** evaluation.  Labels are not needed for this experiment:
  compare the sparse-spliced output with the frozen original model output on
  the same held-out images. Start with 45,000/5,000 for CIFAR-100 and the proportional
  45,000/5,000 split for CIFAR-10 unless cache availability requires a documented
  equivalent stratified split.
- Imagenette uses its official train split for fitting. Before the first run,
  sort the official validation image IDs, stratify by class using fixed seed
  `20260827`, and split them 50/50 into checkpoint-selection validation and an
  untouched label-free fidelity holdout. Save both manifests and their hashes.
  The holdout measures original-versus-sparse behavior only; it is not a
  task-accuracy result.
- Fit activation normalization on training activations only.  Save its values,
  hash, sample IDs, frozen-backbone hash, section/block layout, prefix-token
  count, split manifests, seeds, resolved configuration, checkpoints, metrics,
  selected indices/values, and quantization metadata.
- Decoder-only replay from saved sparse messages must reproduce the recorded
  reconstruction.  Decoders may not receive source activations, encoder states,
  unmasked coefficients, or reconstruction residuals.

### Training policy for every family

All four families use the improved training policy from
`.brainstorm/trainingSAE.md`, not the historical immediately-final-mask path:

1. invariant/no-bypass/tiny-overfit gate;
2. five epochs of dense strict-decoder stabilization;
3. one epoch of selector bootstrap from detached coefficient magnitude;
4. five epochs at an exact easy 50%-of-latent budget;
5. ten epochs of geometric continuation to the declared `M`;
6. at least 20 target-budget epochs (up to 30 planned), with validation-selected
   checkpointing; and
7. only after a stable primary configuration, lower-budget continuations from
   its selected checkpoint rather than fresh random starts.

The continuation DAG is fixed for each backbone, field/block, SAE path, and
seed:

```text
initialize -> 8% -> 4% -> 2% -> 1%
```

`field_old`, the Field-A strict path, and the strict-vector path each follow
that chain. At every budget node, Field-B starts from that node's selected
Field-A strict checkpoint; the practical vector model starts from that node's
selected strict-vector checkpoint. Their correction/refiner warm-start and
joint-refinement phases run at that node's final budget. No 4%, 2%, or 1%
model starts from random initialization.

Use AdamW `(0.9, 0.95)`, weight decay `1e-4` excluding biases/norms/LayerScale/
normalized atoms, peak learning rate `3e-4`, 5% warmup, cosine decay to
`3e-5`, global gradient clipping at 1.0, FP32 support/loss/normalization, and
bf16 autocast when available.  The optimizer, phase state, recovery attempts,
and raw-versus-EMA validation choice are artifacts, not implicit state.

Training never routes the original activation to a decoder.  Optional
downstream-KL refinement, if used, is a separately saved task-aware checkpoint
with scheduled frozen-tail splice replacement and is never substituted for the
reconstruction-selected checkpoint.

## SAE families

All families are evaluated under the common procedure, budgets, seeds, and
metrics.  “Old” identifies the architectural baseline, not the obsolete
historical trainer.

| ID | Family | Dictionary and decoder claim |
|---|---|---|
| `field_old` | Improved old FieldSAE baseline | Historical shallow three-block FieldSAE topology and `K=8C`, upgraded only with the common exact-mask, clean-split, signed-score/value-head, and training policy.  This isolates procedure gain from the two new Field structures. |
| `field_strict` | `StrictAdditiveFieldSAE` (Field-A) | `K=4C`, `d=min(2C,256)`, six mixed-dilation Field blocks around one pooled attention correction, separate score/value heads, and a strictly linear additive 3/5/9/15 spatial-stamp decoder.  Effective atoms are unit-normalized. |
| `field_recovery` | `PracticalRecoveryFieldSAE` (Field-B) | The exact Field-A encoder and strict decoder plus a zero-initialized six-block mixed-dilation nonlinear correction.  It receives only charged sparse values and the transmitted hard support; it begins from the corresponding selected Field-A checkpoint. |
| `vector_context` | `PracticalContextVectorSAE` | `K=4C`, `d=min(2C,256)`, four residual convolution blocks with dilations 1/2/4/1, one pooled-to-at-most-8x8 pre-norm four-head attention correction with zero-initialized output, separate score/value heads, unit-normalized direct 1x1 vector atoms, and a zero-initialized four-block sparse-code-only spatial refiner. |

Field-A's pooled branch is after its first three blocks; it pools to at most
8x8, uses attention width 128 and fixed 2D positions, then projects back through
a zero-initialized 1x1 convolution.  Field-B's correction receives the sparse
value field and transmitted support mask only.  The practical vector decoder
receives the sparse value field only; the strict direct-vector decoder is its
required ablation, not an additional primary family.

Every model uses independent, unconstrained selection-score and signed-value
1x1 heads.  The old Field topology must retain its old shallow structural shape;
the other mandatory training-path changes are procedural fairness, not a claim
that it is the untouched historical checkpoint.

### Parent-checkpoint dependencies

For every backbone, field/block, budget, and seed:

- `field_strict` trains and selects a validation checkpoint first;
- `field_recovery` starts from that exact checkpoint, warms only the correction
  for three target-budget epochs, then jointly refines for 20 epochs at a peak
  `1e-4` learning rate;
- `vector_context` first trains its strict direct-vector path under the common
  curriculum, then enables its zero-initialized sparse-code-only refiner,
  warms it, and jointly refines it under the same recovery schedule.

These are required continuations within the Field-B and practical-vector cells.
They are not eligibility gates that can remove those families from ViT-Base.
If a continuation collapses, apply the bounded recovery policy in
`.brainstorm/trainingSAE.md` (maximum two recovery attempts); retain the failure
artifact and classify the seed as failed after a third collapse.

## Complete run matrix

Each cell below is one final SAE family/budget/seed result.  The prerequisite
strict phases described above are part of the cell's execution.

| Domain | Fields | Families | Budgets | Seeds | Final cells |
|---|---:|---:|---:|---:|---:|
| CNN small (R20/C10) | 5 sections | 4 | 4 | 3 | 240 |
| CNN large (R110/C100) | 5 sections | 4 | 4 | 3 | 240 |
| ViT small | 5 blocks | 4 | 4 | 3 | 240 |
| ViT base | 5 blocks | 4 | 4 | 3 | 240 |
| **Total** | **20 fields** | **4** | **4** | **3** | **960** |

No old result directory is overwritten.  Use a new versioned root such as
`runs/sae_sparse_recovery_redo/<timestamp>/`, partitioned by domain/backbone,
field, family, budget, and seed.  A parent checkpoint and all of its dependent
continuations live in the same immutable cell directory; a retry receives a
new suffixed root and references the failed attempt in metadata.

## Evaluation and reporting

Validation selects checkpoints. Once the complete matrix configuration and
selection rules are frozen, run each declared selected checkpoint on its
untouched fidelity holdout once. No result from that pass changes an
architecture, schedule, budget, or seed decision. This is a held-out measure of
whether sparse reconstruction preserves the frozen original model's outputs,
not a claim that the sparse reconstruction independently solves the task.

For each cell report:

- normalized reconstruction loss, relative L2, FVU, cosine, mean and worst
  channel error;
- frozen-model prediction agreement and original-to-spliced KL;
- exact `M`, numerical nonzero count, support bits, signed-value bits,
  side-information bits, and bits per original activation node;
- collapse state, recovery attempt count, training phase, raw/EMA choice,
  strict-path versus nonlinear-correction norms, feature use/dead-feature rate,
  parameters, FLOPs, peak memory, and throughput.

Report every seed and also summaries that separate stable from collapsed seeds.
Never average a collapsed seed into a headline mean without showing the
distribution.

For each validation-selected final configuration run the following controls
with the same decoder and exact message budget: random support, magnitude
support, constant selected values, shuffled selected values, selected values on
random support, fixed support across examples, empty message, attention disabled,
and the relevant strict/recovery ablations.  The recovery Field additionally
reports correction-disabled and strict-path-disabled results.  These controls
are evaluations, not extra architecture sweep rows.

## Required CNN chain phase: re-grounding retained

Re-grounding is not an SAE fitting technique.  It is the mechanism that made
the earlier learned sparse **chains** work: after each predicted destination
code, decode it, convert it with the destination train-fitted normalization,
re-encode it with the destination SAE, and apply the destination's exact
original-node-relative Top-M mask.  The next transition receives this projected
sparse code rather than an unconstrained dense prediction.

It is required after the 960-cell reconstruction matrix, not omitted:

1. For each CNN backbone, select the best non-collapsed **new** SAE family at
   1% original-node sparsity. A family is eligible only when at least two of
   three seeds are non-collapsed in every Q1–Q5 section. Rank eligible families
   by mean validation original-versus-sparse prediction agreement over all five
   sections; lower mean original-to-spliced KL breaks a tie. Keep `field_old`
   as its matched baseline.
2. Use the selected 1% SAE checkpoints for Q1–Q5 and train the four adjacent
   transition predictors Q1→Q2→Q3→Q4→Q5 for both the selected family and the
   old Field baseline. Train on the clean train split only, validate on the
   clean validation split, and evaluate the fidelity holdout once after
   selection.
3. Use **only the full-BPTT Family-II re-grounded procedure**, not the old
   post-hoc evaluation projection.  During training, scheduled-sampling
   probability rises from 0 to 1: later transitions increasingly consume the
   chain's own re-grounded predicted carrier rather than a teacher code.  For a
   predicted destination code `zhat_b = T_a_to_b(z_a)`, form the in-graph
   carrier

   ```text
   z_b = exact_topM_b(E_b(normalize_b(D_b(zhat_b))))
   ```

   where `D_b` and `E_b` are the frozen destination SAE decoder and encoder and
   `exact_topM_b` is the destination's declared original-node `M` with the
   hard-forward straight-through gradient.  Backpropagation remains connected
   through this complete carrier to `T_a_to_b`; do not detach it, put it under
   `no_grad`, or replace it with an encoder output from the true destination
   activation.  Apply gradient clipping and save all transition, SAE,
   normalization, support, and gradient-path artifacts.
4. Train a matched raw-carrier Family-II comparator with the same scheduled
   sampling, full-rollout BPTT, losses, optimizer, clipping, seeds, and split;
   its next carrier is `zhat_b` directly.  This specifically tests whether
   re-grounding helps beyond rollout-aligned training.  It is not a fallback
   that may silently replace the primary re-grounded path.
5. Report three seeds for every backbone/family chain and compare raw rollout,
   exact re-mask-only rollout, re-grounded rollout, and the frozen-block hybrid
   causal upper bound. Report each adjacent transition as well as end-to-end
   Q1→Q5 behavior.

The transition predictor is never allowed to inherit the old latent-fraction
masking helper: every re-ground step uses the destination field's declared
original-node `M`. This phase is only defined for the ordered CNN Q1–Q5 chain;
it does not fabricate an analogous ViT hierarchy experiment.

## Gates, queue, and completion

### Tier-2 read-only validity review

Before any full GPU queue is launched, one fresh read-only reviewer performs a
bounded review of the frozen contract, changed sparse-selection/SAE/trainer/
launcher paths, and smoke configuration.

```text
Objective: Catch a quiet experimental-validity defect before the 960-cell run.
Scope: exact Top-M/rate accounting, train-validation-holdout isolation,
decoder no-bypass, and full-BPTT re-grounding carrier provenance/gradients.
Write authority: none.
Checks: read-only inspection and focused checks that do not launch the matrix.
Deliverable: evidence-backed pass or concrete blocking finding with paths.
Completion: the implementation owner records the finding, fixes confirmed
blocking issues, and reruns the affected focused checks before launch.
```

This is a one-time review gate. It does not edit code, manage tmux, poll jobs,
or create a runtime harness.

Before launching the 960-cell queue, implement and pass focused checks for:

- exact `M` including tied scores; finite selector gradients; hard-forward mask;
- train/validation/fidelity-holdout manifest disjointness and train-only normalization;
- decoder-only replay and no source/encoder bypass;
- strict Field decoder linearity and atom normalization;
- finite gradients through zero-initialized attention and recovery/refiner
  branches after their output projections begin learning;
- tiny-batch overfit and dense-warmup improvement;
- one real-CUDA bounded smoke for each of the four domain families and each
  parent/continuation launcher path; and
- a chain smoke proving that the full-BPTT re-grounded carrier has a finite,
  nonzero gradient to every transition predictor, is unchanged when only the
  true destination activation is perturbed after `zhat_b` is formed, and uses
  exactly the destination original-node `M` under score ties.

Use one tmux session per available GPU, each running a static text FIFO queue
with `CUDA_VISIBLE_DEVICES` pinned to that GPU.  Queue entries run one command
at a time, append start/end/exit status to a durable log, and then advance
without agent-side polling.  Strict parent jobs are enqueued before their
dependent recovery jobs in the same cell chain; unrelated cells are distributed
across the five queues.  The queue runner resumes by skipping only cells whose
completion manifest, checkpoint, and result hashes pass validation.  It never
overwrites partial or historical outputs.

Completion means all 960 declared cells have either a complete validated result
or a preserved, classified terminal failure; all control and one-shot test
evaluations are saved; the required selected-family-versus-old-baseline CNN
re-grounded chain phase is complete; and the final summary compares all four
families at the same original-node `M` for all four requested backbone families.
Causal and taxonomy follow-ups remain separately scoped.

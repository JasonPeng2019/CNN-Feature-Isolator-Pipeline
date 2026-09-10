# Safe versus ambitious sparse-recovery experiment

Status: proposed successor experiment, written 2026-09-09. The experimental
matrix is specified below; the source-access decision and final confirmation
dataset remain open in section 10. This document does not authorize a GPU
launch. Existing Level-2 results retain their original contract and names.

## 1. What we are trying to learn

Find a sparse autoencoder that reconstructs frozen model activations and
preserves model predictions using very few transmitted coefficients. Give two
serious methods a fair chance: a practical one-pass method, and a more ambitious
method that repeatedly improves its sparse code. Greater recovery is a
hypothesis, not a promised outcome of either method.

The main comparison uses the **same multiscale architecture and decoder
capacity**, with two encoding/training procedures. The existing
`field_recovery` remains the practical incumbent: the new safe proposal must
earn its place by comparison with it. `field_strict` is an additive control and
a prerequisite for that incumbent, not the ambitious challenger.

Success supports sparse-message sufficiency: selected locations and values,
together with a trained decoder, recover useful activations. It does not by
itself establish individual feature interpretability or sparse computation in
the original backbone.

## 2. Source reconciliation and scope decisions

The six source files are in `/jumbo/lisp/f003x5w/.brainstorm/`.
They are design history; this document
states the proposed successor's choices explicitly instead of inheriting all
their conflicting instructions.

| Source | What this plan retains | What requires correction or a scope decision |
|---|---|---|
| `scratch_ideas.md` | Preferred multiscale one-pass versus iterative pair; residual fallback; optional global-context candidate; sparse-message sufficiency and controls | Its earlier six-system grid is an expansion option, not the initial required grid. Its residual-based refinement conflicts with the revised Field note. |
| `fieldSAErec.md` | Same architecture for the primary pair; shared rate; stronger multiscale processing; distinct iterative and distilled results | The ban on rereading the source during refinement is unresolved; see section 10. |
| `trainingSAE.md` | Train-only fitting, exact selection, continuation, long target-budget training, numerical safeguards, logged retries | Mandatory strict-parent training belongs to existing recovery models. It does not define training for both new primary methods. |
| `vectorSAEcross.md` | Contextual vector model and its direct additive control | Its 40%/20% original-node budgets and prohibition on iterative VectorSAE design do not set the new Field experiment's budgets or methods. |
| `sparsity_training-budgets.md` | 8%/4%/2%/1% original-node curve; cross-domain evaluation; winner-focused chain follow-up | Its strict/recovery primary pair is superseded for this proposal. Its R56 inclusion and winner-only ViT-Base promotion conflict with the later frozen four-backbone scope. |
| `brainstorm-proposal-safe-v-ambitious.md` | Explanation that the missing ambitious method is iterative sparse inference | Its description and line references for the former `fieldSAErec.md` are historical, because that file was subsequently rewritten. |

Scope judgment: retain R20, R110, ViT-Small, and ViT-Base from the explicit
frozen Level-2 contract, excluding R56. Use representative fields first to
debug and freeze the new methods, then cover all four backbones. A pilot
failure can postpone an expensive run for diagnosis; it cannot silently remove
a difficult domain from the final report or turn it into a successful result.

The authoritative existing-run records remain
[SAE_REDO_LEVEL2_CONTRACT.md](SAE_REDO_LEVEL2_CONTRACT.md) and
[SAE_REDO_LEVEL2_TOPOLOGY.md](SAE_REDO_LEVEL2_TOPOLOGY.md). This proposal changes
the next scientific comparison, not those records.

## 3. Systems and architecture

| Proposed result ID | Role | Encoding and training | Dictionary |
|---|---|---|---|
| `field_old` | Historical architecture with the improved exact-selection trainer | Existing Level-2 baseline recipe | `K=8C` |
| `field_recovery` | Practical incumbent that the new proposals must beat | Existing strict-parent plus correction warmup and joint training | `K=4C` |
| `field_multiscale_safe` | New practical primary | One-pass selection, continuation, then matched extra target-budget training | `K=4C` |
| `field_multiscale_iterative` | New ambitious primary | Same multiscale base; fixed-round sparse-code refinement and joint training | `K=4C` |

The two new IDs are proposed, not currently accepted launcher/model options.
`field_multiscale_base` is their shared training checkpoint, not a fifth
headline candidate. `field_multiscale_distilled` is a conditional later result,
not a substitute for reporting the actual iterative model.

Proposed architecture to freeze before the pilot:

- Channelwise train-fitted normalization; latent field `K x H x W`, `K=4C`.
- Hidden width `d=min(2C,256)`; one full-resolution path and context paths at
  `max(1,ceil(H/2)) x max(1,ceil(W/2))` and
  `max(1,ceil(H/4)) x max(1,ceil(W/4))`.
- Project input to width d. At each scale use two existing-style residual
  blocks: depthwise 5x5 convolution, GroupNorm, pointwise expansion by four,
  GELU, pointwise projection, and residual scaling initialized to 1e-3.
  Pool the projected input to form the context paths; bilinearly resize their
  outputs, concatenate all three paths, and fuse with a 1x1 projection.
- Independent FP32-ranked selection scores and signed coefficient values.
  One full-resolution coefficient address space avoids an uncounted branch ID.
- Decoder projects the sparse value field to width d, processes the same three
  scales independently, fuses them, and projects to C channels. It may also
  use a binary support tensor reconstructed from the transmitted indices;
  this same input convention is used by both new methods.
- All decoder features originate in the sparse message. Encoder feature skips
  into the decoder are prohibited. The primary decoder is nonlinear; strict
  additivity is not imposed on this recovery experiment.

This is a concrete starting design, not a claim that its capacity is optimal.
The implementation pilot must record parameters, memory and per-step time.
Any width/depth change creates a new configuration for both new methods.
Attention is not needed in the initial primary pair. A residual-only fallback
or pooled-attention expansion gets a separate architecture ID and must be
compared with both training methods before attributing its gains to training.

## 4. Matrix, budgets, and actual workload

Every main result is one backbone/field/system/budget/seed combination.

| Backbone and data | Fields | Main systems | Original-node budgets | Seeds | Main results |
|---|---|---:|---|---|---:|
| ResNet-20 / CIFAR-10 | Q1,Q2,Q3,Q4,Q5 | 4 | 8%,4%,2%,1% | 0,1,2 | 240 |
| ResNet-110 / CIFAR-100 | Q1,Q2,Q3,Q4,Q5 | 4 | 8%,4%,2%,1% | 0,1,2 | 240 |
| ViT-Small / Imagenette | patch blocks 2,4,6,8,10 | 4 | 8%,4%,2%,1% | 0,1,2 | 240 |
| ViT-Base / Imagenette | patch blocks 2,4,6,8,10 | 4 | 8%,4%,2%,1% | 0,1,2 | 240 |
| Total | 20 fields | 4 | 4 budgets | 3 | **960** |

For each image let `N=C*H*W`, `L=K*H*W`, and
`M=max(1,round(budget*N))`, using the repository's existing rounding convention.
Validate `M<=L`; never silently clamp an invalid request. Every method selects
exactly M unique addresses. Selected zero values still count as transmitted
entries. Use score descending, index ascending as a deterministic tie rule;
do not assume a tiny FP32 perturbation distinguishes every tied score.

| Original-node budget | Retention with K=4C | Retention with K=8C | Purpose |
|---:|---:|---:|---|
| 8% | 2% | 1% | Reference and continuation starting point |
| 4% | 1% | 0.5% | Practical sparse target |
| 2% | 0.5% | 0.25% | Main low-rate comparison |
| 1% | 0.25% | 0.125% | Aggressive target |

The new pair matches both M and address-space size. Comparison with
`field_old` matches M but not support bits, since its dictionary is larger.
Report both coefficient-count curves and measured/declared bit-rate curves.
Do not call all four systems bit-rate matched merely because M matches.
Use a common declared 16-bit signed value encoding for the new pair, save its
actual quantized payload, and report reconstruction after decoding that
payload. Charge quantizer scales and other sample-dependent metadata.
Unquantized metrics may be supplementary, not substitutes for payload replay.

The 960 main results are not 960 independent training jobs:

| Additional required checkpoint family | Count | Why |
|---|---:|---|
| `field_strict` | 240 | Same-cell parent required by existing `field_recovery` |
| `field_multiscale_base` | 240 | Shared initialization for safe and iterative extra training |
| Main plus these prerequisites | **1,440** | Upper bound on planned checkpoint-producing segments before reuse, retries or controls |

Within each path budgets depend on higher-budget checkpoints. Existing valid
baseline/strict/recovery artifacts can reduce new training only after matching
backbone, data IDs, normalization, architecture, budget, seed, training recipe,
value encoding and provenance. Reuse is recorded per cell. They cannot count
as new multiscale results. Selection uses validation evidence even if an old
artifact already contains holdout metrics.

The incumbent and old baseline retain their established recipes, while the new
pair shares the recipe below. Comparisons against the incumbent therefore test
whole systems; they do not isolate architecture alone. The safe-versus-iterative
comparison is the controlled training/inference comparison.

## 5. Training comparison and dependencies

First fit the shared multiscale base at 8%: five exact easy-budget epochs,
ten geometric-annealing epochs, then thirty target-budget epochs. Easy budget
starts at 50% of latent addresses. Jointly train encoder, value/score heads
and decoder; no strict-additive parent is required. Bootstrap score ordering
from detached magnitude for one initial epoch if selected as the common pilot
recipe. Dense reconstruction is a separate diagnostic, not a hidden message
path. Freeze the bootstrap choice before comparing seeds.

For lower budgets, continue the selected base checkpoint through
8% -> 4% -> 2% -> 1%, with five annealing and thirty target-budget epochs at
each new budget. Base continuation does not consume an iterative checkpoint.
Select target-budget checkpoints by validation mean per-image normalized
squared reconstruction error. Never select an easier-budget checkpoint for a
lower-budget result.

At each budget, copy that exact base checkpoint into two independent branches:

| Branch | Extra target-budget training | Final inference |
|---|---|---|
| Safe | Twenty epochs of ordinary one-pass joint training | One encoding pass |
| Iterative | Twenty epochs of joint fixed-round refinement training | Initial pass plus R=3 refinement rounds |

Use identical minibatch order, optimizer-update count, data, effective batch
size, base checkpoint and decoder topology for these branches. Restart AdamW
at peak 1e-4 with cosine decay to 1e-5 for both. Initial base training uses
peak 3e-4, floor 3e-5, betas (0.9,0.95), weight decay 1e-4, global clipping 1,
and 5% learning-rate warmup. Exclude normalization, bias and residual-scale
parameters from decay. Use effective batches 128 for CNNs and 32 for ViTs,
with accumulation when necessary. FP32 ranking/loss and bf16 compute follow
the existing numerical policy. Raw/EMA selection is validation-only and uses
the same rule in both branches.

R=3 is a proposed finite starting value, not an empirically established optimum.
At every round form an exact-M message, decode it, and update candidate scores
and signed values through one shared residual refinement module, with its
output projection initially zero. Use hard selection forward and a
straight-through gradient during fitting. New support locations must be
eligible: refining only the currently selected values does not implement the
intended support-swapping method. The final decoder receives only the final
message; intermediate messages and refinement state are never transmitted.

The refinement module's available inputs depend on question 1 in section 10.
With source access it can use the normalized source-minus-reconstruction
residual. Without source access it can use only the initial code, current
code/reconstruction and state derived from those messages. The latter cannot
directly observe what source information was omitted by the initial code.
These are different experiments and require distinct configuration identities.

The iterative branch adds encoder parameters and compute. Equal update counts
do not imply equal FLOPs or equal wall time. Report both costs; make no
compute-efficiency claim without a supplementary time-matched one-pass run.
Record measured per-cell cost before expanding; derive a GPU-hour estimate
from the actual remaining segments and include retries separately.

If iteration helps, distill its final codes into a one-pass student with the
selected decoder fixed, fitting on train only. Evaluate reconstruction as well
as support/value imitation. Count student fitting as additional work; report
student and teacher separately. Distillation is conditional and outside the
960 main results. Downstream-KL fine-tuning is also a separate optional
artifact for both methods, not a replacement for reconstruction-only results.

## 6. Staged work and implementation requirements

1. **Resolve and freeze inputs.** Settle section 10; inventory reusable
   artifacts and data exposure. Freeze architecture, refinement inputs,
   optimizer phases, quantizer, selection criteria, and split manifests.
2. **Implement and check the two new methods.** The inspected code currently
   exposes the old/strict/recovery/vector families in `src/sae.py:build_sae`
   and fixes the Level-2 families in `scripts/launch_sae_redo_level2.py`.
   Add a versioned new experiment configuration/entrypoint rather than
   relabeling an existing family. Reuse common normalization, sparse-message
   and provenance utilities after checking their invariants. Extend the phase
   policy for shared-base branching and iterative encoding. Existing tests in
   `tests/test_sae_redo_level2.py` remain regression checks, not evidence that
   the new methods work. Add focused tests for the behaviors below.
3. **Run representative development conditions first.** R110 Q1/Q3/Q5 and
   ViT-Small blocks 2/6/10; all four main systems; 8%/4%/2%; seeds 0/1/2.
   This is 216 main results plus 54 strict and 54 shared-base checkpoints,
   or 324 segments before reuse. These are subsets of the full matrix, not
   additional final cells. Early one-seed smoke runs prove execution only.
4. **Freeze after development.** Compare the two new branches and the incumbent
   on validation. If configuration changes, old pilot outputs remain development
   evidence and do not fill final cells of the revised configuration. Failed
   seeds remain visible. If both new methods fail, test the residual fallback
   on the same representative fields before expanding. A global-context
   candidate is a separately costed expansion, not an automatic six-way sweep.
5. **Complete the declared matrix.** Add missing fields, 1% budgets, R20 and
   ViT-Base. First exercise ViT-Base blocks 10 and 6 as infrastructure/transfer
   checks; all declared blocks remain in the result inventory. Record a
   scientific failure separately from an unrun or dependency-blocked cell.
6. **Freeze final selections, evaluate and report.** Evaluate the declared
   held-out split once per selected final checkpoint; archive controls and
   replay evidence. Address the confirmation-data limitation in section 10.
7. **Carry forward the CNN chain experiment.** Choose a new family per CNN
   from recovery/safe/iterative using validation, requiring at least two stable
   seeds in every Q1-Q5 section at 1%; rank mean agreement, then lower KL.
   Compare with old Field using the inherited full-BPTT re-grounded chain and
   matched raw-carrier training. Two CNNs x two families x two training modes
   x three chain seeds = 24 chain fits, each covering four adjacent edges.
   Re-mask-only and frozen-block hybrid are additional evaluations. If no
   candidate qualifies, record chain selection as blocked; a higher rate is
   a separately declared amendment. Iterative destination encoding in a chain
   may inspect only the decoded predicted activation, never the true
   destination activation. Preserve gradients through the complete encoding
   procedure. No ViT chain analogue is assumed.

Implementation checks must prove exact counts and stable tie ordering;
finite selector/refiner gradients after zero projections begin learning;
R=0 equivalence to the base; initial zero-refiner equivalence; actual eligibility
of unselected locations during refinement; inference with frozen weights and
fixed R; decoder independence from source and encoder state; quantized
decoder-only replay; split isolation; tiny-batch overfit; and branch/checkpoint
provenance. Exercise both CNN and ViT paths and parent-failure queue behavior.
An ordinary collapsed method must not prevent unrelated methods from running.

Numerical collapse receives at most two logged recovery attempts, restoring
a valid checkpoint and lowering learning rate by three. Restore at an easier
budget only where a curriculum phase exists, and record extra steps. Never
change R, source access or architecture as a retry. Rate violations, replay
failure or leakage require an implementation fix and new affected artifacts.
Report first-attempt stability separately from success after recovery. A method
with two of three stable seeds is pilot-eligible, not proven universally stable.

## 7. Controls without another large primary sweep

For main selected checkpoints evaluate empty message, random exact-M support,
magnitude support, fixed support across images, correct support with constant
or shuffled values, and selected values placed on random support. Distinguish
random support with values gathered at the new locations from relocating the
original selected values. Freeze random-control seeds and constant-value rules
using train/validation. An information-destroying control must be compared
under the same decoder and payload accounting. An empty message is the explicit
zero-rate exception to the exact-M rule.

Report incumbent correction-disabled results and new iterative R=0 results.
Ablating a trained model is not equivalent to independently training that
ablated model. The shared base and extended safe branch provide the actual
trained one-pass comparisons.

On the representative development fields retain a strict local-vector-decoder
comparison using `vector_context_strict`: two domains x three fields x three
budgets x three seeds = 54 checkpoint segments, outside the main count. Its
encoder has context; do not call it an entirely local encoder. Reuse matching
Level-2 artifacts where valid. A full recovery-vector sweep is optional.

Fit a flattened-field PCA/low-rank baseline on train per representative field;
choose rank from the same payload-bit budget, charging its transmitted scores
and reporting decoder storage. This is a rate comparison, not an exact sparse
address-count comparison. Also train a dense version of the new architecture
on each of those six representative fields as an empirical reconstruction
reference. Simply disabling the mask of a sparsely trained model is not a
trained dense ceiling. These fits, vector controls, distillation, fallbacks
and chain work are additional to the 1,440-segment accounting.

## 8. What counts as good, bad, or inconclusive

Keep two judgments separate: recovery quality and training reliability.

- **Safe proposal improves on the incumbent:** lower paired validation
  reconstruction loss, with agreement no worse and KL no higher, at the same
  M; then check whether the conclusion also holds against actual payload bits.
- **Ambitious method improves on safe:** the same comparison against the
  extended safe branch sharing its initial base checkpoint, with its additional
  compute and failure frequency shown explicitly.
- **Mixed result:** reconstruction and downstream behavior disagree, benefits
  depend on section/backbone, or seed differences change the ordering. Show
  those cases; do not force a single winner.
- **Bad recovery:** the learned message does not improve over empty/random
  controls, or loses to its comparator on both reconstruction and behavior.
  Stable training alone is not evidence of useful recovery.
- **Training failure:** non-finite computation or persistent reconstruction
  collapse. Use the existing normalized-error/FVU boundaries as logged alerts;
  an isolated poor early epoch is not sufficient to declare a terminal failure.
  Exact failure triggers and phase grace periods must be frozen before fitting.
- **Unrun/invalid:** missing prerequisite, unfinished job, invalid rate or
  provenance. These are not numerical failures or successful completed cells.

Primary checkpoint selection is minimum validation normalized reconstruction
loss at the exact target. Architecture development emphasizes 2% on R110 and
ViT-Small, with all representative fields weighted equally; 4% and 8% show
continuity and 1% is the aggressive extension. A method is broadly preferable
only when both domains support that conclusion. Do not pick a new metric or
omit a difficult field after viewing results.

Report every seed, paired differences, means and standard deviations, stable
seeds/attempted seeds, retries and elapsed compute. Use paired image bootstrap
intervals within each seed if uncertainty intervals are reported; images are
not independent training replicates, and three training seeds provide limited
precision. Show the entire rate curve and per-field results before averaging.
Absolute fidelity targets such as 95% or 99% agreement require the user choice
below; comparative improvement alone does not mean near-perfect recovery.

## 9. Data, artifacts, and execution handoff

Fit normalization and weights on train; make all choices on validation.
Retain existing CIFAR 45,000/5,000 train/validation and Imagenette official-train
plus the stratified official-validation halves (seed 20260827) where their
manifests verify. ViT prefix tokens remain untouched and outside the sparse
field, and reports explicitly say patch-field fidelity. Downstream agreement
and original-to-spliced KL use frozen-backbone outputs, not a claim of improved
label accuracy.

Prior Level-2 results have informed this research conversation. Audit whether
the evidence used for design came exclusively from validation or also from
holdout metrics; this planning pass has not established that exposure history.
If holdout metrics informed choices, the same holdout can support a disclosed
follow-up benchmark but cannot be described as fresh confirmation evidence.
Randomly reshuffling already inspected examples does not create fresh data.

Use a new root such as `runs/sae_safe_ambitious/<experiment-version>/` with
backbone/field/system/budget/seed/attempt partitioning. Store resolved config,
source revision plus dirty-diff digest, environment/device facts, command,
ordered sample IDs, backbone/cache/normalization hashes, parent checkpoint hash,
optimizer/phase state, all attempts, metric histories, payloads, quantizer,
decoder checkpoint and replay checks. Complete manifests must bind the actual
files. A resumed job skips only verified completed work. Do not count metadata
files or cached development checks as new final scientific results.

Use the repository queue mechanism after adapting its explicit dependency graph
and validating it. Discover available resources at launch rather than assuming
five free GPUs. Training deadlines must be estimated from smoke throughput and
data/epoch counts under the repository's bounded-command policy. This planning
task does not start, stop or replace any current queue.

Completion of this proposed experiment means all declared cells have validated
results or explicit terminal statuses, controls and costs are accounted for,
comparisons show failures and missing evidence, and the eligible CNN chain
follow-up is completed or its exact failed prerequisite is recorded.

## 10. Questions needed before implementation is frozen

1. **May the ambitious encoder reread the source activation during its fixed
   refinement rounds to compute reconstruction error?** Recommendation: yes,
   inside the encoder only; weights stay frozen at evaluation and only the
   final sparse message reaches the decoder. This implements the original
   residual-guided idea. The revised `fieldSAErec.md` explicitly says no, so
   this boundary should not be silently reversed. Keeping that restriction
   instead gives a code-only refinement experiment with less information.
2. **What absolute fidelity should qualify as a successful low-rate result?**
   The plan already defines fair comparative judgments. For your headline,
   should the target be, for example, at least 95% agreement, at least 99%,
   or simply the strongest achieved rate/fidelity curve? Specify whether that
   must hold in every field or on average. Until answered, no numeric absolute
   success threshold is asserted.
3. **If the exposure audit shows earlier holdout results informed the design,
   is a disclosed follow-up on that benchmark sufficient, or do you want
   fresh confirmation data?**
   Recommendation: use existing train/validation for development and reserve
   an independently uninspected, appropriate dataset for final confirmation
   if one is available. Its identity cannot be invented from these notes.

These questions hold only the dependent specification or final claim. The
matrix, role definitions, shared architecture, controls, implementation work
inventory and workload calculations above are ready for review now.

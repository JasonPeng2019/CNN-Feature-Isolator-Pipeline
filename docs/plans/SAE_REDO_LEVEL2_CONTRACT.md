# Sparse-recovery redo: frozen execution contract

This contract instantiates the Tier-2 coordinated plan in
`SAE_REDO_LEVEL2_TOPOLOGY.md`. It supersedes the former one-candidate
`ContextVectorSAE` contract. A result created under that former contract is not
part of this matrix.

## Frozen matrix

Every final cell is the Cartesian product:

```text
backbone field × SAE family × original-node budget × seed
```

| Backbone family | Fields | SAE families | Budgets | Seeds |
|---|---|---|---|---|
| ResNet-20 / CIFAR-10 | Q1,Q2,Q3,Q4,Q5 | `field_old`, `field_strict`, `field_recovery`, `vector_context` | 0.08, 0.04, 0.02, 0.01 | 0,1,2 |
| ResNet-110 / CIFAR-100 | Q1,Q2,Q3,Q4,Q5 | same | same | same |
| `vit_small_patch16_224` / Imagenette | patch blocks 2,4,6,8,10 | same | same | same |
| `vit_base_patch16_224` / Imagenette | patch blocks 2,4,6,8,10 | same | same | same |

This is 960 final cells. ResNet-56, ViT-Large, an architecture-promotion
decision, and a block-10-only shortcut are excluded.

## Shared exact-message invariant

Given each individual activation map `(C,H,W)`, every family uses:

```text
M = max(1, round(fraction * C * H * W)).
```

`M` is computed afresh per field shape and is the same integer for all four
families in that cell. Ranking standardizes scores independently per sample in
FP32 and uses deterministic `topk(..., sorted=False).indices` plus boolean
scatter. The saved indices define the support. A threshold comparison, a
realized-nonzero proxy, or an uncharged support side channel cannot define the
active count.

The old Field baseline remains `K=8C`; each new family is `K=4C` with
`d=min(2C,256)`. Consequently this is a matched-active-count comparison, not a
matched-latent-fraction comparison. Each result stores exact `M`, support bits
using `ceil(log2(K*H*W))`, signed value bits and quantizer, metadata bits, and
total bits per original activation node.

## Family definitions and permitted decoder inputs

| Family | Frozen architecture | Decoder may receive |
|---|---|---|
| `field_old` | Historical shallow three-block FieldSAE structural topology, `K=8C`; upgraded to the common signed score/value heads and trainer. | charged sparse coefficient field only |
| `field_strict` | Six stable Field blocks, dilation 1/1/2 then pooled 8x8-or-less attention then 1/4/1; strict factorized 3/5/9/15 additive field decoder, unit-normalized effective atoms. | charged sparse coefficient field only |
| `field_recovery` | `field_strict` plus zero-initialized six-block 1/1/2/1/4/1 correction. | charged sparse coefficient field and transmitted support mask only |
| `vector_context` | Four stable convolution blocks 1/2 then pooled attention then 4/1; unit-normalized direct 1x1 vector atoms plus zero-initialized four-block 1/2/4/1 refiner. | charged sparse coefficient field only |

Both new-family attention branches pool to at most 8x8 tokens, use width 128,
fixed 2D positions, one pre-norm four-head attention/MLP-2 block, bilinear
upsampling, and a zero-initialized projection. Every family has separate,
unconstrained 1x1 score and signed-value heads.

No decoder receives an original activation, encoder hidden state, unmasked
values, selection logits, source residual, or hidden data-dependent bypass.
Shape/layout metadata is only an artifact/replay aid, never a reconstruction
input. ViT prefix tokens are preserved in the frozen backbone and excluded from
the encoded node count.

`field_recovery` initializes from the validation-selected `field_strict` parent
in the identical backbone/field/budget/seed cell. `vector_context` trains its
strict direct-vector path first, then enables its zero-initialized refiner.
These prerequisites are mandatory, not decisions that can omit a family.

The per-path budget graph is fixed:

```text
initialize -> 8% -> 4% -> 2% -> 1%
```

`field_old`, `field_strict`, and the strict-vector path use the selected parent
checkpoint at every arrow. `field_recovery` at a budget starts from the
matching `field_strict` checkpoint at that same budget; `vector_context` starts
from the matching strict-vector checkpoint at that same budget. Thus, no final
4%, 2%, or 1% cell starts from random weights.

## Training, data, and selection

Every family uses the improved curriculum exactly as detailed in
`.brainstorm/trainingSAE.md`: dense strict stabilization (5 epochs), selector
bootstrap (1), exact 50%-latent easy budget (5), geometric continuation (10),
and final-budget convergence (minimum 20 final epochs; 30 planned). Lower
budgets continue from a selected higher-budget checkpoint; never train 1% from
random initialization. AdamW, warmup/cosine, clipping, FP32 support/loss,
bf16 when supported, atom normalization, bounded collapse recovery, and
raw-versus-EMA validation choice follow that same document.

Recovery continuations retain the exact target budget: warm only the new
correction/refiner three epochs, then jointly refine twenty epochs at peak
learning rate `1e-4`. A third collapse after the two bounded retries is a
terminal failed seed, never an unrecorded reschedule.

Fit all model parameters and normalization on train only. Select checkpoint,
EMA choice, and any bounded recovery using validation only. CIFAR test data is
never accessed in training/validation. It is a label-free fidelity holdout:
compare the original frozen-model output with the sparse-spliced output on the
same images. For Imagenette, sort official validation image IDs, stratify by
class using fixed seed `20260827`, and split them 50/50 into checkpoint-
selection validation and the fidelity holdout. Save both ordered manifests and
hashes before model construction. After all configuration choices are frozen,
each selected checkpoint receives exactly one fidelity-holdout evaluation.

## Required artifacts and completed-cell condition

A completed result directory contains the resolved configuration, split and
normalization manifests/hashes, frozen-backbone/cache hash, seed, complete
checkpoint(s), selected indices/values and quantization metadata, logs,
validation and fidelity-holdout metrics, rate accounting, replay result, and a completion
manifest with content hashes.

It reports reconstruction loss, relative L2, FVU, cosine, per-channel errors,
prediction agreement, KL, correction/direct norms, feature
usage, collapse status, parameters/FLOPs/memory/throughput, and all sparse-rate
terms. Failed cells preserve the same evidence plus recovery-attempt state;
they never overwrite the parent or prior attempt.

Every selected configuration also receives the named support/value/decoder
controls from the coordinated plan. Strict Field linearity, atom normalization,
no-bypass, and decoder-only replay are hard pre-launch checks.

## Required post-matrix CNN re-grounded chains

For each CNN backbone, choose one new family for the chain phase. It is eligible
only if at least two of three 1%-budget seeds are non-collapsed in every Q1–Q5
section. Among eligible families, choose the highest mean validation prediction
agreement across Q1–Q5; lower mean KL breaks a tie. Compare it with `field_old`
at the same 1% original-node budget. This validation choice is frozen before
any chain fidelity-holdout evaluation.

For the selected family and baseline, use their Q1–Q5 SAE checkpoints to train
the four adjacent transition predictors with clean train/validation/fidelity-holdout
manifests, scheduled sampling, gradient clipping, and **full BPTT through
re-grounding**. This is the mandatory Family-II implementation, not the older
post-hoc re-grounded evaluation. At every transition destination, re-grounding
is exactly:

```text
predicted code -> destination SAE decode -> destination normalization
-> destination SAE encode -> exact destination original-node M Top-M
```

For `zhat_b = T_a_to_b(z_a)`, the carrier is computed in graph as
`exact_topM_b(E_b(normalize_b(D_b(zhat_b))))`; it is the next predictor input.
The frozen SAE parameters do not update, but gradients from later transition
losses must reach `zhat_b` through decoder, encoder, and the hard-forward
straight-through Top-M surrogate. The carrier must not be detached, placed
under `no_grad`, or formed from true destination activation `H_b`. Ground-truth
destination activations are allowed only as supervised reconstruction/code
targets, never as a carrier input.

The transition chain never uses the historical latent-fraction mask helper.
It uses the exact destination original-node `M`. A required matched raw-carrier
comparator keeps the same scheduled-sampling schedule, full rollout BPTT,
losses, optimizer, clipping, seeds, and data split, but supplies `zhat_b`
directly to the next transition. Three transition seeds report raw rollout,
exact re-mask-only rollout, full-BPTT re-grounded rollout, frozen-block hybrid
upper bound, every adjacent edge, and end-to-end Q1→Q5 behavior.

Before a chain launch, a focused smoke must prove finite nonzero gradients from
every transition loss through the re-grounding carrier, exact destination `M`
under tied scores, and carrier independence from a perturbation of `H_b` after
`zhat_b` is constructed. This required phase applies to the ordered CNN
sections only, not to an invented ViT equivalent.

## Run roots and static GPU queues

Use `runs/sae_sparse_recovery_redo/<UTC timestamp>/` with a unique hierarchy:

```text
<domain>/<backbone>/<section-or-block>/<family>/b<budget>/seed<seed>/
```

Five tmux sessions pin `CUDA_VISIBLE_DEVICES` and run independent static queue
files. Each runner executes a line to completion, durably records start/end,
exit code and output path, then advances without agent polling. Parent and
dependent continuation commands form one ordered per-cell chain; unrelated
chains are load-balanced across queues. A resume pass admits only a cell whose
completion manifest verifies; partial and failed directories remain immutable.

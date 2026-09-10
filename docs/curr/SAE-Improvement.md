# SAE Improvement

Last updated: 2026-06-26

This note records the current recommended improvement path for the SAE training and architecture, with a special focus on improving the `Field` SAE for ViT activation recovery.

The main context behind this document is:

- `ViT-small` showed real signal, but weaker results than the CNN side
- `ViT-base` showed a strong non-collapse regime
- the current evidence suggests that a large part of the remaining problem is SAE training / stability rather than a clean falsification of the sparsity hypothesis

The practical strategy is:

- use `ViT-small` as a cheap debugging sandbox
- apply successful changes to `ViT-base`
- prioritize reducing collapse and improving reconstruction quality before major architectural overhauls

## Main diagnosis

The current `Field` SAE is promising, but it likely has multiple limitations:

1. ViT training is currently harsher than the CNN-side SAE training.
2. The current `Field` SAE may be too shallow / too narrow for complex transformer activation maps.
3. The current local receptive field may be too limited for ViT token-grid structure.
4. The current latent and masking setup may make optimization harder than necessary.

## Recommended ordering

These are ordered from easiest / most likely to succeed to hardest / least likely to succeed.

### 1. Add the CNN-style sparsity curriculum to `vit_sae.py`

Priority: highest

Idea:

- warmup with no mask
- anneal from an easy retained fraction down to the target
- then hold the target sparsity

Why:

- the current ViT path trains under the target mask immediately
- this likely makes optimization much harder than the CNN path
- this is the simplest high-probability fix for collapse and weak local minima

Recommended first version:

- warmup: `5` epochs, no mask
- anneal: `10` epochs, from `50%` retained to target
- hold: `10-20` epochs at target

### 2. Add better optimization hygiene

Idea:

- switch to `AdamW`
- add an LR schedule, such as cosine decay
- add gradient clipping
- optionally evaluate EMA checkpoints

Why:

- collapse may be partly optimizer-sensitive
- these changes are cheap and often help stability with little architectural risk

### 3. Add validation-driven model selection

Idea:

- save best checkpoint by `rel_l2`
- save best checkpoint by `pred_agree`
- save best checkpoint by a combined score

Why:

- the current best endpoint is not guaranteed to be the best behavior-preserving checkpoint
- this is easy and often improves outcomes without changing the model

### 4. Increase `Kmult`

Idea:

- try `Kmult = 8`
- then `12`
- maybe `16` for the strongest settings

Why:

- ViT activations may need a larger overcomplete basis
- the sparse support can still stay small in percentage terms while the dictionary becomes richer

### 5. Make the `Field` SAE deeper first, then wider

Idea:

- first increase depth while keeping the overall architecture the same
- then increase width if depth alone helps but is still not enough

Why depth should come first:

- the current `Field` SAE may simply be underpowered
- this is the least disruptive architectural hypothesis to test
- for large, complex activation maps, `3` residual local blocks per side may simply not be expressive enough
- if a deeper plain `Field` SAE already improves performance, that is a cleaner and more interpretable result than immediately jumping to more exotic block designs

Why:

- the current architecture may simply be underpowered for transformer maps
- this is the simplest capacity increase before redesigning the whole model

Recommended first settings:

- `d = 2C`, `n_blocks = 6`
- `d = 2C`, `n_blocks = 8`
- then `d = 4C`, `n_blocks = 6`
- then `d = 4C`, `n_blocks = 8`

### 6. Add a small downstream KL term after reconstruction stabilizes

Idea:

- first optimize only reconstruction
- later add a small `KL(orig || spliced)` term

Why:

- reconstructions with similar `rel_l2` can have different downstream quality
- this may push the SAE toward behaviorally important structure

Recommended first version:

- first half of training: pure reconstruction
- second half: `loss = recon + lambda * KL`
- start with a small `lambda`, e.g. `0.02` to `0.1`

### 7. Add explicit decoder regularization / normalization for `Field`

Idea:

- explicit decoder atom normalization
- decoder norm penalty
- feature diversity / decorrelation penalty

Why:

- this may reduce unstable dictionary geometry
- it may also help prevent degenerate features and improve optimization

### 8. Replace nonnegative-only latent with signed sparse codes

Idea:

- use top-k on `abs(z)` rather than only positive activations
- or split the code into positive / negative channels

Why:

- transformer activations may be more naturally expressed with signed components
- nonnegative-only latents may make reconstruction unnecessarily hard

Why this is not earlier:

- it changes the sparse-code semantics more deeply
- it is more intrusive than the curriculum / optimizer / width fixes

### 9. Make the `Field` SAE multiscale

Idea:

- add a small U-Net / hourglass structure
- or use dilations / mixed kernels / downsample-upsample processing

Why:

- ViT token maps may need larger effective receptive fields
- a purely local stack may miss broader structure

Why this is later:

- more code
- more tuning
- more ways to accidentally destabilize training

### 10. Major redesigns

Examples:

- attention inside the SAE
- hybrid field/vector blocks
- more global latent structures

Why this is last:

- highest engineering cost
- least predictable first-step return
- should only happen after the simpler fixes are exhausted

## Recommended first implementation order

If time and compute are limited, do this order:

1. add sparsity curriculum to `vit_sae.py`
2. add optimizer / scheduler / clipping upgrades
3. add validation-based checkpoint selection
4. run a `Kmult` sweep
5. widen / deepen `Field`
6. add downstream KL fine-tuning
7. only then move to signed-latent and multiscale variants

## Practical interpretation

The most likely near-term story is:

- current `Field` SAE can work
- current ViT training is leaving performance on the table
- fixing training may recover a substantial amount of the gap before large architecture changes are necessary

That is why the first effort should be:

- training fixes first
- deeper / wider plain `Field` second
- architecture redesign third

## How to use this note

For future agents:

- use this as the priority order for SAE-improvement work
- test changes first on `ViT-small`
- promote successful changes to `ViT-base`
- treat `ViT-base` stability as the main paper-facing target

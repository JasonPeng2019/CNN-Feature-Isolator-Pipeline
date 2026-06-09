# Phase 3 Results — Sparse-code hierarchy (Family I + N4 hybrid)

**Date:** 2026-05-31 · Backbone ResNet-56/CIFAR-100 71.83%. Winner SAE = Field (K=8C, 3 blocks), global mask @ 5% retained. orig top-1 = 0.7183.

Family I: freeze SAEs at t and t+1; learn `T_t: Z~_t → Ẑ_{t+1}` (target = independently-learned `Z_{t+1}` + decoded recon of `H_{t+1}`).

## Single-step learned transitions (strong)

| transition | learned spliced top-1 | N4 hybrid (causal bound) | code-MSE | recon relL2 |
|---|---|---|---|---|
| Q1→Q2 | **0.715** | 0.715 | 0.008 | 0.201 |
| Q2→Q3 | 0.691 | 0.711 | 0.044 | 0.428 |
| Q3→Q4 | 0.650 | 0.711 | 0.201 | 0.538 |
| Q4→Q5 | 0.709 | 0.715 | 0.059 | 0.408 |

Q1→Q2 learned transition **matches the causal upper bound exactly**. Deeper, resolution-dropping steps (esp. Q3→Q4, 16²→8²) are harder for the learned predictor.

## Chained Q1→Q5

| variant | top-1 |
|---|---|
| raw learned chain (feed prediction straight on) | **0.012** (chance) |
| + re-mask predicted code each step | 0.213 |
| + re-ground (decode→re-encode→mask) each step | **0.593** |
| N4 frozen-block-hybrid chain (real blocks) | **0.712** |
| original | 0.718 |

## Interpretation — the key result

1. **The information is sufficient (Claims 2-3 supported).** The N4 hybrid chain — propagate the *masked* sparse code through the *real* frozen blocks all the way to Q5 — preserves **0.712** (−0.6 pp). A 5%-retained sparse support at every section is enough to drive the network to near-original accuracy.
2. **Independent-SAE learned chains fail from exposure bias, not missing information.** The raw learned chain collapses to chance (0.012) because each predictor was trained on the true masked code `Z~_t` but at chain-time receives the previous predictor's *dense, off-manifold* output, and errors compound over 4 steps.
3. **Re-grounding nearly fixes it (0.012 → 0.593).** Projecting the predicted code back onto the SAE manifold (decode→re-encode→mask) each step removes most of the compounding error. Re-masking alone recovers 0.012 → 0.213.
4. The residual **0.593 → 0.712** gap is genuine learned-predictor error, concentrated in Q3→Q4.

## Verdict & next step

This is the doc's own contingency made concrete: **independent SAEs reconstruct and transition single-step, but Family-I chains need re-grounding.** The principled fix is **Family II (incremental dependent codes)** — train predictors exposed to their own (re-grounded) predicted inputs (scheduled sampling) and/or train the chain end-to-end with per-step reconstruction. Phase 3b implements this, initialized from the Family-I predictors.

New experiments this motivates: **N-exposure** (scheduled sampling during transition training), **N-reground** (build re-grounding into the architecture as the default), and quantifying the per-step error-compounding curve.

---

## Addendum — Focused `Q3 -> Q4` strong-predictor retry

**Artifact:** `runs/T1_3_Q3Q4_strong/result.json`

Because `Q3 -> Q4` was the clear weakest edge in the original transition table, this follow-up retrained that single step with a stronger predictor (`depth=3`, `hidden=512`, `30` epochs, `lr=0.002`).

### Final result

| transition | learned spliced top-1 | pred agree | KL(orig‖spliced) | relL2 | hybrid top-1 |
|---|---:|---:|---:|---:|---:|
| Q3→Q4 strong retry | **0.5344** | 0.5790 | 1.4641 | 0.6454 | 0.7113 |

### Interpretation

1. **Naively making the predictor larger did not fix the bottleneck.** The strong retry finishes at `0.5344`, which is materially worse than the original Phase 3 `Q3 -> Q4` learned transition (`0.650`).
2. **The hard part is not just capacity.** Even with a deeper/wider predictor and longer training, the learned path remains far from the frozen-block causal upper bound (`0.7113`).
3. **This strengthens the original diagnosis.** The `16^2 -> 8^2` step is not a bottleneck that disappears with a straightforward MLP upgrade; it likely needs a qualitatively better transition mechanism, such as explicit multi-scale structure, architecture-aware downsampling, or chain-aware training.

So the original Phase 3 conclusion still stands, and this follow-up makes it sharper: `Q3 -> Q4` is the stubborn edge, and simple predictor scaling is not an adequate remedy.

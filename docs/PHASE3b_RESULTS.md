# Phase 3b Results — Family II chain-aware training (scheduled sampling + re-grounding)

**Date:** 2026-05-31. Warm-started from Family-I predictors; frozen Field SAEs @5%; 25 epochs, scheduled-sampling p 0→1.

## Chained Q1→Q5 top-1

| chain variant | Family I (Phase 3) | Family II (Phase 3b) | causal bound (hybrid) | orig |
|---|---|---|---|---|
| raw learned (no re-grounding) | 0.012 | 0.013 | — | — |
| **re-grounded each step** | 0.593 | **0.662** | 0.712 | 0.718 |

## Conclusions (definitive for this backbone)

1. **Chain-aware training helps the re-grounded chain (+6.9 pp, 0.593→0.662)**, narrowing the gap to the causal upper bound (0.712). Scheduled sampling on re-grounded predicted inputs is a real improvement over independent Family-I transitions.
2. **Re-grounding is necessary, not optional.** The raw code→code chain stays at chance (0.013) regardless of training. A purely learned dense predictor cannot keep its output mask-consistent and on the SAE sparse-code manifold across 4 compositions; snapping back via decode→re-encode→mask each step is required.
3. **The information is there throughout** (hybrid chain 0.712, −0.6 pp from orig): a 5%-retained sparse support at every section is sufficient to drive the real frozen network to near-original accuracy. The sparsity-tree hypothesis (Claims 2-3) is supported in its re-grounded / causal form.

## What this means for the research claim

The defensible statement: *vision-model activation fields are compressible into a learned sparse transform domain (Field SAE) such that a small feature-location subset reconstructs each layer and preserves downstream behavior; and sparse codes at one section generate the next when re-grounded on the SAE manifold (or propagated through the real blocks), giving evidence for a learned, depth-wise sparsity tree.* The pure learned-code chain is unstable — re-grounding is the mechanism that makes the hierarchy usable.

## Open follow-ups (motivated by these results)
- **Q3→Q4 is the bottleneck** (single-step learned 0.650). The 16²→8² resolution drop is where the predictor loses most. Worth a dedicated architecture (better downsampling predictor) or N1 Matryoshka multi-scale codes.
- **N8 parent→child intervention** to test causal tree structure, not just predictive sufficiency.
- **Architectural re-grounding**: bake the decode→re-encode→mask into the predictor as a differentiable layer (train through it) — may close the 0.662→0.712 gap.
- **Family II from-scratch** (not warm-started) + full BPTT through re-grounding.

---

## Addendum — Full-BPTT Family II follow-up

**Artifact:** `runs/phase3b_bptt/result.json`

This later follow-up keeps the Phase 3b re-grounded carrier, but pushes the chain-aware training further with explicit full BPTT through the rollout.

## Chained Q1→Q5 top-1

| chain variant | Phase 3b original | Phase 3b + BPTT | causal bound (hybrid) | orig |
|---|---:|---:|---:|---:|
| raw learned (no re-grounding) | 0.0128 | 0.0109 | — | — |
| **re-grounded each step** | 0.6624 | **0.7010** | 0.7122 | 0.718 |

## What changed

1. **Full-BPTT closes most of the remaining gap.** The re-grounded chain improves from `0.6624` to **`0.7010`**, leaving only about `1.1` points to the hybrid upper bound (`0.7122`).
2. **The main story is still about re-grounding, not raw rollout.** The raw chain remains at chance (`0.0109`), so stronger optimization alone does not make manifold projection optional in this setup.
3. **Family II is now much closer to a practical chain.** With re-grounding plus rollout-aware optimization, the learned chain is no longer just “better than Family I”; it is close to the best causal path available without handing control back to the frozen real blocks.

## Updated takeaway

The conservative Phase 3b conclusion was that chain-aware training helps but does not fully resolve the hierarchy gap. The newer BPTT follow-up strengthens that: **chain-aware training plus re-grounding gets very close to the causal upper bound**, while the pure raw-code chain still fails. So the most defensible current claim is not merely that re-grounding stabilizes the hierarchy; it is that re-grounded Family II transitions can carry almost all of the useful signal across the full `Q1 -> Q5` chain.

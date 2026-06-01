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

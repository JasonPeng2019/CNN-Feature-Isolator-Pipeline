# Phase 4 Results

## Purpose

Phase 4 is the extension and recovery stage of the project. These runs answer a different question than Phases 1--3b: not whether the core winner path works at all, but whether its conclusions survive scale change, seed variation, architecture shift, and a first transformer-side transfer attempt.

This writeup is rebuilt from the surviving run artifacts after the earlier Phase 4 markdown summary went missing from `docs/`.

## Status Summary

The Phase 4 artifacts are present and complete for the main rerun families:

- `runs/r20c10/grid_summary.json`
- `runs/seeds/grid_summary.json`
- `runs/taxonomy/taxonomy_q4q5.json`
- `runs/r110/F_Q1` through `runs/r110/F_Q5`
- `runs/vit_sae/vit_result.json`

All of the main Phase 4 launcher summaries under `LISP-3-Setup/logs/` report `rc=0`.

## Experiment 1: ResNet-20 / CIFAR-10 Grid (`r20c10`)

### What it tested

This is a scale-down and dataset-shift stress test of the winner recipe:

- backbone: ResNet-20
- dataset: CIFAR-10
- sections: `Q1..Q5`
- retained fractions: `2%`, `5%`, `10%`
- SAE style: `Field + Global`

### Completion

- `15/15` cells completed successfully

### Final results

| Section | 2% | 5% | 10% |
|---|---:|---:|---:|
| Q1 | 0.9220 | 0.9217 | 0.9221 |
| Q2 | 0.9183 | 0.9222 | 0.9221 |
| Q3 | 0.9203 | 0.9210 | 0.9228 |
| Q4 | 0.9224 | 0.9227 | 0.9225 |
| Q5 | 0.9228 | 0.9226 | 0.9228 |

### Interpretation

- Deep sections stay extremely strong under this backbone/dataset shift.
- Performance is basically saturated by `5%` retained in `Q3--Q5`.
- The sparse-sufficiency story is therefore not fragile to the original ResNet-56 / CIFAR-100 anchor.

## Experiment 2: Seed-Stability Grid (`seeds`)

### What it tested

This is a robustness check on the winner setup:

- backbone: ResNet-56
- dataset: CIFAR-100
- retained fraction: `5%`
- seeds: `0, 1, 2`
- sections: `Q1..Q5`

### Completion

- `15/15` cells completed successfully

### Final results by section

| Section | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---:|---:|---:|---:|
| Q1 | 0.7173 | 0.7180 | 0.7181 | 0.7178 |
| Q2 | 0.7174 | 0.7148 | 0.7143 | 0.7155 |
| Q3 | 0.7108 | 0.7142 | 0.7136 | 0.7129 |
| Q4 | 0.7184 | 0.7182 | 0.7151 | 0.7172 |
| Q5 | 0.7174 | 0.7167 | 0.7165 | 0.7169 |

### Interpretation

- The winner-path conclusions are not single-seed accidents.
- Variance exists, but the spread is small enough that the qualitative story does not change.
- `Q3` remains the weakest of the later sections, but not catastrophically so.

## Experiment 3: Transition Taxonomy at `Q4 -> Q5` (`taxonomy_q4q5`)

### What it tested

This run compares what kind of representation best carries the `Q4 -> Q5` relationship:

- sparse
- dense
- cross

### Final results

| Condition | Spliced Top-1 | Pred Agree | KL |
|---|---:|---:|---:|
| sparse | 0.7063 | 0.8415 | 0.2404 |
| dense | 0.6902 | 0.7897 | 0.3815 |
| cross | 0.6099 | 0.6739 | 0.9754 |

### Interpretation

- Sparse is the best transition carrier in this controlled comparison.
- This matters because it supports a stronger claim than “sparse can reconstruct.”
- It suggests the sparse representation is itself the most behavior-preserving transition format among the tested alternatives.

## Experiment 4: ResNet-110 Reconstruction Sweep (`r110`)

### What it tested

This is a deeper-backbone generalization test of the winner recipe:

- backbone: ResNet-110
- dataset: CIFAR-100
- sections: `Q1..Q5`
- retained fraction: `5%`
- SAE style: `Field + Global`

### Completion

- `5/5` section runs completed successfully

### Final results

| Section | Spliced Top-1 | Pred Agree | Frac Retained | Rel L2 |
|---|---:|---:|---:|---:|
| Q1 | 0.7316 | 0.9804 | 0.0498 | 0.0312 |
| Q2 | 0.7311 | 0.9701 | 0.0500 | 0.1065 |
| Q3 | 0.7226 | 0.9356 | 0.0500 | 0.1691 |
| Q4 | 0.7290 | 0.9717 | 0.0500 | 0.1189 |
| Q5 | 0.7318 | 0.9804 | 0.0500 | 0.1082 |

### Interpretation

- The reconstruction story survives transfer to a deeper CNN.
- `Q3` is still the hardest section.
- That persistence is scientifically useful: the project’s hardest region is not a one-backbone fluke.

## Experiment 5: ViT SAE Transfer (`vit_sae`)

### What it tested

This is the first transformer-side transfer artifact:

- model: `vit_small_patch16_224`
- block: `6`
- retained fraction: about `5%`

### Final results

- `pred_agree = 0.9440`
- `kl = 0.0427`
- `frac_retained = 0.0500`
- `rel_l2 = 0.4172`
- `cos = 0.9085`

### Interpretation

- This is a strong self-consistency result: after sparse reconstruction at only `5%` retained, the modified ViT keeps the same top prediction as the original ViT on `94.4%` of validation examples.
- The KL is low enough to suggest the full output distribution usually stays close as well.
- This is not yet a full task-accuracy transfer story, but it is a real artifact and a nontrivial proof that the codepath works.

## Overall Phase 4 Takeaways

Phase 4 substantially strengthens the whole project:

1. The main winner-path story survives backbone and dataset shift (`r20c10`).
2. The winner-path story is not a seed accident (`seeds`).
3. Sparse is not only reconstructive but also a particularly strong transition carrier (`taxonomy_q4q5`).
4. The deeper-backbone story remains consistent, including the persistent `Q3` bottleneck (`r110`).
5. The transfer path to ViTs moved from “blocked” to “real executed artifact” (`vit_sae`).

## Bottom Line

The correct current interpretation is that Phase 4 was not lost. The writeup was missing, but the artifacts show that the reruns and extensions completed successfully and materially strengthened the empirical support for the project’s main claims.

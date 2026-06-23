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

### Completion:

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

## Experiment 6: ViT-Small Sweep on Imagenette (`vit_sweep`)

### What it tested

This is the first full transformer-side sweep rather than a single transfer artifact:

- model: `vit_small_patch16_224`
- blocks: `2, 4, 6, 8, 10`
- retained fractions: `1%`, `2%`, `5%`, `8%`, `12%`
- seeds: `0, 1, 2`
- train subset: `2000` Imagenette train images per run
- validation subset: `500` Imagenette val images per run

### Completion

- `75/75` jobs completed successfully
- `75/75` `vit_result.json` files were recovered
- no missing or duplicate `(block, frac, seed)` cells were found in the sweep

### Headline results

- Mean `pred_agree` across all `75` runs: `0.8914`
- Mean `kl` across all `75` runs: `0.1082`
- Best single run: `block=10`, retained fraction about `12%`, `seed=1`, `pred_agree = 0.9340`
- Meaningful maximum for this metric: `pred_agree = 1.0000`, since the target is exact agreement with the original ViT prediction on each validation example
- Previous one-off ViT artifact for reference: `pred_agree = 0.9440` on the earlier single transfer run (`vit_sae`)

### Best stable settings (mean over 3 seeds)

| Block | Retained | Mean Pred Agree | Seed Std | Mean KL | Mean Rel L2 |
|---|---:|---:|---:|---:|---:|
| 10 | 12% | 0.9307 | 0.0034 | 0.0361 | 0.3558 |
| 4 | 8% | 0.9273 | 0.0038 | 0.0463 | 0.3814 |
| 10 | 8% | 0.9187 | 0.0034 | 0.0456 | 0.4177 |
| 4 | 12% | 0.9173 | 0.0050 | 0.0403 | 0.3188 |

The best stable cell is therefore:

- `block=10`, retained `12%`
- mean `pred_agree = 0.9307`
- seed std `= 0.0034`
- per-seed values: `0.9260`, `0.9340`, `0.9320`
- gap to meaningful max (`1.0000`): `0.0693`

### Aggregate trends

Mean `pred_agree` by block:

- block `2`: `0.8916`
- block `4`: `0.9024`
- block `6`: `0.8792`
- block `8`: `0.8752`
- block `10`: `0.9087`

Mean `pred_agree` by retained fraction:

- `1%`: `0.8569`
- `2%`: `0.8827`
- `5%`: `0.8971`
- `8%`: `0.9075`
- `12%`: `0.9129`

Retained-fraction view relative to the meaningful max (`pred_agree = 1.0000`):

| Retained | Mean Pred Agree | Seed Std | Mean KL | Gap To Max |
|---|---:|---:|---:|---:|
| 1% | 0.8569 | 0.0193 | 0.2129 | 0.1431 |
| 2% | 0.8827 | 0.0151 | 0.1317 | 0.1173 |
| 5% | 0.8971 | 0.0143 | 0.0855 | 0.1029 |
| 8% | 0.9075 | 0.0163 | 0.0619 | 0.0925 |
| 12% | 0.9129 | 0.0118 | 0.0490 | 0.0871 |

### Full mean table (all block × retained combinations)

| Block | Retained | Mean Pred Agree | Seed Std | Mean KL |
|---|---:|---:|---:|---:|
| 2 | 1% | 0.8520 | 0.0150 | 0.2097 |
| 2 | 2% | 0.8873 | 0.0100 | 0.1218 |
| 2 | 5% | 0.8953 | 0.0077 | 0.0754 |
| 2 | 8% | 0.9107 | 0.0118 | 0.0525 |
| 2 | 12% | 0.9127 | 0.0034 | 0.0405 |
| 4 | 1% | 0.8653 | 0.0090 | 0.1740 |
| 4 | 2% | 0.8967 | 0.0050 | 0.1030 |
| 4 | 5% | 0.9053 | 0.0025 | 0.0627 |
| 4 | 8% | 0.9273 | 0.0038 | 0.0463 |
| 4 | 12% | 0.9173 | 0.0050 | 0.0403 |
| 6 | 1% | 0.8467 | 0.0127 | 0.2377 |
| 6 | 2% | 0.8720 | 0.0071 | 0.1507 |
| 6 | 5% | 0.8840 | 0.0085 | 0.0999 |
| 6 | 8% | 0.8907 | 0.0066 | 0.0745 |
| 6 | 12% | 0.9027 | 0.0025 | 0.0628 |
| 8 | 1% | 0.8367 | 0.0050 | 0.3004 |
| 8 | 2% | 0.8627 | 0.0050 | 0.1827 |
| 8 | 5% | 0.8853 | 0.0075 | 0.1241 |
| 8 | 8% | 0.8900 | 0.0000 | 0.0908 |
| 8 | 12% | 0.9013 | 0.0082 | 0.0653 |
| 10 | 1% | 0.8840 | 0.0049 | 0.1427 |
| 10 | 2% | 0.8947 | 0.0082 | 0.1002 |
| 10 | 5% | 0.9153 | 0.0109 | 0.0653 |
| 10 | 8% | 0.9187 | 0.0034 | 0.0456 |
| 10 | 12% | 0.9307 | 0.0034 | 0.0361 |

### Interpretation

- The original ViT artifact was not a fluke: the sweep shows a broad region of stable, nontrivial self-consistency rather than a single lucky cell.
- Later blocks are generally better than middle blocks, but the best regime is not simply “deeper is always better”; block `4` is unusually strong and competitive with block `10`.
- Increasing retained fraction helps monotonically on average, with a clear improvement from `1%` to `8–12%`.
- The most paper-robust current claim is therefore not “one ViT cell worked,” but that **ViT token-grid sparse reconstruction is stable across multiple blocks and seeds, with best settings sustaining about `0.93` prediction agreement on Imagenette validation.**

## Experiment 7: ViT-Base Sweep on Imagenette (`vit_base_imagenette_full`)

### What it tested

This is the first completed larger-model ViT sweep in the repo:

- model: `vit_base_patch16_224`
- blocks: `2, 4, 6, 8, 10`
- requested retained fractions: `2%`, `5%`, `8%`, `12%`
- seeds: `0, 1, 2`
- train subset: `2000` Imagenette train images per run
- validation subset: `500` Imagenette val images per run

### Completion

- `60/60` jobs completed successfully
- `60/60` `vit_result.json` files were recovered
- no duplicate requested `(block, frac, seed)` triplets were found

### Headline results

- Best single run: `block=10`, requested retained fraction `12%`, `seed=1`, `pred_agree = 0.9862`
- Meaningful maximum for this metric: `pred_agree = 1.0000`
- Larger-model best stable setting gap to max: only `0.0149`
- The raw mean across all `60` runs is only `0.6606`, but that average is heavily distorted by seed-collapse cells and should not be read as the main story

### Best stable settings (mean over 3 seeds)

| Block | Requested Retained | Mean Pred Agree | Seed Std | Mean KL | Mean Rel L2 | Collapsed Seeds |
|---|---:|---:|---:|---:|---:|---:|
| 10 | 12% | 0.9851 | 0.0012 | 0.0063 | 0.2932 | 0 |
| 10 | 8% | 0.9790 | 0.0023 | 0.0101 | 0.3583 | 0 |
| 2 | 8% | 0.9787 | 0.0014 | 0.0077 | 0.3342 | 0 |
| 2 | 5% | 0.9745 | 0.0006 | 0.0107 | 0.3924 | 0 |
| 10 | 2% | 0.9690 | 0.0015 | 0.0222 | 0.4570 | 0 |

The best stable cell is therefore:

- `block=10`, requested retained `12%`
- mean `pred_agree = 0.9851`
- seed std `= 0.0012`
- per-seed values: `0.9857`, `0.9862`, `0.9834`
- mean `KL = 0.0063`
- gap to meaningful max (`1.0000`): `0.0149`

### Aggregate trends

Mean `pred_agree` by requested retained fraction:

- `2%`: `0.7009`
- `5%`: `0.7704`
- `8%`: `0.6572`
- `12%`: `0.5139`

Mean `pred_agree` by block:

- block `2`: `0.8101`
- block `4`: `0.3220`
- block `6`: `0.6306`
- block `8`: `0.5689`
- block `10`: `0.9714`

Collapse-prone requested settings (`pred_agree < 0.5` on at least one seed): `12 / 20`

Notable collapse examples:

- block `4`, `12%`: seed values `0.9743`, `0.0000`, `0.0000`
- block `8`, `12%`: seed values `0.0166`, `0.9654`, `0.9676`
- block `2`, `12%`: seed values `0.9804`, `0.0000`, `0.0003`
- block `6`, `8%`: seed values `0.9661`, `0.9682`, `0.0025`

### Realized-vs-requested sparsity note

This sweep exposed a real analysis wrinkle: the realized `frac_retained` recorded in the run JSON can drift materially from the requested sweep fraction in collapse-prone seeds. The analysis therefore groups runs by the requested fraction recovered from the run directory name, not by rounded realized `frac_retained`.

Examples:

- requested `5%`, block `10`, one seed realized only about `2.78%`
- requested `8%`, block `6`, one seed realized only about `2.99%`
- requested `12%`, block `2`, one seed realized only about `3.36%`

### Full mean table (all block × requested retained combinations)

| Block | Requested Retained | Mean Pred Agree | Seed Std | Mean KL | Collapsed Seeds |
|---|---:|---:|---:|---:|---:|
| 2 | 2% | 0.9602 | 0.0026 | 0.0254 | 0 |
| 2 | 5% | 0.9745 | 0.0006 | 0.0107 | 0 |
| 2 | 8% | 0.9787 | 0.0014 | 0.0077 | 0 |
| 2 | 12% | 0.3269 | 0.4621 | 4.1514 | 2 |
| 4 | 2% | 0.3167 | 0.4470 | 3.8476 | 2 |
| 4 | 5% | 0.3222 | 0.4544 | 3.8807 | 2 |
| 4 | 8% | 0.3242 | 0.4584 | 4.1534 | 2 |
| 4 | 12% | 0.3248 | 0.4593 | 3.9818 | 2 |
| 6 | 2% | 0.6319 | 0.4468 | 1.8281 | 1 |
| 6 | 5% | 0.9620 | 0.0015 | 0.0208 | 0 |
| 6 | 8% | 0.6456 | 0.4547 | 1.8513 | 1 |
| 6 | 12% | 0.2830 | 0.3975 | 3.6029 | 2 |
| 8 | 2% | 0.6265 | 0.4365 | 1.6623 | 1 |
| 8 | 5% | 0.6409 | 0.4393 | 1.5287 | 1 |
| 8 | 8% | 0.3582 | 0.4292 | 2.8919 | 2 |
| 8 | 12% | 0.6499 | 0.4478 | 1.7070 | 1 |
| 10 | 2% | 0.9690 | 0.0015 | 0.0222 | 0 |
| 10 | 5% | 0.9524 | 0.0355 | 0.0783 | 0 |
| 10 | 8% | 0.9790 | 0.0023 | 0.0101 | 0 |
| 10 | 12% | 0.9851 | 0.0012 | 0.0063 | 0 |

### Interpretation

- The larger model can do much better than the ViT-small sweep when it lands in the stable regime: the best stable `ViT-Base` cell reaches about `0.985` prediction agreement versus about `0.931` for the best stable `ViT-small` cell.
- The larger model is also much less uniformly stable. Outside the late `block 10` regime, many settings become sharply bimodal across seeds, with one or two seeds preserving the original ViT behavior and another collapsing almost completely.
- `block 10` is the clean winner on the larger model. `block 2` also has several strong cells, but becomes fragile at requested `12%`.
- The raw across-cell averages are misleading on this run family because they conflate near-perfect cells with catastrophic collapse cells. The right summary is therefore "excellent stable late-block regime plus substantial off-regime seed fragility," not simply one scalar sweep average.
- A likely follow-up target is the realized-vs-requested sparsity drift itself: several collapse seeds retain far less than the requested budget, which may reflect mask-threshold / tie pathology or another seed-sensitive failure mode.

## Overall Phase 4 Takeaways

Phase 4 substantially strengthens the whole project:

1. The main winner-path story survives backbone and dataset shift (`r20c10`).
2. The winner-path story is not a seed accident (`seeds`).
3. Sparse is not only reconstructive but also a particularly strong transition carrier (`taxonomy_q4q5`).
4. The deeper-backbone story remains consistent, including the persistent `Q3` bottleneck (`r110`).
5. The transfer path to ViTs moved from “blocked” to “real executed artifact” (`vit_sae`).
6. The transformer-side path now has a completed sweep, and the completed sweep shows stable rather than seed-lucky behavior (`vit_sweep`).
7. The larger-model ViT path reaches a much stronger best-case regime, but it also reveals a new fragility story: `ViT-Base` is excellent at late blocks and substantially less stable elsewhere (`vit_base_imagenette_full`).

## Bottom Line

The correct current interpretation is that Phase 4 was not lost. The writeup was missing, but the artifacts show that the reruns and extensions completed successfully and materially strengthened the empirical support for the project’s main claims.

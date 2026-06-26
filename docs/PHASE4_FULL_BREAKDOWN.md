# Phase 4 Full Breakdown

## Purpose

This document is the long-form interpretation of Phase 4.

The goal is different from [PHASE4_RESULTS.md](/jumbo/lisp/f003x5w/ViT_Proj/CNN-SAE/docs/PHASE4_RESULTS.md), which is still a results summary. Here, the aim is to answer the more serious research question:

- what exactly was run
- what every result family looks like, not just the winners
- where the results are stable versus unstable
- what each experiment implies for the project hypothesis
- what is actually supported now versus what is still open

This writeup is deliberately more verbose and interpretive than the shorter result summaries.

## Reading Guide

Phase 4 is not one experiment. It is a collection of extension and robustness tests.

Each family is testing a different failure mode:

- `r20c10`: does the winner recipe survive backbone and dataset shift?
- `seeds`: is the winner path a seed accident?
- `taxonomy_q4q5`: is sparse only reconstructive, or also the best transition carrier?
- `r110`: does the winner path survive a deeper CNN?
- `vit_sae`: can the sparse reconstruction idea transfer to a ViT at all?
- `vit_sweep`: if the ViT transfer works once, does it work broadly across block / fraction / seed?
- `vit_base_imagenette_full`: does a larger ViT strengthen the claim or expose new fragility?

The central lesson of Phase 4 is that different kinds of robustness behaved differently:

- the CNN-side story is broadly stable
- the ViT-small story is broadly stable but not uniform
- the ViT-base story has a very strong best-case regime and a very real fragility regime

## Metric Interpretation

There are two main types of headline metric in Phase 4.

### `spliced_top1`

This is the classification accuracy after replacing a hidden representation with its sparse reconstruction and then continuing the frozen network forward.

Interpretation:

- high `spliced_top1` means the sparse reconstruction preserves downstream task performance
- if `spliced_top1` remains close to the original model's own accuracy, sparse reconstruction is behaviorally sufficient

### `pred_agree`

This is used heavily on the ViT side.

Interpretation:

- it measures exact agreement with the original model's top prediction
- the meaningful maximum is `1.0000`
- for ViT self-consistency, this is the cleanest "did the sparse reconstruction preserve the model's behavior?" metric

### `KL`

This captures how much the full output distribution changes.

Interpretation:

- low `KL` means the spliced model's entire distribution stays close to the original
- it is stricter than top-1 agreement, because two models can agree on the argmax while differing elsewhere

### `rel_l2` and `cos`

These are reconstruction-quality measures in hidden-state space.

Interpretation:

- lower `rel_l2` is better
- higher `cos` is better
- these help explain whether a result is strong because the hidden state is faithfully reconstructed or because the classifier is tolerant to distortion

## Phase 4 At A Glance

If Phase 4 is compressed to one paragraph, it says:

The sparse winner path is not a fragile one-off. It survives dataset and backbone shifts on the CNN side, remains stable across seeds, and is a better transition carrier than dense or cross alternatives in the tested taxonomy. The first ViT transfer artifact is real, the ViT-small sweep shows a broad stable regime, and the larger ViT-base sweep reaches an extremely strong late-block regime while also revealing substantial seed-sensitive instability outside that region.

That is the headline. The rest of this document explains what is behind it.

## Experiment 1: ResNet-20 / CIFAR-10 Grid (`r20c10`)

### What was done

This is the "scale down and dataset shift" stress test.

- backbone: `ResNet-20`
- dataset: `CIFAR-10`
- sections: `Q1..Q5`
- retained fractions: `2%`, `5%`, `10%`
- SAE family: `Field`
- mask family: `Global`

The point of this run is not to break new ground on CIFAR-10. It is to check whether the same sparse reconstruction recipe that worked on the original ResNet-56 / CIFAR-100 setup still works when both the backbone and dataset are changed.

### Results

These table entries are **spliced top-1 classification accuracy** on CIFAR-10 labels after replacing the section with the sparse reconstruction. They are **not** `pred_agree`.

Baseline for comparison:

- frozen ResNet-20 / CIFAR-10 backbone top-1 is about `0.923`
- for `pred_agree`, the meaningful maximum is `1.0000`

| Section | 2% | 5% | 10% |
|---|---:|---:|---:|
| Q1 | 0.9220 | 0.9217 | 0.9221 |
| Q2 | 0.9183 | 0.9222 | 0.9221 |
| Q3 | 0.9203 | 0.9210 | 0.9228 |
| Q4 | 0.9224 | 0.9227 | 0.9225 |
| Q5 | 0.9228 | 0.9226 | 0.9228 |

Supplementary reconstruction / agreement metrics:

| Section | Retained | Pred Agree | KL | Rel L2 | Cos |
|---|---:|---:|---:|---:|---:|
| Q1 | 2% | 0.9913 | 0.0035 | 0.0634 | 0.9981 |
| Q1 | 5% | 0.9937 | 0.0011 | 0.0435 | 0.9992 |
| Q1 | 10% | 0.9964 | 0.0006 | 0.0354 | 0.9994 |
| Q2 | 2% | 0.9710 | 0.0281 | 0.2439 | 0.9701 |
| Q2 | 5% | 0.9842 | 0.0092 | 0.1564 | 0.9878 |
| Q2 | 10% | 0.9933 | 0.0019 | 0.0526 | 0.9988 |
| Q3 | 2% | 0.9738 | 0.0234 | 0.2809 | 0.9597 |
| Q3 | 5% | 0.9878 | 0.0050 | 0.1368 | 0.9908 |
| Q3 | 10% | 0.9958 | 0.0011 | 0.0424 | 0.9993 |
| Q4 | 2% | 0.9852 | 0.0074 | 0.2141 | 0.9775 |
| Q4 | 5% | 0.9944 | 0.0012 | 0.0516 | 0.9991 |
| Q4 | 10% | 0.9941 | 0.0013 | 0.0471 | 0.9994 |
| Q5 | 2% | 0.9942 | 0.0012 | 0.1486 | 0.9888 |
| Q5 | 5% | 0.9966 | 0.0007 | 0.0870 | 0.9965 |
| Q5 | 10% | 0.9964 | 0.0007 | 0.0573 | 0.9989 |

### What it means

This is one of the cleanest robustness families in Phase 4.

The most important points:

- Performance is already very high at `2%`, especially for deeper sections.
- By `5%`, most of the grid is basically saturated.
- `Q2` and `Q3` show the biggest improvement from `2%` to `5%`, which is exactly what you would expect if the deeper-but-not-final sections need a little more sparse budget.
- `Q4` and `Q5` are almost insensitive to the budget once it reaches `5%`, which suggests the late representation is highly compressible under this recipe.

### Implication

The sparse-sufficiency story is not tied to the original R56/C100 anchor. It survives both a smaller backbone and a simpler dataset. That means the main hypothesis is not "we found one sweet spot in one model," at least on the CNN side.

## Experiment 2: Seed Stability (`seeds`)

### What was done

This is the direct "is the winner path a seed accident?" test.

- backbone: `ResNet-56`
- dataset: `CIFAR-100`
- retained fraction: `5%`
- seeds: `0`, `1`, `2`
- sections: `Q1..Q5`

This is important because the core CNN claim would be much weaker if one seed happened to give the nice result and others drifted badly.

### Results

#### `spliced_top1` by section and seed

Baseline for comparison:

- frozen ResNet-56 / CIFAR-100 backbone top-1 is `0.7183`
- for `pred_agree`, the meaningful maximum is `1.0000`

| Section | Seed 0 | Seed 1 | Seed 2 | Mean | Std |
|---|---:|---:|---:|---:|---:|
| Q1 | 0.7173 | 0.7180 | 0.7181 | 0.7178 | 0.00036 |
| Q2 | 0.7174 | 0.7148 | 0.7143 | 0.7155 | 0.00136 |
| Q3 | 0.7108 | 0.7142 | 0.7136 | 0.7129 | 0.00148 |
| Q4 | 0.7184 | 0.7182 | 0.7151 | 0.7172 | 0.00151 |
| Q5 | 0.7174 | 0.7167 | 0.7165 | 0.7169 | 0.00039 |

#### Supplementary metrics

| Section | Mean Pred Agree | Mean KL | Mean Rel L2 | Mean Cos |
|---|---:|---:|---:|---:|
| Q1 | 0.9812 | 0.0040 | 0.0277 | 0.9997 |
| Q2 | 0.9620 | 0.0144 | 0.1066 | 0.9945 |
| Q3 | 0.9323 | 0.0441 | 0.1830 | 0.9831 |
| Q4 | 0.9656 | 0.0116 | 0.1123 | 0.9938 |
| Q5 | 0.9765 | 0.0051 | 0.1106 | 0.9946 |

### What it means

This is a very stable result family.

The most important points:

- The standard deviations are tiny.
- The means are all clustered tightly.
- `Q3` is again the weakest section, but only modestly so.
- The winner story does not change with seed.

### Implication

The core CNN sparse reconstruction result is not a seed fluke. This is an important support beam for the project, because many later claims depend on the idea that the winner path is real enough to transport, compare, and extend.

## Experiment 3: Transition Taxonomy (`taxonomy_q4q5`)

### What was done

This experiment asks a more specific mechanistic question.

It is not about whether sparse codes can reconstruct a hidden state. It is about whether sparse is the right representation format for carrying a layer-to-layer relationship.

It compares three carrier types for the `Q4 -> Q5` relationship:

- sparse
- dense
- cross

### Results

| Condition | Spliced Top-1 | Pred Agree | KL |
|---|---:|---:|---:|
| sparse | 0.7063 | 0.8415 | 0.2404 |
| dense | 0.6902 | 0.7897 | 0.3815 |
| cross | 0.6099 | 0.6739 | 0.9754 |

### What it means

Sparse wins all three ways:

- best top-1
- best agreement
- best KL

Dense is second-best but clearly worse. Cross is much worse.

The main interpretive point is that the sparse representation is not merely a decent reconstruction space. It is also the most behavior-preserving transition format among the tested alternatives.

### Implication

This strengthens the project claim from:

"Sparse can reconstruct hidden states"

to:

"Sparse is the strongest tested carrier for the layer-to-layer relationship itself."

That is a much more interesting claim mechanistically.

## Experiment 4: ResNet-110 (`r110`)

### What was done

This is the deeper-backbone generalization test.

- backbone: `ResNet-110`
- dataset: `CIFAR-100`
- retained fraction: `5%`
- sections: `Q1..Q5`

Unlike the seed sweep, this is not replicated over seeds here, so there is no standard deviation to report from this run family.

### Results

| Section | Spliced Top-1 | Pred Agree | KL | Rel L2 | Cos |
|---|---:|---:|---:|---:|---:|
| Q1 | 0.7316 | 0.9804 | 0.0046 | 0.0312 | 0.9996 |
| Q2 | 0.7311 | 0.9701 | 0.0096 | 0.1065 | 0.9945 |
| Q3 | 0.7226 | 0.9356 | 0.0439 | 0.1691 | 0.9857 |
| Q4 | 0.7290 | 0.9717 | 0.0087 | 0.1189 | 0.9929 |
| Q5 | 0.7318 | 0.9804 | 0.0043 | 0.1082 | 0.9949 |

### What it means

This is another strong positive robustness result.

The most important points:

- The deeper backbone still supports strong sparse reconstruction.
- `Q1` and `Q5` are especially strong.
- `Q3` is again the hardest section.
- The persistent `Q3` weakness is not random noise anymore. It repeats across backbones and setups.

### Implication

The harder region of the network appears structurally meaningful. The fact that `Q3` remains the hardest point under a deeper backbone suggests the project has identified a real bottleneck region rather than a quirk of one exact architecture.

## Experiment 5: Single ViT Transfer Artifact (`vit_sae`)

### What was done

This is the first transformer-side transfer artifact in the repo.

- model: `vit_small_patch16_224`
- block: `6`
- retained fraction: about `5%`

This is a self-consistency experiment rather than a task-accuracy experiment in the same mold as the CNN runs.

### Results

- `pred_agree = 0.9440`
- `KL = 0.0427`
- `frac_retained = 0.0500`
- `rel_l2 = 0.4172`
- `cos = 0.9085`

### What it means

This is a strong proof-of-concept.

At only `5%` retained, the sparse reconstruction preserves the ViT's original top prediction on `94.4%` of validation examples. The KL is also low, which means the full output distribution usually stays close, not just the argmax.

### Implication

The transformer transfer path is real. This experiment alone is not enough for a broad claim, but it is enough to justify doing the sweep experiments. Without this result, the ViT side would still be just an idea.

## Experiment 6: ViT-Small Sweep (`vit_sweep`)

### What was done

This is the first broad transformer-side sweep.

- model: `vit_small_patch16_224`
- blocks: `2`, `4`, `6`, `8`, `10`
- retained fractions: `1%`, `2%`, `5%`, `8%`, `12%`
- seeds: `0`, `1`, `2`
- train subset: `2000`
- validation subset: `500`
- total cells: `75`

Unlike the single ViT artifact, this is testing whether the result is broad and stable rather than lucky.

### Headline results

- overall mean `pred_agree` across all `75` runs: `0.8914`
- overall mean `KL`: `0.1082`
- best single run: block `10`, `12%`, seed `1`, `pred_agree = 0.9340`
- meaningful max for `pred_agree`: `1.0000`

### Full mean table

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

### Structure in the results

The first thing to notice is that this sweep is not random-looking.

There is structure:

- performance generally improves as retained fraction increases
- later blocks tend to be better than middle blocks
- block `4` is surprisingly strong
- block `8` is weaker than `4` and `10`

Mean by retained fraction:

- `1%`: `0.8569`
- `2%`: `0.8827`
- `5%`: `0.8971`
- `8%`: `0.9075`
- `12%`: `0.9129`

Mean by block:

- `2`: `0.8916`
- `4`: `0.9024`
- `6`: `0.8792`
- `8`: `0.8752`
- `10`: `0.9087`

### What it means

This is a genuine stability result.

The most important points:

- The original ViT transfer artifact was not a fluke.
- There is a broad stable regime, not one single lucky point.
- Larger sparse budgets help on average.
- The best stable region sits around blocks `4` and `10`, especially at `8%` and `12%`.

But there is also an important nuance:

- The sweep is not uniformly wonderful.
- Some blocks are clearly weaker.
- So the correct claim is not "all ViT blocks are equally compressible."

### Implication

The project can now support the stronger transformer-side claim:

"Sparse token-grid reconstruction works across a nontrivial region of ViT depth and budget settings, with best stable settings around `0.93` prediction agreement."

That is a serious upgrade over a one-off artifact.

## Experiment 7: ViT-Small SAE Family Comparison (`phase4_changed`)

### What was done

This run family compares two SAE choices on the same completed `ViT-small` sweep grid:

- model: `vit_small_patch16_224`
- SAE families: `field`, `vector`
- blocks: `2`, `4`, `6`, `8`, `10`
- retained fractions: `1%`, `2%`, `5%`, `8%`, `12%`
- seeds: `0`, `1`, `2`
- train subset: `2000`
- validation subset: `500`
- total cells: `150`

Experiment 6 was `field`-only. Experiment 7 asks whether part of the ViT-small ceiling was really an SAE-family mismatch.

### Headline results

Overall mean performance by SAE family:

| SAE | Mean Pred Agree | Mean KL | Mean Rel L2 |
|---|---:|---:|---:|
| field | 0.8914 | 0.1082 | 0.4678 |
| vector | 0.8887 | 0.1144 | 0.4448 |

That might look like "nothing changed," but that would be too shallow a reading. The family comparison is more structured than that.

### Where each SAE wins

Across the `25` block/fraction settings:

- `vector` wins `16`
- `field` wins `9`

Representative mean `pred_agree` deltas (`vector - field`):

- block `2`, `1%`: `-0.0807`
- block `4`, `1%`: `-0.0780`
- block `2`, `12%`: `+0.0253`
- block `8`, `12%`: `+0.0220`
- block `10`, `12%`: `+0.0087`

So the clean pattern is:

- `field` is better in the harshest early-block / tiny-budget regime
- `vector` is often better once the budget is larger and / or the block is later

### Best stable settings by SAE family

Best stable `field` settings:

| SAE | Block | Retained | Mean Pred Agree | Seed Std | Mean KL | Mean Rel L2 |
|---|---:|---:|---:|---:|---:|---:|
| field | 10 | 12% | 0.9307 | 0.0034 | 0.0361 | 0.3558 |
| field | 4 | 8% | 0.9273 | 0.0038 | 0.0463 | 0.3814 |
| field | 10 | 8% | 0.9187 | 0.0034 | 0.0456 | 0.4177 |

Best stable `vector` settings:

| SAE | Block | Retained | Mean Pred Agree | Seed Std | Mean KL | Mean Rel L2 |
|---|---:|---:|---:|---:|---:|---:|
| vector | 10 | 12% | 0.9393 | 0.0074 | 0.0201 | 0.3361 |
| vector | 2 | 12% | 0.9380 | 0.0102 | 0.0236 | 0.2400 |
| vector | 4 | 12% | 0.9367 | 0.0066 | 0.0263 | 0.2747 |

So the strongest completed `ViT-small` family result is now:

- `vector`, block `10`, retained `12%`
- mean `pred_agree = 0.9393`
- seed std `= 0.0074`
- mean `KL = 0.0201`

### Matched filtered comparison

One of the most informative diagnostics from this family comparison is the matched filter: remove only the catastrophically high-reconstruction-loss `vector` runs, then compare `vector` and `field` on the exact same `(block, frac, seed)` cells.

Using:

- keep only `vector` runs with `rel_l2 <= 0.50`

the matched result is:

| Group | Run Count | Mean Pred Agree | Mean Rel L2 |
|---|---:|---:|---:|
| kept `vector` runs (`rel_l2 <= 0.50`) | 48 | 0.9134 | 0.3745 |
| matched `field` runs on those same cells | 48 | 0.9051 | 0.4026 |

For the dropped `vector` runs:

| Group | Run Count | Mean Pred Agree | Mean Rel L2 |
|---|---:|---:|---:|
| dropped `vector` runs (`rel_l2 > 0.50`) | 27 | 0.8447 | 0.5698 |
| matched `field` runs on those same cells | 27 | 0.8670 | 0.5837 |

This tells us something sharper than the raw family average:

- `field` is slightly better in the unfiltered overall average
- but once `vector` avoids the very high-reconstruction-loss regime, `vector` is actually slightly better than `field` on both reconstruction loss and prediction agreement

How good is `0.9134` here?

It means the reconstructed ViT keeps the **same top prediction as the original ViT on about 91.3% of validation examples**. That is strong enough to support a real sparse-sufficiency claim, but still far enough from `1.0000` to show a meaningful remaining transformer-side gap.

### What it means

This experiment changes the interpretation of the earlier ViT-small story in an important but limited way.

It says:

- the original `field` sweep was not the whole transformer story
- changing SAE family does help in several good regions
- but the gains are not large enough to make the ViT-small ceiling disappear

So after this comparison, both explanations remain alive:

- the current SAE design is still part of the bottleneck
- transformer activations may also be genuinely harder to sparsely reconstruct than the CNN activations

### Implication

The correct updated claim is not:

"the first ViT-small sweep settled the SAE question."

It is:

"SAE family choice matters on ViTs, especially at higher budgets, but it does not by itself eliminate the transformer-side gap."

## Experiment 8: ViT-Base Sweep (`vit_base_imagenette_full`)

### What was done

This is the completed larger-model ViT sweep.

- model: `vit_base_patch16_224`
- blocks: `2`, `4`, `6`, `8`, `10`
- requested retained fractions: `2%`, `5%`, `8%`, `12%`
- seeds: `0`, `1`, `2`
- train subset: `2000`
- validation subset: `500`
- total cells: `60`

This run family is the most subtle and the most important to interpret carefully.

### Headline results

- best single run: block `10`, requested `12%`, seed `1`, `pred_agree = 0.9862`
- best stable setting: block `10`, requested `12%`, mean `0.9851`, std `0.0012`
- meaningful max for `pred_agree`: `1.0000`
- raw mean across all `60` runs: `0.6606`

That raw mean is highly misleading if read alone.

### Full mean table

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

### Why the raw average is misleading

If you average everything together, the sweep looks mediocre:

- raw mean `pred_agree = 0.6606`

But that number mixes two very different regimes:

- a strong, stable regime
- a collapse-prone regime

This is not a smooth degradation story. It is a bimodal story.

### Stable regime

The strong stable regime includes:

- block `10`, `12%`: `0.9851 ± 0.0012`
- block `10`, `8%`: `0.9790 ± 0.0023`
- block `2`, `8%`: `0.9787 ± 0.0014`
- block `2`, `5%`: `0.9745 ± 0.0006`
- block `10`, `2%`: `0.9690 ± 0.0015`
- block `6`, `5%`: `0.9620 ± 0.0015`

These are excellent results.

### Collapse-prone regime

The unstable regime includes many cells where one or more seeds fail almost completely.

Examples:

- block `4`, `12%`: `0.9743 / 0.0000 / 0.0000`
- block `8`, `12%`: `0.0166 / 0.9654 / 0.9676`
- block `2`, `12%`: `0.9804 / 0.0000 / 0.0003`
- block `6`, `8%`: `0.9661 / 0.9682 / 0.0025`

This is why the standard deviations are enormous in many cells:

- block `2`, `12%`: std `0.4621`
- block `4`, all fractions: std about `0.447` to `0.459`
- block `6`, `8%`: std `0.4547`
- block `8`, `8%`: std `0.4292`

### Realized-versus-requested sparsity drift

This is one of the most important technical findings in the whole Phase 4 package.

In several collapse-prone seeds, the realized `frac_retained` is much lower than the requested budget.

Examples:

- requested `5%`, block `10`, one seed realized about `2.78%`
- requested `8%`, block `6`, one seed realized about `2.99%`
- requested `12%`, block `2`, one seed realized about `3.36%`

### Separate SAE reconstruction collapse from SAE success

This is the distinction that matters most for interpreting the sparsity theory.

We can split the `60` runs into:

- `SAE reconstruction collapse`
- `SAE non-collapse`

For this breakdown, a practical reconstruction-collapse rule is:

- `rel_l2 > 1` or `fvu > 1`

Under that rule:

| Group | Run Count | Mean Pred Agree | Mean Rel L2 | Mean FVU |
|---|---:|---:|---:|---:|
| SAE reconstruction collapse | 19 | 0.0087 | 2.1533 | 11.0939 |
| SAE non-collapse | 41 | 0.9627 | 0.4424 | 0.2527 |

This is an extremely important disambiguation.

It means the dramatic near-zero `pred_agree` cells are not, in the current sweep, clean evidence against the sparsity theory after successful reconstruction. They are mostly evidence that the SAE itself failed.

Within the `41` SAE non-collapse runs:

- `0/41` had `pred_agree < 0.5`
- all `41/41` stayed in the behavior-preserving regime

So there is no strong example here of:

- "the sparse code reconstructed the transformer activations well, but the downstream behavior still collapsed anyway"

### What that means for the theory

This does **not** prove the sparsity theory universally. But it does shift the burden of explanation.

The main failure mode exposed by this sweep is:

- seed-sensitive SAE failure

not:

- a clear systematic breakdown of sparse sufficiency after good reconstruction

This matters because it means:

- the failing seed is not merely "using the intended 12% budget badly"
- in some cases it is not even actually retaining anything close to the intended budget

So this sweep should be interpreted by requested setting, not by rounding the realized `frac_retained` field in the JSON.

### What it means

This is not a simple "bigger is better" result.

It is a two-sided result:

- `ViT-Base` can achieve much stronger sparse self-consistency than `ViT-small`
- but it is also much more brittle off-regime

The biggest positive:

- the best stable `ViT-Base` setting is far better than the best stable `ViT-small` setting
- `0.9851` versus `0.9307`

The biggest negative:

- many settings are not merely weaker; they are bimodal or collapse-prone

### Implication

The right scientific claim is:

"Larger ViTs can support extremely faithful sparse token-grid reconstructions in the right late-block regime, but stability across block and budget is not automatic and may be limited by seed-sensitive sparsity dynamics."

That is a strong result, but a more conditional one than a simple winner-only summary would suggest.

## What Is Clearly Supported After Phase 4

These claims now have strong support:

- the CNN-side sparse reconstruction story is robust to seed variation
- it survives backbone and dataset shift on the CNN side
- sparse is not only reconstructive but a particularly strong transition carrier
- the harder CNN region remains `Q3` across multiple backbones
- ViT transfer is real
- ViT-small has a broad stable regime
- ViT-small changes somewhat under SAE-family swap, with `vector` improving several later-block / higher-budget settings
- ViT-base has an extremely strong stable best-case regime

## What Is Only Partially Supported

These claims are supported only in a qualified form:

- "transformer sparse reconstruction works broadly"
  This is true for `ViT-small`, but only conditionally true for `ViT-base`

- "larger model means better sparse reconstruction"
  Best-case yes, uniform stability no

- "later blocks are always better"
  No; `block 4` is unusually strong in `ViT-small`, and several middle-block settings in `ViT-base` are unstable

## What Phase 4 Newly Exposes As An Open Problem

The most important new open problem is:

- why some larger-model seeds collapse and retain far less than their requested sparse budget

This is scientifically important because the `ViT-Base` sweep is not just noisy. It has structure:

- stable late-block success
- unstable off-regime cells
- realized-versus-requested sparsity drift

That suggests the next step is not "abandon the hypothesis." It is "diagnose the seed-sensitive failure mode."

## Bottom Line

The full interpretation of Phase 4 is:

The project's main sparse reconstruction claim became stronger, not weaker. The CNN-side story is robust. The first ViT transfer artifact is real. The ViT-small sweep establishes a broad stable transformer-side regime, and the ViT-small SAE-family comparison shows that SAE choice matters but does not remove the whole gap. The ViT-base sweep raises the best-case ceiling dramatically, but also reveals a real fragility story that prevents a naive uniform scaling claim.

So Phase 4 does not say:

"everything works everywhere."

It says:

"the hypothesis is real, often very strong, and now precise enough that we can see where it succeeds, where it saturates, and where it breaks."

## Pointers To Rawer Tables

If you want the machine-readable tables behind these interpretations:

- ViT-small by setting:
  [runs/vit_sweep/analysis/vit_sweep_by_setting.csv](/jumbo/lisp/f003x5w/ViT_Proj/CNN-SAE/runs/vit_sweep/analysis/vit_sweep_by_setting.csv)
- ViT-base by setting:
  [runs/vit_base_imagenette_full/analysis/vit_scale_by_setting.csv](/jumbo/lisp/f003x5w/ViT_Proj/CNN-SAE/runs/vit_base_imagenette_full/analysis/vit_scale_by_setting.csv)
- ViT-base summary:
  [runs/vit_base_imagenette_full/analysis/vit_scale_summary.json](/jumbo/lisp/f003x5w/ViT_Proj/CNN-SAE/runs/vit_base_imagenette_full/analysis/vit_scale_summary.json)

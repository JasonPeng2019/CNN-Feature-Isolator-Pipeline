# Abundant RPM Technical Presentation Brief: CNN-SAE

## Purpose

This document turns the current `CNN-SAE` repository into a defensible technical-presentation narrative for an Abundant Research Product Manager interview. The story emphasizes problem framing, experiment design, evidence quality, learning from failure, and prioritization. Those themes match Abundant's description of RPMs as people who co-design model capabilities, review literature, run experiments and benchmarks, and build research/data pipelines.

The presentation should use one central question:

> Can a learned sparse, spatially indexed representation replace intermediate vision-model activations while preserving the model's computation, both at one layer and across a sequence of layers?

The strongest version of the story is not that the project has already automated semantic interpretability. The repository establishes sparse reconstruction and behavioral faithfulness much more strongly than it establishes human-interpretable feature meaning.

## Evidence labels used in this brief

- **Established in the current artifact set:** backed by code and saved run artifacts.
- **Qualified:** supported only for the studied models, datasets, metric, or analysis corpus.
- **Open:** planned, partially run, or missing the evidence needed for the claim.
- **Personalization required:** only you can supply this information.

## Personalization required before making the deck

Fill these in before presenting. Do not let the audience infer ownership from the repository.

| Prompt | Your answer |
|---|---|
| What parts did you personally design? | `[FILL IN]` |
| What code and infrastructure did you personally implement? | `[FILL IN]` |
| Which experiments did you personally run or supervise? | `[FILL IN]` |
| Which analysis or evaluation flaw did you personally discover? | `[FILL IN]` |
| Did you work alone, with collaborators, or with coding agents? | `[FILL IN]` |
| How long did the work take and what compute did you use? | `[FILL IN]` |
| What decision was yours versus inherited from prior work? | `[FILL IN]` |

A safe ownership sentence is:

> I owned **[scope]** end to end: I translated the research question into **[experiment plan]**, implemented **[components]**, ran **[experiment families]**, and changed the plan when **[specific result or audit finding]** showed that the original interpretation was too broad.

## Recommended correction to the opening framing

### Suggested project title

**Sparse Auto-Encoding for Vision Activation Fields: From Local Reconstruction to System-Level Faithfulness**

Shorter alternative:

**Can Sparse Codes Preserve a Vision Model's Computation?**

### Suggested opening copy

Manual vision interpretability often begins by selecting a neuron or channel and then using tools such as activation maximization, top-activating dataset examples, and attribution to determine what activates it and what it influences. This can produce detailed circuit hypotheses, but it requires substantial expert-guided analysis.

Sparse autoencoders offer a possible first automation layer. They learn an overcomplete dictionary and represent each hidden activation field with a small set of active coefficients. In this project, each active coefficient has two identities: a learned feature index and a spatial location.

The research question was:

> Can a sparse autoencoder learn a compact set of feature-location coefficients at each layer that reconstructs the hidden state closely enough to preserve the frozen model's outputs, and can learned transitions carry those sparse representations across depth?

The project therefore tested three increasingly demanding properties:

1. **Local reconstruction:** Can one layer be sparsified and reconstructed?
2. **Behavioral faithfulness:** Does the frozen remainder of the model behave similarly after the reconstructed activation is spliced in?
3. **System composition:** Can sparse representations support learned layer-to-layer transitions without error compounding across a full chain?

The eventual interpretability goal is to connect learned feature-location coefficients to recognizable concepts and causal effects. The current evidence does not yet justify calling the learned coefficients automatically isolated semantic features.

### Correct use of the signal-processing motivation

The Elad-Bruckstein uncertainty principle concerns the support sizes of a signal represented in two orthonormal bases. If the bases have mutual coherence

\[
\mu = \max_{j,k}|\langle \tau_j,\omega_k\rangle|,
\]

then the two coefficient supports obey

\[
\|\theta_\tau h\|_0\,\|\theta_\omega h\|_0 \geq \frac{1}{\mu^2}.
\]

Use this as motivation for searching for a basis or dictionary in which a signal is compact. It is not a theorem that CNN or ViT hidden activations must be sparse under the nonlinear, overcomplete SAE used here.

Use the original 2002 paper rather than arXiv:2308.00312, which is a later functional-analysis preprint with a placeholder title: Michael Elad and Alfred Bruckstein, [“A Generalized Uncertainty Principle and Sparse Representation in Pairs of Bases”](https://doi.org/10.1109/TIT.2002.801410), *IEEE Transactions on Information Theory* 48(9), 2002.

### Correct use of LoRA

LoRA shows that some model adaptations can be represented through low-rank weight updates. It does not show that images or hidden activations are sparse. Low rank and sparsity are different structural assumptions. The cleanest deck drops LoRA from the motivation. If you retain it, call it an analogy: both approaches search for lower-complexity structure, but they operate on different objects and use different constraints. See Hu et al., [“LoRA: Low-Rank Adaptation of Large Language Models”](https://arxiv.org/abs/2106.09685), 2021.

### Correct use of the word “feature”

In this project, a feature is initially a learned SAE dictionary element. An activation at `(feature index, spatial location)` identifies where that dictionary element participates in reconstructing a hidden activation field. Calling that element a semantic feature requires additional evidence such as coherent top-activating examples, spatial overlays, automated or human labels, and targeted causal interventions.

Likewise, “effective pixels” is too literal. The code operates on hidden activation grids or ViT patch-token grids. Use **feature-location coefficients** or **spatial support**, then map those positions back to input receptive fields when you have performed that analysis.

## Thirty-second project overview

> I built an experimental pipeline to test whether intermediate vision-model activations could be replaced by sparse, spatially indexed codes without changing the model's behavior. I started with layerwise sparse autoencoders on a frozen ResNet, used reconstruction and downstream splice tests to select an architecture, then trained transitions between sparse layers and composed them into a full chain. The layerwise result was strong, but the first raw chain collapsed to chance. That failure led to chain-aware training, full backpropagation through the rollout, fairer sparsity accounting, and transfer tests on other CNNs and ViTs. The main lesson was that local reconstruction can be easy while system-level composition and evaluation validity are the real bottlenecks.

## One-minute technical setup

Let `H_t` be the activation field at section `t` of a frozen vision model. The SAE learns an encoder `E_t` and decoder `D_t`:

\[
Z_t = E_t(H_t), \qquad \widetilde Z_t = \operatorname{TopM}(Z_t), \qquad \widehat H_t = D_t(\widetilde Z_t).
\]

`TopM` keeps a limited set of coefficients across feature and spatial dimensions. The reconstruction `Hhat_t` is inserted back into the frozen model at section `t`. The project measures both activation reconstruction and how closely the resulting model output matches the original frozen model.

For the hierarchy experiment, a learned transition predicts the next sparse code:

\[
\widehat Z_{t+1} = T_t(\widetilde Z_t).
\]

The difficult test composes four transitions from `Q1` through `Q5`, then asks whether the classifier still behaves correctly.

---

# Slide-by-slide content

## Slide 1 — Project Overview

### Title

**Can Sparse Codes Preserve a Vision Model's Internal Computation?**

### Direct answers to the slide questions

- **What was the project?** I built an experimental pipeline that replaced intermediate CNN and ViT activation fields with learned sparse feature-location codes, spliced the reconstructions back into frozen models, and measured whether their computation was preserved.
- **What was the high-level research goal?** Determine whether a small learned sparse message can preserve both an individual hidden state and the computation that follows it, eventually creating a scalable substrate for feature-level interpretability.
- **Why was it worth investigating?** Existing vision-interpretability workflows often require experts to choose units, inspect activation-maximizing images and dataset examples, and trace attribution manually. An SAE could discover candidate feature directions and locations automatically, but reconstruction, behavioral faithfulness, and system-level composition first had to be established.
- **What did I personally own?** `[FILL IN YOUR ACTUAL SCOPE: research framing, implementation, experiment orchestration, analysis, evaluation audit, or some subset.]`
- **What was the headline outcome?** Sparse layer replacements preserved substantial downstream behavior. The completed clean ResNet-20 `field_recovery` slice averaged `0.9560` prediction agreement and `0.9106` spliced top-1 at an exact 8% original-node budget, versus about `0.923` original top-1. In the historical chain experiment, full-BPTT re-grounding reached `0.7010` versus a `0.7122` frozen-block hybrid and `0.7183` original model.

### Put on the slide

- **Project:** learned sparse, spatially indexed replacements for intermediate CNN and ViT activations.
- **Goal:** preserve frozen-model computation with a small set of feature-location coefficients.
- **My role:** `[INSERT ONE-SENTENCE OWNERSHIP STATEMENT]`.
- **Headline:** layerwise sparsification worked; composing learned sparse transitions exposed the real bottleneck and drove a more rigorous research program.

### Say aloud

“I was not asking whether an autoencoder could minimize reconstruction loss in isolation. I wanted to know whether a sparse internal representation could substitute for the original hidden state and still support the rest of the model's computation. I then raised the bar from one layer to an entire learned chain.”

### Suggested visual

One pipeline diagram:

```text
image → frozen vision model → H_t → encoder → sparse feature × location code
                                      ↓                    ↓
                               original tail       decoder → reconstructed H_t → frozen tail
                                      └──────── compare outputs ────────┘
```

### Evidence

- Project purpose and source-of-truth guidance: `README.md:1-32`.
- SAE input, sparse field, and no-bypass design: `src/sae.py:1-15`.
- Downstream splice evaluation: `src/eval.py:27-37`.

### Transition

“The first decision was defining what preserving the model should mean.”

## Slide 2 — The Core Problem

### Title

**From sparse reconstruction to faithful computation**

### Direct answers to the slide questions

- **What was the central research question?** Can a learned sparse, spatially indexed code replace intermediate vision-model activations while preserving the frozen model's downstream computation?
- **What was the starting hypothesis?** CNN activation fields contain enough redundancy that a learned overcomplete dictionary can reconstruct them from a limited active support, and those sparse codes retain enough information to support later layers.
- **Why might sparse representations be useful?** They reduce each dense hidden field to a small set of feature indices, locations, and values. That creates candidate units for automated analysis and intervention while preserving spatial structure.
- **What did “preserving the model” mean?** It meant low activation reconstruction error, high agreement between original and spliced predictions, low KL divergence between their output distributions, nearly unchanged task accuracy, and successful composition across multiple learned transitions.
- **Was success reconstruction, agreement, downstream computation, or interpretability?** The project directly tested reconstruction, prediction agreement, downstream splice behavior, and chain composition. It only began the interpretability evaluation; semantic feature isolation remains open.
- **What would have shown the idea was failing?** Failure to beat random or PCA controls; good reconstruction with collapsed downstream behavior; one-step transitions that could not compose; results that disappeared under fair rate accounting or a clean holdout; or features that lacked coherent semantic and causal behavior.

### Put on the slide

**Central question**

Can learned sparse feature-location codes preserve both a layer's activation and the downstream behavior of the frozen model?

**Starting hypothesis**

Vision activation fields contain enough redundancy that an overcomplete learned dictionary can reconstruct them from a small active support.

**Success meant all of the following**

- small reconstruction error;
- high prediction agreement or nearly unchanged spliced task accuracy;
- low KL divergence between original and spliced output distributions;
- survival across multiple layers, models, seeds, and datasets.

**Failure conditions**

- trained SAE no better than random or PCA controls;
- reconstruction looks good while downstream behavior collapses;
- one-step transitions work but a composed chain fails;
- results disappear under a fair budget or clean holdout;
- learned coefficients lack semantic or causal coherence.

### Say aloud

“A sparse code can win on reconstruction and still destroy the decision boundary. Conversely, classification accuracy alone can hide large changes in the output distribution. I therefore treated reconstruction, output agreement, KL divergence, and composition as separate tests.”

### Suggested visual

A four-rung evidence ladder:

```text
reconstruction → output faithfulness → compositional stability → semantic/causal meaning
    tested             tested                 tested                 still open
```

### Evidence

- Metric definitions: `src/eval.py:6-37`.
- Original success criterion: `analysis/intent.json`, entry `phase1_anchor`.
- Interpretability work still needed: `docs/CURRENT_PROGRESS_AND_TODO.md:177-187` and `399-405`.

### Transition

“That definition turned one broad idea into a sequence of experiments with explicit gates.”

## Slide 3 — How I Mapped the Research

### Title

**A gated research program, not a single model run**

### Direct answers to the slide questions

- **Can one layer be sparsified? Yes.** In the historical ResNet-56 experiment, trained SAEs preserved near-original downstream top-1 at practical latent-retention settings and beat the random-dictionary and reported PCA controls. The later clean ResNet-20 `field_recovery` slice provides stronger rate-controlled evidence: across five sections and three seeds, it averaged `0.9560` prediction agreement and `0.9106` spliced top-1 at an exact 8% original-node budget, versus about `0.923` original top-1.
- **Which architecture and budget preserve behavior? Historically, FieldSAE with global masking was the most reliable CNN choice.** In the historical Pareto sweep, most sections reached their useful regime around 3–8% latent retention; the project chose 5% latent retention for the transition experiments. This was 5% of an `8C` overcomplete latent, equal to 40% as many selected coefficients as original activation values before support bits. Under the newer exact-rate protocol, `field_recovery` is strong on the completed ResNet-20 slice, but the full four-family, cross-model winner is still unknown because the clean matrix is incomplete.
- **Can sparse codes predict the next layer? Yes, for each tested adjacent ResNet-56 transition, with one clear weak edge.** Learned spliced top-1 was `0.715` for `Q1→Q2`, `0.691` for `Q2→Q3`, `0.650` for `Q3→Q4`, and `0.709` for `Q4→Q5`. The frozen-block hybrid comparison was about `0.711–0.715`, showing that `Q3→Q4` had the largest learned-transition gap.
- **Do the transitions survive a full `Q1→Q5` chain? The initial raw chain did not; chain-aware variants recovered most of the behavior.** Raw independent transitions collapsed to `0.0116` top-1. Re-masking reached `0.2126`, re-grounding reached `0.5930`, scheduled-sampling re-grounding reached `0.6624`, and full-BPTT re-grounding reached `0.7010`, compared with `0.7122` for the frozen-block hybrid and `0.7183` for the original model. A separate rollout-aligned raw model reached `0.6821`. The newer clean replicated chain remains unrun.
- **Why did the weak configurations fail? Several distinct causes were supported.** Independent transitions suffered rollout distribution shift and compounding error; rollout-aligned training rescued the raw chain. A larger `Q3→Q4` predictor became worse (`0.5344`), so capacity alone was not the answer. Fair RF accounting rescued the RF-local branch, showing the old budget was unfair. Overlapping patches improved pointwise coding, while coarse disjoint patches destroyed too much spatial structure. ViT-Base failures were associated with SAE reconstruction collapse and realized sparsity drifting below the requested budget. These experiments narrow the explanation but do not prove one universal failure mechanism.
- **Does the result survive new seeds, backbones, datasets, and ViTs? Layerwise reconstruction does; the transition-chain claim has not been transferred.** The reconstruction result survived three ResNet-56 seeds, ResNet-20/CIFAR-10, ResNet-110/CIFAR-100, and stable regions of ViT-Small and ViT-Base on Imagenette. The first ViT-Small result achieved `0.9440` prediction agreement at 5% latent retention; the best stable historical ViT-Base setting reached `0.9851 ± 0.0012`, although 19 of 60 ViT-Base runs collapsed. Only ResNet-56 has saved transition and chain experiments.
- **Are the evaluation and rate accounting trustworthy? The historical evidence is useful but qualified; the corrected evaluation is only partially complete.** Historical CIFAR trainers inspected the test corpus every epoch, so those numbers support corpus-conditioned mechanistic faithfulness rather than untouched external generalization. Historical percentages measured retention within an overcomplete latent and did not establish net bit compression. The redo fixes this with train/validation/one-shot holdout separation, exact original-node Top-M budgets, bit accounting, controls, and immutable manifests. All 300 clean ResNet-20 job segments are complete, but ResNet-110 is partial and the clean ViT and chain phases are unfinished.

The accurate chronological summary is:

```text
CORE TESTS COMPLETED
Layer reconstruction → architecture/budget selection → adjacent transitions → full chain

FOLLOW-UPS COMPLETED
Diagnose Q3→Q4 and chain failure → test training/budget/context explanations
→ test reconstruction transfer across models and datasets

CURRENT WORK
Clean exact-budget evaluation → complete broader matrix → clean replicated chains
```

### Put on the slide

| Research question | Actual answer |
|---|---|
| Can one layer be sparsified? | **Yes.** Layerwise reconstructions preserved substantial downstream behavior; the clean R20 slice reached `95.60%` agreement at an exact 8% original-node budget. |
| Which architecture and budget preserve behavior? | **Historical answer:** Field + global, usually around 3–8% latent retention; 5% was used for the chain. **Clean cross-model answer:** still open. |
| Can sparse codes predict the next layer? | **Yes, unevenly.** Three edges reached `0.691–0.715` top-1; `Q3→Q4` was weakest at `0.650`. |
| Do transitions survive a full chain? | **Only with system-aware training.** Raw: `0.012`; rollout-aligned raw: `0.682`; full-BPTT re-grounded: `0.701`; hybrid: `0.712`. |
| Why did weak configurations fail? | **Training mismatch, compounding error, unfair budgets, insufficient/incorrect spatial context, and SAE instability all mattered.** |
| Does the result transfer? | **Reconstruction does.** It survived new seeds, CNNs, datasets, and ViTs. Learned transition chains were only tested on ResNet-56. |
| Is the evaluation trustworthy? | **Historical results are qualified; clean results are partial.** Clean R20 is complete, while the broader exact-rate matrix and clean chains remain unfinished. |

### Explain planned versus reactive work

| Planned progression | Added after observing results |
|---|---|
| Layerwise SAE anchor and controls | Fine-grained Pareto sweep after a winner emerged |
| Field versus Vector architecture branch | Larger `Q3→Q4` predictor after that edge became the bottleneck |
| Adjacent transitions and full chain | Scheduled sampling and re-grounding after raw chaining collapsed |
| Transfer and robustness tests | Full BPTT after a residual chain gap remained |
| Feature and intervention audits | Fair RF budgeting after discovering an apples-to-oranges budget |
|  | Rollout-aligned raw training after reconsidering the “re-grounding is necessary” claim |
|  | Overlapping, depth-aware, and disjoint patches after the pointwise Vector baseline looked underspecified |
|  | Clean exact-budget redo after the evaluation audit |

### What this table means in direct English

I started with a planned sequence: first determine whether an individual layer could be sparsified without changing the CNN's behavior; then compare representation designs; then test whether one sparse layer could predict the next; then connect those transitions into a full chain; and finally test whether the result was robust and whether the learned features were meaningful.

The experiments in the right column were **not part of that original sequence**. I added each one after a result exposed a specific gap in my explanation or evaluation:

- After one SAE configuration looked strongest, I added a finer Pareto sweep to find where fidelity began to fall as sparsity increased.
- After `Q3→Q4` became the weakest transition, I increased that predictor's capacity to test whether simple under-capacity caused the bottleneck. It did not fully explain the failure.
- After direct raw chaining collapsed, I added scheduled sampling and periodic re-grounding to test whether training-versus-rollout mismatch and accumulated error were responsible. Both helped substantially.
- After a gap remained between the learned chain and the frozen-block hybrid, I added full backpropagation through the chain so the transitions could optimize for end-to-end behavior rather than only local accuracy.
- After noticing that the receptive-field comparison assigned different effective budgets to different methods, I introduced fair RF budgeting. This meant the earlier comparison could no longer support a clean architectural claim.
- After re-grounding worked well, I questioned whether re-grounding was intrinsically necessary. Rollout-aligned raw training showed that a much stronger raw chain was possible, so the correct conclusion became that rollout mismatch was a major problem, not that ground-truth re-grounding was always required.
- After the pointwise Vector model appeared too weak, I added overlapping, depth-aware, and disjoint patch variants to test whether the missing ingredient was spatial context rather than the general Vector idea.
- After auditing test-set reuse and rate accounting, I began a clean exact-budget redo so future comparisons would use validation data for selection, reserve the test set for final evaluation, and measure sparsity against the original activation size.

The table therefore separates **the research roadmap I expected to run** from **the experiments I chose because the evidence changed what I needed to know next**. Items that share a row are not necessarily direct cause-and-effect pairs; the two columns are parallel lists of planned and reactive work.

### Say aloud

“Each phase earned the next one. I used the cheapest experiment that could eliminate a bad direction, and I added work when a result changed the uncertainty rather than simply filling a grid.”

### Evidence

- Phase hypotheses and gates: `analysis/intent.json`.
- Outcome classifications: `analysis/outcomes.json`.
- Current clean-redo contract: `docs/plans/SAE_REDO_LEVEL2_CONTRACT.md`.

### Transition

“The roadmap only works if the metrics tell me different things.”

## Slide 4 — How I Defined Good Evidence

### Title

**A metric stack for reconstruction, behavior, and rate**

### Direct answers to the slide questions

- **What were the primary metrics?** Relative L2, fraction of variance unexplained, cosine similarity, prediction agreement, original-to-spliced KL divergence, spliced top-1 accuracy, retained support, and later exact sparse-message bit counts.
- **Why choose them?** They separately measure activation reconstruction, preservation of the original decision, preservation of the full output distribution, task behavior, and the cost of the transmitted sparse representation.
- **What baselines or controls were used?** An untrained random dictionary, PCA, competing SAE architectures and masks, multiple seeds, and a frozen real-block hybrid for transition experiments.
- **What did each metric capture that the others did not?** Reconstruction metrics measured hidden-state error; agreement measured matching argmax behavior; KL detected distribution shifts hidden by agreement; spliced top-1 measured labeled task behavior; exact `M` and bits measured the actual rate.
- **How was the compression-versus-faithfulness tradeoff measured?** Historical sweeps plotted fidelity against latent retained fraction. The corrected protocol fixes an exact active count relative to the original activation and counts support-index, value, and metadata bits.
- **What justified moving forward?** The initial gate was to beat random and PCA controls and stay within roughly one percentage point of original CNN top-1 at a practical sparse setting.
- **Which metric was misleading?** Historical “fraction retained” sounded like total compression but measured a fraction of an overcomplete latent. Spliced top-1 alone could also hide prediction changes, and the old RF comparison used an unfair per-window budget.

### Put on the slide

| Metric | What it answers | Important limitation |
|---|---|---|
| Relative L2 / FVU | How much activation error remains? | Low error does not guarantee unchanged outputs. |
| Cosine similarity | Is activation direction preserved? | Can hide scale and localized errors. |
| Prediction agreement | Does the spliced model choose the same class as the original? | Ignores probability-distribution changes. |
| KL(original‖spliced) | How much did the full output distribution move? | Hard to interpret alone. |
| Spliced top-1 | Does task accuracy survive the intervention? | Requires labels and can hide changed predictions. |
| Exact active count `M` and payload bits | What sparse message was actually transmitted? | Added rigorously in the redo; historical “fraction retained” is latent-relative. |

**Controls**

- untrained random dictionary;
- PCA at a matched or larger coefficient budget;
- frozen real-block hybrid as a causal upper bound for transitions;
- architecture comparisons under the same nominal sparse budget;
- multiple seeds and transfer settings.

**Initial gate**

At practical sparsity, remain within roughly one percentage point of the original CNN's top-1 and beat random/PCA controls.

### Say aloud

“The key product decision was to use a stack of metrics. Reconstruction tells me whether information survives; agreement and KL tell me whether the model uses the reconstruction similarly; exact message accounting tells me whether I am buying that result with a genuinely small representation.”

### Important audit note

Historical experiments reported retained fraction over the SAE latent field. Because the Field SAE used `K = 8C`, retaining 5% of `K×H×W` corresponds to roughly 40% as many active coefficients as original activation values before charging support indices. Present those results as **sparsification and faithfulness**, not as a complete bit-compression claim. The clean redo adds an original-node-relative `M` and explicit support/value bit accounting.

### Evidence

- Metric implementation: `src/eval.py:6-37`.
- Historical latent budget: `src/train_sae.py:72-75`, `88-105`.
- Exact message and bit-accounting contract: `docs/plans/SAE_REDO_LEVEL2_CONTRACT.md:22-45`.

### Transition

“With those gates, I started with the smallest experiment that could falsify the idea.”

## Slide 5 — The First Experiment

### Title

**Individual layers were sparse-faithful**

### Direct answers to the slide questions

- **What was the simplest experiment?** Train one SAE for one frozen ResNet section, keep a fixed fraction of its latent coefficients, reconstruct that section, splice it into the frozen model, and compare the resulting output with the original.
- **What varied?** Section `Q1–Q5`, retained fraction, SAE family, and global versus receptive-field-local masking.
- **What stayed fixed?** The ResNet-56/CIFAR-100 backbone, frozen downstream model, cached activation corpus, training curriculum within each sweep, and evaluation metrics.
- **What did the result show?** Trained SAEs reconstructed individual sections well enough to preserve downstream behavior, substantially outperforming random dictionaries and the reported PCA controls.
- **Which configurations worked?** The FieldSAE with global masking was the most robust historical CNN configuration. In its Pareto sweep, most sections reached a useful regime around 3–8% latent retention.
- **Where did performance degrade?** Extreme sparsity hurt early VectorSAE sections; `Q3` continued gaining from extra budget after most sections saturated; pointwise Vector and unfair RF-local variants were weak.
- **What was surprising?** At 2% latent retention, FieldSAE improved over VectorSAE by `7.6 pp` at `Q1`, `8.3 pp` at `Q2`, and `4.3 pp` at `Q3`, showing that spatial context mattered most in early high-resolution layers.
- **What decision followed?** Use FieldSAE with global masking at the 5% historical latent setting for the adjacent-transition and hierarchy experiments.

### Put on the slide

**Setup**

- frozen ResNet-56 on CIFAR-100;
- five activation sections, `Q1` through `Q5`;
- one SAE per section;
- sparse reconstruction spliced back into the frozen model;
- original top-1: `0.7183`.

### Training curriculum — what the graphic means

`f` means the final **nominal** retained fraction for that SAE run. For example, `f = 5%` means that the final stage targets the largest 5% of the **historical expanded latent field**. It does not mean 5% of the original activation map or 5% net bit compression. The historical threshold-based mask could exceed that nominal count under ties; the later clean redo uses an exact count.

```text
first 25% of 40 epochs      → warmup  → no Top-K mask
middle 50% of 40 epochs     → anneal  → retention gradually falls from 50% to f
final 25% of 40 epochs      → fixed   → target the final fraction f
```

- **Warmup** → the encoder and decoder first learn to reconstruct without throwing information away.
- **Anneal** → sparsity becomes gradually harder: the model starts by retaining half of its candidate values and ends at the target budget.
- **Fixed** → the model spends the final quarter of training learning to reconstruct under the actual target constraint.
- **Why not impose full sparsity from the first update?** Early in training, a few random candidate features can win Top-K repeatedly. The unselected features receive little or no useful learning signal and may never become useful. The gradual schedule gives many candidates time to learn before competition becomes strict.

**Training settings in the graphic**

| Label | Meaning |
|---|---|
| `Adam`, `2e-3` | The weight-update method and its learning rate (`0.002`). |
| `40 ep` | 40 full passes through the training activation corpus. |
| `bs 256` | 256 activation maps per weight update. |
| `K = 8C` | The SAE created eight times as many candidate feature channels as the original activation had channels; the historical Top-K mask later targeted only a small subset. |
| `α = 0.1` | The reconstruction loss combined normalized squared error with `0.1 × (1 − cosine similarity)`. Squared error rewarded matching values; cosine similarity rewarded matching the overall activation pattern even when its size differed. |

**Slide-safe interpretation**

```text
dense reconstruction first
→ gradually introduce the sparse bottleneck
→ train at the real sparse budget
→ reduce early feature collapse and make the final sparse reconstruction easier to optimize
```

**Anchor result**

- VectorSAE at 5–10% latent retention stayed within one percentage point of original top-1 at every section.
- Random dictionaries fell near chance (`0.009–0.107` on the selected controls).
- PCA trailed the learned SAE at the reported matched-or-larger coefficient budgets.

**Architecture decision**

At 2% latent retention, the convolutional FieldSAE improved early-section spliced top-1 over VectorSAE by `+7.6 pp` at `Q1`, `+8.3 pp` at `Q2`, and `+4.3 pp` at `Q3`. The FieldSAE became the primary hierarchy model.

**Evidence → interpretation → decision**

```text
Sparse reconstructions preserve downstream behavior
→ the hypothesis survives at one layer
→ select Field + global masking and test transitions
```

### Say aloud

“The control result mattered as much as the headline. A random dictionary could not reproduce the effect, and PCA was weaker at the reported budgets. The architecture comparison then showed that spatial context was most valuable in early high-resolution sections.”

### Suggested visual

- Main: `analysis/figures/figure_phase1_pareto.pdf`.
- Backup: `analysis/figures/figure_phase2_grid.pdf`.

### Evidence

- Anchor table and controls: `docs/PHASE1_RESULTS.md:1-45`.
- Field/Vector comparison: `docs/PHASE2_RESULTS.md:1-43`.
- Pareto result: most sections reached a useful regime around 3–8% latent retention; `Q3` continued improving through 12%: `docs/PHASE1_RESULTS.md:47-78`.

### Transition

“Layerwise success created the harder question: could the sparse representation support the transformation between layers?”

## Slide 6 — The Bottleneck

### Title

**The Q3→Q4 transition exposed the hard part**

### Direct answers to the slide questions

- **What became the main bottleneck?** The learned `Q3→Q4` transition and, at system level, the compounding error of the raw `Q1→Q5` learned chain.
- **How was it identified?** `Q3→Q4` achieved `0.6504` spliced top-1 versus `0.7113` for the frozen-block hybrid, the largest gap among the four adjacent transitions. The raw four-transition chain then fell to `0.0116`.
- **Why was it important?** It showed that strong layerwise reconstruction and mostly strong one-step transitions did not guarantee a usable composed system.
- **What were the competing explanations?** Insufficient predictor capacity; the `16×16→8×8` resolution change; off-manifold predictions; exposure bias from training on true codes but rolling out on predicted codes; and insufficient spatial context.
- **Which explanation did I initially favor?** The first repository interpretation favored exposure bias plus off-manifold drift because each independent transition consumed a state distribution at rollout that it had not seen during training.
- **What evidence was missing?** A capacity-controlled retry, training on predicted rollout states, end-to-end gradients through the chain, and architecture/budget comparisons that isolated spatial context fairly.
- **Why did resolving it matter?** A system that only reconstructs isolated layers would not support the project's stronger claim about automated computation across a depth-wise sparse hierarchy.

### Put on the slide

| Transition | Learned spliced top-1 | Frozen-block hybrid |
|---|---:|---:|
| `Q1→Q2` | `0.715` | `0.715` |
| `Q2→Q3` | `0.691` | `0.711` |
| `Q3→Q4` | **`0.650`** | **`0.711`** |
| `Q4→Q5` | `0.709` | `0.715` |

`Q3→Q4` changes spatial resolution from `16×16` to `8×8`. It had the largest learned-versus-hybrid gap and the highest reconstruction error among adjacent transitions.

**Competing hypotheses**

1. The predictor lacked capacity.
2. The transition needed architecture-aware downsampling or multiscale structure.
3. Independently trained transitions suffered exposure bias when composed.
4. Predictions drifted off the sparse-code manifold.
5. Earlier local-context and budget choices made comparisons unfair.

**A useful negative result**

A deeper, wider `Q3→Q4` predictor reduced spliced top-1 from `0.650` to `0.5344` while the hybrid remained `0.7113`. Capacity scaling alone did not solve the bottleneck.

### Say aloud

“This was the moment when I stopped treating the problem as ordinary model capacity. The real block could transmit the sparse information, so the information was present. The learned transition was failing to transform it robustly.”

### Suggested visual

`analysis/figures/figure_phase3_chain_and_edges.pdf`, using the right-hand transition panel.

### Evidence

- Transition metrics: `docs/PHASE3_RESULTS.md:7-33`.
- Larger-predictor retry: `docs/PHASE3_RESULTS.md:43-63`.

### Transition

“I designed the next experiments to separate capacity, representation geometry, training distribution, and spatial context.”

## Slide 7 — How the Results Changed My Plan

### Title

**Each failure triggered a discriminating experiment**

### Direct answers to the slide questions

- **What assumptions were revisited?** That a larger predictor would solve `Q3→Q4`, that re-grounding was strictly necessary, that RF locality was intrinsically weak, and that the Vector architecture itself was the problem.
- **What follow-up experiments were designed?** A deeper/wider `Q3→Q4` model; scheduled sampling; full-BPTT re-grounded training; rollout-aligned raw training; fair-budget RF masking; and overlapping, depth-aware, and disjoint patch encoders.
- **Which variables were isolated?** Predictor capacity, carrier type, exposure to predicted states, gradient horizon, local budget accounting, patch overlap, and patch size.
- **What made the comparisons fairer?** The RF rerun normalized the retained budget within each window. The raw-versus-re-grounded follow-up used rollout-aware training instead of evaluating a raw carrier trained for a different input distribution.
- **Which explanations were supported?** Exposure bias and rollout mismatch were strongly supported because raw-chain top-1 rose from about `0.012` to `0.6821` under rollout-aligned training. Spatial context was supported because overlapping patches improved on pointwise coding.
- **Which explanations were weakened or ruled out?** Capacity alone was weakened when the larger `Q3→Q4` predictor fell to `0.5344`. The claim that re-grounding was the only viable carrier was ruled out by the `0.6821` raw rollout.
- **What negative result was useful?** Larger disjoint patches sharply degraded performance, demonstrating that overlap and appropriately small context mattered.
- **Did an approach that looked bad improve?** Yes. Fair RF budgeting rescued the RF-local branch, and rollout-aligned training rescued the raw chain.
- **What was the next decision?** Treat carrier distribution and exact budget as first-class experimental variables, then rebuild the evaluation with clean splits, exact Top-M selection, and rate accounting.

### Put on the slide

| Observation | Follow-up | What changed in my belief |
|---|---|---|
| Larger `Q3→Q4` predictor got worse | Stop scaling the same MLP; test chain-aware and architecture-aware approaches | Capacity alone was not the explanation. |
| Raw four-step chain collapsed | Add re-masking, re-grounding, scheduled sampling, and full BPTT | Off-distribution rollout and error compounding were central. |
| Re-grounding looked indispensable | Train directly on raw rollout states with matched stabilization | Raw chaining recovered to `0.6821`; re-grounding was a strong stabilizer rather than the only possible solution. |
| RF-local branch looked weak | Normalize the sparse budget fairly per window | RF-local coding became competitive; the earlier comparison was partly a budgeting artifact. |
| Pointwise VectorSAE looked weak | Add overlapping and disjoint patch encoders | Local overlap helped; coarse disjoint tiling failed. |

**Useful negative result**

At `Q1`, the smallest disjoint patch configuration reached `0.7158`, while larger disjoint tiles fell to `0.5942`, `0.2472`, and `0.1260`. The result ruled out “more local context is always better” and showed that overlap matters.

### Say aloud

“I used failures to update the mechanism, not just the hyperparameters. The raw-chain follow-up is the best example: it overturned my earlier claim that re-grounding was necessary and replaced it with a better claim about rollout alignment.”

### Suggested visual

Use a compact hypothesis-update table or `analysis/figures/figure_changed_patch_suites.pdf` as a backup slide.

### Evidence

- Fair RF and raw-chain results: `docs/PHASE3b_changed_RESULTS.md:34-109`.
- Patch experiments: `docs/PHASE3b_changed_RESULTS.md:110-245`.
- Raw-only artifact: `runs/phase3b_raw_only_changed/result.json`.

### Transition

“These follow-ups explained individual failures, but the system test was still the full chain.”

## Slide 8 — Moving From Components to a System

### Title

**Local success did not compose automatically**

### Direct answers to the slide questions

- **Why was individual reconstruction insufficient?** It only tested one intervention at a time. A learned chain repeatedly feeds predicted states into later predictors, creating distribution shift and accumulated error.
- **What did transition models test?** Whether a sparse code at section `t` could predict the independently learned representation at section `t+1` well enough to preserve downstream behavior.
- **What happened when transitions were chained?** The initial raw chain collapsed to `0.0116` top-1. Re-masking raised it to `0.2126`; re-grounding raised it to `0.5930`.
- **Did error accumulate?** Yes. Most one-step results were strong, yet four-step composition collapsed, which is direct evidence of compounding rollout error.
- **Where did the chain break down most?** The weakest identified edge was `Q3→Q4`, although the raw-chain artifact does not provide a complete per-step causal decomposition of total error.
- **Why introduce chain-aware training and full BPTT?** Scheduled sampling exposed predictors to their own rollout states, while full BPTT let final and later-step losses update earlier transitions.
- **Did it improve the result?** Yes. Scheduled-sampling re-grounding reached `0.6624`; full-BPTT re-grounding reached `0.7010`, close to the `0.7122` hybrid.
- **What did this reveal?** The sparse information was largely sufficient, but learning a stable multi-step transition system required optimizing the rollout rather than independent components alone.

### Put on the slide

| Q1→Q5 chain variant | Top-1 on the CIFAR analysis corpus |
|---|---:|
| Raw independent transitions | `0.012` |
| Re-mask after every transition | `0.213` |
| Re-ground after every transition | `0.593` |
| Re-ground + scheduled sampling | `0.6624` |
| Re-ground + full BPTT | **`0.7010`** |
| Rollout-aligned raw training | `0.6821` |
| Frozen-block hybrid upper bound | `0.7122` |
| Original frozen model | `0.7183` |

**Why individual reconstruction was insufficient**

Each transition was initially trained on a true sparse input but consumed another predictor's output during rollout. Small one-step errors changed the next input distribution and compounded over four transitions.

**Why re-grounding helped**

At each destination, the system decoded the prediction, re-encoded it with the destination SAE, and applied the sparse mask again. This projected the carrier toward the destination SAE's training distribution.

### Training mechanisms — slide-ready version

**Starting problem**

```text
independent training:  real sparse Q1 → T12 → target Q2
                       real sparse Q2 → T23 → target Q3

chain-time reality:    predicted Q1→Q2 output → T23 → predicted Q2→Q3 output → T34 → ...
```

| Mechanism | Training change | Result | Interpretation |
|---|---|---:|---|
| **Scheduled sampling + re-grounding** | Early training → next predictor receives the real sparse code. Later training → increasingly often receives the previous predictor’s **decode → encode → Top-K** code. | `66.24%` | Better exposure to chain-time mistakes; still limited because each transition largely optimized locally. |
| **Full BPTT + re-grounding** | Entire Q1→Q5 rollout remains in one training computation → losses at Q4/Q5 update T12/T23 as well as later predictors. | **`70.10%`** | Better result; earlier predictors learn to avoid errors that damage later stages. |

```text
independent raw chain                 → 1.2%
re-grounded independent chain         → 59.3%
scheduled sampling + re-grounding     → 66.24%
full BPTT + re-grounding              → 70.10%
hybrid real-CNN-block comparison      → 71.22%
```

**Mechanism summary**

```text
scheduled sampling → train on increasingly realistic inputs
full BPTT          → optimize the whole chain for its final outcome
```

**What full BPTT revealed**

Training through the entire re-grounded rollout raised top-1 to `0.7010`, within `1.12 pp` of the `0.7122` hybrid. System-level training closed most of the gap left by independently trained components.

All values on this slide come from the historical 5%-of-latent setup with `K = 8C`. That active count equals 40% of the original activation-node count before support bits are charged.

### Say aloud

“The strongest result here is the shape of the recovery. The raw chain was effectively unusable even though most one-step transitions looked good. Changing the system-level training objective recovered almost all of the hybrid upper bound.”

### Suggested visual

Build a fresh bar chart from the table above. The existing `report/plots/fig5_chain.png` predates the BPTT and rollout-aligned raw results, so it should not be used unchanged.

### Evidence

- Initial chain variants: `docs/PHASE3_RESULTS.md:18-40`.
- Scheduled sampling and BPTT: `docs/PHASE3b_RESULTS.md:1-58`.
- Rollout-aligned raw chain: `docs/PHASE3b_changed_RESULTS.md:63-109`.

### Transition

“Once the system worked on the anchor model, I tested whether the finding was narrow.”

## Slide 9 — Testing Whether the Finding Was General

### Title

**Transfer strengthened the idea and exposed a stability boundary**

### Direct answers to the slide questions

- **Why run transfer experiments?** To determine whether the result was a seed accident or an artifact of one ResNet depth, dataset, or convolutional architecture.
- **What new setups were chosen?** ResNet-20/CIFAR-10 for a backbone-plus-dataset shift; three ResNet-56 seeds for stability; ResNet-110/CIFAR-100 for depth; and ViT-Small and ViT-Base on Imagenette for an architecture shift to patch-token grids.
- **What hypothesis did transfer test?** That sparse reconstruction would preserve the frozen model's behavior across related CNNs and a different token-based vision architecture.
- **What transferred successfully?** Layerwise reconstruction faithfulness transferred across the tested CNN settings and into broad stable regions of ViT-Small and ViT-Base.
- **What did not transfer?** The transition and full-chain experiments were not run beyond the original ResNet-56 setup. ViT-Base stability was also not uniform: 19 of 60 historical sweep runs collapsed.
- **Was the result stronger or weaker than expected?** ViT-Base's best stable result was stronger than expected (`0.9851 ± 0.0012` agreement), while its seed-dependent collapse regions were weaker than expected.
- **How did transfer change confidence?** It increased confidence that sparse reconstruction faithfulness was not unique to one CNN, while lowering confidence in any claim of uniform stability.
- **What new claim became possible?** Sparse reconstruction can preserve predictions in both convolutional activation fields and ViT patch-token grids within the tested regimes. The evidence does not yet transfer the learned hierarchy or semantic-interpretability claims.

### Put on the slide

**CNN robustness, qualified by the evaluation audit**

- ResNet-20 / CIFAR-10 preserved about `0.923` original top-1 across all sections; at 2% latent retention, prediction agreement ranged from `0.9710` to `0.9942`.
- Three ResNet-56 seeds at 5% preserved the same qualitative ordering; `Q3` remained weakest.
- ResNet-110 / CIFAR-100 preserved high agreement at 5%; `Q3` was again lowest (`0.9356`).

**ViT transfer**

- First ViT-Small/Imagenette artifact, block 6 at 5%: `0.9440` prediction agreement, `0.0427` KL.
- ViT-Small sweep, 75 cells: best stable setting `0.9307 ± 0.0034` agreement at block 10 and 12%.
- ViT-Base sweep, 60 cells: best stable setting `0.9851 ± 0.0012` at block 10 and 12%.
- ViT-Base also had 19/60 reconstruction-collapse runs; the 41 non-collapse runs averaged `0.9627` agreement.

**What the transfer changed**

The claim expanded from one ResNet to sparse spatial fields in CNNs and patch-token grids in ViTs. The ViT-Base failures made the claim conditional: high faithfulness is possible, while stable training across depth, budget, and seed remains unresolved.

The historical ViT models used `K = 4D`, so 5% and 12% latent retention correspond to active counts equal to 20% and 48% of the original patch-field node count before support bits. These runs establish sparsification and self-consistency rather than net bit compression.

### Say aloud

“I did not use transfer only to add another benchmark. It tested whether the mechanism depended on convolutional activations. The answer was encouraging on ViT token grids, and the larger model exposed a new product-relevant problem: the best regime was excellent, but reliability was bimodal.”

### Suggested visual

- `analysis/figures/figure_phase4_extensions.pdf` for the broad transfer story.
- `analysis/figures/fig20_vit_base_heatmaps.pdf` or `fig20_vit_base_seed_traces.pdf` to show the stable and collapse-prone regimes.

### Evidence

- ResNet-20 and seed tables: `docs/PHASE4_FULL_BREAKDOWN.md:87-209`.
- ResNet-110 table: `docs/PHASE4_FULL_BREAKDOWN.md:256-293`.
- ViT-Small artifact: `runs/vit_sae/vit_result.json`.
- ViT-Small sweep: `docs/PHASE4_FULL_BREAKDOWN.md:324-428`.
- ViT-Base sweep: `docs/PHASE4_FULL_BREAKDOWN.md:558-741`.

### Transition

“The transfer results increased confidence in the mechanism, but an audit showed that the strength of several claims depended on the evaluation protocol.”

## Slide 10 — Auditing the Evaluation

### Title

**I found flaws in the evidence and narrowed the claim**

### Direct answers to the slide questions

- **What issue was discovered?** Historical CIFAR trainers evaluated the test corpus every epoch; threshold-based Top-K could retain more than the declared count under ties; and retention was measured relative to an overcomplete latent without complete support/value bit accounting.
- **How was it discovered?** A code-and-artifact audit compared the intended experiment contracts with `train_sae.py`, `train_transition.py`, `train_chain.py`, the masking implementation, and saved results.
- **Which results were affected?** Core historical CIFAR SAE, transition, and chain results were exposed repeatedly to the analysis corpus. The feature and intervention audits inherited risk from those SAEs. Historical retained-fraction language affected both CNN and ViT rate claims.
- **What claims could those results no longer support?** Untouched external generalization, literal “5% compression,” exact active-count guarantees under tied values, or strong semantic-feature claims.
- **Which results remained useful?** Historical results remain evidence of corpus-conditioned mechanistic faithfulness. PCA used a clean fit/evaluate split, older ViT runs used separate train and validation sets, and the completed clean ResNet-20 slice now provides one-shot exact-rate holdout evidence.
- **How did the conclusion language change?** “Compression” became “latent retention” for historical runs; CIFAR performance became “corpus-conditioned faithfulness”; and “isolated semantic features” became “learned feature-location coefficients.”
- **Were experiments rerun or reclassified?** Both. Historical families were classified by validity risk, and a clean redo introduced explicit train/validation/holdout splits, exact original-node budgets, bit accounting, manifests, controls, and replay checks.
- **How would I prevent it earlier?** Freeze the split, selection rule, exact sparse-message definition, controls, and result schema before training; keep the holdout inaccessible until selection; and have the pipeline reject over-budget or provenance-incomplete artifacts automatically.

### Put on the slide

**What the audit found**

1. Core CIFAR SAE, transition, and chain trainers evaluated on the cached test split after every epoch. The test set therefore functioned as a repeatedly inspected analysis corpus.
2. Historical Top-K masking used a threshold, so tied values could retain more than the intended count.
3. Historical budgets were fractions of the overcomplete latent field, rather than active coefficients or payload bits relative to the original activation.
4. Some follow-up audits inherited the validity risk of the SAEs on which they depended.

**What changed**

- CIFAR top-1 numbers became **corpus-conditioned mechanistic faithfulness measurements**, not untouched external generalization estimates.
- “5% compression” became **5% latent retention** unless support and value bits were explicitly counted.
- Semantic feature-isolation claims remained open.

**Response implemented in the clean redo**

- deterministic train/validation/holdout splits;
- validation-only selection and one holdout evaluation after choices are frozen;
- exact `M = round(fraction × C × H × W)` active coefficients per activation map;
- explicit support-index, coefficient-value, metadata, and total-bit accounting;
- immutable run roots, manifests, hashes, controls, and replay checks;
- matched families, budgets, and seeds across two CNNs and two ViTs.

**Current status**

The expanded clean redo declares 960 final cells and 1,200 dependency-aware training jobs. At inspection time, execution was stopped and 351 completion manifests existed: all 300 ResNet-20 job segments were complete, 51 ResNet-110 segments were complete, and no clean ViT or clean-chain result existed. The aggregate selection, required clean CNN chain phase, and final cross-model conclusion are still open. A smaller predecessor result exists, but the current contract explicitly supersedes it, so it should stay out of the headline deck.

**Early clean evidence, clearly labeled as one completed slice**

For the `field_recovery` family on ResNet-20/CIFAR-10, the 15 completed holdout cells across five sections and three seeds averaged:

| Exact original-node budget | Mean prediction agreement | Mean spliced top-1 | Frozen-model top-1 |
|---:|---:|---:|---:|
| 8% | `0.9560` | `0.9106` | about `0.923` |
| 2% | `0.9013` | `0.8705` | about `0.923` |

Each artifact records `evaluated_once: true` on the official test split and an exact section-specific active count. This result is strong evidence that the corrected protocol works on one backbone/dataset slice. It is not yet a cross-family winner or cross-model conclusion.

### Say aloud

“The audit did not make the existing mechanistic signal disappear. It changed the level of claim I was willing to make. I converted that finding into an engineering response: clean splits, one-shot holdouts, exact sparse messages, bit accounting, and immutable provenance.”

### Suggested visual

A before/after table:

| Historical pipeline | Clean redo |
|---|---|
| Test inspected during development | Train → validation selection → one holdout |
| Threshold Top-K | Exact index-scatter Top-M |
| Latent-relative fraction | Original-node-relative active count + bits |
| Partial run metadata | Immutable manifests, hashes, controls, replay |

### Evidence

- Confirmed exposure audit: `analysis/leak_audit.json`.
- Historical repeated evaluation: `src/train_sae.py:77-78` and `110-117`; `src/train_transition.py:71-92`; `src/train_chain.py:61-105`.
- Redo findings and response: `analysis/sae_redo_level2_evidence.md:3-18`.
- Current contract: `docs/plans/SAE_REDO_LEVEL2_CONTRACT.md`.
- Expanded matrix manifest: `runs/sae_sparse_recovery_redo/queues/20260829T043453Z_launched/queue_manifest.json`.
- Clean ResNet-20 holdouts: `runs/sae_sparse_recovery_redo/matrix/20260829T043453571667Z/cnn-r20/resnet20_cifar10/Q1..Q5/field_recovery/b08, b02/seed0..2/fidelity_result.json`.

### Transition

“With that calibration, three conclusions remain useful and defensible.”

## Slide 11 — What the Evidence Actually Says

### Title

**Three findings I would defend today**

### Direct answers to the slide questions

- **What is the strongest supported conclusion?** In the studied models, sparse feature-location messages can replace individual hidden fields while preserving much of the frozen model's behavior. Multi-step learned composition is possible but materially harder.
- **What is the strongest quantitative result?** The clean ResNet-20 `field_recovery` slice averaged `0.9560` prediction agreement and `0.9106` spliced top-1 at an exact 8% original-node budget across five sections and three seeds, versus about `0.923` original top-1. The strongest system result was `0.7010` full-chain top-1 versus `0.7122` hybrid and `0.7183` original on the historical CIFAR corpus.
- **What was most surprising?** Rollout-aligned raw training recovered the chain from near chance to `0.6821`, showing that the original failure was substantially a training-distribution problem.
- **Where does the approach work?** Individual layers across the tested ResNets and stable ViT block/budget settings; re-grounded or rollout-aligned chains on ResNet-56.
- **Where does it struggle?** Aggressive exact budgets in early ResNet-20 layers, the historical `Q3→Q4` transition, unaligned raw chaining, and seed-sensitive ViT-Base settings.
- **Which factors matter most?** Sparse-rate definition, spatial context, overlap, layer depth/resolution changes, exposure to rollout states, end-to-end optimization, seed stability, and evaluation hygiene.
- **What did I initially believe that I no longer believe?** I no longer believe re-grounding is strictly necessary. The evidence shows it is one strong stabilization method; rollout alignment can also produce a viable raw chain.
- **What claims am I deliberately not making?** Automatic semantic interpretability, universal model/dataset transfer, net bit compression for historical runs, a universal `Q3` bottleneck, improved task performance, or a fully validated causal feature tree.

### Put on the slide

1. **Sparse spatial representations can preserve behavior in the studied vision models.** Under the clean exact-rate protocol, the completed ResNet-20 `field_recovery` slice averaged 95.60% prediction agreement and 91.06% spliced top-1 at an exact 8% original-node budget, compared with about 92.3% original top-1. Historical ResNet and ViT results broaden the signal, with the evaluation qualifications described in the preceding slide.

2. **Local success is a weak predictor of system success.** Strong layerwise reconstruction and mostly strong one-step transitions still produced a 1.2% raw-chain top-1. Rollout-aware training raised full-chain performance to 68.21%, and re-grounded full-BPTT training reached 70.10% against a 71.22% hybrid bound.

3. **Architecture, spatial context, and training stability matter more than one universal sparsity setting.** Field structure helped early CNN sections, `Q3→Q4` remained the hardest edge, overlap beat coarse disjoint tiling, and ViT-Base produced both excellent stable cells and catastrophic collapsed seeds.

**What surprised me most**

The strongest surprise was not the initial raw-chain failure. It was that rollout-aligned raw training later recovered to `0.6821`, forcing a revision from “re-grounding is necessary” to “distribution alignment and manifold control are both viable stabilization mechanisms.”

**Claims deliberately left open**

- the SAE features are automatically human-interpretable or monosemantic;
- sparse activation locations are causal evidence locations;
- the method achieves net storage or communication compression at the historical retention settings;
- the findings generalize to arbitrary vision models, datasets, or real-world distribution shifts;
- the method improves the original model's task performance.

### Say aloud

“The project produced a positive result, a systems lesson, and a research-discipline lesson. The most valuable outcome is a more precise model of where sparse representations work and what must be controlled before scaling them.”

### Suggested visual

Three large claim cards, each with one number and one limitation.

### Evidence

- Current synthesized analysis: `analysis/report.pdf`, sections 1, 4, and 5.
- Open questions: `analysis/open_questions.md`.

### Transition

“The remaining uncertainty makes the next month easy to prioritize.”

## Slide 12 — What I Would Do Next

### Title

**Finish the validity test before scaling the claim**

### Direct answers to the slide questions

- **What is the highest-value unanswered question?** Whether the sparse-faithfulness advantage survives an exact matched message budget and untouched holdout across all declared CNN and ViT families.
- **What experiment would reduce the most uncertainty?** Complete the frozen four-family, four-budget, three-seed clean matrix; select using validation only; evaluate each winner once on holdout; then run matched raw and re-grounded clean CNN chains.
- **Why prioritize it?** It determines whether the main result survives the evaluation and rate-accounting corrections. Scaling models or adding visualizations cannot repair an invalid comparison.
- **What result would increase confidence?** Consistent paired wins over PCA, random/magnitude controls, and the incumbent SAE at equal exact `M` and payload rate, across sections, seeds, backbones, and untouched holdouts.
- **What result would change or end the hypothesis?** Gains disappearing under matched bits, persistent collapse across seeds, or strong reconstruction repeatedly failing to preserve downstream behavior would force a redesign. Semantic interventions performing no better than controls would end the feature-isolation claim.
- **With limited compute, what comes first?** Three seeds at the known hard and representative fields: early, `Q3`, and late CNN sections plus ViT blocks 4 and 10 at 2% and 5%, with clean splits and controls preserved.
- **With unlimited compute, what scales?** Complete the matrix, add ResNet-50/ImageNet-scale data, larger ViTs and more datasets, build architecture-aware transition models, and run rigorous semantic/causal feature evaluations.
- **What infrastructure should improve?** Automatic split and artifact manifests, exact-budget enforcement, payload accounting, checkpoint selection, completion validation, collapse diagnostics, paired controls, and provenance-linked reporting.

### Put on the slide

**Highest-value question**

Does the observed faithfulness advantage survive an exact, matched sparse-message budget and a genuinely untouched holdout across model families?

**Next-month priority**

1. Resume and finish the declared clean matrix without changing selection rules after seeing results.
2. Validate all completion manifests and report every seed, collapse, retry, rate term, and missing cell.
3. Compare the four SAE families at the same original-node `M` and actual payload bits.
4. Freeze winners using validation only, then run one holdout evaluation.
5. Run the required matched raw versus re-grounded full-BPTT CNN chains.
6. Only after faithfulness is secure, complete feature overlays, semantic labeling, and targeted feature interventions.

**Decision rules**

- **Increase confidence:** the result holds across sections and both CNN/ViT families, beats random/PCA/incumbent controls at matched rate, and survives the untouched holdout.
- **Change direction:** gains vanish under matched bit cost, collapse remains common, or strong reconstruction repeatedly fails to preserve behavior.
- **Abandon the semantic-isolation claim:** feature activations lack coherent examples and causal interventions do not outperform random or matched controls.

**If compute is limited**

Prioritize three seeds at the known hard and representative points: CNN `Q3`, one early and one late section, and ViT blocks 4 and 10 at 2% and 5%. Preserve the clean split and controls rather than expanding the grid.

**If compute is abundant**

Complete the frozen matrix, then extend to ResNet-50/ImageNet-scale data, larger ViTs, additional datasets, paired bootstrap intervals, architecture-aware `Q3→Q4` transitions, and semantic/causal feature evaluation.

**Infrastructure investment**

Automate split manifests, exact-budget checks, result-schema validation, collapse detection, paired controls, and claim-generation from immutable artifacts.

### Say aloud

“I would spend the next month reducing the uncertainty that can invalidate the whole conclusion. A larger model is lower priority than proving that the comparison is fair and the holdout is untouched.”

### Evidence

- Frozen matrix and decision rules: `docs/plans/SAE_REDO_LEVEL2_CONTRACT.md`.
- Remaining clean chain requirement: the contract section “Required post-matrix CNN re-grounded chains.”
- Existing feature-visualization gap: `docs/CURRENT_PROGRESS_AND_TODO.md:177-262`.

### Transition

“That prioritization reflects the general research process I would bring to an RPM role.”

## Slide 13 — How I Approach Research Problems

### Title

**Frame uncertainty, build evidence, and update the plan**

### Direct answers to the slide questions

- **How did I turn an ambiguous idea into a roadmap?** I translated “automate feature isolation” into measurable stages: layer reconstruction, behavioral splice fidelity, adjacent prediction, full-chain composition, failure diagnosis, transfer, and evaluation audit.
- **How did I decide what to measure?** I selected metrics that separated hidden-state fidelity, model-output fidelity, labeled performance, sparse-message cost, and training reliability.
- **How did I choose the next experiment?** I chose the least expensive experiment that distinguished the leading explanations for the current bottleneck: capacity, rollout distribution, manifold drift, spatial context, or budget definition.
- **How did I handle conflicting evidence?** I retained both outcomes and narrowed the claim. Re-grounding improved one chain, while rollout-aligned raw training later showed it was not uniquely necessary.
- **How did I react to a flawed experiment design?** I reclassified the affected results, changed the language of the claims, froze a clean protocol, and implemented exact rate and provenance checks.
- **How did I distinguish interesting from trustworthy?** An interesting result motivated a follow-up. A trustworthy result also needed appropriate controls, fair budgets, multiple seeds where practical, clean data separation, and reproducible artifacts.
- **How did I decide whether to continue or change direction?** I continued when a result preserved the core hypothesis and exposed a testable bottleneck. I changed direction when a follow-up contradicted the mechanism, as with the larger predictor, raw rollout recovery, and RF budgeting.

### Put on the slide

```text
Frame the capability
→ define falsifiable evidence
→ run the cheapest discriminating test
→ inspect the failure mode
→ update the hypothesis
→ test composition and transfer
→ audit the evaluation
→ scale only the trustworthy result
```

### Tie each step to this project

| Research behavior | Project example |
|---|---|
| Frame an ambiguous capability | “Preserve internal computation” became reconstruction + splice faithfulness + chain composition. |
| Define evidence before training | Random/PCA controls, output agreement, KL, and a one-point top-1 gate. |
| Run the cheapest useful test | One SAE per section before investing in transitions. |
| Use failure diagnostically | `Q3→Q4` and raw-chain failures generated targeted follow-ups. |
| Change a belief | Raw rollout recovery weakened the claim that re-grounding was mandatory. |
| Test breadth | Seeds, ResNet-20, ResNet-110, ViT-Small, and ViT-Base. |
| Audit the evidence | Repeated test exposure and sparse-budget accounting changed the claims and infrastructure. |
| Prioritize by uncertainty | Finish clean matched-rate holdouts before larger scaling or semantic claims. |

### Say aloud

“My operating principle is to make the next experiment answer a decision. I want a roadmap in which positive, negative, and flawed results all cause a visible update to the program.”

### Optional Abundant connection

“Abundant describes an RPM as a PM of the model: someone who defines a capability, builds the benchmark and data pipeline, hill-climbs through experiments, and remains the voice of the model user. This project is the clearest example of how I do that work.”

Reference: [Abundant Research Product Manager role](https://jobs.ashbyhq.com/abundant/29a78fe0-e392-47b0-81f4-5a540f2c36aa/), accessed 2026-09-10.

---

# Appendix A — Experiment map

| Stage | Question | Main comparison | Decision or learning |
|---|---|---|---|
| Phase 1 | Can one section be sparsified? | Trained VectorSAE vs random and PCA | Continue: downstream behavior survived. |
| Phase 1 Pareto | Where is the useful budget knee? | 1–20% latent retention by section | Most sections saturated around 3–8%; `Q3` needed more. |
| Phase 2 | Which architecture/mask works? | Field/Vector × global/RF | Select Field + global for hierarchy. |
| Phase 3 | Can sparse codes predict the next section? | Learned transition vs frozen-block hybrid | Most edges strong; `Q3→Q4` bottleneck. |
| Phase 3 chain | Do one-step transitions compose? | Raw, re-mask, re-ground, hybrid | Raw chain collapsed; projection recovered much of the gap. |
| Phase 3b | Can chain-aware training close the gap? | Scheduled sampling and full BPTT | Re-grounded chain reached `0.7010`. |
| Changed suite | Were earlier explanations too strong? | Fair RF, raw rollout, overlap/disjoint patches | Budget, rollout distribution, and overlap all mattered. |
| Phase 4 | Does the finding transfer? | Seeds, CNNs, ViTs | Broad positive signal plus ViT-Base instability. |
| Evaluation audit/redo | Are the comparisons and claims valid? | Clean splits, exact message, bit rate, provenance | Historical claims narrowed; clean matrix remains in progress. |

# Appendix B — Metric definitions for Q&A

### Relative L2

\[
\mathrm{relL2}=\frac{\lVert H-\widehat H\rVert_2}{\lVert H\rVert_2}.
\]

Lower is better. It measures reconstruction error relative to activation magnitude.

### Fraction of variance unexplained

\[
\mathrm{FVU}=\frac{\lVert H-\widehat H\rVert_2^2}{\lVert H-\bar H\rVert_2^2}.
\]

Lower is better. Values above one mean the reconstruction is worse than predicting the mean under this normalization.

### Cosine similarity

The mean cosine similarity between flattened original and reconstructed activation fields. Higher is better, with 1 indicating the same direction.

### Prediction agreement

The fraction of examples on which the original and spliced frozen models have the same argmax prediction. This is label-free self-consistency.

### Spliced top-1

Classification accuracy after replacing an intermediate activation with its sparse reconstruction and running the frozen remainder of the network.

### KL divergence

`KL(original || spliced)` compares full output distributions. Lower is better. It can detect behavior changes that argmax agreement misses.

### Frozen-block hybrid

At each stage, decode the sparse code and pass it through the model's real frozen block. It estimates whether the sparse message contains enough information when the original transition computation is retained. It is an empirical upper bound for the learned transition family, not a mathematical upper bound over every possible predictor.

### Re-grounding

```text
predicted destination code
→ destination decoder
→ destination normalization
→ destination encoder
→ destination sparse mask
→ next transition
```

It projects a predicted carrier back through the destination SAE before the next step.

# Appendix C — Presentation-safe claim audit

| Claim | Verdict | Presentation-safe wording | Main risk / next evidence |
|---|---|---|---|
| Individual activation fields support faithful sparse messages | **SUPPORTED, scoped** | In the completed clean ResNet-20 slice, `field_recovery` averaged `0.9560` agreement and `0.9106` spliced top-1 at an exact 8% original-node budget across five layers and three seeds. Historical CNN and ViT results broaden the signal with stated caveats. | Clean exact-rate evidence currently covers one completed backbone/dataset slice. Finish R110 and ViT holdouts before making a broad claim. |
| FieldSAE is the best architecture | **REVISE** | FieldSAE won the historical aggressive-sparsity CNN comparison. | ViT results are mixed, and the current four-family matched matrix is incomplete. |
| Sparse transitions preserve a full chain | **REVISE** | Re-grounded full-BPTT reached `0.7010` versus a `0.7122` hybrid; rollout-aligned raw training reached `0.6821`. | Historical CIFAR chain is corpus-conditioned; clean matched-rate chain is pending. |
| `Q3→Q4` is the main bottleneck | **REVISE** | `Q3→Q4` was the weakest edge in the original ResNet-56/CIFAR-100 transition experiment and resisted a larger predictor. | No corresponding transition sweep exists across other backbones; the clean R20 reconstruction slice does not establish a universal `Q3` bottleneck. |
| The method transfers to ViTs | **SUPPORTED** in a qualified form | Sparse patch-token reconstructions preserved predictions across broad ViT-Small settings and strong stable ViT-Base settings. | ViT-Base was seed-fragile, and older sweeps used validation as the final reported evaluation set. |
| Learned SAE features are automatically interpretable | **BLOCKED** | The pipeline learns feature-location coefficients; semantic interpretability remains a next-stage evaluation. | Need overlays, top examples, labels, and causal interventions with controls. |
| The sparse code is a causal feature hierarchy | **REVISE** | Hybrid and intervention results suggest causal sufficiency and localized effects in the studied setup. | Intervention artifacts inherit upstream validity risk and do not establish a complete semantic tree. Do not show the `Q2→Q3` 38-million ratio: random ablation slightly improved accuracy and the ratio is dominated by a clamped denominator. |

**CLAIM REVIEW:** Use the qualified language above. The strongest current claims concern mechanistic faithfulness and system composition. Semantic interpretability, universal generalization, and net bit compression remain open.

# Appendix D — Citation audit and recommended references

**CITATION AUDIT: PASS for the corrected references below.** The supplied draft required two revisions: arXiv:2308.00312 is not the original Elad-Bruckstein paper, and LoRA does not support activation sparsity.

All sources below were checked against primary or authoritative records on 2026-09-10. No withdrawal or correction notice was visible on those records at access time.

1. Michael Elad and Alfred M. Bruckstein. [“A Generalized Uncertainty Principle and Sparse Representation in Pairs of Bases.”](https://doi.org/10.1109/TIT.2002.801410) *IEEE Transactions on Information Theory* 48(9):2558–2567, 2002. Use for the coherence-based support uncertainty principle.
2. Chris Olah, Alexander Mordvintsev, and Ludwig Schubert. [“Feature Visualization.”](https://distill.pub/2017/feature-visualization/) *Distill*, 2017. DOI: 10.23915/distill.00007. Use for activation maximization and the distinction between feature visualization and attribution.
3. Nick Cammarata et al. [“Thread: Circuits.”](https://distill.pub/2020/circuits/) *Distill*, 2020. DOI: 10.23915/distill.00024. Use for the labor-intensive, neuron-by-neuron circuit-analysis motivation.
4. Hoagy Cunningham et al. [“Sparse Autoencoders Find Highly Interpretable Features in Language Models.”](https://arxiv.org/abs/2309.08600) arXiv:2309.08600, 2023. Use as background that SAEs can learn candidate feature directions; avoid importing its interpretability conclusion into this vision project without evidence.
5. Leo Gao et al. [“Scaling and Evaluating Sparse Autoencoders.”](https://arxiv.org/abs/2406.04093) arXiv:2406.04093, 2024. Use for Top-K SAE motivation and the need to evaluate both reconstruction and feature quality.
6. Edward J. Hu et al. [“LoRA: Low-Rank Adaptation of Large Language Models.”](https://arxiv.org/abs/2106.09685) arXiv:2106.09685, 2021. Optional analogy only; it concerns low-rank parameter updates rather than sparse hidden activations.

# Appendix E — Likely interviewer questions

### Why use an overcomplete SAE if your goal is compression?

The overcomplete dictionary gives the model more candidate directions, while the sparse active support restricts which directions are used per activation. The historical experiments establish a reconstruction-sparsity tradeoff, not complete communication compression. The redo therefore compares exact active counts and payload bits, including support indices and coefficient values.

### Why is prediction agreement better than task accuracy?

They answer different questions. Task accuracy measures performance against labels. Agreement measures whether the sparse-spliced model reproduces the original frozen model's behavior, including on examples where the original model is wrong. For mechanistic faithfulness, agreement and KL are direct measures; task accuracy remains a useful guardrail.

### Why call the hybrid an upper bound?

It uses the same sparse representation but hands the transition back to the real frozen block. It is an empirical ceiling for the tested learned transition family under that representation and evaluation setup. It does not rule out a better learned architecture exceeding it.

### Why did full BPTT help?

Independent transition losses do not train early predictors for the downstream consequences of their errors. Full BPTT lets later losses update earlier transitions through the rollout, aligning optimization with the final chain objective.

### Did re-grounding solve the chain problem?

It was highly effective: the original chain rose from `0.012` raw to `0.593` with re-grounding, then `0.7010` with full-BPTT training. A later raw-carrier experiment reached `0.6821`, showing that rollout-aligned training can also stabilize composition. The evidence supports two mechanisms: projection toward the learned manifold and exposure to the rollout distribution.

### Why was `Q3→Q4` hard?

The evidence identifies it as the main bottleneck but does not isolate one cause. It coincides with a `16×16 → 8×8` resolution change; a larger generic predictor failed; and local/multiscale context mattered elsewhere. The next discriminating test is an architecture-aware downsampling transition under the clean protocol.

### Does high reconstruction imply interpretability?

No. It shows that the sparse representation retains information. Interpretability needs evidence that individual dictionary elements correspond to coherent concepts and have predictable causal effects. The repository has basic dictionary-health and intervention audits, but the general feature-overlay and semantic-validation pipeline remains unfinished.

### What was the most important research-management decision?

A strong answer is: “I treated the evaluation audit as a change to the research program rather than a footnote. I narrowed the claims, froze a clean selection protocol, added exact rate accounting and immutable provenance, and prioritized completing that evidence before scaling further.” Personalize this to what you actually owned.

### What would make you stop the project?

Stop or substantially redirect if the exact-rate clean holdout shows no advantage over PCA, the incumbent FieldSAE, or simple magnitude/random controls; if training collapse remains common across seeds; or if semantic and causal evaluation shows no improvement over simpler representations. Those outcomes would mean the current complexity is not earning its cost.

# Appendix F — Figure and artifact shortlist

| Use | Preferred artifact | Note |
|---|---|---|
| Layerwise Pareto | `analysis/figures/figure_phase1_pareto.pdf` | Includes the evaluation caveat in its caption. |
| Architecture branch | `analysis/figures/figure_phase2_grid.pdf` | Useful as backup detail. |
| Transition bottleneck | `analysis/figures/figure_phase3_chain_and_edges.pdf` | Shows chain progression and weak edge. |
| Transfer overview | `analysis/figures/figure_phase4_extensions.pdf` | Mixed CNN/ViT evidence. |
| ViT-Small stability | `analysis/figures/figure_vit_sweep_heatmaps.pdf` | Broad sweep view. |
| ViT-Base instability | `analysis/figures/fig20_vit_base_seed_traces.pdf` | Best view of bimodal seeds. |
| ViT-Base stable/collapse map | `analysis/figures/fig20_vit_base_heatmaps.pdf` | Shows mean, spread, KL, and collapse count. |
| Intervention/audit | `analysis/figures/figure_intervention_and_audit.pdf` | Backup only; inherits upstream risk. |
| Exact raw data | `analysis/results_tidy.csv` | Provenance-linked long-form metrics. |
| Current synthesized report | `analysis/report.pdf` | Primary historical analysis source. |

## Final deck-editing rules

- Keep one message per slide and no more than one or two headline numbers.
- Put metric definitions and full tables in backup slides.
- Label historical CIFAR results as corpus-conditioned.
- Say “latent retention” for historical fractions and reserve “compression” for exact bit-accounted results.
- Say “feature-location coefficient” until semantic interpretation is demonstrated.
- Make the main narrative about decisions: what evidence you needed, what failed, what belief changed, and what you prioritized next.
- Spend the most speaking time on Slides 3, 6, 7, 8, 10, and 12. Those slides show the RPM work.
- Keep personal ownership explicit and accurate.

# TODO & Project Handoff — Sparse Signal-Field SAEs

This file gives a new contributor everything needed to (a) understand the project and code, and (b) drive it to a publishable paper. For results/figures see [`report/REPORT.pdf`](report/REPORT.pdf); for setup see [`README.md`](README.md).

---

## 1. What this project is (context)

**Hypothesis.** A frozen vision model's intermediate activations look dense in their native channels but are **sparse in a learned transform domain** — like an image being dense in pixels but sparse after a wavelet transform. We test three nested claims on a frozen CNN:
1. **Layerwise sparse reconstruction** — each layer's activation field can be reconstructed from a tiny masked subset of learned feature-location coefficients, preserving the network's prediction.
2. **Cross-section prediction** — the sparse code at layer *t* predicts the sparse code at layer *t+1*.
3. **Chained hierarchy ("sparsity tree")** — chain those predictions Q1→Q5 and still preserve the classifier output.

**Key construct.** At 5 depths `Q1…Q5` we train a Sparse Autoencoder (SAE) mapping activation field `H (C×H×W)` → coefficient field `Z (K×H×W, K=8C)`, apply a **mask** that keeps only a few % of coefficients (global, or receptive-field-local), decode `Ĥ`, and splice `Ĥ` back into the frozen net to measure accuracy/KL vs the original. Two SAE designs (per-location **VectorSAE**; convolutional **FieldSAE**) × two masks (**global** TopK; **receptive-field-local** TopK) × two training protocols (independent; incremental-dependent) = the 8-experiment grid in the design spec.

**What we found (one line each):**
- Claim 1 ✅: ~71% spliced top-1 at 2–5% retained (97.6% prediction agreement with the CNN); beats PCA + random-dict.
- Architecture: **FieldSAE ≫ VectorSAE** at aggressive/local sparsity.
- Claims 2–3 ✅ *with re-grounding*: frozen-block hybrid chain 0.712; learned chain works only when each step is re-grounded on the SAE manifold (0.012 raw → 0.701 with BPTT re-grounding; bound 0.712).
- Causality (N8): top parents matter 3–7× more than random; influence is 3–55×  localized to the RF.
- Features are clean (0–6% dup, 1–3% dead). Negative: Q3→Q4 bottleneck is a representation gap, not predictor capacity.

**Closest prior art** (situate the paper against these): PatchSAE (ICLR'25), saev/Stevens'25, SAE-V, Prisma; Anthropic cross-layer transcoders & crosscoders, Marks Sparse Feature Circuits; classical CRsAE/SSCAE/Zeiler-deconv/LeCun-PSD. Full list in `report/REPORT.md` §6.

---

## 2. Code & file map (what to read first)

Read in this order to understand the pipeline:
1. `src/backbone.py` — the frozen CNN, the 5 taps, and the **exact** splice helpers (`forward_from`, `forward_between`). Verified zero-error.
2. `src/sae.py` — `FieldSAE` (ConvNeXt-style) and `VectorSAE`. **No skip around the bottleneck.**
3. `src/masks.py` — `mask_global_topk`, `mask_rf_topk` (vectorized).
4. `src/train_sae.py` — layerwise SAE training (warmup→anneal→fixed curriculum) + `--random` control.
5. `src/eval.py` — recon, downstream-preservation, sparsity metrics.
6. `src/transitions.py` + `src/train_transition.py` — Family-I learned transition + N4 frozen-block hybrid.
7. `src/train_chain.py` + `src/eval_chain.py` — chaining, scheduled sampling, `--bptt` re-grounding.
8. `src/intervene.py`, `src/feature_audit.py`, `src/taxonomy.py`, `src/vit_sae.py` — Tier experiments.
9. `scripts/run_grid.py` — multi-GPU grids; `scripts/make_*.py` — figures.

Results live in `runs/<phase>/<cell>/result.json`; checkpoints in `runs/.../sae.pt`; per-phase notes in `docs/PHASE*_RESULTS.md`; design spec in `docs/superpowers/specs/`.

**Environment:** always `source env.sh` (project-local cu12 cuDNN preload — the global env's cuDNN is mismatched).

---

## 3. Current experiment scale vs. what's needed

### What has been run so far

| Backbone | Params | Dataset | Train images | Image size |
|----------|--------|---------|--------------|------------|
| ResNet-20 | ~270K | CIFAR-10 | 50K | 32×32 |
| ResNet-56 | ~850K | CIFAR-100 | 50K | 32×32 |
| ResNet-110 | ~1.7M | CIFAR-100 | 50K | 32×32 |

The FieldSAE itself is ~2–4M params per section (larger than the backbones it probes), but the bottleneck for credibility is the backbone and dataset, not the SAE. CIFAR's 32×32 images produce tiny feature maps (e.g. Q5 is 64×8×8), and the 50K training set is toy-scale.

### What the field expects for a publishable claim

| Backbone | Params | Dataset | Train images | Image size |
|----------|--------|---------|--------------|------------|
| ResNet-50 | **25M** | ImageNet-1K | **1.28M** | 224×224 |
| ViT-B/16 | 86M | ImageNet-1K | 1.28M | 224×224 |

The jump is ~30× more model params, ~25× more training data, and ~49× larger spatial resolution. Q3 on ResNet-50 is 28×28×512 vs. 16×16×32 on ResNet-56 — the feature maps are qualitatively richer and the FieldSAE's convolutional advantage over VectorSAE should be even more pronounced.

### Recommended scale progression

1. **Imagenette (done, partial)** — `runs/vit_sae/vit_result.json` exists (ViT-small, block 6, 94.4% pred_agree at 5%). The run exited non-fatally; finish/clean up the full Imagenette sweep with `src/vit_sae.py`.
2. **ResNet-50 / ImageNet-1K (next major target)** — extend `src/backbone.py` to support torchvision ResNet-50, re-implement the 5 taps at the 4 stage boundaries + one mid-stage tap, run the full phase1/phase2 grid. This is the central credibility ask — any reviewer will require it.
3. **ViT-B/16 / ImageNet-1K** — after ResNet-50; `src/vit_sae.py` is the scaffolding. Note: validate spatial masks with causal patch interventions (ViT activation-location ≠ evidence-location), and handle the CLS token explicitly.
4. **Second CNN family** (ConvNeXt or VGG) — shows the effect is not ResNet-specific.

---

## 4. Completed runs (do not re-run)

- ✅ ResNet-56/CIFAR-100 backbone (71.83%), ResNet-20/CIFAR-10 (92.3%), ResNet-110/CIFAR-100 (73.25%)
- ✅ Phase 1 (VectorSAE + global mask anchor grid, all sections)
- ✅ Phase 2 (FieldSAE, RF-local mask, Pareto sweep)
- ✅ Phase 3 transitions (Family-I learned transition, N4 frozen-block hybrid)
- ✅ Phase 3b chain retraining (scheduled sampling + BPTT re-grounding); chain 0.701 vs bound 0.712
- ✅ N8 causal intervention (top parents 3–7× more impactful, 3–55× RF-localized)
- ✅ Feature audit Q5 (0–6% dup, 1–3% dead)
- ✅ Multi-seed stability: all 5 sections × 3 seeds (`runs/seeds/S_Q{1..5}_s{0,1,2}`)
- ✅ R20/CIFAR-10 recon grid (`runs/r20c10/`)
- ✅ ResNet-110 recon (`runs/r110/F_Q{1..5}`)
- ✅ Taxonomy Q4+Q5: sparse (0.706) > dense (0.690) > crosscoder (0.610) on spliced top-1
- ⚠️ ViT-SAE on Imagenette: partial result exists, exited non-fatally — needs completion

---

## 5. Path to paper-ready

### A. Complete the empirical core

- [ ] **Finish the 8-cell grid explicitly** (E1–E8 table) including the incremental-dependent protocol (Family II) for all SAE×mask combos — reviewers will ask for the full factorial, not just the winner.
- [ ] **Full sparsity–fidelity curves** with downstream KL (not just top-1) and per-class breakdown; add an explicit "coefficients-per-pixel" axis so budgets are comparable to PCA/JPEG-style baselines.
- [ ] **Error bars everywhere** — multi-seed is done; now propagate mean ± std into all headline tables (recon Pareto, chain re-grounding, N8 ratios).
- [ ] **Statistical tests**: bootstrap CIs on spliced accuracy; significance of FieldSAE>VectorSAE and of re-grounding gains.
- [ ] **Extend taxonomy to all sections** (currently only Q4+Q5 in `runs/taxonomy/`); make it a clean table situating the paper in Anthropic's sparse-code / transcoder / crosscoder framework.

### B. Strengthen the hierarchy / causal claims (the novel contribution)

- [ ] **Architectural re-grounding as a method**: make decode→re-encode→mask a differentiable layer, train the full chain end-to-end, report the closed gap to the 0.712 bound. (Prototyped via `--bptt`; productionize + ablate.)
- [ ] **N8 → full causal story**: extend parent→child intervention to (i) graded ablation curves, (ii) the Tree-SAE dual criterion (coverage AND decoder-reconstruction), (iii) compare learned-transition edges vs attribution-patching edges (Marks et al.) — "does our learned tree recover the same circuit?".
- [ ] **Solve / characterize the Q3→Q4 bottleneck**: try Matryoshka multi-scale codes (N1), a resolution-aware predictor, or learned upsampling; if it stays hard, frame it as a finding about resolution-drop representation gaps.
- [ ] **Decision-impact vs activation vs evidence location**: add the patch-intervention test distinguishing the three (the spec's §2.6 caveat) so spatial claims are causal, not positional.
- [ ] **Retrain-only ablation** (no re-grounding): retrain chain transitions on predicted carriers without decode→re-encode→mask in the loop, then compare raw chain accuracy against phase3b. Isolates whether hierarchy-learning or manifold projection is the dominant stabilizing mechanism.

### C. Scale (external validity — biggest reviewer ask)

- [ ] **Imagenette cleanup**: finish the ViT-small sweep, report pred_agree / FVU curves across blocks, handle the CLS token.
- [ ] **ResNet-50 / ImageNet-1K**: adapt `src/backbone.py`, implement 5 taps at the 4 stage boundaries, run phase1+phase2 grid. Report the sparsity–fidelity Pareto at this scale. *This is the single most important remaining experiment.*
- [ ] **ViT-B/16 / ImageNet-1K**: extend `src/vit_sae.py`; validate spatial masks with causal patch interventions.
- [ ] **Second CNN family** (ConvNeXt or VGG) to show the effect is not ResNet-specific.

### D. Baselines & positioning

- [ ] Head-to-head **taxonomy** result (sparse-code vs dense transcoder vs crosscoder) extended to all sections — finish `taxonomy.py`, clean table.
- [ ] Compare against **PatchSAE** directly (their per-token SAE) on a shared backbone — quantify the FieldSAE/spatial-mask advantage.
- [ ] Additional compression baselines: **JPEG/wavelet on activations** and a **pruning** baseline, to argue the SAE is a better learned transform than classical alternatives.
- [ ] **VectorSAE patch-context variants** (3×3 and 5×5 overlapping activation patches): the current VectorSAE is a 1×1 per-location encoder with no local context, which handicaps it vs. FieldSAE. Test patch-context before concluding VectorSAE is intrinsically weak. (For ViTs, disjoint patch/token baselines may be more appropriate than for CNN section activations.)
- [ ] **RF-percentage budgeting**: add an RF-percentage variant (retain top `p_RF%` inside each RF window) so global vs. RF-local comparisons are apples-to-apples at matched effective sparsity. Fixed `K_RF` can imply very different global retained fractions across sections, artificially crashing Q5 performance.

### E. Interpretability & qualitative depth

- [ ] Top-activating-image montages for many features at every layer; label a sample of features; quantify monosemanticity.
- [ ] **Feature universality** across seeds (Hungarian matching of decoder atoms) — supports "these are real features."
- [ ] Show qualitative **feature-location maps** evolving across depth (low-level → semantic), à la the ViT residual-replacement paper.

### F. Writing, reproducibility, release

- [ ] Convert `report/REPORT.md` into a paper draft (abstract, intro, related work, method, experiments, limitations); target a venue (workshop first, then ICLR/NeurIPS).
- [ ] **Vector figures** (regenerate plots as true PDF/SVG), consistent style, clear captions.
- [ ] Pin dependencies (`requirements.txt` / lockfile), seed all runs, add a `make all` that reproduces every figure from scratch.
- [ ] Clean the repo for release (drop large caches or document download/regen), add a LICENSE, write a short model/SAE card.
- [ ] Ablation appendix: K (dictionary size), d (width), n_blocks, mask budget schedule, curriculum on/off, loss terms (cosine, KL).

### G. Risks / open questions to address in the paper

- [ ] Is "sparsity" genuine or an artifact of overcomplete K? (control: vary K, report L0 vs explained variance — Prisma warns vision needs high L0.)
- [ ] Does re-grounding "cheat" by re-reading the true manifold? (control: re-ground from predicted-only, quantify information actually carried by the code vs. injected by re-encoding.)
- [ ] Faithfulness of the learned tree (cross-layer transcoders can learn unfaithful shortcuts — cite the LessWrong caveat; test with interventions).

---

## 6. Suggested milestone ordering

1. **Finish Imagenette + extend taxonomy to all sections** → complete current-scale results tables.
2. **Multi-seed error bars + full 8-cell grid + stats** → defensible empirical core.
3. **Re-grounding-as-method + N8 causal + taxonomy table** → the novel contribution, paper-worthy.
4. **ResNet-50 / ImageNet-1K** → generality; the biggest single reviewer ask.
5. **ViT-B/16 + second CNN family** → full generality claim.
6. **Interpretability depth + writing + release** → submission-ready.

Milestones (1)–(3) are achievable on this box; (4)–(5) need more compute or access to a larger node.

# TODO & Project Handoff — Sparse Signal-Field SAEs

This file gives a new contributor everything needed to (a) understand the project and code, and (b) drive it to a publishable paper. For results/figures see [`report/REPORT.pdf`](report/REPORT.pdf); for setup see [`README.md`](README.md).

---

## 1. What this project is (context)

**Hypothesis.** A frozen vision model's intermediate activations look dense in their native channels but are **sparse in a learned transform domain** — like an image being dense in pixels but sparse after a wavelet transform. We test three nested claims on a frozen CNN:
1. **Layerwise sparse reconstruction** — each layer's activation field can be reconstructed from a tiny masked subset of learned feature-location coefficients, preserving the network's prediction.
2. **Cross-section prediction** — the sparse code at layer *t* predicts the sparse code at layer *t+1*.
3. **Chained hierarchy ("sparsity tree")** — chain those predictions Q1→Q5 and still preserve the classifier output.

**Key construct.** At 5 depths `Q1…Q5` we train a Sparse Autoencoder (SAE) mapping activation field `H (C×H×W)` → coefficient field `Z (K×H×W, K=8C)`, apply a **mask** that keeps only a few % of coefficients, decode `Ĥ`, and splice `Ĥ` back into the frozen net to measure accuracy/KL vs the original. Two SAE designs (per-location **VectorSAE**; convolutional **FieldSAE**) × two masks (**global** TopK; **receptive-field-local** TopK) × two training protocols (independent; incremental-dependent) = the 8-experiment grid in the design spec.

**What we found (one line each):**
- Claim 1 ✅: ~71% spliced top-1 at 2–5% retained (97.6% prediction agreement with the CNN); beats PCA + random-dict.
- Architecture: **FieldSAE ≫ VectorSAE** at aggressive/local sparsity.
- Claims 2–3 ✅ *with re-grounding*: frozen-block hybrid chain 0.712; learned chain works only when each step is re-grounded on the SAE manifold (0.012 raw → 0.701 with BPTT re-grounding; bound 0.712).
- Causality (N8): top parents matter 3–7× more than random; influence is 3–55× localized to the RF.
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

**Environment gotchas (critical):** always `source env.sh` (project-local cu12 cuDNN preload — the global env's cuDNN is mismatched). GPU index 2 currently faulted (Xid 154, needs host reboot); use GPUs 0,1,3 until then.

---

## 3. Immediate unblock (do first)

- [ ] **Reboot the node** (or have admin reset GPU2) to clear the Xid-154 fault that poisons CUDA for new processes.
- [ ] Resume the interrupted runs: `source env.sh && bash scripts/run_tier_rest_01.sh`
      → produces: R20/CIFAR-10 recon grid, multi-seed stability, transcoder taxonomy, ResNet-110 recon, ViT-on-Imagenette.
- [ ] Regenerate figures + report: `python scripts/make_plots.py && python scripts/make_plots_tiers.py && python scripts/make_sae_vs_cnn.py`, then rebuild `report/REPORT.pdf`.

---

## 4. Path to paper-ready (comprehensive next steps)

### A. Complete the empirical core
- [ ] **Multi-seed everything that's headline** (≥3 seeds): recon Pareto, chain re-grounding, N8 ratios. Report mean ± std; the paper needs error bars, not single runs.
- [ ] **Finish the 8-cell grid explicitly** (E1–E8 table) incl. the incremental-dependent protocol (Family II) for all SAE×mask combos, not just the winner — reviewers will ask for the full factorial.
- [ ] **Full sparsity–fidelity curves** with downstream KL (not just top-1) and **per-class** breakdown; add an explicit "coefficients-per-pixel" axis so budgets are comparable to PCA/JPEG-style baselines.
- [ ] **Statistical tests**: bootstrap CIs on spliced accuracy; significance of FieldSAE>VectorSAE and of re-grounding gains.

### B. Strengthen the hierarchy / causal claims (the novel contribution)
- [ ] **Architectural re-grounding as a method**, not a patch: make decode→re-encode→mask a differentiable layer, train the full chain end-to-end, report the closed gap to the 0.712 bound. (Already prototyped via `--bptt`; productionize + ablate.)
- [ ] **N8 → full causal story**: extend parent→child intervention to (i) graded ablation curves, (ii) the Tree-SAE dual criterion (coverage AND decoder-reconstruction), (iii) compare learned-transition edges vs **attribution-patching** edges (Marks et al.) — "does our learned tree recover the same circuit?".
- [ ] **Solve / characterize the Q3→Q4 bottleneck**: try Matryoshka multi-scale codes (N1), a resolution-aware predictor, or learned upsampling; if it stays hard, frame it as a finding about resolution-drop representation gaps.
- [ ] **Decision-impact vs activation vs evidence location**: add the patch-intervention test distinguishing the three (the spec's §2.6 caveat) so spatial claims are causal, not positional.

### C. Generality / scale (external validity)
- [ ] **ImageNet-scale CNN** (ResNet-50/ImageNet), not just CIFAR — the central credibility ask. (Imagenette pipeline exists in `vit_sae.py`; extend to full ResNet-50 recon.)
- [ ] **ViT transfer** (timm ViT): finish `vit_sae.py` results; handle the CLS token; **validate spatial masks with causal patch interventions** (ViT activation-location ≠ evidence-location).
- [ ] **A second CNN family** (e.g., ConvNeXt/VGG) to show the effect isn't ResNet-specific.

### D. Baselines & positioning
- [ ] Head-to-head **taxonomy** result (sparse-code transcoder vs dense transcoder vs crosscoder) — finish `taxonomy.py`, make it a clean table situating us in Anthropic's framework.
- [ ] Compare against **PatchSAE** directly (their per-token SAE) on a shared backbone — quantify the FieldSAE/spatial-mask advantage.
- [ ] Compress baselines: PCA (done), plus **JPEG/wavelet on activations** and a **pruning** baseline, to argue the SAE is a *better learned* transform.

### E. Interpretability & qualitative depth
- [ ] Top-activating-image montages for many features at every layer; label a sample of features; quantify monosemanticity.
- [ ] **Feature universality** across seeds (Hungarian matching of decoder atoms) — supports "these are real features."
- [ ] Show qualitative **feature-location maps** evolving across depth (low-level→semantic), à la the ViT residual-replacement paper.

### F. Writing, reproducibility, release
- [ ] Convert `report/REPORT.md` into a paper draft (abstract, intro, related work, method, experiments, limitations); target a venue (workshop first, then ICLR/NeurIPS).
- [ ] **Vector figures** (regenerate plots as true PDF/SVG), consistent style, clear captions.
- [ ] Pin dependencies (`requirements.txt` / lockfile), seed all runs, add a `make all` that reproduces every figure from scratch.
- [ ] Clean the repo for release (drop large caches or document download/regen), add a LICENSE, write a short model/SAE card.
- [ ] Ablation appendix: K (dictionary size), d (width), n_blocks, mask budget schedule, curriculum on/off, loss terms (cosine, KL).

### G. Risks / open questions to address in the paper
- [ ] Is "sparsity" genuine or an artifact of overcomplete K? (control: vary K, report L0 vs explained variance — Prisma warns vision needs high L0.)
- [ ] Does re-grounding "cheat" by re-reading the true manifold? (control: re-ground from predicted-only, quantify information actually carried by the code vs injected by re-encoding.)
- [ ] Faithfulness of the learned tree (cross-layer transcoders can learn unfaithful shortcuts — cite the LessWrong caveat; test with interventions).

---

## 5. Suggested milestone ordering

1. **Unblock + finish current runs** (§3) → complete results tables.
2. **Multi-seed + full 8-cell grid + stats** (§A) → defensible core.
3. **Re-grounding-as-method + N8 causal + taxonomy** (§B, §D) → the novel contribution, paper-worthy.
4. **ImageNet + ViT** (§C) → generality; biggest reviewer ask.
5. **Interpretability depth + writing + release** (§E, §F).

Each milestone ≈ one self-contained chunk; (1)–(3) are achievable on this box once GPU2 is restored, (4) needs more compute/time.

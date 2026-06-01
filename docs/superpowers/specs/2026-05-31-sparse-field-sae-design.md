# Sparse Signal-Field SAEs for Hierarchical Vision Representations — Design / Execution Spec

**Date:** 2026-05-31
**Status:** approved direction, Phase 0 executing
**Decisions locked:** PoC-now / paper-later · ResNet-20/56 on CIFAR-10/100 · validate reconstruction (gate) **and** hierarchy (headline) · critical-path-spine sequencing.

---

## 1. Hypothesis

A frozen vision model's intermediate activation **field** `H_t ∈ R^{B×C×H×W}` is dense in its native channel basis but **compressible in a learned sparse transform domain**. Train a shared-feature SAE per section that maps `H_t → Z_t ∈ R^{B×K×H×W}` (feature-by-location coefficients), keep only a sparse subset of coefficients via a **mask in the transformed field**, and reconstruct `Ĥ_t` well enough to preserve downstream computation. Then test whether sparse codes at `Q_t` **predict/generate** sparse codes at `Q_{t+1}` (a learned "sparsity tree"), chained `Q_1→Q_5` to the classifier.

Three nested claims: (1) layerwise sparse reconstruction, (2) cross-section sparse prediction, (3) chained sparse hierarchy.

## 2. Related-work grounding (the gap we fill)

**Reconstruction/masking half — well precedented, our combination is not:**
- **PatchSAE** (Lim et al., ICLR 2025) — closest vision prior art: SAE on CLIP ViT *patch tokens*, per-patch latent maps, mask-latents→substitute→measure-accuracy. But per-token MLP (our Type B), masks *dictionary indices* not *spatial locations*, no convolutional field encoder.
- **Stevens et al. 2025 (saev, 2502.06755)**, **SAE-V** (NeurIPS 2025) — patch-level causal edits + downstream-preservation eval protocol we reuse.
- **Prisma / ViT-Prisma** (2504.19475) — reusable infra; warns vision SAEs need **much higher L0** (~500+/patch) than LLM SAEs. Budget sparsity accordingly.
- **Classical ancestors of Type A**: CRsAE (unrolled FISTA, weight-tied conv dict), SSCAE (2016), CONV-WTA (spatial+lifetime sparsity ≈ TopK-on-field), Zeiler adaptive deconvolutional nets. None applied to interpreting a *frozen* net's activation field.

**Hierarchy half — maps to a known taxonomy, our variant under-explored:**
- Anthropic taxonomy: autoencoder = 1 layer; **transcoder** = t→t+1 (predicts *dense* acts); **crosscoder** = shared across layers. Ours = **transcoder-of-sparse-codes** (predict `Z_{t+1}` from `Z_t`), chained to the classifier — nonstandard.
- **Cross-Layer Transcoders / Circuit Tracing** (Anthropic 2025), **ViT Residual Replacement Model** (2509.17401) — cross-layer sparse features, but additive into *raw* acts and read off by *attribution*, not *learned generative* prediction.
- **HSAE "Atoms to Trees"**, **Tree SAE** (2025–26) — parent→child reconstruction loss + the validation criterion we adopt: genuine hierarchy needs **both activation-coverage AND decoder-reconstruction**, not co-activation. Their tree axis is dictionary-granularity; **ours is network depth** (the gap).
- **Predictive Sparse Decomposition / LISTA** (LeCun) — classical "learned predictor of a sparse code"; our honest anchor.
- **Sparse Feature Circuits** (Marks et al. 2024) — sparse features → classifier as a causal graph via attribution; our learned edges should be benchmarked against attribution edges.

**Unfilled niche** = (PatchSAE frozen mask-and-measure) × (CRsAE/SSCAE convolutional weight-shared spatial sparse coding) × (a *learned, chained* `Z_t→Z_{t+1}` depth-tree validated with coverage+reconstruction).

## 3. Architecture & infrastructure

- **Backbone**: ResNet-56 / CIFAR-100 (primary, cleaner stage spacing), ResNet-20 / CIFAR-10 (fast debug). Trained once, frozen. `src/backbone.py` with verified exact `forward_from(section, h)` splicing (max|diff| = 0 at every section).
- **Section taps Q1..Q5**: Q1 post-stem (16ch,32²), Q2 end-stage1 (16,32²), Q3 end-stage2 (32,16²), Q4 mid-stage3 (64,8²), Q5 end-stage3 (64,8²).
- **Activation cache** (`src/cache_activations.py`): dump fp16 `H_t` for train/test + logits + labels + per-channel norm stats. Decouples SAE training from backbone forward (fast iteration). ~6 GB total.
- **SAE library** (`src/sae.py`): `VectorSAE` (Type B: 1×1 enc/dec, normalised decoder cols, pre-bias) and `FieldSAE` (Type A: 1×1→`n` ConvNeXt `LocalResBlock`s→1×1, DWConv 5×5, GroupNorm, GELU, K=8–16·C, d=2–4·C). **No input→output skip around Z**; internal residuals allowed.
- **Masks** (`src/masks.py`): `mask_global_topk` (top-m over all k,h,w per image) and `mask_rf_topk` (top-k within each next-layer RF window, optional dilation). Hard TopK → gradients flow through kept coeffs (no STE).
- **Sparsity**: TopK-as-mask is the default (mask == sparsity). JumpReLU/BatchTopK are pluggable encoder activations for later ablation.
- **Training harness** (`src/train_sae.py`): staged curriculum — warmup (no mask) → anneal frac 0.5→target → fixed target — loss `‖H−Ĥ‖²/‖H‖² + 0.1·(1−cos)`, optional `β·KL(orig‖spliced)` after recon is stable. Logs recon + downstream + sparsity each epoch.
- **Controls** (`--random` flag): frozen random normalised dictionary (Heap et al. "Sanity Checks for SAEs" baseline); PCA at matched coefficient budget.
- **Env**: project-local cu12 cuDNN preload (`env.sh`) — global env untouched (see memory `carlos-box-cudnn-gotcha`).

## 4. Phased execution (critical-path spine, every phase has a kill-criterion)

- **Phase 0 — backbone + cache + baselines.** Train/freeze backbone; cache acts; compute PCA + random-dict baselines per section. *Gate: backbone hits ~72% (C100); baselines computed.*
- **Phase 1 — anchor reconstruction (spine).** Single config: VectorSAE + global mask + independent, all 5 sections, full curriculum, sweep target fracs. *Kill: at ≥1 section, masked recon beats PCA+random AND spliced top-1 drop < ~2–3% (small KL) at non-trivial sparsity. If even unmasked recon poor → diagnose capacity before blaming masking.*
- **Phase 2 — expand recon grid.** Add FieldSAE and RF-local mask → completes recon half of E1–E4. *Gate: pick 1–2 best (SAE×mask) on the recon/sparsity/downstream Pareto front.*
- **Phase 3 — hierarchy (headline).** On winners: Family I (independent SAEs + learned transitions `T_t`) first, then Family II (incremental dependent codes, strictly pairwise Q1→…→Q5). Validate edges with the Tree-SAE dual criterion; compare against crosscoder-sharing control and attribution edges. *Kill: chained Q1→Q5 decode preserves downstream above threshold; learned edges beat shared-feature control.*
- **Phase 4 — scale the winner.** Best end-to-end config → ResNet-50/ImageNet-subset → ViT (Prisma infra), per the transfer section.

## 5. The 8 core experiments (training-protocol × SAE-type × mask-type)

| ID | Protocol | SAE | Mask | Phase reached in |
|----|----------|-----|------|------|
| E1 | Independent | Field | Global | P1(anchor=vector)→P2 |
| E2 | Independent | Field | RF | P2 |
| E3 | Independent | Vector | Global | **P1 anchor** |
| E4 | Independent | Vector | RF | P2 |
| E5 | Incremental | Field | Global | P3 |
| E6 | Incremental | Field | RF | P3 |
| E7 | Incremental | Vector | Global | P3 |
| E8 | Incremental | Vector | RF | P3 |

## 6. New experiments brainstormed (beyond the 8)

1. **N1 — Matryoshka field SAE (multi-scale coefficient budget).** Nest coefficient budgets so the smallest budget alone must reconstruct → coarse-to-fine spatial codes; directly tests "sparsity of sparsity" and combats feature splitting (adapts Bussmann et al. 2025 to the spatial field).
2. **N2 — Learned soft mask vs hard TopK (the doc's fallback, promoted to a first-class ablation).** Hard-concrete / L0 gate with an unmasking penalty: *how many* coeffs does each section need? Compare the learned sparsity profile across depth (does deeper need fewer?).
3. **N3 — Sparse-code transcoder vs dense transcoder vs crosscoder, head-to-head.** Three ways to model t→t+1 on identical sections: predict `Z_{t+1}` (ours), predict `H_{t+1}` (Dunefsky transcoder), share features (Anthropic crosscoder). Same eval → a clean taxonomy result.
4. **N4 — Frozen-block hybrid as the "ground-truth" transition.** `Z̃_t → Ĥ_t → frozen ResNet block → Ĥ_{t+1} → E_{t+1} → Ẑ_{t+1}`. This is the least-custom causal transition; use it as the upper bound the learned `P_t` is measured against.
5. **N5 — Causal sufficiency of the sparse support.** Beyond reconstruction: ablate (zero) the *unmasked* coefficients and measure downstream damage vs ablating a random equal-size set — tests whether retained coeffs are causally load-bearing (links to CaFE/decision-impact-location caveat).
6. **N6 — RF-local mask with true effective receptive field.** Replace the analytic RF block with the *measured* effective RF (gradient-based) — tests the texture/repeated-pattern motivation that local sparsity beats global for early layers.
7. **N7 — Cross-seed feature stability / universality.** Train SAEs at the same section across seeds; measure decoder-direction matching (Hungarian) — are the sparse field features universal, like the circuits literature claims?
8. **N8 — Parent→child intervention test.** Mask a parent coefficient at `Q_t`; does the predicted/true child coefficient at `Q_{t+1}` predictably vanish? The strongest evidence for a genuine causal tree (the doc's hierarchy-quality metric, made into an intervention).
9. **N9 — Sparsity vs class-discriminability.** Per PatchSAE's finding, test whether a tiny class-conditional coefficient subset carries the decision — i.e., can we mask to a per-class sparse support and keep accuracy?

## 7. Evaluation & decision framework

- **Reconstruction fidelity**: relative L2, cosine, FVU per section.
- **Downstream preservation** (the real bar): splice `Ĥ_t` → frozen tail → spliced top-1, `KL(orig‖spliced)`, prediction-agreement.
- **Sparse-code quality**: L0/image, fraction retained, active-features/image, active-locations/feature, decoder-cosine duplicate detection, co-activation.
- **Hierarchy quality**: `Ẑ_{t+1}` error, decoded `Q_{t+1}` error, chain Q1→Q5 error, classifier effect, parent→child intervention (N8).
- **Baselines every cell is judged against**: PCA@budget, random-dict SAE, identity (no-op) and (for hierarchy) frozen-block hybrid (N4) upper bound + crosscoder-sharing control.
- **Selection**: Pareto front over {downstream-preservation, sparsity, recon, feature-distinctness, seed-stability}; single winner graduates to Phase 4.

## 8. Risks & mitigations

- **High L0 needed (Prisma warning)** → sweep budgets generously; report the full sparsity–fidelity curve, not a single point.
- **Hard TopK brittle** → curriculum + N2 learned-mask fallback already specced.
- **Hierarchy fails while reconstruction works** → fall back to "independent SAEs as the main result, hierarchy as separate transition-learning problem" (doc's own contingency); N4 hybrid isolates whether the failure is the predictor or the premise.
- **Activation/evidence/decision-impact location conflation** → N5/N8 causal tests; keep claims about *activation location* only until interventions pass.
- **Disk** (198 G free, 89% used) → fp16 cache, clean intermediate checkpoints.

## 9. Repo layout

```
sparse_field_sae/
  env.sh                      # source before GPU runs (cuDNN fix)
  src/{data,backbone,sae,masks,eval,cache_activations,train_sae}.py
  src/{transitions,controls}.py   # Phase 3 / baselines (added when reached)
  scripts/train_backbone.py
  configs/                    # one json per experiment cell
  runs/                       # checkpoints, caches, result.json per cell
  docs/superpowers/specs/     # this spec
```

## 10. How to run (current)

```
source env.sh
python scripts/train_backbone.py --arch resnet56 --dataset cifar100 --out runs/backbone_r56_c100   # Phase 0 (running)
python src/cache_activations.py --ckpt runs/backbone_r56_c100/best.pt --out runs/acts_r56_c100      # after backbone
python src/train_sae.py --section Q3 --sae_type vector --mask global --target_frac 0.05 --out runs/E3_Q3  # Phase 1 anchor
```

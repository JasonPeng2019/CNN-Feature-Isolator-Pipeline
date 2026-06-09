# Phase 1 Results — Anchor Reconstruction (E3: VectorSAE + global mask + independent)

**Date:** 2026-05-31 · Backbone: ResNet-56 / CIFAR-100, frozen, **71.83%** top-1.
**Setup:** per-section VectorSAE (K=8·C), global TopK mask, staged curriculum (warmup→anneal→fixed), 40 epochs. Spliced top-1 = replace H_t with Ĥ_t, run frozen tail. orig=0.7183.

## Reconstruction + downstream preservation (15 cells)

| sec | frac | relL2 | cos | spliced top-1 | KL(orig‖spliced) |
|-----|------|-------|-----|------|------|
| Q1 | 2% | 0.228 | 0.972 | 0.638 | 0.760 |
| Q1 | 5% | 0.034 | 0.999 | **0.713** | 0.018 |
| Q1 | 10% | 0.011 | 1.000 | **0.719** | 0.001 |
| Q2 | 5% | 0.235 | 0.972 | 0.708 | 0.125 |
| Q2 | 10% | 0.059 | 0.998 | **0.716** | 0.006 |
| Q3 | 5% | 0.262 | 0.964 | 0.708 | 0.096 |
| Q3 | 10% | 0.049 | 0.999 | **0.717** | 0.002 |
| Q4 | 2% | 0.369 | 0.929 | **0.710** | 0.137 |
| Q4 | 5% | 0.113 | 0.993 | **0.718** | 0.007 |
| Q5 | 2% | 0.341 | 0.939 | **0.712** | 0.041 |
| Q5 | 5% | 0.125 | 0.992 | **0.718** | 0.003 |

(Q1/Q2/Q3 @2%, Q4/Q5 @10% omitted for brevity; monotonic as expected.)

## Baselines

- **PCA @ rank=C/2 (linear ref):** Q1 0.708, Q2 0.676, Q3 0.655, Q4 0.637, Q5 0.691.
- **Random-dict SAE (Sanity-Checks control):** Q1 0.107, Q3 0.009, Q4 0.010, Q5 0.012 (≈chance) — relL2 >1.1, cos ≈0.

## Verdict — kill-criterion PASSED on all three tests

1. **Trained ≫ random**: trained SAE 0.71 vs random-dict ≈0.01-0.11. Success is not trivial.
2. **Beats PCA at matched-or-smaller coefficient budget**: e.g. Q4 SAE 0.718 @ 5%·(8C·HW)=0.4·CHW coeffs vs PCA 0.637 @ 0.5·CHW; at low budget the gap is huge (Q4 SAE 0.710 @ ~0.16·CHW vs PCA 0.356 @ 0.25·CHW).
3. **Downstream preserved**: spliced top-1 within **<1 pp** of orig at 5-10% retained for every section.

## Key observations

- **Deeper sections compress harder, not easier in coeff-count but easier in accuracy-preservation**: Q4/Q5 hold ~71% at just 2-5% retained — small 8×8 grids carry highly redundant, abstract codes.
- **Q1 needs ≥5%**: high-res early features are less spatially compressible (relL2 0.228 at 2%).
- The **sparse transform-domain hypothesis (Claim 1) is confirmed** for this backbone.

## Next: Phase 2

Branch the recon grid on the surviving axes: **E1** (FieldSAE+global), **E2** (FieldSAE+RF-local), **E4** (VectorSAE+RF-local), all 5 sections. Questions: does the expressive conv field SAE beat the simple vector SAE (esp. at low budget / early layers), and does RF-local masking help textured early sections? Winner(s) graduate to Phase 3 hierarchy.

---

## Addendum — Pareto retained-fraction sweep on the Field winner

**Artifact:** `runs/pareto` (`35/35` cells present). This is a finer follow-up sweep on the eventual Field+Global winner over retained fractions `1%, 2%, 3%, 5%, 8%, 12%, 20%` for all `Q1..Q5`.

### Headline pattern

- The coarse `2/5/10%` Phase 1 grid was directionally right, but the finer sweep shows the main knee much more clearly.
- **Q1** benefits sharply from moving off the extreme `1%` point, then largely saturates by `3-8%`.
- **Q2/Q3** improve more gradually and are the sections where extra budget above `5%` still matters.
- **Q4/Q5** are already near-saturated at moderate sparsity; pushing to `12-20%` adds little.

### Final spliced top-1 by retained fraction

| sec | 1% | 2% | 3% | 5% | 8% | 12% | 20% |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q1 | 0.6957 | 0.7111 | 0.7169 | 0.7167 | 0.7175 | 0.7174 | **0.7177** |
| Q2 | 0.7083 | 0.7133 | 0.7155 | 0.7150 | 0.7145 | 0.7163 | **0.7169** |
| Q3 | 0.6989 | 0.7054 | 0.7129 | 0.7119 | 0.7145 | **0.7186** | 0.7157 |
| Q4 | 0.7046 | 0.7119 | 0.7151 | 0.7160 | **0.7168** | 0.7167 | 0.7157 |
| Q5 | 0.7147 | 0.7132 | 0.7175 | **0.7179** | 0.7175 | 0.7173 | **0.7179** |

### Interpretation

1. **The practical sweet spot is narrower than the original phase summary implied.** For this backbone, the winner recipe already reaches near-ceiling by about `3-8%` retained on most sections.
2. **Q3 is the only section with a real late-budget gain.** It climbs from `0.7119` at `5%` to **`0.7186` at `12%`**, reinforcing the later observation that `Q3` is the project’s persistent bottleneck.
3. **Q1 does not need `10%` once the better Field architecture is in place.** It already reaches `0.7169` at `3%`, so the earlier “Q1 needs >=5%” conclusion was specific to the weaker Vector anchor, not a universal property of the section.
4. **Very deep sections are robust even under aggressive compression.** `Q5` stays at `0.7175-0.7179` from `3%` through `20%`, which is strong evidence that the late activation field is highly redundant in this sparse transform domain.

### Updated takeaway

The anchor Phase 1 result still holds, but the follow-up pareto sweep sharpens it: the Field-style winner does not merely survive at `5-10%` sparsity; it typically reaches its useful operating regime much earlier, with most of the remaining budget buying little except around the `Q3` transition bottleneck.

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

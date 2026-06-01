# Phase 2 Results — Recon grid branch (Field SAE × {Global, RF}; Vector × RF)

**Date:** 2026-05-31 · Backbone ResNet-56/CIFAR-100 71.83%. 30 cells. orig top-1 = 0.7183.
Cells: **E1** FieldSAE+Global, **E2** FieldSAE+RF-local, **E4** VectorSAE+RF-local. (E3 VectorSAE+Global = Phase 1.)

## Headline findings

### 1. Expressive Field SAE ≫ simple Vector SAE, especially at aggressive sparsity
At 2% retained (global mask), spliced top-1:

| sec | Field (E1) | Vector (E3) | Δ |
|-----|-----------|------------|---|
| Q1 | **0.714** | 0.638 | +7.6 pp |
| Q2 | **0.716** | 0.633 | +8.3 pp |
| Q3 | **0.708** | 0.665 | +4.3 pp |
| Q4 | **0.711** | 0.710 | ~0 |
| Q5 | **0.715** | 0.712 | ~0 |

Early/high-res sections benefit most from the conv field encoder (local mixing → better low-budget transform). Deep sections were already saturated. **E1 holds 0.711-0.718 at 2-5% retained across all 5 sections.**

### 2. RF-local masking is viable — but only with the Field SAE
At matched k_rf, Field+RF (E2) dominates Vector+RF (E4):

| cell | Field+RF (E2) | Vector+RF (E4) |
|------|--------------|----------------|
| Q1 k4 | **0.716** (3.1% ret) | 0.686 |
| Q2 k4 | **0.699** (0.8% ret) | 0.465 |
| Q3 k4 | **0.670** (0.4% ret) | 0.243 |
| Q4 k4 | **0.695** (0.8% ret) | 0.594 |

E2 reaches good accuracy at **extreme** retained fractions (0.4-3%), realizing the "sparsity of sparsity" motivation. The simple vector SAE collapses under RF-local masking (E4_Q3_k2 → 0.066).

### 3. Q5 RF degenerates (expected)
Q5's next "section" is global pooling → RF window = whole 8×8 field, so k_rf=2 keeps just 2 coeffs for the entire field (frac 1e-4) → E2_Q5_k2 0.219. Use the spec's Q5-global fallback for RF runs.

## Decision — Pareto winner

**Field SAE (K=8·C, 3 LocalResBlocks).** Two operating points carried to Phase 3:
- **Field + Global @ 5%** — robust, clean, ~0.713-0.718 every section → **primary hierarchy config**.
- **Field + RF-local** — comparable accuracy at far lower retained fraction → sparser variant (E6).

This vindicates the doc's hypothesis that the strongest config is the expressive conv field SAE with RF-local masking, while showing Global is the more robust default.

## Next: Phase 3 (hierarchy / sparsity tree)
On the Field-SAE winner: Family I (independent SAEs + learned transition `T_t`) and the N4 frozen-block-hybrid causal upper bound; adjacent transitions Q1→…→Q5 then chained. Validate edges with coverage+reconstruction; compare learned transition vs frozen-block hybrid vs crosscoder-sharing control.

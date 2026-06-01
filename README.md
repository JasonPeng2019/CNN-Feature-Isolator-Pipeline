# HIGH_DIM_PROJ — Sparse Signal-Field SAEs for Hierarchical Vision Representations

Testing whether a frozen vision model's internal activations are **secretly sparse** — compressible into a few "feature-at-a-location" coefficients — and whether sparse features at one layer **build** the next layer's (a learned "sparsity tree").

**Backbone:** ResNet-56 / CIFAR-100 (71.83% top-1), frozen. Also ResNet-20/CIFAR-10 (92.3%) and ResNet-110/CIFAR-100 (73.25%).

📄 **Full report:** [`report/REPORT.pdf`](report/REPORT.pdf) (14 pp, 13 figures) · source [`report/REPORT.md`](report/REPORT.md)
📊 **Figures:** [`report/plots/`](report/plots/) (PNG), [`report/figures_pdf/`](report/figures_pdf/) (per-figure PDF), [`report/all_figures.pdf`](report/all_figures.pdf) (combined)

---

## TL;DR of findings

1. **Activations are highly compressible.** Keeping just **2–5% of coefficients** rebuilds any layer so well that accuracy drops **<1 point** — beating PCA and a random dictionary (≈chance). At Q5, the SAE-spliced model agrees with the original CNN on **97.6%** of predictions.
2. **A convolutional "field" SAE wins** (+5–8 pts over a per-location SAE at aggressive sparsity; the only design that survives strict local masking).
3. **The hierarchy is causal.** Ablating top sparse coefficients hurts **3–7×** more than random ones, and their influence is **3–55×** concentrated in the right receptive field.
4. **Chaining needs "re-grounding."** A naive learned chain collapses (compounding error); training *through* a re-grounding step recovers **0.70** vs the **0.712** causal upper bound. Re-grounding is the key mechanism.
5. **Features are clean** (0–6% duplicates, 1–3% dead, distinct atoms). **Negative result:** the hard Q3→Q4 transition is a representation gap, *not* fixed by a bigger predictor.

---

## Repository layout

```
HIGH_DIM_PROJ/
├── env.sh                      # source before any GPU run (project-local cuDNN fix)
├── src/
│   ├── backbone.py             # CIFAR ResNet-20/56/110 + exact tap/splice helpers
│   ├── data.py                 # CIFAR-10/100 loaders
│   ├── sae.py                  # FieldSAE (Type A) + VectorSAE (Type B)
│   ├── masks.py                # global TopK + receptive-field-local TopK
│   ├── cache_activations.py    # dump frozen activations + norm stats
│   ├── train_sae.py            # layerwise SAE trainer (curriculum, controls)
│   ├── controls.py             # PCA baseline
│   ├── eval.py                 # recon / downstream-preservation / sparsity metrics
│   ├── transitions.py          # transition predictor + N4 frozen-block hybrid
│   ├── train_transition.py     # Phase-3 Family-I transition trainer
│   ├── train_chain.py          # Phase-3b chain trainer (scheduled-sampling + BPTT re-grounding)
│   ├── eval_chain.py           # chained Q1→Q5 eval (raw / re-mask / re-ground / hybrid)
│   ├── intervene.py            # N8 parent→child causal intervention
│   ├── feature_audit.py        # dead/duplicate/atom analysis + top-activating images
│   ├── taxonomy.py             # sparse-code vs dense transcoder vs crosscoder
│   └── vit_sae.py              # ViT (timm) transfer on Imagenette
├── scripts/
│   ├── train_backbone.py       # train + freeze a backbone
│   ├── run_grid.py             # multi-GPU experiment grids (phase1/phase2/pareto/seeds/r20c10)
│   ├── run_phase3.sh           # Phase-3 transitions
│   ├── run_tier_rest_01.sh     # remaining tier experiments (GPUs 0,1)
│   ├── make_plots.py           # core report figures (1–6)
│   ├── make_plots_tiers.py     # tier figures (7–9)
│   └── make_sae_vs_cnn.py      # SAE-vs-CNN + sparsity figures (10–13, CPU-only)
├── runs/                       # backbones, activation caches, per-run result.json, checkpoints
├── logs/                       # collated logs + result files
├── docs/
│   ├── superpowers/specs/2026-05-31-sparse-field-sae-design.md   # design spec
│   └── PHASE{1,2,3,3b}_RESULTS.md                                 # per-phase notes
└── report/                     # REPORT.md / .pdf, plots/, figures_pdf/, all_figures.pdf
```

---

## Method in one paragraph

At 5 depths `Q1…Q5` of the frozen CNN we train a small SAE that maps the activation field `H (C×H×W)` to an overcomplete coefficient field `Z (K×H×W, K=8C)`. A **mask** keeps only the top few % of coefficients (global, or receptive-field-local); the decoder reconstructs `Ĥ`. We splice `Ĥ` back into the frozen network and measure accuracy/KL vs the original. For the hierarchy, a predictor maps the masked code at one layer to the next, chained to the classifier; the **frozen-block hybrid** (decode → real frozen block → re-encode) is the causal upper bound.

---

## Reproduce

```bash
source env.sh                                                   # REQUIRED: cuDNN fix (see Environment)
python scripts/train_backbone.py --arch resnet56 --dataset cifar100 --out runs/backbone_r56_c100
python src/cache_activations.py  --ckpt runs/backbone_r56_c100/best.pt --out runs/acts_r56_c100
python src/controls.py                                          # PCA baseline
python scripts/run_grid.py --phase phase1                       # Phase 1: anchor (Vector+global)
python scripts/run_grid.py --phase phase2                       # Phase 2: Field SAE, RF-local mask
bash   scripts/run_phase3.sh                                    # Phase 3: transitions
python src/eval_chain.py                                        # chain variants
python src/train_chain.py --bptt                                # Phase 3b: architectural re-grounding
python src/intervene.py                                         # N8 causal intervention
python src/feature_audit.py --section Q5 --sae runs/phase2/E1_Q5_f0.05/sae.pt
python scripts/make_plots.py && python scripts/make_plots_tiers.py && python scripts/make_sae_vs_cnn.py
```

Regenerate the report PDF:
```bash
cd report && pandoc REPORT.md -o REPORT.pdf --pdf-engine=xelatex --toc -V geometry:margin=0.9in
```

---

## Status & how to resume

**Done:** core PoC (Phases 0–3b), Tier 1 (N8, re-grounding, Q3→Q4), Tier 2 (Pareto sweep, feature audit), ResNet-110 backbone.

**Pending a node reboot:** R20/CIFAR-10 recon + multi-seed stability, transcoder taxonomy, ResNet-110 recon, ViT-on-Imagenette. These were interrupted by a GPU hardware fault (see Environment). Resume with:
```bash
source env.sh && bash scripts/run_tier_rest_01.sh        # runs remaining items on GPUs 0,1
```

---

## Environment notes (this box)

- **cuDNN fix (required):** the global env has a mismatched CUDA-13 cuDNN that breaks the cu128 torch wheel (`CUDNN_STATUS_NOT_INITIALIZED`). `env.sh` preloads a project-local cu12 cuDNN (`.cudnn12/`) — **always `source env.sh` before any GPU run.** The global environment is left untouched.
- **GPU fault:** during the tier runs, GPU index 2 hit Xid 154 ("Node Reboot Required"), which poisons CUDA init for all *new* processes node-wide (`torch.cuda.is_available()` becomes False, even with `CUDA_VISIBLE_DEVICES`). A host reboot clears it; until then no new GPU jobs can start.
- Hardware: 4× Quadro RTX 6000 (24 GB), torch 2.10+cu128.
```

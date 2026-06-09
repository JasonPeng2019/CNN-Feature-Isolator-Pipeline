# HIGH_DIM_PROJ

Sparse autoencoder experiments over frozen vision backbones, aimed at testing whether intermediate activations are compressible into a sparse spatial code and whether those sparse codes can support layer-to-layer transitions.

## Current status

The current source of truth is the playbook-style analysis package under [`analysis/`](analysis/). It reconciles guide docs, code, and surviving run artifacts, then compiles a fresh report to [`analysis/report.pdf`](analysis/report.pdf).

Use this precedence when reading the repo:

1. [`analysis/report.pdf`](analysis/report.pdf) and the supporting manifest files in [`analysis/`](analysis/)
2. phase writeups in [`docs/PHASE1_RESULTS.md`](docs/PHASE1_RESULTS.md) through [`docs/PHASE4_RESULTS.md`](docs/PHASE4_RESULTS.md)
3. archived manuscript material in [`report/`](report/) when you need historical context

## What the analysis currently says

Artifact-backed positives:

- The `FieldSAE` winner remains highly compressive across the main CIFAR families.
- The Phase 1 pareto sweep preserves the same basic story and keeps `Q3` as the persistent bottleneck.
- In the chain experiments, full-BPTT re-grounding reaches `0.7010` against a `0.7122` hybrid upper bound, and rollout-aligned raw chaining reaches `0.6821`.
- The changed-suite follow-ups added real signal instead of noise: fair RF budgeting rescues the RF-local branch, patch overlap helps, and coarse disjoint tilings are a genuine negative result.
- The ViT transfer artifact is one of the cleanest results in the repo: at about `5%` retained it achieves `pred_agree = 0.9440` on Imagenette validation.

Important interpretation caveat:

- The playbook audit found a confirmed repeated test-exposure pattern across the core CIFAR SAE, transition, and chain trainers. Those scripts evaluate on the test split during training and development, but in this repo that split is functioning as an analysis corpus for probing how the frozen CNN executes.
- The consequence is about interpretation, not about whether the mechanistic claim is real: the CIFAR top-1 numbers should be read as corpus-conditioned faithfulness measurements, not as untouched external evaluation numbers.
- The cleanest families in the current repo state are the PCA baseline and the ViT transfer path. Some post-hoc analyses are marked `suspected` rather than `clean` because they inherit upstream SAE validity risk.

For the exact audit trail, see:

- [`analysis/leak_audit.json`](analysis/leak_audit.json)
- [`analysis/deviations.json`](analysis/deviations.json)
- [`analysis/outcomes.json`](analysis/outcomes.json)
- [`analysis/verification.md`](analysis/verification.md)

## Repository map

```text
HIGH_DIM_PROJ/
├── analysis/                  # current synthesized analysis package and PDF
├── docs/                      # per-phase result writeups and design material
├── report/                    # archived manuscript-era report assets
├── runs/                      # checkpoints, cached activations, per-run result files
├── logs/                      # launcher logs and collated JSON outputs
├── scripts/                   # experiment launchers and older plotting helpers
├── src/                       # model, training, eval, intervention, and audit code
├── data/                      # local dataset tarballs
└── env.sh                     # required environment setup for GPU runs
```

Important codepaths:

- [`src/train_sae.py`](src/train_sae.py), [`src/train_sae_changed.py`](src/train_sae_changed.py): layerwise SAE training
- [`src/train_transition.py`](src/train_transition.py): adjacent sparse-code transitions
- [`src/train_chain.py`](src/train_chain.py), [`src/train_chain_changed.py`](src/train_chain_changed.py): chain-aware training
- [`src/intervene.py`](src/intervene.py), [`src/feature_audit.py`](src/feature_audit.py), [`src/taxonomy.py`](src/taxonomy.py): follow-up analyses
- [`src/vit_sae.py`](src/vit_sae.py): ViT transfer artifact
- [`analysis/scripts/build_playbook_analysis.py`](analysis/scripts/build_playbook_analysis.py): rebuild the analysis package

## Rebuild the analysis package

If you want the current report rather than the archived manuscript, rebuild `analysis/` directly:

```bash
source env.sh
python analysis/scripts/build_playbook_analysis.py
```

That script regenerates:

- the experiment registry and repo inventory
- intent, deviation, leakage, and outcome manifests
- `analysis/results_tidy.csv`
- vector figures under `analysis/figures/`
- [`analysis/report.tex`](analysis/report.tex) and [`analysis/report.pdf`](analysis/report.pdf)

## Reproduce core experiment flow

The main historical run flow is still:

```bash
source env.sh
python scripts/train_backbone.py --arch resnet56 --dataset cifar100 --out runs/backbone_r56_c100
python src/cache_activations.py --ckpt runs/backbone_r56_c100/best.pt --out runs/acts_r56_c100
python src/controls.py
python scripts/run_grid.py --phase phase1
python scripts/run_grid.py --phase phase2
bash scripts/run_phase3.sh
python src/train_chain.py --bptt
python src/intervene.py
python src/feature_audit.py --section Q5 --sae runs/phase2/E1_Q5_f0.05/sae.pt
```

For later extension families, the surviving launcher entrypoint is usually:

```bash
source env.sh
bash scripts/run_tier_rest_01.sh
```

## Environment notes

- Always `source env.sh` before GPU work. The repo relies on a project-local cuDNN preload to avoid the host CUDA/cuDNN mismatch described in the historical notes.
- The analysis build itself is CPU-friendly, but compiling [`analysis/report.pdf`](analysis/report.pdf) requires a working LaTeX install with `pdflatex`.

## Reading guide

If you are new to the repo, the fastest accurate path is:

1. Read [`analysis/report.pdf`](analysis/report.pdf).
2. Check [`analysis/open_questions.md`](analysis/open_questions.md) and [`analysis/missing_runs.md`](analysis/missing_runs.md).
3. Use the phase docs in [`docs/`](docs/) only for per-family narrative detail.
4. Treat [`report/REPORT.pdf`](report/REPORT.pdf) as archived context, not the current reconciled conclusion.

# Current Progress And To Do

Last updated: 2026-06-22

This file is a handoff document for autonomous continuation of the `CNN-SAE` project inside `ViT_Proj`.

It is meant to answer two questions quickly:

1. What has already been achieved and should not be rediscovered?
2. What should an agent do next to make the project more paper-robust and more interpretable?

## Source Of Truth

Read the repo in this order:

1. `analysis/report.pdf`
2. `analysis/open_questions.md`
3. `analysis/missing_runs.md`
4. `README.md`
5. `TODO.md`
6. `docs/PHASE1_RESULTS.md` through `docs/PHASE4_RESULTS.md`

Important caveat: the current repo analysis explicitly notes repeated test-set exposure in the core CIFAR SAE/transition/chain workflows. That does not erase the mechanistic signal, but it does weaken claims phrased as untouched external generalization. Any new work should preferentially move toward a clean `train/val/test` setup.

## Current Project State

The project already has real, artifact-backed results. This is not a blank-slate repo.

Main achieved conclusions:

- Field-style sparse autoencoders are the strongest reconstruction path on the CIFAR ResNet setup.
- The reconstruction story survives across multiple CIFAR-family backbones and seeds.
- Sparse codes are not only reconstructive; they also appear to be stronger transition carriers than tested dense/cross alternatives.
- Re-grounding is a major reason chained sparse prediction works.
- The `Q3` region remains the persistent bottleneck across multiple families.
- A ViT transfer artifact already exists and is one of the cleanest results in the repo.
- A full ViT-small block/fraction/seed sweep on Imagenette has now completed cleanly and shows stable signal rather than a one-off artifact.

## What Has Been Achieved

The following are already present and should be treated as completed unless a clean rerun is specifically required:

- ResNet-56 / CIFAR-100 backbone and activation-cache workflow
- ResNet-20 / CIFAR-10 winner-style reconstruction sweep
- ResNet-110 / CIFAR-100 winner-style reconstruction sweep
- Phase 1 grid
- Phase 2 grid
- Pareto sweep
- Multi-seed winner-path sweep
- Transition experiments
- Chain experiments, including BPTT re-grounding
- Causal intervention experiment
- Feature audit on trained SAEs
- Taxonomy comparison at `Q4 -> Q5`
- Initial ViT transfer on Imagenette
- Full ViT-small Imagenette sweep over blocks, retained fractions, and seeds
- Run-ready scale-up infrastructure for larger ViT models and generic ImageFolder-style datasets

Useful surviving result families:

- `runs/phase2/`
- `runs/pareto/`
- `runs/seeds/`
- `runs/r20c10/`
- `runs/r110/`
- `runs/taxonomy/`
- `runs/vit_sae/vit_result.json`
- `runs/vit_sweep/`
- `runs/smoke_vit_scale_grid/`

Key reported Phase 4 results:

- ResNet-110 reconstruction remains strong, with `Q3` still the hardest section.
- ViT-small on Imagenette, block 6, at about 5% retained: `pred_agree = 0.9440`, `kl = 0.0427`, `rel_l2 = 0.4172`, `cos = 0.9085`.

Key newly completed ViT sweep results:

- `75/75` full ViT-small sweep jobs completed successfully under `runs/vit_sweep/`.
- The sweep covered `5` blocks × `5` retained fractions × `3` seeds.
- Mean `pred_agree` across all runs is about `0.8914`.
- The strongest stable setting found so far is `block 10`, retained fraction about `0.12`, with mean `pred_agree ≈ 0.9307`, seed std `≈ 0.0034`, and mean `kl ≈ 0.0361`.
- Another strong stable setting is `block 4`, retained fraction about `0.08`, with mean `pred_agree ≈ 0.9273` and seed std `≈ 0.0038`.

## Important Existing Code Paths

- `src/train_sae.py`: core CIFAR SAE training
- `src/train_sae_changed.py`: changed-suite SAE variants
- `src/train_transition.py`: adjacent sparse transition training
- `src/train_chain.py`: chain training and re-grounding path
- `src/intervene.py`: causal parent-child intervention
- `src/feature_audit.py`: decoder-atom audit and top-activating image montages
- `src/vit_sae.py`: ViT-small Imagenette sparse reconstruction path
- `scripts/run_vit_grid.py`: completed ViT-small sweep launcher
- `scripts/run_vit_scale_grid.py`: larger-model / larger-dataset ViT launcher
- `src/backbone.py`: current frozen CNN backbone code
- `scripts/run_grid.py`: launcher for many CIFAR-side sweeps

Important implementation notes:

- `src/vit_sae.py` already reshapes ViT patch tokens into a 2D grid and trains a FieldSAE over that grid.
- `src/vit_sae.py` now supports both the built-in Imagenette path and generic `ImageFolder` train/val directories.
- `src/vit_sae.py` now records dataset metadata in `vit_result.json`, supports full-split usage via `ntrain=-1` / `nval=-1`, and remains backward compatible with the original Imagenette flow.
- `src/feature_audit.py` already produces top-activating montages and decoder-atom statistics.
- `src/backbone.py` only supports CIFAR ResNet variants right now, not `torchvision` ResNet-50 or larger ImageNet backbones.
- `src/train_sae.py` currently evaluates on `test` activations during training, so new paper-facing experiments should not copy that pattern without introducing a clean validation split.

## Compute Status

Verified on 2026-06-21 with `nvidia-smi`:

- 8x `NVIDIA RTX 6000 Ada` GPUs are visible.
- 7 GPUs were effectively free.
- 1 GPU had an existing Python process using about 12 GB.

Interpretation:

- Large parallel sweeps are realistic on this machine.
- It is reasonable to run multiple SAE families, multiple seeds, and multiple ViT blocks in parallel.

## What The Project Needs Next

There are two immediate priorities:

1. Make the project more paper-robust.
2. Produce interpretable qualitative artifacts from learned sparse features across arbitrary images and classes.

### Priority A: Paper Robustness

These are the most important next steps.

1. Introduce a clean validation protocol.
   The CIFAR-side training flow should stop using `test` as the development loop for any new headline result.

2. Continue scaling the now-completed ViT experiment family.
   The ViT path is no longer a single artifact; it now has a completed sweep and a run-ready scale-up launcher for larger models/datasets.

3. Scale beyond toy/backbone-limited settings.
   The strongest next backbone target is ResNet-50 on ImageNet-scale data.

4. Add more uncertainty estimates and ablations.
   Multi-seed summaries, confidence intervals, and `Kmult` / width / budget ablations are needed for stronger claims.

### Priority B: Feature Interpretability

The repo already has top-activating image montages, but it does not yet produce the exact artifact needed for qualitative interpretability review:

- per-feature activation maps on chosen images
- decoder-side feature visualizations
- saved folders of feature overlays for a chosen semantic class, such as dogs

This should be treated as a first-class deliverable.

## Immediate Next Deliverables

An agent continuing this repo should aim to produce the following concrete outputs.

### 1. Clean ViT Sweep

Status: completed

Completed outputs:

- `scripts/run_vit_grid.py`
- `runs/vit_sweep/`
- `runs/vit_sweep/grid_summary.json`
- post-run checks under `scripts/check_vit_sweep_results.py`
- seed-stability summary under `scripts/analyze_vit_seed_stability.py`
- full figure/summary generation under `scripts/make_vit_sweep_figures.py`
- aggregate CSV/JSON summaries under `runs/vit_sweep/analysis/`
- figure suite under `report/plots/fig14_vit_sweep_*.png` through `fig19_vit_sweep_*.png`

Why it matters:

- The ViT result is now a real experiment family with stable settings, not just a single executed artifact.

### 2. General Feature Visualization Folder

Add a new script, likely `src/vit_feature_maps.py`, that uses a trained ViT SAE and writes a browsable artifact folder for arbitrary selected images and classes.

Dogs were only an example of the kind of semantic probe image that may be useful. The actual goal is broader:

- inspect learned sparse features on many classes
- inspect learned sparse features on hand-picked interesting images
- compare whether some features look class-specific, part-specific, texture-specific, or more distributed
- support both targeted and untargeted qualitative review

Recommended outputs per chosen feature:

- `original.png`
- `feature_heatmap.png`
- `feature_overlay.png`
- `decoder_atom.png`
- `top_activating_examples.png`
- `stats.json`

Recommended folder structure:

```text
runs/vit_feature_maps/
  class_<name>/
    img_<id>/
      feature_<k>_overlay.png
      feature_<k>_heatmap.png
      feature_<k>_decoder_atom.png
      feature_<k>_top_examples.png
      feature_<k>_stats.json
  custom/
    img_<id>/
      feature_<k>_overlay.png
      feature_<k>_heatmap.png
      feature_<k>_decoder_atom.png
      feature_<k>_top_examples.png
      feature_<k>_stats.json
```

Minimum required behavior:

- load chosen validation images from arbitrary classes, or from explicit user-provided image paths
- run it through the chosen ViT block
- encode to sparse features
- save the strongest features' spatial maps back in image space
- save decoder impulse-response style views for those same features

Why:

- This is the missing artifact behind the broader question “are the sparse features actually interpretable on real images?”
- The script should make it easy to inspect many semantic categories, not only dogs.

### 3. Seed Universality Analysis

After the ViT sweep or on the current CIFAR winner SAEs:

- compare decoder atoms across seeds
- use feature matching, such as Hungarian matching on atom cosine similarity
- report whether the same sparse features recur across seeds

Why:

- This materially strengthens the claim that learned features are real structure rather than seed-specific accidents.

### 4. ResNet-50 Backbone Support

Extend `src/backbone.py` to support a `torchvision` ResNet-50 style frozen backbone with 5 taps.

Recommended taps:

- post-stem
- end of layer1
- end of layer2
- mid or late layer3
- end of layer4

Then add activation caching and the same SAE reconstruction pipeline for this larger backbone.

Why:

- This is the single most important next CNN-side credibility jump.
- The repo itself already identifies this as the major reviewer ask.

### 5. Larger ViT Scale-Up Runs

Status: run-ready, not yet executed at full scale

Completed infrastructure:

- `src/vit_sae.py` generic dataset support
- `scripts/run_vit_scale_grid.py`
- process bundle:
  - `docs/superpowers/specs/2026-06-22-vit-scale-paper-robustness.md`
  - `docs/superpowers/builds/2026-06-22-vit-scale-paper-robustness-build.md`
  - `docs/superpowers/evals/2026-06-22-vit-scale-paper-robustness-eval.md`

What this enables:

- `vit_base_patch16_224` sweeps
- larger train/val subsets
- full available validation sets
- future ImageFolder-style dataset roots beyond Imagenette

Recommended first larger-model run:

```bash
cd /jumbo/lisp/f003x5w/ViT_Proj/CNN-SAE
source env.sh
python scripts/run_vit_scale_grid.py \
  --dataset imagefolder \
  --dataset_root ./data/imagenette2-160 \
  --gpus 0,1,2,4,5,6,7 \
  --models vit_base_patch16_224 \
  --blocks 2,4,6,8,10 \
  --fracs 0.02,0.05,0.08,0.12 \
  --seeds 0,1,2 \
  --epochs 20 \
  --ntrain -1 \
  --nval -1 \
  --train_bs 16 \
  --val_bs 16 \
  --num_workers 4 \
  --out_root runs/vit_base_imagenette_full
```

Recommended multi-model comparison run:

```bash
cd /jumbo/lisp/f003x5w/ViT_Proj/CNN-SAE
source env.sh
python scripts/run_vit_scale_grid.py \
  --dataset imagefolder \
  --dataset_root ./data/imagenette2-160 \
  --gpus 0,1,2,4,5,6,7 \
  --models vit_small_patch16_224,vit_base_patch16_224 \
  --blocks 4,10 \
  --fracs 0.05,0.08,0.12 \
  --seeds 0,1,2 \
  --epochs 20 \
  --ntrain -1 \
  --nval -1 \
  --train_bs 16 \
  --val_bs 16 \
  --num_workers 4 \
  --out_root runs/vit_model_compare_full
```

## Recommended Execution Order

If the goal is autonomous high-value progress, follow this order:

1. Add clean validation handling for new experiments.
2. Build the feature-visualization script for the completed ViT winners.
3. Add seed-universality analysis on the strongest stable ViT setting(s).
4. Launch the larger-model ViT scale-up run(s), starting with `vit_base_patch16_224`.
5. Extend the taxonomy comparison beyond `Q4 -> Q5`.
6. Add ResNet-50 support and start activation caching.
7. Only after that, move to ViT-B/16 or a second CNN family.

## Recommended GPU Usage

Given the observed machine state on 2026-06-21, a sensible parallel schedule is:

- GPUs 0-3: ViT-small sweep jobs
- GPU 4: seed-repeat jobs
- GPU 5: visualization export jobs
- GPU 6: backbone extension smoke tests or activation caching
- GPU 7: leave free or avoid if another user process is active

This is only a suggestion. Always re-check `nvidia-smi` before launching a large batch.

## Specific Risks To Address In Future Work

- Data leakage / test exposure in current CIFAR development loops
- Overclaiming generalization from CIFAR-scale backbones
- Interpreting sparse activation location as evidence location without causal checks
- Mistaking overcomplete dictionary effects for meaningful sparsity
- Overstating mechanistic faithfulness without stronger intervention comparisons

## What Not To Waste Time On

- Do not re-run already completed CIFAR sweeps unless the purpose is a clean rerun under improved evaluation hygiene.
- Do not treat the archived `report/` folder as the main source of truth.
- Do not treat the single ViT artifact as sufficient scale evidence.
- Do not stop at top-activating montages; the next qualitative step needs per-feature spatial overlays and decoder-side views.

## Definition Of Success For The Next Agent

The next agent should consider this phase successful if it leaves behind:

1. A reproducible ViT sweep over multiple blocks and sparsity levels.
2. A saved qualitative artifact folder showing sparse features on dog images.
2. A saved qualitative artifact folder showing sparse features on multiple semantic categories or chosen probe images.
3. A cleaner evaluation path than the current CIFAR training loop.
4. At least a scaffold for ResNet-50 / ImageNet-scale continuation.

If only one thing can be done next, do this:

- build the ViT sweep plus the dog feature-visualization folder
- build the ViT sweep plus a general feature-visualization folder for multiple classes / probe images

That is the shortest path to both stronger evidence and more compelling qualitative interpretability.

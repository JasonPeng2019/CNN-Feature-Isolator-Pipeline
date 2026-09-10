# Level-2 CUDA smoke

`scripts/run_sae_redo_cuda_smoke.py` is the bounded, no-tmux readiness path for
the sparse-recovery redo. It is dry-run by default and always allocates a new
output root. It prepares an isolated Q1 cache containing eight train, eight
validation, and eight fidelity examples from the selected real CNN activation
cache. The source cache and historical run directories are never modified.

The production matrix launcher (`scripts/launch_sae_redo_level2.py`) uses the
frozen five-queue topology: without an override it selects the first five
physical NVIDIA indices reported by `nvidia-smi`. Runnable queue materialization
also requires both compatible local ViT checkpoints and records their paths and
SHA-256 values in `queue_manifest.json`:

```bash
python scripts/launch_sae_redo_level2.py \
  --vit-small-weights /absolute/path/to/vit-small.pth \
  --vit-base-weights /absolute/path/to/vit-base.pth \
  --queue-root runs/sae_sparse_recovery_redo/queues/<UTC-or-unique-id>
```

Use `--gpus 0,1,2,3,4` only when those are five distinct indices present in
the `nvidia-smi` inventory. Missing or incompatible weights fail before queue
directory creation; no runnable queue is materialized in that case.

The plan contains, in order:

1. `field_old` at 8%;
2. `field_strict` at 8%;
3. `vector_context_strict` at 8%;
4. `field_recovery`, loaded from the exact `field_strict` parent;
5. `vector_context`, loaded from the exact `vector_context_strict` parent; and
6. a CUDA chain-loss smoke over real cached Q1--Q5 activations using the
   current `field_strict` and `field_old` models, including re-grounded/raw
   carrier checks and finite-gradient checks.

The five trainer commands explicitly carry `--smoke --smoke_disable_collapse`.
This smoke-only policy disables collapse retries on the eight-example fixture so
each cell reaches its validation raw-vs-EMA checkpoint selection and persists
the selected checkpoint path/state. Production runs do not pass these flags:
their normal collapse threshold and two-retry policy are unchanged. The smoke
plan, each trainer's metadata/result/checkpoint, and per-job evidence record the
smoke collapse policy.

Use a fresh output path. First inspect the immutable command/evidence plan:

```bash
source env.sh
python scripts/run_sae_redo_cuda_smoke.py \
  --backbone r20 --gpu 0 \
  --out runs/sae_sparse_recovery_redo/cuda_smoke/r20/<UTC-or-unique-id>
```

After checking the plan, execute explicitly:

```bash
source env.sh
python scripts/run_sae_redo_cuda_smoke.py \
  --backbone r20 --gpu 0 --execute \
  --out runs/sae_sparse_recovery_redo/cuda_smoke/r20/<UTC-or-unique-id>
```

Use `--backbone r110` to exercise the ResNet-110/CIFAR-100 cache. The harness
requires that the requested physical index is reported by `nvidia-smi`, then
sets `CUDA_VISIBLE_DEVICES` to that index and passes `--device cuda:0
--require_cuda` to every child. It never starts tmux and never falls back to
CPU. Each child receives a durable log and `smoke_job_evidence.json`; the root
receives `smoke_plan.json` and, after execution, `smoke_result.json`.

Every child has a 900-second deadline by default (`--job-timeout-seconds`).
This is intentionally generous headroom for the fixed tiny-Q1 curriculum and
post-selection artifacts while remaining a bounded smoke. On expiry, the
runner sends SIGTERM to the child's dedicated process group, waits 15 seconds
(`--cleanup-grace-seconds`), escalates to SIGKILL if needed, and waits again to
reap the group. The job evidence records `status: "timed_out"`, deadline,
signals, reaping, and the preserved log; the root summary is still written and
is not reported as passed.

The ViT Imagenette data is present and has a separate local-weight smoke path.
It is enabled only when a compatible local pretrained checkpoint is supplied;
the harness never downloads weights or starts a ViT job without that file. The
checkpoint must be loadable as a tensor-only PyTorch state dict (directly or
under `state_dict`, `model_state_dict`, or `model`), with keys and tensor
shapes compatible with the requested `timm` architecture. The ViT trainer
constructs the model with `pretrained=False`, loads the supplied state dict,
and records the resolved path and SHA-256 in the plan, continuation provenance,
manifest, metadata, result, and checkpoint.

Use a fresh output path for a ViT-small dry run:

```bash
source env.sh
python scripts/run_sae_redo_cuda_smoke.py \
  --vit-model vit_small_patch16_224 \
  --weights /absolute/path/to/vit-small.pth \
  --data-root data/imagenette2-160 --gpu 0 \
  --out runs/sae_sparse_recovery_redo/cuda_smoke/vit-small/<UTC-or-unique-id>
```

Use `--vit-model vit_base_patch16_224` and a matching ViT-base checkpoint for
the base smoke. Each plan uses eight train, eight selection-validation, and
eight fidelity-holdout ImageFolder examples and orders
`field_old`, `field_strict`, `vector_context_strict`, `field_recovery`, and
`vector_context`; recovery/vector jobs depend on their exact strict parent.
After inspecting `smoke_plan.json`, add `--execute` to run the bounded,
no-tmux CUDA jobs. The same NVIDIA-only selection, per-child deadline, process
group cleanup, and durable evidence described above apply to this path.

If the `--weights` path is missing, planning fails before any job is created.
An existing but unreadable or architecture-incompatible file is discovered by
the first bounded trainer child before training, and is captured as failed job
evidence. Thus the only external prerequisite for a passing ViT smoke is a
compatible local file; no network access or download is required.

The fixed trainer curriculum means this path is bounded but not instantaneous:
the five tiny Q1 cells still run the contract's fixed strict/recovery phases.
No full matrix, queue, or historical/partial output is touched.

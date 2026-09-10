# SAE redo Level 2 evidence record

## STEP 2 pre-implementation review validation (2026-08-23)

| Finding | Independent validation | Classification | Action |
|---|---|---|---|
| Threshold Top-K can retain tied values beyond `M`. | `src/masks.py` used `flat >= threshold`. | Confirmed blocking | Replace with index-scatter exact Top-M. |
| Budgets are latent-relative rather than original-node-relative. | `src/masks.py` and both trainers derive counts from `K*H*W`. | Confirmed blocking | Add shared original-node budget/accounting helpers and use them in new lanes. |
| CNN selects on the cached test split. | `src/train_sae.py` builds `Acts(..., "test", ...)` for every epoch. | Confirmed blocking | Add dedicated clean R110 lane using deterministic 45k/5k cached-train split. |
| ViT has no persisted manifest/metadata or reserved test mode. | `src/vit_sae.py` uses in-memory subsets and only result/checkpoint writes. | Confirmed blocking | Add dedicated clean ViT lane with exclusive run roots and manifests. |
| Context candidate is absent. | `src/sae.py:79-84` supports only field/vector. | Confirmed blocking | Add zero-init `ContextVectorSAE`. |
| Required exact accounting/artifact collision/queue support is absent. | Result schemas omit accounting; old launchers use `exist_ok=True`; required scripts do not exist. | Confirmed blocking | Add fresh lane entrypoints and text queue runner. |
| "strict FieldSAE linearity" conflicts with the frozen FieldSAE baseline. | Existing FieldSAE is nonlinear by established design; contract requires strict linearity only of ContextVectorSAE's decoder. | Accepted exception | Preserve FieldSAE baseline and test the candidate's decoder-only property. |

The reviewer found no existing source-activation bypass in the ordinary
VectorSAE decoder. That is confirmed non-blocking foundation, not proof for the
new candidate; the new implementation must retain the same property.

## STEP 3–4 implementation and review evidence (2026-08-23)

- Focused contract tests: `source env.sh && python -m pytest -q tests/test_sae_redo_level2.py` — **6 passed**.
- Static checks: `python -m py_compile` over all new lane/launcher/selection modules and `git diff --check` — **passed**.
- Fresh review repair cycles closed all confirmed blockers. The final smoke-gate review was **SHIP**: exact Top-M, original-node accounting, immutable manifests, zero-init ContextVectorSAE, full-matrix selection integrity, exclusive artifacts, text queues, and ViT-Large exclusion were confirmed.

## STEP 5 real-GPU tmux smoke evidence (2026-08-23)

Three one-item Bash text queues launched at `runs/sae_redo_level2/queues/20260823T085039711218Z` and exited zero:

- GPU 0 CNN R110: 08:50:40Z–08:50:51Z.
- GPU 1 ViT-Small: 08:50:40Z–08:51:01Z.
- GPU 2 ViT-Base: 08:50:40Z–08:51:02Z.

Every smoke root contains `run.log`, `split_manifest.json`, `run_metadata.json`,
result JSON, and `sae.pt`. Metadata reports CUDA available and an RTX 6000 Ada
device. The runs exercised forward/train/evaluation/splice paths and exact M
accounting. One-epoch, 32-example smoke metrics are technical-path evidence only,
not a collapse decision.

## STEP 6 matrix launch evidence (2026-08-23)

The non–ViT-Large matrix was materialized at
`runs/sae_redo_level2/queues/20260823T085150849083Z` and launched as five static
tmux queue sessions: `sae-redo-level2-g0` through `sae-redo-level2-g4`.
The queues contain 33 commands distributed 7/7/7/6/6 across physical GPUs 0–4.
Each status log recorded its first command as `START` at 08:51:51Z. The Bash
runner consumes one text-file command at a time and appends an exit status before
advancing; no agent-side queue polling is required.

## Matrix completion and decision evidence (2026-08-23)

All five static queue logs completed without failure: GPU queues 0, 1, and 2
each recorded 7 successful exits; queues 3 and 4 each recorded 6. The matrix
therefore produced exactly 33 result artifacts: 1 ResNet-110 accounting run, 8
ViT-Small runs, and 24 ViT-Base decision runs. No ViT-Large directory or command
was created.

The queued post-matrix selection command exited zero at 09:03:41Z. Its strict
artifact gate accepted all 24 ViT-Base results, selected `context_vector`, and
reported no candidate collapse at 4% or 2%. At 2% validation, the baseline mean
prediction agreement/KL were 0.8480/0.4029; the candidate means were
0.8900/0.2006, satisfying the precommitted margin and KL rule.

The same queued command ran the one reserved-test confirmation on the selected
2% seed-0 checkpoint. `test_result.json` reports prediction agreement 0.946,
KL 0.03812, relative L2 0.54838, FVU 0.43054, and exact `M=3011` with separate
support/value/total bit counts 60,220/48,176/108,396. The result is confirmation
only; it did not participate in selection.

"""Clean, manifest-backed ViT SAE lane for the SAE redo Level 2 contract."""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder

sys.path.insert(0, os.path.dirname(__file__))
import eval as E
import masks as M
import sae as S
from sae_redo_common import (attach_run_log, create_run_root, deterministic_indices,
                             normalization_artifact, original_node_budget,
                             sparse_bit_accounting, stratified_validation_split,
                             validate_dictionary_multiplier,
                             write_json_new, write_split_manifest)
from redo_provenance import (build_continuation_provenance, file_sha256,
                             configure_continuation_phase as configure_continuation_phase_impl,
                             continuation_phase_schedule,
                             load_matching_parent_checkpoint, replay_sparse_message,
                             provenance_sha256, save_sparse_message,
                             write_continuation_provenance)
from sae_training_policy import (PhaseSpec, bf16_autocast, build_recovery_schedule,
                                 optimizer_step, run_training_policy)
from sae_redo_controls import control_names, generate_control_reconstructions


def local_weight_provenance(weights, *, model_name=None):
    """Validate and hash an optional local ViT checkpoint without downloading."""
    if weights is None:
        return {"source": "timm_pretrained_default", "path": None, "sha256": None,
                "model": model_name}
    path = Path(weights).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"local ViT weights not found: {path}")
    return {"source": "local", "path": str(path), "sha256": file_sha256(path),
            "model": model_name}


def load_local_pretrained_weights(model, weights):
    """Load a tensor-only local checkpoint and reject architecture mismatches."""
    path = Path(weights).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"local ViT weights not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"unable to safely load local ViT weights: {path}") from error
    state = payload
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if isinstance(payload.get(key), dict):
                state = payload[key]
                break
    if not isinstance(state, dict) or not state or not all(torch.is_tensor(value) for value in state.values()):
        raise ValueError(f"local ViT weights do not contain a tensor state dict: {path}")
    cleaned = {}
    for key, value in state.items():
        key = str(key)
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value
    try:
        incompatible = model.load_state_dict(cleaned, strict=False)
    except (RuntimeError, TypeError) as error:
        # ``strict=False`` permits missing optional keys but PyTorch still
        # raises on tensor shape mismatches.  Surface both cases through the
        # stable checkpoint-compatibility error used by the CLI/tests.
        raise ValueError(
            f"local ViT weights are incompatible with the requested model: {error}"
        ) from error
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            f"local ViT weights are incompatible with the requested model: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    return {"path": str(path), "sha256": file_sha256(path),
            "missing_keys": [], "unexpected_keys": []}


class ViTWrap:
    def __init__(self, name, device, weights=None):
        # Keep pure SAE train-step/unit-test imports CPU-friendly.  timm is a
        # runtime dependency only when a frozen ViT backbone is constructed.
        import timm
        self.weight_provenance = local_weight_provenance(weights, model_name=name)
        self.m = timm.create_model(name, pretrained=weights is None)
        if weights is not None:
            loaded = load_local_pretrained_weights(self.m, weights)
            self.weight_provenance.update(loaded)
        self.m = self.m.to(device).eval()
        for parameter in self.m.parameters():
            parameter.requires_grad_(False)
        cfg = timm.data.resolve_data_config({}, model=self.m)
        self.tf = timm.data.create_transform(**cfg)
        self.D = self.m.embed_dim
        self.grid = int(self.m.patch_embed.num_patches ** 0.5)

    @torch.no_grad()
    def tokens_at(self, x, block):
        x = self.m.patch_embed(x)
        x = self.m._pos_embed(x)
        x = self.m.patch_drop(x)
        x = self.m.norm_pre(x)
        for index, layer in enumerate(self.m.blocks):
            x = layer(x)
            if index == block:
                return x
        raise ValueError(f"block {block} is outside this model")

    @torch.no_grad()
    def head_from(self, tokens, block, patch_tokens=None):
        prefix_count = tokens.shape[1] - self.grid * self.grid
        if patch_tokens is not None:
            tokens = torch.cat([tokens[:, :prefix_count], patch_tokens], dim=1)
        for layer in list(self.m.blocks)[block + 1:]:
            tokens = layer(tokens)
        return self.m.forward_head(self.m.norm(tokens))


def to_grid(tokens, prefix_count, grid):
    return tokens[:, prefix_count:].transpose(1, 2).reshape(tokens.shape[0], -1, grid, grid)


def to_tokens(grid):
    return grid.flatten(2).transpose(1, 2)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_collapse_policy(*, smoke=False, disable=False, threshold=1e9):
    """Resolve the explicit smoke policy while preserving production defaults."""
    if disable and not smoke:
        raise ValueError("smoke_disable_collapse requires --smoke")
    if not smoke:
        return {"mode": "production", "threshold": 1.0,
                "max_recovery_retries": 2}
    if disable:
        return {"mode": "smoke_disabled", "threshold": None,
                "max_recovery_retries": 0}
    threshold = float(threshold)
    if not torch.isfinite(torch.tensor(threshold)) or threshold <= 0:
        raise ValueError("smoke_collapse_threshold must be finite and positive")
    return {"mode": "smoke_threshold", "threshold": threshold,
            "max_recovery_retries": 0}


def collapse_is_detected(metric, policy):
    """Treat a disabled threshold as finite-only collapse detection."""
    finite = bool(torch.isfinite(torch.tensor(float(metric))))
    threshold = policy.get("threshold")
    return (not finite) or (threshold is not None and float(metric) > float(threshold))


def partition_train_indices(total, ntrain, ntest, seed):
    order = deterministic_indices(total, -1, seed)
    ntrain = min(ntrain, total) if ntrain >= 0 else total
    ntest = min(ntest, total - ntrain) if ntest >= 0 else total - ntrain
    return order[:ntrain], order[ntrain:ntrain + ntest]


def image_ids(dataset, root):
    root = Path(root).resolve()
    return sorted(str(Path(path).resolve().relative_to(root)) for path, _ in dataset.samples)


def sorted_image_records(dataset, root):
    """Return ``(stable_id, original_index, class)`` ordered by stable ID."""
    root = Path(root).resolve()
    records = [(str(Path(path).resolve().relative_to(root)), index, label)
               for index, (path, label) in enumerate(dataset.samples)]
    return sorted(records, key=lambda item: item[0])


def validation_split_indices(dataset, root, seed=20260827, selection_count=-1,
                             holdout_count=-1):
    """Build sorted, class-stratified official-validation selection/holdout IDs."""
    records = sorted_image_records(dataset, root)
    requested_total = None
    if selection_count >= 0 or holdout_count >= 0:
        requested_total = (max(0, selection_count) if selection_count >= 0 else 0) + \
                          (max(0, holdout_count) if holdout_count >= 0 else 0)
        if requested_total < 1:
            raise ValueError("validation selection and holdout cannot both be empty")
        records = records[:requested_total]
    ids = [item[0] for item in records]
    labels = [item[2] for item in records]
    selection_ids, holdout_ids = stratified_validation_split(ids, labels, seed)
    if selection_count >= 0 and len(selection_ids) != selection_count:
        # The common splitter balances classes; for tiny bounded smokes, trim
        # or refill deterministically while retaining disjointness.
        all_ids = set(ids)
        selected = set(selection_ids[:selection_count])
        if len(selected) < selection_count:
            selected.update(item for item in ids if item not in selected and item in set(holdout_ids))
        selection_ids = sorted(selected)
        holdout_ids = sorted(all_ids - selected)
    if holdout_count >= 0 and len(holdout_ids) != holdout_count:
        all_ids = set(ids)
        holdout = set(holdout_ids[:holdout_count])
        if len(holdout) < holdout_count:
            holdout.update(item for item in ids if item not in holdout and item not in set(selection_ids))
        holdout_ids = sorted(holdout)
        selection_ids = sorted(all_ids - holdout)
    index_by_id = {item[0]: item[1] for item in records}
    return ([index_by_id[item] for item in selection_ids],
            [index_by_id[item] for item in holdout_ids],
            selection_ids, holdout_ids)


def cache_train_grids(vit, loader, block, device):
    pieces, prefix_count = [], None
    with torch.no_grad():
        for images, _ in loader:
            tokens = vit.tokens_at(images.to(device), block)
            pieces.append(tokens.cpu())
            prefix_count = tokens.shape[1] - vit.grid * vit.grid
    tokens = torch.cat(pieces)
    return to_grid(tokens.to(device), prefix_count, vit.grid), prefix_count


@torch.no_grad()
def evaluate(vit, net, loader, block, prefix_count, mean, std, target_m, device):
    net.eval()
    total, aggregate = 0, {}
    for images, _ in loader:
        images = images.to(device)
        tokens = vit.tokens_at(images, block)
        grid = to_grid(tokens, prefix_count, vit.grid)
        normalized = (grid - mean) / std
        masked, keep, _ = M.encode_and_select(net, normalized, target_m)
        reconstruction = net.decode(masked, keep)
        metrics = E.recon_metrics(normalized, reconstruction)
        original_logits = vit.head_from(tokens, block)
        spliced_logits = vit.head_from(tokens, block, to_tokens(reconstruction * std + mean))
        metrics.update({
            "pred_agree": (original_logits.argmax(1) == spliced_logits.argmax(1)).float().mean().item(),
            "kl": F.kl_div(F.log_softmax(spliced_logits, 1), F.softmax(original_logits, 1), reduction="batchmean").item(),
            "realized_active_count": keep.reshape(keep.shape[0], -1).sum(1).float().mean().item(),
        })
        batch = images.shape[0]
        total += batch
        for key, value in metrics.items():
            aggregate[key] = aggregate.get(key, 0.0) + value * batch
    return {key: value / total for key, value in aggregate.items()}


@torch.no_grad()
def evaluate_fidelity_holdout_once(vit, net, loader, block, prefix_count, mean, std,
                                   target_m, device, *, family, seed=20260827):
    """Evaluate ViT validation holdout and controls in one post-selection pass."""
    net.eval()
    names = control_names(family)
    sums = {name: {} for name in names}
    seen = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        tokens = vit.tokens_at(images, block)
        grid = to_grid(tokens, prefix_count, vit.grid)
        normalized = ((grid.float() - mean.float()) / std.float()).float()
        original_logits = vit.head_from(tokens, block)
        controls = generate_control_reconstructions(
            net, normalized, target_m, family=family, seed=seed,
        )
        batch = images.shape[0]
        seen += batch
        for name in names:
            reconstruction = controls[name]["reconstruction"].float()
            spliced_logits = vit.head_from(
                tokens, block, to_tokens(reconstruction * std + mean)
            )
            metrics = {**E.recon_metrics(normalized, reconstruction),
                       "pred_agree": (original_logits.argmax(1) == spliced_logits.argmax(1)).float().mean().item(),
                       "spliced_top1": (labels == spliced_logits.argmax(1)).float().mean().item(),
                       "kl": F.kl_div(F.log_softmax(spliced_logits, 1),
                                     F.softmax(original_logits, 1), reduction="batchmean").item(),
                       "active_count_m": controls[name]["active_count_m"]}
            for key, value in metrics.items():
                sums[name][key] = sums[name].get(key, 0.0) + float(value) * batch
    if not seen:
        raise ValueError("fidelity holdout is empty")
    metrics_by_control = {
        name: {key: value / seen for key, value in values.items()}
        for name, values in sums.items()
    }
    baseline = {"split": "fidelity_holdout", "evaluated_once": True,
                "control": "selected", **metrics_by_control["selected"]}
    controls_artifact = {
        "schema": "sae-redo-level2-controls-v1", "split": "fidelity_holdout",
        "evaluated_once": True, "family": family, "target_m": int(target_m),
        "control_names": list(names), "metrics": metrics_by_control,
    }
    return baseline, controls_artifact


def _signed_values(net, normalized):
    if hasattr(net, "encode_score_value"):
        return net.encode_score_value(normalized)[1]
    return net.encode(normalized)


def _strict_decode(net, values):
    if hasattr(net, "strict_decode"):
        return net.strict_decode(values)
    if hasattr(net, "decode_strict"):
        return net.decode_strict(values)
    return net.decode(values)


def redo_train_step(net, activation, mean, std, target_m, temperature=0.1, phase=None):
    """Compute one FP32 phase-aware reconstruction/selector loss."""
    normalized = ((activation.float() - mean.float()) / std.float()).float()
    mode = getattr(phase, "mode", "masked") if phase is not None else "masked"
    if mode == "dense":
        values = _signed_values(net, normalized)
        reconstruction = _strict_decode(net, values)
        return (normalized - reconstruction.float()).pow(2).sum() / normalized.pow(2).sum().clamp_min(1e-8)
    if mode == "selector_bootstrap":
        scores, values = net.encode_score_value(normalized)
        reconstruction = _strict_decode(net, values)
        recon_loss = (normalized - reconstruction.float()).pow(2).sum() / normalized.pow(2).sum().clamp_min(1e-8)
        selector_loss = F.smooth_l1_loss(scores.float(), values.detach().abs().float())
        return recon_loss + selector_loss
    latent, keep, _ = M.encode_and_select(net, normalized, target_m, temperature=temperature)
    reconstruction = net.decode(latent, keep)
    return (normalized - reconstruction.float()).pow(2).sum() / normalized.pow(2).sum().clamp_min(1e-8)


def configure_continuation_phase(net, phase):
    """Expose the frozen phase contract at the ViT trainer boundary."""
    return configure_continuation_phase_impl(net, phase)


def continuation_training_phases(family, phase, epochs=30, learning_rate=3e-4):
    """Return the phase plan, ignoring mutable CLI settings for recovery lanes."""
    return continuation_phase_schedule(family, phase, epochs, learning_rate)


def run_continuation_phases(net, family, phase, train_epoch, *, epochs=30, learning_rate=3e-4):
    """Compatibility wrapper routed through the shared policy runner."""
    if family in {"field_recovery", "vector_context"}:
        schedule = build_recovery_schedule(family, 1)
    else:
        count = int(epochs)
        if count < 1:
            raise ValueError("continuation epochs must be positive")
        schedule = (PhaseSpec("strict", count, "masked", (1,) * count),)

    def callback(optimizer, phase_spec, phase_epoch):
        return train_epoch(optimizer, phase_spec.name, phase_epoch)

    policy = run_training_policy(
        net, callback, latent_count=max(1, int(getattr(net, "K", 1))), target_m=1,
        phase_schedule=schedule,
        phase_configurer=lambda phase_name: configure_continuation_phase(net, phase_name),
        peak_lr=1e-4 if family in {"field_recovery", "vector_context"} else float(learning_rate),
        updates_per_epoch=1,
    )
    return policy["logs"], policy["phase_state"]


@torch.no_grad()
def save_replay_artifacts(vit, net, loader, block, prefix_count, mean, std, target_m, out, family, device):
    """Persist one held-out sparse message and verify decoder-only replay."""
    images, _ = next(iter(loader))
    tokens = vit.tokens_at(images.to(device), block)
    grid = to_grid(tokens, prefix_count, vit.grid)
    normalized = (grid - mean) / std
    sparse, support, indices = M.encode_and_select(net, normalized, target_m)
    values = sparse.reshape(sparse.shape[0], -1).gather(1, indices)
    reconstruction = net.decode(sparse, support)
    path = out / "sparse_message.pt"
    save_sparse_message(
        path, indices, values, sparse.shape, reconstruction=reconstruction,
        decoder_requires_support=family == "field_recovery",
    )
    replay = replay_sparse_message(net, path)
    write_json_new(out / "replay.json", {
        "schema": "sae-redo-level2-replay-v1",
        "sparse_message_sha256": file_sha256(path),
        "max_abs_error": replay["max_abs_error"],
        "shape": list(sparse.shape),
    })
    return replay["max_abs_error"], file_sha256(path)


def fixed_mask(net, normalized, target_m):
    latent, keep, _ = M.encode_and_select(net, normalized, target_m)
    return net.decode(latent, keep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["vit_small_patch16_224", "vit_base_patch16_224"])
    parser.add_argument("--block", type=int, required=True)
    parser.add_argument("--backbone_id", default=None)
    parser.add_argument("--sae_type", required=True,
                        choices=["field_old", "field_strict", "field_recovery", "vector_context",
                                 "vector_context_strict"])
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data_seed", type=int, default=0)
    parser.add_argument("--data_root", default="data/imagenette2-160")
    parser.add_argument("--weights", default=None,
                        help="optional local pretrained ViT checkpoint; avoids network download")
    parser.add_argument("--Kmult", type=int, default=None,
                        help="dictionary multiplier; defaults to 8 for field_old and 4 otherwise")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--anneal_epochs", type=int, default=10)
    parser.add_argument("--ntrain", type=int, default=-1,
                        help="official train IDs used for fitting; -1 means all")
    parser.add_argument("--nval", type=int, default=-1,
                        help="selection-validation IDs; -1 uses half of official val")
    parser.add_argument("--ntest", type=int, default=-1,
                        help="fidelity-holdout IDs; -1 uses half of official val")
    parser.add_argument("--train_bs", type=int, default=32)
    parser.add_argument("--val_bs", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="legacy compatibility value; the shared policy fixes peak LR at 3e-4")
    parser.add_argument("--metadata_bits", type=int, default=64)
    parser.add_argument("--value_quantizer", default="symmetric_uniform_int16")
    parser.add_argument("--parent_checkpoint", default=None)
    parser.add_argument("--parent_family", default=None)
    parser.add_argument("--parent_budget", type=float, default=None)
    parser.add_argument("--phase", default="strict",
                        choices=["strict", "recovery_correction_warm", "recovery_joint",
                                 "vector_refiner_warm", "vector_joint"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--require_cuda", action="store_true",
                        help="fail instead of falling back to CPU")
    parser.add_argument("--smoke", action="store_true",
                        help="enable explicitly smoke-only collapse policy")
    parser.add_argument("--smoke_disable_collapse", action="store_true",
                        help="smoke-only: disable collapse detection/retries")
    parser.add_argument("--smoke_collapse_threshold", type=float, default=1e9,
                        help="smoke-only finite collapse threshold")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.Kmult is None:
        args.Kmult = 8 if args.sae_type == "field_old" else 4
    # Resolve this before constructing the collapse detector or shared policy;
    # smoke's disabled threshold intentionally still rejects non-finite loss,
    # while production retains the normal threshold and retry budget.
    collapse_policy = resolve_collapse_policy(
        smoke=args.smoke, disable=args.smoke_disable_collapse,
        threshold=args.smoke_collapse_threshold,
    )
    if args.backbone_id is None:
        args.backbone_id = args.model
    weights_provenance = local_weight_provenance(args.weights, model_name=args.model)
    validate_dictionary_multiplier(args.sae_type, args.Kmult, 1)
    continuation = build_continuation_provenance(
        backbone=args.backbone_id,
        field=f"block{args.block}",
        family=args.sae_type,
        budget=args.budget,
        seed=args.seed,
        parent_checkpoint=args.parent_checkpoint,
        parent_family=args.parent_family,
        parent_budget=args.parent_budget,
        phase=args.phase,
    )
    continuation["backbone_weights"] = weights_provenance
    continuation["sha256"] = provenance_sha256(continuation)
    set_seed(args.seed)
    if args.require_cuda and not (args.device.startswith("cuda") and torch.cuda.is_available()):
        raise RuntimeError("CUDA is required for this launcher cell")
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    out = create_run_root(args.out)
    attach_run_log(out)
    write_continuation_provenance(out / "continuation.json", continuation)
    data_root = Path(args.data_root)
    train_dir, val_dir = data_root / "train", data_root / "val"
    # Image enumeration is model-free, so the immutable split record precedes
    # model construction and every train-side computation.
    raw_train, raw_val = ImageFolder(train_dir), ImageFolder(val_dir)
    train_indices, _ = partition_train_indices(len(raw_train), args.ntrain, -1, args.data_seed)
    val_indices, test_indices, val_selection_ids, val_holdout_ids = validation_split_indices(
        raw_val, data_root, seed=20260827, selection_count=args.nval, holdout_count=args.ntest
    )
    train_source_ids = [item[0] for item in sorted_image_records(raw_train, data_root)]
    val_source_ids = [item[0] for item in sorted_image_records(raw_val, data_root)]
    manifest = write_split_manifest(out, "vit", {
        "dataset": "imagenette-imagefolder-v1",
        "data_root": str(data_root.resolve()),
        "backbone_weights": weights_provenance,
        "train_source_ids": train_source_ids,
        "validation_source_ids": val_source_ids,
        "validation_selection_ids": val_selection_ids,
        "validation_holdout_ids": val_holdout_ids,
    }, {"train": train_indices, "validation": val_indices, "test": test_indices},
       {"data_seed": args.data_seed, "model_seed": args.seed})
    manifest_holdout_indices = list(manifest["selections"]["test"])
    vit = ViTWrap(args.model, device, weights=args.weights)
    train_ds, val_ds = ImageFolder(train_dir, transform=vit.tf), ImageFolder(val_dir, transform=vit.tf)
    import torch.utils.data as tud
    train_loader = tud.DataLoader(tud.Subset(train_ds, train_indices), batch_size=args.train_bs, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = tud.DataLoader(tud.Subset(val_ds, val_indices), batch_size=args.val_bs, shuffle=False,
                                num_workers=args.num_workers, pin_memory=device.type == "cuda")
    # This loader is bound to the sorted, class-stratified official-validation
    # holdout indices recorded in ``split_manifest.json``; it is never passed
    # to training or validation checkpoint selection.
    fidelity_loader = tud.DataLoader(tud.Subset(val_ds, manifest_holdout_indices), batch_size=args.val_bs,
                                     shuffle=False, num_workers=args.num_workers,
                                     pin_memory=device.type == "cuda")
    grids, prefix_count = cache_train_grids(vit, train_loader, args.block, device)
    mean = grids.mean(dim=(0, 2, 3), keepdim=True)
    std = grids.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
    fit_ids = [train_source_ids[index] for index in train_indices]
    norm_record = normalization_artifact(
        mean.flatten().cpu().tolist(), std.flatten().cpu().tolist(), fit_ids,
        source={"data_root": str(data_root.resolve()), "model": args.model,
                "block": args.block, "split": "official_train"},
    )
    write_json_new(out / "normalization.json", norm_record)
    channels, height, width = grids.shape[1:]
    validate_dictionary_multiplier(args.sae_type, args.Kmult, channels)
    coefficient_count = args.Kmult * channels * height * width
    target_m = original_node_budget(args.budget, channels, height, width, coefficient_count)
    net = S.build_sae(args.sae_type, channels, args.Kmult * channels).to(device)
    if args.parent_checkpoint:
        parent = load_matching_parent_checkpoint(
            args.parent_checkpoint,
            backbone=args.backbone_id,
            field=f"block{args.block}",
            family=continuation["parent_family"],
            budget=continuation["parent_budget"],
            seed=args.seed,
        )
        incompatible = net.load_state_dict(parent["state_dict"], strict=False)
        allowed_missing = {name for name in incompatible.missing_keys
                           if name.startswith("correction_") or name.startswith("refiner_")}
        unexpected = set(incompatible.unexpected_keys)
        if unexpected or set(incompatible.missing_keys) - allowed_missing:
            raise ValueError(
                f"parent checkpoint architecture mismatch: missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
    def train_epoch(optimizer, phase, phase_epoch):
        net.train()
        phase_budget = phase.budget_for_epoch(phase_epoch)
        order = torch.randperm(grids.shape[0], device=device)
        loss_total, seen, optimizer_steps = 0.0, 0, 0
        for start in range(0, grids.shape[0], args.train_bs):
            activation = grids[order[start:start + args.train_bs]]
            with bf16_autocast(device):
                loss = redo_train_step(net, activation, mean, std, phase_budget, phase=phase)
            loss_value = optimizer_step(optimizer, net, loss)
            if hasattr(net, "_normalize_decoder"):
                net._normalize_decoder()
            loss_total += loss_value * activation.shape[0]
            seen += activation.shape[0]
            optimizer_steps += 1
        return {"loss": loss_total / max(1, seen), "optimizer_steps": optimizer_steps}

    @torch.no_grad()
    def validation_loss(current):
        was_training = current.training
        current.eval(); total_loss = 0.0; total_count = 0
        for images, _ in val_loader:
            images = images.to(device)
            tokens = vit.tokens_at(images, args.block)
            grid = to_grid(tokens, prefix_count, vit.grid)
            normalized = ((grid.float() - mean.float()) / std.float()).float()
            latent, keep, _ = M.encode_and_select(current, normalized, target_m)
            reconstruction = current.decode(latent, keep).float()
            total_loss += float((normalized - reconstruction).pow(2).sum().item())
            total_count += int(normalized.numel())
        current.train(was_training)
        return total_loss / max(1, total_count)

    def collapse_detector(current, _logs):
        metric = validation_loss(current)
        return {"collapsed": collapse_is_detected(metric, collapse_policy),
                "metric": metric}

    recovery_schedule = None
    phase_configurer = None
    if args.phase in {"recovery_correction_warm", "recovery_joint",
                      "vector_refiner_warm", "vector_joint"}:
        recovery_schedule = build_recovery_schedule(args.sae_type, target_m)
        phase_configurer = lambda phase_name: configure_continuation_phase(net, phase_name)
    policy = run_training_policy(
        net, train_epoch, latent_count=coefficient_count, target_m=target_m,
        validation_fn=validation_loss, target_epochs=args.epochs,
        peak_lr=3e-4, seed=args.seed, collapse_detector=collapse_detector,
        max_recovery_retries=collapse_policy["max_recovery_retries"],
        phase_schedule=recovery_schedule, phase_configurer=phase_configurer,
        updates_per_epoch=max(1, (grids.shape[0] + args.train_bs - 1) // args.train_bs),
    )
    epoch_log, phase_state = policy["logs"], policy["phase_state"]
    # Consume the untouched manifest holdout exactly once, after raw-vs-EMA
    # validation selection has been applied by run_training_policy.
    fidelity_result, controls_result = evaluate_fidelity_holdout_once(
        vit, net, fidelity_loader, args.block, prefix_count, mean, std, target_m,
        device, family=args.sae_type, seed=20260827,
    )
    fidelity_result["split_manifest_sha256"] = manifest["sha256"]
    controls_result["split_manifest_sha256"] = manifest["sha256"]
    write_json_new(out / "fidelity_result.json", fidelity_result)
    write_json_new(out / "controls.json", controls_result)
    metrics = evaluate(vit, net, val_loader, args.block, prefix_count, mean, std, target_m, device)
    accounting = sparse_bit_accounting(
        target_m,
        coefficient_count,
        value_quantizer=args.value_quantizer,
        metadata_bits=args.metadata_bits,
        original_node_count=channels * height * width,
    )
    replay_error, message_hash = save_replay_artifacts(
        vit, net, val_loader, args.block, prefix_count, mean, std, target_m, out,
        args.sae_type, device,
    )
    metadata = {
        "schema": "sae-redo-level2-run-v1", "lane": "vit", "args": vars(args),
        "cuda_available": torch.cuda.is_available(), "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "manifest_sha256": manifest["sha256"], "normalization_sha256": norm_record["sha256"],
        "backbone_weights": weights_provenance,
        "fit_train_ids_sha256": norm_record["fit_train_ids_sha256"],
        "continuation_sha256": continuation["sha256"],
        "sparse_message_sha256": message_hash,
        "replay_max_abs_error": replay_error,
        "original_shape": [channels, height, width],
        "coefficient_count": coefficient_count, "accounting": accounting,
        "phase_state": phase_state, "training_policy": policy["policy_state"],
        "validation_selection": policy["validation_selection"],
        "recovery_attempts": policy["attempts"],
        "collapse_policy": collapse_policy,
        "fidelity_result": fidelity_result, "controls": controls_result,
    }
    write_json_new(out / "run_metadata.json", metadata)
    result = {**metrics, **accounting, "budget": args.budget, "budget_fraction_of_original": args.budget,
              "model": args.model, "block": args.block, "sae_type": args.sae_type,
              "D": channels, "K": args.Kmult * channels, "seed": args.seed,
              "data_seed": args.data_seed, "ntrain": len(train_indices), "nval": len(val_indices),
              "ntest_reserved": len(test_indices), "manifest_sha256": manifest["sha256"],
              "backbone_weights": weights_provenance,
              "normalization_sha256": norm_record["sha256"],
              "continuation_sha256": continuation["sha256"],
              "sparse_message_sha256": message_hash,
              "replay_max_abs_error": replay_error, "epoch_log": epoch_log,
              "phase_state": phase_state, "training_policy": policy["policy_state"],
              "validation_selection": policy["validation_selection"],
              "recovery_attempts": policy["attempts"],
              "collapse_policy": collapse_policy,
              "fidelity_result": fidelity_result, "controls": controls_result}
    write_json_new(out / "result.json", result)
    torch.save({"state_dict": net.state_dict(), "config": vars(args), "mean": mean.cpu(), "std": std.cpu(),
                "normalization": norm_record, "prefix_count": prefix_count,
                "continuation_provenance": continuation,
                "backbone_weights": weights_provenance,
                "manifest_sha256": manifest["sha256"], "phase_state": phase_state,
                "training_policy": policy["policy_state"],
                "optimizer_state": policy["optimizer_state"], "ema_state": policy["ema_state"],
                "validation_selection": policy["validation_selection"],
                "recovery_attempts": policy["attempts"],
                "collapse_policy": collapse_policy,
                "fidelity_result": fidelity_result, "controls": controls_result}, out / "sae.pt")
    print("VIT-REDO DONE", json.dumps({k: result[k] for k in ("pred_agree", "kl", "rel_l2", "fvu", "active_count_m")}), flush=True)


if __name__ == "__main__":
    main()

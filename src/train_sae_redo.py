"""Clean ResNet-110 accounting lane for the SAE redo Level 2 contract."""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import backbone as B
import eval as E
import masks as M
import sae as S
from cache_activations import load_backbone, load_cache_metadata
from sae_redo_common import (attach_run_log, create_run_root, deterministic_indices,
                             normalization_artifact, original_node_budget,
                             sparse_bit_accounting, stratified_partition_indices,
                             validate_dictionary_multiplier,
                             write_json_new, write_split_manifest)
from redo_provenance import (build_continuation_provenance, file_sha256,
                             configure_continuation_phase as configure_continuation_phase_impl,
                             continuation_phase_schedule,
                             load_matching_parent_checkpoint, replay_sparse_message,
                             save_sparse_message, write_continuation_provenance)
from sae_training_policy import (PhaseSpec, bf16_autocast, build_recovery_schedule,
                                 optimizer_step, run_training_policy)
from sae_redo_controls import control_names, generate_control_reconstructions


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


def redo_train_update(net, activation, mean, std, target_m, optimizer, temperature=0.1):
    """Run one bounded optimizer update and enforce decoder atom constraints."""
    optimizer.zero_grad()
    loss = redo_train_step(net, activation, mean, std, target_m, temperature)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    optimizer.step()
    if hasattr(net, "_normalize_decoder"):
        net._normalize_decoder()
    return loss


@torch.no_grad()
def evaluate_fidelity_holdout_once(net, loader, section, mean, std, target_m,
                                   *, frozen_model, seed=20260827, family=None):
    """Evaluate baseline and controls in exactly one untouched holdout pass."""
    net.eval()
    try:
        device = next(net.parameters()).device
    except StopIteration:
        device = mean.device
    family = family or getattr(net, "sae_type", "field_strict")
    names = control_names(family)
    sums = {name: {} for name in names}
    seen = 0
    for activation, original_logits, labels in loader:
        activation = activation.to(device)
        original_logits = original_logits.to(device)
        labels = labels.to(device)
        normalized = ((activation.float() - mean.float()) / std.float()).float()
        controls = generate_control_reconstructions(
            net, normalized, target_m, family=family, seed=seed,
        )
        batch = activation.shape[0]
        seen += batch
        for name in names:
            reconstruction = controls[name]["reconstruction"].float()
            metrics = {**E.recon_metrics(normalized, reconstruction),
                       **E.downstream_metrics(
                           frozen_model, section, reconstruction * std + mean,
                           labels, original_logits,
                       ),
                       "active_count_m": controls[name]["active_count_m"]}
            for key, value in metrics.items():
                sums[name][key] = sums[name].get(key, 0.0) + float(value) * batch
    if not seen:
        raise ValueError("fidelity holdout is empty")
    metrics_by_control = {
        name: {key: value / seen for key, value in values.items()}
        for name, values in sums.items()
    }
    baseline = {"split": "official_test", "evaluated_once": True,
                "control": "selected", **metrics_by_control["selected"]}
    controls_artifact = {
        "schema": "sae-redo-level2-controls-v1", "split": "official_test",
        "evaluated_once": True, "family": family, "target_m": int(target_m),
        "control_names": list(names), "metrics": metrics_by_control,
    }
    return baseline, controls_artifact


def configure_continuation_phase(net, phase):
    """Expose the frozen phase contract at the CNN trainer boundary."""
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


def resolve_collapse_policy(*, smoke=False, disable=False, threshold=1e9):
    """Resolve explicit smoke-only collapse behavior without changing production."""
    if disable and not smoke:
        raise ValueError("smoke_disable_collapse requires --smoke")
    if not smoke:
        return {"mode": "production", "threshold": 1.0, "max_recovery_retries": 2}
    if disable:
        return {"mode": "smoke_disabled", "threshold": None, "max_recovery_retries": 0}
    threshold = float(threshold)
    if not torch.isfinite(torch.tensor(threshold)) or threshold <= 0:
        raise ValueError("smoke_collapse_threshold must be finite and positive")
    return {"mode": "smoke_threshold", "threshold": threshold, "max_recovery_retries": 0}


def collapse_is_detected(metric, policy):
    """Apply a resolved collapse policy, including smoke's None threshold."""
    finite = bool(torch.isfinite(torch.tensor(float(metric))))
    threshold = policy.get("threshold")
    return (not finite) or (threshold is not None and float(metric) > float(threshold))


@torch.no_grad()
def save_replay_artifacts(net, loader, mean, std, target_m, out, family):
    """Persist one held-out sparse message and verify decoder-only replay."""
    activation = next(iter(loader))[0]
    try:
        parameter = next(net.parameters())
        activation, mean, std = activation.to(parameter.device), mean.to(parameter.device), std.to(parameter.device)
    except StopIteration:
        pass
    normalized = (activation - mean) / std
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acts", default="runs/acts_r110_c100")
    parser.add_argument("--ckpt", default="runs/backbone_r110_c100/best.pt")
    parser.add_argument("--section", default="Q1")
    parser.add_argument("--backbone_id", default="resnet110_cifar100")
    parser.add_argument("--sae_type", choices=["field_old", "field_strict", "field_recovery", "vector_context", "vector_context_strict"],
                        default="field_old")
    parser.add_argument("--budget", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_seed", type=int, default=0)
    parser.add_argument("--Kmult", type=int, default=None,
                        help="dictionary multiplier; defaults to 8 for field_old and 4 otherwise")
    parser.add_argument("--n_blocks", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train_bs", type=int, default=256)
    parser.add_argument("--val_bs", type=int, default=256)
    parser.add_argument("--ntrain", type=int, default=45000)
    parser.add_argument("--nval", type=int, default=5000)
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
    collapse_policy = resolve_collapse_policy(
        smoke=args.smoke, disable=args.smoke_disable_collapse,
        threshold=args.smoke_collapse_threshold,
    )
    validate_dictionary_multiplier(args.sae_type, args.Kmult, 1)
    continuation = build_continuation_provenance(
        backbone=args.backbone_id,
        field=args.section,
        family=args.sae_type,
        budget=args.budget,
        seed=args.seed,
        parent_checkpoint=args.parent_checkpoint,
        parent_family=args.parent_family,
        parent_budget=args.parent_budget,
        phase=args.phase,
    )
    torch.manual_seed(args.seed)
    if args.require_cuda and not (args.device.startswith("cuda") and torch.cuda.is_available()):
        raise RuntimeError("CUDA is required for this launcher cell")
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    out = create_run_root(args.out)
    attach_run_log(out)
    write_continuation_provenance(out / "continuation.json", continuation)
    acts = os.path.abspath(args.acts)
    hidden = torch.load(os.path.join(acts, f"train_{args.section}.pt"), weights_only=True).float()
    logits = torch.load(os.path.join(acts, "train_logits.pt"), weights_only=True)
    labels = torch.load(os.path.join(acts, "train_labels.pt"), weights_only=True)
    train_indices, val_indices = stratified_partition_indices(
        labels.tolist(), args.ntrain, args.nval, args.data_seed
    )
    if not train_indices or not val_indices:
        raise ValueError("clean train/validation split is empty")
    cache_metadata = load_cache_metadata(acts)
    reserved_test_count = int(cache_metadata["test_count"])
    manifest = write_split_manifest(out, "cnn-r110", {
        "dataset": "cifar100-cached-train-v1", "acts": acts,
        "train_source_ids": [f"train:{index}" for index in range(hidden.shape[0])],
        "ordered_source_ids": [f"train:{index}" for index in range(hidden.shape[0])],
        "reserved_test_source": os.path.join(acts, f"test_{args.section}.pt"),
        "test_source_ids": list(cache_metadata["test_ids"]),
    }, {"train": train_indices, "validation": val_indices, "test": list(range(reserved_test_count))},
       {"data_seed": args.data_seed, "model_seed": args.seed})
    import torch.utils.data as tud
    train_data = tud.TensorDataset(hidden[train_indices], logits[train_indices], labels[train_indices])
    val_data = tud.TensorDataset(hidden[val_indices], logits[val_indices], labels[val_indices])
    train_loader = tud.DataLoader(train_data, batch_size=args.train_bs, shuffle=True, num_workers=2, pin_memory=device.type == "cuda")
    val_loader = tud.DataLoader(val_data, batch_size=args.val_bs, shuffle=False, num_workers=2, pin_memory=device.type == "cuda")
    channels, height, width = hidden.shape[1:]
    validate_dictionary_multiplier(args.sae_type, args.Kmult, channels)
    fit = hidden[train_indices].float()
    mean = fit.mean(dim=(0, 2, 3), keepdim=True).to(device)
    std = fit.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6).to(device)
    fit_ids = [f"train:{index}" for index in train_indices]
    norm_record = normalization_artifact(
        mean.flatten().cpu().tolist(), std.flatten().cpu().tolist(), fit_ids,
        source={"acts": acts, "section": args.section, "split": "train"},
    )
    write_json_new(out / "normalization.json", norm_record)
    coefficient_count = args.Kmult * channels * height * width
    target_m = original_node_budget(args.budget, channels, height, width, coefficient_count)
    net = S.build_sae(args.sae_type, channels, args.Kmult * channels,
                      n_blocks=args.n_blocks).to(device)
    if args.parent_checkpoint:
        parent = load_matching_parent_checkpoint(
            args.parent_checkpoint,
            backbone=args.backbone_id,
            field=args.section,
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
    model, _ = load_backbone(args.ckpt, device)
    def train_epoch(optimizer, phase, phase_epoch):
        net.train(); loss_total = 0.0; seen = 0; optimizer_steps = 0
        phase_budget = phase.budget_for_epoch(phase_epoch)
        for activation, _, _ in train_loader:
            activation = activation.to(device)
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
        for activation, _, _ in val_loader:
            activation = activation.to(device)
            normalized = ((activation.float() - mean.float()) / std.float()).float()
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
        peak_lr=3e-4, seed=args.seed,
        collapse_detector=None if collapse_policy["threshold"] is None else collapse_detector,
        max_recovery_retries=collapse_policy["max_recovery_retries"],
        phase_schedule=recovery_schedule, phase_configurer=phase_configurer,
        updates_per_epoch=max(1, len(train_loader)),
    )
    epoch_log, phase_state = policy["logs"], policy["phase_state"]
    # The official cached test tensors are opened only after validation has
    # selected the raw or EMA checkpoint.  This loader is consumed exactly
    # once by the combined fidelity/control evaluator below.
    fidelity_hidden = torch.load(os.path.join(acts, f"test_{args.section}.pt"), weights_only=True).float()
    fidelity_logits = torch.load(os.path.join(acts, "test_logits.pt"), weights_only=True)
    fidelity_labels = torch.load(os.path.join(acts, "test_labels.pt"), weights_only=True)
    fidelity_data = tud.TensorDataset(fidelity_hidden, fidelity_logits, fidelity_labels)
    fidelity_loader = tud.DataLoader(
        fidelity_data, batch_size=args.val_bs, shuffle=False, num_workers=2,
        pin_memory=device.type == "cuda",
    )
    fidelity_result, controls_result = evaluate_fidelity_holdout_once(
        net, fidelity_loader, args.section, mean, std, target_m,
        frozen_model=model, seed=20260827, family=args.sae_type,
    )
    fidelity_result["split_manifest_sha256"] = manifest["sha256"]
    controls_result["split_manifest_sha256"] = manifest["sha256"]
    write_json_new(out / "fidelity_result.json", fidelity_result)
    write_json_new(out / "controls.json", controls_result)
    net.eval(); aggregate = {}; total = 0
    with torch.no_grad():
        for activation, original_logits, labels_batch in val_loader:
            activation, original_logits, labels_batch = activation.to(device), original_logits.to(device), labels_batch.to(device)
            normalized = (activation - mean) / std
            latent, keep, _ = M.encode_and_select(net, normalized, target_m)
            reconstruction = net.decode(latent, keep)
            metrics = {**E.recon_metrics(normalized, reconstruction),
                       **E.downstream_metrics(model, args.section, reconstruction * std + mean, labels_batch, original_logits)}
            metrics["realized_active_count"] = keep.reshape(keep.shape[0], -1).sum(1).float().mean().item()
            batch = activation.shape[0]; total += batch
            for key, value in metrics.items(): aggregate[key] = aggregate.get(key, 0.0) + value * batch
    metrics = {key: value / total for key, value in aggregate.items()}
    accounting = sparse_bit_accounting(
        target_m,
        coefficient_count,
        value_quantizer=args.value_quantizer,
        metadata_bits=args.metadata_bits,
        original_node_count=channels * height * width,
    )
    replay_error, message_hash = save_replay_artifacts(
        net, val_loader, mean, std, target_m, out, args.sae_type
    )
    metadata = {"schema": "sae-redo-level2-run-v1", "lane": "cnn-r110", "args": vars(args),
                "cuda_available": torch.cuda.is_available(), "device": str(device),
                "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "manifest_sha256": manifest["sha256"], "normalization_sha256": norm_record["sha256"],
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
                "fidelity_result": fidelity_result, "controls": controls_result}
    write_json_new(out / "run_metadata.json", metadata)
    result = {**metrics, **accounting, "budget_fraction_of_original": args.budget,
              "section": args.section, "sae_type": args.sae_type, "seed": args.seed,
              "ntrain": len(train_indices), "nval": len(val_indices), "manifest_sha256": manifest["sha256"],
              "normalization_sha256": norm_record["sha256"],
              "continuation_sha256": continuation["sha256"],
              "sparse_message_sha256": message_hash,
              "replay_max_abs_error": replay_error,
              "epoch_log": epoch_log, "phase_state": phase_state,
              "training_policy": policy["policy_state"],
              "validation_selection": policy["validation_selection"],
              "recovery_attempts": policy["attempts"],
              "collapse_policy": collapse_policy,
              "fidelity_result": fidelity_result, "controls": controls_result}
    write_json_new(out / "result.json", result)
    torch.save({"state_dict": net.state_dict(), "config": vars(args), "manifest_sha256": manifest["sha256"],
                "normalization": norm_record, "continuation_provenance": continuation,
                "phase_state": phase_state, "training_policy": policy["policy_state"],
                "optimizer_state": policy["optimizer_state"], "ema_state": policy["ema_state"],
                "validation_selection": policy["validation_selection"],
                "recovery_attempts": policy["attempts"],
                "collapse_policy": collapse_policy,
                "fidelity_result": fidelity_result, "controls": controls_result,
                "mean": mean.cpu(), "std": std.cpu()}, out / "sae.pt")
    print("CNN-REDO DONE", json.dumps({k: result[k] for k in ("pred_agree", "kl_orig_spliced", "rel_l2", "fvu", "active_count_m")}), flush=True)


if __name__ == "__main__":
    main()

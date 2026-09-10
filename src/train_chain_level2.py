"""Contract-compliant CNN Level-2 sparse-code chain training.

This module is deliberately separate from the historical Phase-3 scripts.  It
trains the four adjacent Q1->Q2->Q3->Q4->Q5 predictors for one validated 1%
SAE family, with full-BPTT re-grounding as the primary carrier and a matched
raw-carrier comparator.  Destination carriers are formed only from the
predicted code and frozen destination SAE; true destination activations are
used for supervised targets, never for carrier construction.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(__file__))
import backbone as B
import masks as M
import sae as S
import transitions as T
from cache_activations import SECTIONS, load_backbone
from sae_redo_common import original_node_budget, ordered_id_hash


PAIRS = tuple(zip(SECTIONS[:-1], SECTIONS[1:]))
CHAIN_FAMILIES = ("field_old", "field_strict", "field_recovery", "vector_context")
NEW_CHAIN_FAMILIES = ("field_strict", "field_recovery", "vector_context")
CHAIN_BUDGET = 0.01
SEEDS = (0, 1, 2)


def _digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_content_addressed_json(path, label):
    """Load a JSON artifact whose canonical body carries its own SHA-256."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} manifest not found: {path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label} manifest: {path}") from error
    if not isinstance(payload, dict) or not payload.get("sha256"):
        raise ValueError(f"{label} manifest lacks sha256: {path}")
    digest = payload["sha256"]
    canonical = dict(payload)
    canonical.pop("sha256", None)
    if digest != _digest(canonical):
        raise ValueError(f"{label} manifest SHA-256 does not match: {path}")
    return payload


def _score_seed_row(row, label):
    if not isinstance(row, dict):
        raise ValueError(f"{label} seed record must be an object")
    if row.get("seed") not in SEEDS:
        raise ValueError(f"{label} seed must be one of {SEEDS}")
    if not isinstance(row.get("non_collapsed"), bool):
        raise ValueError(f"{label} must declare non_collapsed as bool")
    agreement = row.get("validation_pred_agree", row.get("val_pred_agree"))
    kl = row.get("validation_kl", row.get("val_kl"))
    if not isinstance(agreement, (int, float)) or not isinstance(kl, (int, float)):
        raise ValueError(f"{label} must include validation agreement and KL")
    if not (float("-inf") < float(agreement) < float("inf") and
            float("-inf") < float(kl) < float("inf")):
        raise ValueError(f"{label} metrics must be finite")
    return int(row["seed"]), bool(row["non_collapsed"]), float(agreement), float(kl)


def build_chain_selection_artifact(candidates, *, backbone, fields=SECTIONS,
                                   budget=CHAIN_BUDGET, selected_family=None):
    """Build the content-addressed selector evidence consumed by chain CLI.

    ``candidates`` is ``family -> field -> list[three seed records]``.  Each
    seed record must explicitly carry ``non_collapsed``, validation agreement,
    and validation KL.  A candidate is eligible only when at least two seeds
    are non-collapsed in every ordered Q1..Q5 field.
    """
    if abs(float(budget) - CHAIN_BUDGET) > 1e-9:
        raise ValueError("chain selection is frozen to the 1% budget")
    fields = tuple(fields)
    if fields != tuple(SECTIONS):
        raise ValueError(f"chain selection fields must be exactly {SECTIONS}")
    if set(candidates) != set(NEW_CHAIN_FAMILIES):
        raise ValueError(f"selection must contain exactly {NEW_CHAIN_FAMILIES}")
    evidence, ranking = {}, []
    for family in NEW_CHAIN_FAMILIES:
        per_field, eligible = {}, True
        agreement_values, kl_values = [], []
        for field in fields:
            rows = candidates[family].get(field)
            if not isinstance(rows, list) or len(rows) != 3:
                raise ValueError(f"{family}/{field} must contain exactly three seed records")
            parsed = [_score_seed_row(row, f"{family}/{field}") for row in rows]
            if {seed for seed, *_ in parsed} != set(SEEDS):
                raise ValueError(f"{family}/{field} must contain seeds {SEEDS}")
            noncollapsed = sum(item[1] for item in parsed)
            field_eligible = noncollapsed >= 2
            eligible = eligible and field_eligible
            agreement_values.extend(item[2] for item in parsed)
            kl_values.extend(item[3] for item in parsed)
            per_field[field] = {
                "seed_records": rows,
                "non_collapsed_count": noncollapsed,
                "eligible": field_eligible,
                "mean_validation_pred_agree": sum(item[2] for item in parsed) / 3.0,
                "mean_validation_kl": sum(item[3] for item in parsed) / 3.0,
            }
        mean_agreement = sum(agreement_values) / len(agreement_values)
        mean_kl = sum(kl_values) / len(kl_values)
        evidence[family] = {
            "per_field": per_field,
            "eligible": eligible,
            "mean_validation_pred_agree": mean_agreement,
            "mean_validation_kl": mean_kl,
        }
        if eligible:
            ranking.append((family, mean_agreement, mean_kl))
    ranking.sort(key=lambda item: (-item[1], item[2], item[0]))
    if not ranking:
        raise ValueError("no new chain family is eligible at 1%")
    winner = ranking[0][0]
    if selected_family is not None and selected_family != winner:
        raise ValueError(f"selected family {selected_family!r} is not the ranked winner {winner!r}")
    payload = {
        "schema": "sae-redo-level2-chain-selection-v1",
        "backbone": str(backbone), "fields": list(fields), "budget": CHAIN_BUDGET,
        "new_families": list(NEW_CHAIN_FAMILIES), "candidates": evidence,
        "ranking": [
            {"family": family, "mean_validation_pred_agree": agreement,
             "mean_validation_kl": kl}
            for family, agreement, kl in ranking
        ],
        "selected_family": winner, "paired_baseline": "field_old",
        "pair": {"selected_new_family": winner, "baseline_family": "field_old"},
    }
    payload["sha256"] = _digest(payload)
    return payload


def validate_selection_manifest(path, *, expected_backbone=None):
    """Validate frozen 1% eligibility/ranking and return the selector record."""
    payload = _read_content_addressed_json(path, "chain selection")
    if payload.get("schema") != "sae-redo-level2-chain-selection-v1":
        raise ValueError("unsupported chain selection schema")
    if expected_backbone is not None and payload.get("backbone") != str(expected_backbone):
        raise ValueError("chain selection backbone does not match the requested backbone")
    if payload.get("fields") != list(SECTIONS) or abs(float(payload.get("budget", -1)) - CHAIN_BUDGET) > 1e-9:
        raise ValueError("chain selection must cover ordered Q1..Q5 at 1%")
    if payload.get("new_families") != list(NEW_CHAIN_FAMILIES):
        raise ValueError("chain selection new-family set is not frozen")
    if payload.get("paired_baseline") != "field_old" or payload.get("pair") != {
        "selected_new_family": payload.get("selected_family"), "baseline_family": "field_old"
    }:
        raise ValueError("chain selection must explicitly pair the selected family with field_old")
    evidence = payload.get("candidates", {})
    raw_candidates = {
        family: {
            field: evidence[family]["per_field"][field]["seed_records"]
            for field in SECTIONS
        }
        for family in NEW_CHAIN_FAMILIES
    }
    rebuilt = build_chain_selection_artifact(
        raw_candidates, backbone=payload["backbone"],
        fields=payload["fields"], budget=payload["budget"],
        selected_family=payload.get("selected_family"),
    )
    # Compare all semantic fields, not merely the selected winner.  This
    # rejects edits hidden by an unchanged winner/hash substitution.
    if rebuilt != payload:
        raise ValueError("chain selection evidence/ranking is inconsistent")
    return payload


def set_seed(seed):
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def scheduled_sampling_probability(epoch, epochs):
    """Deterministic 0->1 schedule shared by raw and re-grounded runs."""
    if int(epochs) <= 1:
        return 1.0
    return float(epoch) / float(epochs - 1)


def freeze_sae(net):
    """Put a selected SAE in frozen inference mode without blocking input grads."""
    net.eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    return net


def exact_mask_values(values, target_m):
    """Apply exact original-node Top-M to a predicted value field.

    This uses the shared exact operator directly; no latent-fraction or
    threshold helper is used in the Level-2 chain.
    """
    hard, indices = M.exact_topm(values.float(), int(target_m))
    return values.float() * hard.float(), hard, indices


def reground_carrier(zhat, destination_sae, mean, std, target_m, temperature=0.1):
    """Build the in-graph destination carrier from ``zhat`` only.

    No ``no_grad`` context or detach occurs here.  Frozen SAE parameters still
    permit autograd to carry the loss gradient back through decode and encode
    to the transition predictor input.
    """
    decoded_raw = destination_sae.decode(zhat) * std + mean
    normalized = (decoded_raw - mean) / std
    return M.encode_and_select(destination_sae, normalized, int(target_m), temperature=temperature)


def carrier_from_prediction(zhat, mode, destination_sae, mean, std, target_m, temperature=0.1):
    """Return ``(carrier, support, indices)`` for one named rollout mode."""
    if mode == "raw":
        flat = zhat.float().reshape(zhat.shape[0], -1)
        support = torch.ones_like(zhat, dtype=torch.bool)
        indices = torch.arange(flat.shape[1], device=zhat.device).expand(zhat.shape[0], -1)
        return zhat.float(), support, indices
    if mode == "exact_remask_only":
        return exact_mask_values(zhat, target_m)
    if mode == "regrounded":
        return reground_carrier(zhat, destination_sae, mean, std, target_m, temperature)
    raise ValueError(f"unknown chain carrier mode: {mode}")


def _target_codes(saes, normalized):
    with torch.no_grad():
        return {section: M.encode_and_select(saes[section], normalized[section], 1)[0]
                for section in SECTIONS}


def _sample_predicted(index, probability, sampling_draws):
    if index == 0:
        return True
    if sampling_draws is not None:
        return bool(sampling_draws[index - 1])
    return bool(torch.rand((), device="cpu").item() < probability)


def chain_training_loss(
    predictors,
    saes,
    means,
    stds,
    target_ms,
    activations,
    *,
    scheduled_sampling_probability=1.0,
    carrier_mode="regrounded",
    sampling_draws=None,
    temperature=0.1,
):
    """Compute one full-rollout chain loss while retaining BPTT through carriers."""
    normalized = {section: (activations[section] - means[section]) / stds[section]
                  for section in SECTIONS}
    with torch.no_grad():
        teacher = {}
        for section in SECTIONS:
            teacher[section] = M.encode_and_select(
                saes[section], normalized[section], int(target_ms[section]), temperature=temperature
            )[0]
    carrier = teacher["Q1"]
    losses, trace = [], []
    for index, (source, destination) in enumerate(PAIRS):
        use_prediction = _sample_predicted(index, scheduled_sampling_probability, sampling_draws)
        predictor_input = carrier if use_prediction else teacher[source]
        zhat = predictors[(source, destination)](predictor_input)
        with torch.no_grad():
            z_target = teacher[destination]
        decoded = saes[destination].decode(zhat)
        code_loss = (zhat - z_target).pow(2).mean()
        reconstruction_loss = (decoded - normalized[destination]).pow(2).sum() / \
            normalized[destination].pow(2).sum().clamp_min(1e-8)
        loss = code_loss + reconstruction_loss
        losses.append(loss)
        carrier, support, indices = carrier_from_prediction(
            zhat, carrier_mode, saes[destination], means[destination], stds[destination],
            target_ms[destination], temperature,
        )
        trace.append({
            "source": source, "destination": destination,
            "used_predicted_carrier": use_prediction,
            "code_loss": code_loss, "reconstruction_loss": reconstruction_loss,
            "carrier": carrier, "support": support, "indices": indices,
        })
    return sum(losses), {"losses": losses, "trace": trace, "final_carrier": carrier}


def rollout_carrier(predictors, saes, means, stds, target_ms, source_activation,
                    *, mode="regrounded", temperature=0.1):
    """Roll out one sample batch without consulting true destination activations."""
    normalized = (source_activation - means["Q1"]) / stds["Q1"]
    with torch.no_grad():
        carrier = M.encode_and_select(saes["Q1"], normalized, int(target_ms["Q1"]), temperature)[0]
    trace = []
    for source, destination in PAIRS:
        zhat = predictors[(source, destination)](carrier)
        carrier, support, indices = carrier_from_prediction(
            zhat, mode, saes[destination], means[destination], stds[destination],
            target_ms[destination], temperature,
        )
        trace.append({"source": source, "destination": destination,
                      "zhat": zhat, "carrier": carrier,
                      "support": support, "indices": indices})
    return carrier, trace


def frozen_block_hybrid_rollout(predictors, saes, means, stds, target_ms, activations, model,
                                temperature=0.1):
    """Causal frozen-block hybrid using true frozen backbone transitions."""
    hidden = activations["Q1"]
    trace = []
    with torch.no_grad():
        for source, destination in PAIRS:
            normalized = (hidden - means[source]) / stds[source]
            sparse, support, indices = M.encode_and_select(
                saes[source], normalized, int(target_ms[source]), temperature
            )
            hidden = model.forward_between(
                source, destination,
                saes[source].decode(sparse) * stds[source] + means[source],
            )
            trace.append({"source": source, "destination": destination,
                          "support": support, "indices": indices})
    return hidden, trace


@torch.no_grad()
def evaluate_chain_modes(predictors, saes, means, stds, target_ms, batch_iterator, model=None):
    """Evaluate raw, exact re-mask-only, re-grounded, and hybrid rollouts."""
    for predictor in predictors.values():
        predictor.eval()
    totals = {"raw": 0.0, "exact_remask_only": 0.0, "regrounded": 0.0,
              "frozen_block_hybrid": 0.0}
    count = 0
    for activations in batch_iterator:
        if isinstance(activations, (tuple, list)):
            activations = activations[0]
        q5 = activations["Q5"]
        for mode in ("raw", "exact_remask_only", "regrounded"):
            final_code, _ = rollout_carrier(
                predictors, saes, means, stds, target_ms, activations["Q1"], mode=mode
            )
            reconstruction = saes["Q5"].decode(final_code)
            totals[mode] += (reconstruction - (q5 - means["Q5"]) / stds["Q5"]).pow(2).mean().item() * q5.shape[0]
        if model is not None:
            hidden, _ = frozen_block_hybrid_rollout(
                predictors, saes, means, stds, target_ms, activations, model
            )
            totals["frozen_block_hybrid"] += (hidden - q5).pow(2).mean().item() * q5.shape[0]
        count += q5.shape[0]
    return {key: value / max(1, count) for key, value in totals.items()}


def train_chain_epoch(predictors, saes, means, stds, target_ms, loader, optimizer, device,
                      *, scheduled_probability, carrier_mode, clip_grad=1.0,
                      sampling_draws=None):
    """Train one full-BPTT epoch and return loss/gradient/support provenance."""
    for predictor in predictors.values():
        predictor.train()
    total_loss, batches, gradient_trace = 0.0, 0, []
    for batch_index, (batch, _) in enumerate(loader):
        activations = {section: batch[section].to(device).float() for section in SECTIONS}
        optimizer.zero_grad()
        loss, artifact = chain_training_loss(
            predictors, saes, means, stds, target_ms, activations,
            scheduled_sampling_probability=scheduled_probability,
            carrier_mode=carrier_mode,
            sampling_draws=None if sampling_draws is None else sampling_draws[batch_index],
        )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for predictor in predictors.values() for parameter in predictor.parameters()],
            float(clip_grad),
        )
        optimizer.step()
        total_loss += float(loss.item())
        batches += 1
        gradient_trace.append({
            "batch": batch_index, "loss": float(loss.item()),
            "gradient_norm_before_clip": float(gradient_norm),
            "support_counts": [int(item["support"].reshape(item["support"].shape[0], -1).sum(1).float().mean())
                               for item in artifact["trace"]],
        })
    return {"train_loss": total_loss / max(1, batches), "gradient_trace": gradient_trace}


def _load_norm(acts, section, device):
    acts = Path(acts)
    manifest_path = acts / "norm_manifest.json"
    if manifest_path.is_file():
        manifest = _read_content_addressed_json(manifest_path, "normalization")
        values = manifest["sections"][section]
        return (torch.tensor(values["mean"], device=device).view(1, -1, 1, 1),
                torch.tensor(values["std"], device=device).view(1, -1, 1, 1),
                manifest.get("sha256"))
    stats = json.loads((acts / "norm_stats.json").read_text())[section]
    return (torch.tensor(stats["mean"], device=device).view(1, -1, 1, 1),
            torch.tensor(stats["std"], device=device).view(1, -1, 1, 1), None)


class ChainActs(torch.utils.data.Dataset):
    def __init__(self, acts, split):
        self.hidden = {section: torch.load(Path(acts) / f"{split}_{section}.pt", weights_only=True)
                       for section in SECTIONS}
        self.labels = torch.load(Path(acts) / f"{split}_labels.pt", weights_only=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {section: self.hidden[section][index].float() for section in SECTIONS}, self.labels[index]


def load_chain_datasets(acts, input_provenance):
    """Load only the split named by the validated activation-cache manifest.

    Both ``test`` and ``fidelity_holdout`` may exist in a cache directory for
    compatibility with older cache layouts.  Their presence is deliberately
    not consulted here: ``validate_activation_cache_manifest`` has already
    resolved the contract-bound fidelity split and recorded it in provenance.
    """
    try:
        fidelity_split = input_provenance["activation_cache"]["fidelity_split"]
    except (KeyError, TypeError) as error:
        raise ValueError("validated activation-cache provenance lacks fidelity_split") from error
    if fidelity_split not in {"fidelity_holdout", "test"}:
        raise ValueError(f"unsupported manifest fidelity split: {fidelity_split!r}")
    datasets = {split: ChainActs(acts, split)
                for split in ("train", "validation", fidelity_split)}
    return fidelity_split, datasets


def validate_selected_checkpoint(path, family, section):
    """Validate a selected 1% final-family checkpoint before chain use."""
    path = Path(path)
    if family not in CHAIN_FAMILIES:
        raise ValueError(f"chain family must be one of {CHAIN_FAMILIES}, got {family!r}")
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    if config.get("sae_type") != family:
        raise ValueError(f"{path} is not a {family} checkpoint")
    declared_section = config.get("section", config.get("field"))
    if declared_section is not None and str(declared_section) != str(section):
        raise ValueError(f"{path} is not the selected checkpoint for {section}")
    budget = float(config.get("budget", config.get("budget_fraction_of_original", -1)))
    if abs(budget - CHAIN_BUDGET) > 1e-9:
        raise ValueError(f"chain requires a validated 1% checkpoint, got budget {budget}")
    provenance = payload.get("continuation_provenance", {})
    if provenance and (provenance.get("family") != family or abs(float(provenance.get("budget", -1)) - CHAIN_BUDGET) > 1e-9):
        raise ValueError(f"checkpoint provenance does not match {family} at 1%: {path}")
    return payload


def load_selected_saes(sae_paths, family, device):
    """Load exactly five validated 1% SAEs and freeze all their parameters."""
    if set(sae_paths) != set(SECTIONS):
        raise ValueError("selected chain must provide exactly Q1..Q5 SAE checkpoints")
    saes, provenance = {}, {}
    for section in SECTIONS:
        payload = validate_selected_checkpoint(sae_paths[section], family, section)
        config = payload["config"]
        # Current redo checkpoints persist ``mean`` and ``Kmult`` in their
        # resolved config, but not redundant C/K top-level fields.  Derive
        # dimensions from those persisted artifacts (or the score head) so a
        # real selected checkpoint never reaches build_sae with zero dims.
        mean = payload.get("mean")
        channels = int(config.get("C", payload.get("C", 0)) or 0)
        if not channels and torch.is_tensor(mean) and mean.ndim >= 2:
            channels = int(mean.shape[1])
        if not channels:
            normalization = payload.get("normalization", {})
            channels = len(normalization.get("mean", []))
        state = payload.get("state_dict", {})
        score_weight = state.get("score_head.weight")
        explicit_k = int(config.get("K", payload.get("K", 0)) or 0)
        multiplier = int(config.get("Kmult", 0) or 0)
        dictionary = explicit_k or (multiplier * channels if multiplier else 0)
        if not dictionary and torch.is_tensor(score_weight) and score_weight.ndim >= 1:
            dictionary = int(score_weight.shape[0])
        if channels < 1 or dictionary < 1:
            raise ValueError(f"selected checkpoint lacks recoverable C/K dimensions: {sae_paths[section]}")
        net = S.build_sae(family, channels, dictionary,
                          n_blocks=config.get("n_blocks", 3))
        net.load_state_dict(payload["state_dict"])
        saes[section] = freeze_sae(net.to(device))
        provenance[section] = {
            "checkpoint": str(Path(sae_paths[section]).resolve()),
            "checkpoint_sha256": hashlib.sha256(Path(sae_paths[section]).read_bytes()).hexdigest(),
            "family": family, "budget": CHAIN_BUDGET,
        }
    return saes, provenance


def run_chain_training(predictors, saes, means, stds, target_ms, train_loader, val_loader,
                       test_loader, *, device, epochs, learning_rate, carrier_mode,
                       out, provenance):
    """Train one comparator and persist complete chain/gradient/support provenance."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    provenance = dict(provenance)
    provenance.pop("sha256", None)
    provenance["sha256"] = _digest(provenance)
    optimizer = torch.optim.AdamW(
        [parameter for predictor in predictors.values() for parameter in predictor.parameters()],
        lr=float(learning_rate),
    )
    logs, gradients = [], []
    for epoch in range(int(epochs)):
        started = time.time()
        train = train_chain_epoch(
            predictors, saes, means, stds, target_ms, train_loader, optimizer, device,
            scheduled_probability=scheduled_sampling_probability(epoch, epochs),
            carrier_mode=carrier_mode,
        )
        validation = evaluate_chain_modes(predictors, saes, means, stds, target_ms, val_loader)
        logs.append({"epoch": epoch, "p_ss": scheduled_sampling_probability(epoch, epochs),
                     **train, "validation": validation, "seconds": time.time() - started})
        gradients.extend(train["gradient_trace"])
    fidelity = evaluate_chain_modes(predictors, saes, means, stds, target_ms, test_loader)
    torch.save({"state_dict": {key: predictor.state_dict() for key, predictor in predictors.items()},
                "carrier_mode": carrier_mode, "provenance": provenance,
                "resolved_predictor_config": provenance.get("resolved_predictor_config", {})},
               out / "predictors.pt")
    (out / "chain_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    (out / "normalization_provenance.json").write_text(json.dumps({
        section: {"mean": means[section].flatten().cpu().tolist(),
                  "std": stds[section].flatten().cpu().tolist(),
                  "sha256": provenance.get("normalization_sha256", {}).get(section)}
        for section in SECTIONS
    }, indent=2, sort_keys=True) + "\n")
    torch.save({"target_ms": target_ms, "gradient_trace": gradients}, out / "support_and_gradients.pt")
    result = {"schema": "sae-redo-level2-chain-v1", "carrier_mode": carrier_mode,
              "budget": CHAIN_BUDGET, "logs": logs, "fidelity_holdout": fidelity,
              "provenance": provenance}
    result["sha256"] = _digest(result)
    (out / "chain_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _cache_tensor_entries(payload):
    entries = payload.get("tensors", payload.get("files"))
    if isinstance(entries, dict):
        normalized = {}
        for tensor_path, digest in entries.items():
            if isinstance(digest, dict):
                digest = digest.get("sha256")
            normalized[str(tensor_path)] = digest
        return normalized
    if isinstance(entries, list):
        normalized = {}
        for item in entries:
            if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
                raise ValueError("activation-cache tensor entries need path and sha256")
            normalized[str(item["path"])] = item["sha256"]
        return normalized
    raise ValueError("activation-cache manifest must enumerate tensor paths and sha256 values")


def _cache_tensor_requirements(acts, payload):
    root = Path(acts).resolve()
    fidelity_split = payload.get("fidelity_split", "fidelity_holdout")
    if fidelity_split not in {"fidelity_holdout", "test"}:
        raise ValueError("activation-cache fidelity_split must be fidelity_holdout or test")
    requirements = []
    for split in ("train", "validation", fidelity_split):
        for section in SECTIONS:
            requirements.append(root / f"{split}_{section}.pt")
        requirements.append(root / f"{split}_labels.pt")
    return fidelity_split, requirements


def validate_activation_cache_manifest(path, acts=None):
    """Validate and identify the immutable activation-cache manifest."""
    payload = _read_content_addressed_json(path, "activation-cache")
    if payload.get("schema") not in {
        "sae-redo-level2-activation-cache-v1",
        "sae-redo-level2-cache-v1",
    }:
        raise ValueError("unsupported activation-cache manifest schema")
    if acts is not None and payload.get("cache_root") is not None:
        if Path(payload["cache_root"]).resolve() != Path(acts).resolve():
            raise ValueError("activation-cache manifest root does not match --acts")
    if acts is None:
        acts = payload.get("cache_root")
    if acts is None:
        raise ValueError("activation-cache validation requires a cache root")
    fidelity_split, requirements = _cache_tensor_requirements(acts, payload)
    entries = _cache_tensor_entries(payload)
    root = Path(acts).resolve()
    tensor_hashes = {}
    for required in requirements:
        relative = str(required.relative_to(root))
        digest = entries.get(relative, entries.get(str(required)))
        if not digest:
            raise ValueError(f"activation-cache manifest is missing tensor {relative}")
        if not required.is_file():
            raise FileNotFoundError(f"activation-cache tensor not found: {required}")
        actual = file_sha256(required)
        if actual != digest:
            raise ValueError(f"activation-cache tensor SHA-256 does not match: {relative}")
        tensor_hashes[relative] = actual
    return {"path": str(Path(path).resolve()), "sha256": file_sha256(path),
            "manifest": payload, "fidelity_split": fidelity_split,
            "tensor_hashes": tensor_hashes}


def _split_ids(payload, base_dir=None):
    """Find stable split IDs when a manifest exposes them."""
    for key in ("ordered_ids", "source_ids", "sample_ids", "ids"):
        value = payload.get(key)
        if isinstance(value, list):
            ordered = [str(item) for item in value]
            if not ordered or len(set(ordered)) != len(ordered):
                raise ValueError("split manifest ordered IDs must be non-empty and unique")
            declared_hash = payload.get("ordered_ids_sha256")
            if declared_hash is not None and declared_hash != ordered_id_hash(ordered):
                raise ValueError("split manifest ordered-ID hash does not match")
            return ordered
    artifact = payload.get("id_list_artifact")
    if isinstance(artifact, dict) and artifact.get("path") and artifact.get("sha256"):
        artifact_path = Path(artifact["path"])
        if not artifact_path.is_absolute():
            artifact_path = Path(artifact.get("base_dir", base_dir or ".")) / artifact_path
        if not artifact_path.is_file() or file_sha256(artifact_path) != artifact["sha256"]:
            raise ValueError("bound split ID-list artifact is missing or tampered")
        try:
            text = artifact_path.read_text()
            decoded = json.loads(text)
            value = decoded if isinstance(decoded, list) else text.splitlines()
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("bound split ID-list artifact is not a valid ID list") from error
        ordered = [str(item) for item in value if str(item)]
        if not ordered or len(set(ordered)) != len(ordered):
            raise ValueError("bound split ID-list artifact must contain unique IDs")
        return ordered
    return None


def validate_split_manifest(path, role):
    """Validate a train/validation/fidelity split manifest and hash its bytes."""
    payload = _read_content_addressed_json(path, f"{role} split")
    declared = payload.get("split", payload.get("role"))
    if declared is not None and str(declared) != role:
        raise ValueError(f"split manifest {path} declares {declared!r}, expected {role!r}")
    ordered_ids = _split_ids(payload, Path(path).resolve().parent)
    if ordered_ids is None:
        raise ValueError(f"{role} split manifest must bind an ordered ID list")
    return {"path": str(Path(path).resolve()), "sha256": file_sha256(path),
            "manifest": payload, "ids": set(ordered_ids), "ordered_ids": ordered_ids}


def validate_split_manifests(split_manifests):
    """Validate all three ordered-ID manifests and enforce pairwise disjointness."""
    if set(split_manifests) != {"train", "validation", "fidelity_holdout"}:
        raise ValueError("chain requires train, validation, and fidelity_holdout split manifests")
    splits = {role: validate_split_manifest(split_manifests[role], role)
              for role in ("train", "validation", "fidelity_holdout")}
    roles = tuple(splits)
    for index, first in enumerate(roles):
        for second in roles[index + 1:]:
            if splits[first]["ids"].intersection(splits[second]["ids"]):
                raise ValueError(f"split manifests overlap: {first} and {second}")
    return splits


def validate_chain_inputs(selection_manifest, selected_family, selected_sae_paths,
                          field_old_sae_paths, backbone_checkpoint,
                          activation_cache_manifest, split_manifests,
                          *, expected_backbone=None, acts=None):
    """Validate every immutable input required before a chain can launch."""
    selection = validate_selection_manifest(
        selection_manifest, expected_backbone=expected_backbone,
    )
    if selection["selected_family"] != selected_family:
        raise ValueError("selected-family argument does not match selection artifact")
    for paths, family in ((selected_sae_paths, selected_family),
                          (field_old_sae_paths, "field_old")):
        if set(paths) != set(SECTIONS):
            raise ValueError(f"{family} chain input must provide exactly Q1..Q5 checkpoints")
        for section in SECTIONS:
            validate_selected_checkpoint(paths[section], family, section)
    backbone_checkpoint = Path(backbone_checkpoint)
    if not backbone_checkpoint.is_file():
        raise FileNotFoundError(f"frozen backbone checkpoint not found: {backbone_checkpoint}")
    cache = validate_activation_cache_manifest(activation_cache_manifest, acts=acts)
    splits = validate_split_manifests(split_manifests)
    return {
        "selection": {"path": str(Path(selection_manifest).resolve()),
                      "sha256": file_sha256(selection_manifest)},
        "selected_family": selected_family,
        "selected_saes": {
            section: {"path": str(Path(selected_sae_paths[section]).resolve()),
                      "sha256": file_sha256(selected_sae_paths[section])}
            for section in SECTIONS
        },
        "field_old_saes": {
            section: {"path": str(Path(field_old_sae_paths[section]).resolve()),
                      "sha256": file_sha256(field_old_sae_paths[section])}
            for section in SECTIONS
        },
        "frozen_backbone": {"path": str(backbone_checkpoint.resolve()),
                            "sha256": file_sha256(backbone_checkpoint)},
        "activation_cache": {"path": cache["path"], "sha256": cache["sha256"],
                             "fidelity_split": cache["fidelity_split"],
                             "tensor_hashes": cache["tensor_hashes"]},
        "splits": {role: {"path": evidence["path"], "sha256": evidence["sha256"]}
                   for role, evidence in splits.items()},
    }


def _load_selection_manifest(path, family=None, *, expected_backbone=None):
    """Compatibility wrapper now backed by full selector evidence validation."""
    manifest = validate_selection_manifest(path, expected_backbone=expected_backbone)
    if family is not None and family not in {manifest["selected_family"], "field_old"}:
        raise ValueError("requested family is neither selected nor field_old")
    return manifest


def selected_chain_pair(selection_manifest, *, expected_backbone=None):
    """Return the only permitted pair: validated new winner plus field_old."""
    manifest = validate_selection_manifest(selection_manifest,
                                            expected_backbone=expected_backbone)
    return manifest["selected_family"], "field_old"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-manifest", required=True,
                        help="content-addressed selected-new-family eligibility artifact")
    parser.add_argument("--selected-sae-root", required=True,
                        help="selected-family directory containing Q1/sae.pt through Q5/sae.pt")
    parser.add_argument("--field-old-sae-root", required=True,
                        help="matched field_old directory containing Q1/sae.pt through Q5/sae.pt")
    parser.add_argument("--acts", required=True)
    parser.add_argument("--activation-cache-manifest", required=True)
    parser.add_argument("--train-split-manifest", required=True)
    parser.add_argument("--validation-split-manifest", required=True)
    parser.add_argument("--fidelity-holdout-split-manifest", required=True)
    parser.add_argument("--backbone-id", default=None)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--carrier-mode", choices=["regrounded", "raw", "both"], default="both")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    selection = validate_selection_manifest(args.selection_manifest,
                                            expected_backbone=args.backbone_id)
    selected_family = selection["selected_family"]
    split_manifests = {
        "train": args.train_split_manifest,
        "validation": args.validation_split_manifest,
        "fidelity_holdout": args.fidelity_holdout_split_manifest,
    }
    selected_paths = {section: Path(args.selected_sae_root) / section / "sae.pt"
                      for section in SECTIONS}
    old_paths = {section: Path(args.field_old_sae_root) / section / "sae.pt"
                 for section in SECTIONS}
    input_provenance = validate_chain_inputs(
        args.selection_manifest, selected_family, selected_paths, old_paths,
        args.ckpt, args.activation_cache_manifest, split_manifests,
        expected_backbone=args.backbone_id, acts=args.acts,
    )
    # A legacy norm_stats.json alone is not sufficient provenance for this
    # chain: the train-fitted, content-addressed normalization manifest is a
    # required pre-launch input.
    _read_content_addressed_json(Path(args.acts) / "norm_manifest.json", "normalization")
    if not (args.device.startswith("cuda") and torch.cuda.is_available()):
        raise RuntimeError("Level-2 CNN chain requires an available CUDA device")
    device = torch.device(args.device)
    model, _ = load_backbone(args.ckpt, device)
    selected_saes, selected_provenance = load_selected_saes(selected_paths, selected_family, device)
    old_saes, old_provenance = load_selected_saes(old_paths, "field_old", device)
    means, stds, norm_hashes, norm_paths = {}, {}, {}, {}
    for section in SECTIONS:
        means[section], stds[section], norm_hashes[section] = _load_norm(args.acts, section, device)
        norm_paths[section] = str((Path(args.acts) / (
            "norm_manifest.json" if (Path(args.acts) / "norm_manifest.json").is_file()
            else "norm_stats.json"
        )).resolve())
    shapes = {section: torch.load(Path(args.acts) / f"train_{section}.pt", weights_only=True).shape[1:]
              for section in SECTIONS}
    target_ms_by_family = {
        family: {
            section: original_node_budget(
                CHAIN_BUDGET, *shapes[section],
                family_saes[section].K * shapes[section][1] * shapes[section][2],
            )
            for section in SECTIONS
        }
        for family, family_saes in ((selected_family, selected_saes),
                                    ("field_old", old_saes))
    }
    fidelity_split, datasets = load_chain_datasets(args.acts, input_provenance)
    loaders = {"train": torch.utils.data.DataLoader(datasets["train"], batch_size=args.bs, shuffle=True),
               "validation": torch.utils.data.DataLoader(datasets["validation"], batch_size=args.bs),
               fidelity_split: torch.utils.data.DataLoader(datasets[fidelity_split], batch_size=args.bs)}
    base_provenance = {
        "schema": "sae-redo-level2-chain-provenance-v1",
        "backbone": selection["backbone"], "budget": CHAIN_BUDGET,
        "seed": args.seed, "pairs": PAIRS,
        "normalization_sha256": norm_hashes, "target_ms": target_ms_by_family,
        "normalization": {section: {"path": norm_paths[section], "sha256": norm_hashes[section]}
                          for section in SECTIONS},
        "carrier_modes": ["regrounded", "raw"],
        "split": "train/validation/fidelity_holdout",
        "selection": input_provenance["selection"],
        "frozen_backbone": input_provenance["frozen_backbone"],
        "activation_cache": input_provenance["activation_cache"],
        "splits": input_provenance["splits"],
    }
    mode_list = [args.carrier_mode] if args.carrier_mode != "both" else ["regrounded", "raw"]
    for family, saes, sae_provenance in (
        (selected_family, selected_saes, selected_provenance),
        ("field_old", old_saes, old_provenance),
    ):
        for mode in mode_list:
            # Reset before each matched family/mode so initialization,
            # scheduled-sampling draws, and shuffled train order are explicit.
            set_seed(args.seed)
            predictors = {}
            for source, destination in PAIRS:
                predictor = T.TransitionPredictor(saes[source].K, saes[destination].K,
                                                  shapes[source][-1], shapes[destination][-1]).to(device)
                predictors[source, destination] = predictor
            provenance = {
                **base_provenance, "family": family, "carrier_mode": mode,
                "sae": sae_provenance,
                "target_ms": target_ms_by_family[family],
                "resolved_predictor_config": {
                    "family": family, "budget": CHAIN_BUDGET, "seed": args.seed,
                    "epochs": args.epochs, "learning_rate": args.lr,
                    "clip_grad": 1.0, "scheduled_sampling": "linear_0_to_1",
                    "carrier_mode": mode, "pairs": PAIRS,
                },
            }
            run_chain_training(predictors, saes, means, stds, target_ms_by_family[family], loaders["train"],
                               loaders["validation"], loaders[fidelity_split], device=device,
                               epochs=args.epochs, learning_rate=args.lr, carrier_mode=mode,
                               out=Path(args.out) / family / mode, provenance=provenance)


if __name__ == "__main__":
    main()

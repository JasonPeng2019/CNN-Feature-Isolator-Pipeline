"""Immutable continuation and sparse-message replay provenance helpers."""
import hashlib
import json
import math
from pathlib import Path

import torch


_BUDGETS = (0.08, 0.04, 0.02, 0.01)
_STRICT_FAMILIES = {"field_old", "field_strict", "vector_context_strict"}
_RECOVERY_FAMILIES = {"field_recovery", "vector_context"}


def continuation_budgets():
    """The frozen high-to-low budget graph, in continuation order."""
    return _BUDGETS


def continuation_phase_schedule(family, phase, epochs=30, learning_rate=3e-4):
    """Return the immutable optimizer phases for one continuation family."""
    if family == "field_recovery" and phase in {
        "recovery_correction_warm", "recovery_joint"
    }:
        return (("recovery_correction_warm", 3, 1e-4), ("recovery_joint", 20, 1e-4))
    if family == "vector_context" and phase in {"vector_refiner_warm", "vector_joint"}:
        return (("vector_refiner_warm", 3, 1e-4), ("vector_joint", 20, 1e-4))
    if family in _STRICT_FAMILIES and phase == "strict":
        if int(epochs) < 1 or float(learning_rate) <= 0:
            raise ValueError("strict continuation epochs and learning_rate must be positive")
        return (("strict", int(epochs), float(learning_rate)),)
    raise ValueError(f"invalid continuation phase {phase!r} for family {family!r}")


def configure_continuation_phase(module, phase):
    """Set trainability for one fixed continuation phase and return parameters."""
    if phase == "strict" or phase in {"recovery_joint", "vector_joint"}:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    elif phase in {"recovery_correction_warm", "vector_refiner_warm"}:
        prefix = "correction_" if phase.startswith("recovery_") else "refiner_"
        for name, parameter in module.named_parameters():
            parameter.requires_grad_(name.startswith(prefix))
    else:
        raise ValueError(f"unknown continuation phase: {phase}")
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def provenance_sha256(record):
    """Compute the canonical digest of a continuation record without its hash."""
    canonical_record = dict(record)
    canonical_record.pop("sha256", None)
    canonical = json.dumps(canonical_record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _budget(value):
    value = float(value)
    for known in _BUDGETS:
        if math.isclose(value, known, rel_tol=0.0, abs_tol=1e-9):
            return known
    raise ValueError(f"budget must be one of {_BUDGETS}, got {value}")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_file_sha256 = file_sha256


def _parent_budget(budget):
    index = _BUDGETS.index(budget)
    return None if index == 0 else _BUDGETS[index - 1]


def build_continuation_provenance(
    *,
    backbone,
    field,
    family,
    budget,
    seed,
    parent_checkpoint,
    parent_family=None,
    parent_budget=None,
    phase="strict",
):
    """Validate and build the immutable dependency record for one cell."""
    budget = _budget(budget)
    if family not in _STRICT_FAMILIES | _RECOVERY_FAMILIES:
        raise ValueError(f"unknown continuation family: {family}")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    expected_parent_family = family
    expected_parent_budget = _parent_budget(budget)
    if family == "field_recovery":
        expected_parent_family = "field_strict"
        expected_parent_budget = budget
        if phase not in {"recovery_correction_warm", "recovery_joint"}:
            raise ValueError("field_recovery requires an explicit recovery phase")
    elif family == "vector_context":
        expected_parent_family = "vector_context_strict"
        expected_parent_budget = budget
        if phase not in {"vector_refiner_warm", "vector_joint"}:
            raise ValueError("vector_context requires an explicit refiner phase")
    elif phase != "strict":
        raise ValueError(f"{family} continuation must use phase='strict'")

    if expected_parent_budget is None:
        if parent_checkpoint is not None:
            raise ValueError("the 8% root must initialize without a parent checkpoint")
        if parent_family is not None or parent_budget is not None:
            raise ValueError("the 8% root cannot declare a parent")
        start_mode = "initialize"
    else:
        if family in _STRICT_FAMILIES and parent_family is None:
            parent_family = family
        if parent_checkpoint is None:
            raise ValueError(
                f"budget {budget:g} cannot start randomly; expected {expected_parent_budget:g} parent"
            )
        if parent_family != expected_parent_family:
            raise ValueError(
                f"{family} at {budget:g} requires parent family {expected_parent_family}"
            )
        if parent_budget is None:
            parent_budget = expected_parent_budget
        if not math.isclose(float(parent_budget), expected_parent_budget, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"{family} at {budget:g} requires parent budget {expected_parent_budget:g}, got {parent_budget}"
            )
        parent_checkpoint = Path(parent_checkpoint)
        if not parent_checkpoint.is_file():
            raise FileNotFoundError(parent_checkpoint)
        start_mode = "continuation" if family in _STRICT_FAMILIES else "strict_parent"

    parent_path = None if parent_checkpoint is None else str(Path(parent_checkpoint).resolve())
    record = {
        "schema": "sae-redo-level2-continuation-v1",
        "backbone": str(backbone),
        "field": str(field),
        "family": family,
        "budget": budget,
        "seed": seed,
        "continuation_graph": list(_BUDGETS),
        "parent_family": parent_family,
        "parent_budget": None if expected_parent_budget is None else float(parent_budget),
        "parent_checkpoint": parent_path,
        "parent_checkpoint_sha256": None if parent_path is None else _file_sha256(parent_path),
        "start_mode": start_mode,
        "phase": phase,
    }
    if family in _RECOVERY_FAMILIES:
        record["phase_plan"] = {
            "correction_warm_epochs": 3,
            "joint_refinement_epochs": 20,
            "joint_peak_learning_rate": 1e-4,
        }
    else:
        record["phase_plan"] = {}
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
    record["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return record


def write_continuation_provenance(path, record):
    """Write one continuation record without overwriting a prior attempt."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def load_matching_parent_checkpoint(path, *, backbone, field, family, budget, seed):
    """Load a parent only when its persisted continuation identity matches."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    provenance = payload.get("continuation_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"parent checkpoint lacks continuation provenance: {path}")
    if provenance.get("sha256") != provenance_sha256(provenance):
        raise ValueError(f"parent checkpoint provenance hash does not match: {path}")
    expected_budget = _budget(budget)
    for key, expected in (("backbone", str(backbone)), ("field", str(field)),
                          ("family", family), ("budget", expected_budget), ("seed", int(seed))):
        actual = provenance.get(key)
        if key == "budget":
            valid = actual is not None and math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-9)
        else:
            valid = actual == expected
        if not valid:
            raise ValueError(f"parent checkpoint {path} does not match {key}={expected!r}")
    return payload


def save_sparse_message(
    path,
    indices,
    values,
    shape,
    *,
    reconstruction=None,
    decoder_requires_support=False,
):
    """Save only charged sparse indices/values and optional replay evidence."""
    path = Path(path)
    shape = tuple(int(item) for item in shape)
    if len(shape) != 4 or any(item < 1 for item in shape):
        raise ValueError("shape must be (batch, channels, height, width)")
    indices = torch.as_tensor(indices).detach().cpu().long()
    values = torch.as_tensor(values).detach().cpu().float()
    if indices.ndim != 2 or values.shape != indices.shape or indices.shape[0] != shape[0]:
        raise ValueError("indices and values must both have shape (batch, M)")
    capacity = shape[1] * shape[2] * shape[3]
    if indices.numel() and (indices.min() < 0 or indices.max() >= capacity):
        raise ValueError("sparse message index is outside the coefficient field")
    if indices.shape[1] < 1:
        raise ValueError("sparse message must contain at least one value")
    if any(torch.unique(row).numel() != indices.shape[1] for row in indices):
        raise ValueError("sparse message support indices must be unique per sample")
    if not torch.isfinite(values).all():
        raise ValueError("sparse message values must be finite")
    payload = {
        "schema": "sae-redo-level2-sparse-message-v1",
        "shape": shape,
        "indices": indices,
        "values": values,
        "decoder_requires_support": bool(decoder_requires_support),
    }
    if reconstruction is not None:
        payload["recorded_reconstruction"] = torch.as_tensor(reconstruction).detach().cpu()
    forbidden = {"source_activation", "encoder_state", "unmasked_values", "residual"}
    if forbidden.intersection(payload):
        raise ValueError("sparse message contains forbidden source-side data")
    if path.exists():
        raise FileExistsError(path)
    torch.save(payload, path)
    return payload


def replay_sparse_message(decoder, path, atol=1e-6, rtol=1e-5):
    """Replay a saved message through a decoder without source-side inputs."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    allowed = {"schema", "shape", "indices", "values", "recorded_reconstruction",
               "decoder_requires_support"}
    if set(payload) - allowed or payload.get("schema") != "sae-redo-level2-sparse-message-v1":
        raise ValueError("invalid or non-decoder-only sparse message artifact")
    shape = tuple(payload["shape"])
    indices, values = payload["indices"], payload["values"]
    decode = decoder.decode if hasattr(decoder, "decode") else decoder
    try:
        parameter = next(decoder.parameters())
        device, dtype = parameter.device, parameter.dtype
    except (AttributeError, StopIteration):
        device, dtype = values.device, values.dtype
    indices, values = indices.to(device), values.to(device=device, dtype=dtype)
    flat = torch.zeros(shape[0], shape[1] * shape[2] * shape[3], device=device, dtype=dtype)
    flat.scatter_(1, indices, values)
    sparse_values = flat.reshape(shape)
    support = torch.zeros_like(flat, dtype=torch.bool).scatter_(1, indices, True).reshape(shape)
    with torch.no_grad():
        if payload.get("decoder_requires_support", False):
            reconstructed = decode(sparse_values, support)
        else:
            reconstructed = decode(sparse_values)
    recorded = payload.get("recorded_reconstruction")
    max_abs_error = 0.0
    if recorded is not None:
        recorded = recorded.to(device=device, dtype=reconstructed.dtype)
        if recorded.shape != reconstructed.shape:
            raise ValueError("recorded reconstruction shape does not match decoder replay")
        max_abs_error = float((reconstructed - recorded).abs().max().item())
        if not torch.allclose(reconstructed, recorded, atol=atol, rtol=rtol):
            raise ValueError(f"decoder-only replay mismatch: max_abs_error={max_abs_error}")
    return {"reconstruction": reconstructed, "max_abs_error": max_abs_error}

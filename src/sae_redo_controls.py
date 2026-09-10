"""Matched sparse-message controls for a validation-selected SAE cell."""
from contextlib import contextmanager

import torch

from masks import select_topm


BASE_CONTROL_NAMES = (
    "selected",
    "random_support",
    "magnitude_support",
    "constant_values",
    "shuffled_values",
    "selected_values_on_random_support",
    "fixed_support_across_examples",
    "empty_message",
    "attention_disabled",
)
RECOVERY_CONTROL_NAMES = ("recovery_correction_disabled", "recovery_strict_disabled")


def control_names(family):
    """Return the immutable control order for one final SAE family."""
    names = list(BASE_CONTROL_NAMES)
    if family == "field_recovery":
        names.extend(RECOVERY_CONTROL_NAMES)
    return tuple(names)


def _encode_score_value(net, normalized):
    if hasattr(net, "encode_score_value"):
        return net.encode_score_value(normalized)
    values = net.encode(normalized)
    return values.abs(), values


def _support_from_indices(indices, capacity, shape):
    support = torch.zeros(indices.shape[0], capacity, dtype=torch.bool, device=indices.device)
    support.scatter_(1, indices, True)
    return support.reshape(shape)


def _message_from_indices(values, indices, shape):
    flat_values = values.reshape(values.shape[0], -1)
    capacity = flat_values.shape[1]
    support = _support_from_indices(indices, capacity, shape)
    sparse = torch.zeros_like(flat_values)
    sparse.scatter_(1, indices, flat_values.gather(1, indices))
    return sparse.reshape(shape), support


def _random_indices(batch, capacity, m, seed, device):
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    rows = [torch.randperm(capacity, generator=generator)[:m] for _ in range(batch)]
    return torch.stack(rows).to(device=device)


@contextmanager
def _attention_disabled(net):
    """Temporarily disable contextual attention without mutating final state."""
    modules = []
    for module in net.modules():
        if module.__class__.__name__ == "PooledAttentionCorrection":
            out_proj = getattr(module, "out_proj", None)
            if out_proj is not None:
                modules.append((out_proj, out_proj.weight.detach().clone(),
                                None if out_proj.bias is None else out_proj.bias.detach().clone()))
    try:
        with torch.no_grad():
            for module, weight, bias in modules:
                module.weight.zero_()
                if module.bias is not None:
                    module.bias.zero_()
        yield
    finally:
        with torch.no_grad():
            for module, weight, bias in modules:
                module.weight.copy_(weight)
                if module.bias is not None and bias is not None:
                    module.bias.copy_(bias)


def _decode(net, sparse, support, name):
    if name == "recovery_correction_disabled":
        return net.strict_decode(sparse)
    if name == "recovery_strict_disabled":
        return net.decode(sparse, support) - net.strict_decode(sparse)
    return net.decode(sparse, support)


@torch.no_grad()
def generate_control_reconstructions(net, normalized, target_m, *, family, seed=20260827):
    """Build all named controls with one exact support budget.

    Returned entries contain reconstruction plus sparse message/support data so
    callers can persist the actual control evidence rather than only metrics.
    """
    if normalized.ndim != 4:
        raise ValueError("normalized activations must have shape (B,C,H,W)")
    scores, values = _encode_score_value(net, normalized.float())
    selected, selected_support, selected_indices = select_topm(values.float(), scores.float(), target_m)
    shape = tuple(selected.shape)
    flat_values = values.float().reshape(values.shape[0], -1)
    capacity = flat_values.shape[1]
    m = selected_indices.shape[1]
    random_indices = _random_indices(values.shape[0], capacity, m, seed, values.device)
    random_sparse, random_support = _message_from_indices(values.float(), random_indices, shape)
    selected_values = selected.reshape(values.shape[0], -1).gather(1, selected_indices)
    random_selected_sparse = torch.zeros_like(flat_values)
    random_selected_sparse.scatter_(1, random_indices, selected_values)
    random_selected_sparse = random_selected_sparse.reshape(shape)
    constant_values = selected_values.mean(dim=1, keepdim=True).expand_as(selected_values)
    constant_sparse = torch.zeros_like(flat_values)
    constant_sparse.scatter_(1, selected_indices, constant_values)
    constant_sparse = constant_sparse.reshape(shape)
    shuffled_values = torch.roll(selected_values, shifts=1, dims=1)
    shuffled_sparse = torch.zeros_like(flat_values)
    shuffled_sparse.scatter_(1, selected_indices, shuffled_values)
    shuffled_sparse = shuffled_sparse.reshape(shape)
    fixed_indices = selected_indices[:1].expand(values.shape[0], -1)
    fixed_sparse = torch.zeros_like(flat_values)
    fixed_sparse.scatter_(1, fixed_indices, selected_values)
    fixed_sparse = fixed_sparse.reshape(shape)
    fixed_support = _support_from_indices(fixed_indices, capacity, shape)
    magnitude, magnitude_support, magnitude_indices = select_topm(
        values.float(), values.float().abs(), target_m
    )
    empty_sparse = torch.zeros_like(values.float())
    empty_support = torch.zeros_like(values, dtype=torch.bool)

    messages = {
        "selected": (selected, selected_support, selected_indices),
        "random_support": (random_sparse, random_support, random_indices),
        "magnitude_support": (magnitude, magnitude_support, magnitude_indices),
        "constant_values": (constant_sparse, selected_support, selected_indices),
        "shuffled_values": (shuffled_sparse, selected_support, selected_indices),
        "selected_values_on_random_support": (random_selected_sparse, random_support, random_indices),
        "fixed_support_across_examples": (fixed_sparse, fixed_support, fixed_indices),
        "empty_message": (empty_sparse, empty_support, torch.empty(values.shape[0], 0, dtype=torch.long,
                                                                      device=values.device)),
    }
    with _attention_disabled(net):
        attention_scores, attention_values = _encode_score_value(net, normalized.float())
        attention_sparse, attention_support, attention_indices = select_topm(
            attention_values.float(), attention_scores.float(), target_m
        )
    messages["attention_disabled"] = (attention_sparse, attention_support, attention_indices)
    if family == "field_recovery":
        messages["recovery_correction_disabled"] = messages["selected"]
        messages["recovery_strict_disabled"] = messages["selected"]

    outputs = {}
    for name in control_names(family):
        sparse, support, indices = messages[name]
        if name == "attention_disabled":
            with _attention_disabled(net):
                reconstruction = _decode(net, sparse, support, name)
        else:
            reconstruction = _decode(net, sparse, support, name)
        outputs[name] = {
            "reconstruction": reconstruction,
            "sparse": sparse,
            "support": support,
            "indices": indices,
            "active_count_m": int(support.reshape(support.shape[0], -1).sum(1)[0].item()),
        }
    return outputs

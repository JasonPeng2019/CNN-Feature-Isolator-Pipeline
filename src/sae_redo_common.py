"""Shared invariants and artifact helpers for the SAE redo Level 2 lanes."""
import hashlib
import json
import math
import sys
from pathlib import Path

import torch


def original_node_budget(budget, channels, height, width, capacity):
    if not 0 < budget <= 1:
        raise ValueError(f"budget must be in (0, 1], got {budget}")
    original_nodes = int(channels) * int(height) * int(width)
    target = max(1, round(float(budget) * original_nodes))
    if target > int(capacity):
        raise ValueError(
            f"requested original-node budget {target} exceeds coefficient capacity {capacity}"
        )
    return target


def validate_dictionary_multiplier(sae_type, multiplier, channels):
    """Enforce the frozen dictionary size for a redo family."""
    if int(channels) < 1:
        raise ValueError("channels must be positive")
    expected = 8 if sae_type == "field_old" else 4 if sae_type in {
        "field_strict", "field_recovery", "vector_context", "vector_context_strict"
    } else None
    if expected is None:
        raise ValueError(f"unknown redo SAE family: {sae_type}")
    if int(multiplier) != expected:
        raise ValueError(
            f"{sae_type} requires K={expected}C (Kmult={expected}), got Kmult={multiplier}"
        )
    return expected


def sparse_bit_accounting(
    m,
    coefficient_count,
    coefficient_value_bits=16,
    *,
    value_quantizer="symmetric_uniform_int16",
    metadata_bits=0,
    original_node_count=None,
):
    """Return explicit rate accounting for one exact sparse message.

    ``m`` counts transmitted support entries, not numerical nonzeros.  The
    support is represented by a fixed-width index into the flattened
    coefficient field; values use a signed quantizer with the declared bit
    width.  Quantizer/normalization/shape information is charged through
    ``metadata_bits``.  A caller that has an activation shape should pass its
    ``C*H*W`` product to obtain bits per original activation node.
    """
    if m < 1 or coefficient_count < m:
        raise ValueError("invalid sparse accounting dimensions")
    if int(coefficient_value_bits) < 1:
        raise ValueError("coefficient_value_bits must be positive")
    if int(metadata_bits) < 0:
        raise ValueError("metadata_bits must be non-negative")
    if original_node_count is not None and int(original_node_count) < 1:
        raise ValueError("original_node_count must be positive")
    index_width = max(1, math.ceil(math.log2(coefficient_count)))
    support_bits = int(m) * index_width
    value_bits = int(m) * int(coefficient_value_bits)
    total_bits = support_bits + value_bits + int(metadata_bits)
    result = {
        "active_count_m": int(m),
        "support_index_width_bits": index_width,
        "support_index_bits": support_bits,
        "signed_value_quantizer": value_quantizer,
        "value_quantizer": value_quantizer,
        "signed_value_bits": value_bits,
        "value_bits": value_bits,
        # Keep the old key as a compatibility alias for existing summaries.
        "coefficient_value_bits": value_bits,
        "metadata_bits": int(metadata_bits),
        "total_sparse_bits": total_bits,
        "total_bits": total_bits,
    }
    if original_node_count is not None:
        result["original_node_count"] = int(original_node_count)
        result["bits_per_original_activation_node"] = total_bits / int(original_node_count)
        result["bits_per_original_node"] = result["bits_per_original_activation_node"]
    else:
        result["original_node_count"] = None
        result["bits_per_original_activation_node"] = None
        result["bits_per_original_node"] = None
    return result


def ordered_id_hash(ids):
    """Hash an ordered sequence of stable sample IDs."""
    text = "\n".join(str(item) for item in ids) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_id_hash = ordered_id_hash


def stratified_validation_split(source_ids, labels, seed=20260827):
    """Split sorted sample IDs into deterministic class-stratified halves.

    The returned values are ordered by sample ID, while the seeded per-class
    permutation decides which members enter each half.  This makes the
    manifests stable across ImageFolder enumeration order and preserves class
    coverage whenever a class has at least two examples.
    """
    ids = [str(item) for item in source_ids]
    labels = list(labels)
    if len(ids) != len(labels):
        raise ValueError("source_ids and labels must have equal length")
    if len(set(ids)) != len(ids):
        raise ValueError("source_ids must be unique")
    ordered = sorted(zip(ids, labels), key=lambda pair: pair[0])
    groups = {}
    for index, (_, label) in enumerate(ordered):
        groups.setdefault(str(label), []).append(index)
    target_selection = len(ordered) // 2
    generator = torch.Generator().manual_seed(int(seed))
    selected = set()
    remainders = []
    for label in sorted(groups):
        group = groups[label]
        permutation = torch.randperm(len(group), generator=generator).tolist()
        base = len(group) // 2
        selected.update(group[index] for index in permutation[:base])
        remainders.append((label, [group[index] for index in permutation[base:]]))
    # Balance odd per-class groups so the global split is exactly half when
    # possible, without making the assignment depend on filesystem order.
    remaining = target_selection - len(selected)
    for _, candidates in remainders:
        if remaining <= 0:
            break
        if candidates:
            selected.add(candidates[0])
            remaining -= 1
    selection = sorted(ordered[index][0] for index in selected)
    holdout = sorted(ordered[index][0] for index in range(len(ordered)) if index not in selected)
    return selection, holdout


def stratified_partition_indices(labels, train_count, validation_count, seed=20260827):
    """Partition labeled rows into deterministic disjoint train/validation IDs."""
    labels = list(labels)
    total = len(labels)
    train_count = total if int(train_count) < 0 else int(train_count)
    validation_count = total - train_count if int(validation_count) < 0 else int(validation_count)
    if train_count < 1 or validation_count < 1 or train_count + validation_count > total:
        raise ValueError("requested stratified partition exceeds available labeled rows")
    groups = {}
    for index, label in enumerate(labels):
        groups.setdefault(str(label), []).append(index)
    generator = torch.Generator().manual_seed(int(seed))
    permutations = {
        label: [group[item] for item in torch.randperm(len(group), generator=generator).tolist()]
        for label, group in sorted(groups.items())
    }

    def allocate(count, pools):
        sizes = {label: len(pool) for label, pool in pools.items()}
        pool_total = sum(sizes.values())
        raw = {label: count * size / pool_total for label, size in sizes.items()}
        allocation = {label: min(size, int(raw[label])) for label, size in sizes.items()}
        remaining = count - sum(allocation.values())
        for label in sorted(sizes, key=lambda item: (-(raw[item] - int(raw[item])), item)):
            if remaining <= 0:
                break
            if allocation[label] < sizes[label]:
                allocation[label] += 1
                remaining -= 1
        if remaining:
            raise ValueError("could not satisfy stratified partition counts")
        return allocation

    train_alloc = allocate(train_count, permutations)
    train, remaining = [], {}
    for label, pool in permutations.items():
        take = train_alloc[label]
        train.extend(pool[:take])
        remaining[label] = pool[take:]
    val_alloc = allocate(validation_count, remaining)
    validation = [index for label, pool in remaining.items() for index in pool[:val_alloc[label]]]
    return sorted(train), sorted(validation)


def normalization_artifact(mean, std, train_ids, *, source=None):
    """Build a content-addressed train-only normalization record."""
    ids = [str(item) for item in train_ids]
    payload = {
        "schema": "sae-redo-level2-normalization-v1",
        "mean": [float(value) for value in mean],
        "std": [max(float(value), 1e-6) for value in std],
        "fit_train_ids": ids,
        "fit_train_ids_sha256": ordered_id_hash(ids),
    }
    if source is not None:
        payload["source"] = source
    payload["sha256"] = _digest(payload)
    return payload


def deterministic_indices(total, count, seed):
    if count < 0:
        return list(range(total))
    return torch.randperm(total, generator=torch.Generator().manual_seed(seed))[: min(total, count)].tolist()


def _digest(payload):
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def write_split_manifest(out_dir, lane, source, selections, seeds):
    path = Path(out_dir) / "split_manifest.json"
    if path.exists():
        raise FileExistsError(path)
    payload = {
        "schema": "sae-redo-level2-split-v1",
        "lane": lane,
        "source": source,
        "selections": selections,
        "seeds": seeds,
        "counts": {name: len(indices) for name, indices in selections.items()},
    }
    payload["sha256"] = _digest(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def create_run_root(path):
    root = Path(path)
    root.mkdir(parents=True, exist_ok=False)
    return root


class _Tee:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, value):
        self.stream.write(value)
        self.file.write(value)
        self.file.flush()
        return len(value)

    def flush(self):
        self.stream.flush()
        self.file.flush()


def attach_run_log(out_dir):
    """Mirror run stdout/stderr to a per-run log after exclusive root creation."""
    file = open(Path(out_dir) / "run.log", "a", buffering=1)
    sys.stdout = _Tee(sys.stdout, file)
    sys.stderr = _Tee(sys.stderr, file)
    return file


def write_json_new(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

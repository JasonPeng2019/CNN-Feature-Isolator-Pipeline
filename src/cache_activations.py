"""Cache frozen-backbone activations H_t and per-channel norm stats (Phase 0)."""
import argparse, os, sys, json
import hashlib
import torch
sys.path.insert(0, os.path.dirname(__file__))
import data as D
import backbone as B
from sae_redo_common import deterministic_indices, ordered_id_hash

SECTIONS = ["Q1", "Q2", "Q3", "Q4", "Q5"]


def _metadata_digest(payload):
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def write_cache_metadata(out_dir, *, dataset, backbone, train_count, test_count):
    """Persist immutable split counts/IDs without reopening test tensors."""
    path = os.path.join(out_dir, "cache_metadata.json")
    if os.path.exists(path):
        raise FileExistsError(path)
    payload = {
        "schema": "sae-redo-level2-cache-metadata-v1",
        "dataset": str(dataset), "backbone": str(backbone),
        "train_count": int(train_count), "test_count": int(test_count),
        "train_ids": [f"train:{index}" for index in range(int(train_count))],
        "test_ids": [f"test:{index}" for index in range(int(test_count))],
    }
    payload["sha256"] = _metadata_digest(payload)
    with open(path, "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
    return payload


def load_cache_metadata(out_dir):
    """Load and validate immutable cache split counts/IDs."""
    path = os.path.join(out_dir, "cache_metadata.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"activation cache metadata is required before reading fidelity tensors: {path}"
        )
    with open(path) as stream:
        payload = json.load(stream)
    supplied = payload.get("sha256") if isinstance(payload, dict) else None
    canonical = dict(payload) if isinstance(payload, dict) else {}
    canonical.pop("sha256", None)
    if not supplied or supplied != _metadata_digest(canonical):
        raise ValueError(f"activation cache metadata SHA-256 does not match: {path}")
    schema = payload.get("schema")
    if schema not in {
        "sae-redo-level2-cache-metadata-v1",
        "sae-redo-level2-cache-metadata-v1-compact",
    }:
        raise ValueError(f"unsupported activation cache metadata schema: {path}")
    if schema.endswith("-compact"):
        # The original cache roots predate this sidecar and use deterministic
        # split IDs.  Keep their immutable metadata small while exposing the
        # same validated ID contract to trainers as freshly-written caches.
        for prefix_key in ("train_id_prefix", "test_id_prefix"):
            prefix = payload.get(prefix_key)
            if not isinstance(prefix, str) or not prefix:
                raise ValueError(f"compact cache metadata requires {prefix_key}")
        payload["train_ids"] = [
            f"{payload['train_id_prefix']}{index}"
            for index in range(int(payload.get("train_count", 0)))
        ]
        payload["test_ids"] = [
            f"{payload['test_id_prefix']}{index}"
            for index in range(int(payload.get("test_count", 0)))
        ]
    for count_key, ids_key in (("train_count", "train_ids"), ("test_count", "test_ids")):
        count = payload.get(count_key)
        ids = payload.get(ids_key)
        if not isinstance(count, int) or count < 1 or not isinstance(ids, list) or len(ids) != count:
            raise ValueError(f"activation cache metadata has invalid {count_key}/{ids_key}")
        if len(set(map(str, ids))) != count:
            raise ValueError(f"activation cache metadata {ids_key} must be unique")
    return payload


def load_backbone(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    model = {"resnet20": B.resnet20, "resnet56": B.resnet56, "resnet110": B.resnet110}[ck["arch"]](
        D.num_classes(ck["dataset"]))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


@torch.no_grad()
def cache_split(model, loader, device, out_dir, split):
    feats = {q: [] for q in SECTIONS}
    logits_all, labels_all = [], []
    for x, y in loader:
        x = x.to(device)
        logits, taps = model.forward_with_taps(x)
        for q in SECTIONS:
            feats[q].append(taps[q].half().cpu())
        logits_all.append(logits.cpu())
        labels_all.append(y)
    os.makedirs(out_dir, exist_ok=True)
    for q in SECTIONS:
        torch.save(torch.cat(feats[q]), os.path.join(out_dir, f"{split}_{q}.pt"))
    torch.save(torch.cat(logits_all), os.path.join(out_dir, f"{split}_logits.pt"))
    torch.save(torch.cat(labels_all), os.path.join(out_dir, f"{split}_labels.pt"))


def compute_norm_stats(out_dir, train_indices=None, train_ids=None):
    """Fit norm stats on selected cached train IDs and persist their hash.

    Omitting ``train_indices`` retains the legacy all-train behavior but still
    records the fitted population explicitly.  Redo trainers pass their clean
    fit IDs so validation and fidelity holdout activations cannot contribute.
    """
    stats = {}
    first = torch.load(os.path.join(out_dir, "train_Q1.pt"), weights_only=True)
    total = first.shape[0]
    indices = list(range(total)) if train_indices is None else [int(i) for i in train_indices]
    if not indices or min(indices) < 0 or max(indices) >= total:
        raise ValueError("normalization train_indices must be non-empty and within cached train split")
    ids = [str(i) for i in range(total)] if train_ids is None else [str(i) for i in train_ids]
    if len(ids) != total:
        raise ValueError("train_ids must align with cached train tensors")
    fit_ids = [ids[index] for index in indices]
    for q in SECTIONS:
        H = torch.load(os.path.join(out_dir, f"train_{q}.pt"), weights_only=True).float()[indices]
        mean = H.mean(dim=(0, 2, 3))
        std = H.std(dim=(0, 2, 3)).clamp_min(1e-6)
        stats[q] = {"mean": mean.tolist(), "std": std.tolist(),
                    "fit_train_ids": fit_ids,
                    "fit_train_ids_sha256": ordered_id_hash(fit_ids)}
    manifest = {"schema": "sae-redo-level2-normalization-manifest-v1",
                "fit_train_indices": indices, "fit_train_ids": fit_ids,
                "fit_train_ids_sha256": stats[SECTIONS[0]]["fit_train_ids_sha256"],
                "sections": stats}
    import hashlib
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    with open(os.path.join(out_dir, "norm_stats.json"), "w") as f:
        json.dump(stats, f, sort_keys=True)
    with open(os.path.join(out_dir, "norm_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return stats, manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="./runs/backbone_r56_c100/best.pt")
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--out", default="./runs/acts_r56_c100")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--norm_train_count", type=int, default=-1,
                    help="number of cached train IDs used for normalization; -1 means all")
    ap.add_argument("--norm_seed", type=int, default=20260827)
    args = ap.parse_args()
    model, ck = load_backbone(args.ckpt, args.device)
    # no augmentation -> deterministic activations
    train_loader, test_loader = D.get_loaders(args.dataset, root="./data", batch_size=256,
                                               augment=False, shuffle_train=False)
    cache_split(model, train_loader, args.device, args.out, "train")
    cache_split(model, test_loader, args.device, args.out, "test")
    write_cache_metadata(
        args.out, dataset=args.dataset, backbone=ck.get("arch", "unknown"),
        train_count=len(train_loader.dataset), test_count=len(test_loader.dataset),
    )
    count = len(train_loader.dataset) if args.norm_train_count < 0 else args.norm_train_count
    indices = deterministic_indices(len(train_loader.dataset), count, args.norm_seed)
    compute_norm_stats(args.out, indices, [str(i) for i in range(len(train_loader.dataset))])
    print(f"CACHE DONE -> {args.out} (backbone acc {ck['acc']:.2f})", flush=True)


if __name__ == "__main__":
    main()

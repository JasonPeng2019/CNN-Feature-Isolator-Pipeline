"""Evaluate one selected SAE redo ViT checkpoint once on its reserved test set."""
import argparse
import json
import sys
from pathlib import Path

import torch
from torchvision.datasets import ImageFolder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import sae as S
from sae_redo_common import (attach_run_log, create_run_root, sparse_bit_accounting,
                             write_json_new, write_split_manifest)
from vit_sae_redo import ViTWrap, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    source = Path(args.source_run)
    checkpoint = torch.load(source / "sae.pt", map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    source_manifest = json.loads((source / "split_manifest.json").read_text())
    test_indices = source_manifest["selections"]["test"]
    if not test_indices:
        raise ValueError("source run has no reserved test split")
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    out = create_run_root(args.out)
    attach_run_log(out)
    data_root = Path(config["data_root"])
    raw_val = ImageFolder(data_root / "val")
    manifest = write_split_manifest(out, "vit-test-confirmation", {
        "source_run": str(source.resolve()), "source_manifest_sha256": source_manifest["sha256"],
        "dataset": "imagenette-imagefolder-v1", "data_root": str(data_root.resolve()),
        "validation_source_ids": sorted(str(Path(path).resolve().relative_to(data_root.resolve()))
                                         for path, _ in raw_val.samples),
    }, {"train": [], "validation": [], "test": test_indices}, {"model_seed": config["seed"], "data_seed": config["data_seed"]})
    vit = ViTWrap(config["model"], device)
    test_ds = ImageFolder(data_root / "val", transform=vit.tf)
    import torch.utils.data as tud
    loader = tud.DataLoader(tud.Subset(test_ds, test_indices), batch_size=config["val_bs"], shuffle=False,
                            num_workers=config["num_workers"], pin_memory=device.type == "cuda")
    channels = vit.D
    kwargs = {"d": 2 * channels, "n_blocks": 2} if config["sae_type"] == "field" else {}
    net = S.build_sae(config["sae_type"], channels, config["Kmult"] * channels, **kwargs).to(device)
    net.load_state_dict(checkpoint["state_dict"])
    mean, std = checkpoint["mean"].to(device), checkpoint["std"].to(device)
    coefficient_count = config["Kmult"] * channels * vit.grid * vit.grid
    result_path = source / "result.json"
    if not result_path.exists():  # Historical pre-contract runs retain this name.
        result_path = source / "vit_result.json"
    result_source = json.loads(result_path.read_text())
    target_m = result_source["active_count_m"]
    metrics = evaluate(vit, net, loader, config["block"], checkpoint["prefix_count"], mean, std, target_m, device)
    accounting = sparse_bit_accounting(
        target_m,
        coefficient_count,
        value_quantizer=config.get("value_quantizer", "symmetric_uniform_int16"),
        metadata_bits=config.get("metadata_bits", 64),
        original_node_count=channels * vit.grid * vit.grid,
    )
    write_json_new(out / "run_metadata.json", {"schema": "sae-redo-level2-test-v1", "source_run": str(source.resolve()),
        "source_manifest_sha256": source_manifest["sha256"], "manifest_sha256": manifest["sha256"], "device": str(device),
        "cuda_available": torch.cuda.is_available(), "accounting": accounting})
    write_json_new(out / "test_result.json", {**metrics, **accounting, "source_run": str(source.resolve()),
        "source_manifest_sha256": source_manifest["sha256"], "manifest_sha256": manifest["sha256"], "split": "reserved_test"})
    print("VIT-REDO TEST DONE", json.dumps({k: metrics[k] for k in ("pred_agree", "kl", "rel_l2", "fvu")}))


if __name__ == "__main__":
    main()

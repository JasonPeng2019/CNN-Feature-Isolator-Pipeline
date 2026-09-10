"""Apply the frozen ViT-Base decision rule and optionally start one test confirmation."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def collect(matrix_root):
    rows = []
    for path in Path(matrix_root).glob("vit_base_*/*"):
        if path.name == "vit_result.json":
            rows.append((path.parent, json.loads(path.read_text())))
    return rows


def group(rows, sae_type, budget):
    matches = [(path, result) for path, result in rows if result.get("sae_type") == sae_type and result.get("budget_fraction_of_original") == budget]
    if len(matches) != 3 or {result["seed"] for _, result in matches} != {0, 1, 2}:
        raise ValueError(f"expected exactly three seeds for {sae_type} budget {budget}, found {len(matches)}")
    return sorted(matches, key=lambda pair: pair[1]["seed"])


def validate_artifacts(items):
    """Require the split and run metadata evidence before selection."""
    for path, result in items:
        manifest_path, metadata_path, checkpoint_path = path / "split_manifest.json", path / "run_metadata.json", path / "sae.pt"
        if not all(path.exists() for path in (manifest_path, metadata_path, checkpoint_path)):
            raise ValueError(f"missing required artifact under {path}")
        manifest = json.loads(manifest_path.read_text())
        metadata = json.loads(metadata_path.read_text())
        digest = manifest.get("sha256")
        canonical = dict(manifest)
        canonical.pop("sha256", None)
        actual_digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if digest != actual_digest:
            raise ValueError(f"manifest SHA-256 does not match its content under {path}")
        if manifest.get("counts") != {name: len(indices) for name, indices in manifest.get("selections", {}).items()}:
            raise ValueError(f"manifest selection counts are invalid under {path}")
        if not digest or result.get("manifest_sha256") != digest or metadata.get("manifest_sha256") != digest:
            raise ValueError(f"manifest digest mismatch under {path}")
        if result.get("active_count_m") != metadata.get("accounting", {}).get("active_count_m"):
            raise ValueError(f"active-count metadata mismatch under {path}")
        if result.get("ntrain") != manifest["counts"].get("train") or result.get("nval") != manifest["counts"].get("validation"):
            raise ValueError(f"result split counts mismatch under {path}")
        if result.get("ntest_reserved") != manifest["counts"].get("test"):
            raise ValueError(f"result test count mismatch under {path}")
        if (result.get("ntrain"), result.get("nval"), result.get("ntest_reserved")) != (2000, 500, 500):
            raise ValueError(f"decision run is not at the frozen full-run split sizes under {path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})
        if checkpoint.get("manifest_sha256") != digest:
            raise ValueError(f"checkpoint manifest digest mismatch under {path}")
        for key in ("model", "block", "sae_type", "budget", "seed"):
            if config.get(key) != result.get(key):
                raise ValueError(f"checkpoint config mismatch for {key} under {path}")
        if (config.get("Kmult"), config.get("epochs"), config.get("ntrain"), config.get("nval"), config.get("ntest"), config.get("data_seed")) != (4, 20, 2000, 500, 500, 0):
            raise ValueError(f"checkpoint does not match the frozen ViT decision configuration under {path}")
        budget = result.get("budget")
        if budget not in (0.08, 0.04, 0.02, 0.01) or result.get("budget_fraction_of_original") != budget:
            raise ValueError(f"invalid or inconsistent budget under {path}")
        shape = metadata.get("original_shape")
        capacity = metadata.get("coefficient_count")
        if not shape or not capacity:
            raise ValueError(f"missing original shape/accounting capacity under {path}")
        if result.get("D") != shape[0] or result.get("K") != 4 * shape[0] or capacity != 4 * math.prod(shape):
            raise ValueError(f"latent capacity does not match frozen K=4D under {path}")
        expected_m = min(max(1, round(budget * math.prod(shape))), capacity)
        index_width = max(1, math.ceil(math.log2(capacity)))
        expected_accounting = {
            "active_count_m": expected_m,
            "support_index_bits": expected_m * index_width,
            "coefficient_value_bits": expected_m * 16,
            "total_sparse_bits": expected_m * (index_width + 16),
        }
        if any(result.get(key) != value or metadata.get("accounting", {}).get(key) != value
               for key, value in expected_accounting.items()):
            raise ValueError(f"budget/accounting mismatch under {path}")


def mean(items, key):
    return sum(result[key] for _, result in items) / len(items)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--run-test", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    rows = collect(args.matrix_root)
    if len(rows) != 24:
        raise ValueError(f"expected exactly 24 ViT-Base decision results, found {len(rows)}")
    if any(result.get("model") != "vit_base_patch16_224" or result.get("block") != 10 for _, result in rows):
        raise ValueError("all decision results must be ViT-Base block 10")
    all_groups = {(sae_type, budget): group(rows, sae_type, budget)
                  for sae_type in ("field", "context_vector")
                  for budget in (0.08, 0.04, 0.02, 0.01)}
    validate_artifacts([item for items in all_groups.values() for item in items])
    baseline_2 = all_groups[("field", 0.02)]
    candidate_2 = all_groups[("context_vector", 0.02)]
    candidate_4 = all_groups[("context_vector", 0.04)]
    collapsed = [(path, result) for path, result in candidate_2 + candidate_4
                 if result["rel_l2"] > 1 or result["fvu"] > 1]
    baseline_agreement, baseline_kl = mean(baseline_2, "pred_agree"), mean(baseline_2, "kl")
    candidate_agreement, candidate_kl = mean(candidate_2, "pred_agree"), mean(candidate_2, "kl")
    selected = not collapsed and candidate_agreement >= baseline_agreement + 0.005 and candidate_kl <= baseline_kl + 0.005
    decision = {
        "schema": "sae-redo-level2-selection-v1", "candidate_eligible": not bool(collapsed),
        "selected": "context_vector" if selected else "no_candidate_promoted",
        "baseline_2pct_mean_pred_agree": baseline_agreement, "baseline_2pct_mean_kl": baseline_kl,
        "candidate_2pct_mean_pred_agree": candidate_agreement, "candidate_2pct_mean_kl": candidate_kl,
        "collapsed_candidate_runs": [str(path) for path, _ in collapsed],
    }
    (out / "selection.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps(decision, indent=2))
    if args.run_test and selected:
        source_run = next(path for path, result in candidate_2 if result["seed"] == 0)
        command = [sys.executable, str(ROOT / "scripts" / "evaluate_sae_redo_test.py"), "--source-run", str(source_run),
                   "--out", str(out / "test_confirmation"), "--device", args.device]
        subprocess.run(command, check=True)
    elif args.run_test:
        print("No candidate promoted; no test confirmation launched.")


if __name__ == "__main__":
    main()

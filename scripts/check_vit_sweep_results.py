"""Post-run consistency checks for a completed ViT sweep."""
import argparse
import glob
import json
import math
import os
from collections import Counter


def parse_int_list(text):
    return [int(x) for x in text.split(",") if x]


def parse_float_list(text):
    return [float(x) for x in text.split(",") if x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_root", default="runs/vit_sweep")
    ap.add_argument("--blocks", default="2,4,6,8,10")
    ap.add_argument("--fracs", default="0.01,0.02,0.05,0.08,0.12")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()

    result_paths = sorted(glob.glob(os.path.join(args.sweep_root, "*", "vit_result.json")))
    print(f"result_files={len(result_paths)}")

    blocks = parse_int_list(args.blocks)
    fracs = parse_float_list(args.fracs)
    seeds = parse_int_list(args.seeds)

    found = Counter()
    bad_metrics = []
    rows = []
    for path in result_paths:
        with open(path) as f:
            data = json.load(f)
        rows.append(data)
        key = (data["block"], round(data["frac_retained"], 2), data["seed"])
        found[key] += 1
        checks = {
            "pred_agree_range": 0.0 <= data["pred_agree"] <= 1.0,
            "frac_range": 0.0 <= data["frac_retained"] <= 1.0,
            "kl_nonneg": data["kl"] >= 0.0,
            "finite_rel_l2": math.isfinite(data["rel_l2"]),
            "finite_fvu": math.isfinite(data["fvu"]),
            "finite_cos": math.isfinite(data["cos"]),
        }
        if not all(checks.values()):
            bad_metrics.append((path, checks))

    missing = []
    duplicates = []
    for block in blocks:
        for frac in fracs:
            for seed in seeds:
                key = (block, round(frac, 2), seed)
                count = found.get(key, 0)
                if count == 0:
                    missing.append(key)
                elif count > 1:
                    duplicates.append((key, count))

    print(f"expected_jobs={len(blocks) * len(fracs) * len(seeds)}")
    print(f"missing_jobs={len(missing)}")
    print(f"duplicate_jobs={len(duplicates)}")
    print(f"bad_metric_files={len(bad_metrics)}")

    if rows:
        for metric in ["pred_agree", "kl", "rel_l2", "fvu"]:
            vals = [r[metric] for r in rows]
            print(
                f"{metric}: min={min(vals):.6f} max={max(vals):.6f} mean={sum(vals)/len(vals):.6f}"
            )

    if missing:
        print("\nMissing jobs:")
        for item in missing:
            print(item)
    if duplicates:
        print("\nDuplicate jobs:")
        for item in duplicates:
            print(item)
    if bad_metrics:
        print("\nBad metric files:")
        for path, checks in bad_metrics:
            print(path, checks)


if __name__ == "__main__":
    main()

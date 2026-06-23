"""Summarize seed stability for a completed ViT sweep."""
import argparse
import glob
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_root", default="runs/vit_sweep")
    ap.add_argument("--sort_by", default="pred_agree", choices=["pred_agree", "kl"])
    args = ap.parse_args()

    by_pair = defaultdict(list)
    for path in glob.glob(f"{args.sweep_root}/*/vit_result.json"):
        with open(path) as f:
            data = json.load(f)
        frac = round(data["frac_retained"], 2)
        by_pair[(data["block"], frac)].append(
            {
                "seed": data["seed"],
                "pred_agree": data["pred_agree"],
                "kl": data["kl"],
                "rel_l2": data["rel_l2"],
                "fvu": data["fvu"],
                "path": path,
            }
        )

    rows = []
    for (block, frac), vals in sorted(by_pair.items()):
        vals = sorted(vals, key=lambda x: x["seed"])
        pred_agree = [v["pred_agree"] for v in vals]
        kl = [v["kl"] for v in vals]
        rel_l2 = [v["rel_l2"] for v in vals]
        fvu = [v["fvu"] for v in vals]

        mean_pa = sum(pred_agree) / len(pred_agree)
        std_pa = (sum((x - mean_pa) ** 2 for x in pred_agree) / len(pred_agree)) ** 0.5
        mean_kl = sum(kl) / len(kl)
        mean_rel_l2 = sum(rel_l2) / len(rel_l2)
        mean_fvu = sum(fvu) / len(fvu)
        rows.append(
            {
                "block": block,
                "frac": frac,
                "mean_pred_agree": mean_pa,
                "std_pred_agree": std_pa,
                "mean_kl": mean_kl,
                "mean_rel_l2": mean_rel_l2,
                "mean_fvu": mean_fvu,
                "seed_values": pred_agree,
            }
        )

    if args.sort_by == "pred_agree":
        rows.sort(key=lambda r: (-r["mean_pred_agree"], r["std_pred_agree"], r["mean_kl"]))
        print("Ranked by mean pred_agree, then lower std, then lower kl\n")
    else:
        rows.sort(key=lambda r: (r["mean_kl"], -r["mean_pred_agree"], r["std_pred_agree"]))
        print("Ranked by mean kl, then higher pred_agree, then lower std\n")

    for row in rows:
        print(
            f"block={row['block']:2d} frac={row['frac']:.2f} "
            f"mean_pa={row['mean_pred_agree']:.4f} std_pa={row['std_pred_agree']:.4f} "
            f"mean_kl={row['mean_kl']:.4f} mean_rel_l2={row['mean_rel_l2']:.4f} "
            f"mean_fvu={row['mean_fvu']:.4f} "
            f"seeds={[round(x, 4) for x in row['seed_values']]}"
        )


if __name__ == "__main__":
    main()

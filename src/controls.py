"""Baselines: PCA reconstruction at matched coefficient budget per section.

Random-dictionary SAE baseline is provided by train_sae.py --random.
PCA gives the linear transform-domain compression reference: keep the top-r
principal components (global, per-image energy) at a budget matched to the SAE's
retained-coefficient count, splice back, measure downstream preservation.
"""
import argparse, os, sys, json
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
import backbone as B, eval as E
from cache_activations import load_backbone, SECTIONS


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="./runs/acts_r56_c100")
    ap.add_argument("--ckpt", default="./runs/backbone_r56_c100/best.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="./runs/baselines_pca.json")
    args = ap.parse_args()
    dev = args.device
    model, _ = load_backbone(args.ckpt, dev)
    stats = json.load(open(os.path.join(args.acts, "norm_stats.json")))
    results = {}
    for sec in SECTIONS:
        C, sp = B.SECTION_INFO[sec]
        mean = torch.tensor(stats[sec]["mean"], device=dev).view(1, -1, 1, 1)
        std = torch.tensor(stats[sec]["std"], device=dev).view(1, -1, 1, 1)
        Htr = torch.load(os.path.join(args.acts, f"train_{sec}.pt")).float().to(dev)
        Hte = torch.load(os.path.join(args.acts, f"test_{sec}.pt")).float().to(dev)
        labels = torch.load(os.path.join(args.acts, "test_labels.pt")).to(dev)
        orig = torch.load(os.path.join(args.acts, "test_logits.pt")).to(dev)
        # PCA over per-location channel vectors (treat each spatial site as a sample)
        Xtr = ((Htr - mean) / std).permute(0, 2, 3, 1).reshape(-1, C)
        Xtr = Xtr - Xtr.mean(0, keepdim=True)
        U, Sv, Vt = torch.linalg.svd(Xtr[:200000], full_matrices=False)  # subsample for speed
        results[sec] = {}
        for r in [max(1, C // 8), max(1, C // 4), C // 2]:
            Xte = ((Hte - mean) / std).permute(0, 2, 3, 1).reshape(-1, C)
            mu = Xte.mean(0, keepdim=True)
            P = Vt[:r].T  # (C,r)
            Xrec = (Xte - mu) @ P @ P.T + mu
            Hhat = Xrec.reshape(Hte.shape[0], sp, sp, C).permute(0, 3, 1, 2)
            rec = E.recon_metrics((Hte - mean) / std, Hhat)
            Hhat_raw = Hhat * std + mean
            d = E.downstream_metrics(model, sec, Hhat_raw, labels, orig)
            results[sec][f"r{r}"] = {**rec, **d, "rank": r, "frac_of_C": r / C}
        print(f"PCA {sec}: " + " ".join(f"r{k.split('r')[1]} top1={v['spliced_top1']:.3f}" for k, v in results[sec].items()), flush=True)
    json.dump(results, open(args.out, "w"))
    print("PCA BASELINE DONE", args.out, flush=True)


if __name__ == "__main__":
    main()

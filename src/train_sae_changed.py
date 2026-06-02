"""Changed layerwise SAE trainer for isolated experiments."""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import backbone as B
import data as D  # noqa: F401
import eval as E
import masks_changed as M
import sae_changed as S
from cache_activations import SECTIONS, load_backbone


class Acts(torch.utils.data.Dataset):
    def __init__(self, acts_dir, split, section):
        self.H = torch.load(os.path.join(acts_dir, f"{split}_{section}.pt"))
        self.logits = torch.load(os.path.join(acts_dir, f"{split}_logits.pt"))
        self.labels = torch.load(os.path.join(acts_dir, f"{split}_labels.pt"))

    def __len__(self):
        return self.H.shape[0]

    def __getitem__(self, i):
        return self.H[i].float(), self.logits[i], self.labels[i]


def load_norm(acts_dir, section, device):
    st = json.load(open(os.path.join(acts_dir, "norm_stats.json")))[section]
    mean = torch.tensor(st["mean"], device=device).view(1, -1, 1, 1)
    std = torch.tensor(st["std"], device=device).view(1, -1, 1, 1)
    return mean, std


def recon_loss(H, Hhat, alpha=0.1):
    mse = (H - Hhat).pow(2).sum() / (H.pow(2).sum() + 1e-8)
    cos = 1 - F.cosine_similarity(H.flatten(1), Hhat.flatten(1), dim=1).mean()
    return mse + alpha * cos


SECTION_BASE_KERNEL = {
    "Q1": 3,
    "Q2": 3,
    "Q3": 3,
    "Q4": 3,
    "Q5": 3,
}

DEPTH_SCHEDULES = {
    "schedule_a": {"Q1": 2.0, "Q2": 2.0, "Q3": 1.0, "Q4": 0.5, "Q5": 0.5},
    "schedule_b": {"Q1": 1.0, "Q2": 1.0, "Q3": 1.0, "Q4": 0.5, "Q5": 0.5},
    "schedule_c": {"Q1": 1.0, "Q2": 1.0, "Q3": 1.0, "Q4": 0.25, "Q5": 0.25},
    "schedule_d": {"Q1": 2.0, "Q2": 2.0, "Q3": 1.0, "Q4": 1.0, "Q5": 1.0},
}


def odd_patch_size(base_kernel, multiplier):
    raw = base_kernel * multiplier
    realized = max(1, int(round(raw)))
    if realized % 2 == 0:
        realized = realized - 1 if raw < realized else realized + 1
    return max(1, realized)


def patch_settings(args):
    base_kernel = SECTION_BASE_KERNEL[args.section]
    schedule_id = args.depth_schedule or ""
    if args.depth_schedule:
        multiplier = DEPTH_SCHEDULES[args.depth_schedule][args.section]
    else:
        multiplier = args.patch_multiplier
    patch_size = odd_patch_size(base_kernel, multiplier)
    patch_stride = 1 if args.patch_mode == "overlap" else patch_size
    return base_kernel, multiplier, patch_size, patch_stride, schedule_id


def mask_for_stage(cfg, target_hw, frac, k_rf):
    if cfg["mask"] == "global":
        return {"type": "global", "frac": frac, "m": None}
    if cfg["mask"] == "rf":
        return {"type": "rf", "k_rf": k_rf, "target_hw": target_hw, "dilate": cfg.get("dilate", 0)}
    if cfg["mask"] == "rf_frac":
        return {"type": "rf_frac", "rf_frac": frac, "target_hw": target_hw, "dilate": cfg.get("dilate", 0)}
    return {"type": "none"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="./runs/acts_r56_c100")
    ap.add_argument("--ckpt", default="./runs/backbone_r56_c100/best.pt")
    ap.add_argument("--section", required=True)
    ap.add_argument("--sae_type", default="vector", choices=["vector", "field", "patchvec"])
    ap.add_argument("--mask", default="global", choices=["global", "rf", "rf_frac", "none"])
    ap.add_argument("--Kmult", type=int, default=8)
    ap.add_argument("--target_frac", type=float, default=0.05)
    ap.add_argument("--rf_frac", type=float, default=0.05)
    ap.add_argument("--k_rf", type=int, default=4)
    ap.add_argument("--n_blocks", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--patch_mode", default="overlap", choices=["overlap", "disjoint"])
    ap.add_argument("--patch_multiplier", type=float, default=1.0)
    ap.add_argument("--depth_schedule", choices=sorted(DEPTH_SCHEDULES))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = args.device
    cfg = vars(args).copy()

    base_kernel, patch_multiplier, patch_size, patch_stride, schedule_id = patch_settings(args)
    cfg.update({
        "section_base_kernel": base_kernel,
        "resolved_patch_multiplier": patch_multiplier,
        "resolved_patch_size": patch_size,
        "resolved_patch_stride": patch_stride,
        "resolved_depth_schedule": schedule_id,
    })

    C, sp = B.SECTION_INFO[args.section]
    K = args.Kmult * C
    next_idx = SECTIONS.index(args.section) + 1
    target_hw = (B.SECTION_INFO[SECTIONS[next_idx]][1],) * 2 if next_idx < len(SECTIONS) else (1, 1)

    tr = torch.utils.data.DataLoader(
        Acts(args.acts, "train", args.section),
        batch_size=args.bs,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    te = torch.utils.data.DataLoader(
        Acts(args.acts, "test", args.section),
        batch_size=args.bs,
        num_workers=2,
        pin_memory=True,
    )
    mean, std = load_norm(args.acts, args.section, dev)
    model, _ = load_backbone(args.ckpt, dev)

    build_kwargs = {}
    if args.sae_type == "patchvec":
        build_kwargs.update({
            "patch_size": patch_size,
            "patch_mode": args.patch_mode,
            "patch_stride": patch_stride,
        })
    net = S.build_sae(args.sae_type, C, K, n_blocks=args.n_blocks, **build_kwargs).to(dev)
    if args.random:
        for p in net.parameters():
            p.requires_grad_(False)
        if hasattr(net, "dec_bias"):
            net.dec_bias.requires_grad_(True)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=args.lr)

    e_warm = max(1, args.epochs // 4)
    e_anneal_end = args.epochs - max(1, args.epochs // 4)
    log = []
    for ep in range(args.epochs):
        if ep < e_warm:
            mcfg = {"type": "none"}
        else:
            t = min(1.0, (ep - e_warm) / max(1, e_anneal_end - e_warm))
            frac = 0.5 * (1 - t) + args.target_frac * t
            if args.mask == "rf_frac":
                frac = 0.5 * (1 - t) + args.rf_frac * t
            k_rf = int(round((sp * sp * 0.5) * (1 - t) + args.k_rf * t)) if args.mask == "rf" else args.k_rf
            mcfg = mask_for_stage(cfg, target_hw, frac, max(1, k_rf))
        net.train()
        t0 = time.time()
        rl = 0.0
        for H, _, _ in tr:
            H = H.to(dev)
            Hn = (H - mean) / std
            z = net.encode(Hn)
            zt, _ = M.apply_mask(z, mcfg)
            Hhat = net.decode(zt)
            loss = recon_loss(Hn, Hhat)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if hasattr(net, "_normalize_decoder"):
                net._normalize_decoder()
            rl += loss.item() * H.size(0)
        met = evaluate(net, te, mcfg, mean, std, model, args.section, dev)
        met.update({"epoch": ep, "train_loss": rl / len(tr.dataset), "sec_per_ep": time.time() - t0})
        log.append(met)
        print(
            f"[{args.section} {args.sae_type} {args.mask}] ep{ep} loss{met['train_loss']:.3f} "
            f"relL2 {met['rel_l2']:.3f} cos {met['cos']:.3f} top1 {met['spliced_top1']:.3f} "
            f"KL {met['kl_orig_spliced']:.3f} frac {met['frac_retained']:.4f}",
            flush=True,
        )
    json.dump({"config": cfg, "log": log, "final": log[-1]}, open(os.path.join(args.out, "result.json"), "w"))
    torch.save({"state_dict": net.state_dict(), "config": cfg, "C": C, "K": K}, os.path.join(args.out, "sae.pt"))
    print("SAE CHANGED DONE", os.path.join(args.out, "result.json"), flush=True)


@torch.no_grad()
def evaluate(net, loader, mcfg, mean, std, model, section, dev):
    net.eval()
    agg = {}
    n = 0
    for H, logits, labels in loader:
        H = H.to(dev)
        logits = logits.to(dev)
        labels = labels.to(dev)
        Hn = (H - mean) / std
        z = net.encode(Hn)
        zt, keep = M.apply_mask(z, mcfg)
        Hhat = net.decode(zt)
        Hhat_raw = Hhat * std + mean
        r = E.recon_metrics(Hn, Hhat)
        s = E.sparsity_metrics(keep)
        d = E.downstream_metrics(model, section, Hhat_raw, labels, logits)
        b = H.size(0)
        n += b
        for dd in (r, s, d):
            for k, v in dd.items():
                agg[k] = agg.get(k, 0.0) + v * b
    return {k: v / n for k, v in agg.items()}


if __name__ == "__main__":
    main()

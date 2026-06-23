"""T3b — transfer a sparse autoencoder to a pretrained ViT on Imagenette.

Extract patch-token activations after a chosen transformer block, reshape the
196 tokens to a 14x14 grid, train an SAE + global mask, then splice the
reconstructed tokens back and continue the ViT. Downstream metric = agreement /
KL vs the ORIGINAL ViT prediction (self-consistency; no task labels needed).
"""
import argparse, os, sys, json, random
import torch, torch.nn as nn, torch.nn.functional as F
import timm
from torchvision.datasets import Imagenette
from torchvision.datasets import ImageFolder
sys.path.insert(0, os.path.dirname(__file__))
import sae as S, masks as M, eval as E


class ViTWrap:
    def __init__(self, name, device):
        self.m = timm.create_model(name, pretrained=True).to(device).eval()
        for p in self.m.parameters(): p.requires_grad_(False)
        self.device = device
        cfg = timm.data.resolve_data_config({}, model=self.m)
        self.tf = timm.data.create_transform(**cfg)
        self.D = self.m.embed_dim; self.grid = int((self.m.patch_embed.num_patches) ** 0.5)

    @torch.no_grad()
    def tokens_at(self, x, L):
        """Run to end of block L; return (all_tokens, prefix_len)."""
        m = self.m
        x = m.patch_embed(x); x = m._pos_embed(x); x = m.patch_drop(x); x = m.norm_pre(x)
        for i, blk in enumerate(m.blocks):
            x = blk(x)
            if i == L: return x
        return x

    @torch.no_grad()
    def head_from(self, x, L, patch_tokens=None):
        """Continue from end of block L (optionally replacing patch tokens) -> logits."""
        m = self.m
        npref = x.shape[1] - self.grid * self.grid
        if patch_tokens is not None:
            x = torch.cat([x[:, :npref], patch_tokens], dim=1)
        for blk in list(m.blocks)[L + 1:]:
            x = blk(x)
        x = m.norm(x)
        return m.forward_head(x)


def to_grid(tok, npref, grid): return tok[:, npref:].transpose(1, 2).reshape(tok.shape[0], -1, grid, grid)
def to_tokens(g): return g.flatten(2).transpose(1, 2)


def parse_device(device):
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device)
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def subset_indices(n_total, limit, seed):
    if limit < 0:
        return list(range(n_total))
    limit = min(limit, n_total)
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(n_total, generator=g)[:limit].tolist()


def load_dataset(args, transform):
    dataset = args.dataset
    if dataset == "imagenette":
        root = args.dataset_root or "./data"
        size = args.imagenette_size
        try:
            train_ds = Imagenette(root, split="train", size=size, download=True, transform=transform)
        except Exception:
            train_ds = Imagenette(root, split="train", size=size, download=False, transform=transform)
        val_ds = Imagenette(root, split="val", size=size, download=False, transform=transform)
        return train_ds, val_ds, {"dataset": dataset, "dataset_root": root, "train_dir": None, "val_dir": None}

    if dataset == "imagefolder":
        if args.train_dir and args.val_dir:
            train_dir = args.train_dir
            val_dir = args.val_dir
        elif args.dataset_root:
            train_dir = os.path.join(args.dataset_root, args.train_split)
            val_dir = os.path.join(args.dataset_root, args.val_split)
        else:
            raise ValueError("imagefolder dataset requires --dataset_root or both --train_dir and --val_dir")
        train_ds = ImageFolder(train_dir, transform=transform)
        val_ds = ImageFolder(val_dir, transform=transform)
        return train_ds, val_ds, {
            "dataset": dataset,
            "dataset_root": args.dataset_root,
            "train_dir": train_dir,
            "val_dir": val_dir,
        }

    raise ValueError(dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="imagenette", choices=["imagenette", "imagefolder"])
    ap.add_argument("--dataset_root", default=None)
    ap.add_argument("--train_dir", default=None)
    ap.add_argument("--val_dir", default=None)
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--val_split", default="val")
    ap.add_argument("--imagenette_size", default="160px")
    ap.add_argument("--model", default="vit_small_patch16_224")
    ap.add_argument("--block", type=int, default=6)
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--sae_type", default="field", choices=["field", "vector"])
    ap.add_argument("--Kmult", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--ntrain", type=int, default=2000)
    ap.add_argument("--nval", type=int, default=500)
    ap.add_argument("--train_bs", type=int, default=32)
    ap.add_argument("--val_bs", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="./runs/vit_sae")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)
    dev = parse_device(args.device)
    vit = ViTWrap(args.model, dev); grid = vit.grid; D = vit.D
    ds_tr, ds_va, ds_meta = load_dataset(args, vit.tf)
    import torch.utils.data as tud
    tr_idx = subset_indices(len(ds_tr), args.ntrain, args.data_seed)
    va_idx = subset_indices(len(ds_va), args.nval, args.data_seed)
    tr = tud.DataLoader(
        tud.Subset(ds_tr, tr_idx),
        batch_size=args.train_bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=dev.type == "cuda",
    )
    va = tud.DataLoader(
        tud.Subset(ds_va, va_idx),
        batch_size=args.val_bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=dev.type == "cuda",
    )

    # cache block-L tokens (train) for SAE training + per-section norm
    feats, npref = [], None
    for x, _ in tr:
        t = vit.tokens_at(x.to(dev), args.block); feats.append(t.cpu())
        npref = t.shape[1] - grid * grid
    Tr = torch.cat(feats)  # (N, npref+196, D)
    gtr = to_grid(Tr.to(dev), npref, grid)
    mean = gtr.mean(dim=(0, 2, 3), keepdim=True); std = gtr.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)

    K = args.Kmult * D
    build_kwargs = {"n_blocks": 2}
    if args.sae_type == "field":
        build_kwargs["d"] = 2 * D
    net = S.build_sae(args.sae_type, D, K, **build_kwargs).to(dev)
    opt = torch.optim.Adam(net.parameters(), args.lr)
    mask = lambda z: M.mask_global_frac(z, args.frac)
    idx = torch.randperm(gtr.shape[0], device=dev)
    for ep in range(args.epochs):
        net.train(); rl = 0
        for b in range(0, gtr.shape[0], 64):
            g = gtr[idx[b:b+64]]; gn = (g - mean) / std
            z = net.encode(gn); zt, _ = mask(z); gh = net.decode(zt)
            loss = (gn - gh).pow(2).sum() / (gn.pow(2).sum() + 1e-8)
            opt.zero_grad(); loss.backward(); opt.step(); rl += loss.item()
            if args.sae_type == "vector" and hasattr(net, "_normalize_decoder"):
                net._normalize_decoder()
    # eval on val: recon + agreement/KL vs original ViT
    net.eval(); agg = {}; n = 0
    with torch.no_grad():
        for x, _ in va:
            x = x.to(dev); t = vit.tokens_at(x, args.block)
            g = to_grid(t, npref, grid); gn = (g - mean) / std
            z = net.encode(gn); zt, keep = mask(z); gh = net.decode(zt)
            g_raw = gh * std + mean
            rec = E.recon_metrics(gn, gh)
            orig_logits = vit.head_from(t, args.block)
            spliced_logits = vit.head_from(t, args.block, patch_tokens=to_tokens(g_raw))
            agree = (orig_logits.argmax(1) == spliced_logits.argmax(1)).float().mean().item()
            kl = F.kl_div(F.log_softmax(spliced_logits, 1), F.softmax(orig_logits, 1), reduction="batchmean").item()
            frac = keep.float().mean().item(); b = x.size(0); n += b
            for k, v in {**rec, "pred_agree": agree, "kl": kl, "frac_retained": frac}.items():
                agg[k] = agg.get(k, 0.0) + v * b
    res = {k: v / n for k, v in agg.items()}
    res.update({
        "model": args.model,
        "block": args.block,
        "sae_type": args.sae_type,
        "D": D,
        "K": K,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "ntrain": len(tr_idx),
        "nval": len(va_idx),
        "epochs": args.epochs,
        "lr": args.lr,
        "device": str(dev),
        "dataset": ds_meta["dataset"],
        "dataset_root": ds_meta["dataset_root"],
        "train_dir": ds_meta["train_dir"],
        "val_dir": ds_meta["val_dir"],
    })
    json.dump(res, open(f"{args.out}/vit_result.json", "w"))
    torch.save({
        "state_dict": net.state_dict(),
        "config": vars(args),
        "model": args.model,
        "block": args.block,
        "D": D,
        "K": K,
        "mean": mean.cpu(),
        "std": std.cpu(),
        "grid": grid,
        "seed": args.seed,
        "data_seed": args.data_seed,
    }, os.path.join(args.out, "sae.pt"))
    print("VIT-SAE:", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()}), flush=True)
    print("VIT DONE", flush=True)


if __name__ == "__main__":
    main()

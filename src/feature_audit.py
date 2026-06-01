"""T2.3 — feature-quality audit for a trained SAE.

Quantitative: dead-feature fraction, near-duplicate fraction (decoder-atom cosine),
active-features-per-image, active-locations-per-feature.
Qualitative: top-activating-image montage for a few high-firing features.

Decoder "atom" for feature k = impulse response: decode a code that is zero except
feature k = 1 at the field center, crop the central CxPxP patch, flatten. Works for
both the linear Vector SAE and the convolutional Field SAE.
"""
import argparse, os, sys, json
import torch, torch.nn.functional as F
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import backbone as B, masks as M, data as D
from train_transition import load_sae, norm
from cache_activations import SECTIONS


@torch.no_grad()
def decoder_atoms(sae, K, C, sp, dev, patch=3):
    """Impulse responses (bias-subtracted) -> (K, C*patch*patch) atom matrix.
    atom_k = decode(impulse_k) - decode(0): isolates feature k's contribution,
    removing the decoder's constant bias/normalization offset (else all atoms
    look identical)."""
    cc = sp // 2; r = patch // 2
    base = sae.decode(torch.zeros(1, K, sp, sp, device=dev))[:, :, cc-r:cc+r+1, cc-r:cc+r+1].reshape(1, -1)
    atoms = []
    for s in range(0, K, 64):
        ks = list(range(s, min(s + 64, K)))
        z = torch.zeros(len(ks), K, sp, sp, device=dev)
        for i, k in enumerate(ks): z[i, k, cc, cc] = 1.0
        out = sae.decode(z)[:, :, cc-r:cc+r+1, cc-r:cc+r+1].reshape(len(ks), -1) - base
        atoms.append(out)
    return torch.cat(atoms)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="./runs/acts_r56_c100")
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--section", default="Q3")
    ap.add_argument("--sae", required=True)
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="./runs/audit")
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True); dev = args.device
    sae, ck = load_sae(args.sae, dev); K, C = ck["K"], ck["C"]
    C_, sp = B.SECTION_INFO[args.section]
    mean, std = norm(args.acts, args.section, dev)
    H = torch.load(f"{args.acts}/test_{args.section}.pt")

    # activation stats over test set (post-mask)
    active_feat = torch.zeros(K, device=dev)         # times feature active anywhere
    loc_per_feat = torch.zeros(K, device=dev)        # total active locations
    feats_per_img = []
    for b in range(0, H.shape[0], 256):
        Hn = (H[b:b+256].float().to(dev) - mean) / std
        z = sae.encode(Hn); zt, keep = M.mask_global_frac(z, args.frac)
        k = keep.float()
        active_feat += k.amax(dim=(2, 3)).sum(0)
        loc_per_feat += k.sum(dim=(0, 2, 3))
        feats_per_img.append((k.amax(dim=(2, 3)).sum(1)).cpu())
    N = H.shape[0]
    dead = (active_feat == 0).float().mean().item()
    feats_per_img = torch.cat(feats_per_img)

    # duplicate detection via decoder-atom cosine
    A = decoder_atoms(sae, K, C, sp, dev)
    A = A / (A.norm(dim=1, keepdim=True) + 1e-8)
    sim = A @ A.T; sim.fill_diagonal_(-1)
    max_off = sim.max(dim=1).values
    dup = (max_off > 0.9).float().mean().item()

    stats = {
        "section": args.section, "K": K, "dead_frac": dead, "duplicate_frac_cos>0.9": dup,
        "mean_active_features_per_img": feats_per_img.mean().item(),
        "median_active_features_per_img": feats_per_img.median().item(),
        "mean_locations_per_active_feature": (loc_per_feat[active_feat > 0] / active_feat[active_feat > 0].clamp_min(1)).mean().item(),
        "mean_atom_max_cosine": max_off.mean().item(),
    }
    json.dump(stats, open(f"{args.out}/audit_{args.section}.json", "w"))
    print(json.dumps(stats), flush=True)

    # top-activating-image montage for 6 most-firing features
    top_features = active_feat.topk(6).indices.tolist()
    _, test = D.get_loaders(args.dataset, root="./data", batch_size=256, augment=False)
    imgs = torch.cat([x for x, _ in test])  # normalized; we'll just min-max per image for display
    # recompute per-image max activation for the chosen features
    fig, axes = plt.subplots(6, 5, figsize=(7, 8))
    for r, k in enumerate(top_features):
        acts_k = []
        for b in range(0, H.shape[0], 256):
            Hn = (H[b:b+256].float().to(dev) - mean) / std
            z = sae.encode(Hn); zt, _ = M.mask_global_frac(z, args.frac)
            acts_k.append(zt[:, k].amax(dim=(1, 2)).cpu())
        acts_k = torch.cat(acts_k); top5 = acts_k.topk(5).indices
        for c, idx in enumerate(top5):
            im = imgs[idx].permute(1, 2, 0).numpy(); im = (im - im.min()) / (np.ptp(im) + 1e-8)
            axes[r, c].imshow(im); axes[r, c].axis("off")
            if c == 0: axes[r, c].set_ylabel(f"feat {k}", fontsize=8)
    fig.suptitle(f"Top-activating images, {args.section} (Field SAE)", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{args.out}/topimg_{args.section}.png", dpi=110); plt.close(fig)
    print("AUDIT DONE", args.section, flush=True)


if __name__ == "__main__":
    main()

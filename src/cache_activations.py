"""Cache frozen-backbone activations H_t and per-channel norm stats (Phase 0)."""
import argparse, os, sys, json
import torch
sys.path.insert(0, os.path.dirname(__file__))
import data as D
import backbone as B

SECTIONS = ["Q1", "Q2", "Q3", "Q4", "Q5"]


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


def compute_norm_stats(out_dir):
    stats = {}
    for q in SECTIONS:
        H = torch.load(os.path.join(out_dir, f"train_{q}.pt")).float()  # (N,C,H,W)
        mean = H.mean(dim=(0, 2, 3))
        std = H.std(dim=(0, 2, 3)).clamp_min(1e-6)
        stats[q] = {"mean": mean.tolist(), "std": std.tolist()}
    with open(os.path.join(out_dir, "norm_stats.json"), "w") as f:
        json.dump(stats, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="./runs/backbone_r56_c100/best.pt")
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--out", default="./runs/acts_r56_c100")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    model, ck = load_backbone(args.ckpt, args.device)
    # no augmentation -> deterministic activations
    train_loader, test_loader = D.get_loaders(args.dataset, root="./data", batch_size=256, augment=False)
    cache_split(model, train_loader, args.device, args.out, "train")
    cache_split(model, test_loader, args.device, args.out, "test")
    compute_norm_stats(args.out)
    print(f"CACHE DONE -> {args.out} (backbone acc {ck['acc']:.2f})", flush=True)


if __name__ == "__main__":
    main()

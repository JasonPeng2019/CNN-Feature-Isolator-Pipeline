"""Phase 3 Family I — learn one adjacent transition T_t: Z~_t -> Z^_{t+1}.

Loads frozen SAEs at t and t+1, trains the predictor against the independently
learned code Z_{t+1} + decoded reconstruction of H_{t+1}, and reports:
  - code prediction error, decoded recon (relL2/cos)
  - downstream preservation (splice D_{t+1}(Z^_{t+1}) -> frozen tail)
  - N4 frozen-block-hybrid baseline (causal upper bound, no learned transition)
"""
import argparse, os, sys, json, time
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
import backbone as B, sae as S, masks as M, eval as E, transitions as T
from cache_activations import load_backbone, SECTIONS


def load_sae(path, device):
    ck = torch.load(path, map_location=device)
    c = ck["config"]
    net = S.build_sae(c["sae_type"], ck["C"], ck["K"], n_blocks=c.get("n_blocks", 3))
    net.load_state_dict(ck["state_dict"]); net.to(device).eval()
    for p in net.parameters(): p.requires_grad_(False)
    return net, ck


class PairActs(torch.utils.data.Dataset):
    def __init__(self, acts, split, st, st1):
        self.Ht = torch.load(os.path.join(acts, f"{split}_{st}.pt"))
        self.Ht1 = torch.load(os.path.join(acts, f"{split}_{st1}.pt"))
        self.logits = torch.load(os.path.join(acts, f"{split}_logits.pt"))
        self.labels = torch.load(os.path.join(acts, f"{split}_labels.pt"))

    def __len__(self): return self.Ht.shape[0]

    def __getitem__(self, i):
        return self.Ht[i].float(), self.Ht1[i].float(), self.logits[i], self.labels[i]


def norm(acts, sec, dev):
    st = json.load(open(os.path.join(acts, "norm_stats.json")))[sec]
    return (torch.tensor(st["mean"], device=dev).view(1, -1, 1, 1),
            torch.tensor(st["std"], device=dev).view(1, -1, 1, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="./runs/acts_r56_c100")
    ap.add_argument("--ckpt", default="./runs/backbone_r56_c100/best.pt")
    ap.add_argument("--src", required=True)       # e.g. Q2
    ap.add_argument("--dst", required=True)       # e.g. Q3
    ap.add_argument("--sae_src", required=True)   # path to sae.pt at src
    ap.add_argument("--sae_dst", required=True)
    ap.add_argument("--frac", type=float, default=0.05)   # mask budget at src
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--depth", type=int, default=1)        # predictor hidden depth (T1.3)
    ap.add_argument("--hidden", type=int, default=0)       # 0 -> auto
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True); dev = args.device

    model, _ = load_backbone(args.ckpt, dev)
    sae_t, ck_t = load_sae(args.sae_src, dev); sae_t1, ck_t1 = load_sae(args.sae_dst, dev)
    mean_t, std_t = norm(args.acts, args.src, dev); mean_t1, std_t1 = norm(args.acts, args.dst, dev)
    Cs, sp_s = B.SECTION_INFO[args.src]; Cd, sp_d = B.SECTION_INFO[args.dst]
    Kt, Kt1 = ck_t["K"], ck_t1["K"]
    mask_fn = lambda z: M.mask_global_frac(z, args.frac)
    pred = T.TransitionPredictor(Kt, Kt1, sp_s, sp_d, hidden=(args.hidden or None), depth=args.depth).to(dev)
    opt = torch.optim.Adam(pred.parameters(), lr=args.lr)

    tr = torch.utils.data.DataLoader(PairActs(args.acts, "train", args.src, args.dst), batch_size=args.bs, shuffle=True, num_workers=2, pin_memory=True)
    te = torch.utils.data.DataLoader(PairActs(args.acts, "test", args.src, args.dst), batch_size=args.bs, num_workers=2, pin_memory=True)

    log = []
    for ep in range(args.epochs):
        pred.train(); t0 = time.time(); rl = 0.0
        for Ht, Ht1, _, _ in tr:
            Ht, Ht1 = Ht.to(dev), Ht1.to(dev)
            Hn_t, Hn_t1 = (Ht - mean_t) / std_t, (Ht1 - mean_t1) / std_t1
            with torch.no_grad():
                z_t = sae_t.encode(Hn_t); zt_t, _ = mask_fn(z_t)
                z_t1_target = sae_t1.encode(Hn_t1)
            zhat = pred(zt_t)
            Hhat_t1 = sae_t1.decode(zhat)
            loss, _ = T.transition_loss(zhat, z_t1_target, Hhat_t1, Hn_t1, beta=1.0)
            opt.zero_grad(); loss.backward(); opt.step(); rl += loss.item() * Ht.size(0)
        met = evaluate(pred, sae_t, sae_t1, mask_fn, te, mean_t, std_t, mean_t1, std_t1, model, args.src, args.dst, dev)
        met.update({"epoch": ep, "train_loss": rl / len(tr.dataset), "sec": time.time() - t0})
        log.append(met)
        print(f"[{args.src}->{args.dst}] ep{ep} loss{met['train_loss']:.3f} codeMSE {met['code_mse']:.3f} "
              f"recRelL2 {met['rel_l2']:.3f} top1 {met['spliced_top1']:.3f} | HYBRID top1 {met['hybrid_top1']:.3f}", flush=True)
    json.dump({"config": vars(args), "log": log, "final": log[-1]}, open(os.path.join(args.out, "result.json"), "w"))
    torch.save({"state_dict": pred.state_dict(), "Kt": Kt, "Kt1": Kt1, "sp_s": sp_s, "sp_d": sp_d},
               os.path.join(args.out, "pred.pt"))
    print("TRANSITION DONE", args.out, flush=True)


@torch.no_grad()
def evaluate(pred, sae_t, sae_t1, mask_fn, loader, mean_t, std_t, mean_t1, std_t1, model, src, dst, dev):
    pred.eval(); agg = {}; n = 0
    for Ht, Ht1, logits, labels in loader:
        Ht, Ht1, logits, labels = Ht.to(dev), Ht1.to(dev), logits.to(dev), labels.to(dev)
        Hn_t, Hn_t1 = (Ht - mean_t) / std_t, (Ht1 - mean_t1) / std_t1
        z_t = sae_t.encode(Hn_t); zt_t, _ = mask_fn(z_t)
        z_t1_target = sae_t1.encode(Hn_t1)
        zhat = pred(zt_t); Hhat_t1 = sae_t1.decode(zhat)
        code_mse = (zhat - z_t1_target).pow(2).mean().item()
        rec = E.recon_metrics(Hn_t1, Hhat_t1)
        # downstream: splice learned-transition recon
        d = E.downstream_metrics(model, dst, Hhat_t1 * std_t1 + mean_t1, labels, logits)
        # N4 frozen-block hybrid baseline
        zhyb, _ = T.frozen_block_hybrid(sae_t, sae_t1, model, src, dst, Hn_t, mean_t, std_t, mean_t1, std_t1, mask_fn)
        Hhyb_raw = sae_t1.decode(zhyb) * std_t1 + mean_t1
        dh = E.downstream_metrics(model, dst, Hhyb_raw, labels, logits)
        b = Ht.size(0); n += b
        batch = {"code_mse": code_mse, **rec, **d, "hybrid_top1": dh["spliced_top1"],
                 "hybrid_kl": dh["kl_orig_spliced"]}
        for k, v in batch.items():
            agg[k] = agg.get(k, 0.0) + v * b
    return {k: v / n for k, v in agg.items()}


if __name__ == "__main__":
    main()

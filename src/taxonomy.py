"""T2.4 — transition taxonomy at Q4->Q5 (both 64x8x8, fair same-shape comparison).

Three ways to model t->t+1, matched sparsity (5% of the field), same eval
(splice predicted H_{t+1} into frozen classifier, measure top-1 / KL):
  (a) sparse-code transcoder (OURS): mask(E_t(H_t)) -> predict sparse Z_{t+1} -> D_{t+1}
  (b) dense transcoder: mask(E_t(H_t)) -> conv -> predict dense H_{t+1} directly
  (c) crosscoder: one shared masked code z=mask(E(H_t)); two decoders D_t,D_{t+1}
      trained to reconstruct BOTH layers; transition = D_{t+1}(z)
E_t for (a)/(b) is our frozen Field SAE @5%; (c) trains its own shared dictionary.
"""
import argparse, os, sys, json, time
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
import backbone as B, sae as S, masks as M, eval as E, transitions as T
from cache_activations import load_backbone, SECTIONS
from train_transition import load_sae, norm, PairActs

SRC, DST = "Q4", "Q5"


class DenseTranscoder(nn.Module):
    def __init__(self, Kt, C, sp):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(Kt, 4 * C, 1), nn.GELU(),
                                 nn.Conv2d(4 * C, 4 * C, 3, 1, 1), nn.GELU(),
                                 nn.Conv2d(4 * C, C, 1))

    def forward(self, z): return self.net(z)


class CrossCoder(nn.Module):
    """Shared dictionary; encode from H_t; decode to both layers."""
    def __init__(self, C, K):
        super().__init__()
        self.enc = nn.Conv2d(C, K, 1)
        self.dec_t = nn.Conv2d(K, C, 1, bias=False)
        self.dec_t1 = nn.Conv2d(K, C, 1, bias=False)
        self.bt = nn.Parameter(torch.zeros(1, C, 1, 1)); self.bt1 = nn.Parameter(torch.zeros(1, C, 1, 1))

    def encode(self, h): return F.relu(self.enc(h))
    def decode_t(self, z): return self.dec_t(z) + self.bt
    def decode_t1(self, z): return self.dec_t1(z) + self.bt1


def relrec(a, b): return (a - b).pow(2).sum() / (b.pow(2).sum() + 1e-8)


@torch.no_grad()
def splice_metrics(model, Hpred_raw, labels, orig):
    return E.downstream_metrics(model, DST, Hpred_raw, labels, orig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="./runs/acts_r56_c100")
    ap.add_argument("--ckpt", default="./runs/backbone_r56_c100/best.pt")
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="./runs/taxonomy")
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True); dev = args.device
    model, _ = load_backbone(args.ckpt, dev)
    sae_t, ck_t = load_sae(f"runs/phase2/E1_{SRC}_f0.05/sae.pt", dev)
    sae_t1, ck_t1 = load_sae(f"runs/phase2/E1_{DST}_f0.05/sae.pt", dev)
    C, sp = B.SECTION_INFO[SRC]; Kt = ck_t["K"]; K = 8 * C
    mt, st = norm(args.acts, SRC, dev); mt1, st1 = norm(args.acts, DST, dev)
    mask = lambda z: M.mask_global_frac(z, args.frac)[0]
    tr = torch.utils.data.DataLoader(PairActs(args.acts, "train", SRC, DST), batch_size=args.bs, shuffle=True, num_workers=2)
    te = torch.utils.data.DataLoader(PairActs(args.acts, "test", SRC, DST), batch_size=args.bs, num_workers=2)

    sparse = T.TransitionPredictor(Kt, ck_t1["K"], sp, sp).to(dev)
    dense = DenseTranscoder(Kt, C, sp).to(dev)
    cross = CrossCoder(C, K).to(dev)
    opts = {"sparse": torch.optim.Adam(sparse.parameters(), 2e-3),
            "dense": torch.optim.Adam(dense.parameters(), 2e-3),
            "cross": torch.optim.Adam(cross.parameters(), 2e-3)}

    for ep in range(args.epochs):
        for Ht, Ht1, _, _ in tr:
            Ht, Ht1 = Ht.to(dev), Ht1.to(dev); Hn, Hn1 = (Ht - mt) / st, (Ht1 - mt1) / st1
            with torch.no_grad():
                zt = mask(sae_t.encode(Hn)); zt1 = sae_t1.encode(Hn1)
            # (a) sparse-code transcoder
            zhat = sparse(zt); loss_a = (zhat - zt1).pow(2).mean() + relrec(sae_t1.decode(zhat), Hn1)
            opts["sparse"].zero_grad(); loss_a.backward(); opts["sparse"].step()
            # (b) dense transcoder
            loss_b = relrec(dense(zt), Hn1)
            opts["dense"].zero_grad(); loss_b.backward(); opts["dense"].step()
            # (c) crosscoder: shared masked code from H_t reconstructs both layers
            z = mask(cross.encode(Hn)); loss_c = relrec(cross.decode_t(z), Hn) + relrec(cross.decode_t1(z), Hn1)
            opts["cross"].zero_grad(); loss_c.backward(); opts["cross"].step()
            cross.dec_t1.weight.data /= (cross.dec_t1.weight.data.norm(dim=0, keepdim=True) + 1e-8)
    # eval
    res = {}
    for name in ["sparse", "dense", "cross"]:
        n = 0; agg = {}
        with torch.no_grad():
            for Ht, Ht1, logits, labels in te:
                Ht, logits, labels = Ht.to(dev), logits.to(dev), labels.to(dev)
                Hn = (Ht - mt) / st
                zt = mask(sae_t.encode(Hn))
                if name == "sparse": Hp = sae_t1.decode(sparse(zt))
                elif name == "dense": Hp = dense(zt)
                else: Hp = cross.decode_t1(mask(cross.encode(Hn)))
                d = splice_metrics(model, Hp * st1 + mt1, labels, logits); b = Ht.size(0); n += b
                for k, v in d.items(): agg[k] = agg.get(k, 0.0) + v * b
        res[name] = {k: v / n for k, v in agg.items()}
        print(f"{name}: spliced_top1 {res[name]['spliced_top1']:.3f} KL {res[name]['kl_orig_spliced']:.3f}", flush=True)
    json.dump(res, open(f"{args.out}/taxonomy_q4q5.json", "w"))
    print("TAXONOMY DONE", flush=True)


if __name__ == "__main__":
    main()

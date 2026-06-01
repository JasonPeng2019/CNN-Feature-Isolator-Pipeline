"""Phase 3 chain eval: compose learned transitions Z~_1 -> Z^_2 -> ... -> Z^_5,
decode at Q5, splice into frozen classifier. Compares against:
  - the frozen-block-hybrid chain (causal upper bound)
  - single-step transitions (no chaining)
"""
import argparse, os, sys, json
import torch
sys.path.insert(0, os.path.dirname(__file__))
import backbone as B, sae as S, masks as M, eval as E, transitions as T
from cache_activations import load_backbone, SECTIONS
from train_transition import load_sae, norm

PAIRS = [("Q1", "Q2"), ("Q2", "Q3"), ("Q3", "Q4"), ("Q4", "Q5")]


def load_pred(path, device):
    ck = torch.load(path, map_location=device)
    p = T.TransitionPredictor(ck["Kt"], ck["Kt1"], ck["sp_s"], ck["sp_d"]).to(device)
    p.load_state_dict(ck["state_dict"]); p.eval()
    return p


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="./runs/acts_r56_c100")
    ap.add_argument("--ckpt", default="./runs/backbone_r56_c100/best.pt")
    ap.add_argument("--p3", default="./runs/phase3")
    ap.add_argument("--sae_q1", default="./runs/winner_saes/Q1/sae.pt")
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="./runs/phase3/chain_result.json")
    args = ap.parse_args(); dev = args.device
    model, _ = load_backbone(args.ckpt, dev)
    saes = {"Q1": load_sae(args.sae_q1, dev)[0]}
    for q in ["Q2", "Q3", "Q4", "Q5"]:
        saes[q] = load_sae(f"runs/phase2/E1_{q}_f0.05/sae.pt", dev)[0]
    preds = {(a, b): load_pred(f"{args.p3}/T_{a}_{b}/pred.pt", dev) for a, b in PAIRS}
    means = {q: norm(args.acts, q, dev) for q in SECTIONS}

    H1 = torch.load(f"{args.acts}/test_Q1.pt").float()
    labels = torch.load(f"{args.acts}/test_labels.pt")
    orig = torch.load(f"{args.acts}/test_logits.pt")
    bs = 256; n = H1.shape[0]
    c_raw = c_masked = c_reground = hybrid_correct = agree = 0
    for i in range(0, n, bs):
        H = H1[i:i+bs].to(dev); y = labels[i:i+bs].to(dev); ol = orig[i:i+bs].to(dev)
        m1, s1 = means["Q1"]
        z0 = saes["Q1"].encode((H - m1) / s1); z0, _ = M.mask_global_frac(z0, args.frac)

        # (1) raw learned chain: feed predicted code straight on
        zc = z0
        for (a, b) in PAIRS: zc = preds[(a, b)](zc)
        l_raw = model.forward_from("Q5", saes["Q5"].decode(zc) * means["Q5"][1] + means["Q5"][0])
        c_raw += (l_raw.argmax(1) == y).sum().item()
        agree += (l_raw.argmax(1) == ol.argmax(1)).sum().item()

        # (2) masked learned chain: re-mask predicted code each step (match train input sparsity)
        zc = z0
        for (a, b) in PAIRS: zc, _ = M.mask_global_frac(preds[(a, b)](zc), args.frac)
        l_msk = model.forward_from("Q5", saes["Q5"].decode(zc) * means["Q5"][1] + means["Q5"][0])
        c_masked += (l_msk.argmax(1) == y).sum().item()

        # (3) re-grounded learned chain: decode->renorm->re-encode->mask each step
        zc = z0
        for (a, b) in PAIRS:
            mb, sb = means[b]
            Hb = saes[b].decode(preds[(a, b)](zc)) * sb + mb     # predicted raw H_{t+1}
            zc = saes[b].encode((Hb - mb) / sb); zc, _ = M.mask_global_frac(zc, args.frac)
        l_rg = model.forward_from("Q5", saes["Q5"].decode(zc) * means["Q5"][1] + means["Q5"][0])
        c_reground += (l_rg.argmax(1) == y).sum().item()

        # hybrid chain: decode + REAL frozen block each step (causal upper bound)
        Hcur = H
        for (a, b) in PAIRS:
            ma, sa = means[a]
            za = saes[a].encode((Hcur - ma) / sa); za, _ = M.mask_global_frac(za, args.frac)
            Hcur = model.forward_between(a, b, saes[a].decode(za) * sa + ma)
        hybrid_correct += (model.forward_from("Q5", Hcur).argmax(1) == y).sum().item()
    res = {"chain_raw_top1": c_raw / n, "chain_masked_top1": c_masked / n,
           "chain_reground_top1": c_reground / n, "hybrid_chain_top1": hybrid_correct / n,
           "chain_agree_orig": agree / n, "orig_top1": 0.7183}
    json.dump(res, open(args.out, "w"))
    print("CHAIN:", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()

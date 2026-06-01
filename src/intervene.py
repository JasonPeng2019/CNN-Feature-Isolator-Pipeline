"""N8 — parent->child causal intervention.

Tests whether the sparse code is a *causal* hierarchy, not just predictive.
Uses the frozen-block hybrid as the causal path (decode -> real frozen block -> re-encode),
so effects are measured through the actual network.

Two metrics over adjacent pairs (t -> t+1):
  M1 IMPORTANCE: ablate top-k parent coeffs (highest-magnitude kept) vs random-k kept.
     If parents are load-bearing, top-k ablation hurts downstream top-1 far more.
  M2 LOCALITY: ablate all kept coeffs in one source spatial block; measure the fraction
     of child-code change energy that lands inside that block's receptive-field window
     in t+1 vs outside. >0.5 (and >> the area-fraction null) => spatially-directed tree.
"""
import argparse, os, sys, json
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
import backbone as B, masks as M
from cache_activations import load_backbone, SECTIONS
from train_transition import load_sae, norm

PAIRS = [("Q1", "Q2"), ("Q2", "Q3"), ("Q3", "Q4"), ("Q4", "Q5")]


@torch.no_grad()
def child_code(sae_t, sae_t1, model, src, dst, z_masked, mean_t, std_t, mean_t1, std_t1):
    Ht_raw = sae_t.decode(z_masked) * std_t + mean_t
    Ht1_raw = model.forward_between(src, dst, Ht_raw)
    return sae_t1.encode((Ht1_raw - mean_t1) / std_t1), Ht1_raw


@torch.no_grad()
def run_pair(src, dst, sae_t, sae_t1, model, acts, nrm, dev, frac=0.05, kabl=8, bs=128, max_batches=8):
    mean_t, std_t = nrm[src]; mean_t1, std_t1 = nrm[dst]
    H = torch.load(f"{acts}/test_{src}.pt")
    labels = torch.load(f"{acts}/test_labels.pt"); orig = torch.load(f"{acts}/test_logits.pt")
    Cs, sp_s = B.SECTION_INFO[src]; Cd, sp_d = B.SECTION_INFO[dst]
    fh = sp_s // sp_d if sp_s >= sp_d else 1                  # source cells per target cell
    n = 0
    drop_top = drop_rand = 0.0; loc_in = loc_out = 0.0; loc_area = 0.0
    for b in range(0, min(max_batches * bs, H.shape[0]), bs):
        Hb = H[b:b+bs].to(dev); y = labels[b:b+bs].to(dev)
        Hn = (Hb - mean_t) / std_t
        z = sae_t.encode(Hn); zt, keep = M.mask_global_frac(z, frac)
        base_logits = model.forward_from(dst, child_code(sae_t, sae_t1, model, src, dst, zt, mean_t, std_t, mean_t1, std_t1)[1])
        base_top1 = (base_logits.argmax(1) == y).float()

        # ---- M1: top-k vs random-k parent ablation (downstream) ----
        Bn = Hb.size(0); flat = zt.reshape(Bn, -1)
        topidx = flat.topk(kabl, dim=1).indices
        z_top = flat.clone(); z_top.scatter_(1, topidx, 0.0); z_top = z_top.reshape(zt.shape)
        # random kept indices
        randmask = (flat > 0).float()
        randidx = (torch.rand_like(flat) * randmask).topk(kabl, dim=1).indices
        z_rand = flat.clone(); z_rand.scatter_(1, randidx, 0.0); z_rand = z_rand.reshape(zt.shape)
        lt = model.forward_from(dst, child_code(sae_t, sae_t1, model, src, dst, z_top, mean_t, std_t, mean_t1, std_t1)[1])
        lr = model.forward_from(dst, child_code(sae_t, sae_t1, model, src, dst, z_rand, mean_t, std_t, mean_t1, std_t1)[1])
        drop_top += (base_top1 - (lt.argmax(1) == y).float()).sum().item()
        drop_rand += (base_top1 - (lr.argmax(1) == y).float()).sum().item()

        # ---- M2: spatial locality of a source-block ablation ----
        zc_base, _ = child_code(sae_t, sae_t1, model, src, dst, zt, mean_t, std_t, mean_t1, std_t1)
        # ablate one central source block (size fh x fh) -> expect child change in matching t+1 cell
        i0 = (sp_s // 2 // max(fh, 1)) * max(fh, 1); j0 = i0
        z_blk = zt.clone(); z_blk[:, :, i0:i0+fh, j0:j0+fh] = 0.0
        zc_abl, _ = child_code(sae_t, sae_t1, model, src, dst, z_blk, mean_t, std_t, mean_t1, std_t1)
        delta = (zc_abl - zc_base).abs().sum(1)              # (B, sp_d, sp_d) child change map
        ui, uj = i0 // max(fh, 1), j0 // max(fh, 1)          # corresponding t+1 cell
        win = torch.zeros(sp_d, sp_d, device=dev)
        win[max(0, ui-1):ui+2, max(0, uj-1):uj+2] = 1.0      # 3x3 effective-RF neighborhood
        e_in = (delta * win).sum().item(); e_tot = delta.sum().item() + 1e-9
        loc_in += e_in; loc_out += (e_tot - e_in)
        loc_area += (win.sum().item() / (sp_d * sp_d)) * Bn
        n += Bn
    return {
        "drop_top_per_img": drop_top / n, "drop_rand_per_img": drop_rand / n,
        "importance_ratio": (drop_top / max(drop_rand, 1e-6)),
        "locality_frac_in_RFcell": loc_in / (loc_in + loc_out + 1e-9),
        "null_area_frac": loc_area / n, "kabl": kabl, "frac": frac,
    }


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="./runs/acts_r56_c100")
    ap.add_argument("--ckpt", default="./runs/backbone_r56_c100/best.pt")
    ap.add_argument("--sae_q1", default="./runs/winner_saes/Q1/sae.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="./runs/intervene_result.json")
    args = ap.parse_args(); dev = args.device
    model, _ = load_backbone(args.ckpt, dev)
    saes = {"Q1": load_sae(args.sae_q1, dev)[0]}
    for q in ["Q2", "Q3", "Q4", "Q5"]:
        saes[q] = load_sae(f"runs/phase2/E1_{q}_f0.05/sae.pt", dev)[0]
    nrm = {q: norm(args.acts, q, dev) for q in SECTIONS}
    res = {}
    for s, d in PAIRS:
        res[f"{s}->{d}"] = run_pair(s, d, saes[s], saes[d], model, args.acts, nrm, dev)
        r = res[f"{s}->{d}"]
        print(f"{s}->{d}: importance top/rand = {r['drop_top_per_img']:.3f}/{r['drop_rand_per_img']:.3f} "
              f"(ratio {r['importance_ratio']:.2f}) | locality {r['locality_frac_in_RFcell']:.3f} "
              f"(null {r['null_area_frac']:.3f})", flush=True)
    json.dump(res, open(args.out, "w"))
    print("INTERVENE DONE", args.out, flush=True)


if __name__ == "__main__":
    main()

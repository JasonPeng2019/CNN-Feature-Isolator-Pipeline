"""Phase 3b — Family II: chain-aware transition training (scheduled sampling).

Fixes the Family-I exposure-bias collapse: each predictor T_t is trained on
re-grounded *predicted* inputs (decode->re-encode->mask of the previous
predictor's output), not just the true masked code. Scheduled-sampling prob p
anneals 0->1 so predictors gradually face their own chain inputs. Carrier is
detached between steps (stop-gradient) -> stable, each predictor decoupled.

Frozen SAEs (Field, winner). Predictors warm-started from Phase-3 Family I.
"""
import argparse, os, sys, json, time
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
import backbone as B, sae as S, masks as M, transitions as T
from cache_activations import load_backbone, SECTIONS
from train_transition import load_sae, norm

PAIRS = [("Q1", "Q2"), ("Q2", "Q3"), ("Q3", "Q4"), ("Q4", "Q5")]


class ChainActs(torch.utils.data.Dataset):
    def __init__(self, acts, split):
        self.H = {q: torch.load(os.path.join(acts, f"{split}_{q}.pt")) for q in SECTIONS}
        self.labels = torch.load(os.path.join(acts, f"{split}_labels.pt"))

    def __len__(self): return self.labels.shape[0]

    def __getitem__(self, i):
        return {q: self.H[q][i].float() for q in SECTIONS}, self.labels[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="./runs/acts_r56_c100")
    ap.add_argument("--ckpt", default="./runs/backbone_r56_c100/best.pt")
    ap.add_argument("--p3", default="./runs/phase3")
    ap.add_argument("--sae_q1", default="./runs/winner_saes/Q1/sae.pt")
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bptt", action="store_true")  # T1.2 architectural re-grounding
    ap.add_argument("--out", default="./runs/phase3b")
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True); dev = args.device
    model, _ = load_backbone(args.ckpt, dev)
    saes = {"Q1": load_sae(args.sae_q1, dev)[0]}
    for q in ["Q2", "Q3", "Q4", "Q5"]:
        saes[q] = load_sae(f"runs/phase2/E1_{q}_f0.05/sae.pt", dev)[0]
    nrm = {q: norm(args.acts, q, dev) for q in SECTIONS}
    # warm-start predictors from Family I
    preds = {}
    for a, b in PAIRS:
        ck = torch.load(f"{args.p3}/T_{a}_{b}/pred.pt", map_location=dev)
        p = T.TransitionPredictor(ck["Kt"], ck["Kt1"], ck["sp_s"], ck["sp_d"]).to(dev)
        p.load_state_dict(ck["state_dict"]); preds[(a, b)] = p
    params = [pp for p in preds.values() for pp in p.parameters()]
    opt = torch.optim.Adam(params, lr=args.lr)
    mask = lambda z: M.mask_global_frac(z, args.frac)[0]

    tr = torch.utils.data.DataLoader(ChainActs(args.acts, "train"), batch_size=args.bs, shuffle=True, num_workers=2, pin_memory=True)
    te = torch.utils.data.DataLoader(ChainActs(args.acts, "test"), batch_size=args.bs, num_workers=2)
    log = []
    for ep in range(args.epochs):
        p_ss = ep / max(1, args.epochs - 1)          # scheduled-sampling prob 0->1
        for p in preds.values(): p.train()
        t0 = time.time(); rl = 0.0
        for Hd, _ in tr:
            Hn = {q: ((Hd[q].to(dev) - nrm[q][0]) / nrm[q][1]) for q in SECTIONS}
            with torch.no_grad():
                zc = mask(saes["Q1"].encode(Hn["Q1"]))      # true masked code at Q1
            loss = 0.0
            if args.bptt:
                # full-chain BPTT through differentiable re-grounding (carrier stays in-graph)
                for (a, b) in PAIRS:
                    mb, sb = nrm[b]
                    with torch.no_grad():
                        z_b_target = saes[b].encode(Hn[b])
                    zhat = preds[(a, b)](zc)
                    Hhat_b = saes[b].decode(zhat)
                    loss = loss + (zhat - z_b_target).pow(2).mean() + \
                        (Hhat_b - Hn[b]).pow(2).sum() / (Hn[b].pow(2).sum() + 1e-8)
                    zc = mask(saes[b].encode((saes[b].decode(zhat) * sb + mb - mb) / sb))  # in-graph re-ground
            else:
                for (a, b) in PAIRS:
                    with torch.no_grad():
                        z_teacher = mask(saes[a].encode(Hn[a]))
                        use_pred = (torch.rand(1).item() < p_ss) and (a != "Q1")
                        inp = zc.detach() if use_pred else z_teacher
                        z_b_target = saes[b].encode(Hn[b])
                    zhat = preds[(a, b)](inp)
                    Hhat_b = saes[b].decode(zhat)
                    code = (zhat - z_b_target).pow(2).mean()
                    rec = (Hhat_b - Hn[b]).pow(2).sum() / (Hn[b].pow(2).sum() + 1e-8)
                    loss = loss + code + rec
                    with torch.no_grad():                        # re-ground carrier for next step
                        mb, sb = nrm[b]
                        Hb_raw = saes[b].decode(preds[(a, b)](zc.detach() if (a != "Q1") else z_teacher)) * sb + mb
                        zc = mask(saes[b].encode((Hb_raw - mb) / sb))
            opt.zero_grad(); loss.backward(); opt.step(); rl += loss.item()
        chain = chain_eval(preds, saes, nrm, model, te, mask, dev)
        log.append({"epoch": ep, "p_ss": p_ss, "loss": rl / len(tr), **chain, "sec": time.time() - t0})
        print(f"ep{ep} p_ss{p_ss:.2f} loss{rl/len(tr):.3f} chain_reground {chain['chain_reground_top1']:.3f} "
              f"chain_raw {chain['chain_raw_top1']:.3f} (hybrid {chain['hybrid_chain_top1']:.3f})", flush=True)
    json.dump({"config": vars(args), "log": log, "final": log[-1]}, open(os.path.join(args.out, "result.json"), "w"))
    for (a, b), p in preds.items():
        torch.save({"state_dict": p.state_dict()}, os.path.join(args.out, f"pred_{a}_{b}.pt"))
    print("PHASE3B DONE", flush=True)


@torch.no_grad()
def chain_eval(preds, saes, nrm, model, loader, mask, dev):
    for p in preds.values(): p.eval()
    raw = rg = hyb = n = 0
    for Hd, y in loader:
        y = y.to(dev); Hn = {q: ((Hd[q].to(dev) - nrm[q][0]) / nrm[q][1]) for q in SECTIONS}
        z0 = mask(saes["Q1"].encode(Hn["Q1"]))
        zc = z0
        for (a, b) in PAIRS: zc = preds[(a, b)](zc)
        raw += (model.forward_from("Q5", saes["Q5"].decode(zc) * nrm["Q5"][1] + nrm["Q5"][0]).argmax(1) == y).sum().item()
        zc = z0
        for (a, b) in PAIRS:
            mb, sb = nrm[b]
            Hb = saes[b].decode(preds[(a, b)](zc)) * sb + mb
            zc = mask(saes[b].encode((Hb - mb) / sb))
        rg += (model.forward_from("Q5", saes["Q5"].decode(zc) * nrm["Q5"][1] + nrm["Q5"][0]).argmax(1) == y).sum().item()
        Hc = Hd["Q1"].to(dev)
        for (a, b) in PAIRS:
            ma, sa = nrm[a]; za = mask(saes[a].encode((Hc - ma) / sa))
            Hc = model.forward_between(a, b, saes[a].decode(za) * sa + ma)
        hyb += (model.forward_from("Q5", Hc).argmax(1) == y).sum().item()
        n += y.numel()
    return {"chain_raw_top1": raw / n, "chain_reground_top1": rg / n, "hybrid_chain_top1": hyb / n}


if __name__ == "__main__":
    main()

"""CPU-only figures: SAE-spliced output vs original CNN, plus sparsity graphs.
Uses cached activations + trained SAE checkpoints + frozen backbone (no GPU)."""
import sys, os, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import backbone as B, masks as M, data as D
from cache_activations import load_backbone, SECTIONS
from train_transition import load_sae, norm

ROOT = os.path.join(os.path.dirname(__file__), ".."); OUT = os.path.join(ROOT, "report", "plots")
dev = "cpu"; ORIG = 0.7183
plt.rcParams.update({"figure.dpi": 130, "font.size": 11})
acts = os.path.join(ROOT, "runs/acts_r56_c100")
model, _ = load_backbone(os.path.join(ROOT, "runs/backbone_r56_c100/best.pt"), dev)
labels = torch.load(f"{acts}/test_labels.pt"); orig_logits = torch.load(f"{acts}/test_logits.pt")
classes = D.datasets.CIFAR100(os.path.join(ROOT, "data"), train=False, download=False).classes

def nrm(sec): return norm(acts, sec, dev)

# ---------- Fig 10: SAE-spliced vs original CNN ----------
sae5, _ = load_sae(os.path.join(ROOT, "runs/phase2/E1_Q5_f0.05/sae.pt"), dev)
mean, std = nrm("Q5"); Hq5 = torch.load(f"{acts}/test_Q5.pt").float()
N = 2000
with torch.no_grad():
    Hn = (Hq5[:N] - mean) / std
    z = sae5.encode(Hn); zt, keep = M.mask_global_frac(z, 0.05)
    Hhat = sae5.decode(zt) * std + mean
    spliced = model.forward_from("Q5", Hhat)
ol = orig_logits[:N]; yl = labels[:N]
agree = (spliced.argmax(1) == ol.argmax(1)).float().mean().item()
sp_acc = (spliced.argmax(1) == yl).float().mean().item()
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
si = torch.randint(0, N, (1500,))
ax[0].scatter(ol[si].flatten()[::3], spliced[si].flatten()[::3], s=2, alpha=0.25, color="#2b8cbe")
lim = [ol[si].min().item(), ol[si].max().item()]; ax[0].plot(lim, lim, "k--", lw=1)
ax[0].set_xlabel("original CNN logits"); ax[0].set_ylabel("SAE-spliced logits")
ax[0].set_title(f"Logits match the original (Q5 SAE @5%)\npred-agreement {agree*100:.1f}%")
# per-section orig vs spliced top1 @5%
secs = ["Q1", "Q2", "Q3", "Q4", "Q5"]
def top1_at(sec):
    for d in (f"runs/phase2/E1_{sec}_f0.05", f"runs/phase1/E3_{sec}_f0.05"):
        p = os.path.join(ROOT, d, "result.json")
        if os.path.exists(p): return json.load(open(p))["final"]["spliced_top1"]
    return np.nan
sp = [top1_at(s) for s in secs]; x = np.arange(5); w = 0.38
ax[1].bar(x - w/2, [ORIG]*5, w, label="original CNN", color="#969696")
ax[1].bar(x + w/2, sp, w, label="SAE-spliced @5%", color="#238b45")
ax[1].set_xticks(x); ax[1].set_xticklabels(secs); ax[1].set_ylim(0.6, 0.74)
ax[1].set_ylabel("test top-1"); ax[1].set_title("SAE reconstruction preserves accuracy at every layer"); ax[1].legend(fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig10_sae_vs_cnn.png"); plt.close(fig)

# ---------- Fig 11: worked examples (CNN pred vs SAE pred) ----------
_, test = D.get_loaders("cifar100", root=os.path.join(ROOT, "data"), batch_size=256, augment=False)
imgs = torch.cat([x for x, _ in test])[:N]
m_, s_ = torch.tensor(D.CIFAR100_MEAN).view(3,1,1), torch.tensor(D.CIFAR100_STD).view(3,1,1)
fig, axes = plt.subplots(2, 6, figsize=(11, 4.2))
sel = torch.randperm(N)[:12]
for a, idx in zip(axes.flat, sel):
    im = (imgs[idx]*s_+m_).clamp(0,1).permute(1,2,0).numpy()
    a.imshow(im); a.axis("off")
    cnn = classes[ol[idx].argmax().item()]; saep = classes[spliced[idx].argmax().item()]
    ok = cnn == saep
    a.set_title(f"CNN: {cnn[:10]}\nSAE: {saep[:10]}", fontsize=7, color=("green" if ok else "red"))
fig.suptitle("Original CNN prediction vs SAE-spliced prediction (Q5 @5% retained)", fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/fig11_examples.png"); plt.close(fig)

# ---------- Fig 12: activation reconstruction heatmap (Q1) ----------
sae1, _ = load_sae(os.path.join(ROOT, "runs/winner_saes/Q1/sae.pt"), dev)
m1, s1 = nrm("Q1"); Hq1 = torch.load(f"{acts}/test_Q1.pt")[:1].float()
with torch.no_grad():
    Hn1 = (Hq1 - m1) / s1; z1 = sae1.encode(Hn1); zt1, _ = M.mask_global_frac(z1, 0.05); Hh1 = sae1.decode(zt1)
o = Hn1[0].abs().mean(0).numpy(); r = Hh1[0].abs().mean(0).numpy(); e = (Hn1-Hh1)[0].abs().mean(0).numpy()
fig, ax = plt.subplots(1, 3, figsize=(10, 3.6))
for a, M_, t in zip(ax, [o, r, e], ["original activation |H|", "SAE reconstruction |Ĥ|", "|error| (5% retained)"]):
    im = a.imshow(M_, cmap="viridis"); a.set_title(t, fontsize=10); a.axis("off"); fig.colorbar(im, ax=a, fraction=0.046)
fig.suptitle("Q1 activation field: original vs SAE reconstruction (one image)", fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/fig12_actmap.png"); plt.close(fig)

# ---------- Fig 13: sparsity graphs ----------
with torch.no_grad():
    nfeat = keep[:N].float().amax(dim=(2,3)).sum(1).numpy()   # active features per image (Q5)
    nloc = keep[:N].float().sum(dim=(1,2,3)).numpy()          # active coeffs per image
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].hist(nfeat, bins=30, color="#6a51a3"); ax[0].set_xlabel("active features per image (Q5)")
ax[0].set_ylabel("count"); ax[0].set_title(f"Sparse usage: ~{nfeat.mean():.0f}/{keep.shape[1]} features active")
# accuracy vs fraction (Q5 pareto)
P = []
for f in glob.glob(os.path.join(ROOT, "runs/pareto/F_Q5_f*/result.json")):
    fin = json.load(open(f))["final"]; P.append((fin["frac_retained"], fin["spliced_top1"]))
P = sorted(P)
if P:
    fr, tp = zip(*P); ax[1].plot(fr, tp, "o-", color="#238b45")
ax[1].axhline(ORIG, ls="--", c="k", lw=1, label="original CNN")
ax[1].set_xscale("log"); ax[1].set_xlabel("fraction of coefficients retained (log)")
ax[1].set_ylabel("spliced top-1"); ax[1].set_title("Sparsity vs accuracy (Q5)"); ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/fig13_sparsity.png"); plt.close(fig)

print(f"SAE-vs-CNN done. Q5@5%: spliced_top1={sp_acc:.3f} agreement={agree*100:.1f}% (orig {ORIG})")
print("figs: fig10_sae_vs_cnn, fig11_examples, fig12_actmap, fig13_sparsity")

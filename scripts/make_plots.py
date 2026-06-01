"""Generate all figures for the report into report/plots/."""
import json, glob, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "report", "plots"); os.makedirs(OUT, exist_ok=True)
SECS = ["Q1", "Q2", "Q3", "Q4", "Q5"]
ORIG = 0.7183
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True, "grid.alpha": 0.3})


def load(pat):
    d = {}
    for f in glob.glob(os.path.join(ROOT, pat)):
        d[f.split("/")[-2]] = json.load(open(f))["final"]
    return d


# ---- Fig 1: Phase 1 sparsity-preservation curves (Vector+global) + baselines ----
p1 = load("runs/phase1/E3_*/result.json")
pca = json.load(open(os.path.join(ROOT, "runs/baselines_pca.json")))
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
colors = plt.cm.viridis(np.linspace(0, 0.85, 5))
for c, sec in zip(colors, SECS):
    pts = sorted([(v["frac_retained"], v["spliced_top1"]) for k, v in p1.items() if k.split("_")[1] == sec])
    if pts:
        x, y = zip(*pts); ax[0].plot(x, y, "o-", color=c, label=sec)
ax[0].axhline(ORIG, ls="--", c="k", lw=1, label="orig (0.718)")
ax[0].set_xlabel("fraction of coefficients retained"); ax[0].set_ylabel("spliced top-1")
ax[0].set_title("Phase 1: sparse reconstruction preserves accuracy\n(Vector SAE + global mask)"); ax[0].legend(fontsize=8)
# right: relL2 vs frac
for c, sec in zip(colors, SECS):
    pts = sorted([(v["frac_retained"], v["rel_l2"]) for k, v in p1.items() if k.split("_")[1] == sec])
    if pts:
        x, y = zip(*pts); ax[1].plot(x, y, "o-", color=c, label=sec)
ax[1].set_xlabel("fraction retained"); ax[1].set_ylabel("relative L2 recon error")
ax[1].set_title("Reconstruction error vs sparsity"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_phase1_curves.png"); plt.close(fig)


# ---- Fig 2: SAE beats baselines (bar, at ~5% / matched) ----
rand = load("runs/phase1_rand/RAND_*/result.json")
fig, ax = plt.subplots(figsize=(7.5, 4.2))
x = np.arange(len(SECS)); w = 0.27
sae5 = [p1.get(f"E3_{s}_f0.05", p1.get(f"E3_{s}_f0.02", {})).get("spliced_top1", np.nan) for s in SECS]
pcah = [list(pca[s].values())[-1]["spliced_top1"] for s in SECS]  # rank C/2
randv = [rand.get(f"RAND_{s}_f0.05", rand.get(f"RAND_{s}_f0.02", {})).get("spliced_top1", np.nan) for s in SECS]
ax.bar(x - w, sae5, w, label="trained SAE @5%", color="#2c7fb8")
ax.bar(x, pcah, w, label="PCA @rank C/2", color="#7fcdbb")
ax.bar(x + w, randv, w, label="random-dict SAE", color="#d95f0e")
ax.axhline(ORIG, ls="--", c="k", lw=1, label="orig")
ax.set_xticks(x); ax.set_xticklabels(SECS); ax.set_ylabel("spliced top-1")
ax.set_title("Phase 1: trained SAE >> PCA and random-dict baselines"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_baselines.png"); plt.close(fig)


# ---- Fig 3: Field vs Vector SAE (global) at 2% retained ----
p2 = load("runs/phase2/*/result.json")
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
field2 = [p2.get(f"E1_{s}_f0.02", {}).get("spliced_top1", np.nan) for s in SECS]
vec2 = [p1.get(f"E3_{s}_f0.02", {}).get("spliced_top1", np.nan) for s in SECS]
x = np.arange(len(SECS)); w = 0.38
ax[0].bar(x - w/2, field2, w, label="Field SAE", color="#225ea8")
ax[0].bar(x + w/2, vec2, w, label="Vector SAE", color="#41b6c4")
ax[0].axhline(ORIG, ls="--", c="k", lw=1)
ax[0].set_xticks(x); ax[0].set_xticklabels(SECS); ax[0].set_ylabel("spliced top-1 @2% retained")
ax[0].set_title("Field SAE >> Vector at aggressive sparsity"); ax[0].legend(fontsize=9)
# RF: field vs vector at k_rf=4
fieldrf = [p2.get(f"E2_{s}_k4", {}).get("spliced_top1", np.nan) for s in SECS]
vecrf = [p2.get(f"E4_{s}_k4", {}).get("spliced_top1", np.nan) for s in SECS]
ax[1].bar(x - w/2, fieldrf, w, label="Field+RF", color="#225ea8")
ax[1].bar(x + w/2, vecrf, w, label="Vector+RF", color="#41b6c4")
ax[1].axhline(ORIG, ls="--", c="k", lw=1)
ax[1].set_xticks(x); ax[1].set_xticklabels(SECS); ax[1].set_ylabel("spliced top-1 (k_rf=4)")
ax[1].set_title("RF-local masking works only with Field SAE"); ax[1].legend(fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_field_vs_vector.png"); plt.close(fig)


# ---- Fig 4: Phase 3 per-transition learned vs hybrid ----
p3 = load("runs/phase3/T_*/result.json")
pairs = ["T_Q1_Q2", "T_Q2_Q3", "T_Q3_Q4", "T_Q4_Q5"]
labels = ["Q1→Q2", "Q2→Q3", "Q3→Q4", "Q4→Q5"]
learned = [p3[p]["spliced_top1"] for p in pairs]; hybrid = [p3[p]["hybrid_top1"] for p in pairs]
fig, ax = plt.subplots(figsize=(7.5, 4.2)); x = np.arange(4); w = 0.38
ax.bar(x - w/2, learned, w, label="learned transition", color="#3690c0")
ax.bar(x + w/2, hybrid, w, label="N4 frozen-block hybrid (causal bound)", color="#016c59")
ax.axhline(ORIG, ls="--", c="k", lw=1, label="orig")
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("spliced top-1")
ax.set_title("Phase 3: single-step transitions (learned vs causal upper bound)"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_transitions.png"); plt.close(fig)


# ---- Fig 5: Chain Q1->Q5 comparison ----
ch = json.load(open(os.path.join(ROOT, "runs/phase3/chain_result.json")))
chb = json.load(open(os.path.join(ROOT, "runs/phase3b/result.json")))["final"]
names = ["raw\nlearned", "+re-mask", "+re-ground\n(Fam I)", "re-ground\n(Fam II)", "frozen-block\nhybrid", "original"]
vals = [ch["chain_raw_top1"], ch["chain_masked_top1"], ch["chain_reground_top1"],
        chb["chain_reground_top1"], ch["hybrid_chain_top1"], ORIG]
fig, ax = plt.subplots(figsize=(8.5, 4.2))
cols = ["#d7301f", "#fc8d59", "#fdcc8a", "#fee8c8", "#2b8cbe", "#000000"]
ax.bar(names, vals, color=cols)
for i, v in enumerate(vals): ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
ax.set_ylabel("chained Q1→Q5 spliced top-1"); ax.set_ylim(0, 0.8)
ax.set_title("Phase 3: chaining the sparse-code hierarchy to the classifier")
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_chain.png"); plt.close(fig)


# ---- Fig 6: Phase 3b training curve ----
log = json.load(open(os.path.join(ROOT, "runs/phase3b/result.json")))["log"]
ep = [r["epoch"] for r in log]
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(ep, [r["chain_reground_top1"] for r in log], "o-", label="re-grounded chain", color="#2b8cbe")
ax.plot(ep, [r["chain_raw_top1"] for r in log], "s-", label="raw chain", color="#d7301f")
ax.axhline(0.7122, ls="--", c="green", lw=1, label="hybrid bound (0.712)")
ax.plot(ep, [r["p_ss"] for r in log], ":", c="gray", label="scheduled-sampling p")
ax.set_xlabel("epoch"); ax.set_ylabel("chain top-1 / p_ss")
ax.set_title("Phase 3b: chain-aware training (Family II)"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig6_phase3b.png"); plt.close(fig)

print("PLOTS DONE:", sorted(os.listdir(OUT)))

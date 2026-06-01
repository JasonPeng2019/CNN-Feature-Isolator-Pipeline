"""Figures for the tier experiments (CPU-only; uses completed result JSONs)."""
import json, glob, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np
ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "report", "plots"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
def J(p): return json.load(open(os.path.join(ROOT, p)))

# ---- Fig 7: N8 intervention (importance ratio + locality vs null) ----
iv = J("runs/intervene_result.json"); pairs = list(iv.keys())
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2)); x = np.arange(len(pairs))
imp = [min(iv[p]["importance_ratio"], 20) for p in pairs]  # clip inf for display
ax[0].bar(x, imp, color="#3690c0"); ax[0].axhline(1, ls="--", c="k", lw=1, label="parity (no hierarchy)")
ax[0].set_xticks(x); ax[0].set_xticklabels(pairs); ax[0].set_ylabel("top-k / random-k damage ratio")
ax[0].set_title("N8 importance: top parents >> random kept coeffs\n(ratio clipped at 20; some →∞)"); ax[0].legend(fontsize=8)
loc = [iv[p]["locality_frac_in_RFcell"] for p in pairs]; null = [iv[p]["null_area_frac"] for p in pairs]
w = 0.38
ax[1].bar(x - w/2, loc, w, label="child-change in RF nbhd", color="#016c59")
ax[1].bar(x + w/2, null, w, label="chance (area null)", color="#bdbdbd")
ax[1].set_xticks(x); ax[1].set_xticklabels(pairs); ax[1].set_ylabel("fraction of child-change energy")
ax[1].set_title("N8 locality: change concentrates in parent's RF"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig7_intervention.png"); plt.close(fig)

# ---- Fig 8: re-grounding progression ----
famI = J("runs/phase3/chain_result.json")["chain_reground_top1"]
sched = J("runs/phase3b/result.json")["final"]["chain_reground_top1"]
bptt = J("runs/phase3b_bptt/result.json")["final"]["chain_reground_top1"]
hyb = J("runs/phase3/chain_result.json")["hybrid_chain_top1"]
raw = J("runs/phase3/chain_result.json")["chain_raw_top1"]
fig, ax = plt.subplots(figsize=(8, 4.2))
names = ["raw\nlearned", "Fam I\nre-ground", "Fam II\nsched-samp", "Fam II\nBPTT", "frozen-block\nhybrid", "original"]
vals = [raw, famI, sched, bptt, hyb, 0.7183]
cols = ["#d7301f", "#fdcc8a", "#fdae6b", "#41ab5d", "#2b8cbe", "#000000"]
ax.bar(names, vals, color=cols)
for i, v in enumerate(vals): ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
ax.set_ylim(0, 0.8); ax.set_ylabel("chained Q1→Q5 spliced top-1")
ax.set_title("T1.2: architectural re-grounding (BPTT) nearly closes the chaining gap")
fig.tight_layout(); fig.savefig(f"{OUT}/fig8_regrounding.png"); plt.close(fig)

# ---- Fig 9: full Pareto sweep (Field SAE, all sections) ----
P = {}
for f in glob.glob(os.path.join(ROOT, "runs/pareto/F_*/result.json")):
    fin = json.load(open(f))["final"]; name = f.split("/")[-2]; sec = name.split("_")[1]
    P.setdefault(sec, []).append((fin["frac_retained"], fin["spliced_top1"]))
fig, ax = plt.subplots(figsize=(7.5, 4.6)); colors = plt.cm.viridis(np.linspace(0, 0.85, 5))
for c, sec in zip(colors, ["Q1", "Q2", "Q3", "Q4", "Q5"]):
    pts = sorted(P.get(sec, []))
    if pts: x, y = zip(*pts); ax.plot(x, y, "o-", color=c, label=sec)
ax.axhline(0.7183, ls="--", c="k", lw=1, label="orig")
ax.set_xscale("log"); ax.set_xlabel("fraction retained (log)"); ax.set_ylabel("spliced top-1")
ax.set_title("T2.2: full sparsity–fidelity Pareto (Field SAE, global mask)"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig9_pareto.png"); plt.close(fig)

print("TIER PLOTS DONE")
# print numbers for the report
print("N8:", {p: {"imp": round(iv[p]["importance_ratio"], 2), "loc": round(iv[p]["locality_frac_in_RFcell"], 3), "null": round(iv[p]["null_area_frac"], 3)} for p in pairs})
print("regrounding raw/famI/sched/bptt/hyb:", [round(v, 3) for v in [raw, famI, sched, bptt, hyb]])
for q in ["Q3", "Q5"]:
    try: print(f"audit {q}:", J(f"runs/audit/audit_{q}.json"))
    except Exception: pass

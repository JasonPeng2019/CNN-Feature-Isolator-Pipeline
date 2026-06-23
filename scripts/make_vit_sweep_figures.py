"""Summarize a completed ViT sweep and generate a full figure suite.

Outputs:
- per-run and aggregated CSV/JSON summaries under runs/vit_sweep/analysis/
- PNG figures under report/plots/
- PDF figures under report/figures_pdf/
- PDF figures for the LaTeX analysis report under analysis/figures/
"""
import csv
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
RUN_ROOT = os.path.join(ROOT, "runs", "vit_sweep")
OUT_ANALYSIS = os.path.join(RUN_ROOT, "analysis")
OUT_PNG = os.path.join(ROOT, "report", "plots")
OUT_PDF = os.path.join(ROOT, "report", "figures_pdf")
OUT_TEX = os.path.join(ROOT, "analysis", "figures")

for path in [OUT_ANALYSIS, OUT_PNG, OUT_PDF, OUT_TEX]:
    os.makedirs(path, exist_ok=True)

plt.rcParams.update({"figure.dpi": 140, "font.size": 11, "axes.grid": True, "grid.alpha": 0.25})


def load_rows():
    rows = []
    for path in sorted(glob.glob(os.path.join(RUN_ROOT, "*", "vit_result.json"))):
        with open(path) as f:
            row = json.load(f)
        row["path"] = path
        row["frac_bucket"] = round(row["frac_retained"], 2)
        rows.append(row)
    if not rows:
        raise RuntimeError(f"No vit_result.json files found under {RUN_ROOT}")
    return rows


def mean(vals):
    return sum(vals) / len(vals)


def std(vals):
    m = mean(vals)
    return (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5


def aggregate(rows):
    by_setting = defaultdict(list)
    by_block = defaultdict(list)
    by_frac = defaultdict(list)
    for row in rows:
        key = (row["block"], row["frac_bucket"])
        by_setting[key].append(row)
        by_block[row["block"]].append(row)
        by_frac[row["frac_bucket"]].append(row)

    settings = []
    for (block, frac), vals in sorted(by_setting.items()):
        pred = [v["pred_agree"] for v in vals]
        kl = [v["kl"] for v in vals]
        rel = [v["rel_l2"] for v in vals]
        fvu = [v["fvu"] for v in vals]
        cos = [v["cos"] for v in vals]
        settings.append({
            "block": block,
            "frac": frac,
            "n": len(vals),
            "mean_pred_agree": mean(pred),
            "std_pred_agree": std(pred),
            "mean_kl": mean(kl),
            "std_kl": std(kl),
            "mean_rel_l2": mean(rel),
            "mean_fvu": mean(fvu),
            "mean_cos": mean(cos),
            "seed_values": [v["pred_agree"] for v in sorted(vals, key=lambda x: x["seed"])],
        })

    settings.sort(key=lambda r: (-r["mean_pred_agree"], r["std_pred_agree"], r["mean_kl"]))

    block_summary = []
    for block, vals in sorted(by_block.items()):
        pred = [v["pred_agree"] for v in vals]
        kl = [v["kl"] for v in vals]
        block_summary.append({
            "block": block,
            "mean_pred_agree": mean(pred),
            "std_pred_agree": std(pred),
            "mean_kl": mean(kl),
        })

    frac_summary = []
    for frac, vals in sorted(by_frac.items()):
        pred = [v["pred_agree"] for v in vals]
        kl = [v["kl"] for v in vals]
        frac_summary.append({
            "frac": frac,
            "mean_pred_agree": mean(pred),
            "std_pred_agree": std(pred),
            "mean_kl": mean(kl),
        })

    return settings, block_summary, frac_summary


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def write_summaries(rows, settings, block_summary, frac_summary):
    write_csv(
        os.path.join(OUT_ANALYSIS, "vit_sweep_per_run.csv"),
        rows,
        ["model", "block", "seed", "frac_retained", "pred_agree", "kl", "rel_l2", "fvu", "cos", "ntrain", "nval", "path"],
    )
    write_csv(
        os.path.join(OUT_ANALYSIS, "vit_sweep_by_setting.csv"),
        settings,
        ["block", "frac", "n", "mean_pred_agree", "std_pred_agree", "mean_kl", "std_kl", "mean_rel_l2", "mean_fvu", "mean_cos", "seed_values"],
    )
    write_csv(
        os.path.join(OUT_ANALYSIS, "vit_sweep_by_block.csv"),
        block_summary,
        ["block", "mean_pred_agree", "std_pred_agree", "mean_kl"],
    )
    write_csv(
        os.path.join(OUT_ANALYSIS, "vit_sweep_by_frac.csv"),
        frac_summary,
        ["frac", "mean_pred_agree", "std_pred_agree", "mean_kl"],
    )
    summary = {
        "num_runs": len(rows),
        "best_setting": settings[0],
        "best_block": max(block_summary, key=lambda r: r["mean_pred_agree"]),
        "best_frac": max(frac_summary, key=lambda r: r["mean_pred_agree"]),
    }
    with open(os.path.join(OUT_ANALYSIS, "vit_sweep_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def save_fig(fig, stem, also_tex=False):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_PNG, f"{stem}.png"))
    fig.savefig(os.path.join(OUT_PDF, f"{stem}.pdf"))
    if also_tex:
        fig.savefig(os.path.join(OUT_TEX, f"{stem}.pdf"))
    plt.close(fig)


def make_heatmaps(settings):
    blocks = sorted({r["block"] for r in settings})
    fracs = sorted({r["frac"] for r in settings})
    pred = np.zeros((len(blocks), len(fracs)))
    kl = np.zeros((len(blocks), len(fracs)))
    pred_std = np.zeros((len(blocks), len(fracs)))
    rel = np.zeros((len(blocks), len(fracs)))
    lookup = {(r["block"], r["frac"]): r for r in settings}
    for i, block in enumerate(blocks):
        for j, frac in enumerate(fracs):
            row = lookup[(block, frac)]
            pred[i, j] = row["mean_pred_agree"]
            kl[i, j] = row["mean_kl"]
            pred_std[i, j] = row["std_pred_agree"]
            rel[i, j] = row["mean_rel_l2"]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    mats = [
        (pred, "Mean pred_agree", "magma"),
        (kl, "Mean KL", "viridis_r"),
        (pred_std, "Seed std of pred_agree", "cividis_r"),
        (rel, "Mean rel_L2", "plasma_r"),
    ]
    for ax, (mat, title, cmap) in zip(axes.ravel(), mats):
        im = ax.imshow(mat, aspect="auto", cmap=cmap)
        ax.set_title(title)
        ax.set_xticks(range(len(fracs)))
        ax.set_xticklabels([f"{f:.2f}" for f in fracs])
        ax.set_yticks(range(len(blocks)))
        ax.set_yticklabels(blocks)
        ax.set_xlabel("retained fraction")
        ax.set_ylabel("block")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=8, color="white")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_fig(fig, "fig14_vit_sweep_heatmaps", also_tex=True)


def make_curves(settings):
    blocks = sorted({r["block"] for r in settings})
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(blocks)))
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    metric_specs = [
        ("mean_pred_agree", "std_pred_agree", "Mean pred_agree"),
        ("mean_kl", "std_kl", "Mean KL"),
        ("mean_rel_l2", None, "Mean rel_L2"),
        ("mean_fvu", None, "Mean FVU"),
    ]
    for ax, (metric, spread, title) in zip(axes.ravel(), metric_specs):
        for color, block in zip(colors, blocks):
            rows = sorted([r for r in settings if r["block"] == block], key=lambda x: x["frac"])
            x = [r["frac"] for r in rows]
            y = [r[metric] for r in rows]
            ax.plot(x, y, "o-", color=color, label=f"block {block}")
            if spread is not None:
                s = [r[spread] for r in rows]
                ax.fill_between(x, [a - b for a, b in zip(y, s)], [a + b for a, b in zip(y, s)], color=color, alpha=0.15)
        ax.set_title(title)
        ax.set_xlabel("retained fraction")
        ax.set_ylabel(metric.replace("_", " "))
    axes[0, 0].legend(fontsize=8)
    save_fig(fig, "fig15_vit_sweep_curves", also_tex=True)


def make_pareto(settings):
    blocks = sorted({r["block"] for r in settings})
    colors = {b: c for b, c in zip(blocks, plt.cm.tab10(np.linspace(0, 1, len(blocks))))}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7))
    for row in settings:
        size = 700 * row["frac"] + 40
        axes[0].scatter(row["mean_kl"], row["mean_pred_agree"], s=size, color=colors[row["block"]], alpha=0.8)
        axes[0].text(row["mean_kl"], row["mean_pred_agree"], f"b{row['block']}\nf{row['frac']:.2f}", fontsize=7, ha="center", va="center")
        axes[1].scatter(row["mean_rel_l2"], row["mean_pred_agree"], s=size, color=colors[row["block"]], alpha=0.8)
        axes[1].text(row["mean_rel_l2"], row["mean_pred_agree"], f"b{row['block']}\nf{row['frac']:.2f}", fontsize=7, ha="center", va="center")
    axes[0].set_xlabel("mean KL")
    axes[0].set_ylabel("mean pred_agree")
    axes[0].set_title("Pareto view: pred_agree vs KL")
    axes[1].set_xlabel("mean rel_L2")
    axes[1].set_ylabel("mean pred_agree")
    axes[1].set_title("Pareto view: pred_agree vs rel_L2")
    save_fig(fig, "fig16_vit_sweep_pareto", also_tex=True)


def make_ranked(settings):
    top = settings[:12]
    labels = [f"b{r['block']} f{r['frac']:.2f}" for r in top]
    vals = [r["mean_pred_agree"] for r in top]
    errs = [r["std_pred_agree"] for r in top]
    kls = [r["mean_kl"] for r in top]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    y = np.arange(len(top))
    axes[0].barh(y, vals, xerr=errs, color="#2b8cbe")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("mean pred_agree")
    axes[0].set_title("Top settings by mean pred_agree")
    axes[1].barh(y, kls, color="#41ab5d")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("mean KL")
    axes[1].set_title("Top settings by KL among top pred_agree")
    save_fig(fig, "fig17_vit_sweep_ranked", also_tex=True)


def make_block_frac_summary(block_summary, frac_summary):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    bx = [r["block"] for r in block_summary]
    bmean = [r["mean_pred_agree"] for r in block_summary]
    bstd = [r["std_pred_agree"] for r in block_summary]
    axes[0].bar(bx, bmean, yerr=bstd, color="#756bb1")
    axes[0].set_xlabel("block")
    axes[0].set_ylabel("mean pred_agree")
    axes[0].set_title("Aggregate by block")

    fx = [r["frac"] for r in frac_summary]
    fmean = [r["mean_pred_agree"] for r in frac_summary]
    fstd = [r["std_pred_agree"] for r in frac_summary]
    axes[1].bar([f"{x:.2f}" for x in fx], fmean, yerr=fstd, color="#dd1c77")
    axes[1].set_xlabel("retained fraction")
    axes[1].set_ylabel("mean pred_agree")
    axes[1].set_title("Aggregate by retained fraction")
    save_fig(fig, "fig18_vit_sweep_aggregates", also_tex=True)


def make_seed_traces(settings):
    blocks = sorted({r["block"] for r in settings})
    fig, axes = plt.subplots(1, len(blocks), figsize=(15, 3.8), sharey=True)
    if len(blocks) == 1:
        axes = [axes]
    for ax, block in zip(axes, blocks):
        rows = sorted([r for r in settings if r["block"] == block], key=lambda x: x["frac"])
        fracs = [r["frac"] for r in rows]
        for seed_idx in range(3):
            vals = [r["seed_values"][seed_idx] for r in rows]
            ax.plot(fracs, vals, "o-", label=f"seed {seed_idx}")
        ax.plot(fracs, [r["mean_pred_agree"] for r in rows], "k--", lw=1.5, label="mean")
        ax.set_title(f"block {block}")
        ax.set_xlabel("retained fraction")
        ax.set_ylabel("pred_agree")
    axes[0].legend(fontsize=7)
    save_fig(fig, "fig19_vit_sweep_seed_traces", also_tex=True)


def write_caption_sidecars():
    captions = {
        "figure_vit_sweep_heatmaps.caption.txt": "Completed ViT-small sweep heatmaps over block and retained fraction. Mean prediction agreement, mean KL, seed-level agreement spread, and mean relative reconstruction error show a broad stable regime and a strong best region at later blocks and larger retained fractions.",
        "figure_vit_sweep_curves.caption.txt": "Completed ViT-small sweep curves. Mean metrics versus retained fraction, with agreement spread bands where applicable, show monotone improvement with larger budgets and identify block-specific sweet spots.",
        "figure_vit_sweep_pareto.caption.txt": "Completed ViT-small sweep Pareto views. Each point is one block/fraction setting aggregated over three seeds. The best settings jointly achieve high prediction agreement and low KL / relative error rather than winning only one metric.",
    }
    for name, text in captions.items():
        with open(os.path.join(OUT_TEX, name), "w") as f:
            f.write(text + "\n")


def main():
    rows = load_rows()
    settings, block_summary, frac_summary = aggregate(rows)
    write_summaries(rows, settings, block_summary, frac_summary)
    make_heatmaps(settings)
    make_curves(settings)
    make_pareto(settings)
    make_ranked(settings)
    make_block_frac_summary(block_summary, frac_summary)
    make_seed_traces(settings)
    write_caption_sidecars()
    print("VIT SWEEP FIGURES DONE")
    print("best_setting", settings[0])


if __name__ == "__main__":
    main()

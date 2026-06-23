"""Summarize a completed larger-model ViT sweep and generate a figure suite.

This script is intentionally analysis-only: it reads completed `vit_result.json`
artifacts, recovers the requested sweep settings from run directory names, and
writes summaries plus report-ready figures.
"""
import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT_PNG = os.path.join(ROOT, "report", "plots")
OUT_PDF = os.path.join(ROOT, "report", "figures_pdf")
OUT_TEX = os.path.join(ROOT, "analysis", "figures")
RUN_RE = re.compile(
    r"(?P<model>.+?)_b(?P<block>\d+)_f(?P<frac>[0-9.]+)_s(?P<seed>\d+)$"
)


def mean(vals):
    return sum(vals) / len(vals)


def std(vals):
    m = mean(vals)
    return (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5


def ensure_dirs(path):
    out_analysis = os.path.join(path, "analysis")
    for directory in [out_analysis, OUT_PNG, OUT_PDF, OUT_TEX]:
        os.makedirs(directory, exist_ok=True)
    return out_analysis


def load_rows(run_root):
    rows = []
    for path in sorted(glob.glob(os.path.join(run_root, "*", "vit_result.json"))):
        run_dir = os.path.basename(os.path.dirname(path))
        match = RUN_RE.match(run_dir)
        if not match:
            raise RuntimeError(f"Could not parse requested settings from {run_dir}")
        row = json.load(open(path))
        row["path"] = path
        row["run_dir"] = run_dir
        row["req_model"] = match.group("model")
        row["req_block"] = int(match.group("block"))
        row["req_frac"] = float(match.group("frac"))
        row["req_seed"] = int(match.group("seed"))
        rows.append(row)
    if not rows:
        raise RuntimeError(f"No vit_result.json files found under {run_root}")
    return rows


def validate_rows(rows):
    triplets = {(r["req_block"], r["req_frac"], r["req_seed"]) for r in rows}
    if len(triplets) != len(rows):
        raise RuntimeError("Duplicate requested (block, frac, seed) triplets found")


def aggregate(rows):
    by_setting = defaultdict(list)
    by_block = defaultdict(list)
    by_frac = defaultdict(list)
    for row in rows:
        key = (row["req_block"], row["req_frac"])
        by_setting[key].append(row)
        by_block[row["req_block"]].append(row)
        by_frac[row["req_frac"]].append(row)

    settings = []
    for (block, frac), vals in sorted(by_setting.items()):
        pred = [v["pred_agree"] for v in vals]
        kl = [v["kl"] for v in vals]
        rel = [v["rel_l2"] for v in vals]
        fvu = [v["fvu"] for v in vals]
        cos = [v["cos"] for v in vals]
        actual_frac = [v["frac_retained"] for v in vals]
        settings.append({
            "block": block,
            "frac": frac,
            "n": len(vals),
            "mean_pred_agree": mean(pred),
            "std_pred_agree": std(pred),
            "min_pred_agree": min(pred),
            "max_pred_agree": max(pred),
            "mean_kl": mean(kl),
            "std_kl": std(kl),
            "mean_rel_l2": mean(rel),
            "mean_fvu": mean(fvu),
            "mean_cos": mean(cos),
            "mean_actual_frac": mean(actual_frac),
            "std_actual_frac": std(actual_frac),
            "gap_to_max": 1.0 - mean(pred),
            "collapse_count": sum(1 for x in pred if x < 0.5),
            "seed_values": [v["pred_agree"] for v in sorted(vals, key=lambda x: x["req_seed"])],
            "actual_frac_values": [v["frac_retained"] for v in sorted(vals, key=lambda x: x["req_seed"])],
            "kl_values": [v["kl"] for v in sorted(vals, key=lambda x: x["req_seed"])],
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
            "collapse_count": sum(1 for x in pred if x < 0.5),
        })

    frac_summary = []
    for frac, vals in sorted(by_frac.items()):
        pred = [v["pred_agree"] for v in vals]
        kl = [v["kl"] for v in vals]
        actual = [v["frac_retained"] for v in vals]
        frac_summary.append({
            "frac": frac,
            "mean_pred_agree": mean(pred),
            "std_pred_agree": std(pred),
            "mean_kl": mean(kl),
            "mean_actual_frac": mean(actual),
            "std_actual_frac": std(actual),
            "collapse_count": sum(1 for x in pred if x < 0.5),
            "gap_to_max": 1.0 - mean(pred),
        })

    return settings, block_summary, frac_summary


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def write_summaries(rows, settings, block_summary, frac_summary, out_analysis):
    write_csv(
        os.path.join(out_analysis, "vit_scale_per_run.csv"),
        rows,
        [
            "req_model", "req_block", "req_frac", "req_seed", "model", "block",
            "pred_agree", "kl", "rel_l2", "fvu", "cos", "frac_retained",
            "ntrain", "nval", "path",
        ],
    )
    write_csv(
        os.path.join(out_analysis, "vit_scale_by_setting.csv"),
        settings,
        [
            "block", "frac", "n", "mean_pred_agree", "std_pred_agree",
            "min_pred_agree", "max_pred_agree", "mean_kl", "std_kl",
            "mean_rel_l2", "mean_fvu", "mean_cos", "mean_actual_frac",
            "std_actual_frac", "gap_to_max", "collapse_count",
            "seed_values", "actual_frac_values", "kl_values",
        ],
    )
    write_csv(
        os.path.join(out_analysis, "vit_scale_by_block.csv"),
        block_summary,
        ["block", "mean_pred_agree", "std_pred_agree", "mean_kl", "collapse_count"],
    )
    write_csv(
        os.path.join(out_analysis, "vit_scale_by_frac.csv"),
        frac_summary,
        [
            "frac", "mean_pred_agree", "std_pred_agree", "mean_kl",
            "mean_actual_frac", "std_actual_frac", "collapse_count", "gap_to_max",
        ],
    )
    best = settings[0]
    summary = {
        "num_runs": len(rows),
        "meaningful_max_pred_agree": 1.0,
        "best_setting": best,
        "best_single_run": max(
            rows,
            key=lambda r: (r["pred_agree"], -r["kl"])
        ),
        "most_stable_high_perf_setting": min(
            [r for r in settings if r["mean_pred_agree"] >= 0.95],
            key=lambda r: (r["std_pred_agree"], -r["mean_pred_agree"], r["mean_kl"]),
        ) if any(r["mean_pred_agree"] >= 0.95 for r in settings) else best,
        "best_block": max(block_summary, key=lambda r: r["mean_pred_agree"]),
        "best_frac": max(frac_summary, key=lambda r: r["mean_pred_agree"]),
        "collapse_settings": [
            {
                "block": r["block"],
                "frac": r["frac"],
                "collapse_count": r["collapse_count"],
                "seed_values": r["seed_values"],
                "actual_frac_values": r["actual_frac_values"],
            }
            for r in settings if r["collapse_count"] > 0
        ],
    }
    with open(os.path.join(out_analysis, "vit_scale_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def save_fig(fig, stem, title_prefix):
    fig.tight_layout()
    png_stem = f"{title_prefix}_{stem}"
    fig.savefig(os.path.join(OUT_PNG, f"{png_stem}.png"))
    fig.savefig(os.path.join(OUT_PDF, f"{png_stem}.pdf"))
    fig.savefig(os.path.join(OUT_TEX, f"{png_stem}.pdf"))
    plt.close(fig)
    return png_stem


def make_heatmaps(settings, tag):
    blocks = sorted({r["block"] for r in settings})
    fracs = sorted({r["frac"] for r in settings})
    lookup = {(r["block"], r["frac"]): r for r in settings}
    pred = np.zeros((len(blocks), len(fracs)))
    kl = np.zeros((len(blocks), len(fracs)))
    pred_std = np.zeros((len(blocks), len(fracs)))
    collapse = np.zeros((len(blocks), len(fracs)))
    for i, block in enumerate(blocks):
        for j, frac in enumerate(fracs):
            row = lookup[(block, frac)]
            pred[i, j] = row["mean_pred_agree"]
            kl[i, j] = row["mean_kl"]
            pred_std[i, j] = row["std_pred_agree"]
            collapse[i, j] = row["collapse_count"]

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.8))
    mats = [
        (pred, "Mean pred_agree", "magma", ".3f"),
        (kl, "Mean KL", "viridis_r", ".3f"),
        (pred_std, "Seed std of pred_agree", "cividis_r", ".3f"),
        (collapse, "Collapse count (< 0.5 pred_agree)", "Reds", ".0f"),
    ]
    for ax, (mat, title, cmap, fmt) in zip(axes.ravel(), mats):
        im = ax.imshow(mat, aspect="auto", cmap=cmap)
        ax.set_title(title)
        ax.set_xticks(range(len(fracs)))
        ax.set_xticklabels([f"{f:.2f}" for f in fracs])
        ax.set_yticks(range(len(blocks)))
        ax.set_yticklabels(blocks)
        ax.set_xlabel("requested retained fraction")
        ax.set_ylabel("block")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center", fontsize=8, color="white")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return save_fig(fig, "heatmaps", tag)


def make_curves(settings, frac_summary, tag):
    blocks = sorted({r["block"] for r in settings})
    colors = plt.cm.plasma(np.linspace(0.08, 0.92, len(blocks)))
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.8))
    for color, block in zip(colors, blocks):
        rows = sorted([r for r in settings if r["block"] == block], key=lambda x: x["frac"])
        x = [r["frac"] for r in rows]
        y = [r["mean_pred_agree"] for r in rows]
        s = [r["std_pred_agree"] for r in rows]
        axes[0, 0].plot(x, y, "o-", color=color, label=f"block {block}")
        axes[0, 0].fill_between(x, [a - b for a, b in zip(y, s)], [a + b for a, b in zip(y, s)], color=color, alpha=0.13)
        axes[0, 1].plot(x, [r["mean_kl"] for r in rows], "o-", color=color, label=f"block {block}")
        axes[1, 0].plot(x, [r["collapse_count"] for r in rows], "o-", color=color, label=f"block {block}")
        axes[1, 1].plot(x, [r["mean_actual_frac"] for r in rows], "o-", color=color, label=f"block {block}")
    axes[0, 0].set_title("Mean pred_agree vs requested fraction")
    axes[0, 1].set_title("Mean KL vs requested fraction")
    axes[1, 0].set_title("Collapse count vs requested fraction")
    axes[1, 1].set_title("Mean realized fraction vs requested fraction")
    for ax in axes.ravel():
        ax.set_xlabel("requested retained fraction")
    axes[0, 0].set_ylabel("pred_agree")
    axes[0, 1].set_ylabel("KL")
    axes[1, 0].set_ylabel("count of collapsed seeds")
    axes[1, 1].set_ylabel("realized frac_retained")
    axes[0, 0].legend(fontsize=8)

    fracs = [r["frac"] for r in frac_summary]
    axes[1, 1].plot(fracs, fracs, "k--", lw=1, label="requested = realized")
    return save_fig(fig, "curves", tag)


def make_pareto(settings, tag):
    blocks = sorted({r["block"] for r in settings})
    colors = {b: c for b, c in zip(blocks, plt.cm.tab10(np.linspace(0, 1, len(blocks))))}
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.9))
    for row in settings:
        size = 700 * row["frac"] + 50
        marker = "X" if row["collapse_count"] > 0 else "o"
        axes[0].scatter(row["mean_kl"], row["mean_pred_agree"], s=size, color=colors[row["block"]], marker=marker, alpha=0.82)
        axes[0].text(row["mean_kl"], row["mean_pred_agree"], f"b{row['block']}\nf{row['frac']:.2f}", fontsize=7, ha="center", va="center")
        axes[1].scatter(row["mean_rel_l2"], row["mean_pred_agree"], s=size, color=colors[row["block"]], marker=marker, alpha=0.82)
        axes[1].text(row["mean_rel_l2"], row["mean_pred_agree"], f"b{row['block']}\nf{row['frac']:.2f}", fontsize=7, ha="center", va="center")
    axes[0].set_xlabel("mean KL")
    axes[0].set_ylabel("mean pred_agree")
    axes[0].set_title("Pareto view: pred_agree vs KL")
    axes[1].set_xlabel("mean rel_L2")
    axes[1].set_ylabel("mean pred_agree")
    axes[1].set_title("Pareto view: pred_agree vs rel_L2")
    return save_fig(fig, "pareto", tag)


def make_ranked(settings, tag):
    top = settings[:10]
    unstable = sorted(settings, key=lambda r: (-r["collapse_count"], r["mean_pred_agree"]))[:8]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.6))

    labels = [f"b{r['block']} f{r['frac']:.2f}" for r in top]
    vals = [r["mean_pred_agree"] for r in top]
    errs = [r["std_pred_agree"] for r in top]
    y = np.arange(len(top))
    axes[0].barh(y, vals, xerr=errs, color="#2b8cbe")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("mean pred_agree")
    axes[0].set_title("Top settings by mean pred_agree")

    labels2 = [f"b{r['block']} f{r['frac']:.2f}" for r in unstable]
    vals2 = [r["collapse_count"] for r in unstable]
    mins = [r["min_pred_agree"] for r in unstable]
    y2 = np.arange(len(unstable))
    axes[1].barh(y2, vals2, color="#cb181d")
    axes[1].set_yticks(y2)
    axes[1].set_yticklabels(labels2)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("collapsed seeds (< 0.5 pred_agree)")
    axes[1].set_title("Most unstable settings")
    for yy, v, m in zip(y2, vals2, mins):
        axes[1].text(v + 0.03, yy, f"min={m:.3f}", va="center", fontsize=8)
    return save_fig(fig, "ranked", tag)


def make_seed_traces(settings, tag):
    selected = sorted(settings, key=lambda r: (r["block"], r["frac"]))
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 8.4), sharex=True)
    x = np.arange(len(selected))
    labels = [f"b{r['block']}\nf{r['frac']:.2f}" for r in selected]
    for seed_idx in range(max(len(r["seed_values"]) for r in selected)):
        ys = [r["seed_values"][seed_idx] for r in selected]
        axes[0].plot(x, ys, "o-", label=f"seed {seed_idx}")
        af = [r["actual_frac_values"][seed_idx] for r in selected]
        axes[1].plot(x, af, "o-", label=f"seed {seed_idx}")
    collapse_positions = [i for i, r in enumerate(selected) if r["collapse_count"] > 0]
    for ax in axes:
        for pos in collapse_positions:
            ax.axvspan(pos - 0.45, pos + 0.45, color="#fdd0a2", alpha=0.18)
    axes[0].set_title("Per-seed pred_agree across requested settings")
    axes[0].set_ylabel("pred_agree")
    axes[1].set_title("Per-seed realized frac_retained across requested settings")
    axes[1].set_ylabel("realized frac_retained")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[0].legend(fontsize=8)
    return save_fig(fig, "seed_traces", tag)


def make_requested_vs_realized(rows, settings, tag):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
    req = [r["req_frac"] for r in rows]
    actual = [r["frac_retained"] for r in rows]
    pred = [r["pred_agree"] for r in rows]
    sc = axes[0].scatter(req, actual, c=pred, cmap="viridis", s=28, alpha=0.85)
    axes[0].plot([min(req), max(req)], [min(req), max(req)], "k--", lw=1)
    axes[0].set_xlabel("requested retained fraction")
    axes[0].set_ylabel("realized frac_retained")
    axes[0].set_title("Requested vs realized sparsity")
    fig.colorbar(sc, ax=axes[0], label="pred_agree")

    by_frac = sorted(settings, key=lambda r: (r["frac"], r["block"]))
    xpos = np.arange(len(by_frac))
    means = [r["mean_actual_frac"] for r in by_frac]
    errs = [r["std_actual_frac"] for r in by_frac]
    axes[1].errorbar(xpos, means, yerr=errs, fmt="o", color="#756bb1", ecolor="#bcbddc", capsize=3)
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels([f"b{r['block']}\nf{r['frac']:.2f}" for r in by_frac], fontsize=8)
    axes[1].set_ylabel("mean realized frac_retained")
    axes[1].set_title("Realized sparsity by requested setting")
    return save_fig(fig, "fraction_drift", tag)


def write_captions(tag):
    captions = {
        f"{tag}_heatmaps.caption.txt": "Larger-model ViT sweep heatmaps over requested block and retained fraction. Mean prediction agreement, mean KL, seed spread, and collapse count make visible both the strong late-block regime and the seed-fragile collapse regions.",
        f"{tag}_curves.caption.txt": "Larger-model ViT sweep curves. Requested retained fraction improves some settings strongly, but collapse behavior and realized-fraction drift make the story non-monotone outside the stable late-block regime.",
        f"{tag}_pareto.caption.txt": "Larger-model ViT Pareto views. Stable winners jointly achieve high prediction agreement and low KL, while collapse-prone settings stand out as weak tradeoffs rather than merely lower-ranked versions of the same pattern.",
        f"{tag}_ranked.caption.txt": "Larger-model ViT ranking and instability summary. The best stable settings cluster at block 10, while several mid-block settings suffer one- or two-seed collapse events.",
        f"{tag}_seed_traces.caption.txt": "Per-seed traces across requested settings for the larger-model ViT sweep. Good settings remain tightly clustered across seeds, whereas unstable settings bifurcate sharply into success and collapse outcomes.",
        f"{tag}_fraction_drift.caption.txt": "Requested-versus-realized retained-fraction diagnostics for the larger-model ViT sweep. The analysis groups settings by requested fraction recovered from run directory names so that realized-fraction drift does not scramble the sweep.",
    }
    for name, text in captions.items():
        with open(os.path.join(OUT_TEX, name), "w") as f:
            f.write(text + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--tag", default="vit_scale")
    args = ap.parse_args()

    run_root = os.path.join(ROOT, args.run_root) if not os.path.isabs(args.run_root) else args.run_root
    out_analysis = ensure_dirs(run_root)
    rows = load_rows(run_root)
    validate_rows(rows)
    settings, block_summary, frac_summary = aggregate(rows)
    write_summaries(rows, settings, block_summary, frac_summary, out_analysis)
    stems = {
        "heatmaps": make_heatmaps(settings, args.tag),
        "curves": make_curves(settings, frac_summary, args.tag),
        "pareto": make_pareto(settings, args.tag),
        "ranked": make_ranked(settings, args.tag),
        "seed_traces": make_seed_traces(settings, args.tag),
        "fraction_drift": make_requested_vs_realized(rows, settings, args.tag),
    }
    write_captions(args.tag)
    print("VIT SCALE FIGURES DONE")
    print(json.dumps({
        "run_root": run_root,
        "num_runs": len(rows),
        "best_setting": settings[0],
        "figure_stems": stems,
    }, indent=2))


if __name__ == "__main__":
    main()

"""Run a Phase 4 changed ViT family-comparison sweep.

This launcher is the clean "does vector beat field on token grids?" surface.
It compares SAE families on the same ViT blocks / retained fractions / seeds
and writes runs under `runs/phase4_changed/`.
"""
import argparse
import json
import os
import subprocess
import time
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def parse_list(text, cast=str):
    return [cast(x) for x in text.split(",") if x]


def slug_model(name):
    return name.replace("/", "_")


def make_jobs(args):
    jobs = []
    for model in parse_list(args.models, str):
        model_tag = slug_model(model)
        for sae_type in parse_list(args.sae_types, str):
            for block in parse_list(args.blocks, int):
                for frac in parse_list(args.fracs, float):
                    for seed in parse_list(args.seeds, int):
                        frac_tag = f"{frac:.4f}".rstrip("0").rstrip(".")
                        out = os.path.join(
                            args.out_root,
                            f"{model_tag}_{sae_type}_b{block}_f{frac_tag}_s{seed}",
                        )
                        jobs.append({
                            "dataset": args.dataset,
                            "dataset_root": args.dataset_root,
                            "train_dir": args.train_dir,
                            "val_dir": args.val_dir,
                            "train_split": args.train_split,
                            "val_split": args.val_split,
                            "imagenette_size": args.imagenette_size,
                            "model": model,
                            "block": block,
                            "frac": frac,
                            "sae_type": sae_type,
                            "Kmult": args.Kmult,
                            "epochs": args.epochs,
                            "ntrain": args.ntrain,
                            "nval": args.nval,
                            "train_bs": args.train_bs,
                            "val_bs": args.val_bs,
                            "num_workers": args.num_workers,
                            "lr": args.lr,
                            "seed": seed,
                            "data_seed": args.data_seed,
                            "out": out,
                        })
    return jobs


def render_args(job):
    parts = []
    for key, value in job.items():
        if value is None:
            continue
        parts.append(f"--{key}")
        parts.append(str(value))
    return " ".join(parts)


def run(jobs, gpus):
    env_prefix = f"source {ROOT}/env.sh && "
    pending = deque(jobs)
    running = {}
    free = list(gpus)
    logs = []
    while pending or running:
        while free and pending:
            gpu = free.pop(0)
            job = pending.popleft()
            os.makedirs(job["out"], exist_ok=True)
            cmd = env_prefix + f"python {ROOT}/src/vit_sae.py --device cuda:{gpu} {render_args(job)}"
            logf = os.path.join(job["out"], "run.log")
            fh = open(logf, "w")
            proc = subprocess.Popen(["bash", "-lc", cmd], stdout=fh, stderr=subprocess.STDOUT)
            running[gpu] = (proc, job, fh, logf)
            print(f"LAUNCH gpu{gpu} {job['out']}", flush=True)
        time.sleep(2)
        for gpu, (proc, job, fh, logf) in list(running.items()):
            if proc.poll() is not None:
                fh.close()
                rc = proc.returncode
                result_path = os.path.join(job["out"], "vit_result.json")
                ok = rc == 0 and os.path.exists(result_path)
                record = {"job": job, "rc": rc, "ok": ok, "log": logf}
                if os.path.exists(result_path):
                    with open(result_path) as f:
                        record["result"] = json.load(f)
                logs.append(record)
                print(f"FINISH gpu{gpu} rc={rc} ok={ok} {job['out']}", flush=True)
                del running[gpu]
                free.append(gpu)
    return logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="imagenette", choices=["imagenette", "imagefolder"])
    ap.add_argument("--dataset_root", default=None)
    ap.add_argument("--train_dir", default=None)
    ap.add_argument("--val_dir", default=None)
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--val_split", default="val")
    ap.add_argument("--imagenette_size", default="160px")
    ap.add_argument("--models", default="vit_small_patch16_224")
    ap.add_argument("--sae_types", default="field,vector")
    ap.add_argument("--blocks", default="4,10")
    ap.add_argument("--fracs", default="0.05,0.08,0.12")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--Kmult", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--ntrain", type=int, default=2000)
    ap.add_argument("--nval", type=int, default=500)
    ap.add_argument("--train_bs", type=int, default=32)
    ap.add_argument("--val_bs", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data_seed", type=int, default=0)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--out_root", default=os.path.join(ROOT, "runs", "phase4_changed"))
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    jobs = make_jobs(args)
    gpus = parse_list(args.gpus, int)
    print(f"{len(jobs)} jobs on gpus {gpus}", flush=True)
    logs = run(jobs, gpus)
    summary = {
        "dataset": args.dataset,
        "dataset_root": args.dataset_root,
        "models": parse_list(args.models, str),
        "sae_types": parse_list(args.sae_types, str),
        "blocks": parse_list(args.blocks, int),
        "fracs": parse_list(args.fracs, float),
        "seeds": parse_list(args.seeds, int),
        "job_count": len(jobs),
        "ok_count": sum(1 for x in logs if x["ok"]),
        "logs": logs,
    }
    with open(os.path.join(args.out_root, "grid_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"GRID DONE {summary['ok_count']}/{summary['job_count']} ok", flush=True)


if __name__ == "__main__":
    main()

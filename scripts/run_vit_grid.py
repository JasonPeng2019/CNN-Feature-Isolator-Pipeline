"""Run a configurable ViT SAE sweep across blocks, sparsity levels, and seeds.

Each job calls `src/vit_sae.py` and writes one run directory with `vit_result.json`.
The launcher writes `grid_summary.json` after all jobs finish.
"""
import argparse
import json
import os
import subprocess
import time
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def parse_int_list(text):
    return [int(x) for x in text.split(",") if x]


def parse_float_list(text):
    return [float(x) for x in text.split(",") if x]


def make_jobs(args):
    jobs = []
    for block in parse_int_list(args.blocks):
        for frac in parse_float_list(args.fracs):
            for seed in parse_int_list(args.seeds):
                frac_tag = f"{frac:.4f}".rstrip("0").rstrip(".")
                out = os.path.join(args.out_root, f"b{block}_f{frac_tag}_s{seed}")
                jobs.append({
                    "model": args.model,
                    "block": block,
                    "frac": frac,
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
    for k, v in job.items():
        parts.append(f"--{k}")
        parts.append(str(v))
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
    ap.add_argument("--model", default="vit_small_patch16_224")
    ap.add_argument("--blocks", default="2,4,6,8,10")
    ap.add_argument("--fracs", default="0.01,0.02,0.05,0.08,0.12")
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
    ap.add_argument("--out_root", default=os.path.join(ROOT, "runs", "vit_sweep"))
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    jobs = make_jobs(args)
    gpus = parse_int_list(args.gpus)
    print(f"{len(jobs)} jobs on gpus {gpus}", flush=True)
    logs = run(jobs, gpus)
    summary = {
        "model": args.model,
        "blocks": parse_int_list(args.blocks),
        "fracs": parse_float_list(args.fracs),
        "seeds": parse_int_list(args.seeds),
        "job_count": len(jobs),
        "ok_count": sum(1 for x in logs if x["ok"]),
        "logs": logs,
    }
    with open(os.path.join(args.out_root, "grid_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"GRID DONE {summary['ok_count']}/{summary['job_count']} ok", flush=True)


if __name__ == "__main__":
    main()

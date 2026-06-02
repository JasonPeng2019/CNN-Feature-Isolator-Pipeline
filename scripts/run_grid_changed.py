"""Changed grid runner for isolated experiment sweeps."""
import argparse
import json
import os
import subprocess
import time
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def make_jobs_phase2_rf_fair(out_root):
    jobs = []
    for sec in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        for frac in [0.02, 0.05, 0.10]:
            jobs.append({
                "section": sec,
                "sae_type": "field",
                "mask": "rf_frac",
                "rf_frac": frac,
                "target_frac": frac,
                "Kmult": 8,
                "n_blocks": 3,
                "epochs": 40,
                "out": os.path.join(out_root, f"RFP_{sec}_f{frac}"),
            })
    return jobs


def make_jobs_patch_overlap(out_root):
    jobs = []
    for sec in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        for frac in [0.02, 0.05, 0.10]:
            for mult in [1.0, 2.0, 3.0]:
                jobs.append({
                    "section": sec,
                    "sae_type": "patchvec",
                    "mask": "global",
                    "patch_mode": "overlap",
                    "patch_multiplier": mult,
                    "target_frac": frac,
                    "Kmult": 8,
                    "epochs": 40,
                    "out": os.path.join(out_root, f"PO_{sec}_m{mult:g}_f{frac}"),
                })
    return jobs


def make_jobs_patch_depth(out_root):
    jobs = []
    for sec in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        for frac in [0.02, 0.05, 0.10]:
            for schedule in ["schedule_a", "schedule_b", "schedule_c", "schedule_d"]:
                jobs.append({
                    "section": sec,
                    "sae_type": "patchvec",
                    "mask": "global",
                    "patch_mode": "overlap",
                    "depth_schedule": schedule,
                    "target_frac": frac,
                    "Kmult": 8,
                    "epochs": 40,
                    "out": os.path.join(out_root, f"PD_{schedule}_{sec}_f{frac}"),
                })
    return jobs


def make_jobs_patch_disjoint(out_root):
    jobs = []
    for sec in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        for mult in [0.5, 1.0, 2.0, 3.0]:
            jobs.append({
                "section": sec,
                "sae_type": "patchvec",
                "mask": "global",
                "patch_mode": "disjoint",
                "patch_multiplier": mult,
                "target_frac": 0.05,
                "Kmult": 8,
                "epochs": 40,
                "out": os.path.join(out_root, f"PDJ_{sec}_m{mult:g}_f0.05"),
            })
    return jobs


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
            args = " ".join(f"--{k} {v}" for k, v in job.items())
            cmd = env_prefix + f"python {ROOT}/src/train_sae_changed.py --device cuda:{gpu} {args}"
            logf = job["out"] + ".log"
            os.makedirs(os.path.dirname(logf), exist_ok=True)
            fh = open(logf, "w")
            p = subprocess.Popen(["bash", "-lc", cmd], stdout=fh, stderr=subprocess.STDOUT)
            running[gpu] = (p, job, fh)
            print(f"LAUNCH gpu{gpu} {job['out']}", flush=True)
        time.sleep(5)
        for gpu, (p, job, fh) in list(running.items()):
            if p.poll() is not None:
                fh.close()
                rc = p.returncode
                ok = os.path.exists(os.path.join(job["out"], "result.json"))
                print(f"FINISH gpu{gpu} rc={rc} ok={ok} {job['out']}", flush=True)
                logs.append({"job": job, "rc": rc, "ok": ok})
                del running[gpu]
                free.append(gpu)
    return logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["phase2_rf_fair", "patch_overlap", "patch_depth", "patch_disjoint"])
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--gpus", default="0,1")
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]
    makers = {
        "phase2_rf_fair": make_jobs_phase2_rf_fair,
        "patch_overlap": make_jobs_patch_overlap,
        "patch_depth": make_jobs_patch_depth,
        "patch_disjoint": make_jobs_patch_disjoint,
    }
    jobs = makers[args.phase](args.out_root)
    print(f"{len(jobs)} jobs on gpus {gpus}", flush=True)
    logs = run(jobs, gpus)
    os.makedirs(args.out_root, exist_ok=True)
    json.dump(logs, open(os.path.join(args.out_root, "grid_summary.json"), "w"))
    print(f"GRID CHANGED DONE {sum(l['ok'] for l in logs)}/{len(logs)} ok", flush=True)


if __name__ == "__main__":
    main()

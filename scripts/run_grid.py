"""Run a list of train_sae jobs across multiple GPUs with bounded concurrency.

Each job is a dict of CLI args for src/train_sae.py. GPUs are assigned round-robin;
at most one job per GPU at a time.
"""
import argparse, json, os, subprocess, sys, time
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def make_jobs_phase1(out_root):
    """Anchor: vector SAE + global mask + independent, all sections x target fracs."""
    jobs = []
    for sec in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        for frac in [0.02, 0.05, 0.10]:
            jobs.append({
                "section": sec, "sae_type": "vector", "mask": "global",
                "target_frac": frac, "Kmult": 8, "epochs": 40,
                "out": os.path.join(out_root, f"E3_{sec}_f{frac}"),
            })
    return jobs


def make_jobs_phase2(out_root):
    """Branch the recon grid: E1 field+global, E2 field+rf, E4 vector+rf, all sections."""
    secs = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    jobs = []
    for sec in secs:
        for frac in [0.02, 0.05]:                      # E1: Field + Global
            jobs.append({"section": sec, "sae_type": "field", "mask": "global",
                         "target_frac": frac, "Kmult": 8, "n_blocks": 3, "epochs": 40,
                         "out": os.path.join(out_root, f"E1_{sec}_f{frac}")})
        for krf in [2, 4]:                             # E2: Field + RF
            jobs.append({"section": sec, "sae_type": "field", "mask": "rf",
                         "k_rf": krf, "Kmult": 8, "n_blocks": 3, "epochs": 40,
                         "out": os.path.join(out_root, f"E2_{sec}_k{krf}")})
        for krf in [2, 4]:                             # E4: Vector + RF
            jobs.append({"section": sec, "sae_type": "vector", "mask": "rf",
                         "k_rf": krf, "Kmult": 8, "epochs": 40,
                         "out": os.path.join(out_root, f"E4_{sec}_k{krf}")})
    return jobs


def make_jobs_pareto(out_root):
    jobs = []
    for sec in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        for frac in [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20]:
            jobs.append({"section": sec, "sae_type": "field", "mask": "global",
                         "target_frac": frac, "Kmult": 8, "n_blocks": 3, "epochs": 40,
                         "out": os.path.join(out_root, f"F_{sec}_f{frac}")})
    return jobs


def make_jobs_seeds(out_root):
    jobs = []
    for sec in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        for seed in [0, 1, 2]:
            jobs.append({"section": sec, "sae_type": "field", "mask": "global",
                         "target_frac": 0.05, "Kmult": 8, "n_blocks": 3, "epochs": 40,
                         "seed": seed, "out": os.path.join(out_root, f"S_{sec}_s{seed}")})
    return jobs


def make_jobs_r20c10(out_root):
    jobs = []
    for sec in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        for frac in [0.02, 0.05, 0.10]:
            jobs.append({"section": sec, "sae_type": "field", "mask": "global",
                         "target_frac": frac, "Kmult": 8, "n_blocks": 3, "epochs": 40,
                         "acts": "runs/acts_r20_c10", "ckpt": "runs/backbone_r20_c10/best.pt",
                         "dataset": "cifar10", "out": os.path.join(out_root, f"R_{sec}_f{frac}")})
    return jobs


def run(jobs, gpus):
    env_prefix = f"source {ROOT}/env.sh && "
    pending = deque(jobs)
    running = {}  # gpu -> (proc, job)
    free = list(gpus)
    logs = []
    while pending or running:
        while free and pending:
            gpu = free.pop(0); job = pending.popleft()
            args = " ".join(f"--{k} {v}" for k, v in job.items())
            cmd = env_prefix + f"python {ROOT}/src/train_sae.py --device cuda:{gpu} {args}"
            logf = job["out"] + ".log"; os.makedirs(os.path.dirname(logf), exist_ok=True)
            fh = open(logf, "w")
            p = subprocess.Popen(["bash", "-lc", cmd], stdout=fh, stderr=subprocess.STDOUT)
            running[gpu] = (p, job, fh)
            print(f"LAUNCH gpu{gpu} {job['out']}", flush=True)
        time.sleep(5)
        for gpu, (p, job, fh) in list(running.items()):
            if p.poll() is not None:
                fh.close(); rc = p.returncode
                ok = os.path.exists(os.path.join(job["out"], "result.json"))
                print(f"FINISH gpu{gpu} rc={rc} ok={ok} {job['out']}", flush=True)
                logs.append({"job": job, "rc": rc, "ok": ok})
                del running[gpu]; free.append(gpu)
    return logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="phase1")
    ap.add_argument("--out_root", default=os.path.join(ROOT, "runs", "phase1"))
    ap.add_argument("--gpus", default="0,1,2,3")
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]
    jobs = {"phase1": make_jobs_phase1, "phase2": make_jobs_phase2, "pareto": make_jobs_pareto,
            "seeds": make_jobs_seeds, "r20c10": make_jobs_r20c10}[args.phase](args.out_root)
    print(f"{len(jobs)} jobs on gpus {gpus}", flush=True)
    logs = run(jobs, gpus)
    json.dump(logs, open(os.path.join(args.out_root, "grid_summary.json"), "w"))
    print(f"GRID DONE {sum(l['ok'] for l in logs)}/{len(logs)} ok", flush=True)


if __name__ == "__main__":
    main()

"""Bounded real-CUDA smoke path for the Level-2 CNN redo contract.

This module is intentionally separate from the matrix launcher.  It prepares a
new, tiny cache derived from an existing activation cache and can then execute
five sequential trainer cells (the four final families plus the strict vector
parent), followed by a direct CUDA chain-loss smoke.  It never uses CPU
fallback, never starts tmux, and never writes below an existing run root.

The default is ``--dry-run``: only a command/evidence plan is written.  An
actual run requires the explicit ``--execute`` flag and a fresh ``--out``
directory.  ViT is not executed here; its local pretrained-weight prerequisite
is reported in the plan so a missing download cannot be mistaken for a pass.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from launch_sae_redo_level2 import enumerate_nvidia_gpus  # noqa: E402


SMOKE_FAMILIES = (
    "field_old", "field_strict", "vector_context_strict",
    "field_recovery", "vector_context",
)
FINAL_FAMILIES = ("field_old", "field_strict", "field_recovery", "vector_context")
REQUIRED_SAE_ARTIFACTS = (
    "sae.pt", "result.json", "run_metadata.json", "continuation.json",
    "normalization.json", "sparse_message.pt", "replay.json",
    "fidelity_result.json", "controls.json",
)
REQUIRED_CHAIN_ARTIFACTS = ("chain_inputs.pt", "chain_smoke.json")
DEFAULT_JOB_TIMEOUT_SECONDS = 900
DEFAULT_CLEANUP_GRACE_SECONDS = 15
SMOKE_COLLAPSE_POLICY = {
    "mode": "smoke_disabled", "threshold": None, "max_recovery_retries": 0,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _source_spec(backbone: str) -> dict[str, str]:
    if backbone == "r20":
        return {
            "backbone_id": "resnet20_cifar10",
            "acts": str((ROOT / "runs/acts_r20_c10").resolve()),
            "checkpoint": str((ROOT / "runs/backbone_r20_c10/best.pt").resolve()),
        }
    if backbone == "r110":
        return {
            "backbone_id": "resnet110_cifar100",
            "acts": str((ROOT / "runs/acts_r110_c100").resolve()),
            "checkpoint": str((ROOT / "runs/backbone_r110_c100/best.pt").resolve()),
        }
    raise ValueError(f"unsupported CNN smoke backbone: {backbone!r}")


def vit_pretrained_weight_report() -> dict:
    """Report local ViT weight availability without importing timm or downloading."""
    configured = os.environ.get("SAE_REDO_VIT_WEIGHTS", "").strip()
    candidates = [Path(configured)] if configured else []
    candidates.extend([
        ROOT / "weights/vit_small_patch16_224.pth",
        ROOT / "weights/vit_base_patch16_224.pth",
        Path.home() / ".cache/torch/hub/checkpoints/vit_small_patch16_224.pth",
        Path.home() / ".cache/torch/hub/checkpoints/vit_base_patch16_224.pth",
    ])
    existing = [str(path.resolve()) for path in candidates if path.is_file()]
    return {
        "status": "available" if existing else "missing",
        "searched": [str(path) for path in candidates],
        "paths": existing,
        "action": "use an existing local weight or explicitly authorize a network download",
    }


def _quote_command(gpu: int, argv: list[object]) -> str:
    return "CUDA_VISIBLE_DEVICES=" + str(gpu) + " " + " ".join(
        shlex.quote(str(value)) for value in argv
    )


def _family_output(root: Path, family: str) -> Path:
    return root / "families" / family


def build_smoke_plan(backbone: str, output_root: Path | str, *, gpu: int = 0,
                     ntrain: int = 8, nval: int = 8,
                     chain_samples: int = 2,
                     job_timeout_seconds: int = DEFAULT_JOB_TIMEOUT_SECONDS,
                     cleanup_grace_seconds: int = DEFAULT_CLEANUP_GRACE_SECONDS) -> dict:
    """Build a deterministic, CUDA-only plan without touching the filesystem."""
    if int(ntrain) != 8 or int(nval) != 8:
        raise ValueError("the contract smoke is fixed to 8 train and 8 validation samples")
    if int(chain_samples) < 1 or int(chain_samples) > 8:
        raise ValueError("chain_samples must be in [1, 8]")
    if int(job_timeout_seconds) < 1:
        raise ValueError("job_timeout_seconds must be positive")
    if int(cleanup_grace_seconds) < 1 or int(cleanup_grace_seconds) > 120:
        raise ValueError("cleanup_grace_seconds must be in [1, 120]")
    source = _source_spec(backbone)
    root = Path(output_root).resolve()
    prepared = root / "prepared_acts_q1"
    jobs: list[dict] = []
    paths = {family: _family_output(root, family) for family in SMOKE_FAMILIES}

    def add_sae(family: str, *, depends_on: list[str] | None = None,
                parent_family: str | None = None,
                phase: str = "strict") -> None:
        output = paths[family]
        argv = [
            sys.executable, str(SRC / "train_sae_redo.py"),
            "--acts", prepared, "--ckpt", source["checkpoint"],
            "--section", "Q1", "--backbone_id", source["backbone_id"],
            "--sae_type", family, "--budget", "0.08", "--seed", "0",
            "--ntrain", "8", "--nval", "8", "--train_bs", "8", "--val_bs", "8",
            "--Kmult", "8" if family == "field_old" else "4",
            "--smoke", "--smoke_disable_collapse",
            "--phase", phase, "--device", "cuda:0", "--require_cuda",
            "--out", output,
        ]
        if parent_family is not None:
            argv += [
                "--parent_checkpoint", paths[parent_family] / "sae.pt",
                "--parent_family", parent_family, "--parent_budget", "0.08",
            ]
        job_id = f"cuda-smoke:{backbone}:Q1:{family}:b08:s0"
        jobs.append({
            "job_id": job_id, "kind": "sae", "family": family,
            "phase": phase, "output": str(output),
            "depends_on": list(depends_on or []),
            "required_artifacts": list(REQUIRED_SAE_ARTIFACTS),
            "smoke_collapse_policy": dict(SMOKE_COLLAPSE_POLICY),
            "timeout_seconds": int(job_timeout_seconds),
            "cleanup_grace_seconds": int(cleanup_grace_seconds),
            "argv": [str(value) for value in argv],
            "command": _quote_command(gpu, argv),
        })

    add_sae("field_old")
    add_sae("field_strict")
    add_sae("vector_context_strict")
    add_sae("field_recovery", depends_on=[jobs[1]["job_id"]],
            parent_family="field_strict", phase="recovery_joint")
    add_sae("vector_context", depends_on=[jobs[2]["job_id"]],
            parent_family="vector_context_strict", phase="vector_joint")

    chain_output = root / "chain"
    chain_argv = [
        sys.executable, str(Path(__file__).resolve()), "--chain-only",
        "--source-acts", source["acts"], "--backbone", backbone,
        "--samples", chain_samples, "--gpu", gpu, "--device", "cuda:0", "--require_cuda",
        "--out", chain_output,
    ]
    jobs.append({
        "job_id": f"cuda-smoke:{backbone}:chain", "kind": "chain",
        "family": "field_strict+field_old", "phase": "regrounded_and_raw",
        "output": str(chain_output),
        "depends_on": [job["job_id"] for job in jobs],
        "required_artifacts": list(REQUIRED_CHAIN_ARTIFACTS),
        "timeout_seconds": int(job_timeout_seconds),
        "cleanup_grace_seconds": int(cleanup_grace_seconds),
        "argv": [str(value) for value in chain_argv],
        "command": _quote_command(gpu, chain_argv),
    })
    plan = {
        "schema": "sae-redo-level2-cuda-smoke-plan-v1",
        "backbone": backbone, "gpu": int(gpu),
        "output_root": str(root),
        "source": source, "prepared_cache": str(prepared),
        "families": list(FINAL_FAMILIES),
        "bounded_config": {"ntrain": int(ntrain), "nval": int(nval),
                           "chain_samples": int(chain_samples)},
        "jobs": jobs,
        "execution": {
            "cuda_only": True, "tmux": False,
            "job_timeout_seconds": int(job_timeout_seconds),
            "cleanup_grace_seconds": int(cleanup_grace_seconds),
            "timeout_basis": (
                "900 seconds gives the tiny Q1 fixed curriculum and post-selection "
                "artifact pass headroom; cleanup is capped at 15 seconds."
            ),
            "smoke_collapse_policy": dict(SMOKE_COLLAPSE_POLICY),
            "vit_pretrained_weight_check": vit_pretrained_weight_report(),
        },
    }
    plan["sha256"] = _canonical_digest(plan)
    return plan


def _require_local_vit_inputs(model: str, weights: Path | str,
                              data_root: Path | str) -> tuple[dict, Path]:
    """Validate the no-download ViT smoke inputs before constructing jobs."""
    if weights is None:
        raise FileNotFoundError(
            "local ViT weights are required for ViT smoke; no network download is permitted"
        )
    weight_path = Path(weights).expanduser().resolve()
    if not weight_path.is_file():
        raise FileNotFoundError(f"local ViT weights not found: {weight_path}")
    data_path = Path(data_root).expanduser().resolve()
    missing = [split for split in ("train", "val")
               if not (data_path / split).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Imagenette ImageFolder data is missing split directories {missing}: {data_path}"
        )
    weight_record = {
        "source": "local",
        "model": str(model),
        "path": str(weight_path),
        "sha256": _sha256(weight_path),
    }
    return weight_record, data_path


def build_vit_smoke_plan(model: str, output_root: Path | str, *, weights: Path | str,
                         gpu: int = 0, block: int = 2,
                         data_root: Path | str = ROOT / "data/imagenette2-160",
                         ntrain: int = 8, nval: int = 8, ntest: int = 8,
                         job_timeout_seconds: int = DEFAULT_JOB_TIMEOUT_SECONDS,
                         cleanup_grace_seconds: int = DEFAULT_CLEANUP_GRACE_SECONDS) -> dict:
    """Build a bounded local-weight ViT plan without importing timm/downloading."""
    if model not in {"vit_small_patch16_224", "vit_base_patch16_224"}:
        raise ValueError(f"unsupported ViT smoke model: {model!r}")
    if int(ntrain) != 8 or int(nval) != 8 or int(ntest) != 8:
        raise ValueError("the contract ViT smoke is fixed to 8 train, 8 validation, and 8 fidelity samples")
    if int(block) < 0:
        raise ValueError("ViT block must be non-negative")
    if int(job_timeout_seconds) < 1:
        raise ValueError("job_timeout_seconds must be positive")
    if int(cleanup_grace_seconds) < 1 or int(cleanup_grace_seconds) > 120:
        raise ValueError("cleanup_grace_seconds must be in [1, 120]")
    weight_record, data_path = _require_local_vit_inputs(model, weights, data_root)
    root = Path(output_root).resolve()
    paths = {family: _family_output(root, family) for family in SMOKE_FAMILIES}
    jobs: list[dict] = []

    def add_sae(family: str, *, depends_on: list[str] | None = None,
                parent_family: str | None = None,
                phase: str = "strict") -> None:
        output = paths[family]
        argv = [
            sys.executable, str(SRC / "vit_sae_redo.py"),
            "--model", model, "--block", str(block), "--backbone_id", model,
            "--weights", weight_record["path"], "--data_root", str(data_path),
            "--sae_type", family, "--budget", "0.08", "--seed", "0",
            "--ntrain", "8", "--nval", "8", "--ntest", "8",
            "--train_bs", "8", "--val_bs", "8", "--num_workers", "0",
            "--Kmult", "8" if family == "field_old" else "4",
            "--smoke", "--smoke_disable_collapse", "--phase", phase,
            "--device", "cuda:0", "--require_cuda", "--out", output,
        ]
        if parent_family is not None:
            argv += [
                "--parent_checkpoint", paths[parent_family] / "sae.pt",
                "--parent_family", parent_family, "--parent_budget", "0.08",
            ]
        job_id = f"cuda-smoke:{model}:block{block}:{family}:b08:s0"
        jobs.append({
            "job_id": job_id, "kind": "vit_sae", "family": family,
            "phase": phase, "output": str(output),
            "depends_on": list(depends_on or []),
            "required_artifacts": list(REQUIRED_SAE_ARTIFACTS),
            "weights": dict(weight_record),
            "smoke_collapse_policy": dict(SMOKE_COLLAPSE_POLICY),
            "timeout_seconds": int(job_timeout_seconds),
            "cleanup_grace_seconds": int(cleanup_grace_seconds),
            "argv": [str(value) for value in argv],
            "command": _quote_command(gpu, argv),
        })

    add_sae("field_old")
    add_sae("field_strict")
    add_sae("vector_context_strict")
    add_sae("field_recovery", depends_on=[jobs[1]["job_id"]],
            parent_family="field_strict", phase="recovery_joint")
    add_sae("vector_context", depends_on=[jobs[2]["job_id"]],
            parent_family="vector_context_strict", phase="vector_joint")
    plan = {
        "schema": "sae-redo-level2-vit-cuda-smoke-plan-v1",
        "model": model, "block": int(block), "gpu": int(gpu),
        "output_root": str(root), "data_root": str(data_path),
        "weights": weight_record,
        "families": list(FINAL_FAMILIES),
        "bounded_config": {"ntrain": int(ntrain), "nval": int(nval),
                           "ntest": int(ntest)},
        "jobs": jobs,
        "execution": {
            "cuda_only": True, "tmux": False, "network_download": False,
            "job_timeout_seconds": int(job_timeout_seconds),
            "cleanup_grace_seconds": int(cleanup_grace_seconds),
            "timeout_basis": (
                "900 seconds gives each tiny ImageFolder cell fixed-curriculum and "
                "post-selection artifact headroom; cleanup is capped at 15 seconds."
            ),
            "smoke_collapse_policy": dict(SMOKE_COLLAPSE_POLICY),
            "local_weight_sha256": weight_record["sha256"],
        },
    }
    plan["sha256"] = _canonical_digest(plan)
    return plan


def require_cuda_gpu(gpu: int) -> None:
    """Fail closed unless the requested physical NVIDIA GPU is discoverable."""
    available = enumerate_nvidia_gpus()
    if int(gpu) not in available:
        raise RuntimeError(f"requested GPU {gpu} is not reported by nvidia-smi; available={available}")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA runtime is unavailable; refusing CPU fallback")


def build_job_evidence(job: dict, *, returncode: int, started: str, ended: str,
                       status: str | None = None, error: str | None = None,
                       cleanup: dict | None = None,
                       timeout_seconds: int | None = None) -> dict:
    """Build durable evidence and reject a false successful cell."""
    output = Path(job["output"])
    required = list(job.get("required_artifacts", ()))
    missing = [name for name in required if not (output / name).is_file()]
    status = status or ("passed" if int(returncode) == 0 else "failed")
    if status == "passed" and missing:
        raise ValueError(f"successful smoke job is missing artifacts: {missing}")
    artifacts = {
        name: _sha256(output / name)
        for name in required if (output / name).is_file()
    }
    record = {
        "schema": "sae-redo-level2-cuda-smoke-job-v1",
        "job_id": job["job_id"], "kind": job["kind"],
        "family": job.get("family", "unknown"), "phase": job.get("phase", "unknown"),
        "returncode": int(returncode), "status": status,
        "started": started, "ended": ended,
        "output": str(output), "missing_artifacts": missing,
        "artifacts": artifacts,
    }
    if error:
        record["error"] = str(error)
    if cleanup is not None:
        record["cleanup"] = cleanup
    if timeout_seconds is not None:
        record["timeout_seconds"] = int(timeout_seconds)
    return record


def state_dict_unchanged(module, snapshot: dict[str, object]) -> bool:
    """Compare a device-independent CPU snapshot with a live module state."""
    return all(
        name in snapshot and torch_value.detach().cpu().equal(snapshot[name].detach().cpu())
        for name, torch_value in module.state_dict().items()
    )


def _write_json_new(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _prepare_subset(source_acts: Path, destination: Path, *, train_count: int = 16,
                    test_count: int = 8) -> None:
    """Create an isolated Q1 cache from real cached tensors, once per smoke root."""
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    import torch
    from cache_activations import load_cache_metadata, write_cache_metadata
    source_metadata = load_cache_metadata(str(source_acts))
    if int(source_metadata["train_count"]) < train_count or int(source_metadata["test_count"]) < test_count:
        raise ValueError("source cache is smaller than the requested smoke subset")
    for split, count in (("train", train_count), ("test", test_count)):
        for name in ("Q1", "logits", "labels"):
            source = source_acts / f"{split}_{name}.pt"
            if not source.is_file():
                raise FileNotFoundError(source)
            tensor = torch.load(source, map_location="cpu", weights_only=True)
            torch.save(tensor[:count].contiguous(), destination / source.name)
    write_cache_metadata(
        str(destination), dataset=source_metadata["dataset"],
        backbone=source_metadata["backbone"],
        train_count=train_count, test_count=test_count,
    )


def _json_safe(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def run_chain_cuda_smoke(source_acts: Path | str, output: Path | str, *, backbone: str,
                         samples: int, gpu: int, require_cuda: bool = True) -> dict:
    """Run the real Level-2 chain loss/models on a tiny CUDA cache slice."""
    if require_cuda:
        require_cuda_gpu(gpu)
    import torch
    import train_chain_level2 as chain
    import transitions
    import sae

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    source = Path(source_acts)
    if not source.is_dir():
        raise FileNotFoundError(source)
    torch.manual_seed(20260827)
    activations = {}
    for section in chain.SECTIONS:
        source_path = source / f"train_{section}.pt"
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        tensor = torch.load(source_path, map_location="cpu", weights_only=True)
        activations[section] = tensor[:samples].float()
    labels = torch.load(source / "train_labels.pt", map_location="cpu", weights_only=True)[:samples]
    input_path = output / "chain_inputs.pt"
    torch.save({"schema": "sae-redo-level2-cuda-chain-input-v1",
                "activations": activations, "labels": labels}, input_path)
    device = torch.device("cuda:0")
    activations = {section: value.to(device) for section, value in activations.items()}
    results = {}
    for family in ("field_strict", "field_old"):
        torch.manual_seed(20260827)
        saes = {}
        means, stds, target_ms = {}, {}, {}
        for section in chain.SECTIONS:
            channels, height, width = activations[section].shape[1:]
            multiplier = 8 if family == "field_old" else 4
            saes[section] = sae.build_sae(
                family, channels, multiplier * channels, n_blocks=3,
            ).to(device)
            chain.freeze_sae(saes[section])
            means[section] = torch.zeros(1, channels, 1, 1, device=device)
            stds[section] = torch.ones(1, channels, 1, 1, device=device)
            target_ms[section] = chain.original_node_budget(
                0.01, channels, height, width,
                saes[section].K * height * width,
            )
        predictors = {
            (source_section, destination): transitions.TransitionPredictor(
                saes[source_section].K, saes[destination].K,
                activations[source_section].shape[-1], activations[destination].shape[-1],
            ).to(device)
            for source_section, destination in chain.PAIRS
        }
        state_before = {
            section: {name: value.detach().cpu().clone()
                      for name, value in net.state_dict().items()}
            for section, net in saes.items()
        }
        loss, artifact = chain.chain_training_loss(
            predictors, saes, means, stds, target_ms, activations,
            scheduled_sampling_probability=1.0, carrier_mode="regrounded",
        )
        gradient_norms = {}
        for index, transition_loss in enumerate(artifact["losses"]):
            for predictor in predictors.values():
                for parameter in predictor.parameters():
                    parameter.grad = None
            transition_loss.backward(retain_graph=True)
            relevant = list(chain.PAIRS[:index + 1])
            gradient_norms[str(index)] = {
                f"{source_section}->{destination}": float(sum(
                    parameter.grad.detach().float().norm().item()
                    for parameter in predictors[(source_section, destination)].parameters()
                    if parameter.grad is not None
                ))
                for source_section, destination in relevant
            }
        for index, norms in gradient_norms.items():
            if any(not torch.isfinite(torch.tensor(value)) or value <= 0 for value in norms.values()):
                raise RuntimeError(f"non-finite/zero chain gradient for {family} loss {index}: {norms}")
        _, raw_artifact = chain.chain_training_loss(
            predictors, saes, means, stds, target_ms, activations,
            scheduled_sampling_probability=1.0, carrier_mode="raw",
        )
        max_carrier_delta = 0.0
        perturbed = {section: value.clone() for section, value in activations.items()}
        for section in chain.SECTIONS[1:]:
            perturbed[section] = perturbed[section] + 17.0
        _, perturbed_artifact = chain.chain_training_loss(
            predictors, saes, means, stds, target_ms, perturbed,
            scheduled_sampling_probability=1.0, carrier_mode="regrounded",
        )
        for left, right in zip(artifact["trace"], perturbed_artifact["trace"]):
            max_carrier_delta = max(
                max_carrier_delta,
                float((left["carrier"].detach() - right["carrier"].detach()).abs().max().item()),
            )
        ties = torch.ones(1, 4, 2, 2, device=device)
        _, tie_support, tie_indices = chain.exact_mask_values(ties, 3)
        frozen = all(state_dict_unchanged(net, state_before[section])
                     for section, net in saes.items())
        results[family] = {
            "loss": float(loss.detach().item()),
            "gradient_norms_by_transition_loss": gradient_norms,
            "carrier_independence_max_abs_delta": max_carrier_delta,
            "tie_support_count": int(tie_support.sum().item()),
            "tie_indices": tie_indices.detach().cpu().tolist(),
            "raw_vs_regrounded_distinct": not torch.allclose(
                raw_artifact["trace"][-1]["carrier"].detach(),
                artifact["trace"][-1]["carrier"].detach(),
            ),
            "frozen_sae_unchanged": frozen,
        }
        if max_carrier_delta != 0.0 or int(tie_support.sum().item()) != 3 or not frozen:
            raise RuntimeError(f"chain smoke invariant failed for {family}: {results[family]}")
    payload = {
        "schema": "sae-redo-level2-cuda-chain-smoke-v1",
        "backbone": backbone, "device": str(device), "gpu": int(gpu),
        "samples": int(samples), "source_acts": str(source.resolve()),
        "chain_input_sha256": _sha256(input_path), "families": results,
        "cuda_only": True, "real_chain_loss": True,
    }
    payload["sha256"] = _canonical_digest(payload)
    _write_json_new(output / "chain_smoke.json", payload)
    return payload


def _write_plan(plan: dict, path: Path) -> None:
    _write_json_new(path, plan)


def _run_bounded_subprocess(argv: list[str], *, cwd: Path, env: dict,
                            log_path: Path, timeout_seconds: int,
                            cleanup_grace_seconds: int) -> dict:
    """Run one smoke child in its own session and reap its whole process tree."""
    if int(timeout_seconds) < 1:
        raise ValueError("timeout_seconds must be positive")
    if int(cleanup_grace_seconds) < 1:
        raise ValueError("cleanup_grace_seconds must be positive")
    started_clock = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cleanup = {
        "process_group": "new_session",
        "term_sent": False, "kill_sent": False, "reaped": False,
        "grace_seconds": int(cleanup_grace_seconds),
    }
    with log_path.open("x") as stream:
        process = subprocess.Popen(
            argv, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            process.communicate(timeout=int(timeout_seconds))
            cleanup["reaped"] = True
            return {
                "returncode": int(process.returncode), "status": "passed"
                if int(process.returncode) == 0 else "failed",
                "timed_out": False, "cleanup": cleanup,
                "elapsed_seconds": time.monotonic() - started_clock,
            }
        except subprocess.TimeoutExpired as error:
            cleanup["timeout_exception"] = type(error).__name__
            # start_new_session=True makes the child PID the process-group ID.
            try:
                os.killpg(process.pid, signal.SIGTERM)
                cleanup["term_sent"] = True
            except ProcessLookupError:
                cleanup["term_process_missing"] = True
            try:
                process.communicate(timeout=int(cleanup_grace_seconds))
                cleanup["reaped"] = True
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    cleanup["kill_sent"] = True
                except ProcessLookupError:
                    cleanup["kill_process_missing"] = True
                # No timeout is used for the final reap: the process group has
                # received SIGKILL and must not be left orphaned.
                process.communicate()
                cleanup["reaped"] = True
            return {
                "returncode": 124, "status": "timed_out", "timed_out": True,
                "error": f"deadline exceeded after {int(timeout_seconds)} seconds",
                "cleanup": cleanup,
                "elapsed_seconds": time.monotonic() - started_clock,
            }


def _execute_jobs(plan: dict, root: Path) -> dict:
    """Execute an already validated plan serially with durable evidence."""
    statuses = {}
    for job in plan["jobs"]:
        output = Path(job["output"])
        evidence_path = output / "smoke_job_evidence.json"
        if evidence_path.is_file():
            statuses[job["job_id"]] = json.loads(evidence_path.read_text())
            continue
        failed_dependencies = [dependency for dependency in job["depends_on"]
                               if statuses.get(dependency, {}).get("status") != "passed"]
        if failed_dependencies:
            output.mkdir(parents=True, exist_ok=False)
            evidence = {
                "schema": "sae-redo-level2-cuda-smoke-job-v1",
                "job_id": job["job_id"], "kind": job["kind"],
                "family": job["family"], "phase": job["phase"],
                "returncode": 125, "status": "failed",
                "error": "dependency failed", "failed_dependencies": failed_dependencies,
                "output": str(output), "missing_artifacts": list(job["required_artifacts"]),
                "artifacts": {},
            }
            _write_json_new(evidence_path, evidence)
            statuses[job["job_id"]] = evidence
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        started = dt.datetime.now(dt.timezone.utc).isoformat()
        log_path = root / "logs" / (job["job_id"].replace(":", "_") + ".log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(plan["gpu"])
        run = _run_bounded_subprocess(
            job["argv"], cwd=ROOT, env=env, log_path=log_path,
            timeout_seconds=job["timeout_seconds"],
            cleanup_grace_seconds=job["cleanup_grace_seconds"],
        )
        ended = dt.datetime.now(dt.timezone.utc).isoformat()
        try:
            evidence = build_job_evidence(
                job, returncode=run["returncode"], started=started, ended=ended,
                status=run["status"], error=run.get("error"),
                cleanup=run["cleanup"], timeout_seconds=job["timeout_seconds"],
            )
        except ValueError as error:
            evidence = {
                "schema": "sae-redo-level2-cuda-smoke-job-v1",
                "job_id": job["job_id"], "kind": job["kind"],
                "family": job["family"], "phase": job["phase"],
                "returncode": int(run["returncode"]), "status": run["status"],
                "error": str(error), "output": str(output),
                "missing_artifacts": [name for name in job["required_artifacts"]
                                       if not (output / name).is_file()],
                "artifacts": {},
            }
        evidence["log"] = str(log_path)
        evidence["elapsed_seconds"] = run["elapsed_seconds"]
        _write_json_new(evidence_path, evidence)
        statuses[job["job_id"]] = evidence
    summary = {
        "schema": "sae-redo-level2-cuda-smoke-result-v1",
        "plan_sha256": plan["sha256"], "jobs": statuses,
        "passed": all(item.get("status") == "passed" for item in statuses.values()),
        "cuda_only": True, "tmux": False,
    }
    summary["sha256"] = _canonical_digest(summary)
    _write_json_new(root / "smoke_result.json", summary)
    return summary


def execute_smoke_plan(plan: dict) -> dict:
    """Execute the CNN plan, preparing only its new isolated cache first."""
    require_cuda_gpu(plan["gpu"])
    root = Path(plan.get("output_root", Path(plan["prepared_cache"]).parent))
    source = Path(plan["source"]["acts"])
    _prepare_subset(source, Path(plan["prepared_cache"]))
    return _execute_jobs(plan, root)


def execute_vit_smoke_plan(plan: dict) -> dict:
    """Execute a local-weight ViT plan without any implicit downloads."""
    require_cuda_gpu(plan["gpu"])
    weight_path = Path(plan["weights"]["path"])
    if not weight_path.is_file():
        raise FileNotFoundError(f"local ViT weights not found: {weight_path}")
    actual_hash = _sha256(weight_path)
    if actual_hash != plan["weights"]["sha256"]:
        raise ValueError(
            f"local ViT weights changed after plan creation: {weight_path}"
        )
    root = Path(plan["output_root"])
    return _execute_jobs(plan, root)


def _default_output(backbone: str) -> Path:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return ROOT / "runs" / "sae_sparse_recovery_redo" / "cuda_smoke" / backbone / timestamp


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=("r20", "r110"), default="r20")
    parser.add_argument("--vit-model", choices=("vit_small_patch16_224", "vit_base_patch16_224"),
                        default=None, help="build the local-weight ViT smoke plan")
    parser.add_argument("--weights", type=Path, default=None,
                        help="required local ViT checkpoint for --vit-model; never downloaded")
    parser.add_argument("--block", type=int, default=2,
                        help="ViT block for --vit-model (default: 2)")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/imagenette2-160",
                        help="existing Imagenette ImageFolder root for --vit-model")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--ntrain", type=int, default=8)
    parser.add_argument("--nval", type=int, default=8)
    parser.add_argument("--ntest", type=int, default=8)
    parser.add_argument("--chain-samples", type=int, default=2)
    parser.add_argument("--job-timeout-seconds", type=int,
                        default=DEFAULT_JOB_TIMEOUT_SECONDS,
                        help="per-child deadline; default 900 seconds")
    parser.add_argument("--cleanup-grace-seconds", type=int,
                        default=DEFAULT_CLEANUP_GRACE_SECONDS,
                        help="SIGTERM-to-SIGKILL cleanup grace; default 15 seconds")
    parser.add_argument("--execute", action="store_true",
                        help="execute the bounded CUDA smoke; default is dry-run")
    parser.add_argument("--dry-run", action="store_true",
                        help="write only the command/evidence plan (default)")
    parser.add_argument("--chain-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--source-acts", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--samples", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--require-cuda", "--require_cuda", dest="require_cuda",
                        action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cuda:0", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.chain_only:
        if args.source_acts is None or args.out is None or args.samples is None:
            parser.error("--chain-only requires --source-acts, --samples, and --out")
        run_chain_cuda_smoke(
            args.source_acts, args.out, backbone=args.backbone, samples=args.samples,
            gpu=args.gpu, require_cuda=True,
        )
        return 0
    if args.execute and args.dry_run:
        parser.error("--execute and --dry-run are mutually exclusive")
    if args.vit_model is not None:
        output = (args.out or _default_output(args.vit_model)).resolve()
        if output.exists():
            raise FileExistsError(f"smoke output root already exists: {output}")
        plan = build_vit_smoke_plan(
            args.vit_model, output, weights=args.weights, gpu=args.gpu,
            block=args.block, data_root=args.data_root, ntrain=args.ntrain,
            nval=args.nval, ntest=args.ntest,
            job_timeout_seconds=args.job_timeout_seconds,
            cleanup_grace_seconds=args.cleanup_grace_seconds,
        )
        output.mkdir(parents=True, exist_ok=False)
        _write_plan(plan, output / "smoke_plan.json")
        print(json.dumps({"out": str(output), "plan_sha256": plan["sha256"],
                          "jobs": len(plan["jobs"]), "execute": bool(args.execute),
                          "model": args.vit_model}, sort_keys=True))
        if not args.execute:
            print("DRY RUN: no CUDA job, tmux session, or ViT download started", flush=True)
            return 0
        summary = execute_vit_smoke_plan(plan)
        return 0 if summary["passed"] else 1
    output = (args.out or _default_output(args.backbone)).resolve()
    if output.exists():
        raise FileExistsError(f"smoke output root already exists: {output}")
    plan = build_smoke_plan(
        args.backbone, output, gpu=args.gpu, ntrain=args.ntrain,
        nval=args.nval, chain_samples=args.chain_samples,
        job_timeout_seconds=args.job_timeout_seconds,
        cleanup_grace_seconds=args.cleanup_grace_seconds,
    )
    output.mkdir(parents=True, exist_ok=False)
    _write_plan(plan, output / "smoke_plan.json")
    print(json.dumps({"out": str(output), "plan_sha256": plan["sha256"],
                      "jobs": len(plan["jobs"]), "execute": bool(args.execute)},
                     sort_keys=True))
    if not args.execute:
        print("DRY RUN: no CUDA job, tmux session, or ViT download started", flush=True)
        return 0
    summary = execute_smoke_plan(plan)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Build the frozen sparse-recovery redo queues without CPU fallback.

The launcher is deliberately separate from execution: it writes static queue
files by default and starts tmux only when ``--launch`` is explicitly supplied.
Every runnable command is pinned to an NVIDIA GPU discovered through nvidia-smi.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path
import shlex
import subprocess


ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (0.08, 0.04, 0.02, 0.01)
SEEDS = (0, 1, 2)
FINAL_FAMILIES = ("field_old", "field_strict", "field_recovery", "vector_context")
DEFAULT_QUEUE_COUNT = 5
REQUIRED_COMPLETION_FILES = (
    "sae.pt", "result.json", "run_metadata.json", "continuation.json",
    "split_manifest.json", "normalization.json", "sparse_message.pt",
    "replay.json", "fidelity_result.json", "controls.json", "run.log",
)


@dataclasses.dataclass(frozen=True)
class Job:
    job_id: str
    chain_id: str
    lane: str
    backbone: str
    section: str
    model: str | None
    block: int | None
    family: str
    budget: float
    seed: int
    output: Path
    parent_job_id: str | None
    final: bool
    weight_path: Path | None = None


def stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _budget_name(budget: float) -> str:
    return f"b{int(round(budget * 100)):02d}"


def _job_id(lane: str, section: str, family: str, budget: float, seed: int) -> str:
    return f"{lane}:{section}:{family}:{_budget_name(budget)}:seed{seed}"


def _cell_root(run_root: Path, lane: str, backbone: str, section: str,
               family: str, budget: float, seed: int) -> Path:
    return run_root / lane / backbone / section / family / _budget_name(budget) / f"seed{seed}"


def _domain_specs():
    return (
        ("cnn-r20", "resnet20_cifar10", None, ("Q1", "Q2", "Q3", "Q4", "Q5")),
        ("cnn-r110", "resnet110_cifar100", None, ("Q1", "Q2", "Q3", "Q4", "Q5")),
        ("vit-small", "vit_small_patch16_224", "vit_small_patch16_224", (2, 4, 6, 8, 10)),
        ("vit-base", "vit_base_patch16_224", "vit_base_patch16_224", (2, 4, 6, 8, 10)),
    )


def build_jobs(dry_run: bool, run_stamp: str, *, run_root: Path | str | None = None,
               vit_small_weights: Path | str | None = None,
               vit_base_weights: Path | str | None = None) -> list[Job]:
    """Return the 960 final cells plus 240 strict-vector prerequisites.

    Ordering is topological inside each backbone/field/seed chain.  This lets a
    queue runner execute a static file serially without polling or dynamic work
    scheduling while still respecting continuation parents.
    """
    root = (Path(run_root).resolve() if run_root is not None else
            ROOT / "runs" / "sae_sparse_recovery_redo" /
            ("smoke" if dry_run else "matrix") / run_stamp)
    vit_weight_paths = {
        "vit_small_patch16_224": Path(vit_small_weights).expanduser()
        if vit_small_weights is not None else None,
        "vit_base_patch16_224": Path(vit_base_weights).expanduser()
        if vit_base_weights is not None else None,
    }
    jobs: list[Job] = []
    for lane, backbone, model, fields in _domain_specs():
        for raw_field in fields:
            section = f"block{raw_field}" if model is not None else raw_field
            for seed in SEEDS:
                chain_id = f"{lane}:{section}:seed{seed}"
                previous: dict[str, str | None] = {"field_old": None, "field_strict": None,
                                                   "vector_context_strict": None}
                for budget in BUDGETS:
                    # Strict continuation paths are first so same-budget recovery
                    # children can appear after their exact parents.
                    for family, final in (("field_old", True), ("field_strict", True),
                                          ("vector_context_strict", False)):
                        job_id = _job_id(lane, section, family, budget, seed)
                        jobs.append(Job(
                            job_id, chain_id, lane, backbone, section, model,
                            raw_field if model is not None else None, family, budget, seed,
                            _cell_root(root, lane, backbone, section, family, budget, seed),
                            previous[family], final, vit_weight_paths.get(model),
                        ))
                        previous[family] = job_id
                    for family, parent_family in (("field_recovery", "field_strict"),
                                                  ("vector_context", "vector_context_strict")):
                        job_id = _job_id(lane, section, family, budget, seed)
                        jobs.append(Job(
                            job_id, chain_id, lane, backbone, section, model,
                            raw_field if model is not None else None, family, budget, seed,
                            _cell_root(root, lane, backbone, section, family, budget, seed),
                            previous[parent_family], True, vit_weight_paths.get(model),
                        ))
    expected = 1200
    if len(jobs) != expected:
        raise AssertionError(f"frozen job construction produced {len(jobs)}, expected {expected}")
    return jobs


def _quote(value: object) -> str:
    return shlex.quote(str(value))


def _trainer_command(job: Job, output: Path, parent_output: Path | None = None,
                     vit_weights: dict[str, dict[str, object]] | None = None) -> str:
    common = ["--sae_type", job.family, "--budget", job.budget, "--seed", job.seed,
              "--backbone_id", job.backbone, "--out", output, "--device", "cuda:0",
              "--require_cuda"]
    if job.model is None:
        # Existing trainer defaults cover R110; keep R20 inputs explicit so queue
        # entries cannot silently reuse the wrong frozen backbone/cache.
        if job.lane == "cnn-r20":
            common += ["--acts", "runs/acts_r20_c10", "--ckpt", "runs/backbone_r20_c10/best.pt"]
        command = ["python", ROOT / "src" / "train_sae_redo.py", "--section", job.section, *common]
    else:
        weight_record = (vit_weights or {}).get(job.model)
        weight_path = (weight_record or {}).get("path") if weight_record else job.weight_path
        if weight_path is None:
            raise FileNotFoundError(f"missing local ViT weights for {job.model}")
        command = ["python", ROOT / "src" / "vit_sae_redo.py", "--model", job.model,
                   "--block", job.block, "--weights", weight_path, *common]
    if job.parent_job_id is not None:
        parent = parent_output if parent_output is not None else _parent_output(job)
        parent_family = "field_strict" if job.family == "field_recovery" else (
            "vector_context_strict" if job.family == "vector_context" else job.family
        )
        parent_budget = job.budget if job.family in {"field_recovery", "vector_context"} else BUDGETS[BUDGETS.index(job.budget) - 1]
        phase = ("recovery_joint" if job.family == "field_recovery" else
                 "vector_joint" if job.family == "vector_context" else "strict")
        command += ["--parent_checkpoint", parent / "sae.pt", "--parent_family", parent_family,
                    "--parent_budget", parent_budget, "--phase", phase]
    return " ".join(_quote(item) for item in command)


def _parent_output(job: Job) -> Path:
    if job.parent_job_id is None:
        raise ValueError(f"job has no parent: {job.job_id}")
    if job.family in {"field_recovery", "vector_context"}:
        family = "field_strict" if job.family == "field_recovery" else "vector_context_strict"
        budget = job.budget
    else:
        family = job.family
        budget = BUDGETS[BUDGETS.index(job.budget) - 1]
    return job.output.parents[2] / family / _budget_name(budget) / f"seed{job.seed}"


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_state_dict(path: Path) -> dict[str, object]:
    """Load a local checkpoint without executing pickle globals."""
    try:
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"local ViT weights are unreadable: {path}") from error
    state = payload
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if isinstance(payload.get(key), dict):
                state = payload[key]
                break
    if not isinstance(state, dict) or not state:
        raise ValueError(f"local ViT weights are incompatible: no tensor state dict in {path}")
    cleaned = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            raise ValueError(f"local ViT weights are incompatible: non-tensor key {key!r}")
        key = str(key)
        while key.startswith("module.") or key.startswith("model."):
            key = key.split(".", 1)[1]
        cleaned[key] = value
    return cleaned


def _create_vit_for_validation(model: str):
    """Construct a validation model without allowing timm to fetch weights."""
    try:
        import timm
    except Exception as error:
        raise RuntimeError(
            "timm is required for prequeue ViT compatibility validation; "
            "install the repository dependency before materializing queues"
        ) from error
    try:
        return timm.create_model(model, pretrained=False).cpu().eval()
    except Exception as error:
        raise RuntimeError(
            f"unable to construct {model} with pretrained=False for local-weight validation"
        ) from error


def validate_local_vit_weight(path: Path | str, model: str) -> dict[str, object]:
    """Strictly validate and hash a local ViT checkpoint for one architecture."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing local ViT weights for {model}: {path}")
    if model not in {"vit_small_patch16_224", "vit_base_patch16_224"}:
        raise ValueError(f"unsupported ViT model for local weights: {model}")
    state = _tensor_state_dict(path)
    model_instance = _create_vit_for_validation(model)
    try:
        incompatible = model_instance.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(
            f"local ViT weights are incompatible with {model}: {error}"
        ) from error
    missing = list(getattr(incompatible, "missing_keys", ()))
    unexpected = list(getattr(incompatible, "unexpected_keys", ()))
    if missing or unexpected:
        raise ValueError(
            f"local ViT weights are incompatible with {model}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {
        "model": model, "path": str(path), "sha256": _digest(path),
        "format": "strict_tensor_state_dict_v1", "tensor_count": len(state),
    }


def _resolve_vit_weight_sources(
    jobs: list[Job], *, vit_small_weights: Path | str | None = None,
    vit_base_weights: Path | str | None = None,
) -> dict[str, dict[str, object]]:
    """Resolve both ViT sources before any runnable queue directory is made."""
    explicit = {
        "vit_small_patch16_224": vit_small_weights,
        "vit_base_patch16_224": vit_base_weights,
    }
    records: dict[str, dict[str, object]] = {}
    for model in ("vit_small_patch16_224", "vit_base_patch16_224"):
        model_jobs = [job for job in jobs if job.model == model]
        if not model_jobs:
            continue
        raw_path = explicit[model]
        if raw_path is None:
            raw_path = next((job.weight_path for job in model_jobs
                             if job.weight_path is not None), None)
        if raw_path is None:
            raise FileNotFoundError(f"missing local ViT weights for {model}")
        records[model] = validate_local_vit_weight(raw_path, model)
    return records


def write_completion_manifest(run_dir: Path | str, job_id: str) -> dict:
    run_dir = Path(run_dir)
    files = {}
    for name in REQUIRED_COMPLETION_FILES:
        path = run_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"cannot complete {job_id}: missing {path}")
        files[name] = _digest(path)
    payload = {"schema": "sae-redo-level2-completion-v1", "job_id": job_id, "files": files}
    payload["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path = run_dir / "completion_manifest.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def completion_manifest_is_valid(run_dir: Path | str, expected_job_id: str) -> bool:
    path = Path(run_dir) / "completion_manifest.json"
    try:
        payload = json.loads(path.read_text())
        supplied = payload.pop("sha256")
        expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if supplied != expected or payload.get("job_id") != expected_job_id:
            return False
        files = payload.get("files")
        if set(files or ()) != set(REQUIRED_COMPLETION_FILES):
            return False
        return all((Path(run_dir) / name).is_file() and _digest(Path(run_dir) / name) == digest
                   for name, digest in files.items())
    except (OSError, ValueError, TypeError, KeyError):
        return False


def select_cell_output(requested: Path | str, job_id: str) -> Path | None:
    """Return a fresh output path, or None only for a valid completed cell."""
    requested = Path(requested)
    if not requested.exists():
        return requested
    if completion_manifest_is_valid(requested, job_id):
        return None
    index = 1
    while (candidate := requested.with_name(f"{requested.name}.attempt{index}")).exists():
        index += 1
    return candidate


def _next_attempt(requested: Path) -> Path:
    """Allocate a fresh attempt without modifying the prior partial output."""
    if not requested.exists():
        return requested
    index = 1
    while (candidate := requested.with_name(f"{requested.name}.attempt{index}")).exists():
        index += 1
    return candidate


def enumerate_nvidia_gpus() -> list[int]:
    """Discover NVIDIA GPU indices and fail closed rather than accepting CPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, check=False,
        )
    except OSError as error:
        raise RuntimeError("nvidia-smi is required; refusing CPU fallback") from error
    if result.returncode != 0:
        raise RuntimeError("nvidia-smi failed; refusing CPU fallback")
    values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not values:
        raise RuntimeError("no NVIDIA GPUs reported; refusing CPU fallback")
    try:
        return [int(value) for value in values]
    except ValueError as error:
        raise RuntimeError("nvidia-smi returned invalid NVIDIA GPU indices") from error


def select_queue_gpus(discovered: list[int], override=None) -> list[int]:
    """Select the five physical GPUs used by the frozen queue topology.

    Without an override, NVIDIA discovery order is preserved and only the
    first five indices are selected. An explicit override can use one to five
    unique NVIDIA GPUs, which supports smaller, user-requested resumes.
    """
    available = [int(gpu) for gpu in discovered]
    if override is None:
        if len(available) < DEFAULT_QUEUE_COUNT:
            raise RuntimeError(
                f"the launcher requires five NVIDIA GPUs, found {len(available)}"
            )
        return available[:DEFAULT_QUEUE_COUNT]
    if isinstance(override, str):
        fields = [field.strip() for field in override.split(",") if field.strip()]
    else:
        fields = list(override)
    try:
        selected = [int(field) for field in fields]
    except (TypeError, ValueError) as error:
        raise ValueError("--gpus must be a comma-separated list of NVIDIA indices") from error
    if not selected or len(selected) > DEFAULT_QUEUE_COUNT or len(set(selected)) != len(selected):
        raise ValueError("--gpus must contain one to five unique NVIDIA indices")
    missing = [gpu for gpu in selected if gpu not in available]
    if missing:
        raise RuntimeError(f"requested NVIDIA GPUs are not reported: {missing}")
    return selected


def _chains(jobs: list[Job]) -> OrderedDict[str, list[Job]]:
    grouped: OrderedDict[str, list[Job]] = OrderedDict()
    for job in jobs:
        grouped.setdefault(job.chain_id, []).append(job)
    return grouped


def write_queues(
    jobs: list[Job], queue_root: Path | str, gpus: list[int] | None = None, *,
    vit_small_weights: Path | str | None = None,
    vit_base_weights: Path | str | None = None,
) -> dict[int, list[str]]:
    """Materialize static queues after validating every requested ViT source."""
    if gpus is None:
        gpus = select_queue_gpus(enumerate_nvidia_gpus())
    if not gpus:
        raise ValueError("at least one NVIDIA GPU is required")
    vit_weights = _resolve_vit_weight_sources(
        jobs, vit_small_weights=vit_small_weights,
        vit_base_weights=vit_base_weights,
    )
    queue_root = Path(queue_root)
    queue_root.mkdir(parents=True, exist_ok=False)
    queues: dict[int, list[str]] = {gpu: [] for gpu in gpus}
    queue_jobs = []
    for index, chain in enumerate(_chains(jobs).values()):
        gpu = gpus[index % len(gpus)]
        effective_outputs: dict[str, Path] = {}
        rerun_ancestors: set[str] = set()
        for job in chain:
            output = select_cell_output(job.output, job.job_id)
            parent_reruns = job.parent_job_id in rerun_ancestors
            if output is None and parent_reruns:
                output = _next_attempt(job.output)
            if output is None:
                effective_outputs[job.job_id] = job.output
                continue
            effective_outputs[job.job_id] = output
            rerun_ancestors.add(job.job_id)
            parent_output = effective_outputs.get(job.parent_job_id) if job.parent_job_id else None
            command = _trainer_command(job, output, parent_output, vit_weights)
            finalize = " ".join((_quote("python"), _quote(ROOT / "scripts" / "launch_sae_redo_level2.py"),
                                 "--finalize", "--run", _quote(output), "--job-id", _quote(job.job_id)))
            queue_command = f"CUDA_VISIBLE_DEVICES={gpu} {command} && {finalize}"
            queues[gpu].append(queue_command)
            queue_jobs.append({
                "job_id": job.job_id, "gpu": int(gpu), "lane": job.lane,
                "model": job.model, "output": str(output),
                "command_sha256": hashlib.sha256(queue_command.encode()).hexdigest(),
                "weight": vit_weights.get(job.model) if job.model else None,
            })
    metadata = {
        "schema": "sae-redo-level2-queue-v1", "gpus": [int(gpu) for gpu in gpus],
        "job_count": sum(len(commands) for commands in queues.values()),
        "weights": vit_weights, "jobs": queue_jobs,
    }
    metadata["sha256"] = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (queue_root / "queue_manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    for gpu, commands in queues.items():
        (queue_root / f"gpu{gpu}.txt").write_text("\n".join(commands) + ("\n" if commands else ""))
    return queues


def start_tmux(queues: dict[int, list[str]], queue_root: Path | str, session_prefix: str) -> None:
    """Start serial static queues only after NVIDIA discovery by main()."""
    runner = ROOT / "scripts" / "run_sae_redo_queue.sh"
    for gpu, commands in queues.items():
        if not commands:
            continue
        session = f"{session_prefix}-g{gpu}"
        if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
            raise RuntimeError(f"tmux session already exists: {session}")
        queue = Path(queue_root) / f"gpu{gpu}.txt"
        status = Path(queue_root) / f"gpu{gpu}.status.log"
        command = (f"cd {_quote(ROOT)} && export CUDA_VISIBLE_DEVICES={gpu} && "
                   f"{_quote(runner)} {_quote(queue)} {_quote(status)}")
        subprocess.run(["tmux", "new-session", "-d", "-s", session, "bash", "-lc", command], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--gpus", default=None,
        help="optional comma-separated override of one to five discovered NVIDIA indices",
    )
    parser.add_argument("--vit-small-weights", "--vit_small_weights", dest="vit_small_weights",
                        type=Path, default=None,
                        help="required local tensor checkpoint for ViT-small queue cells")
    parser.add_argument("--vit-base-weights", "--vit_base_weights", dest="vit_base_weights",
                        type=Path, default=None,
                        help="required local tensor checkpoint for ViT-base queue cells")
    parser.add_argument("--queue-root", default=None)
    parser.add_argument(
        "--resume-run", type=Path, default=None,
        help="existing matrix run root; completed cells are skipped and partial cells receive new attempts",
    )
    parser.add_argument("--launch", action="store_true", help="start tmux sessions after queue materialization")
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--run")
    parser.add_argument("--job-id")
    args = parser.parse_args()
    if args.finalize:
        if not args.run or not args.job_id:
            parser.error("--finalize requires --run and --job-id")
        write_completion_manifest(args.run, args.job_id)
        return
    if args.launch and args.print_only:
        parser.error("--launch and --print-only are mutually exclusive")
    if args.dry_run and args.resume_run is not None:
        parser.error("--dry-run cannot resume a production matrix run")
    if args.resume_run is not None:
        args.resume_run = args.resume_run.resolve()
        if not args.resume_run.is_dir():
            parser.error(f"--resume-run must name an existing directory: {args.resume_run}")
    jobs = build_jobs(
        args.dry_run, stamp(), run_root=args.resume_run,
        vit_small_weights=args.vit_small_weights,
        vit_base_weights=args.vit_base_weights,
    )
    if args.print_only:
        for job in jobs:
            print(job.job_id)
        return
    # Validate both local sources before nvidia-smi or queue-root creation so a
    # production invocation cannot leave a runnable partial queue behind.
    _resolve_vit_weight_sources(
        jobs, vit_small_weights=args.vit_small_weights,
        vit_base_weights=args.vit_base_weights,
    )
    gpus = select_queue_gpus(enumerate_nvidia_gpus(), args.gpus)
    default = ROOT / "runs" / "sae_sparse_recovery_redo" / "queues" / stamp()
    queue_root = Path(args.queue_root) if args.queue_root else default
    queues = write_queues(
        jobs, queue_root, gpus, vit_small_weights=args.vit_small_weights,
        vit_base_weights=args.vit_base_weights,
    )
    print(f"WROTE {sum(map(len, queues.values()))} jobs across {len(gpus)} NVIDIA queues at {queue_root}")
    if args.launch:
        start_tmux(queues, queue_root, "sae-redo-smoke" if args.dry_run else "sae-redo-level2")


if __name__ == "__main__":
    main()

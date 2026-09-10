"""Run one immutable SAE cell and write its completion manifest.

Queue entries call this wrapper so resume can distinguish a validated complete
cell from a partial/failed directory.  Invalid existing directories are never
reused; a fresh ``.attemptN`` directory is allocated instead.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from launch_sae_redo_level2 import (completion_manifest_is_valid, file_sha256,
                                    select_cell_output, write_completion_manifest)


def _replace_option(values, option, replacement):
    values = list(values)
    for index, value in enumerate(values):
        if value == option and index + 1 < len(values):
            values[index + 1] = str(replacement)
            return values
        prefix = option + "="
        if value.startswith(prefix):
            values[index] = prefix + str(replacement)
            return values
    raise ValueError(f"trainer arguments do not contain {option}")


def _resolve_parent_checkpoint(value):
    """Resolve a canonical parent path to a valid immutable retry attempt."""
    checkpoint = Path(value)
    cell = checkpoint.parent
    if completion_manifest_is_valid(cell):
        return str(checkpoint)
    parent = cell.parent
    candidates = sorted(parent.glob(cell.name + ".attempt*"))
    for candidate in reversed(candidates):
        if completion_manifest_is_valid(candidate):
            resolved = candidate / checkpoint.name
            if resolved.is_file():
                return str(resolved)
    return str(checkpoint)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--result-file", default="result.json")
    parser.add_argument("--trainer", required=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    trainer_args = list(args.trainer_args)
    if trainer_args[:1] == ["--"]:
        trainer_args = trainer_args[1:]
    if args.require_cuda or "--require_cuda" in trainer_args:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        device_values = [trainer_args[index + 1] for index, value in enumerate(trainer_args[:-1])
                         if value == "--device"]
        if not visible or not device_values or not device_values[0].startswith("cuda"):
            print("CUDA_VISIBLE_DEVICES and a CUDA trainer device are required", file=sys.stderr)
            return 2
        if "--require_cuda" not in trainer_args:
            trainer_args.append("--require_cuda")
    checkpoint_index = None
    for index, value in enumerate(trainer_args):
        if value == "--parent_checkpoint":
            checkpoint_index = index + 1
            break
        if value.startswith("--parent_checkpoint="):
            checkpoint_index = index
            break
    if checkpoint_index is not None:
        old = trainer_args[checkpoint_index]
        if old.startswith("--parent_checkpoint="):
            trainer_args[checkpoint_index] = "--parent_checkpoint=" + _resolve_parent_checkpoint(old.split("=", 1)[1])
        else:
            trainer_args[checkpoint_index] = _resolve_parent_checkpoint(old)

    requested_out = Path(args.out)
    actual_out = select_cell_output(requested_out, args.job_id)
    if actual_out is None:
        print(f"SKIP COMPLETE {args.job_id}: {requested_out}")
        return 0
    trainer_args = _replace_option(trainer_args, "--out", actual_out)
    trainer = Path(args.trainer)
    if not trainer.is_absolute():
        trainer = ROOT / trainer
    command = [sys.executable, str(trainer), *trainer_args]
    env = os.environ.copy()
    if args.require_cuda and not env.get("CUDA_VISIBLE_DEVICES", "").strip():
        print("CUDA_VISIBLE_DEVICES is required for a GPU cell", file=sys.stderr)
        return 2
    completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
    if completed.returncode != 0:
        return completed.returncode
    write_completion_manifest(actual_out, args.job_id, args.result_file)
    print(f"COMPLETE {args.job_id}: {actual_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

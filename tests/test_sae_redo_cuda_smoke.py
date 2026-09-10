import json
import os
import signal
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import run_sae_redo_cuda_smoke as smoke


def test_cuda_smoke_plan_covers_families_and_parent_order(tmp_path):
    plan = smoke.build_smoke_plan("r20", tmp_path / "smoke", gpu=5,
                                  ntrain=8, nval=8, chain_samples=2)
    jobs = plan["jobs"]
    assert [job["family"] for job in jobs[:5]] == [
        "field_old", "field_strict", "vector_context_strict",
        "field_recovery", "vector_context",
    ]
    assert jobs[3]["depends_on"] == [jobs[1]["job_id"]]
    assert jobs[4]["depends_on"] == [jobs[2]["job_id"]]
    assert jobs[-1]["kind"] == "chain"
    assert set(jobs[-1]["depends_on"]) == {job["job_id"] for job in jobs[:-1]}
    for job in jobs:
        assert job["command"].startswith("CUDA_VISIBLE_DEVICES=5 ")
        assert "--device cuda:0" in job["command"]
        assert "--require_cuda" in job["command"]
    assert all(job["smoke_collapse_policy"] == smoke.SMOKE_COLLAPSE_POLICY
               for job in jobs[:-1])
    assert plan["execution"]["smoke_collapse_policy"] == smoke.SMOKE_COLLAPSE_POLICY
    assert plan["bounded_config"] == {"ntrain": 8, "nval": 8, "chain_samples": 2}
    vit_report = plan["execution"]["vit_pretrained_weight_check"]
    assert vit_report["status"] in {"missing", "available"}
    assert bool(vit_report["paths"]) == (vit_report["status"] == "available")
    assert "network download" in vit_report["action"]


def test_cuda_smoke_plan_supports_both_real_cnn_cache_roots(tmp_path):
    for backbone, acts, ckpt in (
        ("r20", "runs/acts_r20_c10", "runs/backbone_r20_c10/best.pt"),
        ("r110", "runs/acts_r110_c100", "runs/backbone_r110_c100/best.pt"),
    ):
        plan = smoke.build_smoke_plan(backbone, tmp_path / backbone, gpu=0)
        assert plan["source"]["acts"] == os.path.join(ROOT, acts)
        assert plan["source"]["checkpoint"] == os.path.join(ROOT, ckpt)


def test_cuda_smoke_evidence_requires_all_artifacts(tmp_path):
    output = tmp_path / "job"
    output.mkdir()
    job = {"job_id": "field_old", "kind": "sae", "output": str(output),
           "required_artifacts": ["sae.pt", "result.json"]}
    with pytest.raises(ValueError, match="missing"):
        smoke.build_job_evidence(job, returncode=0, started="s", ended="e")
    (output / "sae.pt").write_bytes(b"checkpoint")
    (output / "result.json").write_text(json.dumps({"ok": True}))
    record = smoke.build_job_evidence(job, returncode=0, started="s", ended="e")
    assert record["status"] == "passed"
    assert set(record["artifacts"]) == {"sae.pt", "result.json"}


def test_cuda_smoke_execution_guard_fails_closed_without_requested_gpu(monkeypatch):
    monkeypatch.setattr(smoke, "enumerate_nvidia_gpus", lambda: [0, 2])
    with pytest.raises(RuntimeError, match="GPU 5"):
        smoke.require_cuda_gpu(5)


def test_cuda_smoke_dry_run_writes_content_addressed_plan(tmp_path):
    output = tmp_path / "dry-run"
    assert smoke.main(["--backbone", "r20", "--gpu", "0", "--dry-run",
                       "--out", str(output)]) == 0
    plan = json.loads((output / "smoke_plan.json").read_text())
    supplied = plan.pop("sha256")
    assert supplied == smoke._canonical_digest(plan)
    assert not (output / "prepared_acts_q1").exists()


def test_vit_smoke_plan_requires_local_weights_before_jobs(tmp_path):
    with pytest.raises(FileNotFoundError, match="local ViT weights"):
        smoke.build_vit_smoke_plan(
            "vit_small_patch16_224", tmp_path / "vit", weights=tmp_path / "missing.pth",
            gpu=0,
        )


def test_vit_smoke_plan_covers_families_and_hashes_weights(tmp_path):
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"local-vit-weights")
    plan = smoke.build_vit_smoke_plan(
        "vit_base_patch16_224", tmp_path / "vit", weights=weights, gpu=2,
        ntrain=8, nval=8, ntest=8,
    )
    assert [job["family"] for job in plan["jobs"]] == [
        "field_old", "field_strict", "vector_context_strict",
        "field_recovery", "vector_context",
    ]
    assert plan["weights"]["path"] == str(weights.resolve())
    assert len(plan["weights"]["sha256"]) == 64
    assert all("--weights" in job["command"] for job in plan["jobs"])
    assert all("--pretrained" not in job["command"] for job in plan["jobs"])
    assert plan["jobs"][3]["depends_on"] == [plan["jobs"][1]["job_id"]]
    assert plan["jobs"][4]["depends_on"] == [plan["jobs"][2]["job_id"]]


def test_bounded_child_terminates_process_group_and_reaps_after_timeout(monkeypatch, tmp_path):
    calls = []

    class HangingProcess:
        pid = 1234
        returncode = None

        def communicate(self, timeout=None):
            calls.append(timeout)
            if len(calls) == 1:
                raise subprocess.TimeoutExpired(["child"], timeout)
            self.returncode = -signal.SIGTERM

    import subprocess
    monkeypatch.setattr(smoke.subprocess, "Popen", lambda *args, **kwargs: HangingProcess())
    killed = []
    monkeypatch.setattr(smoke.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    result = smoke._run_bounded_subprocess(
        ["child"], cwd=tmp_path, env={}, log_path=tmp_path / "child.log",
        timeout_seconds=3, cleanup_grace_seconds=2,
    )
    assert result["status"] == "timed_out"
    assert result["timed_out"] is True
    assert result["cleanup"] == {
        "process_group": "new_session", "term_sent": True,
        "kill_sent": False, "reaped": True, "grace_seconds": 2,
        "timeout_exception": "TimeoutExpired",
    }
    assert killed == [(1234, signal.SIGTERM)]
    assert calls == [3, 2]


def test_bounded_child_escalates_to_sigkill_and_reaps(monkeypatch, tmp_path):
    calls = []

    class NeverEndingProcess:
        pid = 5678
        returncode = None

        def communicate(self, timeout=None):
            calls.append(timeout)
            if timeout is not None:
                raise subprocess.TimeoutExpired(["child"], timeout)
            self.returncode = -signal.SIGKILL

    import subprocess
    monkeypatch.setattr(smoke.subprocess, "Popen", lambda *args, **kwargs: NeverEndingProcess())
    killed = []
    monkeypatch.setattr(smoke.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    result = smoke._run_bounded_subprocess(
        ["child"], cwd=tmp_path, env={}, log_path=tmp_path / "child.log",
        timeout_seconds=4, cleanup_grace_seconds=1,
    )
    assert result["status"] == "timed_out"
    assert result["cleanup"]["kill_sent"] is True
    assert result["cleanup"]["reaped"] is True
    assert killed == [(5678, signal.SIGTERM), (5678, signal.SIGKILL)]
    assert calls == [4, 1, None]


def test_timeout_evidence_is_durable_status_not_false_pass(tmp_path):
    job = {"job_id": "timeout", "kind": "sae", "family": "field_old",
           "phase": "strict", "output": str(tmp_path),
           "required_artifacts": ["sae.pt"], "timeout_seconds": 9}
    evidence = smoke.build_job_evidence(
        job, returncode=124, status="timed_out", started="s", ended="e",
        error="deadline exceeded", timeout_seconds=9,
        cleanup={"term_sent": True, "kill_sent": True, "reaped": True},
    )
    assert evidence["status"] == "timed_out"
    assert evidence["missing_artifacts"] == ["sae.pt"]
    assert evidence["cleanup"]["reaped"] is True


def test_chain_frozen_state_snapshot_is_device_safe():
    module = torch.nn.Linear(3, 2)
    snapshot = {name: value.detach().cpu().clone()
                for name, value in module.state_dict().items()}
    assert smoke.state_dict_unchanged(module, snapshot)
    with torch.no_grad():
        module.weight.add_(1.0)
    assert not smoke.state_dict_unchanged(module, snapshot)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_chain_frozen_state_snapshot_handles_cuda_module():
    module = torch.nn.Linear(3, 2).cuda()
    snapshot = {name: value.detach().cpu().clone()
                for name, value in module.state_dict().items()}
    assert smoke.state_dict_unchanged(module, snapshot)

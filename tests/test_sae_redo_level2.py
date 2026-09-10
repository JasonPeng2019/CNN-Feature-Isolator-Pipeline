import json
import inspect
import math
import os
import subprocess
import sys
import time
import types

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import masks
import sae
import cache_activations
import train_sae_redo as cnn_redo
import vit_sae_redo as vit_redo
import train_chain_level2 as chain_redo
from redo_provenance import (build_continuation_provenance, continuation_budgets,
                              continuation_phase_schedule,
                              load_matching_parent_checkpoint, replay_sparse_message,
                              save_sparse_message)
from sae_redo_common import validate_dictionary_multiplier
from sae_redo_common import (original_node_budget, sparse_bit_accounting,
                             normalization_artifact, stratified_partition_indices,
                             stratified_validation_split,
                             write_split_manifest)
from sae_training_policy import (BoundedCollapseRecovery, ExponentialMovingAverage,
                                 PhaseSpec,
                                 WarmupCosineScheduler, build_curriculum_schedule,
                                 build_optimizer, build_optimizer_param_groups,
                                 choose_validation_checkpoint, run_training_policy)
from sae_redo_controls import (BASE_CONTROL_NAMES, control_names,
                               generate_control_reconstructions)

sys.path.insert(0, os.path.join(ROOT, "scripts"))
import launch_sae_redo_level2 as launcher


def test_global_topm_keeps_exactly_m_under_ties():
    z = torch.ones(2, 1, 2, 3)
    _, keep = masks.mask_global_topk(z, 4)
    assert keep.reshape(2, -1).sum(dim=1).tolist() == [4, 4]


def test_exact_selector_uses_fp32_scores_and_hard_forward_ste_under_ties():
    values = torch.tensor([[[[2.0, 3.0], [4.0, 5.0]]]], requires_grad=True)
    scores = torch.ones_like(values, dtype=torch.float16, requires_grad=True)
    selected, hard, indices = masks.select_topm(values, scores, 2, temperature=0.25)

    assert selected.dtype == torch.float32
    assert hard.dtype == torch.bool
    assert hard.reshape(1, -1).sum().item() == 2
    assert indices.shape == (1, 2)
    assert torch.equal(selected, values.float() * hard.float())

    selected.square().sum().backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad.float()).all()
    assert torch.count_nonzero(scores.grad).item() > 0

    # Ties must not depend on an implementation-defined threshold comparison.
    _, hard_again, indices_again = masks.select_topm(values.detach(), scores.detach(), 2)
    assert torch.equal(hard, hard_again)
    assert torch.equal(indices, indices_again)


def test_original_node_budget_and_bit_accounting_are_explicit():
    # Original activation is C*H*W = 3*2*5 = 30, independently of K=12.
    m = original_node_budget(budget=0.04, channels=3, height=2, width=5, capacity=120)
    assert m == 1
    accounting = sparse_bit_accounting(
        m=7,
        coefficient_count=120,
        original_node_count=30,
        coefficient_value_bits=16,
        value_quantizer="symmetric_uniform_int16",
        metadata_bits=64,
    )
    index_bits = 7 * math.ceil(math.log2(120))
    value_bits = 7 * 16
    assert accounting["active_count_m"] == 7
    assert accounting["support_index_bits"] == index_bits
    assert accounting["signed_value_bits"] == value_bits
    assert accounting["value_quantizer"] == "symmetric_uniform_int16"
    assert accounting["metadata_bits"] == 64
    assert accounting["total_sparse_bits"] == index_bits + value_bits + 64
    assert accounting["bits_per_original_activation_node"] == (index_bits + value_bits + 64) / 30


def test_validation_split_is_sorted_stratified_and_disjoint():
    ids = ["z/4.jpg", "a/1.jpg", "z/2.jpg", "a/3.jpg", "z/1.jpg", "a/2.jpg"]
    labels = [1, 0, 1, 0, 1, 0]
    selection, holdout = stratified_validation_split(ids, labels, seed=20260827)
    assert set(selection).isdisjoint(holdout)
    assert sorted(selection + holdout) == sorted(ids)
    assert len(selection) == len(holdout) == 3
    assert {labels[ids.index(item)] for item in selection} == {0, 1}
    assert {labels[ids.index(item)] for item in holdout} == {0, 1}


def test_normalization_artifact_records_only_ordered_train_ids():
    artifact = normalization_artifact([1.0, 2.0], [0.5, 0.25], ["train/2", "train/1"])
    assert artifact["fit_train_ids"] == ["train/2", "train/1"]
    assert len(artifact["fit_train_ids_sha256"]) == 64
    assert len(artifact["sha256"]) == 64
    assert "test" not in artifact["fit_train_ids_sha256"]


def test_cifar_partition_is_disjoint_and_class_stratified():
    labels = [0] * 10 + [1] * 10 + [2] * 10
    train, validation = stratified_partition_indices(labels, 21, 9, seed=20260827)
    assert len(train) == 21 and len(validation) == 9
    assert set(train).isdisjoint(validation)
    assert sorted(train + validation) == list(range(30))
    assert {labels[index] for index in train} == {0, 1, 2}
    assert {labels[index] for index in validation} == {0, 1, 2}


def test_cnn_cache_metadata_binds_split_counts_without_loading_test_tensors(tmp_path):
    metadata = cache_activations.write_cache_metadata(
        str(tmp_path), dataset="cifar10", backbone="resnet20",
        train_count=3, test_count=2,
    )
    assert metadata["test_ids"] == ["test:0", "test:1"]
    assert cache_activations.load_cache_metadata(str(tmp_path)) == metadata
    metadata_path = tmp_path / "cache_metadata.json"
    metadata_path.write_text(metadata_path.read_text().replace('"test_count": 2', '"test_count": 3'))
    with pytest.raises(ValueError, match="SHA-256"):
        cache_activations.load_cache_metadata(str(tmp_path))


def test_cnn_cache_metadata_accepts_compact_legacy_sidecar(tmp_path):
    """Existing immutable caches can bind IDs without a 60k-entry JSON list."""
    payload = {
        "schema": "sae-redo-level2-cache-metadata-v1-compact",
        "dataset": "cifar100", "backbone": "resnet110",
        "train_count": 50000, "test_count": 10000,
        "train_id_prefix": "train:", "test_id_prefix": "test:",
    }
    payload["sha256"] = cache_activations._metadata_digest(payload)
    (tmp_path / "cache_metadata.json").write_text(json.dumps(payload))
    loaded = cache_activations.load_cache_metadata(str(tmp_path))
    assert loaded["train_ids"][:2] == ["train:0", "train:1"]
    assert loaded["test_ids"][-1] == "test:9999"
    assert loaded["train_count"] == 50000 and loaded["test_count"] == 10000


def test_cnn_main_does_not_load_test_tensor_before_policy_selection(monkeypatch, tmp_path):
    acts = tmp_path / "acts"
    acts.mkdir()
    torch.save(torch.randn(2, 2, 2, 2), acts / "train_Q1.pt")
    torch.save(torch.zeros(2, 2), acts / "train_logits.pt")
    torch.save(torch.tensor([0, 1]), acts / "train_labels.pt")
    cache_activations.write_cache_metadata(
        str(acts), dataset="cifar10", backbone="resnet20", train_count=2, test_count=3,
    )

    class TinyFrozen:
        pass

    monkeypatch.setattr(cnn_redo, "load_backbone", lambda _ckpt, _device: (TinyFrozen(), {}))
    monkeypatch.setattr(cnn_redo, "attach_run_log", lambda _out: None)
    monkeypatch.setattr(cnn_redo, "write_continuation_provenance", lambda *_args: None)
    monkeypatch.setattr(cnn_redo, "run_training_policy",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("POLICY_SENTINEL")))
    real_load = torch.load
    load_paths = []

    def tracking_load(path, *args, **kwargs):
        load_paths.append(str(path))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", tracking_load)
    output = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["train_sae_redo.py", "--acts", str(acts),
                                       "--ckpt", str(tmp_path / "backbone.pt"),
                                       "--section", "Q1", "--sae_type", "field_strict",
                                       "--budget", "0.08", "--seed", "0", "--ntrain", "1",
                                       "--nval", "1", "--train_bs", "1", "--val_bs", "1",
                                       "--device", "cpu", "--out", str(output)])
    with pytest.raises(RuntimeError, match="POLICY_SENTINEL"):
        cnn_redo.main()
    assert not any(os.path.basename(path).startswith("test_") for path in load_paths)


def test_context_vector_starts_as_vector_and_decoder_accepts_only_codes():
    torch.manual_seed(4)
    baseline = sae.VectorSAE(3, 7)
    candidate = sae.ContextVectorSAE(3, 7)
    candidate.load_vector_state(baseline)
    x = torch.randn(2, 3, 4, 4)
    assert torch.count_nonzero(candidate.context.weight) == 0
    assert torch.allclose(candidate.encode(x), baseline.encode(x))
    z = torch.randn(2, 7, 4, 4)
    assert torch.allclose(candidate.decode(z), baseline.decode(z))


def test_redo_families_have_independent_score_and_signed_value_heads_with_gradients():
    torch.manual_seed(11)
    x = torch.randn(2, 3, 5, 5)
    for family in ("field_old", "field_strict", "field_recovery", "vector_context"):
        model = sae.build_sae(family, 3, 12)
        assert model.score_head is not model.value_head
        assert model.score_head.weight is not model.value_head.weight
        selected, keep, _ = masks.encode_and_select(model, x, 7)
        assert selected.shape == (2, 12, 5, 5)
        assert keep.reshape(2, -1).sum(dim=1).tolist() == [7, 7]
        model.decode(selected).square().mean().backward()
        gradient = model.score_head.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient).item() > 0


def test_redo_paths_do_not_fallback_to_legacy_encoder_outputs():
    torch.manual_seed(12)
    x = torch.randn(2, 3, 4, 4)
    cnn = sae.build_sae("field_old", 3, 12)
    vit = sae.build_sae("vector_context", 3, 12)
    cnn_sparse, _, _ = masks.encode_and_select(cnn, x, 5)
    vit_sparse, _, _ = masks.encode_and_select(vit, x, 5)
    assert not torch.equal(cnn_sparse, cnn.encode(x))
    assert not torch.equal(vit_sparse, vit.encode(x))


def test_cnn_and_vit_redo_train_steps_backprop_to_score_heads():
    torch.manual_seed(13)
    activation = torch.randn(2, 3, 4, 4)
    mean = torch.zeros(1, 3, 1, 1)
    std = torch.ones(1, 3, 1, 1)
    cnn = sae.build_sae("field_old", 3, 12)
    vit = sae.build_sae("vector_context", 3, 12)
    cnn_loss = cnn_redo.redo_train_step(cnn, activation, mean, std, 5)
    vit_loss = vit_redo.redo_train_step(vit, activation, mean, std, 5)
    (cnn_loss + vit_loss).backward()
    for model in (cnn, vit):
        gradient = model.score_head.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient).item() > 0


def test_strict_decoder_is_linear_and_recovery_starts_at_strict_path():
    torch.manual_seed(14)
    strict = sae.build_sae("field_strict", 3, 12)
    recovery = sae.build_sae("field_recovery", 3, 12)
    z1, z2 = torch.randn(1, 12, 5, 5), torch.randn(1, 12, 5, 5)
    assert torch.allclose(strict.decode(torch.zeros_like(z1)), torch.zeros(1, 3, 5, 5), atol=1e-6)
    assert torch.allclose(strict.decode(z1 + z2), strict.decode(z1) + strict.decode(z2), atol=1e-5)
    strict._normalize_decoder()
    assert torch.allclose(strict.decoder.effective_atom_norms(), torch.ones(12), atol=1e-4)
    assert torch.allclose(recovery.decode(z1), recovery.strict_decode(z1), atol=1e-6)
    assert "source" not in str(inspect.signature(recovery.decode))


def test_vector_context_has_unit_normalized_direct_atoms_and_code_only_decoder():
    vector = sae.build_sae("vector_context", 3, 12)
    norms = vector.direct.weight.squeeze(-1).squeeze(-1).norm(dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert "source" not in str(inspect.signature(vector.decode))


def test_recovery_and_refiner_use_only_transmitted_message_and_start_zeroed():
    torch.manual_seed(15)
    z = torch.randn(1, 12, 4, 4)
    support_a = torch.zeros_like(z, dtype=torch.bool)
    support_b = support_a.clone()
    support_b[:, 0, 0, 0] = True
    recovery = sae.build_sae("field_recovery", 3, 12)
    assert torch.allclose(recovery.decode(z, support_a), recovery.strict_decode(z), atol=1e-6)
    recovery.correction_in.weight.data[:, 12:] = 1.0
    recovery.correction_out.weight.data.fill_(1.0)
    assert not torch.allclose(recovery.decode(z, support_a), recovery.decode(z, support_b))

    vector = sae.build_sae("vector_context", 3, 12)
    assert torch.count_nonzero(vector.refiner_out.weight) == 0
    assert torch.allclose(vector.decode(z), vector.decode_strict(z), atol=1e-6)
    # A decoder replay with the same sparse message is independent of any
    # source/encoder activation that may have produced that message.
    assert torch.allclose(vector.decode(z), vector.decode(z.clone()), atol=1e-6)


def test_family_dictionary_multipliers_are_frozen():
    for family, expected in (("field_old", 8), ("field_strict", 4),
                             ("field_recovery", 4), ("vector_context", 4)):
        assert validate_dictionary_multiplier(family, expected, 16) == expected
        try:
            validate_dictionary_multiplier(family, 3, 16)
        except ValueError:
            pass
        else:
            raise AssertionError(f"incompatible K multiplier accepted for {family}")


def test_cnn_redo_update_renormalizes_strict_atoms_after_optimizer_step():
    torch.manual_seed(16)
    model = sae.build_sae("field_strict", 3, 12)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mean, std = torch.zeros(1, 3, 1, 1), torch.ones(1, 3, 1, 1)
    cnn_redo.redo_train_update(model, torch.randn(2, 3, 5, 5), mean, std, 7, optimizer)
    assert torch.allclose(model.decoder.effective_atom_norms(), torch.ones(12), atol=1e-4)


def test_continuation_provenance_requires_matching_nonrandom_parent(tmp_path):
    assert continuation_budgets() == (0.08, 0.04, 0.02, 0.01)
    parent = tmp_path / "strict_8.pt"
    parent.write_bytes(b"strict-parent")
    strict = build_continuation_provenance(
        backbone="resnet20", field="Q1", family="field_strict", budget=0.08,
        seed=0, parent_checkpoint=None,
    )
    assert strict["start_mode"] == "initialize"
    lower = build_continuation_provenance(
        backbone="resnet20", field="Q1", family="field_strict", budget=0.04,
        seed=0, parent_checkpoint=parent,
    )
    assert lower["parent_budget"] == 0.08
    assert len(lower["parent_checkpoint_sha256"]) == 64
    assert lower["start_mode"] == "continuation"
    with pytest.raises(ValueError):
        build_continuation_provenance(
            backbone="resnet20", field="Q1", family="field_strict", budget=0.04,
            seed=0, parent_checkpoint=None,
        )


def test_recovery_provenance_requires_same_budget_strict_parent_and_phase_plan(tmp_path):
    parent = tmp_path / "strict_4.pt"
    parent.write_bytes(b"strict-parent-4")
    record = build_continuation_provenance(
        backbone="vit_small", field="block2", family="field_recovery", budget=0.04,
        seed=2, parent_checkpoint=parent, parent_family="field_strict",
        parent_budget=0.04, phase="recovery_joint",
    )
    assert record["parent_family"] == "field_strict"
    assert record["parent_budget"] == 0.04
    assert record["phase_plan"] == {
        "correction_warm_epochs": 3,
        "joint_refinement_epochs": 20,
        "joint_peak_learning_rate": 1e-4,
    }
    with pytest.raises(ValueError):
        build_continuation_provenance(
            backbone="vit_small", field="block2", family="field_recovery", budget=0.04,
            seed=2, parent_checkpoint=parent, parent_family="field_strict",
            parent_budget=0.08,
        )


def test_recovery_phase_schedule_is_fixed_and_target_budgeted():
    assert continuation_phase_schedule("field_recovery", "recovery_joint") == (
        ("recovery_correction_warm", 3, 1e-4), ("recovery_joint", 20, 1e-4)
    )
    assert continuation_phase_schedule("vector_context", "vector_joint") == (
        ("vector_refiner_warm", 3, 1e-4), ("vector_joint", 20, 1e-4)
    )
    assert continuation_phase_schedule("field_strict", "strict", epochs=9, learning_rate=3e-4) == (
        ("strict", 9, 3e-4),
    )


def test_parent_checkpoint_loader_rejects_wrong_identity(tmp_path):
    provenance = build_continuation_provenance(
        backbone="resnet20", field="Q1", family="field_strict", budget=0.08,
        seed=1, parent_checkpoint=None,
    )
    parent = tmp_path / "parent.pt"
    torch.save({"state_dict": {}, "continuation_provenance": provenance}, parent)
    loaded = load_matching_parent_checkpoint(
        parent, backbone="resnet20", field="Q1", family="field_strict", budget=0.08, seed=1
    )
    assert loaded["continuation_provenance"]["sha256"] == provenance["sha256"]
    with pytest.raises(ValueError):
        load_matching_parent_checkpoint(
            parent, backbone="resnet20", field="Q2", family="field_strict", budget=0.08, seed=1
        )
    tampered = dict(provenance)
    tampered["budget"] = 0.04
    torch.save({"state_dict": {}, "continuation_provenance": tampered}, tmp_path / "tampered.pt")
    with pytest.raises(ValueError, match="provenance hash"):
        load_matching_parent_checkpoint(
            tmp_path / "tampered.pt", backbone="resnet20", field="Q1",
            family="field_strict", budget=0.04, seed=1
        )


def test_vector_context_strict_has_no_refiner_and_is_a_valid_parent():
    strict = sae.build_sae("vector_context_strict", 3, 12)
    practical = sae.build_sae("vector_context", 3, 12)
    assert not hasattr(strict, "refiner_out")
    assert torch.allclose(strict.decode(torch.zeros(1, 12, 4, 4)), torch.zeros(1, 3, 4, 4))
    assert practical.score_head.out_channels == strict.score_head.out_channels == 12
    parent = build_continuation_provenance(
        backbone="vit_small", field="block2", family="vector_context_strict", budget=0.08,
        seed=0, parent_checkpoint=None,
    )
    assert parent["start_mode"] == "initialize"


def test_trainer_phase_configuration_freezes_base_and_uses_fixed_recovery_schedule():
    model = sae.build_sae("field_recovery", 3, 12)
    cnn_redo.configure_continuation_phase(model, "recovery_correction_warm")
    assert all(not parameter.requires_grad for name, parameter in model.named_parameters()
               if not name.startswith("correction_"))
    assert any(parameter.requires_grad for name, parameter in model.named_parameters()
               if name.startswith("correction_"))
    cnn_redo.configure_continuation_phase(model, "recovery_joint")
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert cnn_redo.continuation_training_phases("field_recovery", "recovery_joint") == (
        ("recovery_correction_warm", 3, 1e-4), ("recovery_joint", 20, 1e-4)
    )
    vector = sae.build_sae("vector_context", 3, 12)
    vit_redo.configure_continuation_phase(vector, "vector_refiner_warm")
    assert all(not parameter.requires_grad for name, parameter in vector.named_parameters()
               if not name.startswith("refiner_"))
    assert any(parameter.requires_grad for name, parameter in vector.named_parameters()
               if name.startswith("refiner_"))


def test_both_trainers_run_fixed_warm_then_joint_phases_and_ignore_mutable_limits():
    """Exercise the actual trainer runners with tiny no-data epoch callbacks."""
    cnn = sae.build_sae("field_recovery", 2, 8)
    cnn_seen = []

    def cnn_epoch(_optimizer, phase, phase_epoch):
        trainable = {name for name, parameter in cnn.named_parameters() if parameter.requires_grad}
        cnn_seen.append((phase, phase_epoch, trainable))
        assert (all(name.startswith("correction_") for name in trainable)
                if phase.startswith("recovery_correction")
                else all(parameter.requires_grad for parameter in cnn.parameters()))
        _optimizer.zero_grad()
        _optimizer.step()
        return 0.25

    cnn_logs, cnn_state = cnn_redo.run_continuation_phases(
        cnn, "field_recovery", "recovery_joint", cnn_epoch, epochs=1, learning_rate=9.0
    )
    assert len(cnn_logs) == 23
    assert [item["epochs"] for item in cnn_state] == [3, 20]
    assert [item["peak_learning_rate"] for item in cnn_state] == [1e-4, 1e-4]
    assert cnn_seen[0][0] == "recovery_correction_warm"
    assert cnn_seen[-1][0] == "recovery_joint"

    vit = sae.build_sae("vector_context", 2, 8)
    vit_seen = []

    def vit_epoch(_optimizer, phase, phase_epoch):
        trainable = {name for name, parameter in vit.named_parameters() if parameter.requires_grad}
        vit_seen.append((phase, phase_epoch, trainable))
        assert (all(name.startswith("refiner_") for name in trainable)
                if phase.startswith("vector_refiner")
                else all(parameter.requires_grad for parameter in vit.parameters()))
        _optimizer.zero_grad()
        _optimizer.step()
        return 0.5

    vit_logs, vit_state = vit_redo.run_continuation_phases(
        vit, "vector_context", "vector_joint", vit_epoch, epochs=2, learning_rate=7.0
    )
    assert len(vit_logs) == 23
    assert [item["epochs"] for item in vit_state] == [3, 20]
    assert [item["peak_learning_rate"] for item in vit_state] == [1e-4, 1e-4]
    assert vit_seen[0][0] == "vector_refiner_warm"
    assert vit_seen[-1][0] == "vector_joint"


def test_shared_curriculum_has_frozen_five_stage_schedule_and_target_budget():
    phases = build_curriculum_schedule(latent_count=1000, target_m=11, target_epochs=30)
    assert [(phase.name, phase.epochs) for phase in phases] == [
        ("dense_strict", 5), ("selector_bootstrap", 1), ("half_latent", 5),
        ("geometric_continuation", 10), ("target_budget", 30),
    ]
    assert phases[0].mode == "dense"
    assert phases[1].mode == "selector_bootstrap"
    assert phases[2].budgets == (500,) * 5
    assert phases[3].budgets[0] == 500
    assert phases[3].budgets[-1] == 11
    assert all(left >= right for left, right in zip(phases[3].budgets, phases[3].budgets[1:]))
    assert phases[4].budgets == (11,) * 30
    with pytest.raises(ValueError, match="at least 20"):
        build_curriculum_schedule(latent_count=1000, target_m=11, target_epochs=19)


def test_shared_optimizer_excludes_bias_norm_layerscale_and_normalized_atoms():
    model = sae.build_sae("field_strict", 3, 12)
    groups = build_optimizer_param_groups(model)
    grouped = {id(parameter): (group["weight_decay"], name)
               for group in groups for parameter, name in zip(group["params"], group["param_names"])}
    assert len(grouped) == sum(parameter.requires_grad for parameter in model.parameters())
    assert grouped[id(model.encoder.in_proj.bias)][0] == 0.0
    assert grouped[id(model.encoder.first[0].norm.weight)][0] == 0.0
    assert grouped[id(model.encoder.first[0].layer_scale)][0] == 0.0
    assert grouped[id(model.decoder.branches[0]["mix"].weight)][0] == 0.0
    assert grouped[id(model.encoder.in_proj.weight)][0] == 1e-4
    optimizer = build_optimizer(model)
    assert optimizer.defaults["betas"] == (0.9, 0.95)
    assert optimizer.defaults["lr"] == 3e-4


def test_shared_warmup_cosine_lr_reaches_peak_then_declared_floor():
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=3e-4)
    scheduler = WarmupCosineScheduler(optimizer, total_steps=100)
    rates = [scheduler.step() for _ in range(100)]
    assert rates[0] == pytest.approx(3e-4 / 5)
    assert rates[4] == pytest.approx(3e-4)
    assert max(rates) == pytest.approx(3e-4)
    assert rates[-1] == pytest.approx(3e-5)
    assert all(left >= right for left, right in zip(rates[4:], rates[5:]))


def test_shared_ema_selection_and_bounded_collapse_retry_state():
    model = torch.nn.Linear(2, 1)
    ema = ExponentialMovingAverage(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(2.0)
    ema.update(model)
    assert ema.state_dict()["updates"] == 1
    selection = choose_validation_checkpoint(raw_metric=0.4, ema_metric=0.2)
    assert selection["selected"] == "ema"
    assert selection["raw_metric"] == 0.4 and selection["ema_metric"] == 0.2

    recovery = BoundedCollapseRecovery(max_retries=2)
    assert recovery.observe(True)["action"] == "retry"
    assert recovery.observe(True)["action"] == "retry"
    terminal = recovery.observe(True)
    assert terminal["action"] == "terminal_failure"
    assert terminal["attempts"] == 2
    assert terminal["terminal_failure"] is True
    assert recovery.state_dict()["max_retries"] == 2


def test_shared_policy_runner_persists_phase_ema_optimizer_and_recovery_state():
    torch.manual_seed(18)
    model = torch.nn.Linear(1, 1, bias=False)
    seen = []

    def train_epoch(optimizer, phase, phase_epoch):
        seen.append((phase.name, phase_epoch, phase.budget_for_epoch(phase_epoch)))
        optimizer.zero_grad()
        loss = (model.weight - 1.0).square().mean()
        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "optimizer_steps": 1}

    result = run_training_policy(
        model, train_epoch, latent_count=100, target_m=5, target_epochs=20,
        validation_fn=lambda module: float((module.weight - 1.0).square().mean()),
        seed=18,
    )
    assert len(result["logs"]) == 41
    assert result["policy_state"]["optimizer"]["betas"] == [0.9, 0.95]
    assert result["policy_state"]["ema"]["updates"] == 41
    assert result["validation_selection"]["selected"] in {"raw", "ema"}
    assert result["policy_state"]["recovery"]["attempts"] == 0
    assert seen[0][0] == "dense_strict" and seen[-1][0] == "target_budget"


def test_shared_policy_selects_raw_or_ema_in_terminal_recovery_phase():
    torch.manual_seed(19)
    model = torch.nn.Linear(1, 1, bias=False)
    phases = (
        PhaseSpec("recovery_correction_warm", 1, "masked", (2,)),
        PhaseSpec("recovery_joint", 2, "masked", (1, 1)),
    )

    def train_epoch(optimizer, phase, phase_epoch):
        optimizer.zero_grad()
        loss = (model.weight - 1.0).square().mean()
        loss.backward()
        optimizer.step()
        return {"loss": loss.item(), "optimizer_steps": 1}

    result = run_training_policy(
        model, train_epoch, latent_count=8, target_m=1,
        validation_fn=lambda module: float((module.weight - 1.0).square().mean()),
        phase_schedule=phases, seed=19,
    )
    selection = result["validation_selection"]
    assert selection["selected"] in {"raw", "ema"}
    assert selection["phase"] == "recovery_joint"
    assert result["policy_state"]["ema"]["raw_vs_ema_selection"]["phase"] == "recovery_joint"


def test_cnn_collapse_policy_keeps_production_default_and_explicit_smoke_disable():
    assert cnn_redo.resolve_collapse_policy(smoke=False) == {
        "mode": "production", "threshold": 1.0, "max_recovery_retries": 2,
    }
    assert cnn_redo.resolve_collapse_policy(smoke=True, disable=True) == {
        "mode": "smoke_disabled", "threshold": None, "max_recovery_retries": 0,
    }
    assert cnn_redo.resolve_collapse_policy(smoke=True, threshold=7.5) == {
        "mode": "smoke_threshold", "threshold": 7.5, "max_recovery_retries": 0,
    }
    with pytest.raises(ValueError, match="smoke_disable"):
        cnn_redo.resolve_collapse_policy(smoke=False, disable=True)


def test_cnn_smoke_disable_detector_is_nonfinite_only():
    # These are the exact flags emitted by the isolated CUDA smoke harness.
    policy = cnn_redo.resolve_collapse_policy(smoke=True, disable=True)
    assert not cnn_redo.collapse_is_detected(0.5, policy)
    assert not cnn_redo.collapse_is_detected(1e30, policy)
    assert cnn_redo.collapse_is_detected(float("nan"), policy)
    assert cnn_redo.collapse_is_detected(float("inf"), policy)
    production = cnn_redo.resolve_collapse_policy(smoke=False)
    assert cnn_redo.collapse_is_detected(1.1, production)


def test_vit_main_passes_smoke_collapse_policy_to_shared_runner(monkeypatch, tmp_path):
    class TinyImageFolder(torch.utils.data.Dataset):
        def __init__(self, root, transform=None):
            root = os.fspath(root)
            self.samples = [
                (os.path.join(root, "class0", "a.jpg"), 0),
                (os.path.join(root, "class1", "b.jpg"), 1),
            ]
            self.transform = transform

        def __getitem__(self, index):
            image = torch.zeros(3, 2, 2)
            if self.transform is not None:
                image = self.transform(image)
            return image, self.samples[index][1]

        def __len__(self):
            return len(self.samples)

    class TinyVit:
        tf = None
        grid = 1

    captured = {}

    def fake_policy(*_args, **kwargs):
        captured.update(kwargs)
        return {
            "logs": [], "phase_state": [], "policy_state": {},
            "validation_selection": {"selected": "raw"}, "attempts": [],
            "optimizer_state": {}, "ema_state": {},
        }

    data_root = tmp_path / "imagenette"
    (data_root / "train").mkdir(parents=True)
    (data_root / "val").mkdir()
    monkeypatch.setattr(vit_redo, "ImageFolder", TinyImageFolder)
    monkeypatch.setattr(vit_redo, "ViTWrap", lambda *_args, **_kwargs: TinyVit())
    monkeypatch.setattr(
        vit_redo, "cache_train_grids",
        lambda *_args, **_kwargs: (torch.ones(1, 2, 1, 1), 0),
    )
    monkeypatch.setattr(vit_redo.S, "build_sae", lambda *_args, **_kwargs: torch.nn.Linear(4, 4))
    monkeypatch.setattr(vit_redo, "run_training_policy", fake_policy)
    monkeypatch.setattr(
        vit_redo, "evaluate_fidelity_holdout_once",
        lambda *_args, **_kwargs: ({"pred_agree": 1.0}, {"control_names": []}),
    )
    monkeypatch.setattr(
        vit_redo, "evaluate",
        lambda *_args, **_kwargs: {
            "pred_agree": 1.0, "kl": 0.0, "rel_l2": 0.0,
            "fvu": 0.0, "active_count_m": 1.0,
        },
    )
    monkeypatch.setattr(vit_redo, "save_replay_artifacts", lambda *_args, **_kwargs: (0.0, "hash"))
    monkeypatch.setattr(sys, "argv", [
        "vit_sae_redo.py", "--model", "vit_small_patch16_224", "--block", "2",
        "--sae_type", "field_strict", "--budget", "0.08", "--seed", "0",
        "--data_root", str(data_root), "--Kmult", "4", "--ntrain", "1",
        "--nval", "1", "--ntest", "1", "--train_bs", "1", "--val_bs", "1",
        "--num_workers", "0", "--device", "cpu", "--smoke",
        "--smoke_disable_collapse", "--out", str(tmp_path / "run"),
    ])
    vit_redo.main()
    assert captured["max_recovery_retries"] == 0
    assert captured["collapse_detector"] is not None
    disabled = vit_redo.resolve_collapse_policy(smoke=True, disable=True)
    assert not vit_redo.collapse_is_detected(1e30, disabled)
    assert vit_redo.collapse_is_detected(float("nan"), disabled)


def test_vit_local_weight_loader_hashes_and_validates_state(tmp_path):
    model = torch.nn.Linear(3, 2)
    weights = tmp_path / "vit-small.pth"
    torch.save({"state_dict": {"module.weight": model.weight.detach().clone(),
                                "module.bias": model.bias.detach().clone()}}, weights)
    loaded = vit_redo.load_local_pretrained_weights(torch.nn.Linear(3, 2), weights)
    assert loaded["path"] == str(weights.resolve())
    assert len(loaded["sha256"]) == 64
    assert loaded["missing_keys"] == [] and loaded["unexpected_keys"] == []


def test_vit_local_weight_loader_rejects_incompatible_state(tmp_path):
    weights = tmp_path / "bad.pth"
    torch.save({"state_dict": {"weight": torch.randn(4, 4)}}, weights)
    with pytest.raises(ValueError, match="incompatible"):
        vit_redo.load_local_pretrained_weights(torch.nn.Linear(3, 2), weights)


def test_cnn_and_vit_helpers_consume_dense_selector_and_exact_mask_phases():
    phases = build_curriculum_schedule(latent_count=48, target_m=4, target_epochs=20)
    activation = torch.randn(2, 3, 4, 4)
    mean, std = torch.zeros(1, 3, 1, 1), torch.ones(1, 3, 1, 1)
    for helper, family in ((cnn_redo.redo_train_step, "field_strict"),
                           (vit_redo.redo_train_step, "vector_context")):
        model = sae.build_sae(family, 3, 12)
        dense_loss = helper(model, activation, mean, std, 4, phase=phases[0])
        selector_loss = helper(model, activation, mean, std, 4, phase=phases[1])
        masked_loss = helper(model, activation, mean, std, 4, phase=phases[2])
        assert all(torch.isfinite(loss) for loss in (dense_loss, selector_loss, masked_loss))
        selector_loss.backward()
        assert model.score_head.weight.grad is not None
        assert torch.count_nonzero(model.score_head.weight.grad) > 0


def test_named_controls_have_exact_support_and_recovery_ablations():
    torch.manual_seed(19)
    model = sae.build_sae("field_recovery", 2, 8)
    normalized = torch.randn(2, 2, 3, 3)
    controls = generate_control_reconstructions(model, normalized, target_m=4,
                                                family="field_recovery", seed=20260827)
    assert tuple(controls) == control_names("field_recovery")
    assert set(BASE_CONTROL_NAMES).issubset(controls)
    assert {"recovery_correction_disabled", "recovery_strict_disabled"}.issubset(controls)
    for name, artifact in controls.items():
        if name == "empty_message":
            assert artifact["support"].sum().item() == 0
        else:
            assert artifact["support"].reshape(2, -1).sum(dim=1).tolist() == [4, 4]
        assert torch.isfinite(artifact["reconstruction"]).all()


def test_cnn_fidelity_holdout_evaluator_consumes_loader_once_after_selection():
    class TinyModel:
        def forward_from(self, _section, activation):
            return torch.cat((activation.flatten(1).mean(1, keepdim=True),
                              -activation.flatten(1).mean(1, keepdim=True)), dim=1)

    model = sae.build_sae("field_strict", 2, 8)
    mean, std = torch.zeros(1, 2, 1, 1), torch.ones(1, 2, 1, 1)
    hidden = torch.randn(2, 2, 3, 3)
    labels = torch.tensor([0, 1])
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    class CountingLoader:
        def __init__(self, batch):
            self.batch = batch
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            yield self.batch

    loader = CountingLoader((hidden, logits, labels))
    metrics, controls = cnn_redo.evaluate_fidelity_holdout_once(
        model, loader, "Q1", mean, std, target_m=4, frozen_model=TinyModel(), seed=20260827,
    )
    assert loader.iterations == 1
    assert metrics["split"] == "official_test"
    assert set(controls["metrics"]) == set(control_names("field_strict"))


def test_vit_fidelity_holdout_evaluator_consumes_manifest_holdout_once():
    class TinyViT:
        grid = 2

        def tokens_at(self, images, _block):
            return images.flatten(2).transpose(1, 2)

        def head_from(self, tokens, _block, patch_tokens=None):
            if patch_tokens is not None:
                tokens = patch_tokens
            value = tokens.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
            return torch.cat((value, -value), dim=1)

    model = sae.build_sae("vector_context", 2, 8)
    mean, std = torch.zeros(1, 2, 1, 1), torch.ones(1, 2, 1, 1)
    images = torch.randn(2, 2, 2, 2)
    labels = torch.tensor([0, 1])

    class CountingLoader:
        def __init__(self, batch):
            self.batch = batch
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            yield self.batch

    loader = CountingLoader((images, labels))
    metrics, controls = vit_redo.evaluate_fidelity_holdout_once(
        TinyViT(), model, loader, block=2, prefix_count=0, mean=mean, std=std,
        target_m=4, device=torch.device("cpu"), family="vector_context",
    )
    assert loader.iterations == 1
    assert metrics["split"] == "fidelity_holdout"
    assert set(controls["control_names"]) == set(control_names("vector_context"))


def test_decoder_only_sparse_message_replay_has_no_source_or_unmasked_payload(tmp_path):
    torch.manual_seed(17)
    model = sae.build_sae("field_recovery", 3, 12)
    indices = torch.tensor([[0, 7, 15, 31]])
    values = torch.randn(1, 4)
    shape = (1, 12, 2, 2)
    flat = torch.zeros(shape).reshape(1, -1)
    flat.scatter_(1, indices, values)
    support = torch.zeros_like(flat, dtype=torch.bool)
    support.scatter_(1, indices, True)
    expected = model.decode(flat.reshape(shape), support.reshape(shape))
    path = tmp_path / "message.pt"
    save_sparse_message(path, indices, values, shape, reconstruction=expected,
                        decoder_requires_support=True)
    replay = replay_sparse_message(model, path)
    assert replay["max_abs_error"] == 0.0
    payload = torch.load(path, weights_only=False)
    assert set(payload) <= {"schema", "shape", "indices", "values", "recorded_reconstruction",
                             "decoder_requires_support"}


def test_split_manifest_is_content_addressed_and_immutable(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    payload = write_split_manifest(
        out,
        lane="vit_small",
        source={"dataset": "fixture", "train": ["a", "b"], "validation": ["v"]},
        selections={"train": [1], "validation": [0], "test": []},
        seeds={"data_seed": 0},
    )
    saved = json.loads((out / "split_manifest.json").read_text())
    assert saved == payload
    assert len(saved["sha256"]) == 64
    try:
        write_split_manifest(out, "vit_small", {}, {}, {})
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing manifest must not be overwritten")


def test_full_launcher_materializes_frozen_matrix_and_ordered_internal_prerequisites(tmp_path, monkeypatch):
    _patch_launcher_vit_factory(monkeypatch)
    vit_small_weights = _write_launcher_vit_fixture(
        tmp_path / "vit-small.pth", "vit_small_patch16_224"
    )
    vit_base_weights = _write_launcher_vit_fixture(
        tmp_path / "vit-base.pth", "vit_base_patch16_224"
    )
    jobs = launcher.build_jobs(
        dry_run=False, run_stamp="fixture",
        vit_small_weights=vit_small_weights, vit_base_weights=vit_base_weights,
    )
    assert len(jobs) == 1200
    finals = [job for job in jobs if job.final]
    internal = [job for job in jobs if not job.final]
    assert len(finals) == 960
    assert len(internal) == 240
    assert {job.family for job in finals} == {
        "field_old", "field_strict", "field_recovery", "vector_context"
    }
    assert {job.family for job in internal} == {"vector_context_strict"}
    assert {job.model for job in finals if job.lane == "vit-small"} == {"vit_small_patch16_224"}
    assert {job.block for job in finals if job.lane == "vit-small"} == {2, 4, 6, 8, 10}
    assert {job.block for job in finals if job.lane == "vit-base"} == {2, 4, 6, 8, 10}
    assert {job.section for job in finals if job.lane == "cnn-r20"} == {"Q1", "Q2", "Q3", "Q4", "Q5"}
    assert {job.section for job in finals if job.lane == "cnn-r110"} == {"Q1", "Q2", "Q3", "Q4", "Q5"}
    # Every prerequisite appears earlier in its chain than its dependent.
    positions = {job.job_id: index for index, job in enumerate(jobs)}
    for job in jobs:
        if job.parent_job_id is not None:
            assert positions[job.parent_job_id] < positions[job.job_id]
    chain_ids = [job.chain_id for job in jobs]
    assert all(chain_ids[index] == chain_ids[index + 1] or chain_ids[index] != chain_ids[index + 1]
               for index in range(len(chain_ids) - 1))
    queues = launcher.write_queues(
        jobs, tmp_path / "queues", [0, 1, 2, 3, 4],
        vit_small_weights=vit_small_weights, vit_base_weights=vit_base_weights,
    )
    assert sum(len(commands) for commands in queues.values()) == 1200
    assert all((tmp_path / "queues" / f"gpu{gpu}.txt").exists() for gpu in range(5))
    for gpu, commands in queues.items():
        assert all(command.startswith(f"CUDA_VISIBLE_DEVICES={gpu} ") for command in commands)
        assert all(" --device cuda:0 --require_cuda" in command for command in commands)


class _LauncherFixtureViT(torch.nn.Module):
    """Complete CPU stub used to exercise strict launcher loading without timm."""

    def __init__(self, model):
        super().__init__()
        width = 3 if model == "vit_small_patch16_224" else 5
        self.weight = torch.nn.Parameter(torch.empty(2, width))
        self.bias = torch.nn.Parameter(torch.empty(2))


def _patch_launcher_vit_factory(monkeypatch):
    monkeypatch.setattr(
        launcher, "_create_vit_for_validation",
        lambda model: _LauncherFixtureViT(model),
    )


def _write_launcher_vit_fixture(path, model):
    """Create a complete tensor-only checkpoint for the injected CPU stub."""
    torch.save({"state_dict": _LauncherFixtureViT(model).state_dict()}, path)
    return path


def test_launcher_rejects_missing_or_incompatible_vit_weights_before_queue_creation(tmp_path, monkeypatch):
    _patch_launcher_vit_factory(monkeypatch)
    valid_base = _write_launcher_vit_fixture(
        tmp_path / "vit-base.pth", "vit_base_patch16_224"
    )
    jobs = launcher.build_jobs(
        dry_run=False, run_stamp="missing-weight-fixture",
        vit_small_weights=tmp_path / "missing-small.pth", vit_base_weights=valid_base,
    )
    queue_root = tmp_path / "missing-queues"
    with pytest.raises(FileNotFoundError, match="vit_small_patch16_224"):
        launcher.write_queues(jobs, queue_root, [0])
    assert not queue_root.exists()

    # Six plausible ViT anchor tensors are still incomplete: queue validation
    # must use the complete requested model's strict state-dict contract.
    bad_small = tmp_path / "bad-small.pth"
    torch.save({"state_dict": {
        "cls_token": torch.randn(1, 1, 384),
        "pos_embed": torch.randn(1, 197, 384),
        "patch_embed.proj.weight": torch.randn(384, 3, 16, 16),
        "blocks.0.attn.qkv.weight": torch.randn(1152, 384),
        "blocks.11.norm1.weight": torch.randn(384),
        "norm.weight": torch.randn(384),
    }}, bad_small)
    jobs = launcher.build_jobs(
        dry_run=False, run_stamp="incompatible-weight-fixture",
        vit_small_weights=bad_small, vit_base_weights=valid_base,
    )
    queue_root = tmp_path / "incompatible-queues"
    with pytest.raises(ValueError, match="incompatible"):
        launcher.write_queues(jobs, queue_root, [0])
    assert not queue_root.exists()


def test_launcher_queue_metadata_hashes_and_commands_pin_both_vit_weights(tmp_path, monkeypatch):
    _patch_launcher_vit_factory(monkeypatch)
    small = _write_launcher_vit_fixture(tmp_path / "small.pth", "vit_small_patch16_224")
    base = _write_launcher_vit_fixture(tmp_path / "base.pth", "vit_base_patch16_224")
    jobs = launcher.build_jobs(
        dry_run=False, run_stamp="weight-metadata-fixture",
        vit_small_weights=small, vit_base_weights=base,
    )
    queue_root = tmp_path / "queues"
    launcher.write_queues(jobs, queue_root, [0])
    metadata = json.loads((queue_root / "queue_manifest.json").read_text())
    assert metadata["weights"]["vit_small_patch16_224"]["path"] == str(small.resolve())
    assert metadata["weights"]["vit_base_patch16_224"]["path"] == str(base.resolve())
    assert len(metadata["weights"]["vit_small_patch16_224"]["sha256"]) == 64
    assert len(metadata["weights"]["vit_base_patch16_224"]["sha256"]) == 64
    vit_commands = [
        command for command in (queue_root / "gpu0.txt").read_text().splitlines()
        if "vit_sae_redo.py" in command
    ]
    assert vit_commands
    assert all("--weights" in command for command in vit_commands)
    assert any(str(small.resolve()) in command for command in vit_commands)
    assert any(str(base.resolve()) in command for command in vit_commands)


def test_launcher_timm_validation_is_lazy_and_never_pretrained(monkeypatch):
    calls = []
    fake_timm = types.SimpleNamespace(
        create_model=lambda model, pretrained: calls.append((model, pretrained))
        or _LauncherFixtureViT(model)
    )
    monkeypatch.setitem(sys.modules, "timm", fake_timm)
    launcher._create_vit_for_validation("vit_base_patch16_224")
    assert calls == [("vit_base_patch16_224", False)]


def test_launcher_selects_exactly_five_physical_gpus_by_default_and_validates_override():
    discovered = [7, 3, 9, 1, 8, 2]
    assert launcher.select_queue_gpus(discovered) == [7, 3, 9, 1, 8]
    assert len(launcher.select_queue_gpus(discovered)) == 5
    assert launcher.select_queue_gpus(discovered, "2,3,7,8,9") == [2, 3, 7, 8, 9]
    with pytest.raises(RuntimeError, match="five"):
        launcher.select_queue_gpus([0, 1, 2, 3])
    assert launcher.select_queue_gpus(discovered, "1,2,3") == [1, 2, 3]
    with pytest.raises(ValueError, match="one to five"):
        launcher.select_queue_gpus(discovered, "1,2,3,7,8,9")
    with pytest.raises(RuntimeError, match="not reported"):
        launcher.select_queue_gpus(discovered, "0,1,2,3,99")


def test_write_queues_default_materializes_five_physical_gpu_files(tmp_path, monkeypatch):
    monkeypatch.setattr(
        launcher, "enumerate_nvidia_gpus", lambda: [11, 4, 8, 2, 9, 13]
    )
    job = launcher.Job(
        "cnn", "chain", "cnn-r20", "resnet20_cifar10", "Q1", None, None,
        "field_old", 0.08, 0, tmp_path / "cell", None, True,
    )
    queues = launcher.write_queues([job], tmp_path / "queues")
    assert list(queues) == [11, 4, 8, 2, 9]
    assert all((tmp_path / "queues" / f"gpu{gpu}.txt").exists() for gpu in queues)
    for gpu, commands in queues.items():
        assert all(command.startswith(f"CUDA_VISIBLE_DEVICES={gpu} ") for command in commands)
        assert all(" --device cuda:0 --require_cuda" in command for command in commands)


def test_launcher_completion_manifest_requires_matching_hashes(tmp_path):
    run = tmp_path / "cell"
    run.mkdir()
    for filename in launcher.REQUIRED_COMPLETION_FILES:
        (run / filename).write_bytes(filename.encode())
    launcher.write_completion_manifest(run, "job-1")
    assert launcher.completion_manifest_is_valid(run, expected_job_id="job-1")
    (run / "result.json").write_bytes(b"tampered")
    assert not launcher.completion_manifest_is_valid(run, expected_job_id="job-1")


def test_launcher_completion_requires_replay_and_provenance_evidence(tmp_path):
    run = tmp_path / "cell"
    run.mkdir()
    for filename in launcher.REQUIRED_COMPLETION_FILES:
        (run / filename).write_bytes(filename.encode())
    launcher.write_completion_manifest(run, "job-1")
    (run / "replay.json").unlink()
    assert not launcher.completion_manifest_is_valid(run, expected_job_id="job-1")


def test_launcher_completion_requires_fidelity_and_controls_evidence(tmp_path):
    assert "fidelity_result.json" in launcher.REQUIRED_COMPLETION_FILES
    assert "controls.json" in launcher.REQUIRED_COMPLETION_FILES
    run = tmp_path / "cell"
    run.mkdir()
    for filename in launcher.REQUIRED_COMPLETION_FILES:
        (run / filename).write_bytes(filename.encode())
    launcher.write_completion_manifest(run, "job-1")
    (run / "fidelity_result.json").unlink()
    assert not launcher.completion_manifest_is_valid(run, expected_job_id="job-1")


def test_launcher_resume_skips_only_valid_cells_and_suffixes_partial_cells(tmp_path):
    complete = tmp_path / "complete"
    complete.mkdir()
    for filename in launcher.REQUIRED_COMPLETION_FILES:
        (complete / filename).write_bytes(filename.encode())
    launcher.write_completion_manifest(complete, "job-complete")
    assert launcher.select_cell_output(complete, "job-complete") is None

    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "sae.pt").write_bytes(b"partial")
    retry = launcher.select_cell_output(partial, "job-partial")
    assert retry == tmp_path / "partial.attempt1"
    assert partial.exists() and (partial / "sae.pt").read_bytes() == b"partial"


def test_launcher_builds_jobs_against_an_explicit_existing_run_root(tmp_path, monkeypatch):
    _patch_launcher_vit_factory(monkeypatch)
    small = _write_launcher_vit_fixture(tmp_path / "small.pth", "vit_small_patch16_224")
    base = _write_launcher_vit_fixture(tmp_path / "base.pth", "vit_base_patch16_224")
    existing = tmp_path / "matrix" / "existing-run"
    jobs = launcher.build_jobs(
        dry_run=False, run_stamp="ignored", run_root=existing,
        vit_small_weights=small, vit_base_weights=base,
    )
    assert len(jobs) == 1200
    assert all(job.output.is_relative_to(existing) for job in jobs)


def test_queue_runner_records_interruption_and_reaps_active_child(tmp_path):
    queue = tmp_path / "queue.txt"
    status = tmp_path / "queue.status.log"
    queue.write_text("sleep 30\n")
    runner = os.path.join(ROOT, "scripts", "run_sae_redo_queue.sh")
    process = subprocess.Popen([runner, str(queue), str(status)], cwd=ROOT)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if status.exists() and "PROCESS_STARTED" in status.read_text():
            break
        time.sleep(0.05)
    else:
        process.kill()
        pytest.fail("queue runner did not record an active child")
    process.terminate()
    assert process.wait(timeout=10) == 143
    records = status.read_text()
    assert "INTERRUPTED signal=TERM" in records
    assert "EXIT=143" in records
    assert "QUEUE_EXIT=143" in records


def test_launcher_retries_dependents_after_a_parent_retry(tmp_path):
    parent_output = tmp_path / "parent"
    child_output = tmp_path / "child"
    parent_output.mkdir()
    (parent_output / "sae.pt").write_bytes(b"partial")
    child_output.mkdir()
    for filename in launcher.REQUIRED_COMPLETION_FILES:
        (child_output / filename).write_bytes(filename.encode())
    launcher.write_completion_manifest(child_output, "child")
    parent = launcher.Job("parent", "chain", "cnn-r20", "r20", "Q1", None, None,
                          "field_strict", 0.08, 0, parent_output, None, True)
    child = launcher.Job("child", "chain", "cnn-r20", "r20", "Q1", None, None,
                         "field_recovery", 0.08, 0, child_output, "parent", True)
    commands = launcher.write_queues([parent, child], tmp_path / "queues", [0])[0]
    assert len(commands) == 2
    assert "parent.attempt1/sae.pt" in commands[1]
    assert "child.attempt1" in commands[1]


def test_launcher_fails_closed_when_nvidia_is_unavailable(monkeypatch):
    class Result:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: Result())
    with pytest.raises(RuntimeError, match="nvidia-smi"):
        launcher.enumerate_nvidia_gpus()


def test_launcher_rejects_empty_nvidia_inventory(monkeypatch):
    class Result:
        returncode = 0
        stdout = "\n"

    monkeypatch.setattr(launcher.subprocess, "run", lambda *args, **kwargs: Result())
    with pytest.raises(RuntimeError, match="NVIDIA"):
        launcher.enumerate_nvidia_gpus()


def _tiny_chain_fixture():
    torch.manual_seed(101)
    saes = {section: sae.build_sae("field_strict", 2, 8) for section in chain_redo.SECTIONS}
    for model in saes.values():
        chain_redo.freeze_sae(model)
    predictors = {(source, destination): __import__("transitions").TransitionPredictor(
        8, 8, 4, 4, hidden=8
    ) for source, destination in chain_redo.PAIRS}
    means = {section: torch.zeros(1, 2, 1, 1) for section in chain_redo.SECTIONS}
    stds = {section: torch.ones(1, 2, 1, 1) for section in chain_redo.SECTIONS}
    target_ms = {section: 3 for section in chain_redo.SECTIONS}
    activations = {section: torch.randn(2, 2, 4, 4) for section in chain_redo.SECTIONS}
    return predictors, saes, means, stds, target_ms, activations


def test_level2_chain_regrounding_has_full_bptt_gradients_and_frozen_saes():
    predictors, saes, means, stds, target_ms, activations = _tiny_chain_fixture()
    before = {section: {name: value.detach().clone() for name, value in model.state_dict().items()}
              for section, model in saes.items()}
    loss, artifact = chain_redo.chain_training_loss(
        predictors, saes, means, stds, target_ms, activations,
        scheduled_sampling_probability=1.0, carrier_mode="regrounded",
        sampling_draws=[True, True, True],
    )
    # Each later transition loss must retain a graph to every upstream
    # predictor through the primary in-graph re-grounded carrier.
    for loss_index, transition_loss in enumerate(artifact["losses"]):
        for predictor in predictors.values():
            predictor.zero_grad(set_to_none=True)
        transition_loss.backward(retain_graph=loss_index < len(artifact["losses"]) - 1)
        for predictor_index, predictor in enumerate(predictors.values()):
            if predictor_index > loss_index:
                continue
            gradients = [parameter.grad for parameter in predictor.parameters()]
            assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
            assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)
    assert all(parameter.grad is None for model in saes.values() for parameter in model.parameters())
    assert all(torch.equal(before[section][name], value)
               for section, model in saes.items() for name, value in model.state_dict().items())
    assert [item["support"].reshape(2, -1).sum(1).tolist() for item in artifact["trace"]] == [[3, 3]] * 4


def test_level2_chain_carrier_is_independent_of_true_destination_activation_and_ties_are_exact():
    predictors, saes, means, stds, target_ms, activations = _tiny_chain_fixture()
    _, artifact_a = chain_redo.chain_training_loss(
        predictors, saes, means, stds, target_ms, activations,
        scheduled_sampling_probability=1.0, carrier_mode="regrounded",
        sampling_draws=[True, True, True],
    )
    perturbed = dict(activations)
    perturbed["Q2"] = activations["Q2"] + 1000
    _, artifact_b = chain_redo.chain_training_loss(
        predictors, saes, means, stds, target_ms, perturbed,
        scheduled_sampling_probability=1.0, carrier_mode="regrounded",
        sampling_draws=[True, True, True],
    )
    assert torch.equal(artifact_a["trace"][0]["indices"], artifact_b["trace"][0]["indices"])
    assert torch.equal(artifact_a["trace"][0]["support"], artifact_b["trace"][0]["support"])
    assert torch.equal(artifact_a["trace"][0]["carrier"], artifact_b["trace"][0]["carrier"])

    # A tied predicted value field still has exactly the destination M.
    values = torch.ones(2, 8, 4, 4)
    selected, support, indices = chain_redo.exact_mask_values(values, 3)
    assert support.reshape(2, -1).sum(1).tolist() == [3, 3]
    assert selected.reshape(2, -1).count_nonzero(dim=1).tolist() == [3, 3]


def test_level2_chain_loads_real_checkpoint_dimensions_from_mean_and_kmult(tmp_path):
    paths = {}
    for section in chain_redo.SECTIONS:
        model = sae.build_sae("field_strict", 2, 8)
        path = tmp_path / f"{section}.pt"
        torch.save({
            "config": {"sae_type": "field_strict", "budget": 0.01, "Kmult": 4},
            "state_dict": model.state_dict(),
            "mean": torch.zeros(1, 2, 1, 1), "std": torch.ones(1, 2, 1, 1),
        }, path)
        paths[section] = path
    loaded, _ = chain_redo.load_selected_saes(paths, "field_strict", torch.device("cpu"))
    assert all(model.C == 2 and model.K == 8 for model in loaded.values())


def _chain_selection_fixture():
    def rows(agreement, kl, collapsed=()):
        return [{"seed": seed, "non_collapsed": seed not in collapsed,
                 "validation_pred_agree": agreement + seed * 0.001,
                 "validation_kl": kl + seed * 0.0001} for seed in chain_redo.SEEDS]

    return {
        "field_strict": {field: rows(0.91, 0.021) for field in chain_redo.SECTIONS},
        "field_recovery": {field: rows(0.88, 0.025) for field in chain_redo.SECTIONS},
        "vector_context": {field: rows(0.89, 0.023) for field in chain_redo.SECTIONS},
    }


def test_chain_selector_is_content_addressed_and_ranks_only_eligible_new_families(tmp_path):
    artifact = chain_redo.build_chain_selection_artifact(
        _chain_selection_fixture(), backbone="resnet110_cifar100"
    )
    assert artifact["selected_family"] == "field_strict"
    assert artifact["paired_baseline"] == "field_old"
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    loaded = chain_redo.validate_selection_manifest(
        path, expected_backbone="resnet110_cifar100"
    )
    assert chain_redo.selected_chain_pair(path) == ("field_strict", "field_old")
    assert loaded["ranking"][0]["family"] == "field_strict"

    tampered = dict(artifact)
    tampered["selected_family"] = "vector_context"
    path.write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="SHA-256"):
        chain_redo.validate_selection_manifest(path)


def test_chain_selector_rejects_ineligible_winner_and_bad_baseline_pair(tmp_path):
    candidates = _chain_selection_fixture()
    candidates["field_strict"]["Q3"][0]["non_collapsed"] = False
    candidates["field_strict"]["Q3"][1]["non_collapsed"] = False
    artifact = chain_redo.build_chain_selection_artifact(candidates, backbone="r20")
    assert artifact["selected_family"] == "vector_context"
    artifact["paired_baseline"] = "field_recovery"
    canonical = dict(artifact)
    canonical.pop("sha256")
    artifact["sha256"] = chain_redo._digest(canonical)
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(artifact, sort_keys=True))
    with pytest.raises(ValueError, match="field_old"):
        chain_redo.validate_selection_manifest(path)


def test_chain_input_provenance_requires_hashed_selection_cache_backbone_and_splits(tmp_path):
    selection = chain_redo.build_chain_selection_artifact(
        _chain_selection_fixture(), backbone="r20"
    )
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection, sort_keys=True))
    selected_paths, old_paths = {}, {}
    for family, destination in (("field_strict", selected_paths), ("field_old", old_paths)):
        for section in chain_redo.SECTIONS:
            path = tmp_path / family / section / "sae.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"config": {"sae_type": family, "budget": 0.01,
                                    "section": section}}, path)
            destination[section] = path

    def hashed_json(path, payload):
        payload = dict(payload)
        payload["sha256"] = chain_redo._digest(payload)
        path.write_text(json.dumps(payload, sort_keys=True))

    acts = tmp_path / "acts"
    acts.mkdir()
    tensor_hashes = {}
    for role in ("train", "validation", "fidelity_holdout"):
        for name in [f"{role}_{section}.pt" for section in chain_redo.SECTIONS] + [f"{role}_labels.pt"]:
            tensor = acts / name
            torch.save(torch.tensor([1]), tensor)
            tensor_hashes[name] = chain_redo.file_sha256(tensor)
    cache = tmp_path / "cache.json"
    cache_payload = {"schema": "sae-redo-level2-cache-v1", "cache_root": str(acts),
                     "fidelity_split": "fidelity_holdout", "tensors": tensor_hashes}
    hashed_json(cache, cache_payload)
    original_cache_text = cache.read_text()
    backbone = tmp_path / "backbone.pt"
    backbone.write_bytes(b"frozen")
    splits = {}
    for role in ("train", "validation", "fidelity_holdout"):
        split = tmp_path / f"{role}.json"
        hashed_json(split, {"schema": "split-v1", "split": role, "source_ids": [role]})
        splits[role] = split
    evidence = chain_redo.validate_chain_inputs(
        selection_path, "field_strict", selected_paths, old_paths, backbone,
        cache, splits, acts=acts,
    )
    assert evidence["selection"]["sha256"] == chain_redo.file_sha256(selection_path)
    assert set(evidence["splits"]) == {"train", "validation", "fidelity_holdout"}
    cache.write_text(cache.read_text().replace("cache-v1", "cache-tampered"))
    with pytest.raises(ValueError, match="SHA-256"):
        chain_redo.validate_chain_inputs(
            selection_path, "field_strict", selected_paths, old_paths, backbone,
            cache, splits, acts=acts,
        )
    cache.write_text(original_cache_text)
    (acts / "train_Q1.pt").write_bytes(b"tampered-tensor")
    with pytest.raises(ValueError, match="tensor SHA-256"):
        chain_redo.validate_chain_inputs(
            selection_path, "field_strict", selected_paths, old_paths, backbone,
            cache, splits, acts=acts,
        )


def test_chain_split_manifests_require_bound_ordered_ids_and_disjointness(tmp_path):
    def write_manifest(path, payload):
        payload = dict(payload)
        payload["sha256"] = chain_redo._digest(payload)
        path.write_text(json.dumps(payload, sort_keys=True))

    absent = tmp_path / "absent.json"
    write_manifest(absent, {"schema": "split-v1", "split": "train"})
    with pytest.raises(ValueError, match="ordered ID"):
        chain_redo.validate_split_manifest(absent, "train")

    overlap = tmp_path / "overlap.json"
    write_manifest(overlap, {"schema": "split-v1", "split": "validation",
                             "ordered_ids": ["sample-1", "sample-2"]})
    disjoint = tmp_path / "disjoint.json"
    write_manifest(disjoint, {"schema": "split-v1", "split": "fidelity_holdout",
                              "ordered_ids": ["sample-2", "sample-3"]})
    train = tmp_path / "train.json"
    write_manifest(train, {"schema": "split-v1", "split": "train",
                           "ordered_ids": ["sample-1"]})
    evidence = {"train": train, "validation": overlap, "fidelity_holdout": disjoint}
    with pytest.raises(ValueError, match="overlap"):
        chain_redo.validate_split_manifests(evidence)
    assert chain_redo.validate_split_manifest(train, "train")["ordered_ids"] == ["sample-1"]
    assert chain_redo.validate_split_manifest(overlap, "validation")["ordered_ids"] == ["sample-1", "sample-2"]


def test_chain_fidelity_dataset_selection_is_manifest_bound(tmp_path):
    acts = tmp_path / "acts"
    acts.mkdir()
    # Keep both possible fidelity names on disk.  The cache manifest is the
    # authority; file existence must not silently select the alternate split.
    for split, value in (("train", 1), ("validation", 2),
                         ("test", 3), ("fidelity_holdout", 4)):
        for section in chain_redo.SECTIONS:
            torch.save(torch.full((1, 1, 1, 1), float(value)),
                       acts / f"{split}_{section}.pt")
        torch.save(torch.tensor([value]), acts / f"{split}_labels.pt")
    provenance = {"activation_cache": {"fidelity_split": "test"}}
    fidelity_split, datasets = chain_redo.load_chain_datasets(acts, provenance)
    assert fidelity_split == "test"
    assert datasets[fidelity_split][0][0]["Q1"].item() == 3


def test_activation_cache_manifest_rejects_missing_required_tensor(tmp_path):
    acts = tmp_path / "acts"
    acts.mkdir()
    tensor_hashes = {}
    for split in ("train", "validation", "fidelity_holdout"):
        for name in [f"{split}_{section}.pt" for section in chain_redo.SECTIONS] + [f"{split}_labels.pt"]:
            tensor = acts / name
            torch.save(torch.tensor([1]), tensor)
            tensor_hashes[name] = chain_redo.file_sha256(tensor)
    missing = acts / "validation_Q3.pt"
    missing.unlink()
    cache = tmp_path / "cache.json"
    payload = {"schema": "sae-redo-level2-cache-v1", "cache_root": str(acts),
               "fidelity_split": "fidelity_holdout", "tensors": tensor_hashes}
    payload["sha256"] = chain_redo._digest(payload)
    cache.write_text(json.dumps(payload, sort_keys=True))
    with pytest.raises(FileNotFoundError, match="validation_Q3.pt"):
        chain_redo.validate_activation_cache_manifest(cache, acts=acts)


def test_level2_chain_raw_comparator_is_distinct_from_regrounded_rollout():
    predictors, saes, means, stds, target_ms, activations = _tiny_chain_fixture()
    raw, _ = chain_redo.rollout_carrier(
        predictors, saes, means, stds, target_ms, activations["Q1"], mode="raw"
    )
    regrounded, _ = chain_redo.rollout_carrier(
        predictors, saes, means, stds, target_ms, activations["Q1"], mode="regrounded"
    )
    assert raw.shape == regrounded.shape
    assert not torch.equal(raw, regrounded)


def test_level2_chain_rejects_non_one_percent_or_unsupported_selection(tmp_path):
    path = tmp_path / "sae.pt"
    torch.save({"config": {"sae_type": "field_strict", "budget": 0.04}}, path)
    with pytest.raises(ValueError, match="1%"):
        chain_redo.validate_selected_checkpoint(path, "field_strict", "Q1")
    with pytest.raises(ValueError, match="chain family"):
        chain_redo.validate_selected_checkpoint(path, "vector_context_strict", "Q1")


def test_selection_rule_uses_all_three_candidate_seeds_and_the_4pct_gate(tmp_path):
    for sae_type in ("field", "context_vector"):
        for budget, agreement, kl in [(0.08, 0.92, 0.018), (0.04, 0.91, 0.019), (0.02, 0.91 if sae_type == "context_vector" else 0.90, 0.022 if sae_type == "context_vector" else 0.020), (0.01, 0.85, 0.03)]:
            for seed in range(3):
                run = tmp_path / f"vit_base_{sae_type}_b{budget}_s{seed}"
                run.mkdir()
                manifest = write_split_manifest(run, "fixture", {"dataset": "fixture"}, {"train": list(range(2000)), "validation": list(range(500)), "test": list(range(500))}, {"seed": seed})
                active_count = 12
                width = int(active_count / budget)
                coefficient_count = 4 * width
                accounting = {"active_count_m": active_count, "support_index_bits": active_count * math.ceil(math.log2(coefficient_count)), "coefficient_value_bits": 192, "total_sparse_bits": active_count * (math.ceil(math.log2(coefficient_count)) + 16)}
                (run / "vit_result.json").write_text(json.dumps({
                    "model": "vit_base_patch16_224", "block": 10, "sae_type": sae_type, "budget": budget, "budget_fraction_of_original": budget, "seed": seed, "D": int(active_count / budget), "K": 4 * int(active_count / budget),
                    "pred_agree": agreement, "kl": kl, "rel_l2": 0.5, "fvu": 0.25,
                    "manifest_sha256": manifest["sha256"], **accounting, "ntrain": 2000, "nval": 500, "ntest_reserved": 500,
                }))
                (run / "run_metadata.json").write_text(json.dumps({"manifest_sha256": manifest["sha256"], "accounting": accounting, "original_shape": [width, 1, 1], "coefficient_count": coefficient_count}))
                torch.save({"manifest_sha256": manifest["sha256"], "config": {"model": "vit_base_patch16_224", "block": 10, "sae_type": sae_type, "budget": budget, "seed": seed, "Kmult": 4, "epochs": 20, "ntrain": 2000, "nval": 500, "ntest": 500, "data_seed": 0}}, run / "sae.pt")
    out = tmp_path / "selection"
    result = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "select_sae_redo_level2.py"),
         "--matrix-root", str(tmp_path), "--out", str(out)],
        check=True, capture_output=True, text=True,
    )
    assert json.loads((out / "selection.json").read_text())["selected"] == "context_vector"
    assert "context_vector" in result.stdout

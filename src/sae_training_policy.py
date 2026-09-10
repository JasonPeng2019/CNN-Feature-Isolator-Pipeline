"""Shared, deterministic training policy for the Level-2 SAE final cells.

The CNN and ViT lanes have different activation loaders, but their training
curriculum and optimizer policy are intentionally identical.  This module
keeps those rules in one place and returns serializable state alongside the
mutable optimizer/EMA state so a run can be resumed or audited without
reconstructing implicit training decisions.
"""
from contextlib import nullcontext
from dataclasses import dataclass
import copy
import math
import random

import torch


CURRICULUM_TARGET_EPOCHS = 30
MIN_TARGET_EPOCHS = 20
WARMUP_FRACTION = 0.05
PEAK_LEARNING_RATE = 3e-4
FLOOR_LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-4
ADAMW_BETAS = (0.9, 0.95)
GRADIENT_CLIP_NORM = 1.0
EMA_DECAY = 0.999
MAX_RECOVERY_RETRIES = 2


@dataclass(frozen=True)
class PhaseSpec:
    """One immutable curriculum stage."""

    name: str
    epochs: int
    mode: str
    budgets: tuple

    def budget_for_epoch(self, epoch):
        if not 0 <= int(epoch) < self.epochs:
            raise IndexError(f"phase epoch {epoch} is outside {self.name} ({self.epochs})")
        return self.budgets[int(epoch)]

    def as_dict(self):
        return {"name": self.name, "epochs": self.epochs, "mode": self.mode,
                "budgets": list(self.budgets)}


def _validate_budget_dimensions(latent_count, target_m, target_epochs):
    latent_count = int(latent_count)
    target_m = int(target_m)
    target_epochs = int(target_epochs)
    if latent_count < 1:
        raise ValueError("latent_count must be positive")
    if not 1 <= target_m <= latent_count:
        raise ValueError("target_m must be between one and latent_count")
    if target_epochs < MIN_TARGET_EPOCHS:
        raise ValueError(f"target budget requires at least {MIN_TARGET_EPOCHS} epochs")
    return latent_count, target_m, target_epochs


def _geometric_budgets(start, end, steps):
    if int(steps) < 1:
        raise ValueError("geometric continuation needs at least one step")
    start, end, steps = int(start), int(end), int(steps)
    if end > start:
        raise ValueError("target budget cannot exceed the 50% latent budget")
    if steps == 1:
        return (end,)
    ratio = float(end) / float(start)
    values = [max(end, min(start, int(round(start * ratio ** (index / (steps - 1))))))
              for index in range(steps)]
    values[0], values[-1] = start, end
    # Rounding a geometric sequence can otherwise introduce a one-step rise.
    values = [min(values[index], values[index - 1]) if index else values[index]
              for index in range(len(values))]
    values[-1] = end
    return tuple(values)


def build_curriculum_schedule(latent_count, target_m, target_epochs=CURRICULUM_TARGET_EPOCHS):
    """Build the frozen dense-to-target curriculum for one final SAE cell.

    ``target_m`` is already the exact original-node-relative count.  The easy
    phase is exactly half of the latent field, while the geometric phase moves
    monotonically from that count to ``target_m``.  The target stage is never
    shorter than the contract's 20 epochs and defaults to the planned 30.
    """
    latent_count, target_m, target_epochs = _validate_budget_dimensions(
        latent_count, target_m, target_epochs
    )
    half_latent = max(1, round(0.5 * latent_count))
    if target_m > half_latent:
        raise ValueError("target_m must not exceed the exact 50% latent budget")
    return (
        PhaseSpec("dense_strict", 5, "dense", (None,) * 5),
        PhaseSpec("selector_bootstrap", 1, "selector_bootstrap", (None,)),
        PhaseSpec("half_latent", 5, "masked", (half_latent,) * 5),
        PhaseSpec("geometric_continuation", 10, "masked",
                  _geometric_budgets(half_latent, target_m, 10)),
        PhaseSpec("target_budget", target_epochs, "masked", (target_m,) * target_epochs),
    )


# Descriptive aliases keep call sites readable without duplicating policy.
curriculum_phase_schedule = build_curriculum_schedule
build_phase_schedule = build_curriculum_schedule


def build_recovery_schedule(family, target_m):
    """Return the fixed same-budget correction/refiner continuation phases."""
    if family not in {"field_recovery", "vector_context"}:
        raise ValueError(f"recovery schedule is not valid for {family!r}")
    target_m = int(target_m)
    if target_m < 1:
        raise ValueError("target_m must be positive")
    warm_name = "recovery_correction_warm" if family == "field_recovery" else "vector_refiner_warm"
    joint_name = "recovery_joint" if family == "field_recovery" else "vector_joint"
    return (PhaseSpec(warm_name, 3, "masked", (target_m,) * 3),
            PhaseSpec(joint_name, 20, "masked", (target_m,) * 20))


def _is_no_decay(name, parameter):
    lowered = str(name).lower()
    if parameter.ndim <= 1:
        return True
    if "bias" in lowered:
        return True
    if "norm" in lowered or "layer_scale" in lowered or "layerscale" in lowered:
        return True
    # These are projected/1x1 atom parameters renormalized by the SAE after
    # each update; weight decay would fight the explicit unit-norm invariant.
    if lowered.endswith("direct.weight") or lowered.endswith(".dec.weight"):
        return True
    if (lowered.startswith("decoder.") or ".decoder." in lowered) and lowered.endswith(".mix.weight"):
        return True
    return False


def build_optimizer_param_groups(module, weight_decay=WEIGHT_DECAY):
    """Return deterministic AdamW groups with contract-specific exclusions."""
    decay, no_decay = [], []
    decay_names, no_decay_names = [], []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if _is_no_decay(name, parameter):
            no_decay.append(parameter)
            no_decay_names.append(name)
        else:
            decay.append(parameter)
            decay_names.append(name)
    if not decay and not no_decay:
        raise ValueError("training module has no trainable parameters")
    groups = []
    if decay:
        groups.append({"params": decay, "param_names": decay_names,
                       "weight_decay": float(weight_decay)})
    if no_decay:
        groups.append({"params": no_decay, "param_names": no_decay_names,
                       "weight_decay": 0.0})
    return groups


def build_optimizer(module, peak_lr=PEAK_LEARNING_RATE, weight_decay=WEIGHT_DECAY):
    """Construct the shared AdamW optimizer policy."""
    peak_lr = float(peak_lr)
    if peak_lr <= 0:
        raise ValueError("peak learning rate must be positive")
    return torch.optim.AdamW(
        build_optimizer_param_groups(module, weight_decay),
        lr=peak_lr,
        betas=ADAMW_BETAS,
        weight_decay=float(weight_decay),
    )


class WarmupCosineScheduler:
    """A small auditable 5%-warmup/cosine scheduler.

    ``step`` is called once per optimizer update by the shared runner.  The
    first warmup update is one fifth of peak for a 100-step schedule, and the
    final update is exactly the declared floor.
    """

    def __init__(self, optimizer, total_steps, *, warmup_fraction=WARMUP_FRACTION,
                 min_lr=FLOOR_LEARNING_RATE):
        if int(total_steps) < 1:
            raise ValueError("total_steps must be positive")
        if not 0 < float(warmup_fraction) < 1:
            raise ValueError("warmup_fraction must be between zero and one")
        if float(min_lr) < 0:
            raise ValueError("min_lr must be non-negative")
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = max(1, int(math.ceil(self.total_steps * float(warmup_fraction))))
        self.min_lr = float(min_lr)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.last_step = -1
        self.last_lrs = list(self.base_lrs)

    def lr_at(self, step):
        step = max(0, min(int(step), self.total_steps - 1))
        if step < self.warmup_steps:
            fraction = float(step + 1) / float(self.warmup_steps)
            return [base * fraction for base in self.base_lrs]
        elif self.total_steps == self.warmup_steps:
            fraction = 0.0
        else:
            denominator = max(1, self.total_steps - self.warmup_steps - 1)
            progress = min(1.0, max(0.0, float(step - self.warmup_steps) / denominator))
            fraction = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [self.min_lr + (base - self.min_lr) * fraction for base in self.base_lrs]

    def step(self):
        self.last_step += 1
        self.last_lrs = self.lr_at(self.last_step)
        for group, rate in zip(self.optimizer.param_groups, self.last_lrs):
            group["lr"] = rate
        return self.last_lrs[0]

    def get_last_lr(self):
        return list(self.last_lrs)

    def state_dict(self):
        return {"total_steps": self.total_steps, "warmup_steps": self.warmup_steps,
                "min_lr": self.min_lr, "last_step": self.last_step,
                "last_lrs": list(self.last_lrs)}


def bf16_autocast(device, enabled=True):
    """Use bf16 only where CUDA advertises support; otherwise remain FP32."""
    if not enabled:
        return nullcontext()
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


class ExponentialMovingAverage:
    """EMA shadow state with explicit swap/restore for validation."""

    def __init__(self, module, decay=EMA_DECAY):
        if not 0 <= float(decay) < 1:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = float(decay)
        self.updates = 0
        self.shadow = {name: value.detach().clone()
                       for name, value in module.state_dict().items()}

    @torch.no_grad()
    def update(self, module):
        current = module.state_dict()
        if set(current) != set(self.shadow):
            raise ValueError("EMA module state keys changed during training")
        self.updates += 1
        for name, value in current.items():
            shadow = self.shadow[name]
            if torch.is_floating_point(shadow):
                shadow.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                shadow.copy_(value.detach())

    @torch.no_grad()
    def apply_to(self, module):
        backup = {name: value.detach().clone() for name, value in module.state_dict().items()}
        state = module.state_dict()
        for name, value in self.shadow.items():
            state[name].copy_(value)
        return backup

    @torch.no_grad()
    def restore(self, module, backup):
        state = module.state_dict()
        for name, value in backup.items():
            state[name].copy_(value)

    def state_dict(self):
        return {"decay": self.decay, "updates": self.updates,
                "shadow": {name: value.detach().clone() for name, value in self.shadow.items()}}


def _finite_metric(value, label):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} validation metric must be finite")
    return value


def choose_validation_checkpoint(raw_metric, ema_metric, *, minimize=True):
    """Choose raw or EMA validation state with deterministic tie-to-raw rule."""
    raw_metric = _finite_metric(raw_metric, "raw")
    ema_metric = _finite_metric(ema_metric, "EMA")
    if minimize:
        selected = "ema" if ema_metric < raw_metric else "raw"
    else:
        selected = "ema" if ema_metric > raw_metric else "raw"
    return {"selected": selected, "raw_metric": raw_metric,
            "ema_metric": ema_metric, "minimize": bool(minimize)}


class BoundedCollapseRecovery:
    """Track at most two retries and preserve a third-collapse terminal state."""

    def __init__(self, max_retries=MAX_RECOVERY_RETRIES):
        if int(max_retries) < 0:
            raise ValueError("max_retries must be non-negative")
        self.max_retries = int(max_retries)
        self.attempts = 0
        self.terminal_failure = False
        self.events = []

    def observe(self, collapsed, *, phase=None, metric=None):
        collapsed = bool(collapsed)
        if not collapsed:
            action = "stable"
        elif self.attempts < self.max_retries:
            self.attempts += 1
            action = "retry"
        else:
            self.terminal_failure = True
            action = "terminal_failure"
        event = {"collapsed": collapsed, "action": action, "attempts": self.attempts,
                 "terminal_failure": self.terminal_failure,
                 "phase": phase, "metric": None if metric is None else float(metric)}
        self.events.append(event)
        return dict(event)

    def state_dict(self):
        return {"max_retries": self.max_retries, "attempts": self.attempts,
                "terminal_failure": self.terminal_failure, "events": copy.deepcopy(self.events)}


def _clone_state(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _load_state(module, state):
    module.load_state_dict(state, strict=True)


def _call_collapse_detector(detector, module, logs):
    if detector is None:
        return False, None
    result = detector(module, logs)
    if isinstance(result, dict):
        return bool(result.get("collapsed", False)), result.get("metric")
    return bool(result), None


def run_training_policy(module, train_epoch, *, latent_count, target_m,
                        validation_fn=None, target_epochs=CURRICULUM_TARGET_EPOCHS,
                        peak_lr=PEAK_LEARNING_RATE, min_lr=FLOOR_LEARNING_RATE,
                        seed=None, ema_decay=EMA_DECAY, collapse_detector=None,
                        max_recovery_retries=MAX_RECOVERY_RETRIES,
                        phase_schedule=None, phase_configurer=None,
                        updates_per_epoch=1):
    """Run the common curriculum and return logs plus resumable policy state.

    ``train_epoch`` owns the model-specific batch loss but must perform one
    complete optimizer update per batch using the optimizer it receives and
    the phase's ``budget_for_epoch``.  Its signature is
    ``train_epoch(optimizer, phase_spec, phase_epoch)``.  The shared runner
    owns phase order, scheduler, EMA, validation choice, and retry accounting.
    """
    phases = tuple(phase_schedule) if phase_schedule is not None else build_curriculum_schedule(
        latent_count, target_m, target_epochs
    )
    if not phases:
        raise ValueError("training policy requires at least one phase")
    updates_per_epoch = int(updates_per_epoch)
    if updates_per_epoch < 1:
        raise ValueError("updates_per_epoch must be positive")
    total_steps = sum(phase.epochs for phase in phases) * updates_per_epoch
    initial_state = _clone_state(module)
    recovery = BoundedCollapseRecovery(max_recovery_retries)
    attempts = []
    selected_checkpoint = None
    final_optimizer = None
    final_ema = None
    final_logs = []
    final_phase_state = []
    while True:
        attempt_number = len(attempts)
        if attempt_number:
            _load_state(module, initial_state)
        if seed is not None:
            random.seed(int(seed) + attempt_number)
            torch.manual_seed(int(seed) + attempt_number)
        optimizer = build_optimizer(module, peak_lr=peak_lr)
        scheduler = WarmupCosineScheduler(optimizer, total_steps, min_lr=min_lr)
        ema = ExponentialMovingAverage(module, decay=ema_decay)
        logs, phase_state = [], []
        best_score, best_state, best_selection = None, None, None
        global_epoch = 0
        for phase in phases:
            if phase_configurer is not None:
                phase_configurer(phase.name)
            state = {**phase.as_dict(), "epoch_start": global_epoch,
                     "peak_learning_rate": float(peak_lr), "scheduler_start_step": scheduler.last_step}
            for phase_epoch in range(phase.epochs):
                train_result = train_epoch(optimizer, phase, phase_epoch)
                if isinstance(train_result, dict):
                    train_loss = _finite_metric(train_result.get("loss", train_result.get("train_loss", 0.0)), "train")
                    extra = dict(train_result)
                    extra.pop("loss", None); extra.pop("train_loss", None)
                else:
                    train_loss = _finite_metric(train_result, "train")
                    extra = {}
                optimizer_steps = int(extra.pop("optimizer_steps", updates_per_epoch))
                if optimizer_steps < 1:
                    raise ValueError("train_epoch must report at least one optimizer step")
                for _ in range(optimizer_steps):
                    scheduler.step()
                ema.update(module)
                entry = {"epoch": global_epoch, "phase": phase.name,
                         "phase_epoch": phase_epoch, "budget": phase.budget_for_epoch(phase_epoch),
                         "mode": phase.mode, "train_loss": train_loss,
                         "learning_rate": scheduler.get_last_lr()[0], **extra}
                logs.append(entry)
                global_epoch += 1
                # Validation selection belongs to the terminal phase of the
                # supplied schedule.  Continuation schedules terminate in
                # recovery_joint/vector_joint rather than target_budget, and
                # fidelity evaluation must see the selected raw/EMA state.
                if validation_fn is not None and phase.name == phases[-1].name:
                    raw_metric = _finite_metric(validation_fn(module), "raw")
                    backup = ema.apply_to(module)
                    try:
                        ema_metric = _finite_metric(validation_fn(module), "EMA")
                    finally:
                        ema.restore(module, backup)
                    choice = choose_validation_checkpoint(raw_metric, ema_metric)
                    chosen_metric = choice[f"{choice['selected']}_metric"]
                    if best_score is None or chosen_metric < best_score:
                        best_score = chosen_metric
                        best_selection = {**choice, "epoch": global_epoch - 1,
                                          "phase": phase.name}
                        best_state = (_clone_state(module) if choice["selected"] == "raw"
                                      else {name: value.detach().clone() for name, value in ema.shadow.items()})
            state["epoch_end"] = global_epoch
            state["scheduler_end_step"] = scheduler.last_step
            phase_state.append(state)
        collapsed, collapse_metric = _call_collapse_detector(collapse_detector, module, logs)
        event = recovery.observe(collapsed, phase=phases[-1].name, metric=collapse_metric)
        attempts.append({"attempt": attempt_number, "logs": logs,
                         "phase_state": phase_state, "collapse": event,
                         "validation_selection": best_selection})
        final_optimizer, final_ema = optimizer, ema
        final_logs, final_phase_state = logs, phase_state
        if not collapsed:
            selected_checkpoint = best_selection or {
                "selected": "raw", "reason": "validation_not_supplied"
            }
            if best_state is not None:
                _load_state(module, best_state)
            break
        if event["action"] == "terminal_failure":
            selected_checkpoint = {"selected": "none", "terminal_failure": True}
            break
    policy_state = {
        "schema": "sae-redo-level2-training-policy-v1",
        "curriculum": [phase.as_dict() for phase in phases],
        "optimizer": {"name": "AdamW", "betas": list(ADAMW_BETAS),
                      "peak_learning_rate": float(peak_lr),
                      "weight_decay": WEIGHT_DECAY, "gradient_clip_norm": GRADIENT_CLIP_NORM},
        "scheduler": {"warmup_fraction": WARMUP_FRACTION, "floor_learning_rate": float(min_lr),
                      "total_steps": total_steps, "updates_per_epoch": updates_per_epoch},
        "ema": {"decay": float(ema_decay), "updates": final_ema.updates,
                "raw_vs_ema_selection": selected_checkpoint},
        "recovery": recovery.state_dict(),
        "attempt_count": len(attempts),
        "terminal_failure": recovery.terminal_failure,
    }
    return {
        "logs": final_logs,
        "phase_state": final_phase_state,
        "attempts": attempts,
        "policy_state": policy_state,
        "state": policy_state,
        "optimizer_state": final_optimizer.state_dict(),
        "ema_state": final_ema.state_dict(),
        "validation_selection": selected_checkpoint,
    }


def optimizer_step(optimizer, module, loss, *, clip_norm=GRADIENT_CLIP_NORM):
    """Shared FP32 loss/backward/clipping/update helper for trainer callbacks."""
    loss = loss.float()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(module.parameters(), float(clip_norm))
    optimizer.step()
    return float(loss.detach().item())

"""Interruption-equivalence harness for full-resume training (Requirement 14.7, 19.6).

A run that is interrupted after ``K`` updates, checkpointed through
:mod:`~pursuit_evasion_rl.research.training.checkpoint`, reloaded into a *fresh*
:class:`~pursuit_evasion_rl.research.training.trainer.ResearchTrainer` and
continued for the remaining ``N - K`` updates must produce exactly the trace a
continuous ``N``-update run produces, for every ``0 <= K <= N``.

Two comparison paths are kept separate.  The CPU path is bitwise: parameter
tensors must satisfy :func:`torch.equal`, and losses/capture-rates must be
equal as exact floats.  The GPU path re-runs the same comparison with the
protocol's ``rtol``/``atol`` and is skipped cleanly -- reported, never failed --
when no CUDA device is present.  A divergence is never downgraded to a pass: it
becomes an :class:`EquivalenceReport` with ``equivalent=False`` and a content
hash, and :func:`assert_equivalent` raises ``INTERRUPTION_DIVERGENCE``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Sequence
from uuid import uuid4

import torch
from torch import Tensor

from ..canonical import content_hash
from ..errors import ResearchValidationError
from ..maps.splits import DataHandle, HandleKind, SplitScope, TuningDataView
from ..policies.masked_mappo import PPOLoss
from .checkpoint import (
    CompatibilityContract,
    ResumeState,
    capture_trainer_resume_state,
    load_resume_state,
    restore_trainer_resume_state,
    save_resume_state,
)
from .trainer import ResearchTrainer, TrainerConfig, UpdateOutcome

EQUIVALENCE_SCHEMA_VERSION = "interruption-equivalence-v1"

#: Protocol tolerances for the GPU path only; the CPU path is always bitwise.
DEFAULT_GPU_RTOL = 1e-5
DEFAULT_GPU_ATOL = 1e-7

_EQUIVALENCE_CONDITION_ID = "interruption-equivalence"


def _fail(code: str, message: str, **kwargs) -> None:
    raise ResearchValidationError(code, message, **kwargs)


def _tuning_view() -> TuningDataView:
    return TuningDataView(
        train=(DataHandle(SplitScope.TRAIN, HandleKind.MAP, "equivalence-train"),),
        validation=(DataHandle(SplitScope.VALIDATION, HandleKind.MAP, "equivalence-validation"),),
    )


def _digest(tensors: Sequence[Tensor]) -> str:
    accumulator = hashlib.sha256()
    for tensor in tensors:
        detached = tensor.detach().cpu().contiguous()
        accumulator.update(str(detached.dtype).encode("utf-8"))
        accumulator.update(str(tuple(detached.shape)).encode("utf-8"))
        accumulator.update(detached.numpy().tobytes())
    return accumulator.hexdigest()


def _policy_parameters(trainer: ResearchTrainer) -> tuple[Tensor, ...]:
    return tuple(
        parameter.detach().cpu().clone()
        for parameter in (*trainer.policy.actor.parameters(), *trainer.policy.critic.parameters())
    )


def _optimizer_tensors(trainer: ResearchTrainer) -> tuple[Tensor, ...]:
    collected: list[Tensor] = []
    for group in trainer.policy.optimizer.state_dict()["state"].values():
        for key in sorted(group):
            value = group[key]
            # Cloned: the optimizer's own state tensors keep mutating in later updates.
            collected.append(value.detach().cpu().clone() if isinstance(value, Tensor) else torch.tensor(float(value)))
    return tuple(collected)


@dataclass(frozen=True, slots=True)
class UpdateTrace:
    """One update's externally observable result plus its full internal state fingerprint."""

    update_index: int
    transition_count: int
    train_capture_rate: float
    validation_capture_rate: float
    losses: tuple[tuple[float, ...], ...]
    parameter_digest: str
    optimizer_digest: str
    training_generator_digest: str
    episode_index: int
    environment_step_index: int
    parameters: tuple[Tensor, ...] = field(repr=False, default=())
    optimizer_tensors: tuple[Tensor, ...] = field(repr=False, default=())

    @property
    def scalar_summary(self) -> dict[str, Any]:
        return {
            "update_index": self.update_index,
            "transition_count": self.transition_count,
            "train_capture_rate": self.train_capture_rate,
            "validation_capture_rate": self.validation_capture_rate,
            "losses": [list(item) for item in self.losses],
            "parameter_digest": self.parameter_digest,
            "optimizer_digest": self.optimizer_digest,
            "training_generator_digest": self.training_generator_digest,
            "episode_index": self.episode_index,
            "environment_step_index": self.environment_step_index,
        }


def _loss_values(loss: PPOLoss) -> tuple[float, ...]:
    return (loss.policy_loss, loss.value_loss, loss.entropy, loss.total_loss, loss.ratio_mean)


def _trace(trainer: ResearchTrainer, outcome: UpdateOutcome) -> UpdateTrace:
    parameters = _policy_parameters(trainer)
    optimizer_tensors = _optimizer_tensors(trainer)
    return UpdateTrace(
        update_index=outcome.update_index,
        transition_count=outcome.transition_count,
        train_capture_rate=outcome.train_capture_rate,
        validation_capture_rate=outcome.validation_capture_rate,
        losses=tuple(_loss_values(item) for item in outcome.losses),
        parameter_digest=_digest(parameters),
        optimizer_digest=_digest(optimizer_tensors),
        training_generator_digest=_digest((trainer._training_rng.get_state(),)),
        episode_index=trainer.episode_index,
        environment_step_index=trainer.environment_step_index,
        parameters=parameters,
        optimizer_tensors=optimizer_tensors,
    )


def trace_hash(trace: Sequence[UpdateTrace]) -> str:
    """Content hash of a whole trace, used as the failure record's artifact hash."""
    return content_hash([item.scalar_summary for item in trace])


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def build_equivalence_trainer(
    network,
    config: TrainerConfig,
    training_seed: int,
    *,
    output_root: str | Path,
    run_id: str | None = None,
) -> ResearchTrainer:
    """A trainer over one tiny network used as both the train and validation map."""
    return ResearchTrainer(
        train_network=network,
        validation_network=network,
        tuning_data=_tuning_view(),
        config=config,
        condition_id=_EQUIVALENCE_CONDITION_ID,
        training_seed=training_seed,
        output_root=output_root,
        run_id=run_id or f"equivalence-{uuid4().hex[:12]}",
    )


def equivalence_contract(trainer: ResearchTrainer) -> CompatibilityContract:
    """The harness's fixed identity half plus the trainer's real tensor shapes."""
    return CompatibilityContract.for_policy(
        trainer.policy,
        code_hash="equivalence-code",
        dependency_hash="equivalence-dependency",
        protocol_hash="equivalence-protocol",
        condition_hash=_EQUIVALENCE_CONDITION_ID,
        map_hash="equivalence-map",
        split="train",
    )


def _validate_updates(updates: int, config: TrainerConfig) -> int:
    resolved = config.updates if updates is None else int(updates)
    if isinstance(updates, bool) or resolved <= 0:
        _fail("INVALID_EQUIVALENCE_RANGE", "updates must be a positive integer", path="updates", actual=updates)
    return resolved


def run_continuous(
    network,
    config: TrainerConfig,
    training_seed: int,
    *,
    updates: int | None = None,
    output_root: str | Path,
) -> tuple[UpdateTrace, ...]:
    """Train straight through ``updates`` updates and return the per-update trace."""
    total = _validate_updates(updates, config)
    trainer = build_equivalence_trainer(network, config, training_seed, output_root=output_root)
    traces: list[UpdateTrace] = []
    for update_index in range(1, total + 1):
        traces.append(_trace(trainer, trainer.execute_update(update_index)))
    return tuple(traces)


def run_interrupted(
    network,
    config: TrainerConfig,
    training_seed: int,
    *,
    updates: int | None = None,
    interrupt_after: int,
    checkpoint_path: str | Path,
    output_root: str | Path,
    restore: Callable[[ResearchTrainer, ResumeState], None] | None = None,
) -> tuple[UpdateTrace, ...]:
    """Run ``interrupt_after`` updates, checkpoint, resume into a fresh trainer, finish.

    ``restore`` overrides how the Resume_State is applied; the mutation tests
    use it to drop a single stream and prove the comparison detects it.
    """
    total = _validate_updates(updates, config)
    if isinstance(interrupt_after, bool) or not isinstance(interrupt_after, int) or not 0 <= interrupt_after <= total:
        _fail(
            "INVALID_EQUIVALENCE_RANGE", "interrupt_after must satisfy 0 <= K <= N",
            path="interrupt_after", expected=[0, total], actual=interrupt_after,
        )
    first = build_equivalence_trainer(network, config, training_seed, output_root=output_root)
    traces: list[UpdateTrace] = []
    for update_index in range(1, interrupt_after + 1):
        traces.append(_trace(first, first.execute_update(update_index)))

    save_resume_state(checkpoint_path, capture_trainer_resume_state(first), equivalence_contract(first))
    second = build_equivalence_trainer(network, config, training_seed, output_root=output_root)
    state = load_resume_state(checkpoint_path, expected_contract=equivalence_contract(second))
    (restore or restore_trainer_resume_state)(second, state)

    for update_index in range(second.update_index + 1, total + 1):
        traces.append(_trace(second, second.execute_update(update_index)))
    return tuple(traces)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EquivalenceReport:
    """A recordable verdict; a divergence is a failure record, never a pass."""

    mode: str
    equivalent: bool
    update_count: int
    continuous_hash: str
    interrupted_hash: str
    divergences: tuple[str, ...] = ()
    rtol: float | None = None
    atol: float | None = None
    skipped_reason: str | None = None
    schema_version: str = EQUIVALENCE_SCHEMA_VERSION

    @property
    def content_hash(self) -> str:
        return content_hash(self)


def _scalar_equal(left: float, right: float, rtol: float | None, atol: float | None) -> bool:
    if rtol is None or atol is None:
        return left == right
    return math.isclose(left, right, rel_tol=rtol, abs_tol=atol)


def _tensors_equal(left: Sequence[Tensor], right: Sequence[Tensor], rtol: float | None, atol: float | None) -> bool:
    if len(left) != len(right):
        return False
    if rtol is None or atol is None:
        return all(torch.equal(a, b) for a, b in zip(left, right))
    return all(a.shape == b.shape and torch.allclose(a, b, rtol=rtol, atol=atol) for a, b in zip(left, right))


def compare_traces(
    continuous: Sequence[UpdateTrace],
    interrupted: Sequence[UpdateTrace],
    *,
    rtol: float | None = None,
    atol: float | None = None,
) -> EquivalenceReport:
    """Compare two traces; ``rtol``/``atol`` omitted means bitwise CPU comparison."""
    exact = rtol is None and atol is None
    mode = "cpu_bitwise" if exact else "gpu_tolerance"
    divergences: list[str] = []
    if len(continuous) != len(interrupted):
        divergences.append(f"update count {len(continuous)} != {len(interrupted)}")
    for index, (left, right) in enumerate(zip(continuous, interrupted)):
        prefix = f"update[{index}]"
        for name in ("update_index", "transition_count", "episode_index", "environment_step_index"):
            if getattr(left, name) != getattr(right, name):
                divergences.append(f"{prefix}.{name} {getattr(left, name)} != {getattr(right, name)}")
        for name in ("train_capture_rate", "validation_capture_rate"):
            if not _scalar_equal(getattr(left, name), getattr(right, name), rtol, atol):
                divergences.append(f"{prefix}.{name} {getattr(left, name)!r} != {getattr(right, name)!r}")
        if len(left.losses) != len(right.losses):
            divergences.append(f"{prefix}.losses length {len(left.losses)} != {len(right.losses)}")
        else:
            for epoch, (loss_left, loss_right) in enumerate(zip(left.losses, right.losses)):
                if any(not _scalar_equal(a, b, rtol, atol) for a, b in zip(loss_left, loss_right)):
                    divergences.append(f"{prefix}.losses[{epoch}] {loss_left!r} != {loss_right!r}")
        if not _tensors_equal(left.parameters, right.parameters, rtol, atol):
            divergences.append(f"{prefix}.parameters diverged")
        if not _tensors_equal(left.optimizer_tensors, right.optimizer_tensors, rtol, atol):
            divergences.append(f"{prefix}.optimizer diverged")
        if exact:
            for name in ("parameter_digest", "optimizer_digest", "training_generator_digest"):
                if getattr(left, name) != getattr(right, name):
                    divergences.append(f"{prefix}.{name} {getattr(left, name)} != {getattr(right, name)}")
        elif left.training_generator_digest != right.training_generator_digest:
            # An RNG state is bytes: it is never "close", so it is exact in both modes.
            divergences.append(f"{prefix}.training_generator_digest diverged")
    return EquivalenceReport(
        mode=mode,
        equivalent=not divergences,
        update_count=len(continuous),
        continuous_hash=trace_hash(continuous),
        interrupted_hash=trace_hash(interrupted),
        divergences=tuple(divergences),
        rtol=rtol,
        atol=atol,
    )


def assert_equivalent(
    continuous: Sequence[UpdateTrace],
    interrupted: Sequence[UpdateTrace],
    *,
    rtol: float | None = None,
    atol: float | None = None,
) -> EquivalenceReport:
    """Return the report when equivalent; raise ``INTERRUPTION_DIVERGENCE`` otherwise."""
    report = compare_traces(continuous, interrupted, rtol=rtol, atol=atol)
    if not report.equivalent:
        _fail(
            "INTERRUPTION_DIVERGENCE",
            "resumed run diverged from the continuous run",
            path="trace",
            expected=report.continuous_hash,
            actual=report.interrupted_hash,
            details={
                "mode": report.mode,
                "record_hash": report.content_hash,
                "divergences": list(report.divergences),
            },
        )
    return report


def compare_traces_on_gpu_path(
    continuous: Sequence[UpdateTrace],
    interrupted: Sequence[UpdateTrace],
    *,
    rtol: float = DEFAULT_GPU_RTOL,
    atol: float = DEFAULT_GPU_ATOL,
) -> EquivalenceReport:
    """The tolerance path, reported as skipped (not failed) with no CUDA device."""
    if not torch.cuda.is_available():
        return EquivalenceReport(
            mode="gpu_tolerance",
            equivalent=True,
            update_count=len(continuous),
            continuous_hash=trace_hash(continuous),
            interrupted_hash=trace_hash(interrupted),
            rtol=rtol,
            atol=atol,
            skipped_reason="no CUDA device is available",
        )
    return compare_traces(continuous, interrupted, rtol=rtol, atol=atol)


__all__ = (
    "DEFAULT_GPU_ATOL",
    "DEFAULT_GPU_RTOL",
    "EQUIVALENCE_SCHEMA_VERSION",
    "EquivalenceReport",
    "UpdateTrace",
    "assert_equivalent",
    "build_equivalence_trainer",
    "compare_traces",
    "compare_traces_on_gpu_path",
    "equivalence_contract",
    "run_continuous",
    "run_interrupted",
    "trace_hash",
)

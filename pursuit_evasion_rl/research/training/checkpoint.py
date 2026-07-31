"""Full Resume_State checkpoint IO and the resume compatibility gate.

Requirement 14.5 demands that a Resume_State carry *every* stream needed to
continue a run bit-exactly: actor/critic, optimizer, scheduler, scaler, the
episode/update/environment-step indices, Python/NumPy/Torch-CPU/Torch-device
RNG, the environment/evader/placement/sampler RNG, curriculum and hysteresis
state, running normalization, pending SMDP transitions, the rollout buffer and
the parent run link.  The trainer in :mod:`~pursuit_evasion_rl.research.training.trainer`
does not yet own a scheduler, scaler, curriculum, hysteresis wrapper, running
normalization or cross-rollout pending SMDP state, so those fields are carried
as explicit ``None`` -- the project preserves ``null`` rather than omitting it,
so a later trainer can populate them without a schema change and a reader can
always tell "not applicable" from "forgotten".

Requirement 14.6-14.8: loading runs a compatibility gate over code, dependency,
protocol, condition, map, split, tensor shapes and the Resume_State content
hash.  A gate failure raises a coded :class:`ResearchValidationError` -- it
never silently starts fresh, so the caller can record a new lineage or a
``failed`` status against the run manifest store.

**What the content hash covers.**  The payload holds raw tensors and is written
with :func:`torch.save`, which is not byte-stable across re-saves, so the hash
is *not* taken over the file.  :func:`resume_state_content_hash` builds a
canonical JSON summary of the whole Resume_State in which every tensor is
replaced by ``{dtype, shape, sha256(raw contiguous bytes)}`` and every ``bytes``
value by ``{length, sha256}``; all other values are canonicalised verbatim.  The
SHA-256 of that canonical JSON is the content hash.  It therefore changes when
any tensor payload, dtype, shape, scalar, index or RNG stream changes, and is
identical for two saves of the same in-memory state.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
import hashlib
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from ..canonical import canonical_data, content_hash
from ..errors import ResearchValidationError
from ..smdp import DecisionEpoch, PendingDecision
from .trainer import ResearchTrainer

RESUME_STATE_SCHEMA_VERSION = "resume-state-v1"

#: Fields a Resume_State cannot be loaded without (Requirement 14.7).
REQUIRED_RESUME_FIELDS = (
    "policy_state_dict",
    "optimizer_state_dict",
    "episode_index",
    "update_index",
    "environment_step_index",
    "training_seed",
    "python_random_state",
    "numpy_random_state",
    "torch_cpu_rng_state",
    "training_generator_state",
)

#: Fields that are part of the schema but legitimately ``None`` today.
OPTIONAL_RESUME_FIELDS = (
    "scheduler_state",
    "scaler_state",
    "torch_device_rng_state",
    "evader_rng_state",
    "placement_rng_state",
    "sampler_rng_state",
    "curriculum_state",
    "hysteresis_state",
    "running_normalization",
    "pending_transitions",
    "rollout_buffer",
    "parent_run_id",
)


def _fail(code: str, message: str, **kwargs) -> None:
    raise ResearchValidationError(code, message, **kwargs)


def _snapshot(value: Any) -> Any:
    """Deep-copy tensors out of live training state so a Resume_State cannot drift."""
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_snapshot(item) for item in value)
    return value


# ---------------------------------------------------------------------------
# RNG stream capture/restore
# ---------------------------------------------------------------------------


def _jsonify(value: Any) -> Any:
    """Convert a NumPy/Python RNG state into canonical-JSON-safe data."""
    if isinstance(value, np.ndarray):
        return [_jsonify(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonify(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonify(item) for item in value]
    return value


def capture_python_random_state() -> dict[str, Any]:
    version, keys, gauss_next = random.getstate()
    return {"version": int(version), "keys": [int(item) for item in keys], "gauss_next": gauss_next}


def restore_python_random_state(state: Mapping[str, Any]) -> None:
    random.setstate((int(state["version"]), tuple(int(item) for item in state["keys"]), state["gauss_next"]))


def capture_numpy_random_state() -> dict[str, Any]:
    """Capture the legacy global ``numpy.random`` MT19937 stream."""
    bit_generator, keys, position, has_gauss, cached_gaussian = np.random.get_state(legacy=True)
    return {
        "bit_generator": str(bit_generator),
        "keys": [int(item) for item in keys],
        "pos": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def restore_numpy_random_state(state: Mapping[str, Any]) -> None:
    np.random.set_state((
        str(state["bit_generator"]),
        np.array(state["keys"], dtype=np.uint32),
        int(state["pos"]),
        int(state["has_gauss"]),
        float(state["cached_gaussian"]),
    ))


def capture_generator_state(rng: np.random.Generator) -> dict[str, Any]:
    """Capture a dedicated ``numpy.random.Generator`` stream (evader/placement/sampler)."""
    if not isinstance(rng, np.random.Generator):
        _fail("INVALID_RNG_STREAM", "a numpy Generator is required", actual=type(rng).__name__)
    return _jsonify(rng.bit_generator.state)


def restore_generator_state(rng: np.random.Generator, state: Mapping[str, Any]) -> None:
    if not isinstance(rng, np.random.Generator):
        _fail("INVALID_RNG_STREAM", "a numpy Generator is required", actual=type(rng).__name__)
    rng.bit_generator.state = _jsonify(state)


def capture_device_rng_state() -> dict[str, Any] | None:
    """Per-device CUDA RNG, or ``None`` on a CPU-only host (never raises)."""
    if not torch.cuda.is_available():
        return None
    states = torch.cuda.get_rng_state_all()
    if not states:
        return None
    return {"cuda": tuple(item.detach().cpu().clone() for item in states)}


def restore_device_rng_state(state: Mapping[str, Any] | None) -> None:
    if not state:
        return
    cuda_states = state.get("cuda")
    if not cuda_states or not torch.cuda.is_available():
        return
    torch.cuda.set_rng_state_all([torch.as_tensor(item).cpu().clone() for item in cuda_states])


# ---------------------------------------------------------------------------
# Pending SMDP transitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PendingTransitionSnapshot:
    """A mid-rollout :class:`PendingDecision` in serialisable form."""

    officer_id: int
    actor_obs: tuple[float, ...]
    action: int
    stored_mask_bytes: bytes
    old_log_prob: float
    critic_context: tuple[float, ...]
    discounted_reward: float
    duration_steps: int

    def __post_init__(self) -> None:
        if isinstance(self.officer_id, bool) or not isinstance(self.officer_id, int) or self.officer_id < 0:
            _fail("INVALID_PENDING_SNAPSHOT", "officer_id must be a nonnegative integer", path="officer_id", actual=self.officer_id)
        if isinstance(self.duration_steps, bool) or not isinstance(self.duration_steps, int) or self.duration_steps < 0:
            _fail("INVALID_PENDING_SNAPSHOT", "duration_steps must be a nonnegative integer", path="duration_steps", actual=self.duration_steps)
        object.__setattr__(self, "actor_obs", tuple(float(item) for item in self.actor_obs))
        object.__setattr__(self, "critic_context", tuple(float(item) for item in self.critic_context))
        object.__setattr__(self, "stored_mask_bytes", bytes(self.stored_mask_bytes))
        object.__setattr__(self, "old_log_prob", float(self.old_log_prob))
        object.__setattr__(self, "discounted_reward", float(self.discounted_reward))

    @classmethod
    def from_pending(cls, pending: PendingDecision) -> "PendingTransitionSnapshot":
        if not isinstance(pending, PendingDecision):
            _fail("INVALID_PENDING_SNAPSHOT", "a PendingDecision is required", actual=type(pending).__name__)
        return cls(
            officer_id=pending.officer_id,
            actor_obs=pending.epoch.actor_obs,
            action=pending.epoch.action,
            stored_mask_bytes=pending.epoch.stored_mask_bytes,
            old_log_prob=pending.epoch.old_log_prob,
            critic_context=pending.epoch.critic_context,
            discounted_reward=pending.discounted_reward,
            duration_steps=pending.duration_steps,
        )

    def to_pending(self) -> PendingDecision:
        epoch = DecisionEpoch.create(
            actor_obs=self.actor_obs,
            action=self.action,
            action_mask=self.stored_mask_bytes,
            old_log_prob=self.old_log_prob,
            critic_context=self.critic_context,
        )
        return PendingDecision(
            officer_id=self.officer_id,
            epoch=epoch,
            discounted_reward=self.discounted_reward,
            duration_steps=self.duration_steps,
        )

    def as_payload(self) -> dict[str, Any]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


# ---------------------------------------------------------------------------
# Resume_State
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResumeState:
    """Everything needed to continue a run exactly where it stopped (Requirement 14.5)."""

    policy_state_dict: Mapping[str, Mapping[str, Tensor]]
    optimizer_state_dict: Mapping[str, Any]
    episode_index: int
    update_index: int
    environment_step_index: int
    training_seed: int
    python_random_state: Mapping[str, Any]
    numpy_random_state: Mapping[str, Any]
    torch_cpu_rng_state: Tensor
    training_generator_state: Tensor
    scheduler_state: Mapping[str, Any] | None = None
    scaler_state: Mapping[str, Any] | None = None
    torch_device_rng_state: Mapping[str, Any] | None = None
    evader_rng_state: Mapping[str, Any] | None = None
    placement_rng_state: Mapping[str, Any] | None = None
    sampler_rng_state: Mapping[str, Any] | None = None
    curriculum_state: Mapping[str, Any] | None = None
    hysteresis_state: Mapping[str, Any] | None = None
    running_normalization: Mapping[str, Any] | None = None
    pending_transitions: tuple[PendingTransitionSnapshot, ...] | None = None
    rollout_buffer: Mapping[str, Any] | None = None
    parent_run_id: str | None = None
    schema_version: str = RESUME_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("policy_state_dict", "optimizer_state_dict", "python_random_state", "numpy_random_state"):
            if not isinstance(getattr(self, name), Mapping):
                _fail("MISSING_RESUME_STATE_FIELD", f"{name} must be a mapping", path=name, actual=type(getattr(self, name)).__name__)
        for name in ("actor", "critic"):
            if name not in self.policy_state_dict:
                _fail("MISSING_RESUME_STATE_FIELD", f"policy_state_dict is missing {name!r}", path=f"policy_state_dict.{name}")
        for name in ("policy_state_dict", "optimizer_state_dict"):
            object.__setattr__(self, name, _snapshot(getattr(self, name)))
        for name in ("episode_index", "update_index", "environment_step_index", "training_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                _fail("INVALID_RESUME_STATE_FIELD", f"{name} must be a nonnegative integer", path=name, actual=value)
        for name in ("torch_cpu_rng_state", "training_generator_state"):
            value = getattr(self, name)
            if not isinstance(value, Tensor):
                _fail("MISSING_RESUME_STATE_FIELD", f"{name} must be a torch RNG state tensor", path=name, actual=type(value).__name__)
            object.__setattr__(self, name, value.detach().cpu().to(torch.uint8).clone())
        if self.pending_transitions is not None:
            snapshots = tuple(self.pending_transitions)
            if any(not isinstance(item, PendingTransitionSnapshot) for item in snapshots):
                _fail("INVALID_RESUME_STATE_FIELD", "pending_transitions must hold PendingTransitionSnapshot values", path="pending_transitions")
            object.__setattr__(self, "pending_transitions", snapshots)
        if self.parent_run_id is not None and (not isinstance(self.parent_run_id, str) or not self.parent_run_id.strip()):
            _fail("INVALID_RESUME_STATE_FIELD", "parent_run_id must be a non-empty run id or None", path="parent_run_id", actual=self.parent_run_id)

    @property
    def content_hash(self) -> str:
        return resume_state_content_hash(self)


def _state_payload(state: ResumeState) -> dict[str, Any]:
    payload = {item.name: getattr(state, item.name) for item in fields(state)}
    if state.pending_transitions is not None:
        payload["pending_transitions"] = [item.as_payload() for item in state.pending_transitions]
    return payload


def _state_from_payload(payload: Mapping[str, Any]) -> ResumeState:
    known = {item.name for item in fields(ResumeState)}
    missing = [name for name in REQUIRED_RESUME_FIELDS if payload.get(name) is None]
    if missing:
        _fail(
            "INCOMPLETE_RESUME_STATE",
            "Resume_State is missing required fields and cannot resume the original run",
            path="state", expected=list(REQUIRED_RESUME_FIELDS), actual=missing,
        )
    unknown = sorted(set(payload) - known)
    if unknown:
        _fail("UNKNOWN_RESUME_STATE_FIELD", "Resume_State carries unknown fields", path="state", actual=unknown)
    arguments = dict(payload)
    pending = arguments.get("pending_transitions")
    if pending is not None:
        arguments["pending_transitions"] = tuple(PendingTransitionSnapshot(**dict(item)) for item in pending)
    return ResumeState(**arguments)


# ---------------------------------------------------------------------------
# Content hash
# ---------------------------------------------------------------------------


def _hashable(value: Any, path: str) -> Any:
    """Canonical, JSON-safe summary of a payload value (see the module docstring)."""
    if isinstance(value, Tensor):
        detached = value.detach().cpu().contiguous()
        raw = detached.numpy().tobytes()
        return {
            "__tensor__": {
                "dtype": str(detached.dtype),
                "shape": list(detached.shape),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        }
    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
        return {"__bytes__": {"length": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}}
    if isinstance(value, np.ndarray):
        return _hashable(torch.as_tensor(value), path)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _hashable(item, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_hashable(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    _fail("UNHASHABLE_RESUME_STATE_VALUE", f"{path} has unsupported type {type(value).__name__}", path=path, actual=type(value).__name__)


def resume_state_content_hash(state: ResumeState) -> str:
    """SHA-256 over the canonical tensor-summarising JSON of the whole Resume_State."""
    if not isinstance(state, ResumeState):
        _fail("INVALID_RESUME_STATE", "a ResumeState is required", actual=type(state).__name__)
    return content_hash(_hashable(_state_payload(state), "$"))


# ---------------------------------------------------------------------------
# Compatibility gate (Requirement 14.6-14.7)
# ---------------------------------------------------------------------------


def policy_hidden_dims(policy: Any) -> tuple[int, ...]:
    """The actor MLP's hidden widths, read back from the built network."""
    widths = [module.out_features for module in policy.actor.network if isinstance(module, nn.Linear)]
    return tuple(int(width) for width in widths[:-1])


@dataclass(frozen=True, slots=True)
class CompatibilityContract:
    """The identity a resumed run must still match (Requirement 14.6)."""

    code_hash: str
    dependency_hash: str
    protocol_hash: str
    condition_hash: str
    map_hash: str
    split: str
    actor_obs_dim: int
    critic_context_dim: int
    action_dim: int
    num_officers: int
    hidden_dims: tuple[int, ...]
    resume_state_hash: str | None = None
    schema_version: str = RESUME_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("code_hash", "dependency_hash", "protocol_hash", "condition_hash", "map_hash", "split"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                _fail("INVALID_COMPATIBILITY_CONTRACT", f"{name} must be a non-empty string", path=name, actual=value)
        for name in ("actor_obs_dim", "critic_context_dim", "action_dim", "num_officers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                _fail("INVALID_COMPATIBILITY_CONTRACT", f"{name} must be a positive integer", path=name, actual=value)
        dims = tuple(int(item) for item in self.hidden_dims)
        if not dims or any(item <= 0 for item in dims):
            _fail("INVALID_COMPATIBILITY_CONTRACT", "hidden_dims must be positive and non-empty", path="hidden_dims", actual=list(dims))
        object.__setattr__(self, "hidden_dims", dims)

    @classmethod
    def for_policy(
        cls,
        policy: Any,
        *,
        code_hash: str,
        dependency_hash: str,
        protocol_hash: str,
        condition_hash: str,
        map_hash: str,
        split: str,
        resume_state_hash: str | None = None,
    ) -> "CompatibilityContract":
        """Read the tensor-shape half of the contract off a live policy."""
        return cls(
            code_hash=code_hash,
            dependency_hash=dependency_hash,
            protocol_hash=protocol_hash,
            condition_hash=condition_hash,
            map_hash=map_hash,
            split=split,
            actor_obs_dim=int(policy.actor_obs_dim),
            critic_context_dim=int(policy.critic_context_dim),
            action_dim=int(policy.action_dim),
            num_officers=int(policy.num_officers),
            hidden_dims=policy_hidden_dims(policy),
            resume_state_hash=resume_state_hash,
        )


_IDENTITY_FIELDS = ("code_hash", "dependency_hash", "protocol_hash", "condition_hash", "map_hash", "split")
_SHAPE_FIELDS = ("actor_obs_dim", "critic_context_dim", "action_dim", "num_officers", "hidden_dims")


def check_resume_compatibility(saved: CompatibilityContract, current: CompatibilityContract) -> None:
    """Fail closed unless the saved run and the current process are the same experiment.

    Raises :class:`ResearchValidationError` with ``RESUME_COMPATIBILITY_GATE_FAILED``
    (identity drift), ``RESUME_TENSOR_SHAPE_MISMATCH`` (architecture drift) or
    ``RESUME_STATE_HASH_MISMATCH`` (state drift).  The caller must treat any of
    these as "refuse to resume the original run" and record a new lineage or a
    ``failed`` status; there is no partial-resume path.
    """
    for value, name in ((saved, "saved"), (current, "current")):
        if not isinstance(value, CompatibilityContract):
            _fail("INVALID_COMPATIBILITY_CONTRACT", f"{name} contract must be a CompatibilityContract", path=name, actual=type(value).__name__)
    if saved.schema_version != current.schema_version:
        _fail(
            "RESUME_COMPATIBILITY_GATE_FAILED", "Resume_State schema version changed",
            path="schema_version", expected=saved.schema_version, actual=current.schema_version,
        )
    for name in _IDENTITY_FIELDS:
        if getattr(saved, name) != getattr(current, name):
            _fail(
                "RESUME_COMPATIBILITY_GATE_FAILED", f"{name} changed since the checkpoint was written",
                path=name, expected=getattr(saved, name), actual=getattr(current, name),
            )
    for name in _SHAPE_FIELDS:
        if getattr(saved, name) != getattr(current, name):
            _fail(
                "RESUME_TENSOR_SHAPE_MISMATCH", f"{name} changed since the checkpoint was written",
                path=name, expected=getattr(saved, name), actual=getattr(current, name),
            )
    # An expected hash is only compared when the caller actually knows it.
    if current.resume_state_hash is not None and saved.resume_state_hash != current.resume_state_hash:
        _fail(
            "RESUME_STATE_HASH_MISMATCH", "Resume_State content hash does not match the expected hash",
            path="resume_state_hash", expected=saved.resume_state_hash, actual=current.resume_state_hash,
        )


# ---------------------------------------------------------------------------
# Atomic checkpoint IO
# ---------------------------------------------------------------------------


def save_resume_state(path: str | Path, state: ResumeState, contract: CompatibilityContract) -> str:
    """Atomically write a Resume_State and return its content hash."""
    if not isinstance(state, ResumeState):
        _fail("INVALID_RESUME_STATE", "a ResumeState is required", actual=type(state).__name__)
    if not isinstance(contract, CompatibilityContract):
        _fail("INVALID_COMPATIBILITY_CONTRACT", "a CompatibilityContract is required", actual=type(contract).__name__)
    digest = resume_state_content_hash(state)
    payload = {
        "schema_version": RESUME_STATE_SCHEMA_VERSION,
        "resume_state_hash": digest,
        "contract": canonical_data(replace(contract, resume_state_hash=digest)),
        "state": _state_payload(state),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    return digest


def load_resume_state(path: str | Path, *, expected_contract: CompatibilityContract) -> ResumeState:
    """Load a Resume_State only if the full compatibility gate passes."""
    source = Path(path)
    if not source.is_file():
        _fail("RESUME_STATE_NOT_FOUND", "no Resume_State exists at the requested path", path="path", actual=str(source))
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ResearchValidationError(
            "CORRUPT_RESUME_STATE", "Resume_State could not be read and cannot resume the original run",
            path="path", actual=str(source), details={"exception_type": type(exc).__name__, "message": str(exc)},
        ) from exc
    if not isinstance(payload, Mapping) or not {"resume_state_hash", "contract", "state"} <= set(payload):
        _fail("CORRUPT_RESUME_STATE", "Resume_State payload is not a complete checkpoint record", path="path", actual=str(source))
    saved_contract = CompatibilityContract(**dict(payload["contract"]))
    state = _state_from_payload(payload["state"])
    recomputed = resume_state_content_hash(state)
    if recomputed != payload["resume_state_hash"]:
        _fail(
            "RESUME_STATE_HASH_MISMATCH", "stored Resume_State hash does not match its content",
            path="resume_state_hash", expected=payload["resume_state_hash"], actual=recomputed,
        )
    check_resume_compatibility(saved_contract, expected_contract)
    return state


# ---------------------------------------------------------------------------
# Trainer integration
# ---------------------------------------------------------------------------


def capture_trainer_resume_state(
    trainer: ResearchTrainer,
    *,
    parent_run_id: str | None = None,
    scheduler_state: Mapping[str, Any] | None = None,
    scaler_state: Mapping[str, Any] | None = None,
    curriculum_state: Mapping[str, Any] | None = None,
    hysteresis_state: Mapping[str, Any] | None = None,
    running_normalization: Mapping[str, Any] | None = None,
    pending_transitions: Sequence[PendingDecision] | None = None,
    rollout_buffer: Mapping[str, Any] | None = None,
    evader_rng: np.random.Generator | None = None,
    placement_rng: np.random.Generator | None = None,
    sampler_rng: np.random.Generator | None = None,
) -> ResumeState:
    """Snapshot a live trainer.

    Every stream the trainer actually owns is read from it.  The keyword
    arguments exist for the components the trainer does not own yet (scheduler,
    scaler, curriculum, hysteresis, running normalization) and for the RNG
    streams the trainer derives per-episode rather than holding across rollouts
    (evader, placement, sampler); when a caller passes nothing they are stored
    as explicit ``None``.
    """
    if not isinstance(trainer, ResearchTrainer):
        _fail("INVALID_TRAINER", "a ResearchTrainer is required", actual=type(trainer).__name__)
    return ResumeState(
        policy_state_dict=trainer.policy.state_dict(),
        optimizer_state_dict=trainer.policy.optimizer.state_dict(),
        episode_index=trainer.episode_index,
        update_index=trainer.update_index,
        environment_step_index=trainer.environment_step_index,
        training_seed=trainer.training_seed,
        python_random_state=capture_python_random_state(),
        numpy_random_state=capture_numpy_random_state(),
        torch_cpu_rng_state=torch.get_rng_state(),
        training_generator_state=trainer._training_rng.get_state(),
        scheduler_state=scheduler_state,
        scaler_state=scaler_state,
        torch_device_rng_state=capture_device_rng_state(),
        evader_rng_state=capture_generator_state(evader_rng) if evader_rng is not None else None,
        placement_rng_state=capture_generator_state(placement_rng) if placement_rng is not None else None,
        sampler_rng_state=capture_generator_state(sampler_rng) if sampler_rng is not None else None,
        curriculum_state=curriculum_state,
        hysteresis_state=hysteresis_state,
        running_normalization=running_normalization,
        pending_transitions=(
            tuple(PendingTransitionSnapshot.from_pending(item) for item in pending_transitions)
            if pending_transitions is not None else None
        ),
        rollout_buffer=rollout_buffer,
        parent_run_id=parent_run_id,
    )


def restore_trainer_resume_state(trainer: ResearchTrainer, state: ResumeState) -> None:
    """Restore every stream a trainer owns, in place."""
    if not isinstance(trainer, ResearchTrainer):
        _fail("INVALID_TRAINER", "a ResearchTrainer is required", actual=type(trainer).__name__)
    if not isinstance(state, ResumeState):
        _fail("INVALID_RESUME_STATE", "a ResumeState is required", actual=type(state).__name__)
    if state.training_seed != trainer.training_seed:
        _fail(
            "RESUME_COMPATIBILITY_GATE_FAILED", "Resume_State training seed does not match the trainer",
            path="training_seed", expected=state.training_seed, actual=trainer.training_seed,
        )
    trainer.policy.load_state_dict({key: dict(value) for key, value in state.policy_state_dict.items()})
    trainer.policy.optimizer.load_state_dict(dict(state.optimizer_state_dict))
    restore_python_random_state(state.python_random_state)
    restore_numpy_random_state(state.numpy_random_state)
    torch.set_rng_state(state.torch_cpu_rng_state.clone())
    trainer._training_rng.set_state(state.training_generator_state.clone())
    restore_device_rng_state(state.torch_device_rng_state)
    trainer.set_progress(
        episode_index=state.episode_index,
        update_index=state.update_index,
        environment_step_index=state.environment_step_index,
    )


__all__ = (
    "OPTIONAL_RESUME_FIELDS",
    "REQUIRED_RESUME_FIELDS",
    "RESUME_STATE_SCHEMA_VERSION",
    "CompatibilityContract",
    "PendingTransitionSnapshot",
    "ResumeState",
    "capture_device_rng_state",
    "capture_generator_state",
    "capture_numpy_random_state",
    "capture_python_random_state",
    "capture_trainer_resume_state",
    "check_resume_compatibility",
    "load_resume_state",
    "policy_hidden_dims",
    "restore_device_rng_state",
    "restore_generator_state",
    "restore_numpy_random_state",
    "restore_python_random_state",
    "restore_trainer_resume_state",
    "resume_state_content_hash",
    "save_resume_state",
)

"""Fail-closed centralized MAPPO using exact rollout-stored action masks.

The actor only receives an officer-local observation and an officer identity
one-hot.  The critic has a separate, explicitly global input.  Action masks
are sealed before sampling and the same byte/hash identity is required during
PPO recomputation; this module has no environment-state mask API.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor, nn

from ..errors import ResearchValidationError

MASKED_MAPPO_SCHEMA_VERSION = "1.0"


def _fail(code: str, message: str, *, path: str | None = None, expected: Any = None, actual: Any = None) -> None:
    raise ResearchValidationError(
        code, message, path=path, expected=expected, actual=actual
    )


def _require_finite(name: str, value: Tensor) -> None:
    if not torch.isfinite(value).all().item():
        _fail("NONFINITE_VALUE", f"{name} contains a nonfinite value", path=name)


def _mask_digest(mask_bytes: bytes) -> str:
    return hashlib.sha256(mask_bytes).hexdigest()


@dataclass(frozen=True, slots=True)
class StoredActionMask:
    """Exact one-byte-per-action mask representation sealed at rollout time."""

    mask_bytes: bytes
    mask_hash: str
    action_dim: int
    schema_version: str = MASKED_MAPPO_SCHEMA_VERSION
    def __post_init__(self) -> None:
        if self.action_dim <= 0:
            _fail("INVALID_ACTION_DIM", "action_dim must be positive", path="action_dim", actual=self.action_dim)
        if not isinstance(self.mask_bytes, bytes):
            _fail("MALFORMED_MASK_BYTES", "mask_bytes must be immutable bytes", path="mask_bytes")
        if len(self.mask_bytes) != self.action_dim:
            _fail(
                "MALFORMED_MASK_BYTES",
                "stored mask byte length must equal action_dim",
                path="mask_bytes",
                expected=self.action_dim,
                actual=len(self.mask_bytes),
            )
        if any(item not in (0, 1) for item in self.mask_bytes):
            _fail("MALFORMED_MASK_BYTES", "stored mask bytes must contain only 0 or 1", path="mask_bytes")
        if not any(self.mask_bytes):
            _fail("EMPTY_ACTION_SUPPORT", "stored action mask has empty support", path="mask_bytes")
        actual_hash = _mask_digest(self.mask_bytes)
        if self.mask_hash != actual_hash:
            _fail(
                "STORED_MASK_HASH_DRIFT",
                "stored mask hash does not match its exact bytes",
                path="mask_hash",
                expected=actual_hash,
                actual=self.mask_hash,
            )

    @classmethod
    def seal(cls, mask: Tensor | Sequence[bool], *, action_dim: int | None = None) -> "StoredActionMask":
        tensor = torch.as_tensor(mask)
        if tensor.ndim != 1:
            _fail("MALFORMED_ACTION_MASK", "action mask must be rank one", path="mask", actual=tuple(tensor.shape))
        if tensor.dtype is not torch.bool:
            _fail("MALFORMED_ACTION_MASK", "action mask must have bool dtype", path="mask", actual=str(tensor.dtype))
        dimension = int(tensor.numel())
        if action_dim is not None and dimension != action_dim:
            _fail("MALFORMED_ACTION_MASK", "action mask width mismatch", expected=action_dim, actual=dimension)
        raw = bytes(int(item) for item in tensor.detach().cpu().tolist())
        return cls(mask_bytes=raw, mask_hash=_mask_digest(raw), action_dim=dimension)

    def as_tensor(self, *, device: torch.device | str | None = None) -> Tensor:
        # Decode only sealed bytes; callers cannot substitute environment state.
        return torch.tensor(list(self.mask_bytes), dtype=torch.bool, device=device)


class MaskedCategorical:
    """Categorical distribution with explicit, auditable support semantics."""

    def __init__(self, logits: Tensor, masks: Tensor, mask_records: tuple[StoredActionMask, ...]) -> None:
        self.logits = logits.masked_fill(~masks, -torch.inf)
        self.masks = masks
        self.mask_records = mask_records
        self._probabilities = torch.softmax(self.logits, dim=-1)
        _require_finite("masked_probabilities", self._probabilities)
        # This explicit assignment guarantees an exact floating-point zero.
        self._probabilities = torch.where(masks, self._probabilities, torch.zeros_like(self._probabilities))

    @property
    def probs(self) -> Tensor:
        return self._probabilities

    @property
    def support_bytes(self) -> tuple[bytes, ...]:
        return tuple(record.mask_bytes for record in self.mask_records)

    @property
    def support_hashes(self) -> tuple[str, ...]:
        return tuple(record.mask_hash for record in self.mask_records)

    def sample(self, *, generator: torch.Generator | None = None) -> Tensor:
        return torch.multinomial(self.probs, 1, replacement=True, generator=generator).squeeze(-1)

    def log_prob(self, actions: Tensor) -> Tensor:
        if actions.ndim != 1 or actions.shape[0] != self.logits.shape[0]:
            _fail("MALFORMED_ACTIONS", "actions must have shape [batch]", actual=tuple(actions.shape))
        if actions.dtype != torch.long:
            _fail("MALFORMED_ACTIONS", "actions must have torch.long dtype", actual=str(actions.dtype))
        if ((actions < 0) | (actions >= self.logits.shape[-1])).any().item():
            _fail("ACTION_OUT_OF_RANGE", "selected action is outside the action space")
        selected_valid = self.masks.gather(1, actions.unsqueeze(1)).squeeze(1)
        if not selected_valid.all().item():
            invalid_rows = (~selected_valid).nonzero(as_tuple=False).flatten().cpu().tolist()
            _fail("SELECTED_INVALID_ACTION", "selected action is outside stored mask support", details={"rows": invalid_rows})
        result = torch.log(self.probs.gather(1, actions.unsqueeze(1)).squeeze(1))
        _require_finite("log_prob", result)
        return result

    def entropy(self) -> Tensor:
        safe_log_probs = torch.where(self.masks, torch.log(self.probs), torch.zeros_like(self.probs))
        entropy = -(self.probs * safe_log_probs).sum(dim=-1)
        _require_finite("valid_action_entropy", entropy)
        return entropy


class MaskedCategoricalFactory:
    """Build distributions exclusively from validated stored-mask records."""

    def __init__(self, action_dim: int) -> None:
        if action_dim <= 0:
            _fail("INVALID_ACTION_DIM", "action_dim must be positive", actual=action_dim)
        self.action_dim = action_dim

    def seal(self, mask: Tensor | Sequence[bool]) -> StoredActionMask:
        return StoredActionMask.seal(mask, action_dim=self.action_dim)

    def create(
        self,
        logits: Tensor,
        stored_masks: StoredActionMask | Sequence[StoredActionMask],
    ) -> MaskedCategorical:
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        if logits.ndim != 2 or logits.shape[1] != self.action_dim:
            _fail(
                "MALFORMED_ACTOR_OUTPUT",
                "actor logits must have shape [batch, action_dim]",
                expected=("batch", self.action_dim),
                actual=tuple(logits.shape),
            )
        _require_finite("actor_logits", logits)
        records = (stored_masks,) if isinstance(stored_masks, StoredActionMask) else tuple(stored_masks)
        if len(records) != logits.shape[0]:
            _fail("MALFORMED_MASK_BATCH", "one stored mask is required per logit row", expected=logits.shape[0], actual=len(records))
        for index, record in enumerate(records):
            if not isinstance(record, StoredActionMask):
                _fail("UNSEALED_ACTION_MASK", "distribution requires StoredActionMask records", path=f"stored_masks[{index}]")
            # Reconstructing catches mutation/bypass even if a malformed instance was forged.
            StoredActionMask(record.mask_bytes, record.mask_hash, record.action_dim)
            if record.action_dim != self.action_dim:
                _fail("MALFORMED_ACTION_MASK", "stored mask action_dim mismatch", expected=self.action_dim, actual=record.action_dim)
        masks = torch.stack([record.as_tensor(device=logits.device) for record in records])
        return MaskedCategorical(logits, masks, records)

    __call__ = create
    from_logits = create
    from_stored_masks = create


class SharedOfficerActor(nn.Module):
    """Parameter-shared actor over local observation plus officer identity."""

    def __init__(self, actor_obs_dim: int, action_dim: int, num_officers: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        self.actor_obs_dim = actor_obs_dim
        self.action_dim = action_dim
        self.num_officers = num_officers
        self.network = _mlp(actor_obs_dim + num_officers, action_dim, hidden_dims)

    def forward(self, actor_obs: Tensor, officer_ids: Tensor) -> Tensor:
        if actor_obs.ndim != 2 or actor_obs.shape[1] != self.actor_obs_dim:
            _fail("MALFORMED_ACTOR_OBSERVATION", "actor_obs must have shape [batch, actor_obs_dim]", expected=("batch", self.actor_obs_dim), actual=tuple(actor_obs.shape))
        if officer_ids.ndim != 1 or officer_ids.shape[0] != actor_obs.shape[0] or officer_ids.dtype != torch.long:
            _fail("MALFORMED_OFFICER_IDS", "officer_ids must be torch.long with shape [batch]")
        if ((officer_ids < 0) | (officer_ids >= self.num_officers)).any().item():
            _fail("MALFORMED_OFFICER_IDS", "officer id is outside configured range")
        _require_finite("actor_obs", actor_obs)
        identity = torch.nn.functional.one_hot(officer_ids, num_classes=self.num_officers).to(dtype=actor_obs.dtype)
        return self.network(torch.cat((actor_obs, identity), dim=-1))


class CentralizedCritic(nn.Module):
    """Value network whose only input is the declared joint/global context."""

    def __init__(self, critic_context_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        self.critic_context_dim = critic_context_dim
        self.network = _mlp(critic_context_dim, 1, hidden_dims)

    def forward(self, critic_context: Tensor) -> Tensor:
        if critic_context.ndim != 2 or critic_context.shape[1] != self.critic_context_dim:
            _fail("MALFORMED_CENTRALIZED_CONTEXT", "critic_context must have exact shape [batch, critic_context_dim]", expected=("batch", self.critic_context_dim), actual=tuple(critic_context.shape))
        _require_finite("critic_context", critic_context)
        values = self.network(critic_context).squeeze(-1)
        _require_finite("critic_values", values)
        return values


def _mlp(input_dim: int, output_dim: int, hidden_dims: Sequence[int]) -> nn.Sequential:
    if input_dim <= 0 or output_dim <= 0 or not hidden_dims or any(width <= 0 for width in hidden_dims):
        _fail("INVALID_NETWORK_SHAPE", "network dimensions must be positive and hidden_dims nonempty")
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_dims:
        layers.extend((nn.Linear(previous, width), nn.Tanh()))
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


@dataclass(frozen=True, slots=True)
class ActionSample:
    action: int
    log_prob: float
    probabilities: Tensor
    stored_mask_bytes: bytes
    stored_mask_hash: str
    support_bytes: bytes
    support_hash: str


@dataclass(slots=True)
class MaskedMAPPOBatch:
    """Decision batch carrying both rollout identity and update-time storage.

    ``sampling_mask_*`` is copied from :class:`ActionSample`; ``stored_masks``
    is the transition payload.  Requiring both makes coordinated replacement
    of transition bytes/hash visible as old/new support drift.
    """

    actor_obs: Tensor
    officer_ids: Tensor
    actions: Tensor
    old_log_probs: Tensor
    advantages: Tensor
    returns: Tensor
    critic_context: Tensor
    stored_masks: tuple[StoredActionMask, ...]
    sampling_mask_bytes: tuple[bytes, ...]
    sampling_mask_hashes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ActionEvaluation:
    new_log_probs: Tensor
    entropy: Tensor
    values: Tensor
    probabilities: Tensor


# Design-level name retained as a concise public alias.
DecisionBatch = MaskedMAPPOBatch


@dataclass(frozen=True, slots=True)
class PPOLoss:
    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    ratio_mean: float
    update_applied: bool = True


def _all_tensors_finite(value: Any) -> bool:
    if isinstance(value, Tensor):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(_all_tensors_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_all_tensors_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


class ResearchMaskedMAPPO:
    """Exact stored-mask MAPPO with one shared actor and centralized critic."""

    def __init__(
        self,
        actor_obs_dim: int,
        critic_context_dim: int,
        action_dim: int,
        num_officers: int = 6,
        *,
        hidden_dims: Sequence[int] = (64, 64),
        learning_rate: float = 3e-4,
        clip_epsilon: float = 0.2,
        value_coefficient: float = 0.5,
        entropy_coefficient: float = 0.01,
        max_grad_norm: float = 0.5,
        device: torch.device | str = "cpu",
    ) -> None:
        numeric = {
            "learning_rate": learning_rate,
            "clip_epsilon": clip_epsilon,
            "value_coefficient": value_coefficient,
            "entropy_coefficient": entropy_coefficient,
            "max_grad_norm": max_grad_norm,
        }
        if actor_obs_dim <= 0 or critic_context_dim <= 0 or action_dim <= 0 or num_officers <= 0:
            _fail("INVALID_MAPPO_CONFIGURATION", "all dimensions and num_officers must be positive")
        if any(not math.isfinite(value) for value in numeric.values()):
            _fail("NONFINITE_VALUE", "MAPPO hyperparameters must be finite")
        if learning_rate <= 0 or not 0 < clip_epsilon < 1 or value_coefficient < 0 or entropy_coefficient < 0 or max_grad_norm <= 0:
            _fail("INVALID_MAPPO_CONFIGURATION", "MAPPO hyperparameters are outside their valid ranges")
        self.actor_obs_dim = actor_obs_dim
        self.critic_context_dim = critic_context_dim
        self.action_dim = action_dim
        self.num_officers = num_officers
        self.clip_epsilon = clip_epsilon
        self.value_coefficient = value_coefficient
        self.entropy_coefficient = entropy_coefficient
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)
        self.actor = SharedOfficerActor(actor_obs_dim, action_dim, num_officers, hidden_dims).to(self.device)
        self.critic = CentralizedCritic(critic_context_dim, hidden_dims).to(self.device)
        self.masked_categorical = MaskedCategoricalFactory(action_dim)
        self.optimizer = torch.optim.Adam((*self.actor.parameters(), *self.critic.parameters()), lr=learning_rate)

    def seal_mask(self, mask: Tensor | Sequence[bool]) -> StoredActionMask:
        """Seal an environment-produced bool mask before any policy sampling."""
        return self.masked_categorical.seal(mask)

    def actor_logits(self, actor_obs: Tensor, officer_ids: Tensor) -> Tensor:
        actor_obs = actor_obs.to(device=self.device, dtype=next(self.actor.parameters()).dtype)
        officer_ids = officer_ids.to(device=self.device)
        logits = self.actor(actor_obs, officer_ids)
        _require_finite("actor_logits", logits)
        return logits

    def critic_values(self, critic_context: Tensor) -> Tensor:
        context = critic_context.to(device=self.device, dtype=next(self.critic.parameters()).dtype)
        return self.critic(context)

    def sample(
        self,
        actor_obs: Tensor,
        stored_mask: StoredActionMask,
        rng: torch.Generator | None = None,
        *,
        officer_id: int = 0,
        logit_bias: Tensor | None = None,
    ) -> ActionSample:
        """Sample from one already-sealed rollout mask (never from environment state).

        ``logit_bias`` is an optional ``[action_dim]`` tensor added to the raw
        actor logits before masking -- e.g. a stabilization Condition's soft
        u-turn penalty (Requirement 10.5).  It is applied *before* masking, so
        it can never make a masked-illegal action selectable; omitting it (the
        default) reproduces this method's prior behavior exactly.
        """
        if not isinstance(stored_mask, StoredActionMask):
            _fail("UNSEALED_ACTION_MASK", "sample requires a StoredActionMask")
        StoredActionMask(stored_mask.mask_bytes, stored_mask.mask_hash, stored_mask.action_dim)
        observation = torch.as_tensor(actor_obs)
        if observation.ndim != 1 or observation.shape[0] != self.actor_obs_dim:
            _fail("MALFORMED_ACTOR_OBSERVATION", "sample actor_obs must have shape [actor_obs_dim]", expected=(self.actor_obs_dim,), actual=tuple(observation.shape))
        _require_finite("actor_obs", observation)
        if not isinstance(officer_id, int) or not 0 <= officer_id < self.num_officers:
            _fail("MALFORMED_OFFICER_IDS", "officer_id is outside configured range", actual=officer_id)
        if logit_bias is not None:
            bias = torch.as_tensor(logit_bias)
            if bias.ndim != 1 or bias.shape[0] != self.action_dim:
                _fail("MALFORMED_LOGIT_BIAS", "logit_bias must have shape [action_dim]", expected=(self.action_dim,), actual=tuple(bias.shape))
            _require_finite("logit_bias", bias)
        with torch.no_grad():
            logits = self.actor_logits(observation.unsqueeze(0), torch.tensor([officer_id], dtype=torch.long, device=self.device))
            if logit_bias is not None:
                logits = logits + bias.to(device=logits.device, dtype=logits.dtype).unsqueeze(0)
            distribution = self.masked_categorical.create(logits, stored_mask)
            action_tensor = distribution.sample(generator=rng)
            log_prob = distribution.log_prob(action_tensor)
        return ActionSample(
            action=int(action_tensor.item()),
            log_prob=float(log_prob.item()),
            probabilities=distribution.probs[0].detach().clone(),
            stored_mask_bytes=stored_mask.mask_bytes,
            stored_mask_hash=stored_mask.mask_hash,
            support_bytes=distribution.support_bytes[0],
            support_hash=distribution.support_hashes[0],
        )

    def _validated_batch(self, batch: MaskedMAPPOBatch) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if not isinstance(batch, MaskedMAPPOBatch):
            _fail("MALFORMED_MAPPO_BATCH", "update requires MaskedMAPPOBatch")
        tensors = {
            "actor_obs": batch.actor_obs,
            "officer_ids": batch.officer_ids,
            "actions": batch.actions,
            "old_log_probs": batch.old_log_probs,
            "advantages": batch.advantages,
            "returns": batch.returns,
            "critic_context": batch.critic_context,
        }
        if any(not isinstance(value, Tensor) for value in tensors.values()):
            _fail("MALFORMED_MAPPO_BATCH", "all batch numeric fields must be tensors")
        if batch.actor_obs.ndim != 2:
            _fail("MALFORMED_ACTOR_OBSERVATION", "batch actor_obs must be rank two")
        size = batch.actor_obs.shape[0]
        if size <= 0:
            _fail("EMPTY_MAPPO_BATCH", "MAPPO update batch must be nonempty")
        expected_vectors = (batch.officer_ids, batch.actions, batch.old_log_probs, batch.advantages, batch.returns)
        if any(item.ndim != 1 or item.shape[0] != size for item in expected_vectors):
            _fail("MALFORMED_MAPPO_BATCH", "per-decision tensors must all have shape [batch]")
        if batch.critic_context.ndim != 2 or batch.critic_context.shape != (size, self.critic_context_dim):
            _fail("MALFORMED_CENTRALIZED_CONTEXT", "critic_context must have exact shape [batch, critic_context_dim]", expected=(size, self.critic_context_dim), actual=tuple(batch.critic_context.shape))
        if batch.actor_obs.shape[1] != self.actor_obs_dim:
            _fail("MALFORMED_ACTOR_OBSERVATION", "actor_obs width mismatch", expected=self.actor_obs_dim, actual=batch.actor_obs.shape[1])
        if batch.officer_ids.dtype != torch.long or batch.actions.dtype != torch.long:
            _fail("MALFORMED_MAPPO_BATCH", "officer_ids and actions must use torch.long dtype")
        for name in ("actor_obs", "old_log_probs", "advantages", "returns", "critic_context"):
            _require_finite(name, tensors[name])
        counts = (len(batch.stored_masks), len(batch.sampling_mask_bytes), len(batch.sampling_mask_hashes))
        if counts != (size, size, size):
            _fail("MALFORMED_MASK_BATCH", "mask identity arrays must contain one item per decision", expected=(size, size, size), actual=counts)
        for index, (stored, sampled_bytes, sampled_hash) in enumerate(zip(batch.stored_masks, batch.sampling_mask_bytes, batch.sampling_mask_hashes)):
            if not isinstance(stored, StoredActionMask):
                _fail("UNSEALED_ACTION_MASK", "batch stored masks must be sealed", path=f"stored_masks[{index}]")
            StoredActionMask(stored.mask_bytes, stored.mask_hash, stored.action_dim)
            if not isinstance(sampled_bytes, bytes) or _mask_digest(sampled_bytes) != sampled_hash:
                _fail("SAMPLING_MASK_IDENTITY_DRIFT", "sampling mask hash/bytes identity is invalid", path=f"sampling_mask_bytes[{index}]")
            if stored.mask_bytes != sampled_bytes:
                _fail("STORED_MASK_BYTE_DRIFT", "transition mask bytes differ from sampling bytes", path=f"stored_masks[{index}]")
            if stored.mask_hash != sampled_hash:
                _fail("STORED_MASK_HASH_DRIFT", "transition mask hash differs from sampling hash", path=f"stored_masks[{index}]")
            action = int(batch.actions[index].item())
            if action < 0 or action >= self.action_dim:
                _fail("ACTION_OUT_OF_RANGE", "selected action is outside action space", path=f"actions[{index}]", actual=action)
            if stored.mask_bytes[action] != 1:
                _fail("SELECTED_INVALID_ACTION", "selected action is invalid under exact stored mask", path=f"actions[{index}]", actual=action)
        float_dtype = next(self.actor.parameters()).dtype
        return (
            batch.actor_obs.to(device=self.device, dtype=float_dtype),
            batch.officer_ids.to(device=self.device),
            batch.actions.to(device=self.device),
            batch.old_log_probs.to(device=self.device, dtype=float_dtype),
            batch.advantages.to(device=self.device, dtype=float_dtype),
            batch.returns.to(device=self.device, dtype=float_dtype),
            batch.critic_context.to(device=self.device, dtype=float_dtype),
        )


    def _forward_batch(self, batch: MaskedMAPPOBatch) -> tuple[ActionEvaluation, tuple[Tensor, ...]]:
        prepared = self._validated_batch(batch)
        actor_obs, officer_ids, actions, _, _, _, critic_context = prepared
        logits = self.actor(actor_obs, officer_ids)
        _require_finite("actor_logits", logits)
        distribution = self.masked_categorical.create(logits, batch.stored_masks)
        if distribution.support_bytes != batch.sampling_mask_bytes or distribution.support_hashes != batch.sampling_mask_hashes:
            _fail("OLD_NEW_SUPPORT_DRIFT", "PPO recomputation support differs from rollout sampling support")
        new_log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.critic(critic_context)
        evaluation = ActionEvaluation(
            new_log_probs=new_log_probs,
            entropy=entropy,
            values=values,
            probabilities=distribution.probs,
        )
        return evaluation, prepared

    def evaluate_actions(self, batch: MaskedMAPPOBatch) -> ActionEvaluation:
        """Recompute policy/value terms without mutating parameters or optimizer."""
        evaluation, _ = self._forward_batch(batch)
        return evaluation

    def recompute_log_probs(self, batch: MaskedMAPPOBatch) -> Tensor:
        """Convenience audit API for unchanged-parameter identity checks."""
        return self.evaluate_actions(batch).new_log_probs

    def update(self, batch: MaskedMAPPOBatch) -> PPOLoss:
        """Apply one fail-closed PPO update after all support/input validation.

        Parameters and optimizer state are snapshotted before mutation and are
        restored if gradient, optimizer-state, parameter, or loss validation
        fails.  Thus a rejected batch cannot leave a partial optimizer update.
        """
        if not _all_tensors_finite(self.optimizer.state):
            _fail("NONFINITE_OPTIMIZER_STATE", "optimizer state contains a nonfinite value")
        evaluation, prepared = self._forward_batch(batch)
        _, _, _, old_log_probs, advantages, returns, _ = prepared
        for name, tensor in (
            ("old_log_probs", old_log_probs),
            ("advantages", advantages),
            ("returns", returns),
            ("new_log_probs", evaluation.new_log_probs),
            ("entropy", evaluation.entropy),
            ("values", evaluation.values),
        ):
            _require_finite(name, tensor)

        log_ratio = evaluation.new_log_probs - old_log_probs
        _require_finite("log_ratio", log_ratio)
        ratio = torch.exp(log_ratio)
        _require_finite("ppo_ratio", ratio)
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
        policy_loss = -torch.minimum(ratio * advantages, clipped_ratio * advantages).mean()
        value_loss = torch.nn.functional.mse_loss(evaluation.values, returns)
        entropy = evaluation.entropy.mean()
        total_loss = policy_loss + self.value_coefficient * value_loss - self.entropy_coefficient * entropy
        for name, tensor in (
            ("policy_loss", policy_loss),
            ("value_loss", value_loss),
            ("entropy", entropy),
            ("total_loss", total_loss),
        ):
            _require_finite(name, tensor.reshape(1))

        model_snapshot = deepcopy(self.state_dict())
        optimizer_snapshot = deepcopy(self.optimizer.state_dict())
        self.optimizer.zero_grad(set_to_none=True)
        try:
            total_loss.backward()
            parameters = tuple(self.actor.parameters()) + tuple(self.critic.parameters())
            gradients = tuple(parameter.grad for parameter in parameters if parameter.grad is not None)
            if not gradients or any(not torch.isfinite(gradient).all().item() for gradient in gradients):
                _fail("NONFINITE_GRADIENT", "PPO backward pass produced missing or nonfinite gradients")
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
            if not torch.isfinite(torch.as_tensor(gradient_norm)).item():
                _fail("NONFINITE_GRADIENT", "gradient norm is nonfinite")
            self.optimizer.step()
            if any(not torch.isfinite(parameter).all().item() for parameter in parameters):
                _fail("NONFINITE_PARAMETER", "optimizer produced a nonfinite model parameter")
            if not _all_tensors_finite(self.optimizer.state):
                _fail("NONFINITE_OPTIMIZER_STATE", "optimizer produced nonfinite state")
        except Exception as exc:
            self.load_state_dict(model_snapshot)
            self.optimizer.load_state_dict(optimizer_snapshot)
            self.optimizer.zero_grad(set_to_none=True)
            if isinstance(exc, ResearchValidationError):
                raise
            raise ResearchValidationError(
                "OPTIMIZER_UPDATE_FAILED",
                "optimizer update failed and was rolled back",
                details={"exception_type": type(exc).__name__, "message": str(exc)},
            ) from exc
        self.optimizer.zero_grad(set_to_none=True)
        return PPOLoss(
            policy_loss=float(policy_loss.detach().item()),
            value_loss=float(value_loss.detach().item()),
            entropy=float(entropy.detach().item()),
            total_loss=float(total_loss.detach().item()),
            ratio_mean=float(ratio.detach().mean().item()),
        )

    def state_dict(self) -> dict[str, dict[str, Tensor]]:
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}

    def load_state_dict(self, state: dict[str, dict[str, Tensor]]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])


__all__ = (
    "MASKED_MAPPO_SCHEMA_VERSION",
    "ActionEvaluation",
    "ActionSample",
    "CentralizedCritic",
    "DecisionBatch",
    "MaskedCategorical",
    "MaskedCategoricalFactory",
    "MaskedMAPPOBatch",
    "PPOLoss",
    "ResearchMaskedMAPPO",
    "SharedOfficerActor",
    "StoredActionMask",
)

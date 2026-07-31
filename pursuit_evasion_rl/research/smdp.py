"""Fail-closed hybrid SMDP transitions for six asynchronous officers.

A decision remains pending while physical simulation steps elapse.  Rewards are
accumulated with decision-relative discounting; zero-time virtual routing never
increments the duration.  Road helpers validate the movement contract used by
``OSMRoadPursuitEnv`` without coupling the research trainer to that environment.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import math
from typing import Iterable, Mapping, Sequence

from pursuit_evasion_rl.osm_demo.models import ModelNetwork, Segment, VehiclePlacement

from .errors import ResearchValidationError

OFFICER_COUNT = 6


def _finite(value: float, path: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ResearchValidationError(
            "NON_FINITE_SMDP_VALUE", f"{path} must be finite", path=path, actual=value
        ) from exc
    if not math.isfinite(result):
        raise ResearchValidationError(
            "NON_FINITE_SMDP_VALUE", f"{path} must be finite", path=path, actual=value
        )
    return result


def _vector(values: Iterable[float], path: str) -> tuple[float, ...]:
    try:
        result = tuple(_finite(value, f"{path}[{index}]") for index, value in enumerate(values))
    except TypeError as exc:
        raise ResearchValidationError(
            "INVALID_SMDP_VECTOR", f"{path} must be a one-dimensional iterable", path=path
        ) from exc
    if not result:
        raise ResearchValidationError(
            "EMPTY_SMDP_VECTOR", f"{path} must not be empty", path=path
        )
    return result


def encode_action_mask(mask: Sequence[bool] | bytes) -> bytes:
    """Encode an exact, non-empty boolean support as one byte per action."""
    raw = bytes(mask)
    if not raw or any(value not in (0, 1) for value in raw):
        raise ResearchValidationError(
            "INVALID_ACTION_MASK_BYTES",
            "action mask must contain only 0/1 bytes and have non-empty support",
            path="action_mask",
            actual=list(raw),
        )
    if not any(raw):
        raise ResearchValidationError(
            "EMPTY_ACTION_SUPPORT", "action mask must permit at least one action", path="action_mask"
        )
    return raw

@dataclass(frozen=True, slots=True)
class DecisionEpoch:
    """Validated state/action snapshot taken immediately before an action."""

    actor_obs: tuple[float, ...]
    action: int
    stored_mask_bytes: bytes
    old_log_prob: float
    critic_context: tuple[float, ...]
    stored_mask_hash: str = field(init=False)

    def __post_init__(self) -> None:
        actor_obs = _vector(self.actor_obs, "actor_obs")
        critic_context = _vector(self.critic_context, "critic_context")
        mask = encode_action_mask(self.stored_mask_bytes)
        if isinstance(self.action, bool) or not isinstance(self.action, int):
            raise ResearchValidationError(
                "INVALID_ACTION", "action must be an integer", path="action", actual=self.action
            )
        if self.action < 0 or self.action >= len(mask) or not mask[self.action]:
            raise ResearchValidationError(
                "SELECTED_INVALID_ACTION",
                "selected action is outside the stored mask support",
                path="action",
                expected=[index for index, valid in enumerate(mask) if valid],
                actual=self.action,
            )
        object.__setattr__(self, "actor_obs", actor_obs)
        object.__setattr__(self, "critic_context", critic_context)
        object.__setattr__(self, "stored_mask_bytes", mask)
        object.__setattr__(self, "old_log_prob", _finite(self.old_log_prob, "old_log_prob"))
        object.__setattr__(self, "stored_mask_hash", hashlib.sha256(mask).hexdigest())

    @classmethod
    def create(
        cls,
        *,
        actor_obs: Iterable[float],
        action: int,
        action_mask: Sequence[bool] | bytes,
        old_log_prob: float,
        critic_context: Iterable[float],
    ) -> "DecisionEpoch":
        return cls(tuple(actor_obs), action, encode_action_mask(action_mask), old_log_prob, tuple(critic_context))


@dataclass(frozen=True, slots=True)
class DecisionTransition:
    """Immutable transition spanning one asynchronous decision interval."""

    officer_id: int
    actor_obs: tuple[float, ...]
    action: int
    stored_mask_bytes: bytes
    stored_mask_hash: str
    old_log_prob: float
    discounted_reward: float
    duration_steps: int
    gamma: float
    critic_context: tuple[float, ...]
    next_critic_context: tuple[float, ...]
    terminal: bool

    def __post_init__(self) -> None:
        if not 0 <= self.officer_id < OFFICER_COUNT:
            raise ResearchValidationError("INVALID_OFFICER_ID", "officer_id must be in [0, 5]")
        if isinstance(self.duration_steps, bool) or self.duration_steps <= 0:
            raise ResearchValidationError(
                "ZERO_DURATION_TRANSITION",
                "a decision transition must include at least one physical step",
                path="duration_steps",
                actual=self.duration_steps,
            )
        mask = encode_action_mask(self.stored_mask_bytes)
        if hashlib.sha256(mask).hexdigest() != self.stored_mask_hash:
            raise ResearchValidationError("MASK_HASH_DRIFT", "stored mask hash does not match mask bytes")
        if self.action < 0 or self.action >= len(mask) or not mask[self.action]:
            raise ResearchValidationError("SELECTED_INVALID_ACTION", "transition action is not supported")
        gamma = _finite(self.gamma, "gamma")
        if not 0.0 <= gamma <= 1.0:
            raise ResearchValidationError("INVALID_GAMMA", "gamma must be in [0, 1]", actual=gamma)
        object.__setattr__(self, "actor_obs", _vector(self.actor_obs, "actor_obs"))
        object.__setattr__(self, "critic_context", _vector(self.critic_context, "critic_context"))
        object.__setattr__(self, "next_critic_context", _vector(self.next_critic_context, "next_critic_context"))
        object.__setattr__(self, "old_log_prob", _finite(self.old_log_prob, "old_log_prob"))
        object.__setattr__(self, "discounted_reward", _finite(self.discounted_reward, "discounted_reward"))
        object.__setattr__(self, "gamma", gamma)

    @property
    def bootstrap_discount(self) -> float:
        """The only valid non-terminal bootstrap multiplier, ``gamma ** tau``."""
        return self.gamma ** self.duration_steps

    def bootstrap_target(self, next_value: float) -> float:
        value = _finite(next_value, "next_value")
        return self.discounted_reward if self.terminal else self.discounted_reward + self.bootstrap_discount * value


@dataclass(frozen=True, slots=True)
class PendingDecision:
    officer_id: int
    epoch: DecisionEpoch
    discounted_reward: float = 0.0
    duration_steps: int = 0

class AsyncDecisionTransitionBuffer:
    """Per-officer pending buffers with atomic asynchronous closure."""

    def __init__(
        self,
        *,
        gamma: float,
        max_virtual_hops: int,
        actor_obs_dim: int | None = None,
        critic_context_dim: int | None = None,
        action_count: int | None = None,
    ) -> None:
        self.gamma = _finite(gamma, "gamma")
        if not 0.0 <= self.gamma <= 1.0:
            raise ResearchValidationError("INVALID_GAMMA", "gamma must be in [0, 1]")
        if isinstance(max_virtual_hops, bool) or not isinstance(max_virtual_hops, int) or max_virtual_hops < 0:
            raise ResearchValidationError("INVALID_VIRTUAL_HOP_BOUND", "max_virtual_hops must be nonnegative")
        for name, value in (("actor_obs_dim", actor_obs_dim), ("critic_context_dim", critic_context_dim), ("action_count", action_count)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ResearchValidationError("INVALID_DIMENSION", f"{name} must be a positive integer")
        self.max_virtual_hops = max_virtual_hops
        self._actor_obs_dim = actor_obs_dim
        self._critic_context_dim = critic_context_dim
        self._action_count = action_count
        self._pending: list[PendingDecision | None] = [None] * OFFICER_COUNT
        self._completed: list[DecisionTransition] = []

    @property
    def pending(self) -> tuple[PendingDecision | None, ...]:
        return tuple(self._pending)

    @property
    def completed(self) -> tuple[DecisionTransition, ...]:
        return tuple(self._completed)

    def _validate_officer(self, officer_id: int) -> None:
        if isinstance(officer_id, bool) or not isinstance(officer_id, int) or not 0 <= officer_id < OFFICER_COUNT:
            raise ResearchValidationError(
                "INVALID_OFFICER_ID", "officer id must be an integer in [0, 5]", actual=officer_id
            )

    def _validate_epoch(self, epoch: DecisionEpoch) -> None:
        if not isinstance(epoch, DecisionEpoch):
            raise ResearchValidationError("INVALID_DECISION_EPOCH", "value must be a DecisionEpoch")
        dimensions = (
            ("actor_obs", len(epoch.actor_obs), self._actor_obs_dim),
            ("critic_context", len(epoch.critic_context), self._critic_context_dim),
            ("action_mask", len(epoch.stored_mask_bytes), self._action_count),
        )
        for name, actual, expected in dimensions:
            if expected is not None and actual != expected:
                raise ResearchValidationError(
                    "SMDP_DIMENSION_MISMATCH", f"{name} dimension changed", path=name,
                    expected=expected, actual=actual,
                )

    def _lock_dimensions(self, epoch: DecisionEpoch) -> None:
        if self._actor_obs_dim is None:
            self._actor_obs_dim = len(epoch.actor_obs)
        if self._critic_context_dim is None:
            self._critic_context_dim = len(epoch.critic_context)
        if self._action_count is None:
            self._action_count = len(epoch.stored_mask_bytes)

    def start_decisions(self, decisions: Mapping[int, DecisionEpoch]) -> tuple[DecisionTransition, ...]:
        """Close and replace only officers deciding at this epoch.

        The complete batch is validated before mutation so a malformed officer,
        dimension, mask, or zero-duration closure cannot partially update state.
        """
        if not decisions:
            return ()
        for officer_id, epoch in decisions.items():
            self._validate_officer(officer_id)
            self._validate_epoch(epoch)
        first = next(iter(decisions.values()))
        prospective = (len(first.actor_obs), len(first.critic_context), len(first.stored_mask_bytes))
        expected = (self._actor_obs_dim, self._critic_context_dim, self._action_count)
        for epoch in decisions.values():
            actual = (len(epoch.actor_obs), len(epoch.critic_context), len(epoch.stored_mask_bytes))
            target = tuple(p if e is None else e for p, e in zip(prospective, expected))
            if actual != target:
                raise ResearchValidationError("SMDP_DIMENSION_MISMATCH", "decision batch dimensions differ")
        closures: list[DecisionTransition] = []
        for officer_id, epoch in decisions.items():
            pending = self._pending[officer_id]
            if pending is not None:
                closures.append(self._close(pending, epoch.critic_context, terminal=False))
        self._lock_dimensions(first)
        for officer_id, epoch in decisions.items():
            self._pending[officer_id] = PendingDecision(officer_id, epoch)
        self._completed.extend(closures)
        return tuple(closures)

    def start_decision(self, officer_id: int, epoch: DecisionEpoch) -> DecisionTransition | None:
        closed = self.start_decisions({officer_id: epoch})
        return closed[0] if closed else None

    def record_physical_step(
        self,
        rewards: Sequence[float],
        *,
        virtual_hops: Mapping[int, int] | Sequence[int] | None = None,
    ) -> None:
        """Accumulate one physical step; virtual hops are checked but cost zero time."""
        if len(rewards) != OFFICER_COUNT:
            raise ResearchValidationError(
                "INVALID_REWARD_DIMENSION", "exactly six officer rewards are required",
                expected=OFFICER_COUNT, actual=len(rewards),
            )
        if any(item is None for item in self._pending):
            raise ResearchValidationError(
                "MISSING_PENDING_TRANSITION",
                "all six officers require pending decisions before a physical step",
                actual=[index for index, item in enumerate(self._pending) if item is None],
            )
        finite_rewards = tuple(_finite(value, f"rewards[{index}]") for index, value in enumerate(rewards))
        if virtual_hops is None:
            hop_counts = (0,) * OFFICER_COUNT
        elif isinstance(virtual_hops, Mapping):
            unknown = set(virtual_hops) - set(range(OFFICER_COUNT))
            if unknown:
                raise ResearchValidationError("INVALID_OFFICER_ID", "virtual-hop map has unknown officers")
            hop_counts = tuple(virtual_hops.get(index, 0) for index in range(OFFICER_COUNT))
        else:
            if len(virtual_hops) != OFFICER_COUNT:
                raise ResearchValidationError(
                    "INVALID_VIRTUAL_HOP_DIMENSION", "virtual hops must have six entries"
                )
            hop_counts = tuple(virtual_hops)
        for officer_id, count in enumerate(hop_counts):
            if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= self.max_virtual_hops:
                raise ResearchValidationError(
                    "VIRTUAL_HOP_BOUND_EXCEEDED",
                    "zero-time virtual hops exceeded the configured bound",
                    path=f"virtual_hops[{officer_id}]", expected=[0, self.max_virtual_hops], actual=count,
                )
        replacements: list[PendingDecision] = []
        for officer_id, reward in enumerate(finite_rewards):
            pending = self._pending[officer_id]
            assert pending is not None
            accumulated = pending.discounted_reward + (self.gamma ** pending.duration_steps) * reward
            replacements.append(replace(pending, discounted_reward=accumulated, duration_steps=pending.duration_steps + 1))
        self._pending[:] = replacements

    def close_terminal(
        self, next_critic_contexts: Mapping[int, Iterable[float]] | Sequence[Iterable[float]]
    ) -> tuple[DecisionTransition, ...]:
        """Atomically close exactly six pending transitions at episode termination."""
        if any(item is None for item in self._pending):
            raise ResearchValidationError(
                "INCOMPLETE_TERMINAL_CLOSURE", "terminal closure requires all six pending transitions",
                actual=[index for index, item in enumerate(self._pending) if item is None],
            )
        if isinstance(next_critic_contexts, Mapping):
            if set(next_critic_contexts) != set(range(OFFICER_COUNT)):
                raise ResearchValidationError(
                    "INVALID_TERMINAL_CONTEXTS", "terminal contexts must identify officers 0 through 5"
                )
            supplied = tuple(next_critic_contexts[index] for index in range(OFFICER_COUNT))
        else:
            if len(next_critic_contexts) != OFFICER_COUNT:
                raise ResearchValidationError(
                    "INVALID_TERMINAL_CONTEXTS", "terminal closure requires six next contexts"
                )
            supplied = tuple(next_critic_contexts)
        contexts = tuple(_vector(value, f"next_critic_contexts[{index}]") for index, value in enumerate(supplied))
        if self._critic_context_dim is not None and any(len(value) != self._critic_context_dim for value in contexts):
            raise ResearchValidationError("SMDP_DIMENSION_MISMATCH", "terminal critic context dimension changed")
        transitions = tuple(
            self._close(self._pending[index], contexts[index], terminal=True)  # type: ignore[arg-type]
            for index in range(OFFICER_COUNT)
        )
        self._pending[:] = [None] * OFFICER_COUNT
        self._completed.extend(transitions)
        return transitions

    def _close(
        self, pending: PendingDecision, next_context: tuple[float, ...], *, terminal: bool
    ) -> DecisionTransition:
        return DecisionTransition(
            officer_id=pending.officer_id,
            actor_obs=pending.epoch.actor_obs,
            action=pending.epoch.action,
            stored_mask_bytes=pending.epoch.stored_mask_bytes,
            stored_mask_hash=pending.epoch.stored_mask_hash,
            old_log_prob=pending.epoch.old_log_prob,
            discounted_reward=pending.discounted_reward,
            duration_steps=pending.duration_steps,
            gamma=self.gamma,
            critic_context=pending.epoch.critic_context,
            next_critic_context=next_context,
            terminal=terminal,
        )

    def drain_completed(self) -> tuple[DecisionTransition, ...]:
        completed = tuple(self._completed)
        self._completed.clear()
        return completed


AsynchronousSMDPBuffer = AsyncDecisionTransitionBuffer
HybridSMDPBuffer = AsyncDecisionTransitionBuffer

def polyline_arc_length(geometry: Sequence[tuple[float, float]]) -> float:
    if len(geometry) < 2:
        raise ResearchValidationError("INVALID_ROAD_GEOMETRY", "polyline needs at least two points")
    points = tuple((_finite(x, "geometry.x"), _finite(y, "geometry.y")) for x, y in geometry)
    return sum(math.dist(first, second) for first, second in zip(points, points[1:]))


def point_at_arc_progress(
    geometry: Sequence[tuple[float, float]], progress: float
) -> tuple[float, float]:
    """Return a metric point at a normalized directed-polyline arc progress."""
    progress = _finite(progress, "progress")
    if not 0.0 <= progress <= 1.0:
        raise ResearchValidationError("INVALID_ARC_PROGRESS", "progress must be in [0, 1]")
    total = polyline_arc_length(geometry)
    if total <= 0.0:
        raise ResearchValidationError("ZERO_LENGTH_PHYSICAL_ARC", "physical road arc must be positive")
    target = progress * total
    walked = 0.0
    for first, second in zip(geometry, geometry[1:]):
        length = math.dist(first, second)
        if length <= 0.0:
            continue
        if walked + length >= target:
            fraction = (target - walked) / length
            return (first[0] + fraction * (second[0] - first[0]), first[1] + fraction * (second[1] - first[1]))
        walked += length
    return tuple(geometry[-1])  # type: ignore[return-value]


def _virtual_reachable(
    network: ModelNetwork, start: int, target: int, max_virtual_hops: int
) -> bool:
    if start == target:
        return True
    segments = {segment.id: segment for segment in network.segments}
    frontier = {(start, 0)}
    visited = {(start, 0)}
    while frontier:
        node, hops = frontier.pop()
        if hops >= max_virtual_hops:
            continue
        intersection = network.intersections[node]
        for segment_id in intersection.outgoing_segment_ids:
            segment = segments[segment_id]
            if not segment.virtual:
                continue
            if segment.end_id == target:
                return True
            state = (segment.end_id, hops + 1)
            if state not in visited:
                visited.add(state)
                frontier.add(state)
    return False


def validate_road_arc_transition(
    network: ModelNetwork,
    before: VehiclePlacement,
    after: VehiclePlacement,
    *,
    distance_budget_m: float,
    max_virtual_hops: int,
    tolerance_m: float = 1e-9,
) -> None:
    """Validate one environment step against directed road-arc invariants.

    Reaching an intersection may be followed by a zero-progress departure in the
    same environment step.  A different physical segment with positive progress,
    a disconnected departure, backward movement, or non-arc-length progress is
    rejected.
    """
    budget = _finite(distance_budget_m, "distance_budget_m")
    tolerance = _finite(tolerance_m, "tolerance_m")
    if budget < 0.0 or tolerance < 0.0 or max_virtual_hops < 0:
        raise ResearchValidationError("INVALID_ROAD_STEP_BOUND", "road-step bounds must be nonnegative")
    segments: dict[int, Segment] = {segment.id: segment for segment in network.segments}

    if before.intersection_id is not None:
        if after.intersection_id is not None:
            if after.intersection_id != before.intersection_id:
                raise ResearchValidationError(
                    "DISCONTINUOUS_SEGMENT_TRANSITION", "parked vehicle changed physical intersection"
                )
            return
        target = segments[int(after.segment_id)]
        if target.virtual or after.progress > tolerance:
            raise ResearchValidationError(
                "DISCONTINUOUS_SEGMENT_TRANSITION", "departure must enter a physical segment at progress zero"
            )
        if not _virtual_reachable(network, before.intersection_id, target.start_id, max_virtual_hops):
            raise ResearchValidationError(
                "DIRECTION_VIOLATION", "departure is not directed-reachable through bounded virtual hops"
            )
        return

    source = segments[int(before.segment_id)]
    if source.virtual:
        raise ResearchValidationError("INVALID_PHYSICAL_STATE", "vehicles cannot persist on virtual segments")
    length = polyline_arc_length(source.geometry_xy)
    remaining = (1.0 - before.progress) * length
    if after.segment_id == before.segment_id:
        expected_distance = min(budget, remaining)
        actual_distance = (after.progress - before.progress) * length
        if actual_distance < -tolerance or abs(actual_distance - expected_distance) > tolerance:
            raise ResearchValidationError(
                "ROAD_ARC_DISTANCE_MISMATCH", "progress must equal directed polyline arc travel",
                expected=expected_distance, actual=actual_distance,
            )
        if budget + tolerance >= remaining:
            raise ResearchValidationError(
                "DISCONTINUOUS_SEGMENT_TRANSITION", "arrived vehicle must resolve at the end intersection"
            )
        return
    if budget + tolerance < remaining:
        raise ResearchValidationError(
            "TELEPORT", "vehicle left its segment before reaching the end intersection"
        )
    if after.intersection_id is not None:
        if after.intersection_id != source.end_id:
            raise ResearchValidationError(
                "DISCONTINUOUS_SEGMENT_TRANSITION", "segment arrival must use its directed end intersection"
            )
        return
    target = segments[int(after.segment_id)]
    if target.virtual or after.progress > tolerance:
        raise ResearchValidationError(
            "DISCONTINUOUS_SEGMENT_TRANSITION", "new physical segment must start at intersection progress zero"
        )
    if not _virtual_reachable(network, source.end_id, target.start_id, max_virtual_hops):
        raise ResearchValidationError(
            "DIRECTION_VIOLATION", "physical segment changed without a directed intersection connection"
        )


__all__ = (
    "OFFICER_COUNT", "DecisionEpoch", "DecisionTransition", "PendingDecision",
    "AsyncDecisionTransitionBuffer", "AsynchronousSMDPBuffer", "HybridSMDPBuffer",
    "encode_action_mask", "polyline_arc_length", "point_at_arc_progress",
    "validate_road_arc_transition",
)

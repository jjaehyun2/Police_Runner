"""Safety-term reward penalties for the KIPoT mentoring feedback item T2.

The environment's core reward (:mod:`research.variants.rewards`,
:mod:`research.variants.remediation`) scores officers purely on road-distance
progress toward the fugitive.  It carries no explicit notion of *how* that
progress is made -- an officer that closes distance by racing through a
residential street scores identically to one that takes a arterial road.  This
module adds two additive, always-nonnegative penalty terms that are SUBTRACTED
from a base per-officer reward:

* :meth:`SafetyPenalties.risk_traversal_penalty` -- an officer pays per metre
  travelled on a risk-class (pedestrian-shared) road segment.
* :meth:`SafetyPenalties.herding_penalty` -- the team pays, split evenly, when
  its collective positioning pushes the fugitive's reachable region to become
  *more* risk-class-heavy than it was before the step (Requirement-shaped
  after the "herding toward danger" mentoring note).

Everything here is a **simulation proxy**, never a field-safety claim.
``road_class`` is an OSM tag, not a pedestrian-density sensor reading; per
``research.paper.scope`` this module never states or implies real-world
safety, and callers must not either.  ``living_street`` is literally OSM's tag
for a street legally shared with pedestrians, and ``residential`` is the OSM
class most associated with dense low-speed pedestrian-shared frontage, which is
why both are the fixed risk set below (see the coarsener's ``road_class``
propagation in ``osm_demo.coarsening._attributes``/``_segment_attributes``, and
the raw tag origin in ``osm_demo.osm_source``).

This module intentionally imports only from ``osm_demo.models``,
``research.policies.baselines`` (for the cached shortest-path ``_Graph``) and
``research.metrics.behavior`` (for reachability) -- never from
``research.training.trainer`` or ``research.variants.remediation`` -- so it can
be wrapped around any ``step_reward_fn`` without creating an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Sequence

from pursuit_evasion_rl.osm_demo.models import (
    POLICE_COUNT,
    DomainValidationError,
    ModelNetwork,
    Segment,
    VehiclePlacement,
)
from pursuit_evasion_rl.research.metrics.behavior import reachable_intersection_ids
from pursuit_evasion_rl.research.policies.baselines import _Graph

SAFETY_SCHEMA_VERSION = "1.0"

# Requirement (mentoring T2): OSM road classes treated as pedestrian-shared-space
# proxies.  ``living_street`` is literally OSM's tag for a street shared with
# pedestrians by law; ``residential`` is the class most associated with dense
# low-speed pedestrian frontage.  This is a simulation proxy (see module
# docstring and ``research.paper.scope``), never a measured pedestrian density.
RISK_ROAD_CLASSES: frozenset[str] = frozenset({"residential", "living_street"})

DEFAULT_RISK_TRAVERSAL_PER_100M = 0.02
DEFAULT_HERDING_RISK_WEIGHT = 0.05


@dataclass(frozen=True, slots=True)
class SafetyPenaltyConfig:
    """Immutable parameter contract for the two safety penalty terms.

    ``risk_traversal_per_100m`` is the penalty charged per 100 m an officer
    travels on a risk-class segment in one step.  ``herding_risk_weight`` is
    the penalty charged, per unit increase of the fugitive's reachable-region
    risk fraction, when the team's step makes the fugitive's remaining escape
    routes more risk-class-heavy than they were before the step.
    """

    risk_traversal_per_100m: float = DEFAULT_RISK_TRAVERSAL_PER_100M
    herding_risk_weight: float = DEFAULT_HERDING_RISK_WEIGHT
    risk_traversal_enabled: bool = True
    herding_enabled: bool = True
    schema_version: str = SAFETY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("risk_traversal_enabled", "herding_enabled"):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise DomainValidationError(
                    "INVALID_SAFETY_FACTOR",
                    f"{name} must be a boolean enable flag",
                    path=name,
                    actual=value,
                )
        for name in ("risk_traversal_per_100m", "herding_risk_weight"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise DomainValidationError(
                    "INVALID_SAFETY_PARAMETER",
                    f"{name} must be finite and nonnegative",
                    path=name,
                    actual=value,
                )
            object.__setattr__(self, name, value)


def segment_is_risk_class(segment: Segment) -> bool:
    """True when ``segment`` carries any risk-class component in its ``road_class``.

    ``road_class`` may be a ``"|"``-joined multi-class string (see
    ``osm_source._source_id`` fan-in for split ways); the segment counts as
    risk-class when ANY joined component is in :data:`RISK_ROAD_CLASSES`.
    """
    road_class = segment.attributes.get("road_class", "")
    if not road_class:
        return False
    return any(part in RISK_ROAD_CLASSES for part in str(road_class).split("|"))


def traversed_metres(
    before_placement: VehiclePlacement, after_placement: VehiclePlacement, graph: _Graph
) -> tuple[int | None, float]:
    """Metres travelled on ``before_placement.segment_id`` during one step.

    The environment advances the officer and only then resolves the next
    decision, so the segment actually traversed during a step is always the
    segment the officer occupied at the *start* of the step
    (``before_placement.segment_id``) -- never the one it decided into.  Three
    cases:

    * parked at an intersection the whole step (``before_placement.segment_id
      is None``): 0 metres on no segment;
    * stayed on the same segment: ``(after.progress - before.progress) *
      length_m``;
    * exited the segment (arrived at an intersection, or moved on to a new
      segment): the officer covered whatever was left of the segment,
      ``(1 - before.progress) * length_m``.
    """
    if before_placement.segment_id is None:
        return None, 0.0
    segment_id = before_placement.segment_id
    length_m = float(graph.seg[segment_id].length_m)
    if after_placement.segment_id == segment_id:
        metres = (after_placement.progress - before_placement.progress) * length_m
    else:
        metres = (1.0 - before_placement.progress) * length_m
    return segment_id, max(0.0, metres)


def _occupied_intersection(placement: VehiclePlacement, graph: _Graph) -> int:
    """The intersection a placement occupies or is about to arrive at.

    Parked vehicles occupy ``intersection_id`` directly; a vehicle mid-segment
    is attributed to the segment's end intersection, matching the "nearest/
    current intersection" convention used for herding-fraction bookkeeping.
    """
    if placement.segment_id is None:
        return int(placement.intersection_id)
    return int(graph.seg[placement.segment_id].end_id)


class SafetyPenalties:
    """Per-network safety penalty bookkeeping for :func:`wrap_step_reward`.

    Builds (or reuses) a cached ``_Graph`` and precomputes, once per network,
    which segments and which intersections are risk-class so that per-step
    calls do no repeated attribute parsing.
    """

    def __init__(
        self,
        network: ModelNetwork,
        config: SafetyPenaltyConfig,
        graph: _Graph | None = None,
    ) -> None:
        if not isinstance(network, ModelNetwork):
            raise DomainValidationError(
                "INVALID_NETWORK", "A ModelNetwork instance is required", actual=type(network).__name__
            )
        if not isinstance(config, SafetyPenaltyConfig):
            raise DomainValidationError(
                "INVALID_SAFETY_CONFIG",
                "A SafetyPenaltyConfig instance is required",
                actual=type(config).__name__,
            )
        self.network = network
        self.config = config
        self.graph = graph if graph is not None else _Graph(network)
        self._segment_risk: dict[int, bool] = {
            segment.id: segment_is_risk_class(segment) for segment in network.segments
        }
        self._intersection_risk: dict[int, bool] = self._build_intersection_risk()

    def _build_intersection_risk(self) -> dict[int, bool]:
        """An intersection is risk-incident when some segment touching it is risk-class."""
        incident: dict[int, bool] = {item.id: False for item in self.network.intersections}
        for segment in self.network.segments:
            if self._segment_risk.get(segment.id, False):
                incident[segment.start_id] = True
                incident[segment.end_id] = True
        return incident

    def risk_traversal_penalty(self, before_state: Any, after_state: Any) -> tuple[float, ...]:
        """Per-officer penalty (>= 0) for metres travelled on a risk-class segment this step."""
        penalties: list[float] = []
        for before_pl, after_pl in zip(before_state.police, after_state.police):
            segment_id, metres = traversed_metres(before_pl, after_pl, self.graph)
            is_risk = segment_id is not None and self._segment_risk.get(segment_id, False)
            if self.config.risk_traversal_enabled and is_risk and metres > 0.0:
                penalty = self.config.risk_traversal_per_100m * (metres / 100.0)
            else:
                penalty = 0.0
            penalties.append(max(0.0, penalty))
        return tuple(penalties)

    def _risk_fraction(self, state: Any) -> float:
        """Fraction of the fugitive's officer-blocked reachable region that is risk-incident."""
        fugitive_node = _occupied_intersection(state.fugitive, self.graph)
        occupied = frozenset(_occupied_intersection(placement, self.graph) for placement in state.police)
        reachable = reachable_intersection_ids(self.network, fugitive_node, removed=occupied)
        if not reachable:
            return 0.0
        risky = sum(1 for node in reachable if self._intersection_risk.get(node, False))
        return risky / len(reachable)

    def herding_penalty(self, before_state: Any, after_state: Any) -> float:
        """Team-level penalty (>= 0) for the step increasing the fugitive's risk-fraction.

        Only the *increase* is penalized (``max(0, after - before)``): a step
        that pushes the fugitive toward safer terrain earns no bonus here --
        this is a penalty term, not a shaped reward -- but it also never
        penalizes a genuine safety improvement.
        """
        if not self.config.herding_enabled:
            return 0.0
        delta = self._risk_fraction(after_state) - self._risk_fraction(before_state)
        return max(0.0, delta) * self.config.herding_risk_weight

    def __call__(self, before: Any, after: Any) -> tuple[float, ...]:
        """Per-officer TOTAL safety penalty: own traversal plus an equal herding share."""
        traversal = self.risk_traversal_penalty(before, after)
        herding_share = self.herding_penalty(before, after) / POLICE_COUNT
        return tuple(max(0.0, value + herding_share) for value in traversal)


StepRewardFn = Callable[[ModelNetwork, Any, Any, bool], Sequence[float]]


def wrap_step_reward(step_reward_fn: StepRewardFn, safety: SafetyPenalties) -> StepRewardFn:
    """Wrap a ``step_reward_fn`` so its per-officer rewards pay the safety penalties.

    Returns a callable with the same ``(network, before, after, captured)``
    signature ``ResearchTrainer.step_reward_fn`` expects.  The wrapped
    baseline reward is computed first and unchanged in shape; safety penalties
    are then SUBTRACTED per officer, so a caller can layer this around any
    conforming reward function (audited defaults, remediation, or otherwise)
    without that function knowing safety terms exist.
    """
    if not isinstance(safety, SafetyPenalties):
        raise DomainValidationError(
            "INVALID_SAFETY_PENALTIES", "A SafetyPenalties instance is required", actual=type(safety).__name__
        )

    def wrapped(network: ModelNetwork, before: Any, after: Any, captured: bool) -> tuple[float, ...]:
        base_rewards = tuple(float(value) for value in step_reward_fn(network, before, after, captured))
        penalties = safety(before, after)
        if len(penalties) != len(base_rewards):
            raise DomainValidationError(
                "SAFETY_PENALTY_DIMENSION_MISMATCH",
                "safety penalties must carry exactly one value per officer reward",
                expected=len(base_rewards),
                actual=len(penalties),
            )
        return tuple(reward - penalty for reward, penalty in zip(base_rewards, penalties))

    return wrapped


def wrap_step_reward_multi(step_reward_fn: StepRewardFn, config: SafetyPenaltyConfig) -> StepRewardFn:
    """Network-agnostic safety wrapper for ``ResearchTrainer.step_reward_fn``.

    The trainer calls its reward function with BOTH the train and the
    validation network (training rollouts vs checkpoint-selection rollouts),
    so a :class:`SafetyPenalties` bound to a single network would score
    validation states against the wrong graph.  This wrapper builds one
    :class:`SafetyPenalties` lazily per network identity and dispatches on
    the network each call actually passes.
    """
    if not isinstance(config, SafetyPenaltyConfig):
        raise DomainValidationError(
            "INVALID_SAFETY_CONFIG", "A SafetyPenaltyConfig instance is required", actual=type(config).__name__
        )
    cache: dict[int, SafetyPenalties] = {}

    def wrapped(network: ModelNetwork, before: Any, after: Any, captured: bool) -> tuple[float, ...]:
        safety = cache.get(id(network))
        if safety is None:
            safety = SafetyPenalties(network, config)
            cache[id(network)] = safety
        base_rewards = tuple(float(value) for value in step_reward_fn(network, before, after, captured))
        penalties = safety(before, after)
        if len(penalties) != len(base_rewards):
            raise DomainValidationError(
                "SAFETY_PENALTY_DIMENSION_MISMATCH",
                "safety penalties must carry exactly one value per officer reward",
                expected=len(base_rewards),
                actual=len(penalties),
            )
        return tuple(reward - penalty for reward, penalty in zip(base_rewards, penalties))

    # Preserve the remediation on_episode hook through the wrapper so the
    # trainer's road-dynamics axis still reaches the underlying reward.
    hook = getattr(step_reward_fn, "on_episode", None)
    if callable(hook):
        wrapped.on_episode = hook  # type: ignore[attr-defined]
    return wrapped


__all__ = (
    "DEFAULT_HERDING_RISK_WEIGHT",
    "DEFAULT_RISK_TRAVERSAL_PER_100M",
    "RISK_ROAD_CLASSES",
    "SAFETY_SCHEMA_VERSION",
    "SafetyPenaltyConfig",
    "SafetyPenalties",
    "StepRewardFn",
    "segment_is_risk_class",
    "traversed_metres",
    "wrap_step_reward",
    "wrap_step_reward_multi",
)

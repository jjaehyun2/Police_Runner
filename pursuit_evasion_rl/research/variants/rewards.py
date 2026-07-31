"""Audited reward component registry and leave-one-out ablation (Requirement 9.3-9.5).

Every coefficient, sign, and application point below is copied from
``pursuit_evasion_rl.research.training.trainer.ResearchTrainer._step_rewards``,
the reward computation the connected training CLI actually exercises (task
4.3).  This module is the single declarative registry: it does not change
trainer behavior, and ``tests/research/test_reward_variants.py`` asserts the
two stay numerically identical so they cannot silently drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import Sequence

from pursuit_evasion_rl.osm_demo.models import DomainValidationError
from pursuit_evasion_rl.research.canonical import content_hash

REWARD_SCHEMA_VERSION = "1.0"

# Audited from ResearchTrainer._step_rewards / TrainerConfig defaults.
AUDITED_TEAM_COEFFICIENT = 1.0
AUDITED_OWN_COEFFICIENT = 0.6
AUDITED_TIME_PENALTY = 0.02
AUDITED_CAPTURE_BONUS = 20.0
AUDITED_REGRESS_MULTIPLIER = 1.6
AUDITED_DISTANCE_SCALE_M = 50.0
# Requirement 9.7: the symmetric distance condition is identical except the
# own-distance regress multiplier no longer penalizes retreat more steeply.
SYMMETRIC_REGRESS_MULTIPLIER = 1.0


class RewardComponent(str, Enum):
    """The four audited reward components (Requirement 9.6)."""

    TEAM = "team"
    OWN = "own"
    TIME = "time"
    CAPTURE = "capture"


@dataclass(frozen=True, slots=True)
class RewardComponentSet:
    """Immutable, auditable coefficients for one reward Condition.

    ``team_coefficient``/``own_coefficient`` are dimensionless weights on a
    per-step distance delta normalized by ``distance_scale_m`` [m].
    ``time_penalty`` [reward/step] is subtracted every physical step.
    ``capture_bonus`` [reward] is added once, on the step capture resolves.
    ``regress_multiplier`` is dimensionless and only scales an officer's own
    term when its distance to the fugitive *increased* (a retreat).
    """

    team_coefficient: float = AUDITED_TEAM_COEFFICIENT
    own_coefficient: float = AUDITED_OWN_COEFFICIENT
    time_penalty: float = AUDITED_TIME_PENALTY
    capture_bonus: float = AUDITED_CAPTURE_BONUS
    regress_multiplier: float = AUDITED_REGRESS_MULTIPLIER
    distance_scale_m: float = AUDITED_DISTANCE_SCALE_M
    schema_version: str = REWARD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "team_coefficient", "own_coefficient", "time_penalty", "capture_bonus", "regress_multiplier",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise DomainValidationError(
                    "INVALID_REWARD_COEFFICIENT", f"{name} must be finite and nonnegative", path=name, actual=value,
                )
            object.__setattr__(self, name, value)
        scale = float(self.distance_scale_m)
        if not math.isfinite(scale) or scale <= 0.0:
            raise DomainValidationError(
                "INVALID_REWARD_COEFFICIENT", "distance_scale_m must be positive and finite",
                path="distance_scale_m", actual=scale,
            )
        object.__setattr__(self, "distance_scale_m", scale)

    @property
    def config_hash(self) -> str:
        return content_hash(self)

    @property
    def retreat_arm(self) -> str:
        return "symmetric" if self.regress_multiplier == SYMMETRIC_REGRESS_MULTIPLIER else "asymmetric"


# The audited full component set, as actually trained by ResearchTrainer.
AUDITED_REWARD_COMPONENTS = RewardComponentSet()


@dataclass(frozen=True, slots=True)
class ComponentTrace:
    """Per-officer, per-component decomposition of one step's reward."""

    officer_id: int
    team: float
    own: float
    time: float
    capture: float

    @property
    def total(self) -> float:
        return self.team + self.own + self.time + self.capture


def _finite_distances(values: Sequence[float], name: str) -> tuple[float, ...]:
    if not values:
        raise DomainValidationError("EMPTY_REWARD_DISTANCES", f"{name} must not be empty", path=name)
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value < 0.0 for value in result):
        raise DomainValidationError(
            "INVALID_REWARD_DISTANCES", f"{name} must contain finite nonnegative distances", path=name, actual=values,
        )
    return result


def compute_component_trace(
    components: RewardComponentSet,
    old_distances: Sequence[float],
    new_distances: Sequence[float],
    *,
    captured: bool,
) -> tuple[ComponentTrace, ...]:
    """Return the per-officer component breakdown for one physical step.

    ``old_distances``/``new_distances`` are each officer's distance to the
    fugitive immediately before/after the step, in the same officer order.
    """
    old = _finite_distances(old_distances, "old_distances")
    new = _finite_distances(new_distances, "new_distances")
    if len(old) != len(new):
        raise DomainValidationError(
            "REWARD_DISTANCE_COUNT_MISMATCH", "old_distances and new_distances must have the same officer count",
            expected=len(old), actual=len(new),
        )
    scale = components.distance_scale_m
    team_term = (min(old) - min(new)) / scale
    team_value = components.team_coefficient * team_term
    time_value = -components.time_penalty
    capture_value = components.capture_bonus if captured else 0.0
    traces = []
    for officer_id, (before, after) in enumerate(zip(old, new)):
        delta = (before - after) / scale
        own_delta = delta if delta >= 0.0 else components.regress_multiplier * delta
        own_value = components.own_coefficient * own_delta
        traces.append(ComponentTrace(officer_id, team_value, own_value, time_value, capture_value))
    return tuple(traces)


def total_rewards(
    components: RewardComponentSet,
    old_distances: Sequence[float],
    new_distances: Sequence[float],
    *,
    captured: bool,
) -> tuple[float, ...]:
    """Per-officer total reward, matching ``ResearchTrainer._step_rewards`` exactly."""
    return tuple(
        trace.total
        for trace in compute_component_trace(components, old_distances, new_distances, captured=captured)
    )


_COMPONENT_FIELDS = {
    RewardComponent.TEAM: "team_coefficient",
    RewardComponent.OWN: "own_coefficient",
    RewardComponent.TIME: "time_penalty",
    RewardComponent.CAPTURE: "capture_bonus",
}


def leave_one_component_out(
    excluded: RewardComponent, base: RewardComponentSet = AUDITED_REWARD_COMPONENTS
) -> RewardComponentSet:
    """Zero exactly one component's coefficient; terminal semantics are unaffected.

    ``captured`` truth and episode termination never change here -- only the
    *reward contribution* of the excluded component is removed, which is what
    Requirement 9.6 means by leave-one-component-out.
    """
    try:
        field = _COMPONENT_FIELDS[RewardComponent(excluded)]
    except ValueError as exc:
        raise DomainValidationError(
            "INVALID_REWARD_COMPONENT", "excluded must be one of the audited reward components", actual=excluded,
        ) from exc
    return replace(base, **{field: 0.0})


def symmetric_retreat_components(base: RewardComponentSet = AUDITED_REWARD_COMPONENTS) -> RewardComponentSet:
    """The symmetric-distance retreat condition (Requirement 9.7)."""
    return replace(base, regress_multiplier=SYMMETRIC_REGRESS_MULTIPLIER)


def asymmetric_retreat_components(base: RewardComponentSet = AUDITED_REWARD_COMPONENTS) -> RewardComponentSet:
    """The audited asymmetric-regress retreat condition (Requirement 9.7)."""
    return replace(base, regress_multiplier=AUDITED_REGRESS_MULTIPLIER)


@dataclass(frozen=True, slots=True)
class RewardAblationResult:
    """One reward-ablation Condition's result row.

    Requirement 9.9 requires every observation/reward ablation report to
    carry Capture_Metric and stability-metric (Anti_Oscillation /
    Physical_Plausibility) references alongside the reward numbers, so this
    schema makes both references mandatory, non-empty fields: a reward delta
    can never be reported without its accompanying behavioral evidence.
    """

    condition_id: str
    components: RewardComponentSet
    excluded_component: RewardComponent | None
    capture_metric_ref: str
    stability_metric_ref: str
    component_trace: tuple[ComponentTrace, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.condition_id, str) or not self.condition_id.strip():
            raise DomainValidationError(
                "MISSING_REQUIRED_FIELD", "condition_id must be non-empty", path="condition_id",
            )
        if self.excluded_component is not None:
            object.__setattr__(self, "excluded_component", RewardComponent(self.excluded_component))
        if not isinstance(self.capture_metric_ref, str) or not self.capture_metric_ref.strip():
            raise DomainValidationError(
                "MISSING_CAPTURE_METRIC_REFERENCE",
                "reward ablation results require a capture metric reference",
                path="capture_metric_ref",
            )
        if not isinstance(self.stability_metric_ref, str) or not self.stability_metric_ref.strip():
            raise DomainValidationError(
                "MISSING_STABILITY_METRIC_REFERENCE",
                "reward ablation results require a stability metric reference",
                path="stability_metric_ref",
            )
        if not self.component_trace:
            raise DomainValidationError(
                "MISSING_REQUIRED_FIELD", "component_trace must not be empty", path="component_trace",
            )
        object.__setattr__(self, "component_trace", tuple(self.component_trace))

    @property
    def config_hash(self) -> str:
        return content_hash(self)


__all__ = (
    "AUDITED_CAPTURE_BONUS",
    "AUDITED_DISTANCE_SCALE_M",
    "AUDITED_OWN_COEFFICIENT",
    "AUDITED_REGRESS_MULTIPLIER",
    "AUDITED_REWARD_COMPONENTS",
    "AUDITED_TEAM_COEFFICIENT",
    "AUDITED_TIME_PENALTY",
    "REWARD_SCHEMA_VERSION",
    "SYMMETRIC_REGRESS_MULTIPLIER",
    "ComponentTrace",
    "RewardAblationResult",
    "RewardComponent",
    "RewardComponentSet",
    "asymmetric_retreat_components",
    "compute_component_trace",
    "leave_one_component_out",
    "symmetric_retreat_components",
    "total_rewards",
)

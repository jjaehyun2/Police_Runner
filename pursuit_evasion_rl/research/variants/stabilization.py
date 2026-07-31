"""The 2x2 stabilization factorial: U-turn suppression x hysteresis (Requirement 10.4-10.11).

Both factors are protocol-fixed, not learned.  U-turn suppression is a *soft*
additive penalty on reversing candidates -- never a mask -- so a reversal stays
selectable when every alternative is worse.  Hysteresis is an explicit state
machine that holds its committed target for at least ``min_hold_k`` decisions
and only switches once a competing option leads by more than ``margin``.

Every result-affecting rule lives on :class:`StabilizationCondition` and so
enters its ``condition_hash``; :class:`StabilizationWrapperState` is the
manifest record Requirement 10.7 asks for.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math
from typing import Sequence

from pursuit_evasion_rl.osm_demo.models import DomainValidationError
from pursuit_evasion_rl.research.canonical import content_hash

STABILIZATION_SCHEMA_VERSION = "1.0"

# Requirement 10.7: stabilization wrappers carry their own named stream so the
# manifest can attribute any stochastic tie-breaking to it.
STABILIZATION_RNG_STREAM = "research.stabilization"

# Requirement 10.5: the reversal predicate and the suppression method are fixed.
REVERSAL_RULE = "candidate_segment_end_intersection_equals_previous_decision_intersection"
U_TURN_METHOD = "soft_additive_score_penalty"
DEFAULT_U_TURN_PENALTY = 1.0

# Requirement 10.6: what is held, and the minimum-hold / switch conditions.
HYSTERESIS_HOLD_TARGET = "committed_goal_intersection_id"
DEFAULT_HYSTERESIS_MIN_HOLD_K = 3
DEFAULT_HYSTERESIS_MARGIN = 0.5

FACTOR_FIELDS: tuple[str, ...] = ("u_turn_suppression", "hysteresis")


@dataclass(frozen=True, slots=True)
class StabilizationCondition:
    """One arm of the 2x2 stabilization factorial.

    ``u_turn_penalty`` is subtracted from a reversing candidate's score
    (dimensionless, in the caller's score units).  ``hysteresis_min_hold_k`` is
    the minimum number of decisions the committed target is held before any
    switch is permitted, and ``hysteresis_margin`` is how far a competitor must
    lead the incumbent before that switch is taken.  The parameter fields stay
    identical across all four arms; only the two boolean factors vary, which is
    what makes a single-factor diff meaningful.
    """

    u_turn_suppression: bool = False
    hysteresis: bool = False
    u_turn_penalty: float = DEFAULT_U_TURN_PENALTY
    reversal_rule: str = REVERSAL_RULE
    u_turn_method: str = U_TURN_METHOD
    hysteresis_hold_target: str = HYSTERESIS_HOLD_TARGET
    hysteresis_min_hold_k: int = DEFAULT_HYSTERESIS_MIN_HOLD_K
    hysteresis_margin: float = DEFAULT_HYSTERESIS_MARGIN
    rng_stream: str = STABILIZATION_RNG_STREAM
    schema_version: str = STABILIZATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in FACTOR_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise DomainValidationError(
                    "INVALID_STABILIZATION_FACTOR",
                    f"{name} must be a boolean factor level",
                    path=name,
                    actual=value,
                )
        for name in ("u_turn_penalty", "hysteresis_margin"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise DomainValidationError(
                    "INVALID_STABILIZATION_PARAMETER",
                    f"{name} must be finite and nonnegative",
                    path=name,
                    actual=value,
                )
            object.__setattr__(self, name, value)
        k = self.hysteresis_min_hold_k
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise DomainValidationError(
                "INVALID_STABILIZATION_PARAMETER",
                "hysteresis_min_hold_k must be an integer of at least 1",
                path="hysteresis_min_hold_k",
                actual=k,
            )

    @property
    def condition_id(self) -> str:
        return (
            f"uturn_{'on' if self.u_turn_suppression else 'off'}"
            f"__hyst_{'on' if self.hysteresis else 'off'}"
        )

    @property
    def condition_hash(self) -> str:
        return content_hash(self)

    @property
    def effective_min_hold_k(self) -> int:
        """Hysteresis off means no dwell requirement: a switch may occur at once."""
        return self.hysteresis_min_hold_k if self.hysteresis else 1

    @property
    def effective_margin(self) -> float:
        """Hysteresis off means any strictly better competitor wins immediately."""
        return self.hysteresis_margin if self.hysteresis else 0.0


def all_stabilization_arms(
    base: StabilizationCondition = StabilizationCondition(),
) -> tuple[StabilizationCondition, ...]:
    """The exactly-four off/on x off/on arms of Requirement 10.4."""
    return tuple(
        replace(base, u_turn_suppression=u_turn, hysteresis=hyst)
        for u_turn in (False, True)
        for hyst in (False, True)
    )


def toggle_factor(condition: StabilizationCondition, factor: str) -> StabilizationCondition:
    """Flip exactly one declared factor, leaving every other field untouched."""
    if factor not in FACTOR_FIELDS:
        raise DomainValidationError(
            "INVALID_STABILIZATION_FACTOR",
            "factor must be one of the declared stabilization factors",
            path="factor",
            expected=list(FACTOR_FIELDS),
            actual=factor,
        )
    return replace(condition, **{factor: not getattr(condition, factor)})


def single_factor_diff(left: StabilizationCondition, right: StabilizationCondition) -> str:
    """Return the one factor separating two arms, or fail.

    Requirement 10.11 fails the contribution analysis when a non-target
    Condition field differs, so undeclared drift is an error here rather than a
    silently tolerated difference.
    """
    differing = [
        item.name
        for item in fields(left)
        if getattr(left, item.name) != getattr(right, item.name)
    ]
    drifted = [name for name in differing if name not in FACTOR_FIELDS]
    if drifted:
        raise DomainValidationError(
            "UNDECLARED_FACTOR_DRIFT",
            "stabilization arms differ in a non-target Condition field",
            actual=drifted,
        )
    if len(differing) != 1:
        raise DomainValidationError(
            "INVALID_SINGLE_FACTOR_DIFF",
            "exactly one stabilization factor must differ between the two arms",
            expected=1,
            actual=differing,
        )
    return differing[0]


def validate_stabilization_matrix(arms: Sequence[StabilizationCondition]) -> None:
    """Reject a factorial with a missing, duplicate, or drifted arm (Requirement 10.11)."""
    levels = [(arm.u_turn_suppression, arm.hysteresis) for arm in arms]
    expected = {(u_turn, hyst) for u_turn in (False, True) for hyst in (False, True)}
    if len(levels) != len(expected) or set(levels) != expected:
        raise DomainValidationError(
            "INVALID_STABILIZATION_MATRIX",
            "the stabilization factorial must contain exactly the four off/on x off/on arms",
            expected=sorted(expected),
            actual=sorted(levels),
        )
    baseline = replace(arms[0], u_turn_suppression=False, hysteresis=False)
    for arm in arms[1:]:
        if replace(arm, u_turn_suppression=False, hysteresis=False) != baseline:
            raise DomainValidationError(
                "UNDECLARED_FACTOR_DRIFT",
                "stabilization arms must differ only in their two declared factors",
                expected=baseline,
                actual=arm,
            )


# ---------------------------------------------------------------------------
# U-turn suppression (Requirement 10.5)
# ---------------------------------------------------------------------------


def is_reversal(
    *, previous_intersection_id: int | None, candidate_end_intersection_id: int
) -> bool:
    """The fixed reversal predicate named by :data:`REVERSAL_RULE`.

    A candidate reverses when it leads back to the intersection the officer
    departed from at its previous decision.  With no previous decision there is
    no direction of travel yet, so nothing counts as a reversal.
    """
    return (
        previous_intersection_id is not None
        and int(candidate_end_intersection_id) == int(previous_intersection_id)
    )


def u_turn_penalties(
    condition: StabilizationCondition,
    candidate_end_intersection_ids: Sequence[int],
    *,
    previous_intersection_id: int | None,
) -> tuple[float, ...]:
    """Per-candidate soft penalty to add to a score; zero when the factor is off.

    The penalty is additive and finite, so a reversing candidate remains
    selectable -- this is deliberately not an action mask.
    """
    if not candidate_end_intersection_ids:
        raise DomainValidationError(
            "EMPTY_CANDIDATE_SET",
            "candidate_end_intersection_ids must not be empty",
            path="candidate_end_intersection_ids",
        )
    if not condition.u_turn_suppression:
        return (0.0,) * len(candidate_end_intersection_ids)
    return tuple(
        -condition.u_turn_penalty
        if is_reversal(
            previous_intersection_id=previous_intersection_id,
            candidate_end_intersection_id=end_id,
        )
        else 0.0
        for end_id in candidate_end_intersection_ids
    )


# ---------------------------------------------------------------------------
# Hysteresis (Requirement 10.6-10.7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HysteresisStateSnapshot:
    """The manifest-recordable hysteresis state (Requirement 10.7)."""

    condition_id: str
    condition_hash: str
    hold_target: str
    target: int
    steps_held: int
    switch_count: int
    min_hold_k: int
    margin: float
    schema_version: str = STABILIZATION_SCHEMA_VERSION

    @property
    def config_hash(self) -> str:
        return content_hash(self)


class HysteresisController:
    """Holds a committed target until a dwell and a margin condition both clear.

    Each :meth:`step` is one decision epoch.  ``steps_held`` counts decisions
    since the current target was committed; a switch requires
    ``steps_held >= min_hold_k`` *and* ``candidate_score - current_score >
    margin`` at that same decision.  A margin that is exceeded only transiently
    while the target is still within its minimum hold is therefore discarded,
    which is exactly the oscillation this factor suppresses.
    """

    def __init__(self, condition: StabilizationCondition, initial_target: int) -> None:
        if not isinstance(condition, StabilizationCondition):
            raise DomainValidationError(
                "INVALID_STABILIZATION_CONDITION",
                "A StabilizationCondition instance is required",
                actual=type(condition).__name__,
            )
        if isinstance(initial_target, bool) or not isinstance(initial_target, int):
            raise DomainValidationError(
                "INVALID_HYSTERESIS_TARGET",
                "initial_target must be an integer target identifier",
                path="initial_target",
                actual=initial_target,
            )
        self.condition = condition
        self._target = int(initial_target)
        self._steps_held = 0
        self._switch_count = 0

    @property
    def state(self) -> int:
        """The currently committed target."""
        return self._target

    @property
    def steps_held(self) -> int:
        return self._steps_held

    @property
    def switch_count(self) -> int:
        return self._switch_count

    def step(self, candidate: int, candidate_score: float, current_score: float) -> bool:
        """Advance one decision; return whether the committed target switched."""
        if isinstance(candidate, bool) or not isinstance(candidate, int):
            raise DomainValidationError(
                "INVALID_HYSTERESIS_TARGET",
                "candidate must be an integer target identifier",
                path="candidate",
                actual=candidate,
            )
        scores = (float(candidate_score), float(current_score))
        if any(not math.isfinite(value) for value in scores):
            raise DomainValidationError(
                "INVALID_HYSTERESIS_SCORE",
                "candidate_score and current_score must be finite",
                actual=scores,
            )
        self._steps_held += 1
        if candidate == self._target:
            return False
        if self._steps_held < self.condition.effective_min_hold_k:
            return False
        if scores[0] - scores[1] <= self.condition.effective_margin:
            return False
        self._target = candidate
        self._steps_held = 0
        self._switch_count += 1
        return True

    def snapshot(self) -> HysteresisStateSnapshot:
        return HysteresisStateSnapshot(
            condition_id=self.condition.condition_id,
            condition_hash=self.condition.condition_hash,
            hold_target=self.condition.hysteresis_hold_target,
            target=self._target,
            steps_held=self._steps_held,
            switch_count=self._switch_count,
            min_hold_k=self.condition.effective_min_hold_k,
            margin=self.condition.effective_margin,
        )


@dataclass(frozen=True, slots=True)
class StabilizationWrapperState:
    """Wrapper/U-turn/hysteresis state plus RNG provenance (Requirement 10.7)."""

    condition_id: str
    condition_hash: str
    previous_intersection_id: int | None
    hysteresis: HysteresisStateSnapshot | None
    rng_stream: str = STABILIZATION_RNG_STREAM
    schema_version: str = STABILIZATION_SCHEMA_VERSION

    @property
    def config_hash(self) -> str:
        return content_hash(self)


# ---------------------------------------------------------------------------
# Result schema (Requirement 10.8-10.10)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StabilizationResult:
    """One stabilization arm's result row.

    Requirement 10.9 requires capture, containment, anti-oscillation and
    physical-plausibility metrics in the same table, so all four references are
    mandatory: a capture number can never be reported without the behavioral
    and physical evidence beside it.  Requirement 10.10 then forbids an
    improvement-only claim -- if capture improved while stability or
    plausibility degraded, ``limitation_ref`` must point at the recorded
    trade-off.
    """

    condition_id: str
    condition: StabilizationCondition
    capture_metric_ref: str
    containment_metric_ref: str
    anti_oscillation_metric_ref: str
    physical_plausibility_metric_ref: str
    capture_improved: bool = False
    stability_degraded: bool = False
    plausibility_degraded: bool = False
    limitation_ref: str | None = None
    schema_version: str = STABILIZATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.condition_id, str) or not self.condition_id.strip():
            raise DomainValidationError(
                "MISSING_REQUIRED_FIELD", "condition_id must be non-empty", path="condition_id"
            )
        for name in (
            "capture_metric_ref",
            "containment_metric_ref",
            "anti_oscillation_metric_ref",
            "physical_plausibility_metric_ref",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise DomainValidationError(
                    "MISSING_TRADE_OFF_METRIC_REFERENCE",
                    "stabilization results require capture, containment, anti-oscillation "
                    "and physical-plausibility metric references in the same row",
                    path=name,
                )
        degraded = self.stability_degraded or self.plausibility_degraded
        if self.capture_improved and degraded:
            if not isinstance(self.limitation_ref, str) or not self.limitation_ref.strip():
                raise DomainValidationError(
                    "UNREPORTED_TRADE_OFF",
                    "a capture improvement alongside a stability or plausibility "
                    "regression must record its trade-off limitation",
                    path="limitation_ref",
                )

    @property
    def config_hash(self) -> str:
        return content_hash(self)


__all__ = (
    "DEFAULT_HYSTERESIS_MARGIN",
    "DEFAULT_HYSTERESIS_MIN_HOLD_K",
    "DEFAULT_U_TURN_PENALTY",
    "FACTOR_FIELDS",
    "HYSTERESIS_HOLD_TARGET",
    "HysteresisController",
    "HysteresisStateSnapshot",
    "REVERSAL_RULE",
    "STABILIZATION_RNG_STREAM",
    "STABILIZATION_SCHEMA_VERSION",
    "StabilizationCondition",
    "StabilizationResult",
    "StabilizationWrapperState",
    "U_TURN_METHOD",
    "all_stabilization_arms",
    "is_reversal",
    "single_factor_diff",
    "toggle_factor",
    "u_turn_penalties",
    "validate_stabilization_matrix",
)

"""Physical plausibility validation and the hard-violation failure ledger.

Requirement 13.7 fixes **exactly eight** hard-violation categories, in this
order: direction violation, contraflow, off-road, speed-limit violation,
teleport, discontinuous segment transition, impossible immediate round-trip and
invalid action.  :class:`ViolationCategory` mirrors that list one-for-one --
never nine, never merged.

Requirement 13.10 makes a single occurrence fatal: if any category reaches a
count of one in an :class:`~EpisodeTrace`, the whole episode is a failure and is
recorded in the ledger with a content-addressed cause trace hash
(:class:`FailureLedgerRecord`).  Requirement 19.4's metric gate then consumes
:func:`account_intention_to_evaluate`, which keeps failed episodes visible in
intention-to-evaluate accounting instead of quietly shrinking the denominator.

Classification contract
-----------------------
Road-arc invariants are **not** reimplemented here.  Each transition is handed
to :func:`pursuit_evasion_rl.research.smdp.validate_road_arc_transition`, and
its stable error codes are translated into the eight reported categories:

======================================  ==================================
``validate_road_arc_transition`` code   reported category
======================================  ==================================
``DIRECTION_VIOLATION``                 ``DIRECTION_VIOLATION``
``ROAD_ARC_DISTANCE_MISMATCH``          ``SPEED_LIMIT_VIOLATION``
``TELEPORT``                            ``TELEPORT``
``DISCONTINUOUS_SEGMENT_TRANSITION``    ``DISCONTINUOUS_SEGMENT_TRANSITION``
``INVALID_PHYSICAL_STATE``              ``OFF_ROAD``
======================================  ==================================

Per-step travel is exact under that contract (it must equal
``min(distance_budget_m, remaining_arc_m)``), so any deviation is a violation of
the speed-derived budget and is reported as ``SPEED_LIMIT_VIOLATION``.

Four categories need checks the road-arc validator does not name separately.
They are evaluated **before** it so the more specific diagnosis wins, giving
every faulty transition exactly one category:

1. ``OFF_ROAD`` -- structural, id-based, never a geometric nearest-road
   distance.  A placement is off-road when it references a ``segment_id`` or
   ``intersection_id`` absent from the network, or when the *departing*
   placement occupies a ``virtual`` segment (vehicles may only cross virtual
   arcs in zero time, never persist on them -- the same condition the road
   validator raises ``INVALID_PHYSICAL_STATE`` for).
2. ``INVALID_ACTION`` -- the executed action is outside the legal support the
   environment itself published for that decision (``OSMRoadPursuitEnv``'s
   ``action_masks()``): out of range, negative, or masked off.
3. ``IMPOSSIBLE_ROUND_TRIP`` -- see below.
4. ``CONTRAFLOW`` -- ``after.progress`` decreased on the *same* directed
   segment.  Directed segments are this codebase's one-way representation, so
   travelling backward along a segment's own polyline is contraflow.  The
   underlying arithmetic overlaps ``ROAD_ARC_DISTANCE_MISMATCH``'s negative-
   movement rejection; the split exists so the reported breakdown names it.

Impossible immediate round-trip rule
------------------------------------
This is the one category with no prior implementation to match, so the exact
rule is fixed here.  Per officer, a *departure* is a transition whose ``before``
sits at intersection ``A`` and whose ``after`` sits on physical segment ``S``.
The officer's next report of being at intersection ``A`` again -- an ``after``
with ``intersection_id == A`` -- closes an immediate round trip.  Let

* ``outbound_m = polyline_arc_length(S.geometry_xy)`` (the validator's own
  notion of length, not the declared ``Segment.length_m``),
* ``budget_max`` = the largest ``distance_budget_m`` in that officer's trace,
  i.e. its per-step reach,
* ``delta`` = arrival step index minus departure step index.

``S.end_id`` is S's only exit, so reaching it costs at least
``ceil(outbound_m / budget_max)`` steps after the departure step, and returning
to ``A`` from there costs at least one further step.  The round trip is
therefore impossible when::

    delta < ceil(outbound_m / budget_max) + (0 if S.start_id == S.end_id else 1)

The ``+1`` is dropped for a self-loop segment, which needs no return leg.  A
non-positive ``budget_max`` makes the requirement unreachable, so any claimed
round trip is flagged.  The bound is deliberately conservative: everything it
flags is genuinely impossible, and it never guesses at return-path lengths.

Transitions are validated pairwise; adjacent-transition continuity
(``trace[i].after == trace[i + 1].before``) is a construction invariant of the
caller reading ``OSMRoadPursuitEnv.episode_state()`` snapshots.  The round-trip
rule is the only check that spans more than one transition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from pursuit_evasion_rl.osm_demo.models import ModelNetwork, Segment, VehiclePlacement
from pursuit_evasion_rl.research.canonical import content_hash
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.smdp import (
    polyline_arc_length,
    validate_road_arc_transition,
)

PHYSICAL_PLAUSIBILITY_SCHEMA_VERSION = "1.0"


class ViolationCategory(str, Enum):
    """The exactly eight hard physical-violation categories of Requirement 13.7."""

    DIRECTION_VIOLATION = "direction_violation"
    CONTRAFLOW = "contraflow"
    OFF_ROAD = "off_road"
    SPEED_LIMIT_VIOLATION = "speed_limit_violation"
    TELEPORT = "teleport"
    DISCONTINUOUS_SEGMENT_TRANSITION = "discontinuous_segment_transition"
    IMPOSSIBLE_ROUND_TRIP = "impossible_round_trip"
    INVALID_ACTION = "invalid_action"


VIOLATION_CATEGORIES: tuple[ViolationCategory, ...] = tuple(ViolationCategory)

_ROAD_CODE_CATEGORIES: Mapping[str, ViolationCategory] = MappingProxyType(
    {
        "DIRECTION_VIOLATION": ViolationCategory.DIRECTION_VIOLATION,
        "ROAD_ARC_DISTANCE_MISMATCH": ViolationCategory.SPEED_LIMIT_VIOLATION,
        "TELEPORT": ViolationCategory.TELEPORT,
        "DISCONTINUOUS_SEGMENT_TRANSITION": ViolationCategory.DISCONTINUOUS_SEGMENT_TRANSITION,
        "INVALID_PHYSICAL_STATE": ViolationCategory.OFF_ROAD,
    }
)


@dataclass(frozen=True, slots=True)
class TraceTransition:
    """One executed physical step of one officer, as observed in an episode trace."""

    step_index: int
    officer_id: int
    before: VehiclePlacement
    after: VehiclePlacement
    distance_budget_m: float
    executed_action: int
    action_mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        for name in ("step_index", "officer_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ResearchValidationError(
                    "INVALID_TRACE_INDEX", f"{name} must be a nonnegative integer",
                    path=name, actual=value,
                )
        budget = float(self.distance_budget_m)
        if not math.isfinite(budget) or budget < 0.0:
            raise ResearchValidationError(
                "INVALID_DISTANCE_BUDGET", "distance_budget_m must be finite and nonnegative",
                path="distance_budget_m", actual=self.distance_budget_m,
            )
        mask = tuple(bool(value) for value in self.action_mask)
        if not mask or not any(mask):
            raise ResearchValidationError(
                "EMPTY_ACTION_SUPPORT", "action mask must permit at least one action",
                path="action_mask", actual=list(mask),
            )
        object.__setattr__(self, "distance_budget_m", budget)
        object.__setattr__(self, "action_mask", mask)


@dataclass(frozen=True, slots=True)
class PhysicalViolation:
    """A single hard violation, carrying enough context to re-derive its cause."""

    category: ViolationCategory
    step_index: int
    officer_id: int
    code: str
    message: str
    before: VehiclePlacement
    after: VehiclePlacement


@dataclass(frozen=True, slots=True)
class FailureLedgerRecord:
    """Requirement 13.10 ledger entry for an episode with >= 1 hard violation."""

    episode_id: str
    counts: tuple[tuple[str, int], ...]
    violations: tuple[PhysicalViolation, ...]
    cause_trace_hash: str = field(default="", init=False)
    schema_version: str = PHYSICAL_PLAUSIBILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ResearchValidationError(
                "MISSING_EPISODE_ID", "a failure ledger record requires an episode identifier"
            )
        if not self.violations:
            raise ResearchValidationError(
                "EMPTY_FAILURE_CAUSE", "a failure ledger record requires at least one violation"
            )
        cause = {"episode_id": self.episode_id, "violations": list(self.violations)}
        object.__setattr__(self, "cause_trace_hash", content_hash(cause))


@dataclass(frozen=True, slots=True)
class PhysicalPlausibilityReport:
    """Per-episode physical plausibility outcome with all eight counts present."""

    episode_id: str
    transition_count: int
    counts: Mapping[ViolationCategory, int]
    violations: tuple[PhysicalViolation, ...]
    failure: FailureLedgerRecord | None

    @property
    def episode_failed(self) -> bool:
        return self.failure is not None

    def count(self, category: ViolationCategory) -> int:
        return self.counts[category]


def _segment_index(network: ModelNetwork) -> dict[int, Segment]:
    return {segment.id: segment for segment in network.segments}


def _off_road_reason(
    segments: Mapping[int, Segment],
    intersection_ids: frozenset[int],
    placement: VehiclePlacement,
    label: str,
) -> str | None:
    if placement.segment_id is not None:
        segment = segments.get(int(placement.segment_id))
        if segment is None:
            return f"{label} references unknown segment {placement.segment_id}"
        if label == "before" and segment.virtual:
            return f"{label} occupies virtual segment {placement.segment_id}"
        return None
    if int(placement.intersection_id) not in intersection_ids:
        return f"{label} references unknown intersection {placement.intersection_id}"
    return None


def _required_round_trip_steps(segment: Segment, budget_max: float) -> float:
    if budget_max <= 0.0:
        return math.inf
    outbound = polyline_arc_length(segment.geometry_xy)
    return_leg = 0 if segment.start_id == segment.end_id else 1
    return math.ceil(outbound / budget_max) + return_leg


def _classify(
    network: ModelNetwork,
    segments: Mapping[int, Segment],
    intersection_ids: frozenset[int],
    transition: TraceTransition,
    *,
    max_virtual_hops: int,
    tolerance_m: float,
    departures: dict[int, tuple[int, Segment]],
    budget_max: float,
) -> PhysicalViolation | None:
    """Return the single most specific violation for one transition, or ``None``."""
    before, after = transition.before, transition.after

    for label, placement in (("before", before), ("after", after)):
        reason = _off_road_reason(segments, intersection_ids, placement, label)
        if reason is not None:
            return _violation(ViolationCategory.OFF_ROAD, transition, "OFF_ROAD", reason)

    action = transition.executed_action
    mask = transition.action_mask
    if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < len(mask) or not mask[action]:
        return _violation(
            ViolationCategory.INVALID_ACTION,
            transition,
            "SELECTED_INVALID_ACTION",
            f"executed action {action!r} is outside the published legal mask support "
            f"{[index for index, legal in enumerate(mask) if legal]}",
        )

    if after.intersection_id is not None:
        departure = departures.get(int(after.intersection_id))
        if departure is not None:
            departure_step, departed_segment = departure
            required = _required_round_trip_steps(departed_segment, budget_max)
            elapsed = transition.step_index - departure_step
            if elapsed < required:
                return _violation(
                    ViolationCategory.IMPOSSIBLE_ROUND_TRIP,
                    transition,
                    "IMPOSSIBLE_ROUND_TRIP",
                    f"returned to intersection {after.intersection_id} {elapsed} step(s) after "
                    f"departing on segment {departed_segment.id}, which needs at least "
                    f"{required} step(s)",
                )

    if (
        before.segment_id is not None
        and after.segment_id is not None
        and int(before.segment_id) == int(after.segment_id)
        and after.progress < before.progress - tolerance_m
    ):
        return _violation(
            ViolationCategory.CONTRAFLOW,
            transition,
            "CONTRAFLOW",
            f"progress decreased from {before.progress} to {after.progress} on directed "
            f"segment {before.segment_id}",
        )

    try:
        validate_road_arc_transition(
            network,
            before,
            after,
            distance_budget_m=transition.distance_budget_m,
            max_virtual_hops=max_virtual_hops,
            tolerance_m=tolerance_m,
        )
    except ResearchValidationError as error:
        category = _ROAD_CODE_CATEGORIES.get(error.code)
        if category is None:
            raise
        return _violation(category, transition, error.code, str(error))
    return None


def _violation(
    category: ViolationCategory, transition: TraceTransition, code: str, message: str
) -> PhysicalViolation:
    return PhysicalViolation(
        category=category,
        step_index=transition.step_index,
        officer_id=transition.officer_id,
        code=code,
        message=message,
        before=transition.before,
        after=transition.after,
    )


def validate_episode_trace(
    network: ModelNetwork,
    transitions: Sequence[TraceTransition],
    *,
    episode_id: str,
    max_virtual_hops: int = 0,
    tolerance_m: float = 1e-9,
) -> PhysicalPlausibilityReport:
    """Classify every transition of one episode into the eight fixed categories.

    All eight counts are always present, including zeros.  The episode fails
    (Requirement 13.10) as soon as any single count reaches one, and the returned
    :class:`FailureLedgerRecord` carries the content-addressed cause trace hash.
    """
    if not episode_id:
        raise ResearchValidationError(
            "MISSING_EPISODE_ID", "episode_id must be non-empty", path="episode_id"
        )
    segments = _segment_index(network)
    intersection_ids = frozenset(item.id for item in network.intersections)
    budgets: dict[int, float] = {}
    for transition in transitions:
        officer = transition.officer_id
        budgets[officer] = max(budgets.get(officer, 0.0), transition.distance_budget_m)

    counts = {category: 0 for category in VIOLATION_CATEGORIES}
    violations: list[PhysicalViolation] = []
    departures: dict[int, dict[int, tuple[int, Segment]]] = {}

    for transition in transitions:
        officer_departures = departures.setdefault(transition.officer_id, {})
        violation = _classify(
            network,
            segments,
            intersection_ids,
            transition,
            max_virtual_hops=max_virtual_hops,
            tolerance_m=tolerance_m,
            departures=officer_departures,
            budget_max=budgets[transition.officer_id],
        )
        if violation is not None:
            counts[violation.category] += 1
            violations.append(violation)

        after = transition.after
        if after.intersection_id is not None:
            officer_departures.pop(int(after.intersection_id), None)
        elif transition.before.intersection_id is not None:
            segment = segments.get(int(after.segment_id))
            if segment is not None:
                officer_departures[int(transition.before.intersection_id)] = (
                    transition.step_index,
                    segment,
                )

    failure = (
        FailureLedgerRecord(
            episode_id=episode_id,
            counts=tuple((category.value, counts[category]) for category in VIOLATION_CATEGORIES),
            violations=tuple(violations),
        )
        if violations
        else None
    )
    return PhysicalPlausibilityReport(
        episode_id=episode_id,
        transition_count=len(transitions),
        counts=MappingProxyType(counts),
        violations=tuple(violations),
        failure=failure,
    )


@dataclass(frozen=True, slots=True)
class IntentionToEvaluateAccounting:
    """Conserved planned-vs-observed accounting; failures stay visible, never dropped."""

    planned_episode_ids: tuple[str, ...]
    valid_episode_ids: tuple[str, ...]
    failed_records: tuple[FailureLedgerRecord, ...]
    missing_episode_ids: tuple[str, ...]

    @property
    def planned_count(self) -> int:
        return len(self.planned_episode_ids)

    @property
    def valid_count(self) -> int:
        return len(self.valid_episode_ids)

    @property
    def failed_count(self) -> int:
        return len(self.failed_records)

    @property
    def missing_count(self) -> int:
        return len(self.missing_episode_ids)

    @property
    def conserved(self) -> bool:
        return self.valid_count + self.failed_count + self.missing_count == self.planned_count

    def cause_trace_hash(self, episode_id: str) -> str:
        for record in self.failed_records:
            if record.episode_id == episode_id:
                return record.cause_trace_hash
        raise ResearchValidationError(
            "UNKNOWN_FAILED_EPISODE", "episode is not recorded in the failure ledger",
            path="episode_id", actual=episode_id,
        )


def account_intention_to_evaluate(
    reports: Iterable[PhysicalPlausibilityReport], *, planned_episode_ids: Sequence[str]
) -> IntentionToEvaluateAccounting:
    """Split the planned episodes into valid, failed-with-cause and never-reported.

    A hard-failed episode keeps its slot in the denominator with a named cause
    (Requirement 13.10 / 19.4), and an episode that was planned but never
    reported surfaces as ``missing`` rather than silently reducing the total.
    """
    planned = tuple(planned_episode_ids)
    if len(set(planned)) != len(planned):
        raise ResearchValidationError(
            "DUPLICATE_PLANNED_EPISODE", "planned episode identifiers must be unique",
            path="planned_episode_ids", actual=list(planned),
        )
    planned_set = set(planned)
    valid: list[str] = []
    failed: list[FailureLedgerRecord] = []
    seen: set[str] = set()
    for report in reports:
        if report.episode_id not in planned_set:
            raise ResearchValidationError(
                "UNPLANNED_EPISODE", "reported episode was never planned",
                path="episode_id", actual=report.episode_id,
            )
        if report.episode_id in seen:
            raise ResearchValidationError(
                "DUPLICATE_EPISODE_REPORT", "an episode was reported more than once",
                path="episode_id", actual=report.episode_id,
            )
        seen.add(report.episode_id)
        if report.failure is None:
            valid.append(report.episode_id)
        else:
            failed.append(report.failure)
    return IntentionToEvaluateAccounting(
        planned_episode_ids=planned,
        valid_episode_ids=tuple(valid),
        failed_records=tuple(failed),
        missing_episode_ids=tuple(item for item in planned if item not in seen),
    )


__all__ = (
    "PHYSICAL_PLAUSIBILITY_SCHEMA_VERSION",
    "VIOLATION_CATEGORIES",
    "FailureLedgerRecord",
    "IntentionToEvaluateAccounting",
    "PhysicalPlausibilityReport",
    "PhysicalViolation",
    "TraceTransition",
    "ViolationCategory",
    "account_intention_to_evaluate",
    "validate_episode_trace",
)

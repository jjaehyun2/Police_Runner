"""Six-officer behavioral, containment and contribution metrics (Requirement 13.1-13.6, 13.8).

Every quantity produced here is registered in :data:`METRIC_REGISTRY` with its
symbol, formula, SI unit, direction, time window and aggregation level, so a
number can never be reported without the semantics Requirement 13.1 fixes.

Two conventions are deliberately inherited rather than reinvented:

* the U-turn predicate is :func:`...variants.stabilization.is_reversal`, i.e.
  the protocol-fixed :data:`REVERSAL_RULE`; and
* the approach/retreat sign convention is the ``delta = (old - new)`` rule of
  ``...variants.rewards.compute_component_trace`` -- a positive delta means the
  officer closed distance on the fugitive.

Angular coverage reuses the audited formula already shared by
``...variants.observations.Observation28DAdapter._extra`` and
``...policies.baselines.EncirclementPolice``: bearings are taken from the
fugitive as ``atan2(officer_y - fugitive_y, officer_x - fugitive_x)``, sorted,
and the circular gaps between consecutive bearings (including the wrap-around
gap from the last back to the first) give
``coverage = 1 - max_gap / (2*pi)``.  Because only the sorted circular gap
structure is used, coverage and maximum gap are invariant under any rigid
rotation or translation of the whole formation (Correctness Property 22).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import math
import statistics
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from pursuit_evasion_rl.osm_demo.models import (
    POLICE_COUNT,
    EpisodeConfig,
    ModelNetwork,
)
from pursuit_evasion_rl.research.canonical import content_hash
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.variants.stabilization import REVERSAL_RULE, is_reversal

BEHAVIOR_SCHEMA_VERSION = "1.0"

# SI (or explicitly dimensionless) unit labels; every metric carries one.
UNIT_METRE = "m"
UNIT_SECOND = "s"
UNIT_RADIAN = "rad"
UNIT_DIMENSIONLESS = "1"
UNIT_COUNT = "count"

DEFAULT_REVISIT_WINDOW_DECISIONS = 5
DEFAULT_EXIT_BLOCK_RADIUS_M = 50.0

# Requirement 13.5: the neutral replacement used by a leave-one-off difference
# must be pre-registered, never chosen after seeing the result.
NEUTRAL_REPLACEMENT_RULES: tuple[str, ...] = ("stationary", "uniform_random")


class MetricDirection(str, Enum):
    """Which way is better for a registered metric (Requirement 13.1)."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


class MetricAvailability(str, Enum):
    """Why a metric holds (or does not hold) a value.

    The three states stay distinct for the same reason
    ``policies.baselines.CostAvailability`` keeps them distinct: an Interior
    scenario genuinely has no exit to block (:attr:`NOT_APPLICABLE`), which must
    never read the same as a Boundary scenario where an officer blocked nothing
    (a measured ``0.0``) or where nobody recorded it (:attr:`NOT_MEASURED`).
    """

    MEASURED = "measured"
    NOT_APPLICABLE = "not_applicable"
    NOT_MEASURED = "not_measured"


@dataclass(frozen=True, slots=True)
class MetricValue:
    """One metric quantity plus the method that produced (or excused) it."""

    availability: MetricAvailability
    unit: str
    value: float | None = None
    method: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "availability", MetricAvailability(self.availability))
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ResearchValidationError(
                "MISSING_REQUIRED_FIELD", "unit must be non-empty", path="unit"
            )
        if not isinstance(self.method, str) or not self.method.strip():
            raise ResearchValidationError(
                "MISSING_METRIC_METHOD",
                "every metric value requires a computation method or an explicit reason",
                path="method",
            )
        if self.availability is MetricAvailability.MEASURED:
            value = float(self.value) if self.value is not None else None
            if value is None or not math.isfinite(value):
                raise ResearchValidationError(
                    "INVALID_METRIC_VALUE",
                    "a measured metric requires a finite value",
                    path="value",
                    actual=self.value,
                )
            object.__setattr__(self, "value", value)
        elif self.value is not None:
            raise ResearchValidationError(
                "INVALID_METRIC_VALUE",
                "only measured metrics may carry a value",
                path="value",
                expected=None,
                actual=self.value,
            )


def measured_metric(value: float, unit: str, method: str) -> MetricValue:
    return MetricValue(MetricAvailability.MEASURED, unit, value=value, method=method)


def not_applicable_metric(unit: str, reason: str) -> MetricValue:
    """A metric this scenario cannot have -- distinct from an unrecorded one."""
    return MetricValue(MetricAvailability.NOT_APPLICABLE, unit, method=reason)


def not_measured_metric(unit: str, reason: str) -> MetricValue:
    """A metric that does exist here but which nobody computed."""
    return MetricValue(MetricAvailability.NOT_MEASURED, unit, method=reason)


_THRESHOLD_REASON = (
    "practical thresholds are registered by the claim-gate protocol, not by the metric definition"
)


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """The fixed semantics of one metric (Requirement 13.1)."""

    key: str
    symbol: str
    formula: str
    unit: str
    direction: MetricDirection
    aggregation_level: str
    time_window: str
    practical_threshold: MetricValue

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", MetricDirection(self.direction))
        if self.aggregation_level not in ("per_officer", "team"):
            raise ResearchValidationError(
                "INVALID_AGGREGATION_LEVEL",
                "aggregation_level must be per_officer or team",
                path="aggregation_level",
                actual=self.aggregation_level,
            )
        for name in ("key", "symbol", "formula", "unit", "time_window"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ResearchValidationError(
                    "MISSING_REQUIRED_FIELD", f"{name} must be non-empty", path=name
                )


def _spec(
    key: str,
    symbol: str,
    formula: str,
    unit: str,
    direction: MetricDirection,
    aggregation_level: str,
    time_window: str,
) -> MetricSpec:
    return MetricSpec(
        key=key,
        symbol=symbol,
        formula=formula,
        unit=unit,
        direction=direction,
        aggregation_level=aggregation_level,
        time_window=time_window,
        practical_threshold=not_measured_metric(unit, _THRESHOLD_REASON),
    )


_DECISION_TRANSITIONS = "decision transitions 1..n-1 of one episode"
_DECISION_EPOCHS = "decision epochs 0..n-1 of one episode"
_SNAPSHOT = "one decision-time state snapshot"

METRIC_REGISTRY: Mapping[str, MetricSpec] = MappingProxyType(
    {
        spec.key: spec
        for spec in (
            _spec(
                "u_turn_rate",
                "R_uturn",
                f"|{{i in 1..n-1 : {REVERSAL_RULE}}}| / (n-1)",
                UNIT_DIMENSIONLESS,
                MetricDirection.LOWER_IS_BETTER,
                "per_officer",
                _DECISION_TRANSITIONS,
            ),
            _spec(
                "revisit_rate",
                "R_revisit",
                "|{i in 1..n-1 : x_i in {x_{i-W}..x_{i-1}}}| / (n-1)",
                UNIT_DIMENSIONLESS,
                MetricDirection.LOWER_IS_BETTER,
                "per_officer",
                _DECISION_TRANSITIONS,
            ),
            _spec(
                "action_switch_rate",
                "R_aswitch",
                "|{i in 1..n-1 : a_i != a_{i-1}}| / (n-1)",
                UNIT_DIMENSIONLESS,
                MetricDirection.LOWER_IS_BETTER,
                "per_officer",
                _DECISION_TRANSITIONS,
            ),
            _spec(
                "role_switch_rate",
                "R_rswitch",
                "|{i in 1..n-1 : g_i != g_{i-1}}| / (n-1)",
                UNIT_DIMENSIONLESS,
                MetricDirection.LOWER_IS_BETTER,
                "per_officer",
                _DECISION_TRANSITIONS,
            ),
            _spec(
                "idle_rate",
                "R_idle",
                "|{i in 0..n-1 : legal_move_available_i and displacement_i == 0}| / n",
                UNIT_DIMENSIONLESS,
                MetricDirection.LOWER_IS_BETTER,
                "per_officer",
                _DECISION_EPOCHS,
            ),
            _spec(
                "approach_distance_m",
                "D_approach",
                "sum_i max(d_{i-1} - d_i, 0)",
                UNIT_METRE,
                MetricDirection.HIGHER_IS_BETTER,
                "per_officer",
                _DECISION_TRANSITIONS,
            ),
            _spec(
                "retreat_distance_m",
                "D_retreat",
                "sum_i max(d_i - d_{i-1}, 0)",
                UNIT_METRE,
                MetricDirection.LOWER_IS_BETTER,
                "per_officer",
                _DECISION_TRANSITIONS,
            ),
            _spec(
                "zero_displacement_time_s",
                "T_zero",
                "sum_{i : displacement_i == 0} duration_i",
                UNIT_SECOND,
                MetricDirection.LOWER_IS_BETTER,
                "per_officer",
                _DECISION_EPOCHS,
            ),
            _spec(
                "capture_radius_occupancy_steps",
                "N_cap",
                "|{i in 0..n-1 : d_i <= capture_radius_m}|",
                UNIT_COUNT,
                MetricDirection.HIGHER_IS_BETTER,
                "per_officer",
                _DECISION_EPOCHS,
            ),
            _spec(
                "capture_radius_occupancy_time_s",
                "T_cap",
                "sum_{i : d_i <= capture_radius_m} duration_i",
                UNIT_SECOND,
                MetricDirection.HIGHER_IS_BETTER,
                "per_officer",
                _DECISION_EPOCHS,
            ),
            _spec(
                "capture_radius_entry_events",
                "E_cap",
                "|{i : d_i <= capture_radius_m and (i == 0 or d_{i-1} > capture_radius_m)}|",
                UNIT_COUNT,
                MetricDirection.HIGHER_IS_BETTER,
                "per_officer",
                _DECISION_EPOCHS,
            ),
            _spec(
                "exit_block_duration_s",
                "T_exit",
                "sum_{i : min_e dist(x_i, e) <= exit_block_radius_m} duration_i",
                UNIT_SECOND,
                MetricDirection.HIGHER_IS_BETTER,
                "per_officer",
                _DECISION_EPOCHS,
            ),
            _spec(
                "angular_coverage",
                "C_ang",
                "1 - max_circular_bearing_gap / (2*pi)",
                UNIT_DIMENSIONLESS,
                MetricDirection.HIGHER_IS_BETTER,
                "team",
                _SNAPSHOT,
            ),
            _spec(
                "max_angular_gap_rad",
                "G_max",
                "max_k ((b_{k+1} - b_k) mod 2*pi) over bearings sorted ascending",
                UNIT_RADIAN,
                MetricDirection.LOWER_IS_BETTER,
                "team",
                _SNAPSHOT,
            ),
            _spec(
                "blocked_exit_fraction",
                "F_block",
                "|{e : min_k dist(p_k, e) <= exit_block_radius_m}| / |E|",
                UNIT_DIMENSIONLESS,
                MetricDirection.HIGHER_IS_BETTER,
                "team",
                _SNAPSHOT,
            ),
            _spec(
                "reachable_region_reduction",
                "R_reach",
                "(|reach(f)| - |reach(f) with occupied nodes removed|) / |reach(f)|",
                UNIT_DIMENSIONLESS,
                MetricDirection.HIGHER_IS_BETTER,
                "team",
                _SNAPSHOT,
            ),
        )
    }
)

# The per-officer metrics Requirement 13.8 aggregates into six rows plus a team row.
OFFICER_METRIC_KEYS: tuple[str, ...] = tuple(
    key for key, spec in METRIC_REGISTRY.items() if spec.aggregation_level == "per_officer"
)


def metric_spec(key: str) -> MetricSpec:
    try:
        return METRIC_REGISTRY[key]
    except KeyError as exc:
        raise ResearchValidationError(
            "UNREGISTERED_METRIC",
            "metric must be registered before it can be reported",
            path="key",
            expected=sorted(METRIC_REGISTRY),
            actual=key,
        ) from exc


# ---------------------------------------------------------------------------
# Protocol configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BehaviorMetricProtocol:
    """Every protocol-fixed constant these metrics depend on.

    ``revisit_window_decisions`` is the W of the revisit rate.
    ``capture_radius_m`` mirrors ``EpisodeConfig.capture_radius_m``.
    ``exit_block_radius_m`` is the single blocking rule shared by the
    per-officer exit-block duration and the team blocked-exit fraction: an
    officer blocks an exit when its metric distance to that exit intersection is
    at most this radius.
    """

    capture_radius_m: float
    revisit_window_decisions: int = DEFAULT_REVISIT_WINDOW_DECISIONS
    exit_block_radius_m: float = DEFAULT_EXIT_BLOCK_RADIUS_M
    reversal_rule: str = REVERSAL_RULE
    schema_version: str = BEHAVIOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("capture_radius_m", "exit_block_radius_m"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ResearchValidationError(
                    "INVALID_METRIC_PROTOCOL",
                    f"{name} must be positive and finite",
                    path=name,
                    actual=getattr(self, name),
                )
            object.__setattr__(self, name, value)
        window = self.revisit_window_decisions
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ResearchValidationError(
                "INVALID_METRIC_PROTOCOL",
                "revisit_window_decisions must be an integer of at least 1",
                path="revisit_window_decisions",
                actual=window,
            )

    @classmethod
    def from_episode_config(
        cls,
        config: EpisodeConfig,
        *,
        revisit_window_decisions: int = DEFAULT_REVISIT_WINDOW_DECISIONS,
        exit_block_radius_m: float = DEFAULT_EXIT_BLOCK_RADIUS_M,
    ) -> "BehaviorMetricProtocol":
        """Bind the capture radius to the episode config that produced the trace."""
        if not isinstance(config, EpisodeConfig):
            raise ResearchValidationError(
                "INVALID_METRIC_PROTOCOL",
                "An EpisodeConfig instance is required",
                path="config",
                actual=type(config).__name__,
            )
        return cls(
            capture_radius_m=config.capture_radius_m,
            revisit_window_decisions=revisit_window_decisions,
            exit_block_radius_m=exit_block_radius_m,
        )

    @property
    def protocol_hash(self) -> str:
        return content_hash(self)


# ---------------------------------------------------------------------------
# Trace input
# ---------------------------------------------------------------------------


def _finite(value: object, name: str, *, minimum: float | None = None) -> float:
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number):
        raise ResearchValidationError(
            "INVALID_TRACE_VALUE", f"{name} must be finite", path=name, actual=value
        )
    if minimum is not None and number < minimum:
        raise ResearchValidationError(
            "INVALID_TRACE_VALUE",
            f"{name} must be at least {minimum}",
            path=name,
            expected=minimum,
            actual=number,
        )
    return number


def _index(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResearchValidationError(
            "INVALID_TRACE_VALUE",
            f"{name} must be a nonnegative integer",
            path=name,
            actual=value,
        )
    return int(value)


@dataclass(frozen=True, slots=True)
class OfficerDecision:
    """One decision epoch of one officer.

    ``intersection_id`` is the decision intersection (the officer's decision
    point, as ``EncirclementPolice._decision`` defines it).
    ``selected_end_intersection_id`` is the end intersection of the chosen
    candidate, which equals ``intersection_id`` for a stay.
    ``distance_to_fugitive_m`` is measured at the start of the epoch, and
    ``displacement_m`` is the metric displacement realised during it.
    """

    step: int
    duration_s: float
    intersection_id: int
    action_index: int
    selected_end_intersection_id: int
    distance_to_fugitive_m: float
    displacement_m: float
    legal_move_available: bool
    goal_intersection_id: int | None = None

    def __post_init__(self) -> None:
        for name in ("step", "intersection_id", "action_index", "selected_end_intersection_id"):
            object.__setattr__(self, name, _index(getattr(self, name), name))
        object.__setattr__(self, "duration_s", _finite(self.duration_s, "duration_s", minimum=0.0))
        object.__setattr__(
            self,
            "distance_to_fugitive_m",
            _finite(self.distance_to_fugitive_m, "distance_to_fugitive_m", minimum=0.0),
        )
        object.__setattr__(
            self, "displacement_m", _finite(self.displacement_m, "displacement_m", minimum=0.0)
        )
        if not isinstance(self.legal_move_available, bool):
            raise ResearchValidationError(
                "INVALID_TRACE_VALUE",
                "legal_move_available must be a boolean",
                path="legal_move_available",
                actual=self.legal_move_available,
            )
        if self.goal_intersection_id is not None:
            object.__setattr__(
                self, "goal_intersection_id", _index(self.goal_intersection_id, "goal_intersection_id")
            )


@dataclass(frozen=True, slots=True)
class OfficerTrace:
    """One officer's decision sequence for one episode.

    At least two decisions are required: the anti-oscillation metrics are
    defined on decision *transitions*, so a shorter trace has no evaluable
    transition and would leave a zero-denominator rate behind.
    """

    officer_id: int
    decisions: tuple[OfficerDecision, ...]
    schema_version: str = BEHAVIOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        officer_id = _index(self.officer_id, "officer_id")
        if officer_id >= POLICE_COUNT:
            raise ResearchValidationError(
                "INVALID_OFFICER_ID",
                f"officer_id must be in 0..{POLICE_COUNT - 1}",
                path="officer_id",
                expected=POLICE_COUNT - 1,
                actual=officer_id,
            )
        object.__setattr__(self, "officer_id", officer_id)
        decisions = tuple(self.decisions)
        if any(not isinstance(item, OfficerDecision) for item in decisions):
            raise ResearchValidationError(
                "INVALID_TRACE_VALUE",
                "decisions must contain OfficerDecision entries",
                path="decisions",
            )
        if len(decisions) < 2:
            raise ResearchValidationError(
                "TRACE_TOO_SHORT",
                "an officer trace requires at least two decisions to have an evaluable transition",
                path="decisions",
                expected=2,
                actual=len(decisions),
            )
        steps = [item.step for item in decisions]
        if any(later <= earlier for earlier, later in zip(steps, steps[1:])):
            raise ResearchValidationError(
                "NONMONOTONIC_TRACE",
                "decision steps must be strictly increasing",
                path="decisions",
                actual=steps,
            )
        object.__setattr__(self, "decisions", decisions)

    @property
    def decision_count(self) -> int:
        return len(self.decisions)

    @property
    def content_hash(self) -> str:
        return content_hash(self)


# ---------------------------------------------------------------------------
# Per-officer metrics (Requirement 13.3-13.5)
# ---------------------------------------------------------------------------


def u_turn_rate(trace: OfficerTrace) -> float:
    """Fraction of decision transitions that reverse, per :data:`REVERSAL_RULE`."""
    decisions = trace.decisions
    reversals = sum(
        1
        for previous, current in zip(decisions, decisions[1:])
        if is_reversal(
            previous_intersection_id=previous.intersection_id,
            candidate_end_intersection_id=current.selected_end_intersection_id,
        )
    )
    return reversals / (len(decisions) - 1)


def revisit_rate(trace: OfficerTrace, *, window_decisions: int) -> float:
    """Fraction of decision transitions returning to an intersection seen in the last W decisions."""
    if isinstance(window_decisions, bool) or not isinstance(window_decisions, int) or window_decisions < 1:
        raise ResearchValidationError(
            "INVALID_METRIC_PROTOCOL",
            "window_decisions must be an integer of at least 1",
            path="window_decisions",
            actual=window_decisions,
        )
    decisions = trace.decisions
    revisits = 0
    for index in range(1, len(decisions)):
        history = decisions[max(0, index - window_decisions) : index]
        if any(item.intersection_id == decisions[index].intersection_id for item in history):
            revisits += 1
    return revisits / (len(decisions) - 1)


def action_switch_rate(trace: OfficerTrace) -> float:
    """Fraction of decision transitions where the selected action index changed."""
    decisions = trace.decisions
    switches = sum(
        1
        for previous, current in zip(decisions, decisions[1:])
        if previous.action_index != current.action_index
    )
    return switches / (len(decisions) - 1)


def role_switch_rate(trace: OfficerTrace) -> MetricValue:
    """Fraction of transitions where the designated goal intersection changed.

    An officer whose trace never records a goal target has no role concept to
    measure, which is not the same as never switching role.
    """
    decisions = trace.decisions
    if any(item.goal_intersection_id is None for item in decisions):
        return not_applicable_metric(
            UNIT_DIMENSIONLESS,
            "the trace records no goal/role target for at least one decision",
        )
    switches = sum(
        1
        for previous, current in zip(decisions, decisions[1:])
        if previous.goal_intersection_id != current.goal_intersection_id
    )
    return measured_metric(
        switches / (len(decisions) - 1),
        UNIT_DIMENSIONLESS,
        "designated goal intersection changed between consecutive decisions",
    )


def idle_rate(trace: OfficerTrace) -> float:
    """Fraction of decision epochs with a legal move available but zero displacement."""
    idle = sum(
        1 for item in trace.decisions if item.legal_move_available and item.displacement_m == 0.0
    )
    return idle / len(trace.decisions)


def approach_distance_m(trace: OfficerTrace) -> float:
    """Total distance closed on the fugitive, using the reward ``delta = old - new`` sign."""
    decisions = trace.decisions
    return sum(
        max(previous.distance_to_fugitive_m - current.distance_to_fugitive_m, 0.0)
        for previous, current in zip(decisions, decisions[1:])
    )


def retreat_distance_m(trace: OfficerTrace) -> float:
    """Total distance opened up from the fugitive, reported as a positive magnitude."""
    decisions = trace.decisions
    return sum(
        max(current.distance_to_fugitive_m - previous.distance_to_fugitive_m, 0.0)
        for previous, current in zip(decisions, decisions[1:])
    )


def zero_displacement_time_s(trace: OfficerTrace) -> float:
    """Total simulated time spent with exactly zero displacement."""
    return sum(item.duration_s for item in trace.decisions if item.displacement_m == 0.0)


def capture_radius_occupancy(
    trace: OfficerTrace, *, capture_radius_m: float
) -> tuple[int, float, int]:
    """Return ``(steps_inside, time_inside_s, entry_events)`` for the capture radius."""
    radius = _finite(capture_radius_m, "capture_radius_m", minimum=0.0)
    steps = 0
    time_s = 0.0
    events = 0
    previously_inside = False
    for item in trace.decisions:
        inside = item.distance_to_fugitive_m <= radius
        if inside:
            steps += 1
            time_s += item.duration_s
            if not previously_inside:
                events += 1
        previously_inside = inside
    return steps, time_s, events


# ---------------------------------------------------------------------------
# Map geometry helpers
# ---------------------------------------------------------------------------


def exit_intersection_ids(network: ModelNetwork) -> frozenset[int]:
    """Boundary/exit intersections of a Boundary_Escape map.

    An intersection is an exit when it carries a ``boundary_kind`` or is an
    endpoint of a segment that crosses the boundary.  ``make_interior_network``
    strips both markers, so an empty set is exactly an Interior_Contained map.
    """
    if not isinstance(network, ModelNetwork):
        raise ResearchValidationError(
            "INVALID_NETWORK", "A ModelNetwork instance is required", path="network"
        )
    ids = {item.id for item in network.intersections if item.boundary_kind is not None}
    for segment in network.segments:
        if segment.crosses_boundary:
            ids.add(segment.start_id)
            ids.add(segment.end_id)
    return frozenset(ids)


def is_boundary_scenario(network: ModelNetwork) -> bool:
    return bool(exit_intersection_ids(network))


def _positions(network: ModelNetwork) -> dict[int, tuple[float, float]]:
    return {item.id: item.position_xy for item in network.intersections}


_INTERIOR_REASON = "Interior_Contained map has no boundary exit to block"


def exit_block_duration_s(
    trace: OfficerTrace, *, network: ModelNetwork, exit_block_radius_m: float
) -> MetricValue:
    """Total time this officer sat within the blocking radius of any exit."""
    exits = exit_intersection_ids(network)
    if not exits:
        return not_applicable_metric(UNIT_SECOND, _INTERIOR_REASON)
    radius = _finite(exit_block_radius_m, "exit_block_radius_m", minimum=0.0)
    positions = _positions(network)
    exit_positions = [positions[node] for node in sorted(exits)]
    total = 0.0
    for item in trace.decisions:
        here = positions.get(item.intersection_id)
        if here is None:
            raise ResearchValidationError(
                "UNKNOWN_INTERSECTION",
                "trace references an intersection absent from the network",
                path="intersection_id",
                actual=item.intersection_id,
            )
        if any(math.dist(here, target) <= radius for target in exit_positions):
            total += item.duration_s
    return measured_metric(
        total,
        UNIT_SECOND,
        f"decision-epoch duration summed while within {radius} m of a boundary exit",
    )


# ---------------------------------------------------------------------------
# Containment geometry (Requirement 13.2, 13.6)
# ---------------------------------------------------------------------------


def _point(value: object, name: str) -> tuple[float, float]:
    try:
        x, y = value  # type: ignore[misc]
    except (TypeError, ValueError) as exc:
        raise ResearchValidationError(
            "INVALID_POSITION", f"{name} must be an (x, y) pair", path=name, actual=value
        ) from exc
    return (_finite(x, f"{name}.x"), _finite(y, f"{name}.y"))


def _team_positions(police_positions: Sequence[object]) -> tuple[tuple[float, float], ...]:
    positions = tuple(_point(item, "police_position") for item in police_positions)
    if len(positions) != POLICE_COUNT:
        raise ResearchValidationError(
            "INVALID_POLICE_COUNT",
            f"containment geometry requires exactly {POLICE_COUNT} officer positions",
            path="police_positions",
            expected=POLICE_COUNT,
            actual=len(positions),
        )
    return positions


@dataclass(frozen=True, slots=True)
class ContainmentAngles:
    """Rotation- and translation-invariant angular containment of the six officers."""

    bearings_rad: tuple[float, ...]
    sorted_gaps_rad: tuple[float, ...]
    max_angular_gap_rad: float
    angular_coverage: float
    schema_version: str = BEHAVIOR_SCHEMA_VERSION


def containment_angles(
    police_positions: Sequence[object], fugitive_position: object
) -> ContainmentAngles:
    """Angular coverage and maximum angular gap, using the audited 28D/encirclement formula."""
    positions = _team_positions(police_positions)
    fugitive = _point(fugitive_position, "fugitive_position")
    bearings = sorted(
        math.atan2(p[1] - fugitive[1], p[0] - fugitive[0]) for p in positions
    )
    gaps = tuple(
        (bearings[(index + 1) % len(bearings)] - bearings[index]) % (2 * math.pi)
        for index in range(len(bearings))
    )
    max_gap = max(gaps)
    return ContainmentAngles(
        bearings_rad=tuple(bearings),
        sorted_gaps_rad=gaps,
        max_angular_gap_rad=max_gap,
        angular_coverage=1.0 - (max_gap / (2 * math.pi)),
    )


def blocked_exit_fraction(
    *,
    network: ModelNetwork,
    police_positions: Sequence[object],
    exit_block_radius_m: float,
) -> MetricValue:
    """Fraction of the map's exits currently within the blocking radius of some officer."""
    exits = exit_intersection_ids(network)
    if not exits:
        return not_applicable_metric(UNIT_DIMENSIONLESS, _INTERIOR_REASON)
    positions = _team_positions(police_positions)
    radius = _finite(exit_block_radius_m, "exit_block_radius_m", minimum=0.0)
    node_positions = _positions(network)
    blocked = sum(
        1
        for node in sorted(exits)
        if any(math.dist(officer, node_positions[node]) <= radius for officer in positions)
    )
    return measured_metric(
        blocked / len(exits),
        UNIT_DIMENSIONLESS,
        f"exit is blocked when some officer is within {radius} m of it",
    )


def reachable_intersection_ids(
    network: ModelNetwork, source: int, *, removed: Iterable[int] = ()
) -> frozenset[int]:
    """Forward-reachable intersections over the directed segment graph.

    The adjacency is ``start_id -> end_id``, identical to the forward adjacency
    of ``policies.baselines._Graph``.  Removing the source itself leaves nothing
    reachable: the fugitive's own node is occupied.
    """
    if not isinstance(network, ModelNetwork):
        raise ResearchValidationError(
            "INVALID_NETWORK", "A ModelNetwork instance is required", path="network"
        )
    blocked = frozenset(_index(item, "removed") for item in removed)
    known = network.intersection_ids
    if source not in known:
        raise ResearchValidationError(
            "UNKNOWN_INTERSECTION",
            "source must be an intersection of the network",
            path="source",
            actual=source,
        )
    if source in blocked:
        return frozenset()
    adjacency: dict[int, list[int]] = {item.id: [] for item in network.intersections}
    for segment in network.segments:
        adjacency[segment.start_id].append(segment.end_id)
    seen = {source}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neighbour in adjacency[node]:
            if neighbour in blocked or neighbour in seen:
                continue
            seen.add(neighbour)
            queue.append(neighbour)
    return frozenset(seen)


@dataclass(frozen=True, slots=True)
class ReachableRegionReduction:
    """How much the fugitive's directed-reachable region shrank under containment."""

    baseline_reachable_count: int
    contained_reachable_count: int
    removed_intersection_ids: tuple[int, ...]
    schema_version: str = BEHAVIOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.baseline_reachable_count < 1:
            raise ResearchValidationError(
                "INVALID_REACHABLE_REGION",
                "the fugitive's own node is always reachable, so the baseline is at least 1",
                path="baseline_reachable_count",
                actual=self.baseline_reachable_count,
            )
        if not 0 <= self.contained_reachable_count <= self.baseline_reachable_count:
            raise ResearchValidationError(
                "INVALID_REACHABLE_REGION",
                "removing nodes can never grow the reachable region",
                path="contained_reachable_count",
                expected=self.baseline_reachable_count,
                actual=self.contained_reachable_count,
            )
        object.__setattr__(
            self, "removed_intersection_ids", tuple(sorted(set(self.removed_intersection_ids)))
        )

    @property
    def reduction_count(self) -> int:
        return self.baseline_reachable_count - self.contained_reachable_count

    @property
    def reduction_fraction(self) -> float:
        return self.reduction_count / self.baseline_reachable_count

    @property
    def metric_value(self) -> MetricValue:
        return measured_metric(
            self.reduction_fraction,
            UNIT_DIMENSIONLESS,
            "directed BFS reachability from the fugitive with officer-occupied nodes removed",
        )


def reachable_region_reduction(
    *,
    network: ModelNetwork,
    fugitive_intersection_id: int,
    occupied_intersection_ids: Iterable[int],
) -> ReachableRegionReduction:
    """Compare the fugitive's reachable set with and without officer-occupied nodes."""
    baseline = reachable_intersection_ids(network, fugitive_intersection_id)
    occupied = frozenset(_index(item, "occupied_intersection_ids") for item in occupied_intersection_ids)
    contained = reachable_intersection_ids(network, fugitive_intersection_id, removed=occupied)
    return ReachableRegionReduction(
        baseline_reachable_count=len(baseline),
        contained_reachable_count=len(contained),
        removed_intersection_ids=tuple(sorted(occupied)),
    )


@dataclass(frozen=True, slots=True)
class ContainmentGeometry:
    """The four team-level containment quantities of Requirement 13.6."""

    angles: ContainmentAngles
    blocked_exit_fraction: MetricValue
    reachable_region_reduction: ReachableRegionReduction
    schema_version: str = BEHAVIOR_SCHEMA_VERSION

    @property
    def angular_coverage(self) -> float:
        return self.angles.angular_coverage

    @property
    def max_angular_gap_rad(self) -> float:
        return self.angles.max_angular_gap_rad

    @property
    def content_hash(self) -> str:
        return content_hash(self)


def containment_geometry(
    *,
    network: ModelNetwork,
    police_positions: Sequence[object],
    fugitive_position: object,
    fugitive_intersection_id: int,
    occupied_intersection_ids: Iterable[int],
    protocol: BehaviorMetricProtocol,
) -> ContainmentGeometry:
    """Assemble angular coverage, maximum gap, blocked-exit fraction and reachable reduction."""
    return ContainmentGeometry(
        angles=containment_angles(police_positions, fugitive_position),
        blocked_exit_fraction=blocked_exit_fraction(
            network=network,
            police_positions=police_positions,
            exit_block_radius_m=protocol.exit_block_radius_m,
        ),
        reachable_region_reduction=reachable_region_reduction(
            network=network,
            fugitive_intersection_id=fugitive_intersection_id,
            occupied_intersection_ids=occupied_intersection_ids,
        ),
    )


# ---------------------------------------------------------------------------
# Leave-one-off contribution (Requirement 13.5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LeaveOneOffDifference:
    """One officer's contribution to a team metric under a pre-registered neutral replacement.

    Both operands are team-level metric values computed elsewhere; this schema
    only registers their labelled difference, so the counterfactual re-simulation
    stays the evaluation orchestrator's responsibility.
    """

    officer_id: int
    metric: str
    unit: str
    direction: MetricDirection
    neutral_replacement_rule: str
    with_officer_value: float
    neutral_replacement_value: float
    schema_version: str = BEHAVIOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        officer_id = _index(self.officer_id, "officer_id")
        if officer_id >= POLICE_COUNT:
            raise ResearchValidationError(
                "INVALID_OFFICER_ID",
                f"officer_id must be in 0..{POLICE_COUNT - 1}",
                path="officer_id",
                actual=officer_id,
            )
        object.__setattr__(self, "officer_id", officer_id)
        if self.neutral_replacement_rule not in NEUTRAL_REPLACEMENT_RULES:
            raise ResearchValidationError(
                "UNREGISTERED_NEUTRAL_REPLACEMENT",
                "the neutral replacement policy must be pre-registered",
                path="neutral_replacement_rule",
                expected=list(NEUTRAL_REPLACEMENT_RULES),
                actual=self.neutral_replacement_rule,
            )
        object.__setattr__(self, "direction", MetricDirection(self.direction))
        for name in ("with_officer_value", "neutral_replacement_value"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))

    @property
    def difference(self) -> float:
        return self.with_officer_value - self.neutral_replacement_value

    @property
    def officer_helps(self) -> bool:
        """Whether keeping this officer moves the metric in its better direction."""
        if self.direction is MetricDirection.HIGHER_IS_BETTER:
            return self.difference > 0.0
        return self.difference < 0.0

    @property
    def content_hash(self) -> str:
        return content_hash(self)


def leave_one_off_difference(
    *,
    officer_id: int,
    metric: str,
    with_officer_value: float,
    neutral_replacement_value: float,
    neutral_replacement_rule: str,
) -> LeaveOneOffDifference:
    """Register the with/without difference of a team metric for one officer."""
    spec = metric_spec(metric)
    return LeaveOneOffDifference(
        officer_id=officer_id,
        metric=spec.key,
        unit=spec.unit,
        direction=spec.direction,
        neutral_replacement_rule=neutral_replacement_rule,
        with_officer_value=with_officer_value,
        neutral_replacement_value=neutral_replacement_value,
    )


# ---------------------------------------------------------------------------
# Per-officer rows and six-row aggregation (Requirement 13.8)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OfficerBehaviorRow:
    """One officer's raw metric row; exactly six of these make an episode report."""

    officer_id: int
    decision_count: int
    u_turn_rate: float
    revisit_rate: float
    action_switch_rate: float
    role_switch_rate: MetricValue
    idle_rate: float
    approach_distance_m: float
    retreat_distance_m: float
    zero_displacement_time_s: float
    capture_radius_occupancy_steps: int
    capture_radius_occupancy_time_s: float
    capture_radius_entry_events: int
    exit_block_duration_s: MetricValue
    schema_version: str = BEHAVIOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for key in OFFICER_METRIC_KEYS:
            value = getattr(self, key)
            if isinstance(value, MetricValue):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ResearchValidationError(
                    "INVALID_METRIC_VALUE",
                    f"{key} must be a finite number or an explicit MetricValue",
                    path=key,
                    actual=value,
                )

    @property
    def content_hash(self) -> str:
        return content_hash(self)


def officer_behavior_row(
    trace: OfficerTrace, *, network: ModelNetwork, protocol: BehaviorMetricProtocol
) -> OfficerBehaviorRow:
    """Compute every per-officer metric of Requirement 13.3-13.5 for one trace."""
    if not isinstance(trace, OfficerTrace):
        raise ResearchValidationError(
            "INVALID_TRACE_VALUE", "An OfficerTrace instance is required", path="trace"
        )
    steps, time_s, events = capture_radius_occupancy(
        trace, capture_radius_m=protocol.capture_radius_m
    )
    return OfficerBehaviorRow(
        officer_id=trace.officer_id,
        decision_count=trace.decision_count,
        u_turn_rate=u_turn_rate(trace),
        revisit_rate=revisit_rate(trace, window_decisions=protocol.revisit_window_decisions),
        action_switch_rate=action_switch_rate(trace),
        role_switch_rate=role_switch_rate(trace),
        idle_rate=idle_rate(trace),
        approach_distance_m=approach_distance_m(trace),
        retreat_distance_m=retreat_distance_m(trace),
        zero_displacement_time_s=zero_displacement_time_s(trace),
        capture_radius_occupancy_steps=steps,
        capture_radius_occupancy_time_s=time_s,
        capture_radius_entry_events=events,
        exit_block_duration_s=exit_block_duration_s(
            trace, network=network, exit_block_radius_m=protocol.exit_block_radius_m
        ),
    )


def worst_officer(metric: str, values: Sequence[float]) -> tuple[int, float]:
    """Return ``(officer_id, value)`` of the worst officer for this metric's own direction.

    Lower-is-better metrics make the largest value worst and higher-is-better
    metrics make the smallest value worst; ties resolve to the lowest officer id
    so the selection is deterministic.
    """
    spec = metric_spec(metric)
    numbers = [_finite(item, metric) for item in values]
    if len(numbers) != POLICE_COUNT:
        raise ResearchValidationError(
            "INVALID_OFFICER_ROW_COUNT",
            f"exactly {POLICE_COUNT} officer values are required",
            path=metric,
            expected=POLICE_COUNT,
            actual=len(numbers),
        )
    indices = range(POLICE_COUNT)
    if spec.direction is MetricDirection.LOWER_IS_BETTER:
        index = max(indices, key=lambda i: (numbers[i], -i))
    else:
        index = min(indices, key=lambda i: (numbers[i], i))
    return index, numbers[index]


@dataclass(frozen=True, slots=True)
class MetricAggregate:
    """Team mean/median plus the direction-aware worst officer for one metric."""

    metric: str
    unit: str
    direction: MetricDirection
    availability: MetricAvailability
    values: tuple[float, ...] = ()
    mean: float | None = None
    median: float | None = None
    worst_officer_id: int | None = None
    worst_value: float | None = None
    method: str = ""
    schema_version: str = BEHAVIOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", MetricDirection(self.direction))
        object.__setattr__(self, "availability", MetricAvailability(self.availability))
        object.__setattr__(self, "values", tuple(self.values))
        if not isinstance(self.method, str) or not self.method.strip():
            raise ResearchValidationError(
                "MISSING_METRIC_METHOD",
                "every aggregate requires a computation method or an explicit reason",
                path="method",
            )
        stats = (self.mean, self.median, self.worst_officer_id, self.worst_value)
        if self.availability is MetricAvailability.MEASURED:
            if len(self.values) != POLICE_COUNT:
                raise ResearchValidationError(
                    "INVALID_OFFICER_ROW_COUNT",
                    f"a measured aggregate requires exactly {POLICE_COUNT} officer values",
                    path="values",
                    expected=POLICE_COUNT,
                    actual=len(self.values),
                )
            if any(item is None for item in stats):
                raise ResearchValidationError(
                    "INVALID_METRIC_VALUE",
                    "a measured aggregate requires mean, median and worst-officer values",
                    path=self.metric,
                )
        else:
            if self.values or any(item is not None for item in stats):
                raise ResearchValidationError(
                    "INVALID_METRIC_VALUE",
                    "only measured aggregates may carry values",
                    path=self.metric,
                    expected=None,
                    actual=self.values,
                )


def aggregate_officer_metric(metric: str, rows: Sequence[OfficerBehaviorRow]) -> MetricAggregate:
    """Aggregate one per-officer metric across exactly six rows."""
    spec = metric_spec(metric)
    if len(rows) != POLICE_COUNT:
        raise ResearchValidationError(
            "INVALID_OFFICER_ROW_COUNT",
            f"aggregation requires exactly {POLICE_COUNT} officer rows",
            path="rows",
            expected=POLICE_COUNT,
            actual=len(rows),
        )
    raw = [getattr(row, metric) for row in rows]
    availabilities = {
        item.availability for item in raw if isinstance(item, MetricValue)
    }
    if availabilities and availabilities != {MetricAvailability.MEASURED}:
        if len(availabilities) > 1:
            raise ResearchValidationError(
                "INCONSISTENT_METRIC_AVAILABILITY",
                "the six officers disagree on whether this metric applies",
                path=metric,
                actual=sorted(item.value for item in availabilities),
            )
        availability = next(iter(availabilities))
        reason = next(item.method for item in raw if isinstance(item, MetricValue))
        return MetricAggregate(
            metric=spec.key,
            unit=spec.unit,
            direction=spec.direction,
            availability=availability,
            method=reason,
        )
    values = tuple(
        float(item.value) if isinstance(item, MetricValue) else float(item) for item in raw
    )
    index, value = worst_officer(metric, values)
    return MetricAggregate(
        metric=spec.key,
        unit=spec.unit,
        direction=spec.direction,
        availability=MetricAvailability.MEASURED,
        values=values,
        mean=statistics.fmean(values),
        median=statistics.median(values),
        worst_officer_id=index,
        worst_value=value,
        method=f"six-officer aggregate; worst officer selected by {spec.direction.value}",
    )


@dataclass(frozen=True, slots=True)
class OfficerBehaviorReport:
    """Six raw officer rows plus their team aggregates (Requirement 13.8)."""

    rows: tuple[OfficerBehaviorRow, ...]
    aggregates: Mapping[str, MetricAggregate] = field(default_factory=dict)
    protocol_hash: str = ""
    schema_version: str = BEHAVIOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        rows = tuple(self.rows)
        if len(rows) != POLICE_COUNT:
            raise ResearchValidationError(
                "INVALID_OFFICER_ROW_COUNT",
                f"an episode report requires exactly {POLICE_COUNT} officer rows",
                path="rows",
                expected=POLICE_COUNT,
                actual=len(rows),
            )
        if [row.officer_id for row in rows] != list(range(POLICE_COUNT)):
            raise ResearchValidationError(
                "INVALID_OFFICER_ROW_COUNT",
                f"officer rows must cover ids 0..{POLICE_COUNT - 1} exactly once, in order",
                path="rows",
                expected=list(range(POLICE_COUNT)),
                actual=[row.officer_id for row in rows],
            )
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "aggregates", MappingProxyType(dict(self.aggregates)))

    @property
    def content_hash(self) -> str:
        return content_hash(self)


def episode_behavior_report(
    traces: Sequence[OfficerTrace],
    *,
    network: ModelNetwork,
    protocol: BehaviorMetricProtocol,
) -> OfficerBehaviorReport:
    """Compute the six officer rows and every registered per-officer aggregate."""
    if len(traces) != POLICE_COUNT:
        raise ResearchValidationError(
            "INVALID_OFFICER_ROW_COUNT",
            f"an episode requires exactly {POLICE_COUNT} officer traces",
            path="traces",
            expected=POLICE_COUNT,
            actual=len(traces),
        )
    rows = tuple(
        officer_behavior_row(trace, network=network, protocol=protocol)
        for trace in sorted(traces, key=lambda item: item.officer_id)
    )
    return OfficerBehaviorReport(
        rows=rows,
        aggregates={key: aggregate_officer_metric(key, rows) for key in OFFICER_METRIC_KEYS},
        protocol_hash=protocol.protocol_hash,
    )


__all__ = (
    "BEHAVIOR_SCHEMA_VERSION",
    "DEFAULT_EXIT_BLOCK_RADIUS_M",
    "DEFAULT_REVISIT_WINDOW_DECISIONS",
    "METRIC_REGISTRY",
    "NEUTRAL_REPLACEMENT_RULES",
    "OFFICER_METRIC_KEYS",
    "UNIT_COUNT",
    "UNIT_DIMENSIONLESS",
    "UNIT_METRE",
    "UNIT_RADIAN",
    "UNIT_SECOND",
    "BehaviorMetricProtocol",
    "ContainmentAngles",
    "ContainmentGeometry",
    "LeaveOneOffDifference",
    "MetricAggregate",
    "MetricAvailability",
    "MetricDirection",
    "MetricSpec",
    "MetricValue",
    "OfficerBehaviorReport",
    "OfficerBehaviorRow",
    "OfficerDecision",
    "OfficerTrace",
    "ReachableRegionReduction",
    "action_switch_rate",
    "aggregate_officer_metric",
    "approach_distance_m",
    "blocked_exit_fraction",
    "capture_radius_occupancy",
    "containment_angles",
    "containment_geometry",
    "episode_behavior_report",
    "exit_block_duration_s",
    "exit_intersection_ids",
    "idle_rate",
    "is_boundary_scenario",
    "leave_one_off_difference",
    "measured_metric",
    "metric_spec",
    "not_applicable_metric",
    "not_measured_metric",
    "officer_behavior_row",
    "reachable_intersection_ids",
    "reachable_region_reduction",
    "retreat_distance_m",
    "revisit_rate",
    "role_switch_rate",
    "u_turn_rate",
    "worst_officer",
    "zero_displacement_time_s",
)

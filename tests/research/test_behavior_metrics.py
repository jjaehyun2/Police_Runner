"""Task 7.2 regressions for six-officer behavior, containment and contribution metrics."""
from __future__ import annotations

import math

import pytest

from pursuit_evasion_rl.osm_demo.metrics import placement_position
from pursuit_evasion_rl.osm_demo.models import (
    POLICE_COUNT,
    EpisodeConfig,
    Intersection,
    ModelNetwork,
    Segment,
    VehiclePlacement,
)
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.variants.observations import (
    AUGMENTED_FIELD_ORDER,
    Observation28DAdapter,
)
from pursuit_evasion_rl.research.metrics.behavior import (
    METRIC_REGISTRY,
    OFFICER_METRIC_KEYS,
    BehaviorMetricProtocol,
    MetricAvailability,
    MetricDirection,
    OfficerBehaviorReport,
    OfficerDecision,
    OfficerTrace,
    aggregate_officer_metric,
    blocked_exit_fraction,
    containment_angles,
    containment_geometry,
    episode_behavior_report,
    exit_intersection_ids,
    leave_one_off_difference,
    officer_behavior_row,
    reachable_intersection_ids,
    reachable_region_reduction,
    worst_officer,
)

pytestmark = pytest.mark.offline

_GRID_SIZE = 4
_SPACING = 100.0
_EXIT_NODE = 15
_CAPTURE_RADIUS_M = 30.0
_EXIT_BLOCK_RADIUS_M = 50.0


def _grid_network(*, boundary_nodes: tuple[int, ...] = ()) -> ModelNetwork:
    """Bidirectional 4x4 grid; node ``r*4+c`` sits at ``(c*100, r*100)`` metres."""
    positions = {
        row * _GRID_SIZE + col: (col * _SPACING, row * _SPACING)
        for row in range(_GRID_SIZE)
        for col in range(_GRID_SIZE)
    }
    segments: list[Segment] = []
    for row in range(_GRID_SIZE):
        for col in range(_GRID_SIZE):
            node = row * _GRID_SIZE + col
            neighbours = []
            if col + 1 < _GRID_SIZE:
                neighbours.append(node + 1)
            if row + 1 < _GRID_SIZE:
                neighbours.append(node + _GRID_SIZE)
            for neighbour in neighbours:
                for start, end in ((node, neighbour), (neighbour, node)):
                    segments.append(
                        Segment(
                            id=len(segments),
                            start_id=start,
                            end_id=end,
                            length_m=_SPACING,
                            geometry_xy=(positions[start], positions[end]),
                        )
                    )
    intersections = tuple(
        Intersection(
            id=node,
            position_xy=positions[node],
            source_signature=f"grid-{node}",
            boundary_kind="exit" if node in boundary_nodes else None,
            outgoing_segment_ids=tuple(item.id for item in segments if item.start_id == node),
            incoming_segment_ids=tuple(item.id for item in segments if item.end_id == node),
        )
        for node in range(_GRID_SIZE * _GRID_SIZE)
    )
    return ModelNetwork(intersections=intersections, segments=tuple(segments))


def _boundary_network() -> ModelNetwork:
    return _grid_network(boundary_nodes=(_EXIT_NODE,))


def _interior_network() -> ModelNetwork:
    return _grid_network()


def _protocol(**overrides) -> BehaviorMetricProtocol:
    kwargs = {
        "capture_radius_m": _CAPTURE_RADIUS_M,
        "revisit_window_decisions": 5,
        "exit_block_radius_m": _EXIT_BLOCK_RADIUS_M,
    }
    kwargs.update(overrides)
    return BehaviorMetricProtocol(**kwargs)


# ---------------------------------------------------------------------------
# The hand-computed six-officer trace
#
# Officer 0 is worked out by hand below; officers 1..5 walk distinct grid nodes
# in a straight line, so none of them reverses or revisits.
# ---------------------------------------------------------------------------

_OFFICER_0_NODES = (0, 1, 0, 1)
_OFFICER_0_ENDS = (1, 0, 1, 1)
_OFFICER_0_ACTIONS = (0, 1, 1, 2)
_OFFICER_0_GOALS = (10, 10, 11, 11)
_OFFICER_0_DISTANCES = (300.0, 200.0, 250.0, 240.0)
_OFFICER_0_DISPLACEMENTS = (100.0, 100.0, 100.0, 0.0)

_STRAIGHT_NODES = {
    1: (4, 5, 6, 7),
    2: (8, 9, 10, 11),
    3: (1, 2, 3, 7),
    4: (12, 13, 14, 15),
    5: (2, 6, 10, 14),
}
_STRAIGHT_DISTANCES = {
    1: (100.0, 25.0, 20.0, 60.0),
    2: (400.0, 380.0, 360.0, 340.0),
    3: (500.0, 450.0, 400.0, 350.0),
    4: (250.0, 240.0, 230.0, 220.0),
    5: (10.0, 40.0, 15.0, 10.0),
}
_DURATION_S = 2.0


def _officer_0_trace() -> OfficerTrace:
    return OfficerTrace(
        officer_id=0,
        decisions=tuple(
            OfficerDecision(
                step=step,
                duration_s=_DURATION_S,
                intersection_id=_OFFICER_0_NODES[step],
                action_index=_OFFICER_0_ACTIONS[step],
                selected_end_intersection_id=_OFFICER_0_ENDS[step],
                distance_to_fugitive_m=_OFFICER_0_DISTANCES[step],
                displacement_m=_OFFICER_0_DISPLACEMENTS[step],
                legal_move_available=True,
                goal_intersection_id=_OFFICER_0_GOALS[step],
            )
            for step in range(4)
        ),
    )


def _straight_trace(officer_id: int) -> OfficerTrace:
    nodes = _STRAIGHT_NODES[officer_id]
    ends = (*nodes[1:], nodes[-1])
    distances = _STRAIGHT_DISTANCES[officer_id]
    return OfficerTrace(
        officer_id=officer_id,
        decisions=tuple(
            OfficerDecision(
                step=step,
                duration_s=_DURATION_S,
                intersection_id=nodes[step],
                action_index=0,
                selected_end_intersection_id=ends[step],
                distance_to_fugitive_m=distances[step],
                displacement_m=100.0,
                legal_move_available=True,
                goal_intersection_id=nodes[-1],
            )
            for step in range(4)
        ),
    )


def _episode_traces() -> tuple[OfficerTrace, ...]:
    return (_officer_0_trace(), *(_straight_trace(k) for k in range(1, POLICE_COUNT)))


# ---------------------------------------------------------------------------
# 1. Hand trace vs reference computation (Requirement 13.3-13.5)
# ---------------------------------------------------------------------------


def test_hand_traced_officer_row_matches_reference_computation() -> None:
    row = officer_behavior_row(
        _officer_0_trace(), network=_boundary_network(), protocol=_protocol()
    )

    # Reversals at decisions 1 (0 -> 1 -> back to 0) and 2 (1 -> 0 -> back to 1).
    assert row.u_turn_rate == pytest.approx(2 / 3)
    # Node 0 recurs at decision 2 and node 1 at decision 3, within the 5-decision window.
    assert row.revisit_rate == pytest.approx(2 / 3)
    # Action indices 0,1,1,2 change at two of the three transitions.
    assert row.action_switch_rate == pytest.approx(2 / 3)
    # Goal 10,10,11,11 changes once out of three transitions.
    assert row.role_switch_rate.availability is MetricAvailability.MEASURED
    assert row.role_switch_rate.value == pytest.approx(1 / 3)
    # Only decision 3 has a legal move available and zero displacement.
    assert row.idle_rate == pytest.approx(1 / 4)
    # delta = old - new over 300,200,250,240 gives +100, -50, +10.
    assert row.approach_distance_m == pytest.approx(110.0)
    assert row.retreat_distance_m == pytest.approx(50.0)
    assert row.zero_displacement_time_s == pytest.approx(2.0)
    # Every distance stays outside the 30 m capture radius.
    assert row.capture_radius_occupancy_steps == 0
    assert row.capture_radius_occupancy_time_s == pytest.approx(0.0)
    assert row.capture_radius_entry_events == 0
    assert row.decision_count == 4


def test_capture_radius_entry_counts_occupancy_and_transitions() -> None:
    network = _boundary_network()
    protocol = _protocol()

    # Distances 100, 25, 20, 60: inside for two consecutive decisions, one entry.
    first = officer_behavior_row(_straight_trace(1), network=network, protocol=protocol)
    assert first.capture_radius_occupancy_steps == 2
    assert first.capture_radius_occupancy_time_s == pytest.approx(4.0)
    assert first.capture_radius_entry_events == 1

    # Distances 10, 40, 15, 10: inside at decisions 0, 2 and 3, so two entries.
    second = officer_behavior_row(_straight_trace(5), network=network, protocol=protocol)
    assert second.capture_radius_occupancy_steps == 3
    assert second.capture_radius_occupancy_time_s == pytest.approx(6.0)
    assert second.capture_radius_entry_events == 2


def test_revisit_window_shortening_drops_out_of_window_returns() -> None:
    trace = _officer_0_trace()
    wide = officer_behavior_row(trace, network=_boundary_network(), protocol=_protocol())
    narrow = officer_behavior_row(
        trace, network=_boundary_network(), protocol=_protocol(revisit_window_decisions=1)
    )

    # With W=1 only the immediately preceding intersection counts, and the
    # officer alternates 0,1,0,1 so neither return is within one decision.
    assert wide.revisit_rate == pytest.approx(2 / 3)
    assert narrow.revisit_rate == pytest.approx(0.0)


def test_episode_report_aggregates_match_hand_computed_team_values() -> None:
    report = episode_behavior_report(
        _episode_traces(), network=_boundary_network(), protocol=_protocol()
    )

    assert len(report.rows) == POLICE_COUNT
    assert [row.officer_id for row in report.rows] == list(range(POLICE_COUNT))
    assert set(report.aggregates) == set(OFFICER_METRIC_KEYS)

    u_turn = report.aggregates["u_turn_rate"]
    assert u_turn.values == pytest.approx((2 / 3, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert u_turn.mean == pytest.approx(1 / 9)
    assert u_turn.median == pytest.approx(0.0)
    assert u_turn.worst_officer_id == 0
    assert u_turn.worst_value == pytest.approx(2 / 3)

    approach = report.aggregates["approach_distance_m"]
    assert approach.values == pytest.approx((110.0, 80.0, 60.0, 150.0, 30.0, 30.0))
    assert approach.mean == pytest.approx(460.0 / 6)
    assert approach.median == pytest.approx(70.0)
    # Higher is better, so the smallest approach is worst; officers 4 and 5 tie
    # at 30 m and the tie resolves to the lower officer id.
    assert approach.worst_officer_id == 4
    assert approach.worst_value == pytest.approx(30.0)

    # Only officer 4 stands on the exit node, and only at its last decision.
    exit_block = report.aggregates["exit_block_duration_s"]
    assert exit_block.availability is MetricAvailability.MEASURED
    assert exit_block.values == pytest.approx((0.0, 0.0, 0.0, 0.0, 2.0, 0.0))
    assert exit_block.worst_officer_id == 0
    assert exit_block.worst_value == pytest.approx(0.0)

    assert report.protocol_hash == _protocol().protocol_hash
    assert report.content_hash == episode_behavior_report(
        _episode_traces(), network=_boundary_network(), protocol=_protocol()
    ).content_hash


def test_every_registered_metric_declares_unit_direction_and_formula() -> None:
    assert set(OFFICER_METRIC_KEYS) <= set(METRIC_REGISTRY)
    for key, spec in METRIC_REGISTRY.items():
        assert spec.key == key
        assert spec.symbol and spec.formula and spec.time_window
        assert spec.unit in {"m", "s", "rad", "1", "count"}
        assert isinstance(spec.direction, MetricDirection)
        assert spec.aggregation_level in {"per_officer", "team"}
        assert spec.practical_threshold.unit == spec.unit


def test_trace_shorter_than_two_decisions_is_rejected() -> None:
    single = _officer_0_trace().decisions[:1]
    with pytest.raises(ResearchValidationError) as excinfo:
        OfficerTrace(officer_id=0, decisions=single)
    assert excinfo.value.code == "TRACE_TOO_SHORT"


# ---------------------------------------------------------------------------
# 2. Containment geometry: bounds plus rotation/translation invariance
# ---------------------------------------------------------------------------


def _ring(angles_deg: tuple[float, ...], radius: float = 100.0) -> tuple[tuple[float, float], ...]:
    return tuple(
        (radius * math.cos(math.radians(a)), radius * math.sin(math.radians(a)))
        for a in angles_deg
    )


def test_evenly_spaced_officers_match_hand_computed_coverage() -> None:
    angles = containment_angles(_ring((0.0, 60.0, 120.0, 180.0, 240.0, 300.0)), (0.0, 0.0))

    assert angles.max_angular_gap_rad == pytest.approx(math.pi / 3)
    assert angles.angular_coverage == pytest.approx(5 / 6)
    assert sum(angles.sorted_gaps_rad) == pytest.approx(2 * math.pi)


def test_clustered_officers_leave_a_hand_computed_wrap_around_gap() -> None:
    angles = containment_angles(_ring((0.0, 10.0, 20.0, 30.0, 40.0, 50.0)), (0.0, 0.0))

    # Five 10-degree gaps plus the 310-degree wrap-around gap.
    assert angles.max_angular_gap_rad == pytest.approx(math.radians(310.0))
    assert angles.angular_coverage == pytest.approx(50.0 / 360.0)


def test_angular_coverage_matches_the_audited_28d_observation_feature() -> None:
    network = _interior_network()
    police = tuple(VehiclePlacement(intersection_id=node) for node in (0, 3, 12, 15, 1, 7))
    fugitive = VehiclePlacement(intersection_id=5)

    adapter = Observation28DAdapter(network)
    audited = float(adapter._extra(0, police, fugitive)[AUGMENTED_FIELD_ORDER.index("angular_coverage")])
    computed = containment_angles(
        tuple(placement_position(network, item) for item in police),
        placement_position(network, fugitive),
    ).angular_coverage

    # The adapter stores its features as float32, so only that rounding differs.
    assert computed == pytest.approx(audited, abs=1e-7)
    assert computed != pytest.approx(0.0)


@pytest.mark.parametrize(
    "angles_deg",
    [
        (0.0, 60.0, 120.0, 180.0, 240.0, 300.0),
        (0.0, 10.0, 20.0, 30.0, 40.0, 50.0),
        (5.0, 87.0, 91.0, 200.0, 201.0, 349.0),
    ],
)
def test_containment_angles_are_rotation_and_translation_invariant(
    angles_deg: tuple[float, ...]
) -> None:
    police = _ring(angles_deg)
    fugitive = (0.0, 0.0)
    theta = 0.7317
    offset = (123.4, -56.78)

    def moved(point: tuple[float, float]) -> tuple[float, float]:
        x = point[0] * math.cos(theta) - point[1] * math.sin(theta) + offset[0]
        y = point[0] * math.sin(theta) + point[1] * math.cos(theta) + offset[1]
        return (x, y)

    original = containment_angles(police, fugitive)
    transformed = containment_angles(tuple(moved(p) for p in police), moved(fugitive))

    assert transformed.angular_coverage == pytest.approx(original.angular_coverage, abs=1e-12)
    assert transformed.max_angular_gap_rad == pytest.approx(
        original.max_angular_gap_rad, abs=1e-12
    )
    assert 0.0 <= original.angular_coverage <= 1.0
    assert 0.0 <= original.max_angular_gap_rad <= 2 * math.pi


def test_containment_angles_require_exactly_six_officers() -> None:
    for count in (5, 7):
        with pytest.raises(ResearchValidationError) as excinfo:
            containment_angles(_ring(tuple(60.0 * k for k in range(count))), (0.0, 0.0))
        assert excinfo.value.code == "INVALID_POLICE_COUNT"


def test_reachable_region_reduction_is_monotone_and_hand_checkable() -> None:
    network = _interior_network()

    assert len(reachable_intersection_ids(network, 5)) == 16

    # Removing every neighbour of node 5 leaves the fugitive with only its own node.
    sealed = reachable_region_reduction(
        network=network, fugitive_intersection_id=5, occupied_intersection_ids=(1, 4, 6, 9)
    )
    assert sealed.baseline_reachable_count == 16
    assert sealed.contained_reachable_count == 1
    assert sealed.reduction_count == 15
    assert sealed.reduction_fraction == pytest.approx(15 / 16)
    assert sealed.metric_value.availability is MetricAvailability.MEASURED

    partial = reachable_region_reduction(
        network=network, fugitive_intersection_id=5, occupied_intersection_ids=(1, 4)
    )
    assert partial.contained_reachable_count >= sealed.contained_reachable_count
    assert partial.reduction_fraction <= sealed.reduction_fraction

    none_removed = reachable_region_reduction(
        network=network, fugitive_intersection_id=5, occupied_intersection_ids=()
    )
    assert none_removed.reduction_fraction == pytest.approx(0.0)


def test_containment_geometry_composes_angles_exits_and_reachability() -> None:
    network = _boundary_network()
    police = ((300.0, 300.0), (0.0, 0.0), (300.0, 0.0), (0.0, 300.0), (100.0, 0.0), (0.0, 100.0))
    geometry = containment_geometry(
        network=network,
        police_positions=police,
        fugitive_position=(100.0, 100.0),
        fugitive_intersection_id=5,
        occupied_intersection_ids=(1, 4, 6, 9),
        protocol=_protocol(),
    )

    assert geometry.angular_coverage == pytest.approx(
        containment_angles(police, (100.0, 100.0)).angular_coverage
    )
    assert 0.0 <= geometry.max_angular_gap_rad <= 2 * math.pi
    # The single exit at node 15 == (300, 300) is occupied by the first officer.
    assert geometry.blocked_exit_fraction.availability is MetricAvailability.MEASURED
    assert geometry.blocked_exit_fraction.value == pytest.approx(1.0)
    assert geometry.reachable_region_reduction.reduction_fraction == pytest.approx(15 / 16)


# ---------------------------------------------------------------------------
# 3. Exactly six officer rows (Requirement 13.8 / Correctness Property 23)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("count", [5, 7])
def test_aggregation_rejects_officer_row_counts_other_than_six(count: int) -> None:
    report = episode_behavior_report(
        _episode_traces(), network=_boundary_network(), protocol=_protocol()
    )
    rows = report.rows[:count] if count < POLICE_COUNT else (*report.rows, report.rows[-1])

    with pytest.raises(ResearchValidationError) as aggregate_error:
        aggregate_officer_metric("u_turn_rate", rows)
    assert aggregate_error.value.code == "INVALID_OFFICER_ROW_COUNT"

    with pytest.raises(ResearchValidationError) as report_error:
        OfficerBehaviorReport(rows=rows)
    assert report_error.value.code == "INVALID_OFFICER_ROW_COUNT"


@pytest.mark.parametrize("count", [5, 7])
def test_episode_report_rejects_trace_counts_other_than_six(count: int) -> None:
    traces = _episode_traces()
    supplied = traces[:count] if count < POLICE_COUNT else (*traces, traces[-1])

    with pytest.raises(ResearchValidationError) as excinfo:
        episode_behavior_report(supplied, network=_boundary_network(), protocol=_protocol())
    assert excinfo.value.code == "INVALID_OFFICER_ROW_COUNT"


def test_report_rejects_six_rows_that_do_not_cover_every_officer() -> None:
    rows = episode_behavior_report(
        _episode_traces(), network=_boundary_network(), protocol=_protocol()
    ).rows
    duplicated = (*rows[:5], rows[4])

    with pytest.raises(ResearchValidationError) as excinfo:
        OfficerBehaviorReport(rows=duplicated)
    assert excinfo.value.code == "INVALID_OFFICER_ROW_COUNT"


def test_worst_officer_requires_six_values() -> None:
    with pytest.raises(ResearchValidationError) as excinfo:
        worst_officer("u_turn_rate", (0.1, 0.2, 0.3, 0.4, 0.5))
    assert excinfo.value.code == "INVALID_OFFICER_ROW_COUNT"


# ---------------------------------------------------------------------------
# 4. Worst-officer selection follows each metric's own direction
# ---------------------------------------------------------------------------


def test_worst_officer_respects_per_metric_direction() -> None:
    values = (0.1, 0.5, 0.2, 0.3, 0.4, 0.0)

    assert METRIC_REGISTRY["u_turn_rate"].direction is MetricDirection.LOWER_IS_BETTER
    assert worst_officer("u_turn_rate", values) == (1, pytest.approx(0.5))

    assert METRIC_REGISTRY["approach_distance_m"].direction is MetricDirection.HIGHER_IS_BETTER
    assert worst_officer("approach_distance_m", values) == (5, pytest.approx(0.0))


def test_worst_officer_ties_resolve_to_the_lowest_officer_id() -> None:
    tied = (0.4, 0.4, 0.1, 0.1, 0.2, 0.2)

    assert worst_officer("retreat_distance_m", tied)[0] == 0  # lower is better -> max value
    assert worst_officer("capture_radius_occupancy_time_s", tied)[0] == 2  # higher is better -> min


def test_worst_officer_rejects_an_unregistered_metric() -> None:
    with pytest.raises(ResearchValidationError) as excinfo:
        worst_officer("made_up_metric", (0.0,) * POLICE_COUNT)
    assert excinfo.value.code == "UNREGISTERED_METRIC"


# ---------------------------------------------------------------------------
# 5. Interior vs Boundary: not-applicable is distinguishable from a real zero
# ---------------------------------------------------------------------------


def test_interior_exit_block_is_not_applicable_while_boundary_zero_is_measured() -> None:
    protocol = _protocol()
    trace = _straight_trace(1)  # never stands on the exit node

    interior = officer_behavior_row(trace, network=_interior_network(), protocol=protocol)
    boundary = officer_behavior_row(trace, network=_boundary_network(), protocol=protocol)

    assert exit_intersection_ids(_interior_network()) == frozenset()
    assert exit_intersection_ids(_boundary_network()) == frozenset({_EXIT_NODE})

    assert interior.exit_block_duration_s.availability is MetricAvailability.NOT_APPLICABLE
    assert interior.exit_block_duration_s.value is None
    assert interior.exit_block_duration_s.method

    assert boundary.exit_block_duration_s.availability is MetricAvailability.MEASURED
    assert boundary.exit_block_duration_s.value == pytest.approx(0.0)

    assert interior.exit_block_duration_s != boundary.exit_block_duration_s


def test_interior_aggregate_and_blocked_exit_fraction_stay_not_applicable() -> None:
    report = episode_behavior_report(
        _episode_traces(), network=_interior_network(), protocol=_protocol()
    )
    aggregate = report.aggregates["exit_block_duration_s"]

    assert aggregate.availability is MetricAvailability.NOT_APPLICABLE
    assert aggregate.values == ()
    assert aggregate.mean is None and aggregate.median is None
    assert aggregate.worst_officer_id is None
    assert aggregate.method

    fraction = blocked_exit_fraction(
        network=_interior_network(),
        police_positions=_ring((0.0, 60.0, 120.0, 180.0, 240.0, 300.0)),
        exit_block_radius_m=_EXIT_BLOCK_RADIUS_M,
    )
    assert fraction.availability is MetricAvailability.NOT_APPLICABLE
    assert fraction.value is None


def test_boundary_blocked_exit_fraction_reports_a_measured_zero_when_nobody_blocks() -> None:
    fraction = blocked_exit_fraction(
        network=_boundary_network(),
        police_positions=((0.0, 0.0),) * POLICE_COUNT,
        exit_block_radius_m=_EXIT_BLOCK_RADIUS_M,
    )

    assert fraction.availability is MetricAvailability.MEASURED
    assert fraction.value == pytest.approx(0.0)


def test_role_switch_rate_is_not_applicable_without_a_recorded_goal() -> None:
    decisions = tuple(
        OfficerDecision(
            step=step,
            duration_s=_DURATION_S,
            intersection_id=step,
            action_index=0,
            selected_end_intersection_id=step + 1,
            distance_to_fugitive_m=100.0,
            displacement_m=100.0,
            legal_move_available=True,
        )
        for step in range(3)
    )
    row = officer_behavior_row(
        OfficerTrace(officer_id=0, decisions=decisions),
        network=_boundary_network(),
        protocol=_protocol(),
    )

    assert row.role_switch_rate.availability is MetricAvailability.NOT_APPLICABLE
    assert row.role_switch_rate.value is None


# ---------------------------------------------------------------------------
# Leave-one-off contribution (Requirement 13.5)
# ---------------------------------------------------------------------------


def test_leave_one_off_difference_labels_officer_metric_and_neutral_rule() -> None:
    difference = leave_one_off_difference(
        officer_id=3,
        metric="angular_coverage",
        with_officer_value=0.83,
        neutral_replacement_value=0.61,
        neutral_replacement_rule="stationary",
    )

    assert difference.officer_id == 3
    assert difference.metric == "angular_coverage"
    assert difference.unit == "1"
    assert difference.direction is MetricDirection.HIGHER_IS_BETTER
    assert difference.neutral_replacement_rule == "stationary"
    assert difference.difference == pytest.approx(0.22)
    assert difference.officer_helps is True
    assert difference.content_hash


def test_leave_one_off_direction_flips_for_a_lower_is_better_metric() -> None:
    difference = leave_one_off_difference(
        officer_id=2,
        metric="max_angular_gap_rad",
        with_officer_value=1.2,
        neutral_replacement_value=2.5,
        neutral_replacement_rule="uniform_random",
    )

    assert difference.difference == pytest.approx(-1.3)
    assert difference.officer_helps is True


def test_leave_one_off_rejects_an_unregistered_neutral_replacement() -> None:
    with pytest.raises(ResearchValidationError) as excinfo:
        leave_one_off_difference(
            officer_id=0,
            metric="angular_coverage",
            with_officer_value=0.5,
            neutral_replacement_value=0.4,
            neutral_replacement_rule="whatever_helps",
        )
    assert excinfo.value.code == "UNREGISTERED_NEUTRAL_REPLACEMENT"


def test_protocol_binds_the_episode_capture_radius() -> None:
    config = EpisodeConfig(
        dt_s=1.0,
        police_speed_mps=14.0,
        fugitive_speed_mps=10.0,
        capture_radius_m=42.0,
        max_steps=100,
    )
    protocol = BehaviorMetricProtocol.from_episode_config(config)

    assert protocol.capture_radius_m == pytest.approx(42.0)
    assert protocol.protocol_hash != _protocol().protocol_hash

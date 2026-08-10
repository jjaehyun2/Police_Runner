"""Tests for the T2 safety-reward terms and the T6 tech/safety/ops KPI table.

Track B of the KIPoT mentoring feedback: explicit safety terms in the reward
(``research.variants.safety``) and a 3-axis KPI aggregation
(``research.metrics.kpi``).  Fixtures build tiny hand-specified raw OSM graphs
directly (rather than reusing ``osm_demo.fixtures.daejeon``, which tags every
edge ``road_class="secondary"`` and therefore carries no risk-class segment at
all) so risk-class detection and the herding penalty have something to bite on.
"""

from __future__ import annotations

import math

import pytest

from pursuit_evasion_rl.osm_demo.coarsening import coarsen_raw_graph, prepare_model_network
from pursuit_evasion_rl.osm_demo.metrics import EpisodeOutcome
from pursuit_evasion_rl.osm_demo.models import (
    POLICE_COUNT,
    DomainValidationError,
    EpisodeState,
    RawEdge,
    RawNode,
    RawOSMGraph,
    Segment,
    VehiclePlacement,
)
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.metrics.kpi import (
    ContainmentCompletion,
    build_kpi_report,
    compute_containment_completion,
)
from pursuit_evasion_rl.research.variants.safety import (
    RISK_ROAD_CLASSES,
    SafetyPenaltyConfig,
    SafetyPenalties,
    segment_is_risk_class,
    traversed_metres,
    wrap_step_reward,
)

pytestmark = pytest.mark.offline

_LAT, _LON = 36.35, 127.40


# ---------------------------------------------------------------------------
# Fixture: a 3-node star (hub -> residential leaf, hub -> secondary leaf).
# Leaves have no outgoing edge, so (unlike a straight chain) neither collapses
# into the hub's segment during coarsening -- each stays its own intersection,
# the same shape as osm_demo.fixtures.degree_six's one-way spokes.
# ---------------------------------------------------------------------------


def _star_network():
    positions = {"hub": (0.0, 0.0), "r_leaf": (100.0, 0.0), "s_leaf": (-100.0, 0.0)}
    nodes = tuple(RawNode(name, _LAT, _LON, x, y, {}) for name, (x, y) in positions.items())

    def edge(u: str, v: str, key: str, road_class: str) -> RawEdge:
        start, end = positions[u], positions[v]
        return RawEdge(
            u, v, key,
            length_m=math.dist(start, end),
            geometry_xy=(start, end),
            way_refs=(key,),
            oneway=True,
            road_class=road_class,
            tags={},
        )

    edges = (
        edge("hub", "r_leaf", "hub-r", "residential"),
        edge("hub", "s_leaf", "hub-s", "secondary"),
    )
    raw = RawOSMGraph(nodes=nodes, edges=edges)
    prepared = prepare_model_network(coarsen_raw_graph(raw))
    ids = prepared.mapping.raw_node_to_intersection
    return prepared.network, ids["hub"], ids["r_leaf"], ids["s_leaf"]


def _filler_placement(offset: int) -> VehiclePlacement:
    """A parked placement far outside the star's id range -- inert for BFS."""
    return VehiclePlacement(intersection_id=100_000 + offset)


def _filler_police(count: int = POLICE_COUNT) -> tuple[VehiclePlacement, ...]:
    return tuple(_filler_placement(index) for index in range(count))


# ---------------------------------------------------------------------------
# segment_is_risk_class -- including "|"-joined multi-class strings
# ---------------------------------------------------------------------------


def _segment(road_class: str | None) -> Segment:
    attributes = {} if road_class is None else {"road_class": road_class}
    return Segment(
        id=0, start_id=0, end_id=1, length_m=10.0,
        geometry_xy=((0.0, 0.0), (10.0, 0.0)),
        attributes=attributes,
    )


@pytest.mark.parametrize(
    "road_class,expected",
    [
        ("residential", True),
        ("living_street", True),
        ("secondary", False),
        ("primary", False),
        ("residential|primary", True),  # any joined component in the risk set
        ("primary|residential", True),  # order does not matter
        ("primary|trunk", False),
        (None, False),  # missing road_class entirely
        ("", False),
    ],
)
def test_segment_is_risk_class(road_class: str | None, expected: bool) -> None:
    assert segment_is_risk_class(_segment(road_class)) is expected


def test_risk_road_classes_is_the_documented_pair() -> None:
    assert RISK_ROAD_CLASSES == frozenset({"residential", "living_street"})


# ---------------------------------------------------------------------------
# traversed_metres -- the three attribution cases
# ---------------------------------------------------------------------------


class _FakeGraph:
    """Minimal stand-in exposing only the ``.seg`` mapping traversed_metres reads."""

    def __init__(self, segment: Segment) -> None:
        self.seg = {segment.id: segment}


def test_traversed_metres_same_segment_uses_progress_delta() -> None:
    segment = Segment(id=0, start_id=10, end_id=11, length_m=200.0, geometry_xy=((0.0, 0.0), (200.0, 0.0)))
    graph = _FakeGraph(segment)
    before = VehiclePlacement(segment_id=0, progress=0.25)
    after = VehiclePlacement(segment_id=0, progress=0.75)
    segment_id, metres = traversed_metres(before, after, graph)
    assert segment_id == 0
    assert metres == pytest.approx(0.5 * 200.0)


def test_traversed_metres_segment_exit_uses_remaining_progress() -> None:
    segment = Segment(id=0, start_id=10, end_id=11, length_m=200.0, geometry_xy=((0.0, 0.0), (200.0, 0.0)))
    graph = _FakeGraph(segment)
    before = VehiclePlacement(segment_id=0, progress=0.6)
    after = VehiclePlacement(intersection_id=11)  # arrived at the segment's end
    segment_id, metres = traversed_metres(before, after, graph)
    assert segment_id == 0
    assert metres == pytest.approx(0.4 * 200.0)


def test_traversed_metres_parked_is_zero_on_no_segment() -> None:
    segment = Segment(id=0, start_id=10, end_id=11, length_m=200.0, geometry_xy=((0.0, 0.0), (200.0, 0.0)))
    graph = _FakeGraph(segment)
    before = VehiclePlacement(intersection_id=10)  # parked the whole step
    after = VehiclePlacement(intersection_id=11)
    segment_id, metres = traversed_metres(before, after, graph)
    assert segment_id is None
    assert metres == 0.0


# ---------------------------------------------------------------------------
# SafetyPenaltyConfig validation
# ---------------------------------------------------------------------------


def test_safety_penalty_config_rejects_negative_parameter() -> None:
    with pytest.raises(DomainValidationError):
        SafetyPenaltyConfig(risk_traversal_per_100m=-0.01)


def test_safety_penalty_config_rejects_non_bool_flag() -> None:
    with pytest.raises(DomainValidationError):
        SafetyPenaltyConfig(herding_enabled="yes")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# SafetyPenalties.risk_traversal_penalty -- per-officer, risk-class only
# ---------------------------------------------------------------------------


def test_risk_traversal_penalty_charges_only_the_risk_class_traveler() -> None:
    network, hub_id, r_leaf_id, s_leaf_id = _star_network()
    residential_seg = next(s for s in network.segments if s.start_id == hub_id and s.end_id == r_leaf_id)
    secondary_seg = next(s for s in network.segments if s.start_id == hub_id and s.end_id == s_leaf_id)

    police_before = list(_filler_police())
    police_after = list(_filler_police())
    # Officer 0 fully traverses the residential leaf (risk-class).
    police_before[0] = VehiclePlacement(segment_id=residential_seg.id, progress=0.0)
    police_after[0] = VehiclePlacement(intersection_id=r_leaf_id)
    # Officer 1 fully traverses the secondary leaf (non-risk).
    police_before[1] = VehiclePlacement(segment_id=secondary_seg.id, progress=0.0)
    police_after[1] = VehiclePlacement(intersection_id=s_leaf_id)

    before = EpisodeState(step=0, simulated_s=0.0, police=tuple(police_before), fugitive=VehiclePlacement(intersection_id=hub_id))
    after = EpisodeState(step=1, simulated_s=1.0, police=tuple(police_after), fugitive=VehiclePlacement(intersection_id=hub_id))

    safety = SafetyPenalties(network, SafetyPenaltyConfig())
    penalties = safety.risk_traversal_penalty(before, after)

    assert len(penalties) == POLICE_COUNT
    assert penalties[0] == pytest.approx(0.02 * (residential_seg.length_m / 100.0))
    assert penalties[1] == pytest.approx(0.0)
    assert all(value >= 0.0 for value in penalties)
    assert all(value == pytest.approx(0.0) for value in penalties[2:])


def test_risk_traversal_penalty_disabled_flag_zeroes_all_officers() -> None:
    network, hub_id, r_leaf_id, _ = _star_network()
    residential_seg = next(s for s in network.segments if s.start_id == hub_id and s.end_id == r_leaf_id)
    police_before = list(_filler_police())
    police_after = list(_filler_police())
    police_before[0] = VehiclePlacement(segment_id=residential_seg.id, progress=0.0)
    police_after[0] = VehiclePlacement(intersection_id=r_leaf_id)
    before = EpisodeState(step=0, simulated_s=0.0, police=tuple(police_before), fugitive=VehiclePlacement(intersection_id=hub_id))
    after = EpisodeState(step=1, simulated_s=1.0, police=tuple(police_after), fugitive=VehiclePlacement(intersection_id=hub_id))

    safety = SafetyPenalties(network, SafetyPenaltyConfig(risk_traversal_enabled=False))
    assert safety.risk_traversal_penalty(before, after) == (0.0,) * POLICE_COUNT


# ---------------------------------------------------------------------------
# SafetyPenalties.herding_penalty -- blocking the safe exit raises risk-fraction
# ---------------------------------------------------------------------------


def test_herding_penalty_rises_when_officers_close_the_safe_exit() -> None:
    network, hub_id, r_leaf_id, s_leaf_id = _star_network()

    fugitive = VehiclePlacement(intersection_id=hub_id)
    police_before = list(_filler_police())
    police_after = list(_filler_police())
    # Officer 0 starts parked off-map (no exit blocked) and moves to sit on
    # the secondary (non-risk) leaf, closing off the fugitive's only safe exit.
    police_after[0] = VehiclePlacement(intersection_id=s_leaf_id)

    before = EpisodeState(step=0, simulated_s=0.0, police=tuple(police_before), fugitive=fugitive)
    after = EpisodeState(step=1, simulated_s=1.0, police=tuple(police_after), fugitive=fugitive)

    safety = SafetyPenalties(network, SafetyPenaltyConfig())
    penalty = safety.herding_penalty(before, after)

    # hub and r_leaf are both risk-incident (they touch the residential leaf);
    # s_leaf is not.  Before: {hub, r_leaf, s_leaf} -> 2/3 risky.  After
    # (s_leaf blocked): {hub, r_leaf} -> 2/2 risky.  Delta = 1/3.
    expected = (1.0 - (2.0 / 3.0)) * SafetyPenaltyConfig().herding_risk_weight
    assert penalty == pytest.approx(expected)
    assert penalty > 0.0


def test_herding_penalty_is_zero_when_disabled() -> None:
    network, hub_id, _, s_leaf_id = _star_network()
    fugitive = VehiclePlacement(intersection_id=hub_id)
    police_before = list(_filler_police())
    police_after = list(_filler_police())
    police_after[0] = VehiclePlacement(intersection_id=s_leaf_id)
    before = EpisodeState(step=0, simulated_s=0.0, police=tuple(police_before), fugitive=fugitive)
    after = EpisodeState(step=1, simulated_s=1.0, police=tuple(police_after), fugitive=fugitive)

    safety = SafetyPenalties(network, SafetyPenaltyConfig(herding_enabled=False))
    assert safety.herding_penalty(before, after) == 0.0


def test_herding_penalty_guards_division_by_zero_when_fugitive_node_is_blocked() -> None:
    """An officer parked exactly on the fugitive's own node empties the reachable
    set on both sides of the step; the zero-division guard must return 0.0
    rather than raising."""
    network, hub_id, _, _ = _star_network()
    fugitive = VehiclePlacement(intersection_id=hub_id)
    police = list(_filler_police())
    police[0] = VehiclePlacement(intersection_id=hub_id)  # occupies the fugitive's node
    state = EpisodeState(step=0, simulated_s=0.0, police=tuple(police), fugitive=fugitive)

    safety = SafetyPenalties(network, SafetyPenaltyConfig())
    assert safety.herding_penalty(state, state) == 0.0


# ---------------------------------------------------------------------------
# SafetyPenalties.__call__ / wrap_step_reward -- shape and subtraction
# ---------------------------------------------------------------------------


def test_safety_penalties_call_returns_one_nonnegative_value_per_officer() -> None:
    network, hub_id, r_leaf_id, s_leaf_id = _star_network()
    fugitive = VehiclePlacement(intersection_id=hub_id)
    police_before = list(_filler_police())
    police_after = list(_filler_police())
    police_after[0] = VehiclePlacement(intersection_id=s_leaf_id)
    before = EpisodeState(step=0, simulated_s=0.0, police=tuple(police_before), fugitive=fugitive)
    after = EpisodeState(step=1, simulated_s=1.0, police=tuple(police_after), fugitive=fugitive)

    safety = SafetyPenalties(network, SafetyPenaltyConfig())
    totals = safety(before, after)

    assert len(totals) == POLICE_COUNT
    assert all(value >= 0.0 for value in totals)
    herd_share = safety.herding_penalty(before, after) / POLICE_COUNT
    traversal = safety.risk_traversal_penalty(before, after)
    for total, own in zip(totals, traversal):
        assert total == pytest.approx(own + herd_share)


def test_wrap_step_reward_subtracts_penalties_and_preserves_length() -> None:
    network, hub_id, _, s_leaf_id = _star_network()
    fugitive = VehiclePlacement(intersection_id=hub_id)
    police_before = list(_filler_police())
    police_after = list(_filler_police())
    police_after[0] = VehiclePlacement(intersection_id=s_leaf_id)
    before = EpisodeState(step=0, simulated_s=0.0, police=tuple(police_before), fugitive=fugitive)
    after = EpisodeState(step=1, simulated_s=1.0, police=tuple(police_after), fugitive=fugitive)

    safety = SafetyPenalties(network, SafetyPenaltyConfig())

    def base_reward_fn(network_arg, before_arg, after_arg, captured_arg):
        assert captured_arg is False
        return (1.0,) * POLICE_COUNT

    wrapped = wrap_step_reward(base_reward_fn, safety)
    result = wrapped(network, before, after, False)
    expected_penalties = safety(before, after)

    assert len(result) == POLICE_COUNT
    for value, base, penalty in zip(result, (1.0,) * POLICE_COUNT, expected_penalties):
        assert value == pytest.approx(base - penalty)


def test_wrap_step_reward_rejects_non_safety_penalties_argument() -> None:
    with pytest.raises(DomainValidationError):
        wrap_step_reward(lambda *args: (0.0,) * POLICE_COUNT, object())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# KPI: compute_containment_completion -- threshold logic + zero-division guard
# ---------------------------------------------------------------------------


def _ring_positions(radius: float, degrees: list[float]) -> tuple[tuple[float, float], ...]:
    return tuple(
        (radius * math.cos(math.radians(deg)), radius * math.sin(math.radians(deg)))
        for deg in degrees
    )


def test_compute_containment_completion_threshold_logic() -> None:
    fugitive = (0.0, 0.0)
    # Evenly spread: max angular gap is 60 deg -> coverage = 1 - 60/360 = 5/6 ~= 0.833.
    even = _ring_positions(50.0, [0, 60, 120, 180, 240, 300])
    # Clustered within a 25 deg arc -> coverage well under the default 0.75 threshold.
    clustered = _ring_positions(50.0, [0, 5, 10, 15, 20, 25])

    result = compute_containment_completion(
        [
            [(even, fugitive)],       # trace form: reaches threshold
            [(clustered, fugitive)],  # trace form: never reaches threshold
            True,                     # precomputed bool form
            False,                    # precomputed bool form
        ]
    )

    assert isinstance(result, ContainmentCompletion)
    assert result.per_episode == (True, False, True, False)
    assert result.rate == pytest.approx(0.5)
    assert result.threshold == pytest.approx(0.75)


def test_compute_containment_completion_reached_partway_through_trace_counts() -> None:
    fugitive = (0.0, 0.0)
    clustered = _ring_positions(50.0, [0, 5, 10, 15, 20, 25])
    even = _ring_positions(50.0, [0, 60, 120, 180, 240, 300])
    # Formation tightens on the second recorded step -- "ever reached" must count it.
    result = compute_containment_completion([[(clustered, fugitive), (even, fugitive)]])
    assert result.per_episode == (True,)


def test_compute_containment_completion_empty_trace_list_guards_zero_division() -> None:
    result = compute_containment_completion([])
    assert result.per_episode == ()
    assert result.rate == 0.0


def test_compute_containment_completion_rejects_invalid_threshold() -> None:
    with pytest.raises(ResearchValidationError):
        compute_containment_completion([True], threshold=1.5)


# ---------------------------------------------------------------------------
# KPI: build_kpi_report -- assembly, TTC captured-only, fail-closed inputs
# ---------------------------------------------------------------------------


def test_build_kpi_report_assembles_all_three_axes() -> None:
    outcomes = [EpisodeOutcome.CAPTURE, EpisodeOutcome.CAPTURE, EpisodeOutcome.ESCAPE, EpisodeOutcome.TIMEOUT]
    step_counts = [10, 20, 5, 8]
    report = build_kpi_report(
        outcomes=outcomes,
        step_counts=step_counts,
        dt_s=2.0,
        risk_traversal_m_per_episode=[12.0, 8.0, 0.0, 4.0],
        latency_summary={"mean_ms": 3.2},
        containment_completions=[True, False, True, False],
        risk_zone_fraction_final=[0.4, 0.6, 0.2, 0.5],
    )

    assert report.n_episodes == 4
    assert report.tech.capture_rate == pytest.approx(0.5)
    # Captured episodes only: (10*2 + 20*2) / 2 = 30.
    assert report.tech.mean_time_to_capture_s == pytest.approx(30.0)
    assert report.tech.containment_completion_rate == pytest.approx(0.5)
    assert report.tech.mean_inference_latency_ms == pytest.approx(3.2)
    assert report.safety.risk_class_traversal_m_per_episode == pytest.approx(6.0)
    assert report.safety.risk_zone_fraction_final == pytest.approx(0.425)
    assert report.ops.notes.strip()  # non-empty, no fabricated ops metric


def test_build_kpi_report_time_to_capture_is_none_without_captures() -> None:
    report = build_kpi_report(
        outcomes=[EpisodeOutcome.ESCAPE, EpisodeOutcome.TIMEOUT],
        step_counts=[5, 5],
        dt_s=1.0,
        risk_traversal_m_per_episode=[0.0, 0.0],
        containment_completions=[False, False],
    )
    assert report.tech.capture_rate == pytest.approx(0.0)
    assert report.tech.mean_time_to_capture_s is None
    assert report.tech.mean_inference_latency_ms is None
    assert report.safety.risk_zone_fraction_final is None


def test_build_kpi_report_accepts_containment_traces_instead_of_booleans() -> None:
    fugitive = (0.0, 0.0)
    even = _ring_positions(50.0, [0, 60, 120, 180, 240, 300])
    report = build_kpi_report(
        outcomes=[EpisodeOutcome.CAPTURE],
        step_counts=[3],
        dt_s=1.0,
        risk_traversal_m_per_episode=[0.0],
        containment_traces=[[(even, fugitive)]],
    )
    assert report.tech.containment_completion_rate == pytest.approx(1.0)


def test_build_kpi_report_rejects_ambiguous_containment_input() -> None:
    with pytest.raises(ResearchValidationError):
        build_kpi_report(
            outcomes=[EpisodeOutcome.CAPTURE],
            step_counts=[1],
            dt_s=1.0,
            risk_traversal_m_per_episode=[0.0],
            containment_completions=[True],
            containment_traces=[[]],
        )
    with pytest.raises(ResearchValidationError):
        build_kpi_report(
            outcomes=[EpisodeOutcome.CAPTURE],
            step_counts=[1],
            dt_s=1.0,
            risk_traversal_m_per_episode=[0.0],
        )


def test_build_kpi_report_rejects_length_mismatch() -> None:
    with pytest.raises(ResearchValidationError):
        build_kpi_report(
            outcomes=[EpisodeOutcome.CAPTURE, EpisodeOutcome.ESCAPE],
            step_counts=[1, 2],
            dt_s=1.0,
            risk_traversal_m_per_episode=[0.0],  # only one entry for two episodes
            containment_completions=[True, False],
        )


def test_build_kpi_report_rejects_empty_outcomes() -> None:
    with pytest.raises(ResearchValidationError):
        build_kpi_report(
            outcomes=[],
            step_counts=[],
            dt_s=1.0,
            risk_traversal_m_per_episode=[],
            containment_completions=[],
        )

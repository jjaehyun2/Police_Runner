"""Property 21 coverage: per-officer behavioral metrics equal reference trace computations."""

from __future__ import annotations

from hypothesis import given, settings, strategies as st
import pytest

from pursuit_evasion_rl.research.metrics.behavior import (
    OfficerDecision,
    OfficerTrace,
    action_switch_rate,
    approach_distance_m,
    capture_radius_occupancy,
    idle_rate,
    retreat_distance_m,
    revisit_rate,
    u_turn_rate,
    zero_displacement_time_s,
)

# **Property 21: Per-officer behavioral metrics equal reference trace computations**
# **Validates: Requirements 13.1-13.4, 19.4**

pytestmark = pytest.mark.offline

_PBT_SETTINGS = settings(max_examples=100, deadline=None)


@st.composite
def officer_traces(draw: st.DrawFn) -> OfficerTrace:
    n = draw(st.integers(min_value=2, max_value=10))
    decisions = []
    for step in range(n):
        decisions.append(
            OfficerDecision(
                step=step,
                duration_s=draw(st.floats(min_value=0.0, max_value=30.0, allow_nan=False)),
                intersection_id=draw(st.integers(min_value=0, max_value=4)),
                action_index=draw(st.integers(min_value=0, max_value=3)),
                selected_end_intersection_id=draw(st.integers(min_value=0, max_value=4)),
                distance_to_fugitive_m=draw(st.floats(min_value=0.0, max_value=2000.0, allow_nan=False)),
                displacement_m=draw(st.sampled_from([0.0, 0.0, 0.0])) if draw(st.booleans()) else draw(st.floats(min_value=0.0, max_value=200.0, allow_nan=False)),
                legal_move_available=draw(st.booleans()),
                goal_intersection_id=None,
            )
        )
    return OfficerTrace(officer_id=draw(st.integers(min_value=0, max_value=5)), decisions=tuple(decisions))


@_PBT_SETTINGS
@given(trace=officer_traces())
def test_u_turn_rate_matches_a_reference_reversal_count(trace: OfficerTrace) -> None:
    decisions = trace.decisions
    reversals = sum(
        1 for previous, current in zip(decisions, decisions[1:])
        if current.selected_end_intersection_id == previous.intersection_id
    )
    expected = reversals / (len(decisions) - 1)
    assert u_turn_rate(trace) == pytest.approx(expected)


@_PBT_SETTINGS
@given(trace=officer_traces(), window=st.integers(min_value=1, max_value=6))
def test_revisit_rate_matches_a_reference_window_scan(trace: OfficerTrace, window: int) -> None:
    decisions = trace.decisions
    revisits = 0
    for index in range(1, len(decisions)):
        seen = {item.intersection_id for item in decisions[max(0, index - window):index]}
        if decisions[index].intersection_id in seen:
            revisits += 1
    expected = revisits / (len(decisions) - 1)
    assert revisit_rate(trace, window_decisions=window) == pytest.approx(expected)


@_PBT_SETTINGS
@given(trace=officer_traces())
def test_action_switch_rate_matches_a_reference_change_count(trace: OfficerTrace) -> None:
    decisions = trace.decisions
    switches = sum(1 for a, b in zip(decisions, decisions[1:]) if a.action_index != b.action_index)
    expected = switches / (len(decisions) - 1)
    assert action_switch_rate(trace) == pytest.approx(expected)


@_PBT_SETTINGS
@given(trace=officer_traces())
def test_idle_rate_matches_a_reference_legal_zero_displacement_count(trace: OfficerTrace) -> None:
    idle = sum(1 for item in trace.decisions if item.legal_move_available and item.displacement_m == 0.0)
    expected = idle / len(trace.decisions)
    assert idle_rate(trace) == pytest.approx(expected)
    assert 0.0 <= idle_rate(trace) <= 1.0


@_PBT_SETTINGS
@given(trace=officer_traces())
def test_approach_and_retreat_distance_match_a_reference_signed_delta_split(trace: OfficerTrace) -> None:
    decisions = trace.decisions
    reference_approach = sum(max(a.distance_to_fugitive_m - b.distance_to_fugitive_m, 0.0) for a, b in zip(decisions, decisions[1:]))
    reference_retreat = sum(max(b.distance_to_fugitive_m - a.distance_to_fugitive_m, 0.0) for a, b in zip(decisions, decisions[1:]))
    assert approach_distance_m(trace) == pytest.approx(reference_approach)
    assert retreat_distance_m(trace) == pytest.approx(reference_retreat)
    assert approach_distance_m(trace) >= 0.0
    assert retreat_distance_m(trace) >= 0.0
    # Net displacement in distance-to-fugitive terms is exactly approach - retreat.
    net_reference = decisions[0].distance_to_fugitive_m - decisions[-1].distance_to_fugitive_m
    assert approach_distance_m(trace) - retreat_distance_m(trace) == pytest.approx(net_reference)


@_PBT_SETTINGS
@given(trace=officer_traces())
def test_zero_displacement_time_matches_a_reference_duration_sum(trace: OfficerTrace) -> None:
    expected = sum(item.duration_s for item in trace.decisions if item.displacement_m == 0.0)
    assert zero_displacement_time_s(trace) == pytest.approx(expected)
    assert zero_displacement_time_s(trace) >= 0.0


@_PBT_SETTINGS
@given(trace=officer_traces(), radius=st.floats(min_value=0.0, max_value=2000.0, allow_nan=False))
def test_capture_radius_occupancy_matches_a_reference_inside_run_scan(trace: OfficerTrace, radius: float) -> None:
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
    result = capture_radius_occupancy(trace, capture_radius_m=radius)
    assert result == (steps, pytest.approx(time_s), events)
    assert result[0] <= trace.decision_count
    assert result[2] <= result[0]

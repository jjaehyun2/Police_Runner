"""Property 24 coverage: physical violations are complete and fail visibly."""

from __future__ import annotations

from hypothesis import given, settings, strategies as st
import pytest

from pursuit_evasion_rl.osm_demo.models import Intersection, ModelNetwork, Segment, VehiclePlacement
from pursuit_evasion_rl.research.metrics.physical import (
    VIOLATION_CATEGORIES,
    TraceTransition,
    ViolationCategory,
    validate_episode_trace,
)

# **Property 24: Physical violations are complete and fail visibly**
# **Validates: Requirements 13.7, 13.9, 19.4**

pytestmark = pytest.mark.offline

_PBT_SETTINGS = settings(max_examples=100, deadline=None)

BUDGET_M = 4.0
SEGMENT_LENGTH_M = 10.0


def _loop_network() -> ModelNetwork:
    segments = (
        Segment(0, 0, 1, SEGMENT_LENGTH_M, ((0.0, 0.0), (10.0, 0.0))),
        Segment(1, 1, 0, SEGMENT_LENGTH_M, ((10.0, 0.0), (0.0, 0.0))),
    )
    intersections = (
        Intersection(0, (0.0, 0.0), "node-0", outgoing_segment_ids=(0,), incoming_segment_ids=(1,)),
        Intersection(1, (10.0, 0.0), "node-1", outgoing_segment_ids=(1,), incoming_segment_ids=(0,)),
    )
    return ModelNetwork(intersections, segments)


_NETWORK = _loop_network()


def _intersection(identifier: int) -> VehiclePlacement:
    return VehiclePlacement(intersection_id=identifier)


def _on(segment_id: int, progress: float) -> VehiclePlacement:
    return VehiclePlacement(segment_id=segment_id, progress=progress)


def _transition(step_index, officer_id, before, after, *, action=0, mask=(True, True)) -> TraceTransition:
    return TraceTransition(
        step_index=step_index, officer_id=officer_id, before=before, after=after,
        distance_budget_m=BUDGET_M, executed_action=action, action_mask=mask,
    )


def _one_loop(offset: int, officer_id: int) -> list[TraceTransition]:
    """One full directed round trip, transitions numbered from ``offset``."""
    return [
        _transition(offset + 0, officer_id, _intersection(0), _on(0, 0.0)),
        _transition(offset + 1, officer_id, _on(0, 0.0), _on(0, 0.4)),
        _transition(offset + 2, officer_id, _on(0, 0.4), _on(0, 0.8)),
        _transition(offset + 3, officer_id, _on(0, 0.8), _intersection(1)),
        _transition(offset + 4, officer_id, _intersection(1), _on(1, 0.0)),
        _transition(offset + 5, officer_id, _on(1, 0.0), _on(1, 0.4)),
        _transition(offset + 6, officer_id, _on(1, 0.4), _on(1, 0.8)),
        _transition(offset + 7, officer_id, _on(1, 0.8), _intersection(0)),
    ]


def _valid_trace(repeats: int, officer_id: int) -> list[TraceTransition]:
    trace: list[TraceTransition] = []
    for repeat in range(repeats):
        trace.extend(_one_loop(repeat * 8, officer_id))
    return trace


def _fault(category: ViolationCategory, offset: int, officer_id: int) -> list[TraceTransition]:
    if category is ViolationCategory.DIRECTION_VIOLATION:
        return [_transition(offset, officer_id, _intersection(1), _on(0, 0.0))]
    if category is ViolationCategory.CONTRAFLOW:
        return [_transition(offset, officer_id, _on(0, 0.4), _on(0, 0.1))]
    if category is ViolationCategory.OFF_ROAD:
        return [_transition(offset, officer_id, _on(0, 0.4), _on(5, 0.0))]
    if category is ViolationCategory.SPEED_LIMIT_VIOLATION:
        return [_transition(offset, officer_id, _on(0, 0.4), _on(0, 0.9))]
    if category is ViolationCategory.TELEPORT:
        return [_transition(offset, officer_id, _on(0, 0.4), _on(1, 0.0))]
    if category is ViolationCategory.DISCONTINUOUS_SEGMENT_TRANSITION:
        return [_transition(offset, officer_id, _intersection(1), _on(1, 0.1))]
    if category is ViolationCategory.IMPOSSIBLE_ROUND_TRIP:
        return [_transition(offset, officer_id, _on(0, 0.0), _intersection(0))]
    if category is ViolationCategory.INVALID_ACTION:
        return [_transition(offset, officer_id, _on(0, 0.4), _on(0, 0.8), action=1, mask=(True, False))]
    raise AssertionError(f"unhandled category {category}")


# Each fault must attach after a prefix that leaves the officer at the state
# the fault expects; only CONTRAFLOW/OFF_ROAD/SPEED_LIMIT/TELEPORT/INVALID_ACTION
# start from a mid-segment position (after >=2 valid steps), while
# DIRECTION_VIOLATION/DISCONTINUOUS start from an intersection arrival
# (after >=4 valid steps), and IMPOSSIBLE_ROUND_TRIP starts from a fresh departure.
_PREFIX_STEPS = {
    ViolationCategory.DIRECTION_VIOLATION: 4,
    ViolationCategory.CONTRAFLOW: 2,
    ViolationCategory.OFF_ROAD: 2,
    ViolationCategory.SPEED_LIMIT_VIOLATION: 2,
    ViolationCategory.TELEPORT: 2,
    ViolationCategory.DISCONTINUOUS_SEGMENT_TRANSITION: 4,
    ViolationCategory.IMPOSSIBLE_ROUND_TRIP: 1,
    ViolationCategory.INVALID_ACTION: 2,
}


def _faulted_trace(category: ViolationCategory, leading_loops: int, officer_id: int) -> list[TraceTransition]:
    lead = _valid_trace(leading_loops, officer_id)
    offset = leading_loops * 8
    prefix_len = _PREFIX_STEPS[category]
    prefix = _one_loop(offset, officer_id)[:prefix_len]
    fault = _fault(category, offset + prefix_len, officer_id)
    return lead + prefix + fault


@_PBT_SETTINGS
@given(repeats=st.integers(min_value=0, max_value=4), officer_id=st.integers(min_value=0, max_value=5))
def test_a_valid_randomized_length_trace_always_has_zero_violations(repeats: int, officer_id: int) -> None:
    trace = _valid_trace(repeats, officer_id)
    if not trace:
        return
    report = validate_episode_trace(_NETWORK, trace, episode_id="episode-valid")
    assert set(report.counts) == set(VIOLATION_CATEGORIES)
    assert all(report.count(category) == 0 for category in VIOLATION_CATEGORIES)
    assert report.episode_failed is False
    assert report.failure is None


@_PBT_SETTINGS
@given(
    category=st.sampled_from(VIOLATION_CATEGORIES),
    leading_loops=st.integers(min_value=0, max_value=2),
    officer_id=st.integers(min_value=0, max_value=5),
)
def test_each_injected_fault_increments_only_its_own_category_regardless_of_leading_valid_loops(
    category: ViolationCategory, leading_loops: int, officer_id: int
) -> None:
    trace = _faulted_trace(category, leading_loops, officer_id)
    report = validate_episode_trace(_NETWORK, trace, episode_id=f"episode-{category.value}-{leading_loops}")

    assert report.count(category) == 1
    assert all(report.count(other) == 0 for other in VIOLATION_CATEGORIES if other != category)
    assert report.episode_failed is True
    assert report.failure is not None
    assert len(report.failure.cause_trace_hash) == 64
    assert dict(report.failure.counts)[category.value] == 1

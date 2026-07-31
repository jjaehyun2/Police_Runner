"""Task 7.3 physical plausibility validator and failure-ledger regressions."""

from __future__ import annotations

import math

import pytest

from pursuit_evasion_rl.osm_demo.models import (
    Intersection,
    ModelNetwork,
    Segment,
    VehiclePlacement,
)
from pursuit_evasion_rl.research.metrics.physical import (
    VIOLATION_CATEGORIES,
    TraceTransition,
    ViolationCategory,
    account_intention_to_evaluate,
    validate_episode_trace,
)

pytestmark = pytest.mark.offline

BUDGET_M = 4.0
SEGMENT_LENGTH_M = 10.0


def _loop_network() -> ModelNetwork:
    """Two intersections joined by one directed segment in each direction."""
    segments = (
        Segment(0, 0, 1, SEGMENT_LENGTH_M, ((0.0, 0.0), (10.0, 0.0))),
        Segment(1, 1, 0, SEGMENT_LENGTH_M, ((10.0, 0.0), (0.0, 0.0))),
    )
    intersections = (
        Intersection(0, (0.0, 0.0), "node-0", outgoing_segment_ids=(0,), incoming_segment_ids=(1,)),
        Intersection(1, (10.0, 0.0), "node-1", outgoing_segment_ids=(1,), incoming_segment_ids=(0,)),
    )
    return ModelNetwork(intersections, segments)


def _intersection(identifier: int) -> VehiclePlacement:
    return VehiclePlacement(intersection_id=identifier)


def _on(segment_id: int, progress: float) -> VehiclePlacement:
    return VehiclePlacement(segment_id=segment_id, progress=progress)


def _transition(
    step_index: int,
    before: VehiclePlacement,
    after: VehiclePlacement,
    *,
    action: int = 0,
    mask: tuple[bool, ...] = (True, True),
) -> TraceTransition:
    return TraceTransition(
        step_index=step_index,
        officer_id=0,
        before=before,
        after=after,
        distance_budget_m=BUDGET_M,
        executed_action=action,
        action_mask=mask,
    )


def _valid_trace() -> list[TraceTransition]:
    """A full directed round trip 0 -> segment 0 -> 1 -> segment 1 -> 0.

    Each interior step advances exactly ``BUDGET_M`` along the directed polyline
    and every segment change happens at the segment's own end intersection.
    """
    return [
        _transition(0, _intersection(0), _on(0, 0.0)),
        _transition(1, _on(0, 0.0), _on(0, 0.4)),
        _transition(2, _on(0, 0.4), _on(0, 0.8)),
        _transition(3, _on(0, 0.8), _intersection(1)),
        _transition(4, _intersection(1), _on(1, 0.0)),
        _transition(5, _on(1, 0.0), _on(1, 0.4)),
        _transition(6, _on(1, 0.4), _on(1, 0.8)),
        _transition(7, _on(1, 0.8), _intersection(0)),
    ]


def _report(transitions, *, episode_id: str = "episode-0"):
    return validate_episode_trace(_loop_network(), transitions, episode_id=episode_id)


def test_category_set_is_exactly_the_eight_fixed_requirement_13_7_categories():
    assert [category.value for category in VIOLATION_CATEGORIES] == [
        "direction_violation",
        "contraflow",
        "off_road",
        "speed_limit_violation",
        "teleport",
        "discontinuous_segment_transition",
        "impossible_round_trip",
        "invalid_action",
    ]


def test_valid_directed_trace_has_zero_violations_in_every_category():
    report = _report(_valid_trace())

    assert report.transition_count == 8
    assert set(report.counts) == set(VIOLATION_CATEGORIES)
    assert all(report.count(category) == 0 for category in VIOLATION_CATEGORIES)
    assert report.violations == ()
    assert report.episode_failed is False
    assert report.failure is None


# Each entry truncates the valid trace at the injected fault so every preceding
# transition stays legal and the chain of placements remains continuous.
def _direction_violation() -> list[TraceTransition]:
    # Departs intersection 1 onto segment 0, which starts at intersection 0.
    return _valid_trace()[:4] + [_transition(4, _intersection(1), _on(0, 0.0))]


def _contraflow() -> list[TraceTransition]:
    # Progress runs backward along segment 0's own directed polyline.
    return _valid_trace()[:2] + [_transition(2, _on(0, 0.4), _on(0, 0.1))]


def _off_road() -> list[TraceTransition]:
    # Segment id 5 does not exist in the network.
    return _valid_trace()[:2] + [_transition(2, _on(0, 0.4), _on(5, 0.0))]


def _speed_limit_violation() -> list[TraceTransition]:
    # Travels 5 m in a step whose distance budget is 4 m.
    return _valid_trace()[:2] + [_transition(2, _on(0, 0.4), _on(0, 0.9))]


def _teleport() -> list[TraceTransition]:
    # Leaves segment 0 with 6 m of arc remaining and a 4 m budget.
    return _valid_trace()[:2] + [_transition(2, _on(0, 0.4), _on(1, 0.0))]


def _discontinuous_segment_transition() -> list[TraceTransition]:
    # Departure onto a connected segment must start at progress zero.
    return _valid_trace()[:4] + [_transition(4, _intersection(1), _on(1, 0.1))]


def _impossible_round_trip() -> list[TraceTransition]:
    # Back at intersection 0 one step after departing on a 10 m segment.
    return _valid_trace()[:1] + [_transition(1, _on(0, 0.0), _intersection(0))]


def _invalid_action() -> list[TraceTransition]:
    # Placements are legal; the executed action is masked off.
    return _valid_trace()[:2] + [
        _transition(2, _on(0, 0.4), _on(0, 0.8), action=1, mask=(True, False))
    ]


FAULTS = {
    ViolationCategory.DIRECTION_VIOLATION: _direction_violation,
    ViolationCategory.CONTRAFLOW: _contraflow,
    ViolationCategory.OFF_ROAD: _off_road,
    ViolationCategory.SPEED_LIMIT_VIOLATION: _speed_limit_violation,
    ViolationCategory.TELEPORT: _teleport,
    ViolationCategory.DISCONTINUOUS_SEGMENT_TRANSITION: _discontinuous_segment_transition,
    ViolationCategory.IMPOSSIBLE_ROUND_TRIP: _impossible_round_trip,
    ViolationCategory.INVALID_ACTION: _invalid_action,
}


def test_every_category_has_a_fault_injection():
    assert tuple(FAULTS) == VIOLATION_CATEGORIES


@pytest.mark.parametrize("category", VIOLATION_CATEGORIES, ids=lambda item: item.value)
def test_single_injected_fault_increments_only_its_own_category(category):
    transitions = FAULTS[category]()

    # The prefix alone must be violation-free, so the fault is what is detected.
    assert not _report(transitions[:-1]).violations

    report = _report(transitions, episode_id=f"episode-{category.value}")

    assert report.count(category) == 1
    assert all(report.count(other) == 0 for other in VIOLATION_CATEGORIES if other != category)
    assert report.episode_failed is True
    assert report.failure is not None
    assert len(report.failure.cause_trace_hash) == 64
    assert int(report.failure.cause_trace_hash, 16) >= 0
    assert dict(report.failure.counts)[category.value] == 1
    assert [item.category for item in report.violations] == [category]
    assert report.violations[0].step_index == transitions[-1].step_index


def test_cause_trace_hash_is_content_addressed_and_distinguishes_causes():
    first = _report(_contraflow(), episode_id="episode-x").failure
    repeated = _report(_contraflow(), episode_id="episode-x").failure
    other = _report(_teleport(), episode_id="episode-x").failure

    assert first is not None and repeated is not None and other is not None
    assert first.cause_trace_hash == repeated.cause_trace_hash
    assert first.cause_trace_hash != other.cause_trace_hash


def test_round_trip_bound_matches_the_documented_rule():
    # ceil(10 m / 4 m) + 1 = 4 steps, so a return at delta 4 is plausible.
    plausible = [
        _transition(0, _intersection(0), _on(0, 0.0)),
        _transition(1, _on(0, 0.0), _on(0, 0.4)),
        _transition(2, _on(0, 0.4), _on(0, 0.8)),
        _transition(3, _on(0, 0.8), _intersection(1)),
        _transition(4, _intersection(1), _on(1, 0.0)),
    ]
    assert not _report(plausible).violations
    assert math.ceil(SEGMENT_LENGTH_M / BUDGET_M) + 1 == 4

    for delta in (1, 2, 3):
        early = [
            _transition(0, _intersection(0), _on(0, 0.0)),
            _transition(delta, _on(0, 0.0), _intersection(0)),
        ]
        assert _report(early).count(ViolationCategory.IMPOSSIBLE_ROUND_TRIP) == 1


def test_intention_to_evaluate_accounting_conserves_planned_episodes():
    planned = [f"episode-{index}" for index in range(5)]
    reports = [
        _report(_valid_trace(), episode_id="episode-0"),
        _report(_valid_trace(), episode_id="episode-1"),
        _report(_valid_trace(), episode_id="episode-2"),
        _report(_contraflow(), episode_id="episode-3"),
        _report(_teleport(), episode_id="episode-4"),
    ]

    accounting = account_intention_to_evaluate(reports, planned_episode_ids=planned)

    assert accounting.planned_count == 5
    assert accounting.valid_count == 3
    assert accounting.failed_count == 2
    assert accounting.missing_count == 0
    assert accounting.conserved is True
    assert accounting.valid_episode_ids == ("episode-0", "episode-1", "episode-2")
    assert [record.episode_id for record in accounting.failed_records] == [
        "episode-3",
        "episode-4",
    ]
    causes = {
        record.episode_id: dict(record.counts) for record in accounting.failed_records
    }
    assert causes["episode-3"][ViolationCategory.CONTRAFLOW.value] == 1
    assert causes["episode-4"][ViolationCategory.TELEPORT.value] == 1
    assert accounting.cause_trace_hash("episode-3") != accounting.cause_trace_hash("episode-4")


def test_unreported_planned_episode_surfaces_instead_of_shrinking_the_denominator():
    planned = ["episode-0", "episode-1", "episode-2"]
    reports = [_report(_valid_trace(), episode_id="episode-0")]

    accounting = account_intention_to_evaluate(reports, planned_episode_ids=planned)

    assert accounting.planned_count == 3
    assert accounting.valid_count == 1
    assert accounting.failed_count == 0
    assert accounting.missing_episode_ids == ("episode-1", "episode-2")
    assert accounting.conserved is True

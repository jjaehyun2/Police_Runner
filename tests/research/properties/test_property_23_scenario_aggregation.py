"""Property 23 coverage: scenario metrics conserve outcomes and six-officer aggregation."""

from __future__ import annotations

from hypothesis import given, settings, strategies as st
import pytest

from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.evaluation.paired import scenario_outcome_domain
from pursuit_evasion_rl.research.maps.registry import resolve_episode_outcome
from pursuit_evasion_rl.research.domain import EpisodeOutcome, MapScenario
from pursuit_evasion_rl.research.metrics.behavior import (
    OfficerBehaviorReport,
    OfficerBehaviorRow,
    aggregate_officer_metric,
    not_applicable_metric,
    worst_officer,
)

# **Property 23: Scenario metrics conserve outcomes and six-officer aggregation**
# **Validates: Requirements 13.6, 13.8, 19.4**

pytestmark = pytest.mark.offline

_PBT_SETTINGS = settings(max_examples=100, deadline=None)


def _row(officer_id: int, *, u_turn: float, approach: float) -> OfficerBehaviorRow:
    return OfficerBehaviorRow(
        officer_id=officer_id,
        decision_count=5,
        u_turn_rate=u_turn,
        revisit_rate=0.0,
        action_switch_rate=0.0,
        role_switch_rate=not_applicable_metric("dimensionless", "no role concept in this fixture"),
        idle_rate=0.0,
        approach_distance_m=approach,
        retreat_distance_m=0.0,
        zero_displacement_time_s=0.0,
        capture_radius_occupancy_steps=0,
        capture_radius_occupancy_time_s=0.0,
        capture_radius_entry_events=0,
        exit_block_duration_s=not_applicable_metric("s", "interior scenario has no exits to block"),
    )


@st.composite
def six_officer_rows(draw: st.DrawFn) -> list[OfficerBehaviorRow]:
    return [
        _row(
            i,
            u_turn=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
            approach=draw(st.floats(min_value=0.0, max_value=500.0, allow_nan=False)),
        )
        for i in range(6)
    ]


@_PBT_SETTINGS
@given(rows=six_officer_rows())
def test_exactly_six_officer_rows_are_required_for_aggregation_and_the_report(rows) -> None:
    report = OfficerBehaviorReport(rows=tuple(rows))
    assert len(report.rows) == 6
    assert [row.officer_id for row in report.rows] == list(range(6))

    aggregate = aggregate_officer_metric("u_turn_rate", rows)
    assert len(aggregate.values) == 6


@_PBT_SETTINGS
@given(rows=six_officer_rows(), drop_index=st.integers(min_value=0, max_value=5))
def test_five_or_seven_rows_are_rejected_for_both_the_report_and_aggregation(rows, drop_index) -> None:
    five = [row for index, row in enumerate(rows) if index != drop_index]
    seven = [*rows, _row(6, u_turn=0.0, approach=0.0)]
    for bad in (five, seven):
        with pytest.raises(ResearchValidationError) as excinfo:
            OfficerBehaviorReport(rows=tuple(bad))
        assert excinfo.value.code == "INVALID_OFFICER_ROW_COUNT"
        with pytest.raises(ResearchValidationError) as excinfo_agg:
            aggregate_officer_metric("u_turn_rate", bad)
        assert excinfo_agg.value.code == "INVALID_OFFICER_ROW_COUNT"


@_PBT_SETTINGS
@given(rows=six_officer_rows())
def test_worst_officer_direction_is_respected_for_both_lower_and_higher_is_better_metrics(rows) -> None:
    u_turn_values = [row.u_turn_rate for row in rows]
    approach_values = [row.approach_distance_m for row in rows]

    # u_turn_rate is LOWER_IS_BETTER: worst is the officer with the HIGHEST rate.
    worst_uturn_index, worst_uturn_value = worst_officer("u_turn_rate", u_turn_values)
    assert worst_uturn_value == max(u_turn_values)
    assert worst_uturn_index == max(
        range(6), key=lambda i: (u_turn_values[i], -i)
    )

    # approach_distance_m is HIGHER_IS_BETTER: worst is the officer with the LOWEST distance.
    worst_approach_index, worst_approach_value = worst_officer("approach_distance_m", approach_values)
    assert worst_approach_value == min(approach_values)
    assert worst_approach_index == min(
        range(6), key=lambda i: (approach_values[i], i)
    )

    u_turn_aggregate = aggregate_officer_metric("u_turn_rate", rows)
    approach_aggregate = aggregate_officer_metric("approach_distance_m", rows)
    assert u_turn_aggregate.worst_officer_id == worst_uturn_index
    assert approach_aggregate.worst_officer_id == worst_approach_index
    assert u_turn_aggregate.mean == pytest.approx(sum(u_turn_values) / 6)
    assert approach_aggregate.mean == pytest.approx(sum(approach_values) / 6)


# ---------------------------------------------------------------------------
# Scenario outcome-domain conservation (an Interior map can never escape)
# ---------------------------------------------------------------------------


@_PBT_SETTINGS
@given(outcome=st.sampled_from(list(EpisodeOutcome)))
def test_interior_scenarios_outcome_domain_never_admits_escape(outcome: EpisodeOutcome) -> None:
    domain = scenario_outcome_domain(MapScenario.INTERIOR_CONTAINED)
    assert domain == frozenset({EpisodeOutcome.CAPTURE, EpisodeOutcome.TIMEOUT})
    assert (outcome in domain) == (outcome is not EpisodeOutcome.ESCAPE)


@_PBT_SETTINGS
@given(
    capture=st.booleans(),
    boundary_escape=st.booleans(),
    timeout=st.booleans(),
)
def test_capture_always_wins_over_escape_when_both_are_asserted(capture: bool, boundary_escape: bool, timeout: bool) -> None:
    if not (capture or boundary_escape or timeout):
        with pytest.raises(ResearchValidationError) as excinfo:
            resolve_episode_outcome(capture=capture, boundary_escape=boundary_escape, timeout=timeout)
        assert excinfo.value.code == "MISSING_TERMINAL_OUTCOME"
        return
    resolved = resolve_episode_outcome(capture=capture, boundary_escape=boundary_escape, timeout=timeout)
    if capture:
        assert resolved is EpisodeOutcome.CAPTURE
    elif boundary_escape:
        assert resolved is EpisodeOutcome.ESCAPE
    else:
        assert resolved is EpisodeOutcome.TIMEOUT
    # Boundary scenario admits all three outcomes; whichever was resolved must
    # be a member of its outcome domain, and capture never loses precedence.
    assert resolved in scenario_outcome_domain(MapScenario.BOUNDARY_ESCAPE)

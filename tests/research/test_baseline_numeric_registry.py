"""Task 5.4 numeric baseline registry, fairness schema and matrix validation."""
from __future__ import annotations

from dataclasses import replace
import math

import pytest

from pursuit_evasion_rl.osm_demo.models import (
    Intersection,
    ModelNetwork,
    Segment,
    VehiclePlacement,
)
from pursuit_evasion_rl.research.domain import ExecutionStatus, MapScenario
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.policies.baselines import (
    DIRECTED_SHORTEST_PATH_ID,
    ENCIRCLEMENT_ID,
    GREEDY_INTERCEPT_ID,
    LEGACY_SHARED_PPO_ID,
    MINIMUM_REQUIRED_BASELINE_IDS,
    PROPOSED_POLICY_ID,
    REQUIRED_METRIC_FIELDS,
    BaselineRequirement,
    BaselineResultRow,
    ComparisonOutcome,
    CostAvailability,
    EpisodeCaseRef,
    EvaluationStratum,
    MetricSchemaRefs,
    PolicyKind,
    RuntimeMeasurement,
    TrainingBudget,
    check_policy_interface_conformance,
    directed_shortest_path_policy,
    greedy_intercept_policy,
    not_run_baselines,
    numeric_baseline_registry,
    required_baseline_ids,
    validate_primary_matrix,
)

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Offline synthetic fixtures (no OSM / osmnx dependency)
# ---------------------------------------------------------------------------


def _grid_network() -> ModelNetwork:
    """A bidirectional 3x3 grid: branching exits, max out-degree four."""
    positions = {index: (float(index % 3) * 100.0, float(index // 3) * 100.0) for index in range(9)}
    undirected: list[tuple[int, int]] = []
    for index in range(9):
        row, col = divmod(index, 3)
        if col < 2:
            undirected.append((index, index + 1))
        if row < 2:
            undirected.append((index, index + 3))
    directed = [*undirected, *((b, a) for a, b in undirected)]
    segments = tuple(
        Segment(
            segment_id, start, end,
            math.dist(positions[start], positions[end]),
            (positions[start], positions[end]),
        )
        for segment_id, (start, end) in enumerate(directed)
    )
    intersections = tuple(
        Intersection(
            index, positions[index], f"node-{index}",
            outgoing_segment_ids=tuple(s.id for s in segments if s.start_id == index),
            incoming_segment_ids=tuple(s.id for s in segments if s.end_id == index),
        )
        for index in range(9)
    )
    return ModelNetwork(intersections, segments)


def _placements() -> tuple[list[VehiclePlacement], VehiclePlacement]:
    police = [VehiclePlacement(intersection_id=index) for index in (0, 1, 2, 3, 5, 6)]
    return police, VehiclePlacement(intersection_id=8)


def _budget(env_steps: int = 900_000) -> TrainingBudget:
    return TrainingBudget(
        env_steps=env_steps,
        optimizer_updates=2_000,
        resource_ceiling_id="ceiling-cpu-8h",
        model_selection_data="validation split capture rate",
        checkpoint_selection_rule="argmax validation capture rate over updates",
    )


def _measurements() -> dict[str, RuntimeMeasurement]:
    learned = RuntimeMeasurement(
        inference_latency_per_episode_s=0.42,
        method="wall clock via time.perf_counter over 100 evaluation episodes",
        train_wall_clock_s=7_200.0,
        accelerator_time_s=6_100.0,
    )
    heuristic = RuntimeMeasurement(
        inference_latency_per_episode_s=0.03,
        method="wall clock via time.perf_counter over 100 evaluation episodes",
    )
    return {
        PROPOSED_POLICY_ID: learned,
        LEGACY_SHARED_PPO_ID: learned,
        ENCIRCLEMENT_ID: heuristic,
        DIRECTED_SHORTEST_PATH_ID: heuristic,
        GREEDY_INTERCEPT_ID: heuristic,
    }


def _conformance() -> dict[str, object]:
    network = _grid_network()
    police, fugitive = _placements()
    return {
        DIRECTED_SHORTEST_PATH_ID: check_policy_interface_conformance(
            DIRECTED_SHORTEST_PATH_ID, directed_shortest_path_policy(network),
            network=network, police=police, fugitive=fugitive,
        ),
        GREEDY_INTERCEPT_ID: check_policy_interface_conformance(
            GREEDY_INTERCEPT_ID, greedy_intercept_policy(network),
            network=network, police=police, fugitive=fugitive,
        ),
    }


def _full_registry():
    return numeric_baseline_registry(
        budget=_budget(), measurements=_measurements(), conformance=_conformance()
    )


_STRATUM = EvaluationStratum(MapScenario.INTERIOR_CONTAINED, "held_out")


def _case() -> EpisodeCaseRef:
    return EpisodeCaseRef(
        map_hash="map-hash-abc", scenario=MapScenario.INTERIOR_CONTAINED,
        placement_hash="placement-hash-xyz", seed=7,
    )


def _metrics(**extra: str) -> MetricSchemaRefs:
    refs = {name: f"artifact://metrics/{name}" for name in REQUIRED_METRIC_FIELDS}
    refs.update(extra)
    return MetricSchemaRefs(refs)


def _row(baseline_id: str, outcome: ComparisonOutcome, *, metrics: MetricSchemaRefs | None = None) -> BaselineResultRow:
    return BaselineResultRow(
        stratum=_STRATUM, baseline_id=baseline_id, episode_case=_case(),
        metrics=metrics or _metrics(), outcome_vs_proposed=outcome,
    )


def _rows(registry) -> list[BaselineResultRow]:
    """One row per required baseline; the proposed policy loses to encirclement."""
    outcomes = {
        PROPOSED_POLICY_ID: ComparisonOutcome.TIE,
        LEGACY_SHARED_PPO_ID: ComparisonOutcome.PROPOSED_SUPERIOR,
        ENCIRCLEMENT_ID: ComparisonOutcome.PROPOSED_INFERIOR,
        DIRECTED_SHORTEST_PATH_ID: ComparisonOutcome.UNDETERMINED,
        GREEDY_INTERCEPT_ID: ComparisonOutcome.PROPOSED_SUPERIOR,
    }
    return [_row(baseline_id, outcomes[baseline_id]) for baseline_id in required_baseline_ids(registry)]


# ---------------------------------------------------------------------------
# Registry construction and the fair-comparison happy path
# ---------------------------------------------------------------------------


def test_registry_registers_every_named_baseline_with_matched_budget():
    registry = _full_registry()

    assert set(registry) == {
        PROPOSED_POLICY_ID, LEGACY_SHARED_PPO_ID, ENCIRCLEMENT_ID,
        DIRECTED_SHORTEST_PATH_ID, GREEDY_INTERCEPT_ID,
    }
    # Requirement 11.1: the three minimum baselines are unconditionally required.
    for baseline_id in MINIMUM_REQUIRED_BASELINE_IDS:
        assert registry[baseline_id].requirement is BaselineRequirement.MINIMUM_REQUIRED
    # Requirement 11.5: the proposed policy and legacy shared PPO share one budget.
    proposed, legacy = registry[PROPOSED_POLICY_ID], registry[LEGACY_SHARED_PPO_ID]
    assert proposed.policy_kind is legacy.policy_kind is PolicyKind.LEARNED
    assert proposed.training_budget.budget_hash == legacy.training_budget.budget_hash
    # The legacy 21D actor is capacity-matched to the proposed 28D actor within 1%.
    proposed_params = proposed.cost.parameter_count.value
    legacy_params = legacy.cost.parameter_count.value
    assert abs(legacy_params - proposed_params) / proposed_params <= 0.01
    # Requirement 11.6: selection provenance never touches held-out data.
    for registration in registry.values():
        assert set(registration.selection.selection_data_splits) <= {"train", "validation"}
        assert registration.selection.search_space.strip()
        assert registration.selection.selection_rule.strip()


def test_cost_schema_distinguishes_not_applicable_from_unrecorded():
    registry = _full_registry()

    learned_cost = registry[PROPOSED_POLICY_ID].cost
    heuristic_cost = registry[ENCIRCLEMENT_ID].cost

    # A learned policy really incurs training cost, so it is measured with a method.
    assert learned_cost.train_wall_clock_s.availability is CostAvailability.MEASURED
    assert learned_cost.train_wall_clock_s.value == pytest.approx(7_200.0)
    assert learned_cost.train_wall_clock_s.method.strip()
    # A rule-based baseline cannot incur it: not_applicable, with a stated reason.
    assert heuristic_cost.train_wall_clock_s.availability is CostAvailability.NOT_APPLICABLE
    assert heuristic_cost.train_wall_clock_s.value is None
    assert "no training phase" in heuristic_cost.train_wall_clock_s.method
    # Both states are distinct from "we simply never recorded it".
    assert CostAvailability.NOT_MEASURED not in {
        heuristic_cost.train_wall_clock_s.availability,
        learned_cost.train_wall_clock_s.availability,
    }
    assert heuristic_cost.unrecorded_fields == ()
    # Inference latency applies to every policy kind and is measured for both.
    assert learned_cost.inference_latency_per_episode_s.availability is CostAvailability.MEASURED
    assert heuristic_cost.inference_latency_per_episode_s.availability is CostAvailability.MEASURED


def test_learned_policy_missing_runtime_measurement_is_rejected():
    """A learned policy may not silently inherit the rule-based not-applicable path."""
    measurements = _measurements()
    measurements[LEGACY_SHARED_PPO_ID] = RuntimeMeasurement(
        inference_latency_per_episode_s=0.42, method="perf_counter",
    )
    with pytest.raises(ResearchValidationError) as excinfo:
        numeric_baseline_registry(budget=_budget(), measurements=measurements, conformance=_conformance())
    assert excinfo.value.code == "MISSING_COST_MEASUREMENT"


def test_fully_populated_stratum_passes_validation():
    registry = _full_registry()
    reports = validate_primary_matrix(_rows(registry), registry, strata=[_STRATUM])

    assert len(reports) == 1
    report = reports[0]
    assert report.stratum == _STRATUM
    assert len(report.retained_rows) == len(required_baseline_ids(registry)) == 5
    assert report.budget_hash == registry[PROPOSED_POLICY_ID].training_budget.budget_hash
    assert report.not_run == ()


# ---------------------------------------------------------------------------
# 완료 검증 1: a missing minimum baseline fails validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dropped", MINIMUM_REQUIRED_BASELINE_IDS)
def test_missing_minimum_baseline_fails_primary_matrix_validation(dropped):
    registry = _full_registry()
    rows = [row for row in _rows(registry) if row.baseline_id != dropped]

    with pytest.raises(ResearchValidationError) as excinfo:
        validate_primary_matrix(rows, registry, strata=[_STRATUM])

    assert excinfo.value.code == "MISSING_MINIMUM_BASELINE"
    assert excinfo.value.actual == [dropped]


def test_conformant_conditional_baseline_becomes_required():
    """Requirement 11.3: a passing shortest-path baseline is required everywhere."""
    registry = _full_registry()
    assert DIRECTED_SHORTEST_PATH_ID in required_baseline_ids(registry)
    rows = [row for row in _rows(registry) if row.baseline_id != DIRECTED_SHORTEST_PATH_ID]

    with pytest.raises(ResearchValidationError) as excinfo:
        validate_primary_matrix(rows, registry, strata=[_STRATUM])
    assert excinfo.value.code == "MISSING_MINIMUM_BASELINE"


def test_stratum_with_no_results_at_all_fails():
    registry = _full_registry()
    other = EvaluationStratum(MapScenario.BOUNDARY_ESCAPE, "held_out")

    with pytest.raises(ResearchValidationError) as excinfo:
        validate_primary_matrix(_rows(registry), registry, strata=[_STRATUM, other])
    assert excinfo.value.code == "MISSING_STRATUM"


# ---------------------------------------------------------------------------
# 완료 검증 2: a budget mismatch between compared policies fails validation
# ---------------------------------------------------------------------------


def test_budget_mismatch_between_learned_policies_fails_validation():
    registry = _full_registry()
    registry[LEGACY_SHARED_PPO_ID] = replace(
        registry[LEGACY_SHARED_PPO_ID], training_budget=_budget(env_steps=450_000)
    )

    with pytest.raises(ResearchValidationError) as excinfo:
        validate_primary_matrix(_rows(registry), registry, strata=[_STRATUM])

    assert excinfo.value.code == "BUDGET_MISMATCH"
    assert sorted(excinfo.value.actual) == sorted([PROPOSED_POLICY_ID, LEGACY_SHARED_PPO_ID])


def test_checkpoint_selection_rule_difference_is_also_a_budget_mismatch():
    """Requirement 11.5 covers selection rules, not just step counts."""
    registry = _full_registry()
    divergent = replace(_budget(), checkpoint_selection_rule="last update, no validation selection")
    registry[LEGACY_SHARED_PPO_ID] = replace(registry[LEGACY_SHARED_PPO_ID], training_budget=divergent)

    with pytest.raises(ResearchValidationError) as excinfo:
        validate_primary_matrix(_rows(registry), registry, strata=[_STRATUM])
    assert excinfo.value.code == "BUDGET_MISMATCH"


# ---------------------------------------------------------------------------
# 완료 검증 3: a metric schema difference fails validation
# ---------------------------------------------------------------------------


def test_metric_schema_difference_between_rows_fails_validation():
    registry = _full_registry()
    rows = _rows(registry)
    divergent = _metrics(containment_v2="artifact://metrics/containment_v2")
    rows = [
        replace(row, metrics=divergent) if row.baseline_id == ENCIRCLEMENT_ID else row
        for row in rows
    ]

    with pytest.raises(ResearchValidationError) as excinfo:
        validate_primary_matrix(rows, registry, strata=[_STRATUM])

    assert excinfo.value.code == "METRIC_SCHEMA_MISMATCH"
    assert "containment_v2" in excinfo.value.actual or "containment_v2" in excinfo.value.expected


def test_row_missing_a_required_metric_reference_is_rejected_at_construction():
    refs = {name: f"artifact://metrics/{name}" for name in REQUIRED_METRIC_FIELDS}
    refs.pop("containment")

    with pytest.raises(ResearchValidationError) as excinfo:
        MetricSchemaRefs(refs)

    assert excinfo.value.code == "MISSING_METRIC_REFERENCE"
    assert excinfo.value.actual == ["containment"]


def test_rows_scored_on_different_episode_cases_fail_validation():
    """Requirement 11.7: every baseline in a stratum shares one Episode_Case."""
    registry = _full_registry()
    other_case = replace(_case(), seed=99)
    rows = [
        replace(row, episode_case=other_case) if row.baseline_id == ENCIRCLEMENT_ID else row
        for row in _rows(registry)
    ]

    with pytest.raises(ResearchValidationError) as excinfo:
        validate_primary_matrix(rows, registry, strata=[_STRATUM])
    assert excinfo.value.code == "EPISODE_CASE_MISMATCH"


# ---------------------------------------------------------------------------
# 완료 검증 4: not_run preservation and no filtering of losing results
# ---------------------------------------------------------------------------


class _NonConformingPolicy:
    """A candidate that violates the team-cardinality clause of the contract."""

    profile = "broken_candidate"
    experimental = False

    def recommend(self, *, police, fugitive, step, max_steps, incoming_headings=None):
        return ()


def test_conformance_failure_is_recorded_as_not_run_with_a_reason():
    network = _grid_network()
    police, fugitive = _placements()
    report = check_policy_interface_conformance(
        GREEDY_INTERCEPT_ID, _NonConformingPolicy(),
        network=network, police=police, fugitive=fugitive,
    )
    assert not report.conforms
    assert report.status is ExecutionStatus.NOT_RUN
    assert "team_cardinality" in report.failure_reason

    registry = numeric_baseline_registry(
        budget=_budget(), measurements=_measurements(),
        conformance={**_conformance(), GREEDY_INTERCEPT_ID: report},
    )

    # Requirement 11.4: preserved as not_run with a reason, never silently omitted.
    assert GREEDY_INTERCEPT_ID in registry
    registration = registry[GREEDY_INTERCEPT_ID]
    assert registration.status is ExecutionStatus.NOT_RUN
    assert "failed the common policy interface check" in registration.status_reason
    assert GREEDY_INTERCEPT_ID not in required_baseline_ids(registry)
    assert dict(not_run_baselines(registry))[GREEDY_INTERCEPT_ID] == registration.status_reason

    # A not_run baseline must not then report a result row.
    rows = [*_rows(registry), _row(GREEDY_INTERCEPT_ID, ComparisonOutcome.PROPOSED_SUPERIOR)]
    with pytest.raises(ResearchValidationError) as excinfo:
        validate_primary_matrix(rows, registry, strata=[_STRATUM])
    assert excinfo.value.code == "NOT_RUN_BASELINE_HAS_RESULT"

    # ...and the remaining matrix, which excludes it, still validates.
    report_out = validate_primary_matrix(_rows(registry), registry, strata=[_STRATUM])[0]
    assert dict(report_out.not_run)[GREEDY_INTERCEPT_ID] == registration.status_reason


def test_no_conformance_verdict_defaults_to_not_run_rather_than_omission():
    registry = numeric_baseline_registry(budget=_budget(), measurements=_measurements())

    for baseline_id in (DIRECTED_SHORTEST_PATH_ID, GREEDY_INTERCEPT_ID):
        assert registry[baseline_id].status is ExecutionStatus.NOT_RUN
        assert "no common-interface conformance check was recorded" in registry[baseline_id].status_reason
    assert set(required_baseline_ids(registry)) == set(MINIMUM_REQUIRED_BASELINE_IDS)


def test_inferior_and_null_result_rows_are_preserved_not_filtered():
    registry = _full_registry()
    rows = _rows(registry)
    report = validate_primary_matrix(rows, registry, strata=[_STRATUM])[0]

    # Every supplied row survives verbatim, including the losing and null ones.
    assert report.retained_rows == tuple(rows)
    outcomes = {row.baseline_id: row.outcome_vs_proposed for row in report.retained_rows}
    assert outcomes[ENCIRCLEMENT_ID] is ComparisonOutcome.PROPOSED_INFERIOR
    assert outcomes[DIRECTED_SHORTEST_PATH_ID] is ComparisonOutcome.UNDETERMINED
    assert outcomes[PROPOSED_POLICY_ID] is ComparisonOutcome.TIE

    non_winning = {row.baseline_id for row in report.non_winning_rows}
    assert {ENCIRCLEMENT_ID, DIRECTED_SHORTEST_PATH_ID} <= non_winning
    assert report.outcome_counts[ComparisonOutcome.PROPOSED_INFERIOR] == 1
    assert report.outcome_counts[ComparisonOutcome.UNDETERMINED] == 1

    # Deleting the losing row is detected as a missing baseline, not accepted.
    pruned = [row for row in rows if row.outcome_vs_proposed is not ComparisonOutcome.PROPOSED_INFERIOR]
    with pytest.raises(ResearchValidationError) as excinfo:
        validate_primary_matrix(pruned, registry, strata=[_STRATUM])
    assert excinfo.value.code == "MISSING_MINIMUM_BASELINE"
    assert excinfo.value.actual == [ENCIRCLEMENT_ID]


def test_unregistered_baseline_row_is_rejected():
    registry = _full_registry()
    rows = [*_rows(registry), _row("undeclared_policy", ComparisonOutcome.PROPOSED_SUPERIOR)]

    with pytest.raises(ResearchValidationError) as excinfo:
        validate_primary_matrix(rows, registry, strata=[_STRATUM])
    assert excinfo.value.code == "UNREGISTERED_BASELINE_ROW"
    assert excinfo.value.actual == ["undeclared_policy"]


# ---------------------------------------------------------------------------
# 완료 검증 5: the shortest-path baseline drives a legal action sequence
# ---------------------------------------------------------------------------


def test_directed_shortest_path_baseline_produces_a_legal_action_sequence():
    network = _grid_network()
    police, fugitive = _placements()
    policy = directed_shortest_path_policy(network)

    report = check_policy_interface_conformance(
        DIRECTED_SHORTEST_PATH_ID, policy, network=network, police=police, fugitive=fugitive,
    )
    assert report.conforms, report.failure_reason
    assert report.status is ExecutionStatus.COMPLETED

    # Roll the team forward: every officer's chosen exit must stay legal and, for
    # the corner officer, must strictly close the road distance to the fugitive.
    current = list(police)
    positions = {item.id: item.position_xy for item in network.intersections}
    distances = [math.dist(positions[0], positions[8])]
    for step in range(4):
        recommendations = policy.recommend(
            police=current, fugitive=fugitive, step=step, max_steps=10,
        )
        assert len(recommendations) == 6
        assert all(item.valid for item in recommendations)
        current = [VehiclePlacement(intersection_id=item.next_intersection_id) for item in recommendations]
        distances.append(math.dist(positions[current[0].intersection_id], positions[8]))

    assert distances == sorted(distances, reverse=True)
    assert current[0].intersection_id == 8


def test_greedy_intercept_baseline_conforms_and_is_deterministic():
    network = _grid_network()
    police, fugitive = _placements()
    policy = greedy_intercept_policy(network)

    report = check_policy_interface_conformance(
        GREEDY_INTERCEPT_ID, policy, network=network, police=police, fugitive=fugitive,
    )
    assert report.conforms, report.failure_reason

    first = policy.recommend(police=police, fugitive=fugitive, step=0, max_steps=10)
    second = policy.recommend(police=police, fugitive=fugitive, step=0, max_steps=10)
    assert [item.action_index for item in first] == [item.action_index for item in second]
    # Officer 0 sits at the grid corner farthest from the fugitive at node 8; the
    # greedy step must reduce straight-line distance to it.
    positions = {item.id: item.position_xy for item in network.intersections}
    assert math.dist(positions[first[0].next_intersection_id], positions[8]) < math.dist(
        positions[0], positions[8]
    )

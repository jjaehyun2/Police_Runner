"""Property 17 coverage for baseline matrix coverage, fairness and cost semantics."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import math
import string

from hypothesis import assume, given, settings, strategies as st
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
    CONDITIONAL_BASELINE_IDS,
    COST_FIELD_NAMES,
    DIRECTED_SHORTEST_PATH_ID,
    ENCIRCLEMENT_ID,
    GREEDY_INTERCEPT_ID,
    LEGACY_SHARED_PPO_ID,
    MINIMUM_REQUIRED_BASELINE_IDS,
    NON_WINNING_OUTCOMES,
    PROPOSED_POLICY_ID,
    REQUIRED_METRIC_FIELDS,
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
    measured_cost,
    not_applicable_cost,
    not_measured_cost,
    not_run_baselines,
    numeric_baseline_registry,
    required_baseline_ids,
    validate_primary_matrix,
)

# **Property 17: Baseline coverage and fairness hold for every primary stratum**
# **Validates: Requirements 11.1-11.2, 11.4-11.6**

pytestmark = pytest.mark.offline

_PBT_SETTINGS = settings(max_examples=100, deadline=None)


# ---------------------------------------------------------------------------
# Offline synthetic fixtures (no OSM / osmnx dependency)
# ---------------------------------------------------------------------------


def _grid_network() -> ModelNetwork:
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


_BASE_BUDGET = TrainingBudget(
    env_steps=900_000,
    optimizer_updates=2_000,
    resource_ceiling_id="ceiling-cpu-8h",
    model_selection_data="validation split capture rate",
    checkpoint_selection_rule="argmax validation capture rate over updates",
)

_LEARNED_MEASUREMENT = RuntimeMeasurement(
    inference_latency_per_episode_s=0.42,
    method="wall clock via time.perf_counter over 100 evaluation episodes",
    train_wall_clock_s=7_200.0,
    accelerator_time_s=6_100.0,
)
_HEURISTIC_MEASUREMENT = RuntimeMeasurement(
    inference_latency_per_episode_s=0.03,
    method="wall clock via time.perf_counter over 100 evaluation episodes",
)
_MEASUREMENTS = {
    PROPOSED_POLICY_ID: _LEARNED_MEASUREMENT,
    LEGACY_SHARED_PPO_ID: _LEARNED_MEASUREMENT,
    ENCIRCLEMENT_ID: _HEURISTIC_MEASUREMENT,
    DIRECTED_SHORTEST_PATH_ID: _HEURISTIC_MEASUREMENT,
    GREEDY_INTERCEPT_ID: _HEURISTIC_MEASUREMENT,
}


@lru_cache(maxsize=1)
def _conformance_reports() -> dict:
    """Passing interface verdicts for both conditional baselines (built once)."""
    network = _grid_network()
    police = [VehiclePlacement(intersection_id=index) for index in (0, 1, 2, 3, 5, 6)]
    fugitive = VehiclePlacement(intersection_id=8)
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


def _registry(conditionals=(), *, budget: TrainingBudget = _BASE_BUDGET) -> dict:
    """A fresh registry where only ``conditionals`` carry a passing verdict."""
    reports = _conformance_reports()
    return numeric_baseline_registry(
        budget=budget,
        measurements=_MEASUREMENTS,
        conformance={baseline_id: reports[baseline_id] for baseline_id in conditionals},
    )


def _metrics(
    extra: tuple[str, ...] = (),
    *,
    blank: str | None = None,
    drop: str | None = None,
    filler: str = "   ",
) -> MetricSchemaRefs:
    refs = {name: f"artifact://metrics/{name}" for name in (*REQUIRED_METRIC_FIELDS, *extra)}
    if blank is not None:
        refs[blank] = filler
    if drop is not None:
        refs.pop(drop)
    return MetricSchemaRefs(refs)


def _case(stratum: EvaluationStratum, seed: int) -> EpisodeCaseRef:
    return EpisodeCaseRef(
        map_hash=f"map-hash-{stratum.split}",
        scenario=stratum.scenario,
        placement_hash=f"placement-hash-{stratum.split}-{seed}",
        seed=seed,
    )


def _rows_for_stratum(
    stratum: EvaluationStratum,
    baseline_ids,
    outcomes,
    *,
    seed: int,
    metrics_by_baseline=None,
) -> list[BaselineResultRow]:
    case = _case(stratum, seed)
    overrides = metrics_by_baseline or {}
    shared = _metrics()
    return [
        BaselineResultRow(
            stratum=stratum,
            baseline_id=baseline_id,
            episode_case=case,
            metrics=overrides.get(baseline_id, shared),
            outcome_vs_proposed=outcomes[baseline_id],
        )
        for baseline_id in baseline_ids
    ]


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_strata = st.builds(
    EvaluationStratum,
    scenario=st.sampled_from(tuple(MapScenario)),
    split=st.sampled_from(("train", "validation", "held_out", "transfer_city")),
)
_stratum_sets = st.lists(_strata, min_size=1, max_size=3, unique_by=lambda item: item.stratum_id)
_conditional_sets = st.sets(st.sampled_from(CONDITIONAL_BASELINE_IDS))
_outcomes = st.sampled_from(tuple(ComparisonOutcome))
_non_winning = st.sampled_from(tuple(item for item in ComparisonOutcome if item in NON_WINNING_OUTCOMES))
_seeds = st.integers(min_value=0, max_value=10_000)
_identifiers = st.text(alphabet=string.ascii_lowercase + "-_", min_size=1, max_size=16)


def _draw_outcomes(data: st.DataObject, baseline_ids, strategy=_outcomes) -> dict[str, ComparisonOutcome]:
    return {
        baseline_id: data.draw(strategy, label=f"outcome_{baseline_id}")
        for baseline_id in baseline_ids
    }


# ---------------------------------------------------------------------------
# Invariant 1/5: every minimum-required baseline must report a row in every
# primary stratum; a stratum missing one is rejected by name (Req 11.1, 11.4).
# ---------------------------------------------------------------------------
@_PBT_SETTINGS
@given(
    strata=_stratum_sets,
    conditionals=_conditional_sets,
    missing=st.sampled_from(MINIMUM_REQUIRED_BASELINE_IDS),
    defect_index=st.integers(min_value=0, max_value=2),
    seed=_seeds,
    data=st.data(),
)
def test_a_stratum_missing_a_minimum_required_baseline_is_always_rejected(
    strata, conditionals, missing, defect_index, seed, data
) -> None:
    registry = _registry(conditionals)
    required = required_baseline_ids(registry)
    # Requirement 11.1: the three minimum baselines are required no matter which
    # conditional baselines happened to pass their conformance check.
    assert set(MINIMUM_REQUIRED_BASELINE_IDS) <= set(required)
    assert set(required) == set(MINIMUM_REQUIRED_BASELINE_IDS) | conditionals
    # Requirement 11.4: an excluded conditional baseline is preserved as not_run
    # with a recorded reason rather than dropped from the registry.
    preserved = dict(not_run_baselines(registry))
    for baseline_id in set(CONDITIONAL_BASELINE_IDS) - conditionals:
        assert registry[baseline_id].status is ExecutionStatus.NOT_RUN
        assert preserved[baseline_id].strip()

    defective = strata[defect_index % len(strata)]
    rows: list[BaselineResultRow] = []
    for stratum in strata:
        present = list(data.draw(st.permutations(required), label=f"order_{stratum.stratum_id}"))
        if stratum is defective:
            present = [item for item in present if item != missing]
        rows.extend(_rows_for_stratum(stratum, present, _draw_outcomes(data, required), seed=seed))

    with pytest.raises(ResearchValidationError) as excinfo:
        validate_primary_matrix(rows, registry, strata=strata)

    assert excinfo.value.code == "MISSING_MINIMUM_BASELINE"
    assert excinfo.value.actual == [missing]
    assert defective.stratum_id in str(excinfo.value)


# ---------------------------------------------------------------------------
# Invariant 2/5: the two learned policies are compared only under one shared
# training budget; any divergent budget field fails validation (Req 11.5).
# ---------------------------------------------------------------------------
_INT_BUDGET_FIELDS = ("env_steps", "optimizer_updates")
_TEXT_BUDGET_FIELDS = ("resource_ceiling_id", "model_selection_data", "checkpoint_selection_rule")


@_PBT_SETTINGS
@given(
    stratum=_strata,
    conditionals=_conditional_sets,
    field=st.sampled_from(_INT_BUDGET_FIELDS + _TEXT_BUDGET_FIELDS),
    favoured=st.sampled_from((PROPOSED_POLICY_ID, LEGACY_SHARED_PPO_ID)),
    seed=_seeds,
    data=st.data(),
)
def test_learned_policies_on_mismatched_training_budgets_are_always_rejected(
    stratum, conditionals, field, favoured, seed, data
) -> None:
    registry = _registry(conditionals)
    required = required_baseline_ids(registry)
    if field in _INT_BUDGET_FIELDS:
        value = data.draw(st.integers(min_value=1, max_value=5_000_000), label="budget_value")
        assume(value != getattr(_BASE_BUDGET, field))
    else:
        value = f"divergent-{data.draw(_identifiers, label='budget_value')}"

    divergent = replace(_BASE_BUDGET, **{field: value})
    assert divergent.budget_hash != _BASE_BUDGET.budget_hash
    registry[favoured] = replace(registry[favoured], training_budget=divergent)

    rows = _rows_for_stratum(stratum, required, _draw_outcomes(data, required), seed=seed)
    with pytest.raises(ResearchValidationError) as excinfo:
        validate_primary_matrix(rows, registry, strata=[stratum])

    assert excinfo.value.code == "BUDGET_MISMATCH"
    assert sorted(excinfo.value.actual) == sorted((PROPOSED_POLICY_ID, LEGACY_SHARED_PPO_ID))
    # Both compared policies are learned, so both really are budget-bearing.
    for baseline_id in (PROPOSED_POLICY_ID, LEGACY_SHARED_PPO_ID):
        assert registry[baseline_id].policy_kind is PolicyKind.LEARNED


# ---------------------------------------------------------------------------
# Invariant 3/5: every row in a stratum exposes the same metric schema, and a
# required metric reference can never be blanked away (Requirement 11.7).
# ---------------------------------------------------------------------------
@_PBT_SETTINGS
@given(
    stratum=_strata,
    conditionals=_conditional_sets,
    extra_key=st.sampled_from(("containment_v2", "capture_at_k", "custom_probe")),
    seed=_seeds,
    data=st.data(),
)
def test_one_row_with_a_divergent_metric_schema_is_always_rejected(
    stratum, conditionals, extra_key, seed, data
) -> None:
    registry = _registry(conditionals)
    required = required_baseline_ids(registry)
    odd_one_out = data.draw(st.sampled_from(required), label="divergent_baseline")

    rows = _rows_for_stratum(
        stratum, required, _draw_outcomes(data, required), seed=seed,
        metrics_by_baseline={odd_one_out: _metrics(extra=(extra_key,))},
    )
    with pytest.raises(ResearchValidationError) as excinfo:
        validate_primary_matrix(rows, registry, strata=[stratum])

    assert excinfo.value.code == "METRIC_SCHEMA_MISMATCH"
    assert extra_key in excinfo.value.actual or extra_key in excinfo.value.expected


@_PBT_SETTINGS
@given(
    field=st.sampled_from(REQUIRED_METRIC_FIELDS),
    blank_it=st.booleans(),
    filler=st.sampled_from(("", " ", "\t", "\n  ")),
    extra=st.sets(_identifiers.map(lambda item: f"aux_{item}"), max_size=2),
)
def test_a_blanked_or_dropped_metric_reference_is_rejected_at_construction(
    field, blank_it, filler, extra
) -> None:
    """A missing metric reference fails loudly; it never degrades to an empty ref."""
    with pytest.raises(ResearchValidationError) as excinfo:
        if blank_it:
            _metrics(extra=tuple(sorted(extra)), blank=field, filler=filler)
        else:
            _metrics(extra=tuple(sorted(extra)), drop=field)

    assert excinfo.value.code == "MISSING_METRIC_REFERENCE"
    assert field in (excinfo.value.path or "") or excinfo.value.actual == [field]


# ---------------------------------------------------------------------------
# Invariant 4/5: unfavourable results are first-class members of the matrix.
#
# This is an *absence* property: no outcome value may, by itself, cause a row
# to be rejected, dropped, reordered or reweighted.  It is modelled three ways:
#   (a) a stratum in which the proposed policy never wins validates in full;
#   (b) the verdict and the retained rows are invariant under relabelling every
#       outcome to PROPOSED_SUPERIOR -- acceptance cannot depend on the label;
#   (c) deleting an unfavourable row is detected as a coverage failure naming
#       that baseline, so a losing row cannot be silently pruned (Req 11.8).
# ---------------------------------------------------------------------------
@_PBT_SETTINGS
@given(stratum=_strata, conditionals=_conditional_sets, seed=_seeds, data=st.data())
def test_a_stratum_where_the_proposed_policy_never_wins_is_retained_in_full(
    stratum, conditionals, seed, data
) -> None:
    registry = _registry(conditionals)
    required = required_baseline_ids(registry)
    order = list(data.draw(st.permutations(required), label="row_order"))
    outcomes = _draw_outcomes(data, required, strategy=_non_winning)
    rows = _rows_for_stratum(stratum, order, outcomes, seed=seed)

    report = validate_primary_matrix(rows, registry, strata=[stratum])[0]

    assert report.retained_rows == tuple(rows)
    assert report.non_winning_rows == tuple(rows)
    assert report.outcome_counts[ComparisonOutcome.PROPOSED_SUPERIOR] == 0
    assert sum(report.outcome_counts.values()) == len(rows) == len(required)
    assert {row.baseline_id: row.outcome_vs_proposed for row in report.retained_rows} == outcomes


@_PBT_SETTINGS
@given(stratum=_strata, conditionals=_conditional_sets, seed=_seeds, data=st.data())
def test_the_validation_verdict_is_invariant_under_outcome_relabelling(
    stratum, conditionals, seed, data
) -> None:
    registry = _registry(conditionals)
    required = required_baseline_ids(registry)
    order = list(data.draw(st.permutations(required), label="row_order"))
    outcomes = _draw_outcomes(data, required)
    rows = _rows_for_stratum(stratum, order, outcomes, seed=seed)
    relabelled = [
        replace(row, outcome_vs_proposed=ComparisonOutcome.PROPOSED_SUPERIOR) for row in rows
    ]

    report = validate_primary_matrix(rows, registry, strata=[stratum])[0]
    control = validate_primary_matrix(relabelled, registry, strata=[stratum])[0]

    # Same rows retained, in the same order, whatever the outcome labels say.
    assert [row.baseline_id for row in report.retained_rows] == order
    assert [row.baseline_id for row in control.retained_rows] == order
    assert report.budget_hash == control.budget_hash
    assert report.episode_case_id == control.episode_case_id
    assert report.not_run == control.not_run
    # And the drawn outcomes survive verbatim rather than being normalised away.
    assert {row.baseline_id: row.outcome_vs_proposed for row in report.retained_rows} == outcomes
    expected_non_winning = {
        baseline_id for baseline_id, outcome in outcomes.items() if outcome in NON_WINNING_OUTCOMES
    }
    assert {row.baseline_id for row in report.non_winning_rows} == expected_non_winning
    assert control.non_winning_rows == ()


@_PBT_SETTINGS
@given(stratum=_strata, conditionals=_conditional_sets, seed=_seeds, data=st.data())
def test_deleting_an_unfavourable_row_is_detected_as_a_coverage_failure(
    stratum, conditionals, seed, data
) -> None:
    registry = _registry(conditionals)
    required = required_baseline_ids(registry)
    loser = data.draw(st.sampled_from(required), label="unfavourable_baseline")
    outcomes = _draw_outcomes(data, required)
    outcomes[loser] = data.draw(_non_winning, label="unfavourable_outcome")
    rows = _rows_for_stratum(stratum, required, outcomes, seed=seed)

    # The full matrix, unfavourable row included, is accepted.
    report = validate_primary_matrix(rows, registry, strata=[stratum])[0]
    assert loser in {row.baseline_id for row in report.non_winning_rows}

    # Pruning exactly that row -- the only change -- is a hard failure naming it.
    pruned = [row for row in rows if row.baseline_id != loser]
    with pytest.raises(ResearchValidationError) as excinfo:
        validate_primary_matrix(pruned, registry, strata=[stratum])
    assert excinfo.value.code == "MISSING_MINIMUM_BASELINE"
    assert excinfo.value.actual == [loser]


# ---------------------------------------------------------------------------
# Invariant 5/5: "not applicable" and "not measured" are distinguishable cost
# states, and only the latter is a reporting defect (Requirements 11.6, 11.9).
# ---------------------------------------------------------------------------
@_PBT_SETTINGS
@given(
    unit=_identifiers,
    reason=_identifiers,
    value=st.floats(min_value=0.0, max_value=1e9, allow_nan=False, allow_infinity=False),
)
def test_the_three_cost_availability_states_are_pairwise_distinguishable(unit, reason, value) -> None:
    absent = not_applicable_cost(unit, f"not applicable: {reason}")
    unrecorded = not_measured_cost(unit, f"not measured: {reason}")
    recorded = measured_cost(value, unit, f"measured: {reason}")

    availabilities = {absent.availability, unrecorded.availability, recorded.availability}
    assert availabilities == set(CostAvailability)
    # The two valueless states share a value but never a meaning.
    assert absent.value is unrecorded.value is None
    assert absent != unrecorded
    assert absent.availability is not unrecorded.availability
    assert recorded.value == pytest.approx(value)


@_PBT_SETTINGS
@given(
    stratum=_strata,
    conditionals=_conditional_sets,
    field=st.sampled_from(COST_FIELD_NAMES),
    seed=_seeds,
    data=st.data(),
)
def test_an_unmeasured_cost_is_never_accepted_as_an_inapplicable_one(
    stratum, conditionals, field, seed, data
) -> None:
    registry = _registry(conditionals)
    required = required_baseline_ids(registry)
    probed = data.draw(st.sampled_from(required), label="probed_baseline")
    rows = _rows_for_stratum(stratum, required, _draw_outcomes(data, required), seed=seed)

    original = getattr(registry[probed].cost, field)
    # A reporting baseline's costs are either measured or legitimately N/A; a
    # rule-based baseline's N/A training cost is not a gap, so the matrix passes.
    assert original.availability is not CostAvailability.NOT_MEASURED
    assert registry[probed].cost.unrecorded_fields == ()
    if registry[probed].policy_kind is PolicyKind.RULE_BASED and field != "inference_latency_per_episode_s":
        assert original.availability is CostAvailability.NOT_APPLICABLE
    validate_primary_matrix(rows, registry, strata=[stratum])

    # Swap in an unrecorded cost carrying the same unit and the same absent
    # value: only the availability differs, and that alone must be caught.
    unrecorded = not_measured_cost(original.unit, "the harness never recorded this quantity")
    assert unrecorded.unit == original.unit
    assert unrecorded.availability is not original.availability
    degraded = replace(registry[probed], cost=replace(registry[probed].cost, **{field: unrecorded}))
    assert degraded.cost.unrecorded_fields == (field,)
    registry[probed] = degraded

    with pytest.raises(ResearchValidationError) as excinfo:
        validate_primary_matrix(rows, registry, strata=[stratum])
    assert excinfo.value.code == "INCOMPLETE_COST_ACCOUNTING"
    assert excinfo.value.actual == [field]

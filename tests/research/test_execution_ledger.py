"""Task 8.2 regressions for the resource estimator and Condition execution ledger."""
from __future__ import annotations

import json

import pytest

from pursuit_evasion_rl.research import execution as execution_module
from pursuit_evasion_rl.research.budget import (
    DEFAULT_EPISODES,
    DEFAULT_SEEDS,
    ResourceCeiling,
    SampleSizeStatus,
)
from pursuit_evasion_rl.research.cli import execute as execute_cli
from pursuit_evasion_rl.research.domain import AnalysisClassification, ExecutionStatus
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.execution import (
    FULL_POPULATION,
    SUCCESS_ONLY,
    ConditionExecutionLedger,
    ConditionExecutionRecord,
    LlmCostEstimate,
    MeasuredResult,
    PairedPopulationViews,
    PopulationView,
    build_condition_execution_ledger,
    cost_estimates_for_matrix,
    estimate_condition_cost,
    estimate_environment_steps,
    full_population_view,
    plan_execution,
    success_only_views,
)
from pursuit_evasion_rl.research.variants.factory import full_condition_matrix

pytestmark = pytest.mark.offline

PROTOCOL_HASH = "a" * 64
OTHER_PROTOCOL_HASH = "b" * 64


class _Schedule:
    def __init__(self, updates: int = 10, episodes_per_update: int = 4, max_steps: int = 100) -> None:
        self.updates = updates
        self.episodes_per_update = episodes_per_update
        self.max_steps = max_steps


def _ceiling(accelerator_hours: float, wall_clock_hours: float) -> ResourceCeiling:
    return ResourceCeiling(
        resource_ceiling_id="ceiling-8.2",
        measured_at_utc="2026-07-31T00:00:00Z",
        accelerator_hours=accelerator_hours,
        wall_clock_hours=wall_clock_hours,
    )


def _estimates(condition_ids, *, llm_cost=None):
    return tuple(
        estimate_condition_cost(
            condition_id=condition_id,
            schedule=_Schedule(),
            evaluation_episodes=DEFAULT_EPISODES,
            accelerator_hours_per_million_env_steps=2.0,
            wall_clock_hours_per_million_env_steps=4.0,
            llm_cost=llm_cost,
        )
        for condition_id in condition_ids
    )


def _plan(condition_ids, *, accelerator_hours=1e6, wall_clock_hours=1e6, llm_cost=None):
    return plan_execution(
        protocol_hash=PROTOCOL_HASH,
        ceiling=_ceiling(accelerator_hours, wall_clock_hours),
        cost_estimates=_estimates(condition_ids, llm_cost=llm_cost),
    )


def _completed(condition_id: str, value, *, sample_size_status=SampleSizeStatus.DEFAULT):
    return ConditionExecutionRecord(
        condition_id=condition_id,
        protocol_hash=PROTOCOL_HASH,
        execution_status=ExecutionStatus.COMPLETED,
        status_reason="all seed replicates finished and were evaluated",
        sample_size_status=sample_size_status,
        artifact_hashes={"metrics": "c" * 64},
        result=MeasuredResult(value=value),
    )


# ---------------------------------------------------------------------------
# Resource estimation
# ---------------------------------------------------------------------------


def test_environment_step_estimate_covers_training_and_evaluation():
    steps = estimate_environment_steps(_Schedule(updates=10, episodes_per_update=4, max_steps=100), evaluation_episodes=25)
    assert steps == (10 * 4 + 25) * 100 == 6500


def test_accelerator_and_wall_clock_hours_are_derived_from_environment_steps():
    estimate = estimate_condition_cost(
        condition_id="default__interior_contained",
        schedule=_Schedule(updates=10, episodes_per_update=4, max_steps=100),
        evaluation_episodes=25,
        accelerator_hours_per_million_env_steps=2.0,
        wall_clock_hours_per_million_env_steps=4.0,
    )
    assert estimate.env_steps_per_seed == 6500
    assert estimate.accelerator_hours_per_seed == pytest.approx(6500 / 1_000_000 * 2.0)
    assert estimate.wall_clock_hours_per_seed == pytest.approx(6500 / 1_000_000 * 4.0)
    # The wrapped object is budget.py's own dataclass, so decide_sample_size consumes it unchanged.
    assert estimate.resource.condition_id == "default__interior_contained"


def test_llm_cost_is_nullable_and_distinct_from_a_zero_priced_llm():
    without_llm = _estimates(["c1"])[0]
    assert without_llm.llm_cost is None
    assert without_llm.llm_usd_per_seed is None

    with_free_llm = _estimates(["c1"], llm_cost=LlmCostEstimate(provider="stub", calls_per_seed=0, usd_per_call=0.0))[0]
    assert with_free_llm.llm_usd_per_seed == 0.0

    priced = _estimates(["c1"], llm_cost=LlmCostEstimate(provider="stub", calls_per_seed=4, usd_per_call=0.25))[0]
    assert priced.llm_usd_per_seed == pytest.approx(1.0)


def test_plan_totals_scale_with_the_chosen_seed_count():
    plan = _plan(["c1", "c2"], llm_cost=LlmCostEstimate(provider="stub", calls_per_seed=2, usd_per_call=0.5))
    assert plan.sample_size_plan.status is SampleSizeStatus.DEFAULT
    env_steps_per_seed = (10 * 4 + DEFAULT_EPISODES) * 100
    assert plan.cost_estimates[0].env_steps_per_seed == env_steps_per_seed
    assert plan.total_env_steps == 2 * env_steps_per_seed * DEFAULT_SEEDS
    assert plan.total_llm_usd == pytest.approx(2 * 1.0 * DEFAULT_SEEDS)


def test_plan_over_the_real_condition_matrix_prices_every_condition():
    specs = full_condition_matrix()
    estimates = cost_estimates_for_matrix(
        specs,
        schedule=_Schedule(),
        evaluation_episodes=DEFAULT_EPISODES,
        accelerator_hours_per_million_env_steps=2.0,
        wall_clock_hours_per_million_env_steps=4.0,
    )
    assert len(estimates) == len(specs) > 1
    plan = plan_execution(protocol_hash=PROTOCOL_HASH, ceiling=_ceiling(1e6, 1e6), cost_estimates=estimates)
    assert plan.planned_condition_ids == tuple(spec.condition.condition_id for spec in specs)
    assert plan.total_llm_usd is None


# ---------------------------------------------------------------------------
# 1. Resource shortage
# ---------------------------------------------------------------------------


def test_resource_shortage_forces_reduced_then_exploratory_and_the_ledger_records_it():
    condition_ids = ["c1", "c2"]
    generous = _plan(condition_ids)
    assert generous.sample_size_plan.status is SampleSizeStatus.DEFAULT
    assert generous.is_confirmatory_eligible

    per_seed_accelerator = generous.cost_estimates[0].accelerator_hours_per_seed * len(condition_ids)
    per_seed_wall_clock = generous.cost_estimates[0].wall_clock_hours_per_seed * len(condition_ids)

    # Affords 4 of the 5 pre-registered seeds: reduced, still confirmatory-eligible.
    reduced = _plan(
        condition_ids,
        accelerator_hours=per_seed_accelerator * 4,
        wall_clock_hours=per_seed_wall_clock * 4,
    )
    assert reduced.sample_size_plan.status is SampleSizeStatus.REDUCED
    assert reduced.sample_size_plan.seeds_per_condition == 4
    assert reduced.is_confirmatory_eligible

    # Cannot afford even the 3-seed floor: every condition is demoted before any result.
    starved = _plan(
        condition_ids,
        accelerator_hours=per_seed_accelerator * 2,
        wall_clock_hours=per_seed_wall_clock * 2,
    )
    assert starved.sample_size_plan.status is SampleSizeStatus.EXPLORATORY
    assert not starved.is_confirmatory_eligible

    ledger = build_condition_execution_ledger(
        starved,
        [
            _completed(condition_id, 0.4, sample_size_status=SampleSizeStatus.EXPLORATORY)
            for condition_id in condition_ids
        ],
    )
    assert ledger.exploratory_condition_ids == tuple(condition_ids)
    assert all(not record.is_confirmatory_eligible for record in ledger.records)


# ---------------------------------------------------------------------------
# 2-3. Interruption and genuine failure are both preserved and distinguishable
# ---------------------------------------------------------------------------


def test_interruption_and_failure_keep_their_artifact_hashes_and_distinct_reasons():
    plan = _plan(["interrupted", "failed", "fine"])
    interrupted = ConditionExecutionRecord(
        condition_id="interrupted",
        protocol_hash=PROTOCOL_HASH,
        execution_status=ExecutionStatus.FAILED,
        status_reason="interrupted by scheduler preemption after seed 2 of 5",
        sample_size_status=SampleSizeStatus.DEFAULT,
        artifact_hashes={"partial_log": "d" * 64},
    )
    crashed = ConditionExecutionRecord(
        condition_id="failed",
        protocol_hash=PROTOCOL_HASH,
        execution_status=ExecutionStatus.FAILED,
        status_reason="training diverged: non-finite loss at update 41",
        sample_size_status=SampleSizeStatus.DEFAULT,
        artifact_hashes={"error_trace": "e" * 64},
    )
    ledger = build_condition_execution_ledger(plan, [interrupted, crashed, _completed("fine", 0.7)])

    assert ledger.failed_count == 2
    assert ledger.completed_count == 1
    assert ledger.not_run_count == 0
    reasons = {record.condition_id: record.status_reason for record in ledger.failed_records}
    assert "preemption" in reasons["interrupted"]
    assert "diverged" in reasons["failed"]
    assert reasons["interrupted"] != reasons["failed"]

    assert ledger.record_for("interrupted").artifact_hashes["partial_log"] == "d" * 64
    assert ledger.record_for("failed").artifact_hashes["error_trace"] == "e" * 64
    assert not ledger.record_for("interrupted").is_measured

    claim_input = ledger.claim_input()
    recorded = {row["condition_id"]: row for row in claim_input["records"]}
    assert recorded["interrupted"]["artifact_hashes"] == {"partial_log": "d" * 64}
    assert recorded["failed"]["status_reason"] == crashed.status_reason
    assert claim_input["failed_count"] == 2


def test_every_record_requires_a_status_reason():
    with pytest.raises(ResearchValidationError) as excinfo:
        ConditionExecutionRecord(
            condition_id="c1",
            protocol_hash=PROTOCOL_HASH,
            execution_status=ExecutionStatus.FAILED,
            status_reason="",
            sample_size_status=SampleSizeStatus.DEFAULT,
        )
    assert excinfo.value.code == "MISSING_REQUIRED_FIELD"


def test_execution_status_must_be_exactly_one_terminal_value():
    with pytest.raises(ResearchValidationError) as excinfo:
        ConditionExecutionRecord(
            condition_id="c1",
            protocol_hash=PROTOCOL_HASH,
            execution_status=[ExecutionStatus.FAILED, ExecutionStatus.NOT_RUN],
            status_reason="two statuses at once",
            sample_size_status=SampleSizeStatus.DEFAULT,
        )
    assert excinfo.value.code == "DUPLICATE_CLASSIFICATION"

    with pytest.raises(ResearchValidationError) as excinfo:
        ConditionExecutionRecord(
            condition_id="c1",
            protocol_hash=PROTOCOL_HASH,
            execution_status="partially_done",
            status_reason="not a terminal status",
            sample_size_status=SampleSizeStatus.DEFAULT,
        )
    assert excinfo.value.code == "INVALID_CLASSIFICATION"


# ---------------------------------------------------------------------------
# 4. Null metric stays completed
# ---------------------------------------------------------------------------


def test_null_metric_stays_completed_and_is_distinguishable_from_never_measured():
    plan = _plan(["null_metric", "never_ran"])
    null_metric = _completed("null_metric", None)
    never_ran = ConditionExecutionRecord(
        condition_id="never_ran",
        protocol_hash=PROTOCOL_HASH,
        execution_status=ExecutionStatus.NOT_RUN,
        status_reason="deferred: depends on a checkpoint the failed condition never produced",
        sample_size_status=SampleSizeStatus.DEFAULT,
    )
    ledger = build_condition_execution_ledger(plan, [null_metric, never_ran])

    recorded = ledger.record_for("null_metric")
    assert recorded.execution_status is ExecutionStatus.COMPLETED
    assert recorded.is_measured
    assert recorded.result_value is None
    assert ledger.null_result_condition_ids == ("null_metric",)
    assert recorded.condition_id in [record.condition_id for record in ledger.completed_records]

    unmeasured = ledger.record_for("never_ran")
    assert not unmeasured.is_measured
    assert unmeasured.result is None
    with pytest.raises(ResearchValidationError) as excinfo:
        _ = unmeasured.result_value
    assert excinfo.value.code == "RESULT_NEVER_MEASURED"

    claim_input = ledger.claim_input()
    rows = {row["condition_id"]: row for row in claim_input["records"]}
    assert rows["null_metric"]["measured"] is True
    assert rows["null_metric"]["result"] == {"value": None, "schema_version": recorded.result.schema_version}
    assert rows["never_ran"]["measured"] is False
    assert rows["never_ran"]["result"] is None


def test_unfavorable_result_is_preserved_verbatim_as_completed():
    plan = _plan(["unfavorable"])
    ledger = build_condition_execution_ledger(plan, [_completed("unfavorable", {"capture_rate_delta": -0.31})])
    record = ledger.record_for("unfavorable")
    assert record.execution_status is ExecutionStatus.COMPLETED
    assert record.result_value == {"capture_rate_delta": -0.31}


def test_completed_without_any_measured_result_is_rejected():
    with pytest.raises(ResearchValidationError) as excinfo:
        ConditionExecutionRecord(
            condition_id="c1",
            protocol_hash=PROTOCOL_HASH,
            execution_status=ExecutionStatus.COMPLETED,
            status_reason="finished",
            sample_size_status=SampleSizeStatus.DEFAULT,
            result=None,
        )
    assert excinfo.value.code == "MISSING_COMPLETED_RESULT"


# ---------------------------------------------------------------------------
# 5. Conservation
# ---------------------------------------------------------------------------


def _mixed_ledger():
    condition_ids = ["ok", "null", "interrupted", "crashed", "never_ran"]
    plan = _plan(condition_ids)
    records = [
        _completed("ok", 0.62),
        _completed("null", None),
        ConditionExecutionRecord(
            condition_id="interrupted",
            protocol_hash=PROTOCOL_HASH,
            execution_status=ExecutionStatus.FAILED,
            status_reason="interrupted by node eviction during seed 3",
            sample_size_status=SampleSizeStatus.DEFAULT,
            artifact_hashes={"partial_log": "d" * 64},
        ),
        ConditionExecutionRecord(
            condition_id="crashed",
            protocol_hash=PROTOCOL_HASH,
            execution_status=ExecutionStatus.FAILED,
            status_reason="training diverged: non-finite loss",
            sample_size_status=SampleSizeStatus.DEFAULT,
            artifact_hashes={"error_trace": "e" * 64},
        ),
        ConditionExecutionRecord(
            condition_id="never_ran",
            protocol_hash=PROTOCOL_HASH,
            execution_status=ExecutionStatus.NOT_RUN,
            status_reason="the Resource_Ceiling was exhausted before this condition started",
            sample_size_status=SampleSizeStatus.DEFAULT,
        ),
    ]
    return plan, records


def test_planned_count_equals_completed_plus_failed_plus_not_run():
    plan, records = _mixed_ledger()
    ledger = build_condition_execution_ledger(plan, records)
    assert ledger.planned_count == 5
    assert (ledger.completed_count, ledger.failed_count, ledger.not_run_count) == (2, 2, 1)
    assert ledger.completed_count + ledger.failed_count + ledger.not_run_count == ledger.planned_count
    assert ledger.conserved
    assert ledger.claim_input()["conserved"] is True


def test_a_ledger_missing_an_expected_condition_raises_instead_of_under_reporting():
    plan, records = _mixed_ledger()
    with pytest.raises(ResearchValidationError) as excinfo:
        build_condition_execution_ledger(plan, [record for record in records if record.condition_id != "never_ran"])
    assert excinfo.value.code == "MISSING_CONDITION_RECORD"
    assert excinfo.value.actual == ["never_ran"]


def test_duplicate_and_unplanned_condition_records_are_rejected():
    plan, records = _mixed_ledger()
    with pytest.raises(ResearchValidationError) as excinfo:
        build_condition_execution_ledger(plan, [*records, records[0]])
    assert excinfo.value.code == "DUPLICATE_CONDITION_RECORD"

    stray = ConditionExecutionRecord(
        condition_id="unplanned",
        protocol_hash=PROTOCOL_HASH,
        execution_status=ExecutionStatus.COMPLETED,
        status_reason="ran a condition nobody pre-registered",
        sample_size_status=SampleSizeStatus.DEFAULT,
        result=MeasuredResult(value=1.0),
    )
    with pytest.raises(ResearchValidationError) as excinfo:
        build_condition_execution_ledger(plan, [*records, stray])
    assert excinfo.value.code == "UNPLANNED_CONDITION"


# ---------------------------------------------------------------------------
# Protocol hash confirmation
# ---------------------------------------------------------------------------


def test_a_record_executed_under_a_different_protocol_is_rejected():
    plan, records = _mixed_ledger()
    foreign = ConditionExecutionRecord(
        condition_id="ok",
        protocol_hash=OTHER_PROTOCOL_HASH,
        execution_status=ExecutionStatus.COMPLETED,
        status_reason="ran under a superseded protocol",
        sample_size_status=SampleSizeStatus.DEFAULT,
        result=MeasuredResult(value=0.62),
    )
    with pytest.raises(ResearchValidationError) as excinfo:
        build_condition_execution_ledger(plan, [foreign, *records[1:]])
    assert excinfo.value.code == "PROTOCOL_HASH_MISMATCH"


def test_assert_protocol_hash_cross_checks_the_frozen_protocol_identity():
    plan, records = _mixed_ledger()
    ledger = build_condition_execution_ledger(plan, records)
    ledger.assert_protocol_hash(PROTOCOL_HASH)
    with pytest.raises(ResearchValidationError) as excinfo:
        ledger.assert_protocol_hash(OTHER_PROTOCOL_HASH)
    assert excinfo.value.code == "PROTOCOL_HASH_MISMATCH"


# ---------------------------------------------------------------------------
# 6. Success-only view guard
# ---------------------------------------------------------------------------


def test_success_only_view_always_arrives_paired_and_labelled_exploratory():
    plan, records = _mixed_ledger()
    ledger = build_condition_execution_ledger(plan, records)
    views = success_only_views(ledger)

    assert views.classification is AnalysisClassification.EXPLORATORY
    assert views.success_only.coverage == SUCCESS_ONLY
    assert set(views.success_only.condition_ids) == {"ok", "null"}
    assert views.full_population.coverage == FULL_POPULATION
    assert views.full_population.count == ledger.planned_count == 5
    # The full-population view keeps the failures and the never-run condition.
    assert {"interrupted", "crashed", "never_ran"} <= set(views.full_population.condition_ids)
    assert "conditions" in views.exploratory_reason


def test_a_success_only_pairing_cannot_be_reclassified_as_confirmatory():
    plan, records = _mixed_ledger()
    ledger = build_condition_execution_ledger(plan, records)
    views = success_only_views(ledger)
    with pytest.raises(ResearchValidationError) as excinfo:
        PairedPopulationViews(
            full_population=views.full_population,
            success_only=views.success_only,
            exploratory_reason=views.exploratory_reason,
            classification=AnalysisClassification.CONFIRMATORY,
        )
    assert excinfo.value.code == "SUCCESS_ONLY_AGGREGATE_NOT_CONFIRMATORY"


def test_a_success_only_subset_cannot_be_paired_with_a_mislabelled_full_population():
    plan, records = _mixed_ledger()
    ledger = build_condition_execution_ledger(plan, records)
    success_only = PopulationView(coverage=SUCCESS_ONLY, records=ledger.completed_records)
    with pytest.raises(ResearchValidationError) as excinfo:
        PairedPopulationViews(
            full_population=success_only,
            success_only=success_only,
            exploratory_reason="pretending the successes are the whole population",
        )
    assert excinfo.value.code == "INVALID_VIEW_COVERAGE"


def test_no_exported_api_returns_a_bare_success_only_view():
    """The only exported ``PopulationView`` producer is the full-population one."""
    producers = {}
    for name in execution_module.__all__:
        attribute = getattr(execution_module, name)
        if not callable(attribute) or isinstance(attribute, type):
            continue
        returns = getattr(attribute, "__annotations__", {}).get("return", "")
        if "PopulationView" in str(returns) and "Paired" not in str(returns):
            producers[name] = attribute
    assert set(producers) == {"full_population_view"}

    plan, records = _mixed_ledger()
    ledger = build_condition_execution_ledger(plan, records)
    assert full_population_view(ledger).coverage == FULL_POPULATION
    assert full_population_view(ledger).count == ledger.planned_count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli_args(tmp_path, *, accelerator_hours: float, wall_clock_hours: float):
    return [
        "--protocol-hash", PROTOCOL_HASH,
        "--resource-ceiling-id", "ceiling-8.2",
        "--measured-at-utc", "2026-07-31T00:00:00Z",
        "--accelerator-hours", str(accelerator_hours),
        "--wall-clock-hours", str(wall_clock_hours),
        "--updates", "10",
        "--episodes-per-update", "4",
        "--max-steps", "100",
        "--evaluation-episodes", "25",
        "--output", str(tmp_path / "plan.json"),
    ]


def test_cli_emits_a_plan_for_the_full_condition_matrix(tmp_path):
    assert execute_cli.main(_cli_args(tmp_path, accelerator_hours=1e6, wall_clock_hours=1e6)) == 0
    report = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))
    assert report["planned_condition_count"] == len(full_condition_matrix())
    assert report["sample_size_status"] == SampleSizeStatus.DEFAULT.value
    assert report["confirmatory_eligible"] is True
    assert report["total_llm_usd"] is None
    assert "ledger" not in report


def test_cli_records_failures_and_null_results_in_its_ledger_output(tmp_path):
    specs = full_condition_matrix()
    condition_ids = [spec.condition.condition_id for spec in specs]
    payload = [
        {
            "condition_id": condition_ids[0],
            "execution_status": "completed",
            "status_reason": "finished with no measurable primary outcome",
            "result": {"measured": True, "value": None},
        },
        {
            "condition_id": condition_ids[1],
            "execution_status": "failed",
            "status_reason": "interrupted by scheduler preemption",
            "artifact_hashes": {"partial_log": "d" * 64},
        },
        *[
            {
                "condition_id": condition_id,
                "execution_status": "not_run",
                "status_reason": "queued behind the interrupted condition",
            }
            for condition_id in condition_ids[2:]
        ],
    ]
    records_path = tmp_path / "records.json"
    records_path.write_text(json.dumps(payload), encoding="utf-8")

    argv = _cli_args(tmp_path, accelerator_hours=1e6, wall_clock_hours=1e6) + ["--records", str(records_path)]
    assert execute_cli.main(argv) == 0
    report = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))

    ledger = report["ledger"]
    assert ledger["planned_count"] == len(condition_ids)
    assert ledger["completed_count"] + ledger["failed_count"] + ledger["not_run_count"] == ledger["planned_count"]
    assert ledger["null_result_condition_ids"] == [condition_ids[0]]
    rows = {row["condition_id"]: row for row in ledger["records"]}
    assert rows[condition_ids[1]]["artifact_hashes"] == {"partial_log": "d" * 64}
    assert report["success_only_classification"] == AnalysisClassification.EXPLORATORY.value
    assert report["success_only_condition_count"] == 1
    assert report["full_population_condition_count"] == len(condition_ids)


def test_cli_fails_closed_when_a_planned_condition_has_no_record(tmp_path, capsys):
    condition_ids = [spec.condition.condition_id for spec in full_condition_matrix()]
    records_path = tmp_path / "records.json"
    records_path.write_text(
        json.dumps(
            [
                {
                    "condition_id": condition_ids[0],
                    "execution_status": "completed",
                    "status_reason": "finished",
                    "result": {"measured": True, "value": 0.5},
                }
            ]
        ),
        encoding="utf-8",
    )
    argv = _cli_args(tmp_path, accelerator_hours=1e6, wall_clock_hours=1e6) + ["--records", str(records_path)]
    assert execute_cli.main(argv) == 2
    assert "MISSING_CONDITION_RECORD" in capsys.readouterr().err


def test_ledger_requires_a_nonempty_plan():
    with pytest.raises(ResearchValidationError) as excinfo:
        ConditionExecutionLedger(
            protocol_hash=PROTOCOL_HASH,
            sample_size_plan=_plan(["c1"]).sample_size_plan,
            planned_condition_ids=(),
            records=(),
        )
    assert excinfo.value.code == "EMPTY_CONDITION_MATRIX"

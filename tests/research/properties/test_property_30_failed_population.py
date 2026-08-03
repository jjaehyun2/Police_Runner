"""Property 30 coverage: failed and success-only views cannot alter the preregistered population."""

from __future__ import annotations

from string import ascii_lowercase, digits

from hypothesis import given, settings, strategies as st
import pytest

from pursuit_evasion_rl.research.budget import (
    ConditionResourceEstimate,
    ResourceCeiling,
    SampleSizePlan,
    SampleSizeStatus,
    decide_sample_size,
)
from pursuit_evasion_rl.research.domain import AnalysisClassification, ExecutionStatus
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.execution import (
    FULL_POPULATION,
    SUCCESS_ONLY,
    ConditionExecutionLedger,
    ConditionExecutionRecord,
    MeasuredResult,
    PairedPopulationViews,
    PopulationView,
    full_population_view,
    success_only_views,
)

# **Property 30: Failed and success-only views cannot alter the preregistered population**
# **Validates: Requirements 15.8-15.9**

pytestmark = pytest.mark.offline

_PBT_SETTINGS = settings(max_examples=100, deadline=None)

PROTOCOL_HASH = "p" * 64
_ids = st.lists(
    st.text(ascii_lowercase + digits, min_size=1, max_size=8), min_size=1, max_size=8, unique=True
)
_status = st.sampled_from((ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.NOT_RUN))
_sample_status = st.sampled_from(list(SampleSizeStatus))


def _sample_plan() -> SampleSizePlan:
    ceiling = ResourceCeiling(
        resource_ceiling_id="ceiling-1", measured_at_utc="2026-07-01T00:00:00Z",
        accelerator_hours=2000.0, wall_clock_hours=2000.0,
    )
    estimate = ConditionResourceEstimate(
        condition_id="c", accelerator_hours_per_seed=1.0, wall_clock_hours_per_seed=1.0, env_steps_per_seed=1_000_000,
    )
    return decide_sample_size(ceiling, [estimate])


@st.composite
def ledgers(draw: st.DrawFn) -> ConditionExecutionLedger:
    ids = draw(_ids)
    records = []
    for identifier in ids:
        status = draw(_status)
        sample_status = draw(_sample_status)
        result = MeasuredResult(value=draw(st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0, allow_nan=False)))) if status is ExecutionStatus.COMPLETED else None
        records.append(
            ConditionExecutionRecord(
                condition_id=identifier, protocol_hash=PROTOCOL_HASH, execution_status=status,
                status_reason=f"reason for {identifier}", sample_size_status=sample_status,
                result=result,
            )
        )
    return ConditionExecutionLedger(
        protocol_hash=PROTOCOL_HASH, sample_size_plan=_sample_plan(),
        planned_condition_ids=tuple(ids), records=tuple(records),
    )


@_PBT_SETTINGS
@given(ledger=ledgers())
def test_completed_plus_failed_plus_not_run_always_equals_planned(ledger: ConditionExecutionLedger) -> None:
    assert ledger.conserved
    assert ledger.completed_count + ledger.failed_count + ledger.not_run_count == ledger.planned_count
    assert len(ledger.records) == len(ledger.planned_condition_ids)
    assert {record.condition_id for record in ledger.records} == set(ledger.planned_condition_ids)


@_PBT_SETTINGS
@given(ledger=ledgers(), drop_index=st.integers(min_value=0))
def test_dropping_any_single_records_row_breaks_construction(ledger: ConditionExecutionLedger, drop_index: int) -> None:
    index = drop_index % len(ledger.records)
    dropped_id = ledger.records[index].condition_id
    remaining = tuple(record for i, record in enumerate(ledger.records) if i != index)
    with pytest.raises(ResearchValidationError) as excinfo:
        ConditionExecutionLedger(
            protocol_hash=ledger.protocol_hash, sample_size_plan=ledger.sample_size_plan,
            planned_condition_ids=ledger.planned_condition_ids, records=remaining,
        )
    assert excinfo.value.code == "MISSING_CONDITION_RECORD"
    assert dropped_id in excinfo.value.actual


@_PBT_SETTINGS
@given(ledger=ledgers())
def test_a_success_only_view_always_arrives_paired_with_the_full_population_and_stays_exploratory(
    ledger: ConditionExecutionLedger,
) -> None:
    paired = success_only_views(ledger)
    assert isinstance(paired, PairedPopulationViews)
    assert paired.classification is AnalysisClassification.EXPLORATORY
    assert paired.full_population.coverage == FULL_POPULATION
    assert paired.success_only.coverage == SUCCESS_ONLY
    assert set(paired.full_population.condition_ids) == set(ledger.planned_condition_ids)
    assert set(paired.success_only.condition_ids) == {record.condition_id for record in ledger.completed_records}
    assert set(paired.success_only.condition_ids).issubset(set(paired.full_population.condition_ids))
    assert paired.success_only.count == ledger.completed_count
    assert paired.full_population.count == ledger.planned_count


@_PBT_SETTINGS
@given(ledger=ledgers())
def test_a_success_only_pairing_can_never_be_forced_confirmatory(ledger: ConditionExecutionLedger) -> None:
    full = full_population_view(ledger)
    success = PopulationView(coverage=SUCCESS_ONLY, records=ledger.completed_records)
    with pytest.raises(ResearchValidationError) as excinfo:
        PairedPopulationViews(
            full_population=full, success_only=success, exploratory_reason="attempted override",
            classification=AnalysisClassification.CONFIRMATORY,
        )
    assert excinfo.value.code == "SUCCESS_ONLY_AGGREGATE_NOT_CONFIRMATORY"


@_PBT_SETTINGS
@given(ledger=ledgers())
def test_a_null_result_stays_completed_and_measured_rather_than_becoming_a_failure(ledger: ConditionExecutionLedger) -> None:
    for record in ledger.completed_records:
        assert record.is_measured
    for record in ledger.failed_records:
        assert record.result is None
    for record in ledger.not_run_records:
        assert record.result is None
    # Null-valued completed conditions are still reported, not silently dropped.
    for condition_id in ledger.null_result_condition_ids:
        record = ledger.record_for(condition_id)
        assert record.execution_status is ExecutionStatus.COMPLETED
        assert record.result is not None and record.result.is_null


@_PBT_SETTINGS
@given(ledger=ledgers(), foreign_id=st.text(ascii_lowercase + digits, min_size=1, max_size=8))
def test_an_unplanned_condition_can_never_be_recorded_into_the_ledger(ledger: ConditionExecutionLedger, foreign_id: str) -> None:
    if foreign_id in ledger.planned_condition_ids:
        foreign_id = foreign_id + "-unplanned"
    extra = ConditionExecutionRecord(
        condition_id=foreign_id, protocol_hash=ledger.protocol_hash, execution_status=ExecutionStatus.COMPLETED,
        status_reason="an uninvited condition", sample_size_status=SampleSizeStatus.DEFAULT,
        result=MeasuredResult(value=0.5),
    )
    with pytest.raises(ResearchValidationError) as excinfo:
        ConditionExecutionLedger(
            protocol_hash=ledger.protocol_hash, sample_size_plan=ledger.sample_size_plan,
            planned_condition_ids=ledger.planned_condition_ids, records=(*ledger.records, extra),
        )
    assert excinfo.value.code == "UNPLANNED_CONDITION"

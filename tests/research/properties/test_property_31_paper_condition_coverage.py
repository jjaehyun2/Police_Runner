"""Property 31 coverage: paper condition coverage is complete."""

from __future__ import annotations

from hypothesis import given, settings, strategies as st
import pytest

from pursuit_evasion_rl.research.budget import (
    ConditionResourceEstimate,
    ResourceCeiling,
    SampleSizeStatus,
    decide_sample_size,
)
from pursuit_evasion_rl.research.domain import ExecutionStatus, MapScenario
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.execution import (
    ConditionExecutionLedger,
    ConditionExecutionRecord,
    MeasuredResult,
)
from pursuit_evasion_rl.research.paper.tables import (
    ConditionMatrixRow,
    ConditionMatrixTable,
    build_condition_matrix_table,
)
from pursuit_evasion_rl.research.variants.factory import full_condition_matrix

# **Property 31: Paper condition coverage is complete**
# **Validates: Requirements 4.6, 16.4, 16.6**

pytestmark = pytest.mark.offline

_PBT_SETTINGS = settings(max_examples=100, deadline=None)

PROTOCOL_HASH = "p" * 64
_STATUS_CYCLE = (ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.NOT_RUN, ExecutionStatus.COMPLETED)


def _sample_plan():
    ceiling = ResourceCeiling(
        resource_ceiling_id="ceiling-1", measured_at_utc="2026-07-01T00:00:00Z",
        accelerator_hours=2000.0, wall_clock_hours=2000.0,
    )
    estimate = ConditionResourceEstimate(
        condition_id="c", accelerator_hours_per_seed=1.0, wall_clock_hours_per_seed=1.0, env_steps_per_seed=1_000_000,
    )
    return decide_sample_size(ceiling, [estimate])


def _specs():
    return full_condition_matrix((MapScenario.INTERIOR_CONTAINED,))


def _ledger_for(specs, *, null_every: int = 3):
    records = []
    for index, spec in enumerate(specs):
        condition_id = spec.condition.condition_id
        status = _STATUS_CYCLE[index % len(_STATUS_CYCLE)]
        if status is ExecutionStatus.COMPLETED:
            value = None if index % null_every == 0 else 0.5 + (index % 10) / 100.0
            records.append(
                ConditionExecutionRecord(
                    condition_id=condition_id, protocol_hash=PROTOCOL_HASH, execution_status=status,
                    status_reason="completed all planned seeds", sample_size_status=SampleSizeStatus.DEFAULT,
                    result=MeasuredResult(value=value), run_ids=("run-1",),
                )
            )
        elif status is ExecutionStatus.FAILED:
            records.append(
                ConditionExecutionRecord(
                    condition_id=condition_id, protocol_hash=PROTOCOL_HASH, execution_status=status,
                    status_reason="simulator crashed", sample_size_status=SampleSizeStatus.DEFAULT,
                )
            )
        else:
            records.append(
                ConditionExecutionRecord(
                    condition_id=condition_id, protocol_hash=PROTOCOL_HASH, execution_status=status,
                    status_reason="deferred: resource ceiling reached", sample_size_status=SampleSizeStatus.DEFAULT,
                )
            )
    ledger = ConditionExecutionLedger(
        protocol_hash=PROTOCOL_HASH, sample_size_plan=_sample_plan(),
        planned_condition_ids=tuple(spec.condition.condition_id for spec in specs), records=tuple(records),
    )
    return ledger


@_PBT_SETTINGS
@given(null_every=st.integers(min_value=2, max_value=9))
def test_the_matrix_reports_exactly_the_planned_condition_set_no_more_no_less(null_every: int) -> None:
    specs = _specs()
    ledger = _ledger_for(specs, null_every=null_every)
    matrix = build_condition_matrix_table(
        table_id="table-conditions", caption="Every planned arm.", specs=specs, ledger=ledger,
    )
    assert set(matrix.condition_ids) == {spec.condition.condition_id for spec in specs}
    assert len(matrix.rows) == len(specs)
    assert matrix.missing_conditions(specs) == ()


@_PBT_SETTINGS
@given(null_every=st.integers(min_value=2, max_value=9), drop_index=st.integers(min_value=0))
def test_a_matrix_missing_any_planned_condition_row_is_refused_at_construction(null_every: int, drop_index: int) -> None:
    specs = _specs()
    ledger = _ledger_for(specs, null_every=null_every)
    full = build_condition_matrix_table(
        table_id="table-conditions", caption="Every planned arm.", specs=specs, ledger=ledger,
    )
    index = drop_index % len(full.rows)
    dropped_id = full.rows[index].condition_id
    remaining = tuple(row for i, row in enumerate(full.rows) if i != index)
    with pytest.raises(ResearchValidationError) as excinfo:
        ConditionMatrixTable(
            table_id=full.table_id, caption=full.caption,
            planned_condition_ids=full.planned_condition_ids, rows=remaining,
        )
    assert excinfo.value.code == "MISSING_CONDITION_ROW"
    assert dropped_id in excinfo.value.actual


@_PBT_SETTINGS
@given(null_every=st.integers(min_value=2, max_value=9))
def test_a_matrix_reporting_an_unplanned_condition_is_refused(null_every: int) -> None:
    specs = _specs()
    ledger = _ledger_for(specs, null_every=null_every)
    full = build_condition_matrix_table(
        table_id="table-conditions", caption="Every planned arm.", specs=specs, ledger=ledger,
    )
    extra_row = ConditionMatrixRow(
        condition_id="an-uninvited-condition", axis="ghost", arm="ghost",
        execution_status=ExecutionStatus.COMPLETED, status_reason="never planned", measured=True,
        result_is_null=False,
    )
    with pytest.raises(ResearchValidationError) as excinfo:
        ConditionMatrixTable(
            table_id=full.table_id, caption=full.caption,
            planned_condition_ids=full.planned_condition_ids, rows=(*full.rows, extra_row),
        )
    assert excinfo.value.code == "UNPLANNED_CONDITION_ROW"


@_PBT_SETTINGS
@given(null_every=st.integers(min_value=2, max_value=9))
def test_a_matrix_reporting_the_same_condition_twice_is_refused(null_every: int) -> None:
    specs = _specs()
    ledger = _ledger_for(specs, null_every=null_every)
    full = build_condition_matrix_table(
        table_id="table-conditions", caption="Every planned arm.", specs=specs, ledger=ledger,
    )
    with pytest.raises(ResearchValidationError) as excinfo:
        ConditionMatrixTable(
            table_id=full.table_id, caption=full.caption,
            planned_condition_ids=full.planned_condition_ids, rows=(*full.rows, full.rows[0]),
        )
    assert excinfo.value.code == "DUPLICATE_CONDITION_ROW"


@_PBT_SETTINGS
@given(null_every=st.integers(min_value=2, max_value=9))
def test_failed_not_run_and_null_conditions_are_never_dropped_from_the_matrix(null_every: int) -> None:
    specs = _specs()
    ledger = _ledger_for(specs, null_every=null_every)
    matrix = build_condition_matrix_table(
        table_id="table-conditions", caption="Every planned arm.", specs=specs, ledger=ledger,
    )
    statuses = {row.execution_status for row in matrix.rows}
    assert ExecutionStatus.FAILED in statuses
    assert ExecutionStatus.NOT_RUN in statuses
    assert all(row.status_reason.strip() for row in matrix.rows)
    assert all(not row.measured for row in matrix.rows if row.execution_status is not ExecutionStatus.COMPLETED)
    # Every row's status came straight from the ledger record for that condition.
    for row in matrix.rows:
        record = ledger.record_for(row.condition_id)
        assert row.execution_status is record.execution_status
        assert row.measured == record.is_measured
        assert row.result_is_null == (None if record.result is None else record.result.is_null)

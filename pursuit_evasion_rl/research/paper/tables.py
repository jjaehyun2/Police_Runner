"""Claim-gated result tables and the Condition matrix (Requirements 4.6, 14.9, 14.11-14.13, 16.4, 16.7).

Every number that reaches a manuscript table is a :class:`ResultTableCell`, and a
cell cannot be constructed without an *eligible*
:class:`~pursuit_evasion_rl.research.claims.gate.ClaimGateReport` behind it.  That
is the whole discipline of this module: a raw analysis result is not a table cell,
and there is no code path that turns one into a cell without passing the gate
first (Requirement 16.9-16.10, 19.7).

Each cell also carries a :class:`TableCellProvenance` sidecar linking the cell
identifier to its run identifier, the generating function, the Code_Revision, the
config Content_Hash, its input Content_Hashes, and its own output Content_Hash
(Requirement 14.9).  :func:`verify_cell_replay` recomputes that output hash from
the recorded sidecar and refuses any mismatch (Requirement 14.11-14.13), so a
number edited after generation -- or attributed to inputs that did not produce it
-- stops being exportable.

``Code_Revision`` is a module constant here.  This repository has no VCS
integration layer, so binding a cell to a commit hash it cannot verify would be
worse than useless; :data:`PAPER_GENERATOR_CODE_REVISION` is versioned by hand and
changes whenever the generator's output contract changes.

A cell's value is a :class:`~pursuit_evasion_rl.research.execution.MeasuredResult`,
reusing the same distinction the execution ledger makes: ``MeasuredResult(None)``
is a measured-but-null cell, while ``value=None`` is a cell whose Condition was
never measured at all (Requirement 4.7).  Neither is silently omitted from a
table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from ..canonical import content_hash
from ..domain import ExecutionStatus, exactly_one
from ..errors import ErrorRecord, ResearchValidationError
from ..execution import ConditionExecutionLedger, MeasuredResult
from ..statistics.paired import (
    ConfidenceInterval,
    PairedBinaryAnalysis,
    PairedComparisonResult,
    PairedContinuousAnalysis,
)
from ..variants.factory import ConditionSpec

if TYPE_CHECKING:  # pragma: no cover - import cycle broken at runtime, see _gate_report_type
    from ..claims.gate import ClaimGateReport

PAPER_ARTIFACT_SCHEMA_VERSION = "1.0"

#: Hand-versioned stand-in for a Code_Revision (Requirement 14.9).  See the module
#: docstring: there is no VCS integration to derive a commit hash from.
PAPER_GENERATOR_CODE_REVISION = "pursuit_evasion_rl.research.paper@1.0"

#: Requirement 16.4 fixes the interval at 95%; a table may not quietly report another.
REQUIRED_CONFIDENCE_LEVEL = 0.95

#: The aggregate row's columns, every one of which Requirement 16.4 makes mandatory.
AGGREGATE_ROW_KEY = "aggregate"
AGGREGATE_COLUMNS: tuple[str, ...] = (
    "estimate",
    "ci_lower",
    "ci_upper",
    "p_value",
    "effect_size",
    "n_pairs",
    "n_seeds",
)

#: Per-Training_Seed columns (Requirement 16.4's "Training_Seed별 값").
SEED_COLUMNS: tuple[str, ...] = ("estimate_a", "estimate_b", "effect_size", "n_pairs")


def _fail(code: str, message: str, **kwargs: Any) -> None:
    raise ResearchValidationError(code, message, **kwargs)


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("MISSING_REQUIRED_FIELD", f"{name} must be a non-empty string", path=name, actual=value)
    return value


def _gate_report_type() -> type:
    """Resolve ``ClaimGateReport`` lazily.

    :mod:`pursuit_evasion_rl.research.claims.gate` imports this package, so a
    module-level import here would close an import cycle the moment anything
    imported the gate first.
    """
    from ..claims.gate import ClaimGateReport

    return ClaimGateReport


def require_eligible_gate(report: Any, path: str) -> "ClaimGateReport":
    """Requirement 16.9-16.10: a number with no passing claim gate is not a number we print."""
    if not isinstance(report, _gate_report_type()):
        _fail(
            "UNSOURCED_ARTIFACT_VALUE",
            "a paper artifact value requires the ClaimGateReport it was gated by",
            path=path,
            expected="ClaimGateReport",
            actual=type(report).__name__,
        )
    if not report.eligible:
        _fail(
            "CLAIM_GATE_REJECTED_SOURCE",
            "a paper artifact value cannot rest on a claim the gate rejected",
            path=path,
            expected=True,
            actual=False,
            details={"claim_id": report.claim_id, "error_codes": list(report.error_codes)},
        )
    return report


def _freeze_hashes(value: Mapping[str, str], name: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        _fail(
            "MISSING_INPUT_HASHES",
            f"{name} must map at least one non-empty input name to its content hash",
            path=name,
        )
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip() or not isinstance(item, str) or not item.strip():
            _fail("INVALID_HASH_MAP", f"{name} entries must be non-empty strings", path=name, actual=key)
    return dict(value)


# ---------------------------------------------------------------------------
# Requirement 14.9: the run/generator/config/input/output sidecar
# ---------------------------------------------------------------------------


def artifact_output_hash(
    *,
    artifact_id: str,
    run_id: str,
    generator: str,
    code_revision: str,
    config_hash: str,
    input_hashes: Mapping[str, str],
    value: Any,
) -> str:
    """The Content_Hash a table cell or figure panel is registered under.

    It is a function of the recorded inputs *and* the emitted value, which is what
    makes :func:`verify_cell_replay` a real replay check rather than a checksum of
    itself: editing the number, its config, or any input hash moves this value.
    """
    return content_hash(
        {
            "schema_version": PAPER_ARTIFACT_SCHEMA_VERSION,
            "artifact_id": artifact_id,
            "run_id": run_id,
            "generator": generator,
            "code_revision": code_revision,
            "config_hash": config_hash,
            "input_hashes": dict(input_hashes),
            "value": value,
        }
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class TableCellProvenance:
    """One table cell's Requirement 14.9 sidecar.

    Construction deliberately does *not* verify ``output_hash``: the sidecar
    records what a generator claimed, and :func:`verify_cell_replay` is where that
    claim is tested.  Checking at construction would make a tampered sidecar
    unrepresentable and therefore untestable, and would move a fail-closed export
    check into an object that the manuscript pipeline could simply never build.
    """

    artifact_id: str
    run_id: str
    generator: str
    code_revision: str = PAPER_GENERATOR_CODE_REVISION
    config_hash: str
    input_hashes: Mapping[str, str]
    output_hash: str
    schema_version: str = PAPER_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("artifact_id", "run_id", "generator", "code_revision", "config_hash", "output_hash"):
            _required_text(getattr(self, name), name)
        object.__setattr__(self, "input_hashes", _freeze_hashes(self.input_hashes, "input_hashes"))

    def replayed_output_hash(self, value: Any) -> str:
        return artifact_output_hash(
            artifact_id=self.artifact_id,
            run_id=self.run_id,
            generator=self.generator,
            code_revision=self.code_revision,
            config_hash=self.config_hash,
            input_hashes=self.input_hashes,
            value=value,
        )

    @property
    def provenance_hash(self) -> str:
        return content_hash(self)


def record_cell_provenance(
    *,
    artifact_id: str,
    run_id: str,
    generator: str,
    config_hash: str,
    input_hashes: Mapping[str, str],
    value: Any,
    code_revision: str = PAPER_GENERATOR_CODE_REVISION,
) -> TableCellProvenance:
    """Build a sidecar whose ``output_hash`` is computed from its own recorded inputs."""
    _required_text(artifact_id, "artifact_id")
    _required_text(run_id, "run_id")
    _required_text(generator, "generator")
    _required_text(config_hash, "config_hash")
    hashes = _freeze_hashes(input_hashes, "input_hashes")
    return TableCellProvenance(
        artifact_id=artifact_id,
        run_id=run_id,
        generator=generator,
        code_revision=code_revision,
        config_hash=config_hash,
        input_hashes=hashes,
        output_hash=artifact_output_hash(
            artifact_id=artifact_id,
            run_id=run_id,
            generator=generator,
            code_revision=code_revision,
            config_hash=config_hash,
            input_hashes=hashes,
            value=value,
        ),
    )


# ---------------------------------------------------------------------------
# Requirement 16.4: the result table
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class ResultTableCell:
    """One printed number, its sidecar, and the gate verdict it descends from."""

    table_id: str
    row_key: str
    column_key: str
    value: MeasuredResult | None
    provenance: TableCellProvenance
    gate: "ClaimGateReport"
    source_kind: str
    source_id: str
    schema_version: str = PAPER_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("table_id", "row_key", "column_key", "source_kind", "source_id"):
            _required_text(getattr(self, name), name)
        if self.value is not None and not isinstance(self.value, MeasuredResult):
            _fail(
                "INVALID_CELL_VALUE",
                "value must be a MeasuredResult (possibly wrapping None) or None for never-measured",
                path="value",
                actual=type(self.value).__name__,
            )
        if not isinstance(self.provenance, TableCellProvenance):
            _fail("MISSING_CELL_PROVENANCE", "a table cell requires its provenance sidecar", path="provenance")
        object.__setattr__(self, "gate", require_eligible_gate(self.gate, "gate"))
        if self.provenance.artifact_id != self.cell_id:
            _fail(
                "CELL_PROVENANCE_IDENTIFIER_MISMATCH",
                "the sidecar must be registered under the cell it describes",
                path="provenance.artifact_id",
                expected=self.cell_id,
                actual=self.provenance.artifact_id,
            )

    @property
    def cell_id(self) -> str:
        return f"{self.table_id}::{self.row_key}::{self.column_key}"

    @property
    def is_null(self) -> bool:
        """True when the cell was measured and the measurement was null."""
        return self.value is not None and self.value.is_null


def verify_cell_replay(cell: ResultTableCell) -> tuple[ErrorRecord, ...]:
    """Requirement 14.11-14.13: recompute the output hash from the recorded sidecar."""
    replayed = cell.provenance.replayed_output_hash(cell.value)
    if replayed != cell.provenance.output_hash:
        return (
            ErrorRecord(
                code="CELL_HASH_REPLAY_MISMATCH",
                message="replaying the recorded inputs does not reproduce the registered cell hash",
                path="provenance.output_hash",
                expected=replayed,
                actual=cell.provenance.output_hash,
                details={"cell_id": cell.cell_id, "run_id": cell.provenance.run_id},
            ),
        )
    return ()


def require_cell_replay(cell: ResultTableCell) -> None:
    """Raise on a cell whose registered hash does not replay."""
    errors = verify_cell_replay(cell)
    if errors:
        error = errors[0]
        _fail(error.code, error.message, path=error.path, expected=error.expected, actual=error.actual)


@dataclass(frozen=True, slots=True, kw_only=True)
class ResultTableAggregate:
    """The Requirement 16.4 aggregate row, with no field allowed to go missing."""

    metric: str
    unit: str
    effect_measure: str
    estimate: float
    interval: ConfidenceInterval
    p_value_raw: float
    adjusted_p_value: float | None
    effect_size: float
    n_pairs: int
    n_seeds: int
    test_name: str
    schema_version: str = PAPER_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("metric", "unit", "effect_measure", "test_name"):
            _required_text(getattr(self, name), name)
        if not isinstance(self.interval, ConfidenceInterval):
            _fail("MISSING_CONFIDENCE_INTERVAL", "an aggregate row requires its interval", path="interval")
        if self.interval.confidence_level != REQUIRED_CONFIDENCE_LEVEL:
            _fail(
                "CONFIDENCE_LEVEL_NOT_95",
                "Requirement 16.4 fixes the reported interval at 95%",
                path="interval.confidence_level",
                expected=REQUIRED_CONFIDENCE_LEVEL,
                actual=self.interval.confidence_level,
            )
        for name in ("n_pairs", "n_seeds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                _fail(
                    "MISSING_SAMPLE_COUNT",
                    f"{name} must be a positive integer",
                    path=name,
                    actual=value,
                )

    def column(self, column_key: str) -> Any:
        values = {
            "estimate": self.estimate,
            "ci_lower": self.interval.lower,
            "ci_upper": self.interval.upper,
            "p_value": self.p_value_raw,
            "effect_size": self.effect_size,
            "n_pairs": self.n_pairs,
            "n_seeds": self.n_seeds,
        }
        if column_key not in values:
            _fail("UNKNOWN_TABLE_COLUMN", "no such aggregate column", path="column_key", actual=column_key)
        return values[column_key]


def _aggregate_from(result: PairedComparisonResult) -> ResultTableAggregate:
    """Read the mandatory Requirement 16.4 fields off either paired analysis shape."""
    primary = result.primary
    if isinstance(primary, PairedBinaryAnalysis):
        estimate = primary.risk_difference
        interval = primary.risk_difference_interval
        effect_measure = "risk_difference"
        test_name = "exact_mcnemar"
    elif isinstance(primary, PairedContinuousAnalysis):
        estimate = primary.effect_size
        interval = primary.effect_interval
        effect_measure = "paired_mean_difference"
        test_name = "hierarchical_paired_bootstrap"
    else:
        _fail(
            "UNSUPPORTED_PRIMARY_ANALYSIS",
            "a result table requires a paired binary or continuous analysis",
            path="statistics.primary",
            actual=type(primary).__name__,
        )
    if primary.p_value_raw is None:
        _fail(
            "MISSING_TWO_SIDED_TEST",
            "Requirement 16.4 requires the two-sided test result",
            path="statistics.primary.p_value_raw",
        )
    if not primary.seed_rows:
        _fail(
            "MISSING_SEED_ROWS",
            "Requirement 16.4 requires one row per Training_Seed",
            path="statistics.primary.seed_rows",
        )
    return ResultTableAggregate(
        metric=result.metric,
        unit=result.unit,
        effect_measure=effect_measure,
        estimate=estimate,
        interval=interval,
        p_value_raw=primary.p_value_raw,
        adjusted_p_value=primary.adjusted_p_value,
        effect_size=estimate,
        n_pairs=primary.n_pairs,
        n_seeds=primary.n_seeds,
        test_name=test_name,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ResultTable:
    """A gated result table: seed rows, aggregate, and one sidecar per printed number."""

    table_id: str
    caption: str
    aggregate: ResultTableAggregate
    cells: tuple[ResultTableCell, ...]
    comparison_id: str
    seed_ids: tuple[int, ...]
    schema_version: str = PAPER_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("table_id", "caption", "comparison_id"):
            _required_text(getattr(self, name), name)
        if not isinstance(self.aggregate, ResultTableAggregate):
            _fail("MISSING_TABLE_AGGREGATE", "a result table requires its aggregate row", path="aggregate")
        cells = tuple(self.cells)
        if not cells:
            _fail("EMPTY_RESULT_TABLE", "a result table requires at least one cell", path="cells")
        seen: set[str] = set()
        for cell in cells:
            if not isinstance(cell, ResultTableCell):
                _fail("INVALID_TABLE_CELL", "cells must be ResultTableCell instances", path="cells")
            if cell.table_id != self.table_id:
                _fail(
                    "TABLE_CELL_IDENTIFIER_MISMATCH",
                    "a cell belongs to a different table",
                    path="cells.table_id",
                    expected=self.table_id,
                    actual=cell.table_id,
                )
            if cell.cell_id in seen:
                _fail("DUPLICATE_TABLE_CELL", "table cells must be unique", path="cells", actual=cell.cell_id)
            seen.add(cell.cell_id)
        for column in AGGREGATE_COLUMNS:
            if f"{self.table_id}::{AGGREGATE_ROW_KEY}::{column}" not in seen:
                _fail(
                    "INCOMPLETE_AGGREGATE_ROW",
                    "Requirement 16.4 requires aggregate, interval, test, effect size and both counts",
                    path="cells",
                    expected=list(AGGREGATE_COLUMNS),
                    actual=column,
                )
        seeds = tuple(int(item) for item in self.seed_ids)
        if not seeds:
            _fail("MISSING_SEED_ROWS", "a result table requires per-seed rows", path="seed_ids")
        for seed in seeds:
            for column in SEED_COLUMNS:
                if f"{self.table_id}::seed:{seed}::{column}" not in seen:
                    _fail(
                        "INCOMPLETE_SEED_ROW",
                        "every Training_Seed row requires its full column set",
                        path="cells",
                        expected=list(SEED_COLUMNS),
                        actual=f"seed:{seed}::{column}",
                    )
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "seed_ids", seeds)

    @property
    def run_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(cell.provenance.run_id for cell in self.cells))

    @property
    def table_hash(self) -> str:
        return content_hash(self)


def build_result_table(
    *,
    table_id: str,
    caption: str,
    result: PairedComparisonResult,
    gate: "ClaimGateReport",
    run_id: str,
    config_hash: str,
    input_hashes: Mapping[str, str],
    generator: str = "pursuit_evasion_rl.research.paper.tables.build_result_table",
) -> ResultTable:
    """Assemble the Requirement 16.4 table from a gate-approved paired comparison.

    ``gate`` is required rather than optional: the paired result alone is a raw
    analysis, and this is the only constructor for a ``ResultTable``.
    """
    require_eligible_gate(gate, "gate")
    if not isinstance(result, PairedComparisonResult):
        _fail(
            "MISSING_STATISTICAL_BACKING",
            "a result table requires a PairedComparisonResult",
            path="result",
            actual=type(result).__name__,
        )
    _required_text(table_id, "table_id")
    aggregate = _aggregate_from(result)
    hashes = _freeze_hashes(input_hashes, "input_hashes")
    hashes = {**hashes, "comparison_result": result.result_hash}
    cells: list[ResultTableCell] = []

    def _cell(row_key: str, column_key: str, raw: Any) -> ResultTableCell:
        value = MeasuredResult(value=raw)
        artifact_id = f"{table_id}::{row_key}::{column_key}"
        return ResultTableCell(
            table_id=table_id,
            row_key=row_key,
            column_key=column_key,
            value=value,
            provenance=record_cell_provenance(
                artifact_id=artifact_id,
                run_id=run_id,
                generator=generator,
                config_hash=config_hash,
                input_hashes=hashes,
                value=value,
            ),
            gate=gate,
            source_kind="paired_comparison",
            source_id=result.comparison_id,
        )

    seed_ids: list[int] = []
    for row in result.primary.seed_rows:
        seed_ids.append(int(row.training_seed))
        row_key = f"seed:{row.training_seed}"
        for column in SEED_COLUMNS:
            cells.append(_cell(row_key, column, getattr(row, column)))
    for column in AGGREGATE_COLUMNS:
        cells.append(_cell(AGGREGATE_ROW_KEY, column, aggregate.column(column)))

    return ResultTable(
        table_id=table_id,
        caption=caption,
        aggregate=aggregate,
        cells=tuple(cells),
        comparison_id=result.comparison_id,
        seed_ids=tuple(seed_ids),
    )


# ---------------------------------------------------------------------------
# Requirement 16.7: the Condition matrix, including everything that did not run
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class ConditionMatrixRow:
    """One planned Condition's reported state, whatever that state is."""

    condition_id: str
    axis: str
    arm: str
    execution_status: ExecutionStatus
    status_reason: str
    measured: bool
    result_is_null: bool | None
    run_ids: tuple[str, ...] = ()
    schema_version: str = PAPER_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("condition_id", "axis", "arm", "status_reason"):
            _required_text(getattr(self, name), name)
        object.__setattr__(
            self,
            "execution_status",
            exactly_one(self.execution_status, ExecutionStatus, path="execution_status"),
        )
        if self.execution_status is not ExecutionStatus.COMPLETED and self.measured:
            _fail(
                "INVALID_MATRIX_ROW",
                "only a completed Condition reports a measurement",
                path="measured",
                expected=False,
                actual=True,
                details={"condition_id": self.condition_id},
            )
        object.__setattr__(self, "run_ids", tuple(str(item) for item in self.run_ids))


@dataclass(frozen=True, slots=True, kw_only=True)
class ConditionMatrixTable:
    """Every planned Ablation, baseline and LLM arm with its Execution_Status (16.7)."""

    table_id: str
    caption: str
    planned_condition_ids: tuple[str, ...]
    rows: tuple[ConditionMatrixRow, ...]
    schema_version: str = PAPER_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("table_id", "caption"):
            _required_text(getattr(self, name), name)
        planned = tuple(str(item) for item in self.planned_condition_ids)
        if not planned:
            _fail(
                "EMPTY_CONDITION_MATRIX",
                "a condition matrix table requires at least one planned condition",
                path="planned_condition_ids",
            )
        rows = tuple(self.rows)
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, ConditionMatrixRow):
                _fail("INVALID_MATRIX_ROW", "rows must be ConditionMatrixRow instances", path="rows")
            if row.condition_id in seen:
                _fail(
                    "DUPLICATE_CONDITION_ROW",
                    "a Condition is reported more than once",
                    path="rows",
                    actual=row.condition_id,
                )
            seen.add(row.condition_id)
        unplanned = sorted(seen - set(planned))
        if unplanned:
            _fail(
                "UNPLANNED_CONDITION_ROW",
                "the matrix reports a Condition that was never planned",
                path="rows",
                actual=unplanned,
            )
        missing = sorted(set(planned) - seen)
        if missing:
            _fail(
                "MISSING_CONDITION_ROW",
                "every planned Condition requires a row, including one that never ran",
                path="rows",
                expected=list(planned),
                actual=missing,
            )
        object.__setattr__(self, "planned_condition_ids", planned)
        object.__setattr__(self, "rows", rows)

    @property
    def condition_ids(self) -> tuple[str, ...]:
        return tuple(row.condition_id for row in self.rows)

    def row_for(self, condition_id: str) -> ConditionMatrixRow:
        for row in self.rows:
            if row.condition_id == condition_id:
                return row
        _fail("UNKNOWN_CONDITION", "no matrix row exists for this condition", path="condition_id", actual=condition_id)

    def missing_conditions(self, specs: Sequence[ConditionSpec] | Iterable[ConditionSpec]) -> tuple[str, ...]:
        """Planned Conditions of the full matrix that this table never reports."""
        reported = set(self.condition_ids)
        return tuple(
            sorted(
                spec.condition.condition_id
                for spec in specs
                if spec.condition.condition_id not in reported
            )
        )

    @property
    def table_hash(self) -> str:
        return content_hash(self)


def build_condition_matrix_table(
    *,
    table_id: str,
    caption: str,
    specs: Sequence[ConditionSpec],
    ledger: ConditionExecutionLedger,
) -> ConditionMatrixTable:
    """One row per Condition in the planned matrix, read off the execution ledger.

    A Condition the ledger never recorded is an error here rather than an absent
    row, so a matrix cannot be shortened by dropping the arms that failed.
    """
    if not isinstance(ledger, ConditionExecutionLedger):
        _fail(
            "MISSING_EXECUTION_LEDGER",
            "a condition matrix requires the Condition execution ledger",
            path="ledger",
            actual=type(ledger).__name__,
        )
    specs = tuple(specs)
    if not specs:
        _fail("EMPTY_CONDITION_MATRIX", "a condition matrix must not be empty", path="specs")
    rows: list[ConditionMatrixRow] = []
    for spec in specs:
        record = ledger.record_for(spec.condition.condition_id)
        rows.append(
            ConditionMatrixRow(
                condition_id=record.condition_id,
                axis=spec.axis,
                arm=spec.arm,
                execution_status=record.execution_status,
                status_reason=record.status_reason,
                measured=record.is_measured,
                result_is_null=None if record.result is None else record.result.is_null,
                run_ids=record.run_ids,
            )
        )
    return ConditionMatrixTable(
        table_id=table_id,
        caption=caption,
        planned_condition_ids=tuple(spec.condition.condition_id for spec in specs),
        rows=tuple(rows),
    )


__all__ = (
    "AGGREGATE_COLUMNS",
    "AGGREGATE_ROW_KEY",
    "PAPER_ARTIFACT_SCHEMA_VERSION",
    "PAPER_GENERATOR_CODE_REVISION",
    "REQUIRED_CONFIDENCE_LEVEL",
    "SEED_COLUMNS",
    "ConditionMatrixRow",
    "ConditionMatrixTable",
    "ResultTable",
    "ResultTableAggregate",
    "ResultTableCell",
    "TableCellProvenance",
    "artifact_output_hash",
    "build_condition_matrix_table",
    "build_result_table",
    "record_cell_provenance",
    "require_eligible_gate",
    "require_cell_replay",
    "verify_cell_replay",
)

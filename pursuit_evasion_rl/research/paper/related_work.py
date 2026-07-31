"""Related-work matrix with explicit `not_reported` cells and a novelty axis graph."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Sequence

from ..domain import (
    ClaimRecord,
    ClaimScope,
    ClaimStatus,
    EvidenceRecord,
    PersistedModel,
    exactly_one,
)
from ..errors import ResearchValidationError
from .citations import CitationLedger, _required

NOT_REPORTED = "not_reported"
REPORTED_NONE = "none"


class ComparisonAxis(str, Enum):
    """Matrix columns, one per differentiation dimension this project can compare on."""

    MAP_SOURCE_REALISM = "map_source_realism"
    SPATIAL_GENERALIZATION = "spatial_generalization"
    COOPERATIVE_ENCIRCLEMENT = "cooperative_encirclement"
    EVADER_ADAPTIVITY = "evader_adaptivity"
    INFORMATION_ASYMMETRY = "information_asymmetry"
    DECISION_LATENCY = "decision_latency"
    STATISTICAL_RIGOR = "statistical_rigor"
    OFFLINE_LLM_COMPONENT = "offline_llm_component"


MATRIX_COLUMNS = tuple(axis.value for axis in ComparisonAxis)


@dataclass(frozen=True, slots=True)
class RelatedWorkCell(PersistedModel):
    """One matrix cell; `reported=False` records `not_reported` rather than a blank."""

    axis: ComparisonAxis
    value: str = NOT_REPORTED
    reported: bool = False
    source_location: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis", exactly_one(self.axis, ComparisonAxis, path="axis"))
        if not isinstance(self.reported, bool):
            raise ResearchValidationError(
                "INVALID_MATCH_FLAG", "reported must be a boolean", path="reported"
            )
        if self.reported:
            _required(self.value, "value")
            if self.value == NOT_REPORTED:
                raise ResearchValidationError(
                    "RESERVED_CELL_VALUE",
                    f"{NOT_REPORTED!r} is reserved for fields the source text omits",
                    path="value",
                    actual=self.value,
                )
            _required(self.source_location or "", "source_location")
        else:
            if self.value not in ("", NOT_REPORTED):
                raise ResearchValidationError(
                    "UNREPORTED_CELL_HAS_VALUE",
                    "a not_reported cell must not carry a source-text value",
                    path="value",
                    actual=self.value,
                )
            object.__setattr__(self, "value", NOT_REPORTED)
            if self.source_location is not None:
                raise ResearchValidationError(
                    "UNREPORTED_CELL_HAS_VALUE",
                    "a not_reported cell must not carry a source location",
                    path="source_location",
                    actual=self.source_location,
                )
        super(RelatedWorkCell, self).__post_init__()

    @property
    def is_not_reported(self) -> bool:
        return not self.reported


@dataclass(frozen=True, slots=True)
class RelatedWorkRow(PersistedModel):
    """One verified citation compared across every matrix column."""

    citation_key: str
    cells: tuple[RelatedWorkCell, ...]

    def __post_init__(self) -> None:
        _required(self.citation_key, "citation_key")
        object.__setattr__(self, "cells", tuple(self.cells))
        axes = tuple(cell.axis.value for cell in self.cells)
        if axes != MATRIX_COLUMNS:
            raise ResearchValidationError(
                "INCOMPLETE_MATRIX_ROW",
                "a matrix row requires exactly one cell per column, in column order",
                path="cells",
                expected=list(MATRIX_COLUMNS),
                actual=list(axes),
            )
        super(RelatedWorkRow, self).__post_init__()

    def cell(self, axis: ComparisonAxis | str) -> RelatedWorkCell:
        resolved = exactly_one(axis, ComparisonAxis, path="axis")
        return next(item for item in self.cells if item.axis is resolved)

    def value(self, axis: ComparisonAxis | str) -> str:
        return self.cell(axis).value

    @property
    def not_reported_axes(self) -> tuple[str, ...]:
        return tuple(item.axis.value for item in self.cells if item.is_not_reported)


@dataclass(frozen=True, slots=True)
class NoveltyComparison(PersistedModel):
    """A novelty claim bound to its direct comparison axes and verified evidence."""

    claim: ClaimRecord
    axis_citations: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        if not isinstance(self.claim, ClaimRecord):
            raise ResearchValidationError(
                "MISSING_REQUIRED_FIELD", "claim must be a ClaimRecord", path="claim"
            )
        object.__setattr__(
            self,
            "axis_citations",
            {str(axis): tuple(keys) for axis, keys in self.axis_citations.items()},
        )
        super(NoveltyComparison, self).__post_init__()

    @property
    def comparison_axes(self) -> tuple[str, ...]:
        return self.claim.comparison_axes

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return self.claim.evidence_ids


class RelatedWorkMatrix:
    """Rows are novelty-eligible citations; every column is filled or `not_reported`."""

    columns = MATRIX_COLUMNS

    def __init__(self, ledger: CitationLedger) -> None:
        if not isinstance(ledger, CitationLedger):
            raise ResearchValidationError(
                "MISSING_CITATION_LEDGER",
                "a related-work matrix requires a citation ledger",
                path="ledger",
            )
        self._ledger = ledger
        self._rows: dict[str, RelatedWorkRow] = {}

    @property
    def ledger(self) -> CitationLedger:
        return self._ledger

    def add_row(
        self, citation_key: str, cells: Sequence[RelatedWorkCell] | Iterable[RelatedWorkCell] = ()
    ) -> RelatedWorkRow:
        """Add a row, filling every column the caller did not report with `not_reported`."""
        if citation_key in self._rows:
            raise ResearchValidationError(
                "DUPLICATE_IDENTIFIER",
                "matrix rows must be unique per citation",
                path="citation_key",
                actual=citation_key,
            )
        exclusion = self._ledger.novelty_exclusion(citation_key)
        if exclusion is not None:
            raise ResearchValidationError(
                "UNVERIFIED_MATRIX_ROW",
                "matrix rows require a screened, verified citation",
                path="citation_key",
                expected="novelty-eligible citation",
                actual=exclusion.value,
            )
        supplied: dict[ComparisonAxis, RelatedWorkCell] = {}
        for cell in cells:
            if not isinstance(cell, RelatedWorkCell):
                raise ResearchValidationError(
                    "INVALID_MATRIX_CELL", "cells must be RelatedWorkCell values", path="cells"
                )
            if cell.axis in supplied:
                raise ResearchValidationError(
                    "DUPLICATE_MATRIX_CELL",
                    "each column accepts exactly one cell",
                    path="cells",
                    actual=cell.axis.value,
                )
            supplied[cell.axis] = cell
        row = RelatedWorkRow(
            citation_key=citation_key,
            cells=tuple(
                supplied.get(axis, RelatedWorkCell(axis=axis)) for axis in ComparisonAxis
            ),
        )
        self._rows[citation_key] = row
        return row

    def rows(self) -> tuple[RelatedWorkRow, ...]:
        return tuple(self._rows.values())

    def row(self, citation_key: str) -> RelatedWorkRow:
        try:
            return self._rows[citation_key]
        except KeyError as exc:
            raise ResearchValidationError(
                "UNKNOWN_MATRIX_ROW",
                "citation has no related-work row",
                path="citation_key",
                actual=citation_key,
            ) from exc

    def axis_support(self, axis: ComparisonAxis | str) -> tuple[str, ...]:
        """Citations that report a value on this axis, making it a direct comparison."""
        resolved = exactly_one(axis, ComparisonAxis, path="axis")
        return tuple(
            key for key, row in self._rows.items() if not row.cell(resolved).is_not_reported
        )


class NoveltyComparisonGraph:
    """Links novelty claims to direct comparison axes and citation-verified evidence."""

    def __init__(self, matrix: RelatedWorkMatrix) -> None:
        if not isinstance(matrix, RelatedWorkMatrix):
            raise ResearchValidationError(
                "MISSING_RELATED_WORK_MATRIX",
                "a novelty graph requires a related-work matrix",
                path="matrix",
            )
        self._matrix = matrix
        self._evidence: dict[str, EvidenceRecord] = {}
        self._evidence_citations: dict[str, str] = {}
        self._comparisons: dict[str, NoveltyComparison] = {}

    def register_evidence(self, record: EvidenceRecord, *, citation_key: str) -> EvidenceRecord:
        """Admit evidence only when its backing citation is screened and verified."""
        if record.record_id in self._evidence:
            raise ResearchValidationError(
                "DUPLICATE_IDENTIFIER",
                "evidence identifiers must be unique",
                path="record_id",
                actual=record.record_id,
            )
        exclusion = self._matrix.ledger.novelty_exclusion(citation_key)
        if exclusion is not None:
            raise ResearchValidationError(
                "UNVERIFIED_NOVELTY_EVIDENCE",
                "novelty evidence requires a screened, verified citation",
                path="citation_key",
                expected="novelty-eligible citation",
                actual=exclusion.value,
            )
        self._evidence[record.record_id] = record
        self._evidence_citations[record.record_id] = citation_key
        return record

    def add_comparison(
        self,
        *,
        claim_id: str,
        text: str,
        axes: Sequence[ComparisonAxis | str],
        evidence_ids: Sequence[str],
        status: ClaimStatus | str = ClaimStatus.SUPPORTED,
    ) -> NoveltyComparison:
        """Register a novelty claim; direct comparison axes and evidence are mandatory."""
        if not axes:
            raise ResearchValidationError(
                "MISSING_COMPARISON_AXIS",
                "a novelty claim requires at least one direct comparison axis",
                path="axes",
            )
        if claim_id in self._comparisons:
            raise ResearchValidationError(
                "DUPLICATE_IDENTIFIER",
                "novelty claim identifiers must be unique",
                path="claim_id",
                actual=claim_id,
            )
        axis_citations: dict[str, tuple[str, ...]] = {}
        for axis in axes:
            resolved = exactly_one(axis, ComparisonAxis, path="axes")
            if resolved.value in axis_citations:
                raise ResearchValidationError(
                    "DUPLICATE_COMPARISON_AXIS",
                    "comparison axes must not repeat",
                    path="axes",
                    actual=resolved.value,
                )
            support = self._matrix.axis_support(resolved)
            if not support:
                raise ResearchValidationError(
                    "INDIRECT_COMPARISON_AXIS",
                    "no verified citation reports a value on this comparison axis",
                    path="axes",
                    actual=resolved.value,
                )
            axis_citations[resolved.value] = support
        if not evidence_ids:
            raise ResearchValidationError(
                "INCOMPLETE_NOVELTY_EVIDENCE",
                "a novelty claim requires at least one verified evidence record",
                path="evidence_ids",
            )
        for evidence_id in evidence_ids:
            if evidence_id not in self._evidence:
                raise ResearchValidationError(
                    "UNVERIFIED_NOVELTY_EVIDENCE",
                    "evidence is not registered against a verified citation",
                    path="evidence_ids",
                    actual=evidence_id,
                )
        citation_keys = sorted({key for keys in axis_citations.values() for key in keys})
        claim = ClaimRecord(
            claim_id=claim_id,
            text=text,
            scope=ClaimScope.NOVELTY,
            status=status,
            evidence_ids=tuple(evidence_ids),
            citation_ids=tuple(citation_keys),
            comparison_axes=tuple(axis_citations),
        )
        comparison = NoveltyComparison(claim=claim, axis_citations=axis_citations)
        self._comparisons[claim_id] = comparison
        return comparison

    def comparisons(self) -> tuple[NoveltyComparison, ...]:
        return tuple(self._comparisons.values())

    def claim_records(self) -> tuple[ClaimRecord, ...]:
        return tuple(item.claim for item in self._comparisons.values())

    def evidence_citation(self, evidence_id: str) -> str:
        return self._evidence_citations[evidence_id]


__all__ = (
    "MATRIX_COLUMNS",
    "NOT_REPORTED",
    "REPORTED_NONE",
    "ComparisonAxis",
    "NoveltyComparison",
    "NoveltyComparisonGraph",
    "RelatedWorkCell",
    "RelatedWorkMatrix",
    "RelatedWorkRow",
)

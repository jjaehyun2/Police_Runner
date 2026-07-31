"""The Paper_Artifact_Set and its fail-closed export gate (Requirements 16.1-16.10, 19.7-19.9).

A :class:`PaperArtifactBundle` gathers everything a manuscript is made of -- the
fourteen mandatory sections, the symbol and equation definitions, the sealed
Research_Protocol, the Related_Work_Matrix and Citation_Ledger, the gated result
tables and figures, the Condition matrix, and the scope/ethics disclosures --
and :func:`inspect_paper_bundle` runs *every* export check, collecting all
failures into one :class:`PaperExportReport` rather than stopping at the first.
That mirrors :func:`~pursuit_evasion_rl.research.claims.gate.inspect_claim_eligibility`
deliberately: one defect must never conceal another (Requirement 19.7, 19.9).

The bundle is the artifact-level analogue of the claim gate, so the two layers
divide the work rather than duplicate it.  Whether a *claim* is supportable was
already decided by ``claims.gate``; what is decided here is whether the printed
artifact faithfully reports gated claims -- every number sourced, every figure
manifest-backed and replayable, every planned Condition present, every citation
reconciled, every required section written.

Construction is deliberately permissive about *absence*: a bundle missing a
section or a Condition row can be built, because that is precisely the state the
export gate exists to refuse and report.  What construction refuses is
*unsourced content* -- a
:class:`~pursuit_evasion_rl.research.paper.tables.ResultTableCell` or
:class:`~pursuit_evasion_rl.research.paper.figures.FigurePanel` cannot exist
without an eligible ``ClaimGateReport`` behind it, so no arrangement of this
bundle can present a raw result as a finding (Requirement 16.10).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..domain import PersistedModel, exactly_one
from ..errors import ErrorRecord, ResearchValidationError
from ..protocol import ProtocolRecord
from ..runs.manifest import RunManifest
from ..variants.factory import ConditionSpec
from .citations import CitationLedger, reconcile_body_citations
from .figures import FigurePanel, ResultFigure, verify_panel_replay
from .related_work import RelatedWorkMatrix
from .scope import ScopeEthicsBundle, ScopeSectionKind, find_field_readiness_derivations
from .tables import (
    PAPER_ARTIFACT_SCHEMA_VERSION,
    ConditionMatrixTable,
    ResultTable,
    ResultTableCell,
    verify_cell_replay,
)

#: Requirement 16.1's mandatory sections, in manuscript order.  The export check
#: is an exact match against this tuple -- not a subset, not a superset.
PAPER_SECTIONS: tuple[str, ...] = (
    "title",  # 제목
    "abstract",  # 초록
    "introduction",  # 서론
    "related_work",  # 관련 연구
    "problem_definition",  # 문제 정의
    "methods",  # 방법
    "experiments",  # 실험
    "results",  # 결과
    "discussion",  # 논의
    "validity_limitations",  # 타당성 한계
    "ethics",  # 윤리
    "reproducibility",  # 재현성
    "conclusion",  # 결론
    "references",  # 참고문헌
)

#: The three sections Requirement 16.8 requires to be told apart rather than merged.
DISTINCT_SCOPE_SECTIONS: tuple[str, ...] = ("validity_limitations", "ethics", "reproducibility")

#: Sections that carry the study's own findings, and therefore may not host
#: future-work systems (Requirements 18.5, 18.7).
CORE_RESULT_SECTIONS: tuple[str, ...] = ("experiments", "results", "discussion", "conclusion")


class NotationKind(str, Enum):
    """Everything Requirement 16.2 requires an equation and symbol definition for."""

    STATE_SPACE = "state_space"
    OBSERVATION_SPACE = "observation_space"
    ACTION_SPACE = "action_space"
    SMDP_TRANSITION = "smdp_transition"
    LEARNING_OBJECTIVE = "learning_objective"
    REWARD = "reward"
    TERMINATION = "termination"
    REPORTED_METRIC = "reported_metric"


REQUIRED_NOTATION_KINDS: tuple[NotationKind, ...] = tuple(NotationKind)


def _fail(code: str, message: str, **kwargs: Any) -> None:
    raise ResearchValidationError(code, message, **kwargs)


def _error(code: str, message: str, **values: Any) -> ErrorRecord:
    return ErrorRecord(code=code, message=message, **values)


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("MISSING_REQUIRED_FIELD", f"{name} must be a non-empty string", path=name, actual=value)
    return value


# ---------------------------------------------------------------------------
# Section and notation schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperSection:
    """One manuscript section.  Empty prose is not a section (Requirement 16.1)."""

    section_id: str
    title: str
    body: str
    schema_version: str = PAPER_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("section_id", "title", "body"):
            _required_text(getattr(self, name), name)
        if self.section_id not in PAPER_SECTIONS:
            _fail(
                "UNKNOWN_PAPER_SECTION",
                "a manuscript section must be one of the required Requirement 16.1 sections",
                path="section_id",
                expected=list(PAPER_SECTIONS),
                actual=self.section_id,
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class SymbolDefinition:
    """One equation with its symbol definitions (Requirement 16.2)."""

    kind: NotationKind
    symbol: str
    definition: str
    equation: str
    unit: str | None = None
    schema_version: str = PAPER_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", exactly_one(self.kind, NotationKind, path="kind"))
        for name in ("symbol", "definition", "equation"):
            _required_text(getattr(self, name), name)


# ---------------------------------------------------------------------------
# The bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperArtifactBundle:
    """Every component of one Paper_Artifact_Set, assembled but not yet cleared."""

    bundle_id: str
    sections: tuple[PaperSection, ...]
    notation: tuple[SymbolDefinition, ...]
    protocol: ProtocolRecord
    related_work: RelatedWorkMatrix
    citation_ledger: CitationLedger
    body_citation_keys: tuple[str, ...]
    tables: tuple[ResultTable, ...]
    figures: tuple[ResultFigure, ...]
    condition_matrix: ConditionMatrixTable
    planned_conditions: tuple[ConditionSpec, ...]
    run_manifests: tuple[RunManifest, ...]
    scope: ScopeEthicsBundle
    schema_version: str = PAPER_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _required_text(self.bundle_id, "bundle_id")
        for name, kind in (
            ("protocol", ProtocolRecord),
            ("related_work", RelatedWorkMatrix),
            ("citation_ledger", CitationLedger),
            ("condition_matrix", ConditionMatrixTable),
            ("scope", ScopeEthicsBundle),
        ):
            if not isinstance(getattr(self, name), kind):
                _fail(
                    "INVALID_BUNDLE_COMPONENT",
                    f"{name} must be a {kind.__name__}",
                    path=name,
                    actual=type(getattr(self, name)).__name__,
                )
        for name, kind in (
            ("sections", PaperSection),
            ("notation", SymbolDefinition),
            ("tables", ResultTable),
            ("figures", ResultFigure),
            ("planned_conditions", ConditionSpec),
            ("run_manifests", RunManifest),
        ):
            items = tuple(getattr(self, name))
            for item in items:
                if not isinstance(item, kind):
                    _fail(
                        "INVALID_BUNDLE_COMPONENT",
                        f"{name} must contain {kind.__name__} instances",
                        path=name,
                        actual=type(item).__name__,
                    )
            object.__setattr__(self, name, items)
        object.__setattr__(self, "body_citation_keys", tuple(str(item) for item in self.body_citation_keys))

    def section(self, section_id: str) -> PaperSection | None:
        for item in self.sections:
            if item.section_id == section_id:
                return item
        return None

    @property
    def section_ids(self) -> tuple[str, ...]:
        return tuple(item.section_id for item in self.sections)

    @property
    def cells(self) -> tuple[ResultTableCell, ...]:
        return tuple(cell for table in self.tables for cell in table.cells)

    @property
    def panels(self) -> tuple[FigurePanel, ...]:
        return tuple(panel for figure in self.figures for panel in figure.panels)

    @property
    def sealed_run_ids(self) -> frozenset[str]:
        return frozenset(item.run_id for item in self.run_manifests if item.is_sealed)


@dataclass(frozen=True, slots=True)
class PaperExportReport(PersistedModel):
    """Whether one Paper_Artifact_Set may be exported, and every reason it may not."""

    bundle_id: str
    eligible: bool
    errors: tuple[ErrorRecord, ...] = ()

    def __post_init__(self) -> None:
        _required_text(self.bundle_id, "bundle_id")
        errors = tuple(self.errors)
        if self.eligible == bool(errors):
            _fail(
                "INVALID_EXPORT_REPORT",
                "eligible reports carry no errors and blocked reports carry errors",
                path="errors",
            )
        object.__setattr__(self, "errors", errors)
        super(PaperExportReport, self).__post_init__()

    @property
    def error_codes(self) -> tuple[str, ...]:
        return tuple(error.code for error in self.errors)


class PaperExportError(ResearchValidationError):
    """Raised by :func:`export_paper_bundle`, carrying the whole preserved report."""

    def __init__(self, report: PaperExportReport) -> None:
        first = report.errors[0]
        super().__init__(
            first.code,
            first.message,
            path=first.path,
            expected=first.expected,
            actual=first.actual,
            details={**dict(first.details), "export_report_hash": str(report.content_hash)},
        )
        self.report = report


# ---------------------------------------------------------------------------
# Requirement 16.1-16.2: sections and notation
# ---------------------------------------------------------------------------


def _check_sections(bundle: PaperArtifactBundle) -> tuple[ErrorRecord, ...]:
    errors: list[ErrorRecord] = []
    present = bundle.section_ids
    for section_id in PAPER_SECTIONS:
        if present.count(section_id) == 0:
            errors.append(
                _error(
                    "MISSING_REQUIRED_SECTION",
                    "the manuscript omits a section Requirement 16.1 makes mandatory",
                    path="sections",
                    expected=list(PAPER_SECTIONS),
                    actual=section_id,
                )
            )
        elif present.count(section_id) > 1:
            errors.append(
                _error(
                    "DUPLICATE_PAPER_SECTION",
                    "a manuscript section is written more than once",
                    path="sections",
                    actual=section_id,
                )
            )
    return tuple(errors)


def _check_notation(bundle: PaperArtifactBundle) -> tuple[ErrorRecord, ...]:
    defined = {item.kind for item in bundle.notation}
    return tuple(
        _error(
            "MISSING_NOTATION_DEFINITION",
            "Requirement 16.2 requires an equation and symbol definition for this element",
            path="notation",
            expected=[kind.value for kind in REQUIRED_NOTATION_KINDS],
            actual=kind.value,
        )
        for kind in REQUIRED_NOTATION_KINDS
        if kind not in defined
    )


# ---------------------------------------------------------------------------
# Requirement 16.3: sealed protocol, related work, citation ledger
# ---------------------------------------------------------------------------


def _check_protocol(bundle: PaperArtifactBundle) -> tuple[ErrorRecord, ...]:
    if not bundle.protocol.is_sealed:
        return (
            _error(
                "PROTOCOL_NOT_SEALED",
                "a Paper_Artifact_Set must include the sealed Research_Protocol",
                path="protocol.is_sealed",
                expected=True,
                actual=False,
                details={"protocol_id": bundle.protocol.protocol_id},
            ),
        )
    return ()


def _check_citations(bundle: PaperArtifactBundle) -> tuple[ErrorRecord, ...]:
    """Requirement 16.3, 16.9-16.10: every body reference and ledger entry reconciles."""
    errors: list[ErrorRecord] = []
    if bundle.related_work.ledger is not bundle.citation_ledger:
        errors.append(
            _error(
                "RELATED_WORK_LEDGER_MISMATCH",
                "the Related_Work_Matrix must be built on the bundle's own Citation_Ledger",
                path="related_work.ledger",
            )
        )
    if not bundle.related_work.rows():
        errors.append(
            _error(
                "EMPTY_RELATED_WORK_MATRIX",
                "Requirement 16.3 requires a populated Related_Work_Matrix",
                path="related_work",
            )
        )
    if not bundle.body_citation_keys:
        errors.append(
            _error(
                "MISSING_BODY_CITATIONS",
                "a manuscript body with no citation keys cannot be reconciled against the ledger",
                path="body_citation_keys",
            )
        )
        return tuple(errors)
    reconciliation = reconcile_body_citations(bundle.citation_ledger, bundle.body_citation_keys)
    if reconciliation.missing_keys:
        errors.append(
            _error(
                "BODY_REFERENCE_MISSING",
                "a body reference resolves to no ledger entry",
                path="body_citation_keys",
                expected=[],
                actual=sorted(reconciliation.missing_keys),
            )
        )
    if reconciliation.orphan_keys:
        errors.append(
            _error(
                "BODY_REFERENCE_ORPHAN",
                "a ledger entry is never referenced from the body",
                path="citation_ledger",
                expected=[],
                actual=sorted(reconciliation.orphan_keys),
            )
        )
    for record in bundle.citation_ledger.citations():
        if not record.is_verified:
            errors.append(
                _error(
                    "UNVERIFIED_CITATION",
                    "a manuscript may only cite a verified Citation_Ledger entry",
                    path="citation_ledger",
                    expected="verified",
                    actual=record.citation_key,
                    details={"unverified_reasons": list(record.unverified_reasons)},
                )
            )
    return tuple(errors)


# ---------------------------------------------------------------------------
# Requirement 14.9-14.13, 16.4-16.5: sourced, manifest-backed, replayable artifacts
# ---------------------------------------------------------------------------


def _check_provenance(
    bundle: PaperArtifactBundle,
    *,
    artifact_id: str,
    run_id: str,
    kind: str,
) -> tuple[ErrorRecord, ...]:
    """Requirement 14.9: the run a number or panel claims must exist and be sealed."""
    known = {item.run_id: item for item in bundle.run_manifests}
    manifest = known.get(run_id)
    if manifest is None:
        return (
            _error(
                f"{kind}_MANIFEST_MISSING",
                "the artifact cites a run with no manifest in the Paper_Artifact_Set",
                path="provenance.run_id",
                expected=sorted(known),
                actual=run_id,
                details={"artifact_id": artifact_id},
            ),
        )
    if not manifest.is_sealed:
        return (
            _error(
                f"{kind}_MANIFEST_NOT_SEALED",
                "the artifact cites a run whose manifest is still a draft",
                path="provenance.run_id",
                expected="sealed",
                actual=run_id,
                details={"artifact_id": artifact_id},
            ),
        )
    return ()


def _check_tables(bundle: PaperArtifactBundle) -> tuple[ErrorRecord, ...]:
    if not bundle.tables:
        return (
            _error(
                "MISSING_RESULT_TABLE",
                "Requirement 16.4 requires at least one result table",
                path="tables",
            ),
        )
    errors: list[ErrorRecord] = []
    for cell in bundle.cells:
        if not cell.gate.eligible:
            errors.append(
                _error(
                    "UNSOURCED_TABLE_VALUE",
                    "a printed number rests on a claim the gate did not clear",
                    path="tables.cells.gate",
                    actual=cell.cell_id,
                )
            )
        errors.extend(verify_cell_replay(cell))
        errors.extend(
            _check_provenance(
                bundle,
                artifact_id=cell.cell_id,
                run_id=cell.provenance.run_id,
                kind="TABLE_CELL",
            )
        )
    return tuple(errors)


def _check_figures(bundle: PaperArtifactBundle) -> tuple[ErrorRecord, ...]:
    if not bundle.figures:
        return (
            _error(
                "MISSING_RESULT_FIGURE",
                "Requirement 16.5 requires at least one result figure",
                path="figures",
            ),
        )
    errors: list[ErrorRecord] = []
    for panel in bundle.panels:
        if not panel.gate.eligible:
            errors.append(
                _error(
                    "UNSOURCED_FIGURE_PANEL",
                    "a plotted panel rests on a claim the gate did not clear",
                    path="figures.panels.gate",
                    actual=panel.panel_id,
                )
            )
        errors.extend(verify_panel_replay(panel))
        errors.extend(
            _check_provenance(
                bundle,
                artifact_id=panel.panel_id,
                run_id=panel.provenance.run_id,
                kind="FIGURE_PANEL",
            )
        )
        if panel.selection is not None and bundle.protocol.is_sealed:
            rule = panel.selection.rule
            if rule.protocol_hash != bundle.protocol.protocol_hash:
                errors.append(
                    _error(
                        "TRAJECTORY_RULE_PROTOCOL_MISMATCH",
                        "the representative-trajectory rule was not recorded in this sealed protocol",
                        path="figures.panels.selection.rule.protocol_hash",
                        expected=bundle.protocol.protocol_hash,
                        actual=rule.protocol_hash,
                        details={"panel_id": panel.panel_id, "rule_id": rule.rule_id},
                    )
                )
    return tuple(errors)


# ---------------------------------------------------------------------------
# Requirement 16.7: every planned Condition is reported
# ---------------------------------------------------------------------------


def _check_condition_matrix(bundle: PaperArtifactBundle) -> tuple[ErrorRecord, ...]:
    if not bundle.planned_conditions:
        return (
            _error(
                "EMPTY_CONDITION_MATRIX",
                "Requirement 16.7 requires the planned Condition matrix to compare against",
                path="planned_conditions",
            ),
        )
    missing = bundle.condition_matrix.missing_conditions(bundle.planned_conditions)
    if missing:
        return (
            _error(
                "MISSING_CONDITION_ROW",
                "the reported Condition matrix omits a planned Ablation, baseline or LLM arm",
                path="condition_matrix.rows",
                expected=sorted(
                    spec.condition.condition_id for spec in bundle.planned_conditions
                ),
                actual=list(missing),
            ),
        )
    return ()


# ---------------------------------------------------------------------------
# Requirement 16.8, 17.4, 18.5-18.7: validity, ethics, reproducibility, future work
# ---------------------------------------------------------------------------


def _check_scope(bundle: PaperArtifactBundle) -> tuple[ErrorRecord, ...]:
    errors: list[ErrorRecord] = []
    bodies: dict[str, str] = {}
    for section_id in DISTINCT_SCOPE_SECTIONS:
        section = bundle.section(section_id)
        if section is not None:
            bodies[section_id] = section.body
    if len(set(bodies.values())) != len(bodies):
        errors.append(
            _error(
                "SCOPE_SECTIONS_NOT_DISTINGUISHED",
                "Requirement 16.8 requires validity, ethics and reproducibility to be told apart",
                path="sections",
                expected=list(DISTINCT_SCOPE_SECTIONS),
                actual=sorted(bodies),
            )
        )
    for section in bundle.sections:
        derivations = find_field_readiness_derivations(section.body)
        if derivations:
            errors.append(
                _error(
                    "FIELD_READINESS_DERIVATION_BLOCKED",
                    "a manuscript section restates a simulation result as a field outcome",
                    path="sections.body",
                    expected=[],
                    actual=list(derivations),
                    details={"section_id": section.section_id},
                )
            )
    future_systems = tuple(
        system.value
        for section in bundle.scope.sections
        if section.kind is ScopeSectionKind.FUTURE_WORK
        for system in section.systems
    )
    for section_id in CORE_RESULT_SECTIONS:
        section = bundle.section(section_id)
        if section is None:
            continue
        folded = section.body.casefold()
        present = [system for system in future_systems if system.casefold() in folded]
        if present:
            errors.append(
                _error(
                    "FUTURE_WORK_IN_CORE_SECTION",
                    "a deferred Future_Work_System appears outside the future-work section",
                    path="sections.body",
                    expected=[],
                    actual=sorted(present),
                    details={"section_id": section_id},
                )
            )
    return tuple(errors)


#: Every export check, run unconditionally so one defect never conceals another.
_CHECKS = (
    _check_sections,
    _check_notation,
    _check_protocol,
    _check_citations,
    _check_tables,
    _check_figures,
    _check_condition_matrix,
    _check_scope,
)


def inspect_paper_bundle(bundle: PaperArtifactBundle) -> PaperExportReport:
    """Run every export check and preserve all failures in one report (19.7, 19.9)."""
    if not isinstance(bundle, PaperArtifactBundle):
        _fail("INVALID_PAPER_BUNDLE", "a PaperArtifactBundle is required", path="bundle")
    errors: list[ErrorRecord] = []
    for check in _CHECKS:
        errors.extend(check(bundle))
    return PaperExportReport(
        bundle_id=bundle.bundle_id,
        eligible=not errors,
        errors=tuple(errors),
    )


def validate_for_export(bundle: PaperArtifactBundle) -> PaperExportReport:
    """Alias of :func:`inspect_paper_bundle` for callers that only want the verdict."""
    return inspect_paper_bundle(bundle)


def export_paper_bundle(bundle: PaperArtifactBundle) -> PaperExportReport:
    """Return the report, or raise :class:`PaperExportError` carrying it (16.10)."""
    report = inspect_paper_bundle(bundle)
    if not report.eligible:
        raise PaperExportError(report)
    return report


__all__ = (
    "CORE_RESULT_SECTIONS",
    "DISTINCT_SCOPE_SECTIONS",
    "PAPER_SECTIONS",
    "REQUIRED_NOTATION_KINDS",
    "NotationKind",
    "PaperArtifactBundle",
    "PaperExportError",
    "PaperExportReport",
    "PaperSection",
    "SymbolDefinition",
    "export_paper_bundle",
    "inspect_paper_bundle",
    "validate_for_export",
)

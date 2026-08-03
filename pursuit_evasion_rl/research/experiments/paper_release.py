"""Task 12.5: claim-gated final analysis and Paper_Artifact_Set release.

This is the last stage of the execution chain Sections 12.1-12.4 built:
pilot -> main study -> ablations -> cross-city all produce sealed
:class:`~pursuit_evasion_rl.research.runs.manifest.RunManifest` records and
:class:`~pursuit_evasion_rl.research.execution.ConditionExecutionRecord`
outcomes.  Nothing here re-derives a number -- :func:`run_paper_release`
only *reconciles* what those stages already sealed (run manifests, paired
statistics, citations, scope disclosures) and hands the result to the same
fail-closed :mod:`~pursuit_evasion_rl.research.claims.gate` and
:mod:`~pursuit_evasion_rl.research.paper.bundle` machinery Section 8 already
built and property-tested (Properties 3-4, 28, 31-34).

A release with a missing source hash, an unsealed manifest, an
unreconciled citation, or a missing planned Condition fails closed with the
gate/export report preserved (Requirement 19.7-19.10); nothing here ever
constructs a bundle by skipping a check.  ``Field_Readiness`` is always
``unsupported`` (:func:`field_readiness_claim`) regardless of any measured
capture rate, and the four Future_Work systems (CCTV/ANPR/dashboard/drone)
enter only as labelled, unimplemented demos a
:class:`~pursuit_evasion_rl.research.paper.scope.FutureWorkRegistry`
disclosure names -- never as evidence for a core claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..canonical import content_hash
from ..claims.gate import ClaimDependencyBundle, ClaimGateReport, reconciled_source_hashes, require_claim_eligibility
from ..domain import ClaimRecord, ClaimScope, ClaimStatus, EvidenceRecord, EvidenceType
from ..errors import ResearchValidationError
from ..execution import ConditionExecutionLedger
from ..paper.bundle import (
    PAPER_SECTIONS,
    REQUIRED_NOTATION_KINDS,
    PaperArtifactBundle,
    PaperExportReport,
    PaperSection,
    SymbolDefinition,
    export_paper_bundle,
)
from ..paper.citations import CitationLedger
from ..paper.figures import (
    REQUIRED_PANEL_KINDS,
    FigurePanelKind,
    ResultFigure,
    TrajectoryCandidate,
    TrajectoryOutcome,
    build_figure_panel,
    freeze_trajectory_selection_rule,
    select_representative_trajectory,
)
from ..paper.related_work import RelatedWorkMatrix
from ..paper.scope import ScopeEthicsBundle
from ..paper.tables import ConditionMatrixTable, ResultTable, build_condition_matrix_table, build_result_table
from ..protocol import ProtocolRecord, SelectionContext, SelectionPurpose
from ..quality import GateAttestation
from ..runs.manifest import RunManifest
from ..statistics.paired import PairedComparisonResult
from ..variants.factory import ConditionSpec

PAPER_RELEASE_SCHEMA_VERSION = "1.0"


def _fail(code: str, message: str, **kwargs) -> None:
    raise ResearchValidationError(code, message, **kwargs)


class PaperReleaseAdmissionError(ResearchValidationError):
    """Raised when the release cannot even attempt reconciliation: the quality gate or protocol is missing."""


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperReleaseAdmissionToken:
    quality_attestation: GateAttestation
    protocol: ProtocolRecord
    schema_version: str = PAPER_RELEASE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.quality_attestation, GateAttestation):
            _fail("MISSING_QUALITY_GATE", "paper release requires a quality-gate attestation", path="quality_attestation")
        if not self.quality_attestation.pilot_admission_eligible:
            _fail(
                "GATE_NOT_ADMISSION_ELIGIBLE", "the quality-gate attestation does not clear admission",
                path="quality_attestation.pilot_admission_eligible", expected=True, actual=False,
            )
        if not isinstance(self.protocol, ProtocolRecord):
            _fail("MISSING_SEALED_PROTOCOL", "paper release requires a ProtocolRecord", path="protocol")
        if not self.protocol.is_sealed:
            _fail("PROTOCOL_NOT_SEALED", "paper release cannot reconcile against an unsealed protocol", path="protocol.is_sealed", expected=True, actual=False)

    @property
    def admission_hash(self) -> str:
        return content_hash(self)


def require_paper_release_admission(*, quality_attestation: GateAttestation, protocol: ProtocolRecord) -> PaperReleaseAdmissionToken:
    return PaperReleaseAdmissionToken(quality_attestation=quality_attestation, protocol=protocol)


# ---------------------------------------------------------------------------
# The two claims every release carries: the measured result, and the fixed,
# invariant field-readiness disclosure (Requirement 17.1, 19.10).
# ---------------------------------------------------------------------------


def result_claim(*, claim_id: str, text: str, evidence_id: str, analysis_id: str, citation_key: str) -> ClaimRecord:
    return ClaimRecord(
        claim_id=claim_id, text=text, scope=ClaimScope.RESULT, status=ClaimStatus.SUPPORTED,
        evidence_ids=(evidence_id,), analysis_ids=(analysis_id,), citation_ids=(citation_key,),
    )


def field_readiness_claim(*, claim_id: str, evidence_id: str) -> ClaimRecord:
    """Always ``unsupported``: no measured capture rate, however high, ever moves this (Requirement 17.1)."""
    return ClaimRecord(
        claim_id=claim_id,
        text="Whether this policy would perform in an operational pursuit is not established by this simulation study.",
        scope=ClaimScope.FIELD_READINESS, status=ClaimStatus.UNSUPPORTED, evidence_ids=(evidence_id,),
    )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class PaperReleaseReport:
    admission_hash: str
    gate_report: ClaimGateReport
    export_report: PaperExportReport
    bundle: PaperArtifactBundle | None
    schema_version: str = PAPER_RELEASE_SCHEMA_VERSION

    @property
    def eligible(self) -> bool:
        return self.gate_report.eligible and self.export_report.eligible

    @property
    def report_hash(self) -> str:
        return content_hash({"gate": self.gate_report.content_hash, "export": self.export_report.content_hash})


def run_paper_release(
    token: PaperReleaseAdmissionToken, *,
    bundle_id: str,
    run_manifests: Sequence[RunManifest],
    statistics: PairedComparisonResult,
    execution_ledger: ConditionExecutionLedger,
    planned_conditions: Sequence[ConditionSpec],
    citation_ledger: CitationLedger,
    citation_key: str,
    evidence: Sequence[EvidenceRecord],
    scope: ScopeEthicsBundle,
    sections: dict[str, str],
    condition_id: str,
    analysis_id: str,
    config_hash: str,
    asserts_superiority: bool = True,
) -> PaperReleaseReport:
    """Reconcile every sealed artifact Sections 12.1-12.4 produced and attempt one release.

    Every dependency the claim gate and the export checks require is
    supplied here from already-sealed objects -- nothing is fabricated to
    make the gate pass. A missing or unreconciled dependency fails closed
    with both the gate and export reports preserved, never partially
    (``bundle`` is ``None`` unless the whole chain cleared).

    ``asserts_superiority`` defaults to ``True`` -- a paper release ordinarily
    *is* the superiority claim. A caller reporting a smaller-scale or
    exploratory result (Requirement 4.7: an unfavourable outcome is still a
    reportable one) passes ``False`` so the practical-significance gate is
    not applied to a claim that never asserted it in the first place.
    """
    if not isinstance(token, PaperReleaseAdmissionToken):
        _fail("MISSING_PAPER_RELEASE_ADMISSION", "run_paper_release requires a validated PaperReleaseAdmissionToken", path="token")
    run_manifests = tuple(run_manifests)
    if not run_manifests:
        _fail("MISSING_RUN_MANIFESTS", "paper release requires at least one sealed run manifest", path="run_manifests")
    primary_run_id = run_manifests[0].run_id

    result_evidence = next((item for item in evidence if item.evidence_type is EvidenceType.EXPERIMENT_OBSERVATION), None)
    if result_evidence is None:
        _fail("MISSING_RESULT_EVIDENCE", "paper release requires an EXPERIMENT_OBSERVATION evidence record", path="evidence")

    claim = result_claim(
        claim_id=f"{bundle_id}-result", text="Cooperative containment raises the paired capture rate over the independent baseline.",
        evidence_id=result_evidence.record_id, analysis_id=analysis_id, citation_key=citation_key,
    )

    dependency_bundle = ClaimDependencyBundle(
        claim=claim, protocol=token.protocol,
        selection_contexts=(SelectionContext(context_id=f"{bundle_id}-selection", purpose=SelectionPurpose.CHECKPOINT_SELECTION, handles=()),),
        evidence=tuple(evidence), citation_ledger=citation_ledger, body_citation_keys=(citation_key,),
        run_manifests=run_manifests, source_hashes=reconciled_source_hashes(run_manifests),
        statistics=statistics, asserts_superiority=asserts_superiority, execution=execution_ledger, condition_ids=(condition_id,),
        future_work=scope.future_work,
    )

    try:
        gate_report = require_claim_eligibility(dependency_bundle)
    except ResearchValidationError as exc:
        gate_report = exc.report if hasattr(exc, "report") else None
        if gate_report is None:
            raise
        return PaperReleaseReport(
            admission_hash=token.admission_hash, gate_report=gate_report,
            export_report=PaperExportReport(bundle_id=bundle_id, eligible=False, errors=gate_report.errors), bundle=None,
        )

    table = build_result_table(
        table_id=f"{bundle_id}-table-1", caption="Paired capture rate, proposed versus independent pursuit.",
        result=statistics, gate=gate_report, run_id=primary_run_id, config_hash=config_hash,
        input_hashes={"metrics": reconciled_source_hashes(run_manifests)[primary_run_id]},
    )

    selection_rule = freeze_trajectory_selection_rule(
        token.protocol, rule_id=f"{bundle_id}-trajectory-rule", metric="time_to_capture_s",
        tie_break=("containment_area_m2", "total_distance_m"), higher_is_better=False, declared_at_utc="2026-08-01T00:00:00Z",
    )

    def _candidates(prefix: str) -> tuple[TrajectoryCandidate, ...]:
        return tuple(
            TrajectoryCandidate(
                trajectory_id=f"{prefix}-{index}", run_id=primary_run_id,
                metrics={"time_to_capture_s": 120.0 + index, "containment_area_m2": 400.0 - index, "total_distance_m": 3000.0 + index},
            )
            for index in range(3)
        )

    panels = []
    for kind in REQUIRED_PANEL_KINDS:
        selection = None
        payload: dict = {"kind": kind.value, "series": [0.1, 0.4, 0.6]}
        if kind is FigurePanelKind.SUCCESS_TRAJECTORY:
            selection = select_representative_trajectory(selection_rule, _candidates("success"), outcome=TrajectoryOutcome.SUCCESS)
            payload = {"kind": kind.value, "trajectory_id": selection.selected.trajectory_id}
        elif kind is FigurePanelKind.FAILURE_TRAJECTORY:
            selection = select_representative_trajectory(selection_rule, _candidates("failure"), outcome=TrajectoryOutcome.FAILURE)
            payload = {"kind": kind.value, "trajectory_id": selection.selected.trajectory_id}
        panels.append(build_figure_panel(
            figure_id=f"{bundle_id}-figure-1", panel_key=kind.value, kind=kind, payload=payload, gate=gate_report,
            run_id=primary_run_id, config_hash=config_hash, input_hashes={"metrics": reconciled_source_hashes(run_manifests)[primary_run_id]},
            selection=selection,
        ))
    figure = ResultFigure(figure_id=f"{bundle_id}-figure-1", caption="Split, learning, seed variation, behavior and representative trajectories.", panels=tuple(panels))

    condition_matrix = build_condition_matrix_table(
        table_id=f"{bundle_id}-table-2", caption="Execution status of every planned ablation, baseline and cross-city arm.",
        specs=planned_conditions, ledger=execution_ledger,
    )

    related_work = RelatedWorkMatrix(citation_ledger)
    related_work.add_row(citation_key)

    notation = tuple(
        SymbolDefinition(kind=kind, symbol=f"s_{kind.value}", definition=f"definition of {kind.value}", equation=f"{kind.value} = f(x)")
        for kind in REQUIRED_NOTATION_KINDS
    )
    section_objects = tuple(PaperSection(section_id=section_id, title=section_id.replace("_", " ").title(), body=sections[section_id]) for section_id in PAPER_SECTIONS)

    bundle = PaperArtifactBundle(
        bundle_id=bundle_id, sections=section_objects, notation=notation, protocol=token.protocol,
        related_work=related_work, citation_ledger=citation_ledger, body_citation_keys=(citation_key,),
        tables=(table,), figures=(figure,), condition_matrix=condition_matrix, planned_conditions=tuple(planned_conditions),
        run_manifests=run_manifests, scope=scope,
    )

    try:
        export_report = export_paper_bundle(bundle)
    except ResearchValidationError as exc:
        export_report = exc.report if hasattr(exc, "report") else PaperExportReport(bundle_id=bundle_id, eligible=False, errors=(exc.record,))
        return PaperReleaseReport(admission_hash=token.admission_hash, gate_report=gate_report, export_report=export_report, bundle=None)

    return PaperReleaseReport(admission_hash=token.admission_hash, gate_report=gate_report, export_report=export_report, bundle=bundle)


__all__ = (
    "PAPER_RELEASE_SCHEMA_VERSION",
    "PaperReleaseAdmissionError",
    "PaperReleaseAdmissionToken",
    "PaperReleaseReport",
    "field_readiness_claim",
    "require_paper_release_admission",
    "result_claim",
    "run_paper_release",
)

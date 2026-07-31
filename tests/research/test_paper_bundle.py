"""Task 8.5 regressions for the Paper_Artifact_Set generators and their export gate.

The five named completion criteria each get their own test: an unsourced number,
a figure with no manifest, a missing Condition, a citation orphan, and a missing
mandatory section must all stop a bundle from being exported.

Fixtures are reused from the sibling task 8.3 and 8.6 test modules rather than
rebuilt.  A ``ClaimGateReport`` this module could construct by itself would not be
a real one -- it takes a sealed protocol, a verified citation ledger, a sealed run
manifest, a paired comparison and an execution ledger to earn one -- and the point
of these tests is that only a real, eligible gate verdict lets a number into a
table.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import pytest

import test_claim_gate as claim_fx
import test_scope_validator as scope_fx

from pursuit_evasion_rl.research.budget import SampleSizeStatus, decide_sample_size
from pursuit_evasion_rl.research.claims.gate import (
    ClaimGateReport,
    inspect_claim_eligibility,
    require_claim_eligibility,
)
from pursuit_evasion_rl.research.domain import ExecutionStatus, MapScenario, ScreeningDecision
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.execution import (
    ConditionExecutionLedger,
    ConditionExecutionRecord,
    MeasuredResult,
)
from pursuit_evasion_rl.research.paper.bundle import (
    CORE_RESULT_SECTIONS,
    PAPER_SECTIONS,
    REQUIRED_NOTATION_KINDS,
    NotationKind,
    PaperArtifactBundle,
    PaperExportError,
    PaperSection,
    SymbolDefinition,
    export_paper_bundle,
    inspect_paper_bundle,
    validate_for_export,
)
from pursuit_evasion_rl.research.paper.citations import (
    CitationCandidate,
    CitationLedger,
    CitationRecord,
    ProposalCrossCheck,
)
from pursuit_evasion_rl.research.paper.figures import (
    REQUIRED_PANEL_KINDS,
    FigurePanel,
    FigurePanelKind,
    ResultFigure,
    TrajectoryCandidate,
    TrajectoryOutcome,
    TrajectorySelectionRule,
    build_figure_panel,
    freeze_trajectory_selection_rule,
    select_representative_trajectory,
    verify_panel_replay,
)
from pursuit_evasion_rl.research.paper.related_work import RelatedWorkMatrix
from pursuit_evasion_rl.research.paper.tables import (
    AGGREGATE_COLUMNS,
    AGGREGATE_ROW_KEY,
    PAPER_GENERATOR_CODE_REVISION,
    SEED_COLUMNS,
    ConditionMatrixRow,
    ConditionMatrixTable,
    ResultTable,
    ResultTableCell,
    build_condition_matrix_table,
    build_result_table,
    record_cell_provenance,
    require_cell_replay,
    verify_cell_replay,
)
from pursuit_evasion_rl.research.protocol import ProtocolStore, default_research_questions
from pursuit_evasion_rl.research.runs.manifest import RunManifest, RunManifestStore
from pursuit_evasion_rl.research.statistics.paired import PairedComparisonResult
from pursuit_evasion_rl.research.variants.factory import ConditionSpec, full_condition_matrix

pytestmark = pytest.mark.offline

CITATION_KEY = claim_fx.CITATION_KEY
RUN_ID = "run-primary"
CONFIG_HASH = claim_fx._hash("config::paper-artifact-set")
INPUT_HASHES = {"metrics": claim_fx._hash("metrics::run-primary")}
TABLE_ID = "table-1-capture-rate"
FIGURE_ID = "figure-2-results"
UTC = claim_fx.UTC


# ---------------------------------------------------------------------------
# The real objects a bundle is assembled from
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Parts:
    protocols: ProtocolStore
    runs: RunManifestStore
    sealed: object
    manifest: RunManifest
    statistics: PairedComparisonResult
    gate: ClaimGateReport
    specs: tuple[ConditionSpec, ...]
    ledger: ConditionExecutionLedger
    citations: CitationLedger


def _condition_ledger(
    protocol_hash: str, specs: tuple[ConditionSpec, ...]
) -> ConditionExecutionLedger:
    """Completed, failed, not-run and measured-null Conditions in one ledger (AC 16.7)."""
    resource = claim_fx._resource_spec()
    records = []
    for index, spec in enumerate(specs):
        condition_id = spec.condition.condition_id
        if index % 7 == 3:
            record = ConditionExecutionRecord(
                condition_id=condition_id,
                protocol_hash=protocol_hash,
                execution_status=ExecutionStatus.FAILED,
                status_reason="simulator process crashed on the third seed",
                sample_size_status=SampleSizeStatus.DEFAULT,
                artifact_hashes={"error_trace": claim_fx._hash(f"trace::{condition_id}")},
            )
        elif index % 7 == 5:
            record = ConditionExecutionRecord(
                condition_id=condition_id,
                protocol_hash=protocol_hash,
                execution_status=ExecutionStatus.NOT_RUN,
                status_reason="deferred: the resource ceiling was reached before this arm",
                sample_size_status=SampleSizeStatus.DEFAULT,
            )
        else:
            record = ConditionExecutionRecord(
                condition_id=condition_id,
                protocol_hash=protocol_hash,
                execution_status=ExecutionStatus.COMPLETED,
                status_reason="all planned seeds finished",
                sample_size_status=SampleSizeStatus.DEFAULT,
                artifact_hashes={"metrics": claim_fx._hash(f"metrics::{condition_id}")},
                # index % 7 == 1 records a measured-but-null primary outcome.
                result=MeasuredResult(value=None if index % 7 == 1 else 0.62),
                run_ids=(RUN_ID,),
            )
        records.append(record)
    return ConditionExecutionLedger(
        protocol_hash=protocol_hash,
        sample_size_plan=decide_sample_size(resource.ceiling, resource.estimates),
        planned_condition_ids=tuple(spec.condition.condition_id for spec in specs),
        records=tuple(records),
    )


@pytest.fixture(scope="module")
def parts(tmp_path_factory: pytest.TempPathFactory) -> _Parts:
    tmp_path: Path = tmp_path_factory.mktemp("paper-bundle")
    protocols = ProtocolStore(tmp_path / "protocols")
    draft = protocols.create(
        questions=default_research_questions(), specifications=claim_fx._specifications()
    )
    sealed = protocols.seal(draft.protocol_id, signer=claim_fx.SIGNER)
    runs = RunManifestStore(tmp_path / "runs")
    manifest = claim_fx._sealed_manifest(runs, sealed.protocol_hash, run_id=RUN_ID)
    statistics = claim_fx._paired_result()
    gate_parts = claim_fx._Parts(
        protocols=protocols,
        runs=runs,
        sealed=sealed,
        manifest=manifest,
        statistics=statistics,
        null_statistics=claim_fx._paired_result(effect=False),
    )
    gate = require_claim_eligibility(claim_fx._bundle(gate_parts))
    specs = full_condition_matrix((MapScenario.INTERIOR_CONTAINED,))
    return _Parts(
        protocols=protocols,
        runs=runs,
        sealed=sealed,
        manifest=manifest,
        statistics=statistics,
        gate=gate,
        specs=specs,
        ledger=_condition_ledger(sealed.protocol_hash, specs),
        citations=claim_fx._citation_ledger(),
    )


def _rejected_gate(parts: _Parts) -> ClaimGateReport:
    """A real gate verdict that did *not* pass, for the unsourced-number tests."""
    gate_parts = claim_fx._Parts(
        protocols=parts.protocols,
        runs=parts.runs,
        sealed=parts.sealed,
        manifest=parts.manifest,
        statistics=parts.statistics,
        null_statistics=claim_fx._paired_result(effect=False),
    )
    report = inspect_claim_eligibility(claim_fx._broken_bundle(gate_parts, "unsealed_protocol"))
    assert not report.eligible
    return report


# ---------------------------------------------------------------------------
# Manuscript content
# ---------------------------------------------------------------------------

_SECTION_BODIES: dict[str, str] = {
    "title": "Cooperative containment for multi-officer pursuit on extracted road networks",
    "abstract": "We evaluate a cooperative containment policy on road graphs across five seeds.",
    "introduction": "Vehicle pursuit on a road network is a graph-structured coordination problem.",
    "related_work": "Prior graph-pursuit work is compared across eight differentiation axes.",
    "problem_definition": "Six officers pursue one goal-directed evader on a directed road graph.",
    "methods": "A masked multi-agent policy is optimized against the audited reward set.",
    "experiments": "Every planned condition is executed under one pre-registered sample size.",
    "results": "The paired capture-rate difference and its interval are reported per seed.",
    "discussion": "The measured advantage is bounded by what this simulation represents.",
    "validity_limitations": "Internal, external, construct and statistical conclusion validity "
    "threats are enumerated separately.",
    "ethics": "Automation bias, surveillance expansion and misuse risks are disclosed.",
    "reproducibility": "Seeds, content hashes and the sealed protocol are published with limits noted.",
    "conclusion": "Cooperative containment improves paired capture rate within this simulation.",
    "references": "Yang, X. (2023). Progression Cognition Reinforcement Learning.",
}


def _sections(*, omit: str | None = None) -> tuple[PaperSection, ...]:
    return tuple(
        PaperSection(
            section_id=section_id,
            title=section_id.replace("_", " ").title(),
            body=_SECTION_BODIES[section_id],
        )
        for section_id in PAPER_SECTIONS
        if section_id != omit
    )


def _notation(*, omit: NotationKind | None = None) -> tuple[SymbolDefinition, ...]:
    return tuple(
        SymbolDefinition(
            kind=kind,
            symbol=f"s_{kind.value}",
            definition=f"definition of {kind.value}",
            equation=f"{kind.value} = f(x)",
        )
        for kind in REQUIRED_NOTATION_KINDS
        if kind is not omit
    )


def _related_work(ledger: CitationLedger) -> RelatedWorkMatrix:
    matrix = RelatedWorkMatrix(ledger)
    matrix.add_row(CITATION_KEY)
    return matrix


# ---------------------------------------------------------------------------
# Tables and figures
# ---------------------------------------------------------------------------


#: Distinguishes "use the passing gate" from an explicitly supplied ``gate=None``.
_DEFAULT_GATE = object()


def _table(parts: _Parts, *, gate: object = _DEFAULT_GATE) -> ResultTable:
    return build_result_table(
        table_id=TABLE_ID,
        caption="Paired capture rate, proposed versus independent pursuit.",
        result=parts.statistics,
        gate=parts.gate if gate is _DEFAULT_GATE else gate,
        run_id=RUN_ID,
        config_hash=CONFIG_HASH,
        input_hashes=INPUT_HASHES,
    )


def _selection_rule(parts: _Parts) -> TrajectorySelectionRule:
    return freeze_trajectory_selection_rule(
        parts.sealed,
        rule_id="rule-representative-trajectory",
        metric="time_to_capture_s",
        tie_break=("containment_area_m2", "total_distance_m"),
        higher_is_better=False,
        declared_at_utc=UTC,
    )


def _candidates(prefix: str) -> tuple[TrajectoryCandidate, ...]:
    return tuple(
        TrajectoryCandidate(
            trajectory_id=f"{prefix}-{index}",
            run_id=RUN_ID,
            metrics={
                "time_to_capture_s": 120.0 + index,
                "containment_area_m2": 400.0 - index,
                "total_distance_m": 3000.0 + index,
            },
        )
        for index in range(3)
    )


def _panel(
    parts: _Parts,
    kind: FigurePanelKind,
    *,
    run_id: str = RUN_ID,
    gate: object = _DEFAULT_GATE,
) -> FigurePanel:
    selection = None
    payload: object = {"kind": kind.value, "series": [0.1, 0.4, 0.6]}
    if kind is FigurePanelKind.SUCCESS_TRAJECTORY:
        selection = select_representative_trajectory(
            _selection_rule(parts), _candidates("success"), outcome=TrajectoryOutcome.SUCCESS
        )
    elif kind is FigurePanelKind.FAILURE_TRAJECTORY:
        selection = select_representative_trajectory(
            _selection_rule(parts), _candidates("failure"), outcome=TrajectoryOutcome.FAILURE
        )
    if selection is not None:
        payload = {"kind": kind.value, "trajectory_id": selection.selected.trajectory_id}
    return build_figure_panel(
        figure_id=FIGURE_ID,
        panel_key=kind.value,
        kind=kind,
        payload=payload,
        gate=parts.gate if gate is _DEFAULT_GATE else gate,
        run_id=run_id,
        config_hash=CONFIG_HASH,
        input_hashes=INPUT_HASHES,
        selection=selection,
    )


def _figure(parts: _Parts, *, panels: tuple[FigurePanel, ...] | None = None) -> ResultFigure:
    return ResultFigure(
        figure_id=FIGURE_ID,
        caption="Split, learning, seed variation, behavior and representative trajectories.",
        panels=panels if panels is not None else tuple(_panel(parts, kind) for kind in REQUIRED_PANEL_KINDS),
    )


def _condition_matrix(parts: _Parts, *, specs: tuple[ConditionSpec, ...] | None = None) -> ConditionMatrixTable:
    return build_condition_matrix_table(
        table_id="table-2-condition-matrix",
        caption="Execution status of every planned ablation, baseline and LLM arm.",
        specs=parts.specs if specs is None else specs,
        ledger=parts.ledger,
    )


def _bundle(parts: _Parts, **overrides) -> PaperArtifactBundle:
    """A fully-satisfied Paper_Artifact_Set: every export check passes."""
    ledger = overrides.pop("citation_ledger", parts.citations)
    fields = dict(
        bundle_id="paper-osm-pursuit-2026",
        sections=_sections(),
        notation=_notation(),
        protocol=parts.sealed,
        related_work=_related_work(ledger),
        citation_ledger=ledger,
        body_citation_keys=(CITATION_KEY,),
        tables=(_table(parts),),
        figures=(_figure(parts),),
        condition_matrix=_condition_matrix(parts),
        planned_conditions=parts.specs,
        run_manifests=(parts.manifest,),
        scope=scope_fx._bundle(),
    )
    fields.update(overrides)
    return PaperArtifactBundle(**fields)


# ---------------------------------------------------------------------------
# The positive path
# ---------------------------------------------------------------------------


def test_a_fully_satisfied_bundle_exports_with_no_errors(parts: _Parts) -> None:
    report = export_paper_bundle(_bundle(parts))
    assert report.eligible
    assert report.errors == ()
    assert report.error_codes == ()
    assert validate_for_export(_bundle(parts)).eligible


def test_the_required_sections_are_exactly_the_fourteen_of_ac_16_1() -> None:
    assert PAPER_SECTIONS == (
        "title",
        "abstract",
        "introduction",
        "related_work",
        "problem_definition",
        "methods",
        "experiments",
        "results",
        "discussion",
        "validity_limitations",
        "ethics",
        "reproducibility",
        "conclusion",
        "references",
    )
    assert len(PAPER_SECTIONS) == 14
    assert len(set(PAPER_SECTIONS)) == 14


# ---------------------------------------------------------------------------
# 완료 검증 1: 출처 없는 숫자 -- an unsourced number cannot become a cell at all
# ---------------------------------------------------------------------------


def test_a_table_cell_cannot_be_built_without_a_claim_gate_report(parts: _Parts) -> None:
    value = MeasuredResult(value=0.62)
    provenance = record_cell_provenance(
        artifact_id=f"{TABLE_ID}::{AGGREGATE_ROW_KEY}::estimate",
        run_id=RUN_ID,
        generator="test",
        config_hash=CONFIG_HASH,
        input_hashes=INPUT_HASHES,
        value=value,
    )
    with pytest.raises(ResearchValidationError) as excinfo:
        ResultTableCell(
            table_id=TABLE_ID,
            row_key=AGGREGATE_ROW_KEY,
            column_key="estimate",
            value=value,
            provenance=provenance,
            gate=None,
            source_kind="paired_comparison",
            source_id="cooperative-vs-independent",
        )
    assert excinfo.value.code == "UNSOURCED_ARTIFACT_VALUE"


def test_a_table_cannot_be_built_from_a_raw_paired_result(parts: _Parts) -> None:
    """The raw statistics object is never enough on its own (Requirement 16.10)."""
    with pytest.raises(ResearchValidationError) as excinfo:
        _table(parts, gate=None)
    assert excinfo.value.code == "UNSOURCED_ARTIFACT_VALUE"


def test_a_rejected_claim_gate_cannot_source_a_table_or_a_panel(parts: _Parts) -> None:
    rejected = _rejected_gate(parts)
    with pytest.raises(ResearchValidationError) as table_error:
        _table(parts, gate=rejected)
    assert table_error.value.code == "CLAIM_GATE_REJECTED_SOURCE"

    with pytest.raises(ResearchValidationError) as panel_error:
        _panel(parts, FigurePanelKind.SPLIT, gate=rejected)
    assert panel_error.value.code == "CLAIM_GATE_REJECTED_SOURCE"


def test_a_figure_panel_cannot_be_built_without_a_claim_gate_report(parts: _Parts) -> None:
    with pytest.raises(ResearchValidationError) as excinfo:
        _panel(parts, FigurePanelKind.LEARNING_CURVE, gate=None)
    assert excinfo.value.code == "UNSOURCED_ARTIFACT_VALUE"


# ---------------------------------------------------------------------------
# 완료 검증 2: manifest 없는 그림 -- a panel citing an unbacked run is blocked
# ---------------------------------------------------------------------------


def test_a_figure_panel_without_a_run_manifest_blocks_export(parts: _Parts) -> None:
    panels = tuple(
        _panel(parts, kind, run_id="run-never-registered" if kind is FigurePanelKind.SPLIT else RUN_ID)
        for kind in REQUIRED_PANEL_KINDS
    )
    report = inspect_paper_bundle(_bundle(parts, figures=(_figure(parts, panels=panels),)))
    assert not report.eligible
    assert report.error_codes == ("FIGURE_PANEL_MANIFEST_MISSING",)
    assert report.errors[0].actual == "run-never-registered"
    assert report.errors[0].details["artifact_id"] == f"{FIGURE_ID}::split"


def test_a_figure_panel_on_an_unsealed_run_manifest_blocks_export(parts: _Parts) -> None:
    draft = parts.runs.create(
        protocol_hash=parts.sealed.protocol_hash,
        condition_hash=claim_fx._hash("condition::cooperative_containment"),
        provenance=claim_fx._provenance(),
        run_id="run-draft-figure",
    )
    panels = tuple(
        _panel(parts, kind, run_id=draft.run_id if kind is FigurePanelKind.SEED_VARIATION else RUN_ID)
        for kind in REQUIRED_PANEL_KINDS
    )
    report = inspect_paper_bundle(
        _bundle(
            parts,
            figures=(_figure(parts, panels=panels),),
            run_manifests=(parts.manifest, draft),
        )
    )
    assert report.error_codes == ("FIGURE_PANEL_MANIFEST_NOT_SEALED",)
    assert report.errors[0].actual == "run-draft-figure"


def test_a_table_cell_without_a_run_manifest_also_blocks_export(parts: _Parts) -> None:
    report = inspect_paper_bundle(_bundle(parts, run_manifests=()))
    assert not report.eligible
    assert set(report.error_codes) == {"TABLE_CELL_MANIFEST_MISSING", "FIGURE_PANEL_MANIFEST_MISSING"}


# ---------------------------------------------------------------------------
# 완료 검증 3: condition 누락 -- a planned Condition may never be dropped
# ---------------------------------------------------------------------------


def test_a_condition_matrix_missing_one_planned_condition_blocks_export(parts: _Parts) -> None:
    dropped = parts.specs[-1].condition.condition_id
    short = _condition_matrix(parts, specs=parts.specs[:-1])
    assert len(short.rows) == len(parts.specs) - 1

    report = inspect_paper_bundle(_bundle(parts, condition_matrix=short))

    assert not report.eligible
    assert report.error_codes == ("MISSING_CONDITION_ROW",)
    assert report.errors[0].actual == [dropped]


def test_a_condition_matrix_table_refuses_to_omit_a_planned_row(parts: _Parts) -> None:
    full = _condition_matrix(parts)
    with pytest.raises(ResearchValidationError) as excinfo:
        ConditionMatrixTable(
            table_id=full.table_id,
            caption=full.caption,
            planned_condition_ids=full.planned_condition_ids,
            rows=full.rows[:-1],
        )
    assert excinfo.value.code == "MISSING_CONDITION_ROW"
    assert excinfo.value.actual == [parts.specs[-1].condition.condition_id]


def test_the_condition_matrix_reports_failed_not_run_and_null_conditions(parts: _Parts) -> None:
    """Requirement 16.7: every arm appears with a status and a reason, not just the good ones."""
    matrix = _condition_matrix(parts)
    statuses = {row.execution_status for row in matrix.rows}
    assert statuses == {
        ExecutionStatus.COMPLETED,
        ExecutionStatus.FAILED,
        ExecutionStatus.NOT_RUN,
    }
    assert all(row.status_reason.strip() for row in matrix.rows)
    assert any(row.result_is_null for row in matrix.rows), "a measured-null condition stays reported"
    assert all(not row.measured for row in matrix.rows if row.execution_status is not ExecutionStatus.COMPLETED)
    assert len(matrix.rows) == len(parts.specs)


# ---------------------------------------------------------------------------
# 완료 검증 4: citation orphan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body_keys,expected_codes,expected_code,expected_actual",
    [
        (
            (CITATION_KEY, "ghost2024"),
            ("BODY_REFERENCE_MISSING",),
            "BODY_REFERENCE_MISSING",
            ["ghost2024"],
        ),
        (
            ("ghost2024",),
            ("BODY_REFERENCE_MISSING", "BODY_REFERENCE_ORPHAN"),
            "BODY_REFERENCE_ORPHAN",
            [CITATION_KEY],
        ),
    ],
    ids=["missing_body_reference", "orphan_ledger_entry"],
)
def test_an_unreconciled_citation_blocks_export(
    parts: _Parts,
    body_keys: tuple[str, ...],
    expected_codes: tuple[str, ...],
    expected_code: str,
    expected_actual: list[str],
) -> None:
    report = inspect_paper_bundle(_bundle(parts, body_citation_keys=body_keys))
    assert not report.eligible
    assert report.error_codes == expected_codes, "the citation defect must be the only finding"
    error = next(item for item in report.errors if item.code == expected_code)
    assert error.actual == expected_actual


UNVERIFIED_KEY = "unverified2024"


def _ledger_with_an_unverified_entry() -> CitationLedger:
    """A ledger whose first citation is verified and whose second never was."""
    ledger = claim_fx._citation_ledger()
    work_id = "work::unverified-preprint"
    ledger.add_candidate(
        CitationCandidate(
            candidate_id=f"{UNVERIFIED_KEY}-c1",
            canonical_work_id=work_id,
            source_id="arxiv",
            query=claim_fx.QUERY,
            executed_at_utc=UTC,
            result_rank=2,
            record_identifier=f"record::{UNVERIFIED_KEY}",
            result_content_hash=claim_fx._hash(f"search-result::{UNVERIFIED_KEY}"),
        )
    )
    ledger.screen(
        f"{UNVERIFIED_KEY}-c1",
        ScreeningDecision.INCLUDED,
        "road-network pursuit, metadata not yet cross-checked",
        screened_at_utc=UTC,
    )
    ledger.register_citation(
        CitationRecord(
            citation_key=UNVERIFIED_KEY,
            canonical_work_id=work_id,
            title="An unverified preprint on multi-vehicle pursuit",
            authors=("Unknown, A.",),
            year=2024,
            candidate_ids=ledger.candidate_ids_for_work(work_id),
            proposal_cross_check=ProposalCrossCheck.NOT_FROM_PROPOSAL,
        )
    )
    return ledger


def test_an_unverified_citation_blocks_export(parts: _Parts) -> None:
    ledger = _ledger_with_an_unverified_entry()
    report = inspect_paper_bundle(
        _bundle(
            parts,
            citation_ledger=ledger,
            body_citation_keys=(CITATION_KEY, UNVERIFIED_KEY),
        )
    )
    assert not report.eligible
    assert report.error_codes == ("UNVERIFIED_CITATION",)
    assert report.errors[0].actual == UNVERIFIED_KEY


# ---------------------------------------------------------------------------
# 완료 검증 5: 필수 논문 절 누락
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("omitted", list(PAPER_SECTIONS))
def test_a_bundle_missing_any_required_section_blocks_export(parts: _Parts, omitted: str) -> None:
    report = inspect_paper_bundle(_bundle(parts, sections=_sections(omit=omitted)))

    assert not report.eligible
    assert report.error_codes == ("MISSING_REQUIRED_SECTION",), (
        "dropping a section must be the only finding, not a cascade"
    )
    assert report.errors[0].actual == omitted, "the report must name the missing section"


def test_an_unknown_section_identifier_is_refused_at_construction() -> None:
    with pytest.raises(ResearchValidationError) as excinfo:
        PaperSection(section_id="acknowledgements", title="Thanks", body="Thanks to everyone.")
    assert excinfo.value.code == "UNKNOWN_PAPER_SECTION"


def test_an_empty_section_body_is_refused_at_construction() -> None:
    with pytest.raises(ResearchValidationError) as excinfo:
        PaperSection(section_id="ethics", title="Ethics", body="   ")
    assert excinfo.value.code == "MISSING_REQUIRED_FIELD"


# ---------------------------------------------------------------------------
# Requirement 14.11-14.13: the replay check
# ---------------------------------------------------------------------------


def _tampered_cell(cell: ResultTableCell, *, output_hash: str) -> ResultTableCell:
    return dataclasses.replace(
        cell, provenance=dataclasses.replace(cell.provenance, output_hash=output_hash)
    )


def test_a_cell_whose_registered_hash_does_not_replay_is_rejected(parts: _Parts) -> None:
    table = _table(parts)
    good = table.cells[0]
    assert verify_cell_replay(good) == ()
    require_cell_replay(good)

    tampered = _tampered_cell(good, output_hash=claim_fx._hash("a hash nothing produced"))
    errors = verify_cell_replay(tampered)
    assert [item.code for item in errors] == ["CELL_HASH_REPLAY_MISMATCH"]
    assert errors[0].expected == good.provenance.output_hash
    with pytest.raises(ResearchValidationError) as excinfo:
        require_cell_replay(tampered)
    assert excinfo.value.code == "CELL_HASH_REPLAY_MISMATCH"


def test_a_tampered_cell_blocks_the_whole_bundle_export(parts: _Parts) -> None:
    table = _table(parts)
    tampered = _tampered_cell(table.cells[0], output_hash=claim_fx._hash("edited after generation"))
    broken = dataclasses.replace(table, cells=(tampered, *table.cells[1:]))

    with pytest.raises(PaperExportError) as excinfo:
        export_paper_bundle(_bundle(parts, tables=(broken,)))

    assert excinfo.value.code == "CELL_HASH_REPLAY_MISMATCH"
    assert excinfo.value.report.error_codes == ("CELL_HASH_REPLAY_MISMATCH",)
    assert excinfo.value.record.details["export_report_hash"] == str(
        excinfo.value.report.content_hash
    )


def test_editing_a_recorded_input_hash_also_breaks_the_replay(parts: _Parts) -> None:
    """Provenance is content-addressed: changing the inputs invalidates the output hash."""
    cell = _table(parts).cells[0]
    relabelled = dataclasses.replace(
        cell,
        provenance=dataclasses.replace(
            cell.provenance, input_hashes={"metrics": claim_fx._hash("some other run's metrics")}
        ),
    )
    assert [item.code for item in verify_cell_replay(relabelled)] == ["CELL_HASH_REPLAY_MISMATCH"]


def test_a_tampered_panel_blocks_export(parts: _Parts) -> None:
    panels = list(_figure(parts).panels)
    panels[0] = dataclasses.replace(
        panels[0],
        provenance=dataclasses.replace(
            panels[0].provenance, output_hash=claim_fx._hash("panel redrawn by hand")
        ),
    )
    assert [item.code for item in verify_panel_replay(panels[0])] == ["PANEL_HASH_REPLAY_MISMATCH"]
    report = inspect_paper_bundle(_bundle(parts, figures=(_figure(parts, panels=tuple(panels)),)))
    assert report.error_codes == ("PANEL_HASH_REPLAY_MISMATCH",)


def test_the_sidecar_carries_every_requirement_14_9_field(parts: _Parts) -> None:
    provenance = _table(parts).cells[0].provenance
    assert provenance.artifact_id == f"{TABLE_ID}::seed:0::estimate_a"
    assert provenance.run_id == RUN_ID
    assert "build_result_table" in provenance.generator
    assert provenance.code_revision == PAPER_GENERATOR_CODE_REVISION
    assert provenance.config_hash == CONFIG_HASH
    assert provenance.input_hashes["metrics"] == INPUT_HASHES["metrics"]
    assert provenance.input_hashes["comparison_result"] == parts.statistics.result_hash
    assert provenance.output_hash == provenance.replayed_output_hash(
        _table(parts).cells[0].value
    )


# ---------------------------------------------------------------------------
# Requirement 16.4-16.5: the table and figure schemas
# ---------------------------------------------------------------------------


def test_the_result_table_carries_seed_rows_aggregate_ci_test_effect_and_counts(parts: _Parts) -> None:
    table = _table(parts)
    aggregate = table.aggregate
    assert aggregate.interval.confidence_level == 0.95
    assert aggregate.n_seeds == 5
    assert aggregate.n_pairs == 500
    assert aggregate.p_value_raw is not None
    assert table.seed_ids == (0, 1, 2, 3, 4)
    for seed in table.seed_ids:
        for column in SEED_COLUMNS:
            assert any(cell.cell_id == f"{TABLE_ID}::seed:{seed}::{column}" for cell in table.cells)
    for column in AGGREGATE_COLUMNS:
        assert any(cell.cell_id == f"{TABLE_ID}::{AGGREGATE_ROW_KEY}::{column}" for cell in table.cells)


@pytest.mark.parametrize("column", list(AGGREGATE_COLUMNS))
def test_a_result_table_missing_any_mandatory_aggregate_column_is_refused(
    parts: _Parts, column: str
) -> None:
    table = _table(parts)
    dropped = f"{TABLE_ID}::{AGGREGATE_ROW_KEY}::{column}"
    with pytest.raises(ResearchValidationError) as excinfo:
        dataclasses.replace(
            table, cells=tuple(cell for cell in table.cells if cell.cell_id != dropped)
        )
    assert excinfo.value.code == "INCOMPLETE_AGGREGATE_ROW"
    assert excinfo.value.actual == column


@pytest.mark.parametrize("omitted", list(REQUIRED_PANEL_KINDS))
def test_a_result_figure_missing_any_mandatory_panel_is_refused(
    parts: _Parts, omitted: FigurePanelKind
) -> None:
    panels = tuple(_panel(parts, kind) for kind in REQUIRED_PANEL_KINDS if kind is not omitted)
    with pytest.raises(ResearchValidationError) as excinfo:
        _figure(parts, panels=panels)
    assert excinfo.value.code == "INCOMPLETE_RESULT_FIGURE"
    assert excinfo.value.actual == [omitted.value]


# ---------------------------------------------------------------------------
# Requirement 16.6: the trajectory rule is recorded before the labels are seen
# ---------------------------------------------------------------------------


def test_a_trajectory_rule_cannot_be_recorded_against_an_unsealed_protocol(parts: _Parts) -> None:
    draft = parts.protocols.create(
        questions=default_research_questions(), specifications=claim_fx._specifications()
    )
    with pytest.raises(ResearchValidationError) as excinfo:
        freeze_trajectory_selection_rule(
            draft,
            rule_id="rule-after-the-fact",
            metric="time_to_capture_s",
            tie_break=("total_distance_m",),
            higher_is_better=False,
            declared_at_utc=UTC,
        )
    assert excinfo.value.code == "TRAJECTORY_RULE_NOT_PRE_REGISTERED"


def test_a_rule_frozen_under_another_protocol_blocks_export(parts: _Parts) -> None:
    other = parts.protocols.seal(
        parts.protocols.create(
            questions=default_research_questions(), specifications=claim_fx._specifications()
        ).protocol_id,
        signer=claim_fx.SIGNER,
    )
    foreign = freeze_trajectory_selection_rule(
        other,
        rule_id="rule-from-another-study",
        metric="time_to_capture_s",
        tie_break=("total_distance_m",),
        higher_is_better=False,
        declared_at_utc=UTC,
    )
    selection = select_representative_trajectory(
        foreign, _candidates("success"), outcome=TrajectoryOutcome.SUCCESS
    )
    panels = []
    for kind in REQUIRED_PANEL_KINDS:
        if kind is FigurePanelKind.SUCCESS_TRAJECTORY:
            panels.append(
                build_figure_panel(
                    figure_id=FIGURE_ID,
                    panel_key=kind.value,
                    kind=kind,
                    payload={"kind": kind.value, "trajectory_id": selection.selected.trajectory_id},
                    gate=parts.gate,
                    run_id=RUN_ID,
                    config_hash=CONFIG_HASH,
                    input_hashes=INPUT_HASHES,
                    selection=selection,
                )
            )
        else:
            panels.append(_panel(parts, kind))

    report = inspect_paper_bundle(_bundle(parts, figures=(_figure(parts, panels=tuple(panels)),)))
    assert report.error_codes == ("TRAJECTORY_RULE_PROTOCOL_MISMATCH",)


def test_a_trajectory_panel_requires_the_rule_that_selected_it(parts: _Parts) -> None:
    with pytest.raises(ResearchValidationError) as excinfo:
        build_figure_panel(
            figure_id=FIGURE_ID,
            panel_key="success_trajectory",
            kind=FigurePanelKind.SUCCESS_TRAJECTORY,
            payload={"trajectory_id": "success-0"},
            gate=parts.gate,
            run_id=RUN_ID,
            config_hash=CONFIG_HASH,
            input_hashes=INPUT_HASHES,
        )
    assert excinfo.value.code == "MISSING_TRAJECTORY_SELECTION_RULE"


def test_the_rule_orders_by_metric_then_the_recorded_tie_break(parts: _Parts) -> None:
    rule = _selection_rule(parts)
    tied = tuple(
        TrajectoryCandidate(
            trajectory_id=f"tied-{index}",
            run_id=RUN_ID,
            metrics={
                "time_to_capture_s": 100.0,
                "containment_area_m2": float(10 - index),
                "total_distance_m": 3000.0,
            },
        )
        for index in range(3)
    )
    selection = select_representative_trajectory(rule, tied, outcome=TrajectoryOutcome.SUCCESS)
    assert selection.selected.trajectory_id == "tied-2", "the tie-break metric decides, not input order"
    assert (
        select_representative_trajectory(
            rule, tuple(reversed(tied)), outcome=TrajectoryOutcome.SUCCESS
        ).selected.trajectory_id
        == "tied-2"
    )


def test_a_candidate_missing_a_rule_metric_cannot_be_selected(parts: _Parts) -> None:
    rule = _selection_rule(parts)
    with pytest.raises(ResearchValidationError) as excinfo:
        select_representative_trajectory(
            rule,
            (
                TrajectoryCandidate(
                    trajectory_id="unmeasured",
                    run_id=RUN_ID,
                    metrics={"time_to_capture_s": 100.0},
                ),
            ),
            outcome=TrajectoryOutcome.SUCCESS,
        )
    assert excinfo.value.code == "TRAJECTORY_METRIC_NOT_RECORDED"


# ---------------------------------------------------------------------------
# Requirement 16.2-16.3, 16.8: notation, seal, and the scope sections
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("omitted", list(REQUIRED_NOTATION_KINDS))
def test_a_bundle_missing_any_equation_or_symbol_definition_blocks_export(
    parts: _Parts, omitted: NotationKind
) -> None:
    report = inspect_paper_bundle(_bundle(parts, notation=_notation(omit=omitted)))
    assert report.error_codes == ("MISSING_NOTATION_DEFINITION",)
    assert report.errors[0].actual == omitted.value


def test_an_unsealed_protocol_blocks_export(parts: _Parts) -> None:
    draft = parts.protocols.create(
        questions=default_research_questions(), specifications=claim_fx._specifications()
    )
    report = inspect_paper_bundle(_bundle(parts, protocol=draft))
    assert "PROTOCOL_NOT_SEALED" in report.error_codes


def test_indistinct_validity_ethics_and_reproducibility_sections_block_export(parts: _Parts) -> None:
    merged = tuple(
        dataclasses.replace(section, body="All limitations are discussed together.")
        if section.section_id in ("validity_limitations", "ethics")
        else section
        for section in _sections()
    )
    report = inspect_paper_bundle(_bundle(parts, sections=merged))
    assert report.error_codes == ("SCOPE_SECTIONS_NOT_DISTINGUISHED",)


def test_a_field_readiness_derivation_in_any_section_blocks_export(parts: _Parts) -> None:
    overclaimed = tuple(
        dataclasses.replace(
            section, body="Cooperative containment delivers a real-world capture rate gain."
        )
        if section.section_id == "conclusion"
        else section
        for section in _sections()
    )
    report = inspect_paper_bundle(_bundle(parts, sections=overclaimed))
    assert report.error_codes == ("FIELD_READINESS_DERIVATION_BLOCKED",)
    assert report.errors[0].details["section_id"] == "conclusion"


@pytest.mark.parametrize("section_id", list(CORE_RESULT_SECTIONS))
def test_future_work_systems_may_not_appear_in_a_core_result_section(
    parts: _Parts, section_id: str
) -> None:
    contaminated = tuple(
        dataclasses.replace(
            section, body="Our results incorporate cctv detection fusion across the network."
        )
        if section.section_id == section_id
        else section
        for section in _sections()
    )
    report = inspect_paper_bundle(_bundle(parts, sections=contaminated))
    assert report.error_codes == ("FUTURE_WORK_IN_CORE_SECTION",)
    assert report.errors[0].actual == ["cctv"]
    assert report.errors[0].details["section_id"] == section_id


# ---------------------------------------------------------------------------
# Requirement 19.7, 19.9: every defect survives into one report
# ---------------------------------------------------------------------------


def test_simultaneous_defects_are_all_reported_at_once(parts: _Parts) -> None:
    """No single failure short-circuits the others -- five defects, five findings."""
    report = inspect_paper_bundle(
        _bundle(
            parts,
            sections=_sections(omit="ethics"),
            notation=_notation(omit=NotationKind.REWARD),
            body_citation_keys=("ghost2024",),
            condition_matrix=_condition_matrix(parts, specs=parts.specs[:-1]),
            run_manifests=(),
        )
    )
    assert not report.eligible
    assert {
        "MISSING_REQUIRED_SECTION",
        "MISSING_NOTATION_DEFINITION",
        "BODY_REFERENCE_MISSING",
        "BODY_REFERENCE_ORPHAN",
        "MISSING_CONDITION_ROW",
        "TABLE_CELL_MANIFEST_MISSING",
        "FIGURE_PANEL_MANIFEST_MISSING",
    } <= set(report.error_codes)


def test_a_bundle_with_no_table_or_figure_is_blocked(parts: _Parts) -> None:
    report = inspect_paper_bundle(_bundle(parts, tables=(), figures=()))
    assert {"MISSING_RESULT_TABLE", "MISSING_RESULT_FIGURE"} <= set(report.error_codes)


def test_an_export_report_is_content_addressed_and_never_half_eligible(parts: _Parts) -> None:
    report = inspect_paper_bundle(_bundle(parts))
    assert report.eligible and report.errors == ()
    assert report.content_hash == inspect_paper_bundle(_bundle(parts)).content_hash
    assert report.verify_hash()

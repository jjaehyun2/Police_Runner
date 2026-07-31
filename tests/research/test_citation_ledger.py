"""Citation ledger, verification gating, and novelty comparison-axis tests."""

from __future__ import annotations

import hashlib

import pytest

from pursuit_evasion_rl.research.domain import (
    ClaimScope,
    EvidenceRecord,
    EvidenceType,
    ScreeningDecision,
)
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.paper.citations import (
    CitationCandidate,
    CitationLedger,
    CitationRecord,
    MetadataCrossCheck,
    NoveltyEvidenceExclusion,
    PersistentIdentifierKind,
    ProposalCrossCheck,
    ScreeningState,
    SearchProtocolSpec,
    SearchSource,
    SearchSourceKind,
    SourceLocation,
    VerificationStatus,
    reconcile_body_citations,
)
from pursuit_evasion_rl.research.paper.related_work import (
    NOT_REPORTED,
    REPORTED_NONE,
    ComparisonAxis,
    NoveltyComparisonGraph,
    RelatedWorkCell,
    RelatedWorkMatrix,
)

pytestmark = pytest.mark.offline

UTC = "2026-07-31T00:00:00+00:00"
QUERY = "multi-agent reinforcement learning vehicle pursuit road network"


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _sources() -> tuple[SearchSource, ...]:
    return (
        SearchSource(
            source_id="openalex",
            name="OpenAlex",
            kind=SearchSourceKind.ACADEMIC_INDEX,
            operator="OurResearch",
            endpoint="https://api.openalex.org/works",
        ),
        SearchSource(
            source_id="scopus",
            name="Scopus",
            kind=SearchSourceKind.ACADEMIC_INDEX,
            operator="Elsevier",
            endpoint="https://api.elsevier.com/content/search/scopus",
        ),
        SearchSource(
            source_id="arxiv",
            name="arXiv",
            kind=SearchSourceKind.PREPRINT_SERVER,
            operator="Cornell University",
            endpoint="https://export.arxiv.org/api/query",
        ),
    )


def _protocol(sources: tuple[SearchSource, ...] | None = None) -> SearchProtocolSpec:
    return SearchProtocolSpec(
        search_date_utc=UTC,
        queries=(QUERY,),
        sources=_sources() if sources is None else sources,
        period_start="2015-01-01",
        period_end="2026-07-31",
        languages=("en", "ko"),
        inclusion_criteria=("road-network pursuit or interception of a fleeing vehicle",),
        exclusion_criteria=("no evaluation on a road network",),
    )


def _cross_check(*, passed: bool = True) -> MetadataCrossCheck:
    return MetadataCrossCheck(
        identifier="10.1007/s41109-024-00689-1",
        identifier_kind=PersistentIdentifierKind.DOI,
        metadata_source="Crossref",
        checked_at_utc=UTC,
        author_match=True,
        year_match=passed,
        title_match=True,
    )


def _location(token: str) -> SourceLocation:
    return SourceLocation(
        original_text_content_hash=_hash(token),
        section="4 Experiments",
        table="Table 2",
    )


def _ledger() -> CitationLedger:
    return CitationLedger(_protocol())


def _add_work(
    ledger: CitationLedger,
    *,
    key: str,
    work_id: str,
    title: str,
    year: int,
    source_id: str = "openalex",
    rank: int = 1,
    screenings: tuple[tuple[ScreeningDecision, str], ...] = (
        (ScreeningDecision.INCLUDED, "road-network pursuit with a learned policy"),
    ),
    verified: bool = True,
    proposal_label: str | None = None,
) -> CitationRecord:
    candidate = CitationCandidate(
        candidate_id=f"{key}-c1",
        canonical_work_id=work_id,
        source_id=source_id,
        query=QUERY,
        executed_at_utc=UTC,
        result_rank=rank,
        record_identifier=f"record::{key}",
        result_content_hash=_hash(f"search-result::{key}"),
    )
    ledger.add_candidate(candidate)
    for decision, reason in screenings:
        ledger.screen(candidate.candidate_id, decision, reason, screened_at_utc=UTC)
    record = CitationRecord(
        citation_key=key,
        canonical_work_id=work_id,
        title=title,
        authors=("van Droffelaar, I.",) if "droffelaar" in work_id else ("Yang, X.",),
        year=year,
        candidate_ids=ledger.candidate_ids_for_work(work_id),
        proposal_cross_check=(
            ProposalCrossCheck.PROPOSAL_TEXT_VERIFIED
            if proposal_label
            else ProposalCrossCheck.NOT_FROM_PROPOSAL
        ),
        proposal_reference_label=proposal_label,
        metadata_cross_check=_cross_check() if verified else None,
        source_location=_location(key) if verified else None,
    )
    return ledger.register_citation(record)


def _verified_matrix() -> RelatedWorkMatrix:
    ledger = _ledger()
    _add_work(
        ledger,
        key="yang2023",
        work_id="work::yang-progression-cognition",
        title="Progression Cognition Reinforcement Learning for Multi-Vehicle Pursuit",
        year=2023,
        proposal_label="[6]",
    )
    matrix = RelatedWorkMatrix(ledger)
    matrix.add_row(
        "yang2023",
        (
            RelatedWorkCell(
                axis=ComparisonAxis.COOPERATIVE_ENCIRCLEMENT,
                value="joint reward for multi-vehicle capture",
                reported=True,
                source_location="Section 4",
            ),
            RelatedWorkCell(
                axis=ComparisonAxis.MAP_SOURCE_REALISM,
                value="synthetic grid road network",
                reported=True,
                source_location="Section 5.1",
            ),
        ),
    )
    return matrix


def _evidence(record_id: str = "ev-1") -> EvidenceRecord:
    return EvidenceRecord(
        record_id=record_id,
        evidence_type=EvidenceType.PAPER_CLAIM,
        producer="citation-ledger",
        created_at_utc=UTC,
        method="original-text extraction with DOI metadata cross-check",
        source="yang2023 Section 5.1",
        extracted_value="synthetic grid road network",
        verification="Crossref metadata and original-text hash matched",
    )


def test_body_reconciliation_names_missing_and_orphan_keys():
    ledger = _ledger()
    _add_work(
        ledger,
        key="yang2023",
        work_id="work::yang-progression-cognition",
        title="Progression Cognition Reinforcement Learning for Multi-Vehicle Pursuit",
        year=2023,
    )
    _add_work(
        ledger,
        key="droffelaar2024",
        work_id="work::droffelaar-graph-coarsening",
        title="Graph coarsening for fugitive interception",
        year=2024,
        source_id="scopus",
        rank=2,
    )

    report = reconcile_body_citations(ledger, ["yang2023", "kim2025"])

    assert report.missing_keys == ("kim2025",)
    assert report.orphan_keys == ("droffelaar2024",)
    assert report.missing_count == 1
    assert report.orphan_count == 1
    assert report.is_clean is False

    clean = reconcile_body_citations(ledger, ["yang2023", "droffelaar2024"])
    assert clean.missing_keys == ()
    assert clean.orphan_keys == ()
    assert clean.is_clean is True


def test_included_but_unverified_citation_is_excluded_from_novelty_evidence():
    ledger = _ledger()
    record = _add_work(
        ledger,
        key="yang2023",
        work_id="work::yang-progression-cognition",
        title="Progression Cognition Reinforcement Learning for Multi-Vehicle Pursuit",
        year=2023,
        verified=False,
    )

    assert ledger.screening_state("yang2023-c1") is ScreeningState.INCLUDED
    assert record.verification_status is VerificationStatus.UNVERIFIED
    assert record.unverified_reasons == ("metadata_cross_check_missing", "source_location_missing")
    assert ledger.novelty_exclusion("yang2023") is NoveltyEvidenceExclusion.UNVERIFIED
    assert ledger.novelty_eligible_citations() == ()


def test_failed_metadata_cross_check_is_unverified():
    ledger = _ledger()
    candidate = CitationCandidate(
        candidate_id="droffelaar2024-c1",
        canonical_work_id="work::droffelaar-graph-coarsening",
        source_id="scopus",
        query=QUERY,
        executed_at_utc=UTC,
        result_rank=1,
        record_identifier="record::droffelaar2024",
        result_content_hash=_hash("search-result::droffelaar2024"),
    )
    ledger.add_candidate(candidate)
    ledger.screen(
        candidate.candidate_id, ScreeningDecision.INCLUDED, "interception baseline", screened_at_utc=UTC
    )
    record = ledger.register_citation(
        CitationRecord(
            citation_key="droffelaar2024",
            canonical_work_id="work::droffelaar-graph-coarsening",
            title="Graph coarsening for fugitive interception",
            authors=("van Droffelaar, I.",),
            year=2024,
            candidate_ids=(candidate.candidate_id,),
            proposal_cross_check=ProposalCrossCheck.NOT_FROM_PROPOSAL,
            metadata_cross_check=_cross_check(passed=False),
            source_location=_location("droffelaar2024"),
        )
    )

    assert record.unverified_reasons == ("metadata_cross_check_failed",)
    assert ledger.novelty_exclusion("droffelaar2024") is NoveltyEvidenceExclusion.UNVERIFIED


def test_proposal_citation_without_text_cross_check_stays_unverified():
    ledger = _ledger()
    candidate = CitationCandidate(
        candidate_id="droffelaar2024-c1",
        canonical_work_id="work::droffelaar-simopt",
        source_id="scopus",
        query=QUERY,
        executed_at_utc=UTC,
        result_rank=1,
        record_identifier="record::droffelaar-simopt",
        result_content_hash=_hash("search-result::droffelaar-simopt"),
    )
    ledger.add_candidate(candidate)
    ledger.screen(
        candidate.candidate_id, ScreeningDecision.INCLUDED, "cited by the proposal", screened_at_utc=UTC
    )
    record = ledger.register_citation(
        CitationRecord(
            citation_key="droffelaar2024simopt",
            canonical_work_id="work::droffelaar-simopt",
            title="Simulation-optimization for real-time fugitive interception",
            authors=("van Droffelaar, I.",),
            year=2024,
            candidate_ids=(candidate.candidate_id,),
            proposal_cross_check=ProposalCrossCheck.PROPOSAL_TEXT_UNVERIFIED,
            proposal_reference_label="[3]",
            metadata_cross_check=_cross_check(),
            source_location=_location("droffelaar-simopt"),
        )
    )

    assert record.unverified_reasons == ("proposal_text_unverified",)
    assert ledger.novelty_exclusion("droffelaar2024simopt") is NoveltyEvidenceExclusion.UNVERIFIED


def test_screening_problems_are_distinguishable_from_verification_failure():
    ledger = _ledger()
    _add_work(
        ledger,
        key="unscreened",
        work_id="work::unscreened",
        title="Unscreened pursuit study",
        year=2022,
        screenings=(),
    )
    _add_work(
        ledger,
        key="conflicted",
        work_id="work::conflicted",
        title="Conflicted pursuit study",
        year=2021,
        source_id="scopus",
        rank=2,
        screenings=(
            (ScreeningDecision.INCLUDED, "road-network pursuit"),
            (ScreeningDecision.EXCLUDED, "no road-network evaluation"),
        ),
    )
    _add_work(
        ledger,
        key="repeated",
        work_id="work::repeated",
        title="Repeatedly screened pursuit study",
        year=2020,
        source_id="arxiv",
        rank=3,
        screenings=(
            (ScreeningDecision.INCLUDED, "road-network pursuit"),
            (ScreeningDecision.INCLUDED, "road-network pursuit"),
        ),
    )
    _add_work(
        ledger,
        key="rejected",
        work_id="work::rejected",
        title="Screened-out pursuit study",
        year=2019,
        source_id="arxiv",
        rank=4,
        screenings=((ScreeningDecision.EXCLUDED, "aerial pursuit only"),),
    )
    _add_work(
        ledger,
        key="unverified",
        work_id="work::unverified",
        title="Included but unverified pursuit study",
        year=2018,
        source_id="scopus",
        rank=5,
        verified=False,
    )

    assert ledger.screening_state("unscreened-c1") is ScreeningState.MISSING
    assert ledger.screening_state("conflicted-c1") is ScreeningState.INCONSISTENT
    assert ledger.screening_state("repeated-c1") is ScreeningState.DUPLICATE

    assert ledger.novelty_exclusions() == {
        "unscreened": NoveltyEvidenceExclusion.SCREENING_MISSING,
        "conflicted": NoveltyEvidenceExclusion.SCREENING_INCONSISTENT,
        "repeated": NoveltyEvidenceExclusion.SCREENING_DUPLICATE,
        "rejected": NoveltyEvidenceExclusion.SCREENED_EXCLUDED,
        "unverified": NoveltyEvidenceExclusion.UNVERIFIED,
    }
    assert ledger.novelty_eligible_citations() == ()


def test_screening_requires_a_reason():
    ledger = _ledger()
    _add_work(
        ledger,
        key="yang2023",
        work_id="work::yang-progression-cognition",
        title="Progression Cognition Reinforcement Learning for Multi-Vehicle Pursuit",
        year=2023,
        screenings=(),
    )
    with pytest.raises(ResearchValidationError) as excinfo:
        ledger.screen("yang2023-c1", ScreeningDecision.INCLUDED, "  ", screened_at_utc=UTC)
    assert excinfo.value.code == "MISSING_REQUIRED_FIELD"

    with pytest.raises(ResearchValidationError) as missing:
        ledger.screen("yang2023-c1", None, "no decision", screened_at_utc=UTC)
    assert missing.value.code == "MISSING_CLASSIFICATION"

    with pytest.raises(ResearchValidationError) as duplicate:
        ledger.screen(
            "yang2023-c1",
            [ScreeningDecision.INCLUDED, ScreeningDecision.EXCLUDED],
            "both",
            screened_at_utc=UTC,
        )
    assert duplicate.value.code == "DUPLICATE_CLASSIFICATION"


def test_duplicate_candidates_link_to_one_canonical_work_and_are_preserved():
    ledger = _ledger()
    for index, source_id in enumerate(("openalex", "scopus", "arxiv"), start=1):
        candidate = CitationCandidate(
            candidate_id=f"yang2023-c{index}",
            canonical_work_id="work::yang-progression-cognition",
            source_id=source_id,
            query=QUERY,
            executed_at_utc=UTC,
            result_rank=index,
            record_identifier=f"{source_id}::2306.05016",
            result_content_hash=_hash(f"{source_id}-result"),
        )
        ledger.add_candidate(candidate)
        ledger.screen(
            candidate.candidate_id, ScreeningDecision.INCLUDED, "road-network pursuit", screened_at_utc=UTC
        )

    assert ledger.candidate_ids_for_work("work::yang-progression-cognition") == (
        "yang2023-c1",
        "yang2023-c2",
        "yang2023-c3",
    )
    assert ledger.duplicate_links() == {
        "work::yang-progression-cognition": ("yang2023-c1", "yang2023-c2", "yang2023-c3")
    }

    with pytest.raises(ResearchValidationError) as excinfo:
        ledger.register_citation(
            CitationRecord(
                citation_key="yang2023",
                canonical_work_id="work::yang-progression-cognition",
                title="Progression Cognition Reinforcement Learning for Multi-Vehicle Pursuit",
                authors=("Yang, X.",),
                year=2023,
                candidate_ids=("yang2023-c1",),
                proposal_cross_check=ProposalCrossCheck.NOT_FROM_PROPOSAL,
                metadata_cross_check=_cross_check(),
                source_location=_location("yang2023"),
            )
        )
    assert excinfo.value.code == "UNLINKED_DUPLICATE_CANDIDATE"


def test_matrix_records_not_reported_distinctly_from_a_reported_absence():
    matrix = _verified_matrix()
    row = matrix.row("yang2023")

    assert tuple(cell.axis.value for cell in row.cells) == matrix.columns
    assert row.value(ComparisonAxis.COOPERATIVE_ENCIRCLEMENT) == "joint reward for multi-vehicle capture"
    assert row.cell(ComparisonAxis.SPATIAL_GENERALIZATION).value == NOT_REPORTED
    assert row.cell(ComparisonAxis.SPATIAL_GENERALIZATION).is_not_reported is True
    assert ComparisonAxis.SPATIAL_GENERALIZATION.value in row.not_reported_axes
    assert ComparisonAxis.COOPERATIVE_ENCIRCLEMENT.value not in row.not_reported_axes

    reported_absence = RelatedWorkCell(
        axis=ComparisonAxis.OFFLINE_LLM_COMPONENT,
        value=REPORTED_NONE,
        reported=True,
        source_location="Section 3",
    )
    assert reported_absence.value == REPORTED_NONE
    assert reported_absence.is_not_reported is False

    with pytest.raises(ResearchValidationError) as reserved:
        RelatedWorkCell(
            axis=ComparisonAxis.DECISION_LATENCY,
            value=NOT_REPORTED,
            reported=True,
            source_location="Section 3",
        )
    assert reserved.value.code == "RESERVED_CELL_VALUE"

    with pytest.raises(ResearchValidationError) as blank:
        RelatedWorkCell(
            axis=ComparisonAxis.DECISION_LATENCY, value="sub-second inference", reported=False
        )
    assert blank.value.code == "UNREPORTED_CELL_HAS_VALUE"


def test_matrix_rows_require_a_novelty_eligible_citation():
    ledger = _ledger()
    _add_work(
        ledger,
        key="unverified",
        work_id="work::unverified",
        title="Included but unverified pursuit study",
        year=2018,
        verified=False,
    )
    matrix = RelatedWorkMatrix(ledger)

    with pytest.raises(ResearchValidationError) as excinfo:
        matrix.add_row("unverified")
    assert excinfo.value.code == "UNVERIFIED_MATRIX_ROW"
    assert excinfo.value.actual == NoveltyEvidenceExclusion.UNVERIFIED.value


def test_novelty_claim_requires_a_direct_comparison_axis():
    matrix = _verified_matrix()
    graph = NoveltyComparisonGraph(matrix)
    graph.register_evidence(_evidence(), citation_key="yang2023")

    with pytest.raises(ResearchValidationError) as empty:
        graph.add_comparison(
            claim_id="novelty-1",
            text="본 연구는 실제 OSM 도로망에서 협력 포위를 평가한다",
            axes=(),
            evidence_ids=("ev-1",),
        )
    assert empty.value.code == "MISSING_COMPARISON_AXIS"

    with pytest.raises(ResearchValidationError) as indirect:
        graph.add_comparison(
            claim_id="novelty-1",
            text="본 연구는 실제 OSM 도로망에서 협력 포위를 평가한다",
            axes=(ComparisonAxis.SPATIAL_GENERALIZATION,),
            evidence_ids=("ev-1",),
        )
    assert indirect.value.code == "INDIRECT_COMPARISON_AXIS"
    assert indirect.value.actual == ComparisonAxis.SPATIAL_GENERALIZATION.value

    assert graph.comparisons() == ()


def test_novelty_evidence_must_be_backed_by_a_verified_citation():
    matrix = _verified_matrix()
    _add_work(
        matrix.ledger,
        key="unverified",
        work_id="work::unverified",
        title="Included but unverified pursuit study",
        year=2018,
        source_id="scopus",
        rank=2,
        verified=False,
    )
    graph = NoveltyComparisonGraph(matrix)

    with pytest.raises(ResearchValidationError) as excinfo:
        graph.register_evidence(_evidence(), citation_key="unverified")
    assert excinfo.value.code == "UNVERIFIED_NOVELTY_EVIDENCE"

    graph.register_evidence(_evidence(), citation_key="yang2023")
    with pytest.raises(ResearchValidationError) as unregistered:
        graph.add_comparison(
            claim_id="novelty-1",
            text="본 연구는 실제 OSM 도로망에서 협력 포위를 평가한다",
            axes=(ComparisonAxis.MAP_SOURCE_REALISM,),
            evidence_ids=("ev-missing",),
        )
    assert unregistered.value.code == "UNVERIFIED_NOVELTY_EVIDENCE"


def test_novelty_comparison_populates_a_domain_claim_record():
    matrix = _verified_matrix()
    graph = NoveltyComparisonGraph(matrix)
    graph.register_evidence(_evidence(), citation_key="yang2023")

    comparison = graph.add_comparison(
        claim_id="novelty-1",
        text="본 연구는 합성 격자 도로망을 사용한 선행 연구와 달리 실제 OSM 도로망에서 협력 포위를 평가한다",
        axes=(ComparisonAxis.MAP_SOURCE_REALISM, ComparisonAxis.COOPERATIVE_ENCIRCLEMENT),
        evidence_ids=("ev-1",),
    )

    assert comparison.claim.scope is ClaimScope.NOVELTY
    assert comparison.comparison_axes == (
        ComparisonAxis.MAP_SOURCE_REALISM.value,
        ComparisonAxis.COOPERATIVE_ENCIRCLEMENT.value,
    )
    assert comparison.evidence_ids == ("ev-1",)
    assert comparison.claim.citation_ids == ("yang2023",)
    assert comparison.axis_citations == {
        ComparisonAxis.MAP_SOURCE_REALISM.value: ("yang2023",),
        ComparisonAxis.COOPERATIVE_ENCIRCLEMENT.value: ("yang2023",),
    }
    assert graph.claim_records() == (comparison.claim,)
    assert graph.evidence_citation("ev-1") == "yang2023"


def test_priority_language_is_rejected_when_the_graph_builds_the_claim():
    matrix = _verified_matrix()
    graph = NoveltyComparisonGraph(matrix)
    graph.register_evidence(_evidence(), citation_key="yang2023")

    with pytest.raises(ResearchValidationError) as excinfo:
        graph.add_comparison(
            claim_id="novelty-1",
            text="본 연구는 실제 OSM 도로망에 협력 포위를 적용한 최초의 사례이다",
            axes=(ComparisonAxis.MAP_SOURCE_REALISM,),
            evidence_ids=("ev-1",),
        )
    assert excinfo.value.code == "FORBIDDEN_PRIORITY_CLAIM"
    assert graph.comparisons() == ()


def test_search_protocol_requires_three_independent_and_two_academic_sources():
    complete = _protocol()
    assert len(complete.sources) == 3

    with pytest.raises(ResearchValidationError) as too_few:
        _protocol(sources=_sources()[:2])
    assert too_few.value.code == "INSUFFICIENT_SEARCH_SOURCES"
    assert too_few.value.expected == 3
    assert too_few.value.actual == 2

    shared_operator = _sources() + (
        SearchSource(
            source_id="google-scholar",
            name="Google Scholar",
            kind=SearchSourceKind.ACADEMIC_INDEX,
            operator="Elsevier",
            endpoint="https://scholar.google.com",
        ),
    )
    assert len(_protocol(sources=shared_operator).sources) == 4

    one_index = (
        _sources()[0],
        _sources()[2],
        SearchSource(
            source_id="ieee",
            name="IEEE Xplore",
            kind=SearchSourceKind.PUBLISHER_PORTAL,
            operator="IEEE",
            endpoint="https://ieeexplore.ieee.org/rest/search",
        ),
    )
    with pytest.raises(ResearchValidationError) as too_few_indices:
        _protocol(sources=one_index)
    assert too_few_indices.value.code == "INSUFFICIENT_ACADEMIC_INDEX_SOURCES"
    assert too_few_indices.value.actual == 1


def test_candidates_must_match_the_frozen_search_protocol():
    ledger = _ledger()
    with pytest.raises(ResearchValidationError) as unknown_source:
        ledger.add_candidate(
            CitationCandidate(
                candidate_id="x-c1",
                canonical_work_id="work::x",
                source_id="semantic-scholar",
                query=QUERY,
                executed_at_utc=UTC,
                result_rank=1,
                record_identifier="record::x",
                result_content_hash=_hash("x"),
            )
        )
    assert unknown_source.value.code == "UNKNOWN_SEARCH_SOURCE"

    with pytest.raises(ResearchValidationError) as unknown_query:
        ledger.add_candidate(
            CitationCandidate(
                candidate_id="x-c1",
                canonical_work_id="work::x",
                source_id="arxiv",
                query="unregistered ad-hoc query",
                executed_at_utc=UTC,
                result_rank=1,
                record_identifier="record::x",
                result_content_hash=_hash("x"),
            )
        )
    assert unknown_query.value.code == "UNKNOWN_SEARCH_QUERY"


def test_candidate_provenance_fields_are_validated():
    with pytest.raises(ResearchValidationError) as rank:
        CitationCandidate(
            candidate_id="x-c1",
            canonical_work_id="work::x",
            source_id="arxiv",
            query=QUERY,
            executed_at_utc=UTC,
            result_rank=0,
            record_identifier="record::x",
            result_content_hash=_hash("x"),
        )
    assert rank.value.code == "INVALID_RESULT_RANK"

    with pytest.raises(ResearchValidationError) as digest:
        CitationCandidate(
            candidate_id="x-c1",
            canonical_work_id="work::x",
            source_id="arxiv",
            query=QUERY,
            executed_at_utc=UTC,
            result_rank=1,
            record_identifier="record::x",
            result_content_hash="not-a-hash",
        )
    assert digest.value.code == "INVALID_CONTENT_HASH"

    with pytest.raises(ResearchValidationError) as naive:
        CitationCandidate(
            candidate_id="x-c1",
            canonical_work_id="work::x",
            source_id="arxiv",
            query=QUERY,
            executed_at_utc="2026-07-31T00:00:00",
            result_rank=1,
            record_identifier="record::x",
            result_content_hash=_hash("x"),
        )
    assert naive.value.code == "INVALID_UTC_TIMESTAMP"


def test_source_location_requires_a_page_section_or_table():
    with pytest.raises(ResearchValidationError) as excinfo:
        SourceLocation(original_text_content_hash=_hash("t"))
    assert excinfo.value.code == "MISSING_SOURCE_LOCATION"

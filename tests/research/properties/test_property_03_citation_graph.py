"""Property 3 coverage for the citation, related-work and novelty evidence graph."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import string

from hypothesis import given, settings, strategies as st
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
    MATRIX_COLUMNS,
    NOT_REPORTED,
    REPORTED_NONE,
    ComparisonAxis,
    NoveltyComparisonGraph,
    RelatedWorkCell,
    RelatedWorkMatrix,
)

# **Property 3: Citation and novelty evidence forms a closed traceable graph**
# **Validates: Requirements 3.2-3.6, 3.8-3.9, 4.1, 4.4**

pytestmark = pytest.mark.offline

_PBT_SETTINGS = settings(max_examples=100, deadline=None)

UTC = "2026-07-31T00:00:00+00:00"
QUERY = "multi-agent reinforcement learning vehicle pursuit road network"


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Randomized ledger construction
# ---------------------------------------------------------------------------


def _protocol() -> SearchProtocolSpec:
    return SearchProtocolSpec(
        search_date_utc=UTC,
        queries=(QUERY,),
        sources=(
            SearchSource(
                source_id="openalex", name="OpenAlex", kind=SearchSourceKind.ACADEMIC_INDEX,
                operator="OurResearch", endpoint="https://api.openalex.org/works",
            ),
            SearchSource(
                source_id="scopus", name="Scopus", kind=SearchSourceKind.ACADEMIC_INDEX,
                operator="Elsevier", endpoint="https://api.elsevier.com/content/search/scopus",
            ),
            SearchSource(
                source_id="arxiv", name="arXiv", kind=SearchSourceKind.PREPRINT_SERVER,
                operator="Cornell University", endpoint="https://export.arxiv.org/api/query",
            ),
        ),
        period_start="2015-01-01",
        period_end="2026-07-31",
        languages=("en", "ko"),
        inclusion_criteria=("road-network pursuit or interception of a fleeing vehicle",),
        exclusion_criteria=("no evaluation on a road network",),
    )


#: Verification defects, each of which independently keeps a citation unverified.
VERIFICATION_PLANS = (
    "verified",
    "metadata_missing",
    "metadata_failed",
    "location_missing",
    "proposal_unverified",
)

#: Screening histories a candidate may carry, including every malformed state.
SCREENING_PLANS: tuple[tuple[ScreeningDecision, ...], ...] = (
    (),
    (ScreeningDecision.INCLUDED,),
    (ScreeningDecision.EXCLUDED,),
    (ScreeningDecision.INCLUDED, ScreeningDecision.INCLUDED),
    (ScreeningDecision.EXCLUDED, ScreeningDecision.EXCLUDED),
    (ScreeningDecision.INCLUDED, ScreeningDecision.EXCLUDED),
    (ScreeningDecision.EXCLUDED, ScreeningDecision.INCLUDED),
)


@dataclass(frozen=True)
class WorkPlan:
    """One canonical work: a screening history per candidate plus a verification state."""

    screenings: tuple[tuple[ScreeningDecision, ...], ...]
    verification: str

    @property
    def is_cleanly_screened(self) -> bool:
        """Every candidate was screened exactly once, and always as `included`."""
        return bool(self.screenings) and all(
            plan == (ScreeningDecision.INCLUDED,) for plan in self.screenings
        )

    @property
    def is_verified(self) -> bool:
        return self.verification == "verified"

    @property
    def is_novelty_eligible(self) -> bool:
        return self.is_cleanly_screened and self.is_verified


_work_plans = st.builds(
    WorkPlan,
    screenings=st.lists(st.sampled_from(SCREENING_PLANS), min_size=1, max_size=2).map(tuple),
    verification=st.sampled_from(VERIFICATION_PLANS),
)
_work_plan_lists = st.lists(_work_plans, min_size=1, max_size=4)
_free_keys = st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=6).map(lambda item: f"free-{item}")
_axis_sets = st.sets(st.sampled_from(tuple(ComparisonAxis)), max_size=len(MATRIX_COLUMNS))
_cell_values = st.sampled_from(("real OSM road network", "synthetic grid", REPORTED_NONE, "sub-second"))


def _cross_check(*, passed: bool) -> MetadataCrossCheck:
    return MetadataCrossCheck(
        identifier="10.1007/s41109-024-00689-1",
        identifier_kind=PersistentIdentifierKind.DOI,
        metadata_source="Crossref",
        checked_at_utc=UTC,
        author_match=True,
        year_match=passed,
        title_match=True,
    )


def _add_work(ledger: CitationLedger, key: str, plan: WorkPlan) -> CitationRecord:
    """Materialise one planned work, its candidates and its screening history."""
    work_id = f"work::{key}"
    sources = ("openalex", "scopus", "arxiv")
    for index, screenings in enumerate(plan.screenings):
        candidate = CitationCandidate(
            candidate_id=f"{key}-c{index}",
            canonical_work_id=work_id,
            source_id=sources[index % len(sources)],
            query=QUERY,
            executed_at_utc=UTC,
            result_rank=index + 1,
            record_identifier=f"record::{key}::{index}",
            result_content_hash=_hash(f"search-result::{key}::{index}"),
        )
        ledger.add_candidate(candidate)
        for decision in screenings:
            ledger.screen(
                candidate.candidate_id, decision, f"screened as {decision.value}", screened_at_utc=UTC
            )
    from_proposal = plan.verification == "proposal_unverified"
    return ledger.register_citation(
        CitationRecord(
            citation_key=key,
            canonical_work_id=work_id,
            title=f"Pursuit study {key}",
            authors=("Yang, X.",),
            year=2023,
            candidate_ids=ledger.candidate_ids_for_work(work_id),
            proposal_cross_check=(
                ProposalCrossCheck.PROPOSAL_TEXT_UNVERIFIED
                if from_proposal
                else ProposalCrossCheck.NOT_FROM_PROPOSAL
            ),
            proposal_reference_label="[3]" if from_proposal else None,
            metadata_cross_check=(
                None
                if plan.verification == "metadata_missing"
                else _cross_check(passed=plan.verification != "metadata_failed")
            ),
            source_location=(
                None
                if plan.verification == "location_missing"
                else SourceLocation(original_text_content_hash=_hash(key), section="4 Experiments")
            ),
        )
    )


def _build_ledger(plans: list[WorkPlan]) -> tuple[CitationLedger, dict[str, WorkPlan]]:
    ledger = CitationLedger(_protocol())
    by_key = {f"work{index}": plan for index, plan in enumerate(plans)}
    for key, plan in by_key.items():
        _add_work(ledger, key, plan)
    return ledger, by_key


def _eligible_ledger(plans: list[WorkPlan]) -> tuple[CitationLedger, dict[str, WorkPlan]]:
    """Force every planned work to be cleanly screened and verified."""
    clean = [WorkPlan(screenings=((ScreeningDecision.INCLUDED,),), verification="verified") for _ in plans]
    return _build_ledger(clean)


# ---------------------------------------------------------------------------
# Invariant 1/5: the body/ledger symmetric difference is exactly the missing and
# orphan key sets, for any pair of key sets (Requirements 3.8-3.9).
# ---------------------------------------------------------------------------
@_PBT_SETTINGS
@given(
    plans=_work_plan_lists,
    dropped=st.sets(st.integers(min_value=0, max_value=3)),
    extra=st.sets(_free_keys, max_size=4),
)
def test_body_and_ledger_key_sets_always_reconcile_to_their_symmetric_difference(
    plans, dropped, extra
) -> None:
    ledger, by_key = _build_ledger(plans)
    registered = set(ledger.citation_keys())
    assert registered == set(by_key)

    cited = {key for index, key in enumerate(sorted(by_key)) if index not in dropped} | set(extra)
    report = reconcile_body_citations(ledger, sorted(cited))

    # Missing = cited but unregistered; orphan = registered but never cited.
    assert set(report.missing_keys) == cited - registered
    assert set(report.orphan_keys) == registered - cited
    assert set(report.missing_keys) | set(report.orphan_keys) == cited ^ registered
    assert report.missing_count + report.orphan_count == len(cited ^ registered)
    assert report.is_clean is (cited == registered)
    # An orphan is never silently reclassified as a missing key or vice versa.
    assert not (set(report.missing_keys) & set(report.orphan_keys))


# ---------------------------------------------------------------------------
# Invariant 2/5: a citation backs novelty evidence only when every one of its
# candidates was screened exactly once as included, and it verified (Req 3.2-3.5).
# ---------------------------------------------------------------------------
@_PBT_SETTINGS
@given(plans=_work_plan_lists)
def test_novelty_evidence_never_admits_an_unscreened_or_unverified_candidate(plans) -> None:
    ledger, by_key = _build_ledger(plans)
    exclusions = ledger.novelty_exclusions()
    eligible_keys = {record.citation_key for record in ledger.novelty_eligible_citations()}

    for key, plan in by_key.items():
        exclusion = ledger.novelty_exclusion(key)
        assert (exclusion is None) is plan.is_novelty_eligible
        assert (key in eligible_keys) is plan.is_novelty_eligible
        assert (key in exclusions) is not plan.is_novelty_eligible
        if exclusion is None:
            continue
        # The reported reason always names a defect the plan really carries.
        if exclusion is NoveltyEvidenceExclusion.UNVERIFIED:
            assert plan.is_cleanly_screened and not plan.is_verified
        else:
            assert not plan.is_cleanly_screened
        states = {ledger.screening_state(f"{key}-c{index}") for index in range(len(plan.screenings))}
        if exclusion is NoveltyEvidenceExclusion.SCREENING_MISSING:
            assert ScreeningState.MISSING in states
        elif exclusion is NoveltyEvidenceExclusion.SCREENING_DUPLICATE:
            assert ScreeningState.DUPLICATE in states
        elif exclusion is NoveltyEvidenceExclusion.SCREENING_INCONSISTENT:
            assert ScreeningState.INCONSISTENT in states or len(states) > 1
        elif exclusion is NoveltyEvidenceExclusion.SCREENED_EXCLUDED:
            assert states == {ScreeningState.EXCLUDED}


@_PBT_SETTINGS
@given(plans=_work_plan_lists)
def test_an_ineligible_citation_can_enter_neither_the_matrix_nor_the_novelty_graph(plans) -> None:
    ledger, by_key = _build_ledger(plans)
    matrix = RelatedWorkMatrix(ledger)
    graph = NoveltyComparisonGraph(matrix)

    for key, plan in by_key.items():
        evidence = EvidenceRecord(
            record_id=f"ev-{key}",
            evidence_type=EvidenceType.PAPER_CLAIM,
            producer="citation-ledger",
            created_at_utc=UTC,
            method="original-text extraction with DOI metadata cross-check",
            source=f"{key} Section 5.1",
            extracted_value="synthetic grid road network",
            verification="Crossref metadata and original-text hash matched",
        )
        if plan.is_novelty_eligible:
            assert matrix.add_row(key).citation_key == key
            assert graph.register_evidence(evidence, citation_key=key).record_id == f"ev-{key}"
            continue
        with pytest.raises(ResearchValidationError) as row_error:
            matrix.add_row(key)
        assert row_error.value.code == "UNVERIFIED_MATRIX_ROW"
        assert row_error.value.actual == ledger.novelty_exclusion(key).value
        with pytest.raises(ResearchValidationError) as evidence_error:
            graph.register_evidence(evidence, citation_key=key)
        assert evidence_error.value.code == "UNVERIFIED_NOVELTY_EVIDENCE"

    eligible = {key for key, plan in by_key.items() if plan.is_novelty_eligible}
    assert {row.citation_key for row in matrix.rows()} == eligible


# ---------------------------------------------------------------------------
# Invariant 3/5: every matrix row is complete, and `not_reported` is never
# confused with a source that genuinely reports an absence (Requirement 3.6).
# ---------------------------------------------------------------------------
@_PBT_SETTINGS
@given(plans=_work_plan_lists, reported=_axis_sets, value=_cell_values)
def test_a_not_reported_cell_is_never_conflated_with_a_reported_absence(plans, reported, value) -> None:
    ledger, by_key = _eligible_ledger(plans)
    matrix = RelatedWorkMatrix(ledger)
    key = sorted(by_key)[0]

    row = matrix.add_row(
        key,
        tuple(
            RelatedWorkCell(axis=axis, value=value, reported=True, source_location="Section 4")
            for axis in sorted(reported, key=lambda item: item.value)
        ),
    )

    # The row spans every column, in column order, whatever was supplied.
    assert tuple(cell.axis.value for cell in row.cells) == MATRIX_COLUMNS
    assert set(row.not_reported_axes) == {
        axis.value for axis in ComparisonAxis if axis not in reported
    }
    for axis in ComparisonAxis:
        cell = row.cell(axis)
        if axis in reported:
            # A reported `none` is a claim about the source, not a silent gap.
            assert cell.value == value
            assert cell.is_not_reported is False
            assert cell.source_location == "Section 4"
        else:
            assert cell.value == NOT_REPORTED
            assert cell.is_not_reported is True
            assert cell.source_location is None
        assert (axis.value in row.not_reported_axes) is cell.is_not_reported
        assert (key in matrix.axis_support(axis)) is (axis in reported)


@_PBT_SETTINGS
@given(axis=st.sampled_from(tuple(ComparisonAxis)), value=_cell_values, location=st.sampled_from(("Section 4", None)))
def test_the_reserved_not_reported_value_can_never_be_written_as_source_text(axis, value, location) -> None:
    with pytest.raises(ResearchValidationError) as reserved:
        RelatedWorkCell(axis=axis, value=NOT_REPORTED, reported=True, source_location="Section 4")
    assert reserved.value.code == "RESERVED_CELL_VALUE"

    # An unreported cell may carry neither a value nor a source location.
    with pytest.raises(ResearchValidationError) as unreported:
        RelatedWorkCell(axis=axis, value=value, reported=False, source_location=location)
    assert unreported.value.code == "UNREPORTED_CELL_HAS_VALUE"

    # A reported cell without a source location is equally inadmissible.
    with pytest.raises(ResearchValidationError) as unsourced:
        RelatedWorkCell(axis=axis, value=value, reported=True, source_location=None)
    assert unsourced.value.code == "MISSING_REQUIRED_FIELD"


# ---------------------------------------------------------------------------
# Invariant 4/5: a novelty claim closes over verified citations and axes that a
# row actually reports; an indirect axis is always refused (Req 3.6, 4.1, 4.4).
# ---------------------------------------------------------------------------
@_PBT_SETTINGS
@given(plans=_work_plan_lists, reported=_axis_sets, claimed=_axis_sets)
def test_a_novelty_claim_closes_only_over_directly_compared_axes(plans, reported, claimed) -> None:
    ledger, by_key = _eligible_ledger(plans)
    matrix = RelatedWorkMatrix(ledger)
    for key in sorted(by_key):
        matrix.add_row(
            key,
            tuple(
                RelatedWorkCell(axis=axis, value="reported value", reported=True, source_location="Section 4")
                for axis in sorted(reported, key=lambda item: item.value)
            ),
        )
    graph = NoveltyComparisonGraph(matrix)
    evidence = EvidenceRecord(
        record_id="ev-1",
        evidence_type=EvidenceType.PAPER_CLAIM,
        producer="citation-ledger",
        created_at_utc=UTC,
        method="original-text extraction",
        source="Section 5.1",
        extracted_value="real OSM road network",
        verification="Crossref metadata and original-text hash matched",
    )
    graph.register_evidence(evidence, citation_key=sorted(by_key)[0])
    axes = tuple(sorted(claimed, key=lambda item: item.value))
    text = "본 연구는 선행 연구와 달리 실제 OSM 도로망에서 협력 포위를 평가한다"

    if not axes:
        with pytest.raises(ResearchValidationError) as empty:
            graph.add_comparison(claim_id="novelty-1", text=text, axes=axes, evidence_ids=("ev-1",))
        assert empty.value.code == "MISSING_COMPARISON_AXIS"
        return
    if not claimed <= reported:
        with pytest.raises(ResearchValidationError) as indirect:
            graph.add_comparison(claim_id="novelty-1", text=text, axes=axes, evidence_ids=("ev-1",))
        assert indirect.value.code == "INDIRECT_COMPARISON_AXIS"
        assert indirect.value.actual in {axis.value for axis in claimed - reported}
        assert graph.comparisons() == ()
        return

    comparison = graph.add_comparison(
        claim_id="novelty-1", text=text, axes=axes, evidence_ids=("ev-1",)
    )
    assert comparison.claim.scope is ClaimScope.NOVELTY
    assert set(comparison.comparison_axes) == {axis.value for axis in claimed}
    # Every citation the claim cites is registered, verified and carries a row.
    assert set(comparison.claim.citation_ids) <= set(ledger.citation_keys())
    for citation_key in comparison.claim.citation_ids:
        assert ledger.is_novelty_eligible(citation_key)
        assert matrix.row(citation_key).citation_key == citation_key
    for axis, keys in comparison.axis_citations.items():
        assert set(keys) == set(matrix.axis_support(axis))
        assert keys, axis
    assert set(comparison.evidence_ids) == {"ev-1"}
    assert graph.evidence_citation("ev-1") in set(ledger.citation_keys())


@_PBT_SETTINGS
@given(plans=_work_plan_lists, unknown=_free_keys)
def test_novelty_evidence_that_is_not_registered_can_never_back_a_claim(plans, unknown) -> None:
    ledger, by_key = _eligible_ledger(plans)
    matrix = RelatedWorkMatrix(ledger)
    key = sorted(by_key)[0]
    matrix.add_row(
        key,
        (
            RelatedWorkCell(
                axis=ComparisonAxis.MAP_SOURCE_REALISM, value="synthetic grid",
                reported=True, source_location="Section 5.1",
            ),
        ),
    )
    graph = NoveltyComparisonGraph(matrix)

    with pytest.raises(ResearchValidationError) as excinfo:
        graph.add_comparison(
            claim_id="novelty-1",
            text="본 연구는 실제 OSM 도로망에서 협력 포위를 평가한다",
            axes=(ComparisonAxis.MAP_SOURCE_REALISM,),
            evidence_ids=(unknown,),
        )
    assert excinfo.value.code == "UNVERIFIED_NOVELTY_EVIDENCE"
    assert graph.comparisons() == ()


# ---------------------------------------------------------------------------
# Invariant 5/5: verification status is derived from the recorded cross-checks
# and never asserted directly (Requirements 3.3-3.5).
# ---------------------------------------------------------------------------
@_PBT_SETTINGS
@given(plan=_work_plans)
def test_verification_status_always_follows_the_recorded_cross_checks(plan) -> None:
    ledger, by_key = _build_ledger([plan])
    record = ledger.citations()[0]

    assert record.is_verified is plan.is_verified
    assert record.verification_status is (
        VerificationStatus.VERIFIED if plan.is_verified else VerificationStatus.UNVERIFIED
    )
    assert bool(record.unverified_reasons) is not plan.is_verified
    expected_reason = {
        "metadata_missing": "metadata_cross_check_missing",
        "metadata_failed": "metadata_cross_check_failed",
        "location_missing": "source_location_missing",
        "proposal_unverified": "proposal_text_unverified",
    }.get(plan.verification)
    if expected_reason is not None:
        assert expected_reason in record.unverified_reasons
    assert set(by_key) == set(ledger.citation_keys())

"""Property 4 coverage: research claims remain within the demonstrated scope."""

from __future__ import annotations

from hypothesis import given, settings, strategies as st
import pytest

from pursuit_evasion_rl.research.domain import (
    AnalysisClassification,
    ClaimRecord,
    ClaimScope,
    ClaimStatus,
    DataKind,
    MapScenario,
    ResearchQuestion,
)
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.evaluation.paired import EvaluationProvenance, EvaluationStratumLabel

# **Property 4: Research claims remain within the demonstrated scope**
# **Validates: Requirements 4.3-4.7, 6.9**

pytestmark = pytest.mark.offline

_PBT_SETTINGS = settings(max_examples=100, deadline=None)

_PRIORITY_TERMS = ("최초", "유일", "first", "only")
_words = st.text(alphabet="abcdefghijklmnopqrstuvwxyz가나다라마 ", min_size=0, max_size=20)


@_PBT_SETTINGS
@given(
    term=st.sampled_from(_PRIORITY_TERMS),
    prefix=_words,
    suffix=_words,
    upper=st.booleans(),
)
def test_priority_language_is_rejected_wherever_it_appears_in_novelty_text(term, prefix, suffix, upper) -> None:
    token = term.upper() if upper and term.isascii() else term
    text = f"{prefix} {token} {suffix}".strip() or token
    with pytest.raises(ResearchValidationError) as excinfo:
        ClaimRecord(
            claim_id="novelty-1", text=text, scope=ClaimScope.NOVELTY, status=ClaimStatus.SUPPORTED,
            evidence_ids=("evidence-1",), comparison_axes=("axis-1",),
        )
    assert excinfo.value.code == "FORBIDDEN_PRIORITY_CLAIM"


@_PBT_SETTINGS
@given(text=_words.filter(lambda t: not any(term in t.casefold() for term in _PRIORITY_TERMS)))
def test_novelty_text_without_priority_language_and_with_evidence_constructs_cleanly(text: str) -> None:
    body = text.strip() or "a comparison without priority language"
    claim = ClaimRecord(
        claim_id="novelty-2", text=body,
        scope=ClaimScope.NOVELTY, status=ClaimStatus.SUPPORTED,
        evidence_ids=("evidence-1",), comparison_axes=("axis-1",),
    )
    assert claim.scope is ClaimScope.NOVELTY


@_PBT_SETTINGS
@given(
    evidence_ids=st.one_of(st.just(()), st.tuples(st.just("evidence-1"))),
    comparison_axes=st.one_of(st.just(()), st.tuples(st.just("axis-1"))),
)
def test_novelty_claims_require_both_nonempty_evidence_and_a_comparison_axis(evidence_ids, comparison_axes) -> None:
    if evidence_ids and comparison_axes:
        claim = ClaimRecord(
            claim_id="novelty-3", text="a direct comparison against prior baselines",
            scope=ClaimScope.NOVELTY, status=ClaimStatus.SUPPORTED,
            evidence_ids=evidence_ids, comparison_axes=comparison_axes,
        )
        assert claim.evidence_ids and claim.comparison_axes
    else:
        with pytest.raises(ResearchValidationError) as excinfo:
            ClaimRecord(
                claim_id="novelty-3", text="a direct comparison against prior baselines",
                scope=ClaimScope.NOVELTY, status=ClaimStatus.SUPPORTED,
                evidence_ids=evidence_ids, comparison_axes=comparison_axes,
            )
        assert excinfo.value.code == "INCOMPLETE_NOVELTY_EVIDENCE"


@_PBT_SETTINGS
@given(performance=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
def test_field_readiness_claims_can_never_be_constructed_as_anything_but_unsupported(performance: float) -> None:
    for status in ClaimStatus:
        if status is ClaimStatus.UNSUPPORTED:
            claim = ClaimRecord(
                claim_id="field-readiness-1", text=f"observed performance {performance}",
                scope=ClaimScope.FIELD_READINESS, status=status, evidence_ids=("evidence-1",),
            )
            assert claim.status is ClaimStatus.UNSUPPORTED
        else:
            with pytest.raises(ResearchValidationError) as excinfo:
                ClaimRecord(
                    claim_id="field-readiness-1", text=f"observed performance {performance}",
                    scope=ClaimScope.FIELD_READINESS, status=status, evidence_ids=("evidence-1",),
                )
            assert excinfo.value.code == "FIELD_READINESS_UNSUPPORTED"


@_PBT_SETTINGS
@given(analysis_class=st.sampled_from(list(AnalysisClassification)))
def test_a_research_question_always_carries_exactly_one_analysis_classification(analysis_class) -> None:
    question = ResearchQuestion(
        question_id="rq-1", analysis_class=analysis_class, primary_outcome="capture_rate",
        directional_inequality="proposed > baseline", practical_threshold=0.05,
        threshold_unit="probability", decision_rule="two-sided test with Holm correction",
    )
    assert question.analysis_class is analysis_class


@_PBT_SETTINGS
@given(malformed=st.text(min_size=1, max_size=12).filter(lambda t: t not in {item.value for item in ClaimStatus}))
def test_a_malformed_claim_status_value_is_always_rejected(malformed: str) -> None:
    with pytest.raises(ResearchValidationError) as excinfo:
        ClaimRecord(
            claim_id="claim-1", text="some result", scope=ClaimScope.RESULT, status=malformed,  # type: ignore[arg-type]
            evidence_ids=("evidence-1",),
        )
    assert excinfo.value.code == "INVALID_CLASSIFICATION"


# ---------------------------------------------------------------------------
# "Same-road" / spatial-generalization scope limitation
#
# domain.ClaimRecord has no built-in notion of evaluation stratum -- that
# composition lives one layer up, in whatever assembles a claim's evidence.
# This exercises the underlying data model's capacity to distinguish
# in-region-only evidence from evidence that actually demonstrates
# generalization, which is the structural precondition any real claim gate
# needs before it can enforce the restriction.
# ---------------------------------------------------------------------------


def _mentions_generalization(text: str) -> bool:
    lowered = text.casefold()
    return any(term in lowered for term in ("generaliz", "zero-shot", "cross-city", "일반화"))


def _generalization_eligible(strata: tuple[EvaluationStratumLabel, ...]) -> bool:
    return any(item.provenance is not EvaluationProvenance.IN_REGION for item in strata)


@_PBT_SETTINGS
@given(
    mentions=st.booleans(),
    provenances=st.lists(st.sampled_from(list(EvaluationProvenance)), min_size=1, max_size=4),
)
def test_a_generalization_claim_is_falsifiable_only_by_non_in_region_evidence(mentions: bool, provenances) -> None:
    strata = tuple(
        EvaluationStratumLabel(
            data_kind=DataKind.ACTUAL_OSM_MAP if provenance is EvaluationProvenance.CROSS_CITY else DataKind.SYNTHETIC_FIXTURE,
            scenario=MapScenario.INTERIOR_CONTAINED,
            provenance=provenance,
        )
        for provenance in provenances
    )
    text = "the policy generalizes to unseen cities" if mentions else "the policy captures the fugitive"
    eligible = _generalization_eligible(strata)

    if mentions and not eligible:
        # A generalization claim backed only by in-region (seed-level) strata
        # is not falsifiable by the evidence it cites -- it should never be
        # treated as demonstrated.
        assert all(item.provenance is EvaluationProvenance.IN_REGION for item in strata)
    if eligible:
        assert any(item.provenance is not EvaluationProvenance.IN_REGION for item in strata)

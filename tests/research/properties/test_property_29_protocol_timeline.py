"""Property 29 coverage: protocol and analysis classification are immutable and timeline-consistent."""

from __future__ import annotations

from pathlib import Path

from hypothesis import given, settings, strategies as st
import pytest

from pursuit_evasion_rl.research.budget import ConditionResourceEstimate, ResourceCeiling, decide_sample_size
from pursuit_evasion_rl.research.domain import AnalysisClassification, ProtocolState
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.maps.splits import DataHandle, HandleKind, SplitScope
from pursuit_evasion_rl.research.metrics.behavior import MetricDirection
from pursuit_evasion_rl.research.protocol import (
    ConditionRef,
    ExclusionSpec,
    HypothesisSpec,
    MetricDeclaration,
    MetricRole,
    PlannedAnalysis,
    ProtocolSpecifications,
    ProtocolStore,
    ResourceSpec,
    SampleSpec,
    SearchSpec,
    SplitSpec,
    StatisticsSpec,
    StopRule,
    StopSpec,
    ToleranceSpec,
    default_research_questions,
    metric_declaration,
    practical_thresholds_for,
)
from pursuit_evasion_rl.research.statistics.paired import BootstrapPlan, CorrectionMethod, MissingDataPolicy

# **Property 29: Protocol and analysis classification are immutable and timeline-consistent**
# **Validates: Requirements 15.2-15.5**

pytestmark = pytest.mark.offline

_PBT_SETTINGS = settings(max_examples=100, deadline=None)

SIGNER = "principal-investigator"


def _metrics() -> tuple[MetricDeclaration, ...]:
    return (
        MetricDeclaration(
            metric_id="capture_rate", symbol="P_cap", unit="probability",
            direction=MetricDirection.HIGHER_IS_BETTER, role=MetricRole.PRIMARY,
            formula="captured episodes divided by planned episodes",
        ),
        metric_declaration("blocked_exit_fraction", MetricRole.PRIMARY),
    )


def _selection_handles() -> tuple[DataHandle[SplitScope], ...]:
    return (
        DataHandle(SplitScope.TRAIN, HandleKind.MAP, "daejeon-train"),
        DataHandle(SplitScope.VALIDATION, HandleKind.MAP, "daejeon-validation"),
    )


def _resource_spec() -> ResourceSpec:
    return ResourceSpec(
        ceiling=ResourceCeiling(
            resource_ceiling_id="ceiling-2026-07", measured_at_utc="2026-07-01T00:00:00Z",
            accelerator_hours=2000.0, wall_clock_hours=2000.0,
        ),
        estimates=(
            ConditionResourceEstimate(
                condition_id="default_interior", accelerator_hours_per_seed=1.0,
                wall_clock_hours_per_seed=1.0, env_steps_per_seed=1_000_000,
            ),
        ),
    )


def _questions():
    return default_research_questions()


def _analyses(questions):
    return tuple(
        PlannedAnalysis(
            analysis_id=f"A-{question.question_id}", question_id=question.question_id,
            classification=AnalysisClassification.CONFIRMATORY,
            estimand=f"paired difference in {question.primary_outcome}",
            metric_id=question.primary_outcome, comparison=question.directional_inequality,
        )
        for question in questions
    )


def _specifications(questions=None) -> ProtocolSpecifications:
    questions = questions or _questions()
    metrics = _metrics()
    resource = _resource_spec()
    return ProtocolSpecifications(
        hypotheses=tuple(
            HypothesisSpec(
                hypothesis_id=f"H-{question.question_id}", question_id=question.question_id,
                statement=question.directional_inequality,
                null_statement=f"no difference in {question.primary_outcome}",
            )
            for question in questions
        ),
        conditions=(ConditionRef(condition_id="default_interior", condition_hash="9" * 64, axis="baseline", arm="default"),),
        split=SplitSpec(
            metric_crs="EPSG:5186", buffer_m=50.0,
            polygon_hashes={"train": "1" * 64, "validation": "2" * 64, "test": "3" * 64},
            network_hashes={"train": "4" * 64, "validation": "5" * 64, "test": "6" * 64},
            cross_city_city_ids=("busan", "seoul"), split_protocol_hash="7" * 64,
            split_validation_report_hash="8" * 64,
            held_out_evaluation_rule="apply the validation-selected policy unchanged to the test split, once",
        ),
        sample=SampleSpec(plan=decide_sample_size(resource.ceiling, resource.estimates), selection_handles=_selection_handles()),
        resource=resource,
        metrics=metrics,
        statistics=StatisticsSpec(
            bootstrap=BootstrapPlan(), correction=CorrectionMethod.HOLM,
            primary_test="exact_mcnemar", multiplicity_family="the four pre-registered primary outcomes",
        ),
        thresholds=practical_thresholds_for(questions, metrics),
        tolerance=ToleranceSpec(
            rtol=1e-5, atol=1e-7, applies_to=("resume_equivalence",),
            rationale="float32 accumulation differs across resume boundaries on the same device",
        ),
        exclusion=ExclusionSpec(
            outcome_mapping={
                "valid": "use the observed episode outcome",
                "missing": "score the pre-registered worst outcome for both policies",
                "failed": "score the pre-registered worst outcome for both policies",
                "interrupted": "score the pre-registered worst outcome for both policies",
            },
            excludable_conditions=("simulator process crash reproduced twice",),
            primary_policy=MissingDataPolicy.PRE_REGISTERED_WORST_CASE,
            sensitivity_policy=MissingDataPolicy.COMPLETE_CASE,
            planned_case_accounting_rule="every planned episode case is reported as valid, missing, failed or interrupted",
        ),
        stop=StopSpec(rules=(
            StopRule(
                rule_id="validation-plateau", criterion="validation capture rate does not improve for 20 evaluations",
                action="halt the condition and record the reason", evaluated_on=SplitScope.VALIDATION,
            ),
        )),
        search=SearchSpec(
            search_date_utc="2026-07-01T00:00:00Z",
            query="(pursuit evasion) AND (multi-agent reinforcement learning) AND road network",
            sources=("Scopus", "IEEE Xplore", "arXiv"), date_range_start="2015-01-01", date_range_end="2026-06-30",
            languages=("en", "ko"), inclusion_criteria=("multi-pursuer pursuit on a graph or road network",),
            exclusion_criteria=("continuous open-plane pursuit without a road network",),
        ),
        analyses=_analyses(questions),
    )


def _sealed_store(tmp_path: Path) -> tuple[ProtocolStore, str]:
    store = ProtocolStore(tmp_path / "protocols")
    record = store.create(questions=_questions(), specifications=_specifications())
    store.seal(record.protocol_id, signer=SIGNER)
    return store, record.protocol_id


@_PBT_SETTINGS
@given(seed=st.integers(min_value=0, max_value=2**16))
def test_a_sealed_protocol_is_immutable_and_any_edit_must_fork(tmp_path_factory, seed: int) -> None:
    tmp_path = tmp_path_factory.mktemp(f"protocol-{seed}")
    store, protocol_id = _sealed_store(tmp_path)
    sealed = store.read(protocol_id)
    assert sealed.protocol.state is ProtocolState.SEALED

    with pytest.raises(ResearchValidationError) as excinfo:
        store.update_draft(protocol_id, specifications=_specifications())
    assert excinfo.value.code == "SEALED_PROTOCOL_MUTATION_BLOCKED"
    with pytest.raises(ResearchValidationError) as excinfo_seal:
        store.seal(protocol_id, signer=SIGNER)
    assert excinfo_seal.value.code == "SEALED_PROTOCOL_MUTATION_BLOCKED"

    branch = store.fork(protocol_id)
    assert branch.protocol.state is ProtocolState.DRAFT
    assert branch.protocol.parent_hash == sealed.protocol_hash
    # The parent's own content is untouched by the fork.
    assert store.read(protocol_id).protocol_hash == sealed.protocol_hash


@_PBT_SETTINGS
@given(question_index=st.integers(min_value=0, max_value=3))
def test_an_unchanged_preregistered_analysis_is_always_confirmatory(tmp_path_factory, question_index: int) -> None:
    tmp_path = tmp_path_factory.mktemp(f"analysis-conf-{question_index}")
    store, protocol_id = _sealed_store(tmp_path)
    planned = _analyses(_questions())[question_index]
    event = store.register_analysis(protocol_id, planned)
    assert event.classification is AnalysisClassification.CONFIRMATORY
    assert event.is_confirmatory
    assert event.protocol_hash_at_declaration == event.protocol_hash_at_registration


@_PBT_SETTINGS
@given(analysis_id=st.text(alphabet="abcdefghijklmnop", min_size=3, max_size=12))
def test_an_unplanned_analysis_is_always_demoted_to_exploratory(tmp_path_factory, analysis_id: str) -> None:
    tmp_path = tmp_path_factory.mktemp(f"analysis-unplanned-{analysis_id}")
    store, protocol_id = _sealed_store(tmp_path)
    questions = _questions()
    unplanned = PlannedAnalysis(
        analysis_id=f"unplanned-{analysis_id}", question_id=questions[0].question_id,
        classification=AnalysisClassification.CONFIRMATORY,
        estimand="an analysis never declared in the protocol",
        metric_id=questions[0].primary_outcome, comparison=questions[0].directional_inequality,
    )
    event = store.register_analysis(protocol_id, unplanned)
    assert event.classification is AnalysisClassification.EXPLORATORY
    assert not event.is_confirmatory


@_PBT_SETTINGS
@given(question_index=st.integers(min_value=0, max_value=3), new_comparison=st.text(alphabet="abcdefghijklmnop ", min_size=1, max_size=20))
def test_a_changed_analysis_is_demoted_even_if_it_reuses_the_planned_id(tmp_path_factory, question_index: int, new_comparison: str) -> None:
    tmp_path = tmp_path_factory.mktemp(f"analysis-changed-{question_index}-{len(new_comparison)}")
    store, protocol_id = _sealed_store(tmp_path)
    planned = _analyses(_questions())[question_index]
    if new_comparison.strip() == planned.comparison:
        new_comparison = new_comparison + "-changed"
    changed = PlannedAnalysis(
        analysis_id=planned.analysis_id, question_id=planned.question_id,
        classification=planned.classification, estimand=planned.estimand,
        metric_id=planned.metric_id, comparison=new_comparison.strip() or "changed-comparison",
    )
    event = store.register_analysis(protocol_id, changed)
    assert event.classification is AnalysisClassification.EXPLORATORY
    assert event.declared_analysis_hash != event.registered_analysis_hash


@_PBT_SETTINGS
@given(question_index=st.integers(min_value=0, max_value=3))
def test_an_identical_analysis_registered_against_a_forked_branch_is_exploratory(tmp_path_factory, question_index: int) -> None:
    tmp_path = tmp_path_factory.mktemp(f"analysis-branch-{question_index}")
    store, protocol_id = _sealed_store(tmp_path)
    planned = _analyses(_questions())[question_index]

    original_event = store.register_analysis(protocol_id, planned)
    assert original_event.is_confirmatory

    branch = store.fork(protocol_id)
    sealed_branch = store.seal(branch.protocol_id, signer=SIGNER)
    branch_event = store.register_analysis(sealed_branch.protocol_id, planned)

    assert branch_event.classification is AnalysisClassification.EXPLORATORY
    assert branch_event.protocol_hash_at_declaration == store.read(protocol_id).protocol_hash
    assert branch_event.protocol_hash_at_registration == sealed_branch.protocol_hash


@_PBT_SETTINGS
@given(question_index=st.integers(min_value=0, max_value=3))
def test_an_analysis_can_only_be_registered_once_against_the_same_protocol(tmp_path_factory, question_index: int) -> None:
    tmp_path = tmp_path_factory.mktemp(f"analysis-dup-{question_index}")
    store, protocol_id = _sealed_store(tmp_path)
    planned = _analyses(_questions())[question_index]
    store.register_analysis(protocol_id, planned)
    with pytest.raises(ResearchValidationError) as excinfo:
        store.register_analysis(protocol_id, planned)
    assert excinfo.value.code == "DUPLICATE_ANALYSIS_REGISTRATION"


@_PBT_SETTINGS
@given(classification=st.sampled_from(list(AnalysisClassification)))
def test_a_planned_analysis_always_carries_exactly_one_classification(classification: AnalysisClassification) -> None:
    analysis = PlannedAnalysis(
        analysis_id="A-x", question_id="Q-x", classification=classification,
        estimand="an estimand", metric_id="capture_rate", comparison="proposed > baseline",
    )
    assert analysis.classification is classification

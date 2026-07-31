"""Task 8.1 regressions for Research_Protocol freeze, fork and the leakage guard."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from pursuit_evasion_rl.research.budget import (
    ConditionResourceEstimate,
    ResourceCeiling,
    decide_sample_size,
)
from pursuit_evasion_rl.research.domain import (
    AnalysisClassification,
    ExecutionStatus,
    ProtocolState,
    ResearchQuestion,
)
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.maps.splits import DataHandle, HandleKind, SplitScope
from pursuit_evasion_rl.research.metrics.behavior import MetricDirection
from pursuit_evasion_rl.research.protocol import (
    REQUIRED_QUESTION_IDS,
    REQUIRED_SPECIFICATION_SLOTS,
    ConditionRef,
    ConfirmatoryGateError,
    ContaminationRecord,
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
    SelectionContext,
    SelectionLeakageError,
    SelectionPurpose,
    SplitSpec,
    StatisticsSpec,
    StopRule,
    StopSpec,
    ToleranceSpec,
    contaminated_run_lineage,
    default_research_questions,
    guard_selection_context,
    inspect_confirmatory_eligibility,
    inspect_selection_context,
    metric_declaration,
    practical_thresholds_for,
    propagate_contamination,
    require_confirmatory_eligibility,
)
from pursuit_evasion_rl.research.runs.manifest import RunManifestStore, RunProvenance
from pursuit_evasion_rl.research.statistics.paired import (
    BootstrapPlan,
    CorrectionMethod,
    MissingDataPolicy,
)

pytestmark = pytest.mark.offline

SIGNER = "principal-investigator"


def _metrics() -> tuple[MetricDeclaration, ...]:
    return (
        MetricDeclaration(
            metric_id="capture_rate",
            symbol="P_cap",
            unit="probability",
            direction=MetricDirection.HIGHER_IS_BETTER,
            role=MetricRole.PRIMARY,
            formula="captured episodes divided by planned episodes",
        ),
        metric_declaration("blocked_exit_fraction", MetricRole.PRIMARY),
        metric_declaration("u_turn_rate", MetricRole.SECONDARY),
    )


def _selection_handles() -> tuple[DataHandle[SplitScope], ...]:
    return (
        DataHandle(SplitScope.TRAIN, HandleKind.MAP, "daejeon-train"),
        DataHandle(SplitScope.VALIDATION, HandleKind.MAP, "daejeon-validation"),
    )


def _split_spec() -> SplitSpec:
    return SplitSpec(
        metric_crs="EPSG:5186",
        buffer_m=50.0,
        polygon_hashes={"train": "1" * 64, "validation": "2" * 64, "test": "3" * 64},
        network_hashes={"train": "4" * 64, "validation": "5" * 64, "test": "6" * 64},
        cross_city_city_ids=("busan", "seoul"),
        split_protocol_hash="7" * 64,
        split_validation_report_hash="8" * 64,
        held_out_evaluation_rule=(
            "apply the validation-selected policy unchanged to the test split, once"
        ),
    )


def _resource_spec() -> ResourceSpec:
    return ResourceSpec(
        ceiling=ResourceCeiling(
            resource_ceiling_id="ceiling-2026-07",
            measured_at_utc="2026-07-01T00:00:00Z",
            accelerator_hours=2000.0,
            wall_clock_hours=2000.0,
        ),
        estimates=(
            ConditionResourceEstimate(
                condition_id="default_interior",
                accelerator_hours_per_seed=1.0,
                wall_clock_hours_per_seed=1.0,
                env_steps_per_seed=1_000_000,
            ),
        ),
    )


def _questions() -> tuple[ResearchQuestion, ...]:
    return default_research_questions()


def _analyses(questions: tuple[ResearchQuestion, ...]) -> tuple[PlannedAnalysis, ...]:
    return tuple(
        PlannedAnalysis(
            analysis_id=f"A-{question.question_id}",
            question_id=question.question_id,
            classification=AnalysisClassification.CONFIRMATORY,
            estimand=f"paired difference in {question.primary_outcome}",
            metric_id=question.primary_outcome,
            comparison=question.directional_inequality,
        )
        for question in questions
    )


def _specifications(
    questions: tuple[ResearchQuestion, ...] | None = None,
    *,
    selection_handles: tuple[DataHandle[SplitScope], ...] | None = None,
) -> ProtocolSpecifications:
    questions = questions or _questions()
    metrics = _metrics()
    resource = _resource_spec()
    return ProtocolSpecifications(
        hypotheses=tuple(
            HypothesisSpec(
                hypothesis_id=f"H-{question.question_id}",
                question_id=question.question_id,
                statement=question.directional_inequality,
                null_statement=f"no difference in {question.primary_outcome}",
            )
            for question in questions
        ),
        conditions=(
            ConditionRef(
                condition_id="default_interior",
                condition_hash="9" * 64,
                axis="baseline",
                arm="default",
            ),
        ),
        split=_split_spec(),
        sample=SampleSpec(
            plan=decide_sample_size(resource.ceiling, resource.estimates),
            selection_handles=selection_handles or _selection_handles(),
        ),
        resource=resource,
        metrics=metrics,
        statistics=StatisticsSpec(
            bootstrap=BootstrapPlan(),
            correction=CorrectionMethod.HOLM,
            primary_test="exact_mcnemar",
            multiplicity_family="the four pre-registered primary outcomes",
        ),
        thresholds=practical_thresholds_for(questions, metrics),
        tolerance=ToleranceSpec(
            rtol=1e-5,
            atol=1e-7,
            applies_to=("resume_equivalence", "interruption_equivalence"),
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
            planned_case_accounting_rule=(
                "every planned episode case is reported as valid, missing, failed or interrupted"
            ),
        ),
        stop=StopSpec(
            rules=(
                StopRule(
                    rule_id="validation-plateau",
                    criterion="validation capture rate does not improve for 20 evaluations",
                    action="halt the condition and record the reason",
                    evaluated_on=SplitScope.VALIDATION,
                ),
            )
        ),
        search=SearchSpec(
            search_date_utc="2026-07-01T00:00:00Z",
            query="(pursuit evasion) AND (multi-agent reinforcement learning) AND road network",
            sources=("Scopus", "IEEE Xplore", "arXiv"),
            date_range_start="2015-01-01",
            date_range_end="2026-06-30",
            languages=("en", "ko"),
            inclusion_criteria=("multi-pursuer pursuit on a graph or road network",),
            exclusion_criteria=("continuous open-plane pursuit without a road network",),
        ),
        analyses=_analyses(questions),
    )


def _sealed_store(tmp_path: Path) -> tuple[ProtocolStore, str]:
    store = ProtocolStore(tmp_path / "protocols")
    record = store.create(questions=_questions(), specifications=_specifications())
    store.seal(record.protocol_id, signer=SIGNER)
    return store, record.protocol_id


# ---------------------------------------------------------------------------
# Freeze
# ---------------------------------------------------------------------------


def test_seal_records_freeze_timestamp_hash_and_signer(tmp_path: Path) -> None:
    store = ProtocolStore(tmp_path / "protocols")
    draft = store.create(questions=_questions(), specifications=_specifications())
    assert draft.protocol.state is ProtocolState.DRAFT
    assert draft.protocol.frozen_at_utc is None

    sealed = store.seal(draft.protocol_id, signer=SIGNER, frozen_at_utc="2026-07-31T00:00:00Z")
    assert sealed.protocol.state is ProtocolState.SEALED
    assert sealed.protocol.frozen_at_utc == "2026-07-31T00:00:00Z"
    assert sealed.protocol.signer == SIGNER
    assert sealed.protocol_hash != draft.protocol_hash
    assert sealed.protocol.parent_hash is None

    reloaded = store.read(draft.protocol_id)
    assert reloaded.protocol_hash == sealed.protocol_hash
    assert tuple(reloaded.protocol.specifications) == tuple(sealed.protocol.specifications)
    assert set(REQUIRED_SPECIFICATION_SLOTS) <= set(reloaded.protocol.specifications)
    assert tuple(item.question_id for item in reloaded.protocol.questions) == REQUIRED_QUESTION_IDS


def test_seal_rejects_a_protocol_missing_a_research_question(tmp_path: Path) -> None:
    store = ProtocolStore(tmp_path / "protocols")
    questions = _questions()
    record = store.create(questions=questions[:3], specifications=_specifications(questions))

    with pytest.raises(ResearchValidationError) as excinfo:
        store.seal(record.protocol_id, signer=SIGNER)
    assert excinfo.value.code == "MISSING_RESEARCH_QUESTION"
    assert store.read(record.protocol_id).protocol.state is ProtocolState.DRAFT


def test_seal_rejects_a_question_without_a_practical_threshold(tmp_path: Path) -> None:
    store = ProtocolStore(tmp_path / "protocols")
    questions = _questions()
    degraded = (dataclasses.replace(questions[0], practical_threshold=None, content_hash=None),) + questions[1:]
    record = store.create(questions=degraded, specifications=_specifications(questions))

    with pytest.raises(ResearchValidationError) as excinfo:
        store.seal(record.protocol_id, signer=SIGNER)
    assert excinfo.value.code == "MISSING_PRACTICAL_THRESHOLD"


@pytest.mark.parametrize("slot", ["sample", "search", "metrics", "exclusion", "stop", "analyses"])
def test_seal_rejects_a_missing_specification_slot(tmp_path: Path, slot: str) -> None:
    store = ProtocolStore(tmp_path / "protocols")
    specifications = _specifications().as_mapping()
    specifications.pop(slot)
    record = store.create(questions=_questions(), specifications=specifications)

    with pytest.raises(ResearchValidationError) as excinfo:
        store.seal(record.protocol_id, signer=SIGNER)
    assert excinfo.value.code == "MISSING_SPECIFICATION_SLOT"
    assert excinfo.value.path == f"specifications.{slot}"


def test_seal_rejects_an_empty_search_specification(tmp_path: Path) -> None:
    store = ProtocolStore(tmp_path / "protocols")
    specifications = _specifications().as_mapping()
    specifications["search"] = {}
    record = store.create(questions=_questions(), specifications=specifications)

    with pytest.raises(ResearchValidationError) as excinfo:
        store.seal(record.protocol_id, signer=SIGNER)
    assert excinfo.value.code == "EMPTY_SPECIFICATION_SLOT"


def test_seal_rejects_a_primary_outcome_that_is_not_a_declared_metric(tmp_path: Path) -> None:
    store = ProtocolStore(tmp_path / "protocols")
    specifications = _specifications().as_mapping()
    specifications["metrics"] = [
        item for item in specifications["metrics"] if item["metric_id"] != "capture_rate"
    ]
    record = store.create(questions=_questions(), specifications=specifications)

    with pytest.raises(ResearchValidationError) as excinfo:
        store.seal(record.protocol_id, signer=SIGNER)
    assert excinfo.value.code == "UNDECLARED_PRIMARY_OUTCOME"


def test_seal_requires_a_signer(tmp_path: Path) -> None:
    store = ProtocolStore(tmp_path / "protocols")
    record = store.create(questions=_questions(), specifications=_specifications())

    with pytest.raises(ResearchValidationError) as excinfo:
        store.seal(record.protocol_id, signer="  ")
    assert excinfo.value.code == "MISSING_REQUIRED_FIELD"


# ---------------------------------------------------------------------------
# Post-freeze immutability and fork
# ---------------------------------------------------------------------------


def test_a_sealed_protocol_cannot_be_changed_in_place(tmp_path: Path) -> None:
    store, protocol_id = _sealed_store(tmp_path)
    sealed = store.read(protocol_id)

    with pytest.raises(dataclasses.FrozenInstanceError):
        sealed.protocol.signer = "someone-else"

    with pytest.raises(ResearchValidationError) as update_error:
        store.update_draft(protocol_id, questions=_questions()[:3])
    assert update_error.value.code == "SEALED_PROTOCOL_MUTATION_BLOCKED"

    with pytest.raises(ResearchValidationError) as reseal_error:
        store.seal(protocol_id, signer="someone-else")
    assert reseal_error.value.code == "SEALED_PROTOCOL_MUTATION_BLOCKED"

    assert store.read(protocol_id).protocol_hash == sealed.protocol_hash
    assert store.read(protocol_id).protocol.signer == SIGNER


def test_fork_creates_a_new_identity_and_preserves_the_sealed_original(tmp_path: Path) -> None:
    store, protocol_id = _sealed_store(tmp_path)
    sealed = store.read(protocol_id)

    child = store.fork(protocol_id)
    assert child.protocol_id != protocol_id
    assert child.protocol.parent_hash == sealed.protocol_hash
    assert child.parent_id == protocol_id
    assert child.protocol.state is ProtocolState.DRAFT
    assert child.is_branch

    assert store.read(protocol_id).protocol_hash == sealed.protocol_hash
    assert tuple(item.protocol_id for item in store.lineage(child.protocol_id)) == (
        protocol_id,
        child.protocol_id,
    )


def test_a_draft_protocol_cannot_be_forked(tmp_path: Path) -> None:
    store = ProtocolStore(tmp_path / "protocols")
    record = store.create(questions=_questions(), specifications=_specifications())

    with pytest.raises(ResearchValidationError) as excinfo:
        store.fork(record.protocol_id)
    assert excinfo.value.code == "CANNOT_FORK_DRAFT_PROTOCOL"


# ---------------------------------------------------------------------------
# Analysis registration lineage (Requirement 15.3-15.4)
# ---------------------------------------------------------------------------


def test_a_pre_registered_unchanged_analysis_is_confirmatory(tmp_path: Path) -> None:
    store, protocol_id = _sealed_store(tmp_path)
    planned = _analyses(_questions())[0]

    event = store.register_analysis(protocol_id, planned)
    assert event.classification is AnalysisClassification.CONFIRMATORY
    assert event.declared_analysis_hash == event.registered_analysis_hash
    assert event.protocol_hash_at_declaration == event.protocol_hash_at_registration
    assert store.analysis_events(protocol_id) == (event,)


def test_an_analysis_registered_after_freeze_is_demoted_to_exploratory(tmp_path: Path) -> None:
    store, protocol_id = _sealed_store(tmp_path)
    sealed = store.read(protocol_id)
    unplanned = PlannedAnalysis(
        analysis_id="A-post-hoc-subgroup",
        question_id=REQUIRED_QUESTION_IDS[0],
        classification=AnalysisClassification.CONFIRMATORY,
        estimand="capture rate difference on dead-end-heavy subregions",
        metric_id="capture_rate",
        comparison="capture_rate(dead_end_heavy) - capture_rate(other) >= 0.05",
    )

    event = store.register_analysis(protocol_id, unplanned)
    assert event.classification is AnalysisClassification.EXPLORATORY
    assert event.declared_analysis_hash is None
    assert event.registered_analysis_hash == unplanned.declaration_hash
    assert event.protocol_hash_at_registration == sealed.protocol_hash
    assert "absent" in event.reason


def test_a_changed_analysis_is_demoted_even_under_its_registered_identifier(tmp_path: Path) -> None:
    store, protocol_id = _sealed_store(tmp_path)
    planned = _analyses(_questions())[0]
    changed = dataclasses.replace(planned, estimand="unpaired difference in capture_rate")

    event = store.register_analysis(protocol_id, changed)
    assert event.classification is AnalysisClassification.EXPLORATORY
    assert event.declared_analysis_hash is not None
    assert event.declared_analysis_hash != event.registered_analysis_hash
    assert "differs" in event.reason


def test_an_identical_analysis_on_a_forked_protocol_is_exploratory(tmp_path: Path) -> None:
    store, protocol_id = _sealed_store(tmp_path)
    original = store.read(protocol_id)
    child = store.fork(protocol_id)
    store.seal(child.protocol_id, signer=SIGNER)
    planned = _analyses(_questions())[0]

    original_event = store.register_analysis(protocol_id, planned)
    branch_event = store.register_analysis(child.protocol_id, planned)

    assert original_event.classification is AnalysisClassification.CONFIRMATORY
    assert branch_event.classification is AnalysisClassification.EXPLORATORY
    assert branch_event.registered_analysis_hash == original_event.registered_analysis_hash
    assert branch_event.protocol_hash_at_declaration == original.protocol_hash
    assert branch_event.protocol_hash_at_registration != original.protocol_hash
    assert "branch" in branch_event.reason


def test_an_analysis_is_registered_against_a_protocol_exactly_once(tmp_path: Path) -> None:
    store, protocol_id = _sealed_store(tmp_path)
    planned = _analyses(_questions())[0]
    store.register_analysis(protocol_id, planned)

    with pytest.raises(ResearchValidationError) as excinfo:
        store.register_analysis(protocol_id, planned)
    assert excinfo.value.code == "DUPLICATE_ANALYSIS_REGISTRATION"


def test_an_analysis_cannot_be_registered_against_a_draft_protocol(tmp_path: Path) -> None:
    store = ProtocolStore(tmp_path / "protocols")
    record = store.create(questions=_questions(), specifications=_specifications())

    with pytest.raises(ResearchValidationError) as excinfo:
        store.register_analysis(record.protocol_id, _analyses(_questions())[0])
    assert excinfo.value.code == "ANALYSIS_REGISTRATION_REQUIRES_SEALED_PROTOCOL"


# ---------------------------------------------------------------------------
# Leakage guard (Requirement 15.5-15.6)
# ---------------------------------------------------------------------------


def test_the_protocol_sample_specification_cannot_name_a_held_out_handle() -> None:
    with pytest.raises(ResearchValidationError) as excinfo:
        _specifications(
            selection_handles=_selection_handles()
            + (DataHandle(SplitScope.TEST, HandleKind.MAP, "daejeon-test"),)
        )
    assert excinfo.value.code == "PROTOCOL_SELECTION_LEAKAGE"


def test_a_stop_rule_cannot_be_evaluated_on_held_out_data() -> None:
    with pytest.raises(ResearchValidationError) as excinfo:
        StopRule(
            rule_id="test-plateau",
            criterion="test capture rate stops improving",
            action="halt",
            evaluated_on=SplitScope.TEST,
        )
    assert excinfo.value.code == "STOP_RULE_LEAKAGE"


def test_a_clean_selection_context_yields_a_tuning_view_only() -> None:
    context = SelectionContext(
        context_id="ctx-clean",
        purpose=SelectionPurpose.CHECKPOINT_SELECTION,
        handles=_selection_handles(),
    )
    assert inspect_selection_context(context) is None

    view = guard_selection_context(context)
    assert {handle.scope for handle in view.train + view.validation} == {
        SplitScope.TRAIN,
        SplitScope.VALIDATION,
    }


@pytest.mark.parametrize(
    ("scope", "purpose"),
    [
        (SplitScope.TEST, SelectionPurpose.CHECKPOINT_SELECTION),
        (SplitScope.CROSS_CITY, SelectionPurpose.REWARD_DESIGN),
    ],
)
def test_the_guard_refuses_a_selection_context_that_reaches_held_out_data(
    scope: SplitScope, purpose: SelectionPurpose
) -> None:
    context = SelectionContext(
        context_id="ctx-leaky",
        purpose=purpose,
        handles=_selection_handles() + (DataHandle(scope, HandleKind.EPISODE, f"{scope.value}-ep-1"),),
    )

    with pytest.raises(SelectionLeakageError) as excinfo:
        guard_selection_context(context)
    contamination = excinfo.value.contamination
    assert contamination.leaked_scopes == (scope,)
    assert contamination.purpose is purpose
    assert excinfo.value.code == "SELECTION_PATH_LEAKAGE"


# ---------------------------------------------------------------------------
# Confirmatory gate (완료 검증)
# ---------------------------------------------------------------------------


def test_the_confirmatory_gate_accepts_a_sealed_protocol_with_a_clean_selection_path(
    tmp_path: Path,
) -> None:
    store, protocol_id = _sealed_store(tmp_path)
    event = store.register_analysis(protocol_id, _analyses(_questions())[0])
    clean = SelectionContext(
        context_id="ctx-clean",
        purpose=SelectionPurpose.EARLY_STOPPING,
        handles=_selection_handles(),
    )

    report = inspect_confirmatory_eligibility(
        store.read(protocol_id), analysis_event=event, selection_contexts=(clean,)
    )
    assert report.eligible
    assert report.errors == ()
    assert report.analysis_id == event.analysis_id


def test_the_confirmatory_gate_rejects_an_unresolved_test_data_dependency(tmp_path: Path) -> None:
    store, protocol_id = _sealed_store(tmp_path)
    leaky = SelectionContext(
        context_id="ctx-test-dependency",
        purpose=SelectionPurpose.CHECKPOINT_SELECTION,
        handles=_selection_handles()
        + (DataHandle(SplitScope.TEST, HandleKind.METRIC, "held-out-capture-rate"),),
    )

    report = inspect_confirmatory_eligibility(store.read(protocol_id), selection_contexts=(leaky,))
    assert not report.eligible
    assert "TEST_DATA_DEPENDENCY" in report.error_codes
    assert "held-out-capture-rate" in report.errors[0].actual


def test_the_confirmatory_gate_rejects_target_city_adaptation(tmp_path: Path) -> None:
    store, protocol_id = _sealed_store(tmp_path)
    adapting = SelectionContext(
        context_id="ctx-target-city",
        purpose=SelectionPurpose.TUNING,
        handles=_selection_handles()
        + (DataHandle(SplitScope.CROSS_CITY, HandleKind.MAP, "busan-zero-shot"),),
    )

    report = inspect_confirmatory_eligibility(store.read(protocol_id), selection_contexts=(adapting,))
    assert not report.eligible
    assert "TARGET_CITY_ADAPTATION" in report.error_codes
    assert "TEST_DATA_DEPENDENCY" not in report.error_codes


def test_the_confirmatory_gate_rejects_an_exploratory_analysis(tmp_path: Path) -> None:
    store, protocol_id = _sealed_store(tmp_path)
    child = store.fork(protocol_id)
    store.seal(child.protocol_id, signer=SIGNER)
    event = store.register_analysis(child.protocol_id, _analyses(_questions())[0])

    report = inspect_confirmatory_eligibility(store.read(child.protocol_id), analysis_event=event)
    assert not report.eligible
    assert "ANALYSIS_NOT_CONFIRMATORY" in report.error_codes


def test_the_confirmatory_gate_rejects_an_unsealed_protocol(tmp_path: Path) -> None:
    store = ProtocolStore(tmp_path / "protocols")
    record = store.create(questions=_questions(), specifications=_specifications())

    report = inspect_confirmatory_eligibility(record)
    assert not report.eligible
    assert "PROTOCOL_NOT_SEALED" in report.error_codes


def test_contamination_propagates_to_every_descendant_run(tmp_path: Path) -> None:
    runs = RunManifestStore(tmp_path / "runs")
    provenance = RunProvenance(
        code_hash="a" * 64,
        dirty_tree=False,
        dependency_hash="b" * 64,
        runtime="python3.11+torch2.8.0+cpu",
        device="cpu",
        map_hash="c" * 64,
        split="train",
        seed=7,
        input_hashes={"map": "c" * 64},
    )
    origin = runs.create(
        protocol_hash="d" * 64, condition_hash="e" * 64, provenance=provenance, run_id="run-origin"
    )
    runs.seal(origin.run_id, execution_status=ExecutionStatus.COMPLETED, status_reason="finished")
    child = runs.fork(origin.run_id, new_run_id="run-child")
    runs.seal(child.run_id, execution_status=ExecutionStatus.COMPLETED, status_reason="finished")
    runs.fork(child.run_id, new_run_id="run-grandchild")
    runs.create(
        protocol_hash="d" * 64, condition_hash="e" * 64, provenance=provenance, run_id="run-unrelated"
    )

    assert contaminated_run_lineage(runs, "run-origin") == ("run-child", "run-grandchild", "run-origin")

    context = SelectionContext(
        context_id="ctx-leaky",
        purpose=SelectionPurpose.BASELINE_SELECTION,
        handles=_selection_handles()
        + (DataHandle(SplitScope.TEST, HandleKind.EPISODE, "held-out-episode-1"),),
        run_id="run-origin",
    )
    detected = inspect_selection_context(context)
    assert isinstance(detected, ContaminationRecord)

    propagated = propagate_contamination(detected, runs, origin_run_id="run-origin")
    assert propagated.excluded_run_ids == ("run-child", "run-grandchild", "run-origin")
    assert "run-unrelated" not in propagated.excluded_run_ids

    store, protocol_id = _sealed_store(tmp_path)
    with pytest.raises(ConfirmatoryGateError) as excinfo:
        require_confirmatory_eligibility(store.read(protocol_id), contamination=(propagated,))
    assert "CONTAMINATED_LINEAGE" in excinfo.value.report.error_codes
    assert "TEST_DATA_DEPENDENCY" in excinfo.value.report.error_codes

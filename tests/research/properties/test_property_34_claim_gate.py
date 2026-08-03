"""Property 34 coverage: the paper claim gate is fail-closed."""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from hypothesis import given, settings, strategies as st
import pytest

from pursuit_evasion_rl.research.budget import (
    ConditionResourceEstimate,
    ResourceCeiling,
    SampleSizeStatus,
    decide_sample_size,
)
from pursuit_evasion_rl.research.claims.gate import (
    ClaimDependencyBundle,
    ClaimGateError,
    inspect_claim_eligibility,
    reconciled_source_hashes,
    require_claim_eligibility,
)
from pursuit_evasion_rl.research.domain import (
    AnalysisClassification,
    ClaimRecord,
    ClaimScope,
    ClaimStatus,
    EvidenceRecord,
    EvidenceType,
    ExecutionStatus,
    ScreeningDecision,
)
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.execution import (
    ConditionExecutionLedger,
    ConditionExecutionRecord,
    MeasuredResult,
    success_only_views,
)
from pursuit_evasion_rl.research.maps.splits import DataHandle, HandleKind, SplitScope
from pursuit_evasion_rl.research.metrics.behavior import MetricDirection
from pursuit_evasion_rl.research.paper.citations import (
    CitationCandidate,
    CitationLedger,
    CitationRecord,
    MetadataCrossCheck,
    PersistentIdentifierKind,
    ProposalCrossCheck,
    SearchProtocolSpec,
    SearchSource,
    SearchSourceKind,
    SourceLocation,
)
from pursuit_evasion_rl.research.paper.scope import FutureWorkArtifact, FutureWorkRegistry, FutureWorkSystem
from pursuit_evasion_rl.research.protocol import (
    ConditionRef,
    ExclusionSpec,
    HypothesisSpec,
    MetricDeclaration,
    MetricRole,
    PlannedAnalysis,
    ProtocolRecord,
    ProtocolSpecifications,
    ProtocolStore,
    ResourceSpec,
    SampleSpec,
    SearchSpec,
    SelectionContext,
    SelectionPurpose,
    SplitSpec,
    StatisticsSpec,
    StopRule,
    StopSpec,
    ToleranceSpec,
    default_research_questions,
    metric_declaration,
    practical_thresholds_for,
)
from pursuit_evasion_rl.research.runs.manifest import RunManifest, RunManifestStore, RunProvenance
from pursuit_evasion_rl.research.statistics.paired import (
    BootstrapPlan,
    CorrectionMethod,
    EffectDirection,
    MissingDataPolicy,
    PairedCase,
    PairedComparisonResult,
    PracticalThreshold,
    compare_paired_binary,
)

# **Property 34: The paper claim gate is fail-closed**
# **Validates: Requirements 19.1, 19.3, 19.7-19.10**

pytestmark = pytest.mark.offline

_PBT_SETTINGS = settings(max_examples=100, deadline=None)

SIGNER = "principal-investigator"
UTC = "2026-07-31T00:00:00+00:00"
QUERY = "multi-agent reinforcement learning vehicle pursuit road network"
CONDITION_ID = "cooperative_containment"
CITATION_KEY = "yang2023"
FUTURE_WORK_ARTIFACT = "cctv-fusion-demo"
BOOTSTRAP = BootstrapPlan(n_bootstrap=200, rng_seed=11, rationale="unit-test runtime ceiling")


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _metrics():
    return (
        MetricDeclaration(
            metric_id="capture_rate", symbol="P_cap", unit="probability",
            direction=MetricDirection.HIGHER_IS_BETTER, role=MetricRole.PRIMARY,
            formula="captured episodes divided by planned episodes",
        ),
        metric_declaration("blocked_exit_fraction", MetricRole.PRIMARY),
    )


def _selection_handles():
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
                condition_id=CONDITION_ID, accelerator_hours_per_seed=1.0,
                wall_clock_hours_per_seed=1.0, env_steps_per_seed=1_000_000,
            ),
        ),
    )


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


def _specifications() -> ProtocolSpecifications:
    questions = default_research_questions()
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
        conditions=(ConditionRef(condition_id=CONDITION_ID, condition_hash=_hash("condition::cooperative_containment"), axis="coordination", arm="cooperative"),),
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
            search_date_utc="2026-07-01T00:00:00Z", query=QUERY,
            sources=("Scopus", "IEEE Xplore", "arXiv"), date_range_start="2015-01-01", date_range_end="2026-06-30",
            languages=("en", "ko"), inclusion_criteria=("multi-pursuer pursuit on a graph or road network",),
            exclusion_criteria=("continuous open-plane pursuit without a road network",),
        ),
        analyses=_analyses(questions),
    )


def _search_protocol() -> SearchProtocolSpec:
    return SearchProtocolSpec(
        search_date_utc=UTC, queries=(QUERY,),
        sources=(
            SearchSource(source_id="openalex", name="OpenAlex", kind=SearchSourceKind.ACADEMIC_INDEX, operator="OurResearch", endpoint="https://api.openalex.org/works"),
            SearchSource(source_id="scopus", name="Scopus", kind=SearchSourceKind.ACADEMIC_INDEX, operator="Elsevier", endpoint="https://api.elsevier.com/content/search/scopus"),
            SearchSource(source_id="arxiv", name="arXiv", kind=SearchSourceKind.PREPRINT_SERVER, operator="Cornell University", endpoint="https://export.arxiv.org/api/query"),
        ),
        period_start="2015-01-01", period_end="2026-07-31", languages=("en", "ko"),
        inclusion_criteria=("road-network pursuit or interception of a fleeing vehicle",),
        exclusion_criteria=("no evaluation on a road network",),
    )


def _citation_ledger(*, verified: bool = True) -> CitationLedger:
    ledger = CitationLedger(_search_protocol())
    work_id = "work::yang-progression-cognition"
    ledger.add_candidate(
        CitationCandidate(
            candidate_id=f"{CITATION_KEY}-c1", canonical_work_id=work_id, source_id="openalex",
            query=QUERY, executed_at_utc=UTC, result_rank=1, record_identifier=f"record::{CITATION_KEY}",
            result_content_hash=_hash(f"search-result::{CITATION_KEY}"),
        )
    )
    ledger.screen(f"{CITATION_KEY}-c1", ScreeningDecision.INCLUDED, "road-network pursuit with a learned policy", screened_at_utc=UTC)
    ledger.register_citation(
        CitationRecord(
            citation_key=CITATION_KEY, canonical_work_id=work_id,
            title="Progression Cognition Reinforcement Learning for Multi-Vehicle Pursuit",
            authors=("Yang, X.",), year=2023, candidate_ids=ledger.candidate_ids_for_work(work_id),
            proposal_cross_check=ProposalCrossCheck.NOT_FROM_PROPOSAL,
            metadata_cross_check=(
                MetadataCrossCheck(
                    identifier="10.1007/s41109-024-00689-1", identifier_kind=PersistentIdentifierKind.DOI,
                    metadata_source="Crossref", checked_at_utc=UTC, author_match=True, year_match=True, title_match=True,
                ) if verified else None
            ),
            source_location=(SourceLocation(original_text_content_hash=_hash(CITATION_KEY), section="4 Experiments", table="Table 2") if verified else None),
        )
    )
    return ledger


def _provenance() -> RunProvenance:
    return RunProvenance(
        code_hash=_hash("code"), dirty_tree=False, dependency_hash=_hash("deps"),
        runtime="python3.11+torch2.8.0+cpu", device="cpu", map_hash=_hash("daejeon-train"),
        split="train", seed=7, input_hashes={"map": _hash("daejeon-train")},
    )


def _sealed_manifest(store: RunManifestStore, protocol_hash: str, *, run_id: str) -> RunManifest:
    store.create(protocol_hash=protocol_hash, condition_hash=_hash("condition::cooperative_containment"), provenance=_provenance(), run_id=run_id)
    return store.seal(
        run_id, execution_status=ExecutionStatus.COMPLETED, status_reason="all planned seeds finished",
        artifact_hashes={"metrics": _hash(f"metrics::{run_id}")}, result={"capture_rate": 0.62},
    )


def _paired_cases(*, effect: bool = True):
    cases = []
    for seed in range(5):
        for index in range(100):
            if index < 70:
                outcomes = (True, True)
            elif index < 90:
                outcomes = (True, False) if effect else (True, True)
            else:
                outcomes = (False, False)
            cases.append(PairedCase(case_id=f"case-{seed}-{index}", training_seed=seed, outcome_a=outcomes[0], outcome_b=outcomes[1]))
    return tuple(cases)


def _paired_result(*, effect: bool = True) -> PairedComparisonResult:
    return compare_paired_binary(
        _paired_cases(effect=effect), comparison_id="cooperative-vs-independent", metric="capture_rate", unit="probability",
        plan=BOOTSTRAP, threshold=PracticalThreshold(metric="capture_rate", unit="probability", minimum_effect=0.05, direction=EffectDirection.GREATER_IS_BETTER),
    )


def _execution_ledger(protocol_hash, *, capture_rate=0.62, measured=True, execution_status=ExecutionStatus.COMPLETED, sample_size_status=SampleSizeStatus.DEFAULT):
    resource = _resource_spec()
    return ConditionExecutionLedger(
        protocol_hash=protocol_hash, sample_size_plan=decide_sample_size(resource.ceiling, resource.estimates),
        planned_condition_ids=(CONDITION_ID,),
        records=(
            ConditionExecutionRecord(
                condition_id=CONDITION_ID, protocol_hash=protocol_hash, execution_status=execution_status,
                status_reason="all planned seeds finished", sample_size_status=sample_size_status,
                artifact_hashes={"metrics": _hash("metrics::run-primary")},
                result=MeasuredResult(value=capture_rate) if measured else None, run_ids=("run-primary",),
            ),
        ),
    )


def _future_work_registry() -> FutureWorkRegistry:
    return FutureWorkRegistry(
        primary_result_root="artifacts/primary", future_work_root="artifacts/future_work",
        artifacts=(
            FutureWorkArtifact(
                artifact_id=FUTURE_WORK_ARTIFACT, system=FutureWorkSystem.CCTV,
                description="illustrative CCTV detection fusion demo, not evaluated",
                artifact_path="artifacts/future_work/cctv/demo.mp4",
            ),
        ),
    )


def _evidence():
    return (
        EvidenceRecord(
            record_id="ev-capture", evidence_type=EvidenceType.EXPERIMENT_OBSERVATION, producer="paired-evaluator",
            created_at_utc=UTC, method="paired episode evaluation over five training seeds", source="run-primary metrics",
            extracted_value=0.62, verification="metrics artifact hash matches the sealed manifest",
        ),
        EvidenceRecord(
            record_id=FUTURE_WORK_ARTIFACT, evidence_type=EvidenceType.IMPLEMENTATION_EXISTENCE, producer="future-work demo",
            created_at_utc=UTC, method="unevaluated illustrative demo recording", source="artifacts/future_work/cctv/demo.mp4",
            extracted_value="demo exists", verification="file present, no measurement performed",
        ),
    )


def _result_claim(text: str = "Cooperative containment raises the paired capture rate.") -> ClaimRecord:
    return ClaimRecord(
        claim_id="claim-rq2-containment", text=text, scope=ClaimScope.RESULT, status=ClaimStatus.SUPPORTED,
        evidence_ids=("ev-capture",), analysis_ids=("A-RQ2_cooperative_containment_stability",), citation_ids=(CITATION_KEY,),
    )


@dataclass(frozen=True, slots=True)
class _Parts:
    protocols: ProtocolStore
    runs: RunManifestStore
    sealed: ProtocolRecord
    manifest: RunManifest
    statistics: PairedComparisonResult
    null_statistics: PairedComparisonResult


def _build_parts(tmp_path: Path) -> _Parts:
    protocols = ProtocolStore(tmp_path / "protocols")
    draft = protocols.create(questions=default_research_questions(), specifications=_specifications())
    sealed = protocols.seal(draft.protocol_id, signer=SIGNER)
    runs = RunManifestStore(tmp_path / "runs")
    manifest = _sealed_manifest(runs, sealed.protocol_hash, run_id="run-primary")
    return _Parts(
        protocols=protocols, runs=runs, sealed=sealed, manifest=manifest,
        statistics=_paired_result(), null_statistics=_paired_result(effect=False),
    )


def _confirmatory_event(parts: _Parts, analysis_id: str):
    declared = next(item for item in _analyses(default_research_questions()) if item.analysis_id == analysis_id)
    path = parts.protocols.root / parts.sealed.protocol_id / "analyses" / f"{analysis_id}.json"
    path.unlink(missing_ok=True)
    return parts.protocols.register_analysis(parts.sealed.protocol_id, declared)


def _bundle(parts: _Parts, claim: ClaimRecord | None = None, **overrides) -> ClaimDependencyBundle:
    claim = claim if claim is not None else _result_claim()
    fields = dict(
        claim=claim, protocol=parts.sealed,
        analysis_event=(_confirmatory_event(parts, claim.analysis_ids[0]) if claim.analysis_ids else None),
        selection_contexts=(SelectionContext(context_id="ctx-checkpoint-selection", purpose=SelectionPurpose.CHECKPOINT_SELECTION, handles=_selection_handles()),),
        evidence=_evidence(), citation_ledger=_citation_ledger(), body_citation_keys=(CITATION_KEY,),
        run_manifests=(parts.manifest,), source_hashes=reconciled_source_hashes((parts.manifest,)),
        statistics=parts.statistics, asserts_superiority=True, execution=_execution_ledger(parts.sealed.protocol_hash),
        condition_ids=(CONDITION_ID,), future_work=_future_work_registry(),
    )
    fields.update(overrides)
    return ClaimDependencyBundle(**fields)


def _unsealed_protocol(parts: _Parts) -> ProtocolRecord:
    return parts.protocols.create(questions=default_research_questions(), specifications=_specifications())


def _exploratory_event(parts: _Parts):
    unplanned = PlannedAnalysis(
        analysis_id="A-unplanned-subgroup", question_id="RQ2_cooperative_containment_stability",
        classification=AnalysisClassification.CONFIRMATORY, estimand="paired difference in capture rate for dead-end-heavy maps only",
        metric_id="capture_rate", comparison="post-hoc subgroup",
    )
    path = parts.protocols.root / parts.sealed.protocol_id / "analyses" / f"{unplanned.analysis_id}.json"
    path.unlink(missing_ok=True)
    return parts.protocols.register_analysis(parts.sealed.protocol_id, unplanned)


def _tampered_source_hashes(parts: _Parts) -> dict[str, str]:
    hashes = reconciled_source_hashes((parts.manifest,))
    hashes["run-primary"] = _hash("a number that was never produced by this run")
    return hashes


def _draft_manifest(parts: _Parts) -> RunManifest:
    run_id = f"run-unsealed-{uuid4().hex[:12]}"
    return parts.runs.create(protocol_hash=parts.sealed.protocol_hash, condition_hash=_hash("condition::cooperative_containment"), provenance=_provenance(), run_id=run_id)


BREAKS: tuple[tuple[str, str], ...] = (
    ("unsealed_protocol", "PROTOCOL_NOT_SEALED"),
    ("exploratory_analysis", "ANALYSIS_NOT_CONFIRMATORY"),
    ("test_split_leak", "TEST_DATA_DEPENDENCY"),
    ("unverified_citation", "UNVERIFIED_CITATION"),
    ("source_hash_mismatch", "SOURCE_HASH_MISMATCH"),
    ("unsealed_manifest", "RUN_MANIFEST_NOT_SEALED"),
    ("statistics_ineligible", "STATISTICS_NOT_CONFIRMATORY_ELIGIBLE"),
    ("practical_gate_failed", "PRACTICAL_GATE_NOT_PASSED"),
    ("condition_not_run", "CONDITION_NOT_COMPLETED"),
    ("success_only_view", "SUCCESS_ONLY_POPULATION_VIEW"),
    ("future_work_evidence", "FUTURE_WORK_DEPENDENCY_BLOCKED"),
    ("overclaiming_text", "FIELD_READINESS_DERIVATION_BLOCKED"),
)


def _broken_bundle(parts: _Parts, name: str) -> ClaimDependencyBundle:
    if name == "unsealed_protocol":
        return _bundle(parts, protocol=_unsealed_protocol(parts))
    if name == "exploratory_analysis":
        return _bundle(parts, analysis_event=_exploratory_event(parts))
    if name == "test_split_leak":
        return _bundle(parts, selection_contexts=(
            SelectionContext(context_id="ctx-leaky-checkpoint-selection", purpose=SelectionPurpose.CHECKPOINT_SELECTION,
                              handles=_selection_handles() + (DataHandle(SplitScope.TEST, HandleKind.METRIC, "held-out-capture-rate"),)),
        ))
    if name == "unverified_citation":
        return _bundle(parts, citation_ledger=_citation_ledger(verified=False))
    if name == "source_hash_mismatch":
        return _bundle(parts, source_hashes=_tampered_source_hashes(parts))
    if name == "unsealed_manifest":
        draft = _draft_manifest(parts)
        return _bundle(parts, run_manifests=(draft,), source_hashes={draft.run_id: str(draft.run.content_hash)})
    if name == "statistics_ineligible":
        return _bundle(parts, statistics=dataclasses.replace(parts.statistics, confirmatory_eligible=False))
    if name == "practical_gate_failed":
        return _bundle(parts, statistics=parts.null_statistics)
    if name == "condition_not_run":
        return _bundle(parts, execution=_execution_ledger(parts.sealed.protocol_hash, measured=False, execution_status=ExecutionStatus.NOT_RUN))
    if name == "success_only_view":
        ledger = _execution_ledger(parts.sealed.protocol_hash)
        return _bundle(parts, execution=ledger, population_view=success_only_views(ledger).success_only)
    if name == "future_work_evidence":
        return _bundle(parts, dataclasses.replace(
            ClaimRecord(
                claim_id="claim-novelty-osm", text="This study evaluates cooperative containment on extracted OSM road networks.",
                scope=ClaimScope.NOVELTY, status=ClaimStatus.SUPPORTED, evidence_ids=("ev-capture", FUTURE_WORK_ARTIFACT),
                analysis_ids=("A-RQ2_cooperative_containment_stability",), citation_ids=(CITATION_KEY,), comparison_axes=("map_source_realism",),
            ), content_hash=None,
        ))
    if name == "overclaiming_text":
        return _bundle(parts, _result_claim(text="Cooperative containment delivers a real-world capture rate gain."))
    raise AssertionError(f"unhandled break {name!r}")


_BREAK_NAMES = tuple(name for name, _ in BREAKS)
_CODE_FOR = dict(BREAKS)


@pytest.fixture(scope="module")
def parts(tmp_path_factory: pytest.TempPathFactory) -> _Parts:
    return _build_parts(tmp_path_factory.mktemp("claim-gate-property"))


@_PBT_SETTINGS
@given(name=st.sampled_from(_BREAK_NAMES))
def test_each_named_defect_blocks_export_alone_with_its_own_code_and_nothing_else(parts: _Parts, name: str) -> None:
    assert inspect_claim_eligibility(_bundle(parts)).eligible, "the baseline bundle must pass"
    report = inspect_claim_eligibility(_broken_bundle(parts, name))
    assert not report.eligible
    assert report.error_codes == (_CODE_FOR[name],)
    assert report.errors[0].code == _CODE_FOR[name]


@_PBT_SETTINGS
@given(name=st.sampled_from(_BREAK_NAMES))
def test_a_blocked_bundle_raises_and_preserves_the_full_report_in_the_error(parts: _Parts, name: str) -> None:
    with pytest.raises(ClaimGateError) as excinfo:
        require_claim_eligibility(_broken_bundle(parts, name))
    assert excinfo.value.report.eligible is False
    assert excinfo.value.report.error_codes == (_CODE_FOR[name],)
    assert excinfo.value.record.details["gate_report_hash"] == str(excinfo.value.report.content_hash)
    assert excinfo.value.code == _CODE_FOR[name]


@_PBT_SETTINGS
@given(names=st.lists(st.sampled_from(_BREAK_NAMES), min_size=2, max_size=4, unique=True))
def test_the_gate_is_never_partially_eligible_a_rejected_report_always_carries_errors(parts: _Parts, names: list[str]) -> None:
    """Combine several structural facts (not simultaneous injection, since breaks touch shared fields);
    each individually rejected report must satisfy the report invariant on its own."""
    for name in names:
        report = inspect_claim_eligibility(_broken_bundle(parts, name))
        assert report.eligible is False
        assert len(report.errors) >= 1
        assert bool(report.errors) != report.eligible


@_PBT_SETTINGS
@given(capture_rate=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
def test_field_readiness_verdict_is_byte_identical_no_matter_what_was_measured(parts: _Parts, capture_rate: float) -> None:
    claim = ClaimRecord(
        claim_id="claim-field-readiness",
        text="Whether this policy would perform in an operational pursuit is not established by this study.",
        scope=ClaimScope.FIELD_READINESS, status=ClaimStatus.UNSUPPORTED, evidence_ids=("ev-capture",),
    )
    bundle = _bundle(parts, claim, execution=_execution_ledger(parts.sealed.protocol_hash, capture_rate=capture_rate), asserts_superiority=False)
    report = inspect_claim_eligibility(bundle)
    assert report.status is ClaimStatus.UNSUPPORTED
    assert report.eligible
    assert report.errors == ()

    baseline = inspect_claim_eligibility(
        _bundle(parts, claim, execution=_execution_ledger(parts.sealed.protocol_hash, capture_rate=0.5), asserts_superiority=False)
    )
    assert report.content_hash == baseline.content_hash, "no measurement may reach the report"


@_PBT_SETTINGS
@given(missing=st.sampled_from(("citation_ledger", "statistics", "execution")))
def test_a_missing_optional_dependency_is_reported_rather_than_silently_passing(parts: _Parts, missing: str) -> None:
    report = inspect_claim_eligibility(_bundle(parts, **{missing: None}))
    assert not report.eligible
    expected_code = {
        "citation_ledger": "MISSING_CITATION_LEDGER",
        "statistics": "MISSING_STATISTICAL_BACKING",
        "execution": "MISSING_EXECUTION_LEDGER",
    }[missing]
    assert expected_code in report.error_codes


@_PBT_SETTINGS
@given(capture_rate=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
def test_a_fully_satisfied_result_claim_stays_eligible_across_any_measured_capture_rate(parts: _Parts, capture_rate: float) -> None:
    """Requirement 4.7: an unfavourable or null result is still a completed, claimable outcome."""
    report = inspect_claim_eligibility(_bundle(parts, execution=_execution_ledger(parts.sealed.protocol_hash, capture_rate=capture_rate)))
    assert report.eligible
    assert report.errors == ()

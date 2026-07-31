"""Task 8.3 regressions for the fail-closed Paper_Claim_Gate."""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from pathlib import Path

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
    ClaimGateReport,
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
    ResearchQuestion,
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
from pursuit_evasion_rl.research.paper.scope import (
    FutureWorkArtifact,
    FutureWorkRegistry,
    FutureWorkSystem,
)
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
from pursuit_evasion_rl.research.runs.manifest import (
    RunManifest,
    RunManifestStore,
    RunProvenance,
)
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

pytestmark = pytest.mark.offline

SIGNER = "principal-investigator"
UTC = "2026-07-31T00:00:00+00:00"
QUERY = "multi-agent reinforcement learning vehicle pursuit road network"
CONDITION_ID = "cooperative_containment"
CITATION_KEY = "yang2023"
FUTURE_WORK_ARTIFACT = "cctv-fusion-demo"
BOOTSTRAP = BootstrapPlan(n_bootstrap=200, rng_seed=11, rationale="unit-test runtime ceiling")


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Sealed protocol
# ---------------------------------------------------------------------------


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
    )


def _selection_handles() -> tuple[DataHandle[SplitScope], ...]:
    return (
        DataHandle(SplitScope.TRAIN, HandleKind.MAP, "daejeon-train"),
        DataHandle(SplitScope.VALIDATION, HandleKind.MAP, "daejeon-validation"),
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
                condition_id=CONDITION_ID,
                accelerator_hours_per_seed=1.0,
                wall_clock_hours_per_seed=1.0,
                env_steps_per_seed=1_000_000,
            ),
        ),
    )


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


def _specifications() -> ProtocolSpecifications:
    questions = default_research_questions()
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
                condition_id=CONDITION_ID,
                condition_hash=_hash("condition::cooperative_containment"),
                axis="coordination",
                arm="cooperative",
            ),
        ),
        split=SplitSpec(
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
        ),
        sample=SampleSpec(
            plan=decide_sample_size(resource.ceiling, resource.estimates),
            selection_handles=_selection_handles(),
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
            applies_to=("resume_equivalence",),
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
            query=QUERY,
            sources=("Scopus", "IEEE Xplore", "arXiv"),
            date_range_start="2015-01-01",
            date_range_end="2026-06-30",
            languages=("en", "ko"),
            inclusion_criteria=("multi-pursuer pursuit on a graph or road network",),
            exclusion_criteria=("continuous open-plane pursuit without a road network",),
        ),
        analyses=_analyses(questions),
    )


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------


def _search_protocol() -> SearchProtocolSpec:
    return SearchProtocolSpec(
        search_date_utc=UTC,
        queries=(QUERY,),
        sources=(
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
        ),
        period_start="2015-01-01",
        period_end="2026-07-31",
        languages=("en", "ko"),
        inclusion_criteria=("road-network pursuit or interception of a fleeing vehicle",),
        exclusion_criteria=("no evaluation on a road network",),
    )


def _citation_ledger(*, verified: bool = True) -> CitationLedger:
    """A ledger holding exactly one screened citation, verified or not on request."""
    ledger = CitationLedger(_search_protocol())
    work_id = "work::yang-progression-cognition"
    ledger.add_candidate(
        CitationCandidate(
            candidate_id=f"{CITATION_KEY}-c1",
            canonical_work_id=work_id,
            source_id="openalex",
            query=QUERY,
            executed_at_utc=UTC,
            result_rank=1,
            record_identifier=f"record::{CITATION_KEY}",
            result_content_hash=_hash(f"search-result::{CITATION_KEY}"),
        )
    )
    ledger.screen(
        f"{CITATION_KEY}-c1",
        ScreeningDecision.INCLUDED,
        "road-network pursuit with a learned policy",
        screened_at_utc=UTC,
    )
    ledger.register_citation(
        CitationRecord(
            citation_key=CITATION_KEY,
            canonical_work_id=work_id,
            title="Progression Cognition Reinforcement Learning for Multi-Vehicle Pursuit",
            authors=("Yang, X.",),
            year=2023,
            candidate_ids=ledger.candidate_ids_for_work(work_id),
            proposal_cross_check=ProposalCrossCheck.NOT_FROM_PROPOSAL,
            metadata_cross_check=(
                MetadataCrossCheck(
                    identifier="10.1007/s41109-024-00689-1",
                    identifier_kind=PersistentIdentifierKind.DOI,
                    metadata_source="Crossref",
                    checked_at_utc=UTC,
                    author_match=True,
                    year_match=True,
                    title_match=True,
                )
                if verified
                else None
            ),
            source_location=(
                SourceLocation(
                    original_text_content_hash=_hash(CITATION_KEY),
                    section="4 Experiments",
                    table="Table 2",
                )
                if verified
                else None
            ),
        )
    )
    return ledger


# ---------------------------------------------------------------------------
# Runs, statistics, execution
# ---------------------------------------------------------------------------


def _provenance() -> RunProvenance:
    return RunProvenance(
        code_hash=_hash("code"),
        dirty_tree=False,
        dependency_hash=_hash("deps"),
        runtime="python3.11+torch2.8.0+cpu",
        device="cpu",
        map_hash=_hash("daejeon-train"),
        split="train",
        seed=7,
        input_hashes={"map": _hash("daejeon-train")},
    )


def _sealed_manifest(store: RunManifestStore, protocol_hash: str, *, run_id: str) -> RunManifest:
    store.create(
        protocol_hash=protocol_hash,
        condition_hash=_hash("condition::cooperative_containment"),
        provenance=_provenance(),
        run_id=run_id,
    )
    return store.seal(
        run_id,
        execution_status=ExecutionStatus.COMPLETED,
        status_reason="all planned seeds finished",
        artifact_hashes={"metrics": _hash(f"metrics::{run_id}")},
        result={"capture_rate": 0.62},
    )


def _paired_cases(*, effect: bool = True) -> tuple[PairedCase, ...]:
    """Five seeds, one hundred pairs each; ``effect`` gives policy A a 0.20 risk difference."""
    cases: list[PairedCase] = []
    for seed in range(5):
        for index in range(100):
            if index < 70:
                outcomes = (True, True)
            elif index < 90:
                outcomes = (True, False) if effect else (True, True)
            else:
                outcomes = (False, False)
            cases.append(
                PairedCase(
                    case_id=f"case-{seed}-{index}",
                    training_seed=seed,
                    outcome_a=outcomes[0],
                    outcome_b=outcomes[1],
                )
            )
    return tuple(cases)


def _paired_result(*, effect: bool = True) -> PairedComparisonResult:
    return compare_paired_binary(
        _paired_cases(effect=effect),
        comparison_id="cooperative-vs-independent",
        metric="capture_rate",
        unit="probability",
        plan=BOOTSTRAP,
        threshold=PracticalThreshold(
            metric="capture_rate",
            unit="probability",
            minimum_effect=0.05,
            direction=EffectDirection.GREATER_IS_BETTER,
        ),
    )


def _execution_ledger(
    protocol_hash: str,
    *,
    capture_rate: float | None = 0.62,
    measured: bool = True,
    execution_status: ExecutionStatus = ExecutionStatus.COMPLETED,
    sample_size_status: SampleSizeStatus = SampleSizeStatus.DEFAULT,
) -> ConditionExecutionLedger:
    resource = _resource_spec()
    return ConditionExecutionLedger(
        protocol_hash=protocol_hash,
        sample_size_plan=decide_sample_size(resource.ceiling, resource.estimates),
        planned_condition_ids=(CONDITION_ID,),
        records=(
            ConditionExecutionRecord(
                condition_id=CONDITION_ID,
                protocol_hash=protocol_hash,
                execution_status=execution_status,
                status_reason="all planned seeds finished",
                sample_size_status=sample_size_status,
                artifact_hashes={"metrics": _hash("metrics::run-primary")},
                result=MeasuredResult(value=capture_rate) if measured else None,
                run_ids=("run-primary",),
            ),
        ),
    )


def _future_work_registry() -> FutureWorkRegistry:
    return FutureWorkRegistry(
        primary_result_root="artifacts/primary",
        future_work_root="artifacts/future_work",
        artifacts=(
            FutureWorkArtifact(
                artifact_id=FUTURE_WORK_ARTIFACT,
                system=FutureWorkSystem.CCTV,
                description="illustrative CCTV detection fusion demo, not evaluated",
                artifact_path="artifacts/future_work/cctv/demo.mp4",
            ),
        ),
    )


def _evidence() -> tuple[EvidenceRecord, ...]:
    return (
        EvidenceRecord(
            record_id="ev-capture",
            evidence_type=EvidenceType.EXPERIMENT_OBSERVATION,
            producer="paired-evaluator",
            created_at_utc=UTC,
            method="paired episode evaluation over five training seeds",
            source="run-primary metrics",
            extracted_value=0.62,
            verification="metrics artifact hash matches the sealed manifest",
        ),
        EvidenceRecord(
            record_id=FUTURE_WORK_ARTIFACT,
            evidence_type=EvidenceType.IMPLEMENTATION_EXISTENCE,
            producer="future-work demo",
            created_at_utc=UTC,
            method="unevaluated illustrative demo recording",
            source="artifacts/future_work/cctv/demo.mp4",
            extracted_value="demo exists",
            verification="file present, no measurement performed",
        ),
    )


# ---------------------------------------------------------------------------
# Claims and the assembled bundle
# ---------------------------------------------------------------------------


def _result_claim(text: str = "Cooperative containment raises the paired capture rate.") -> ClaimRecord:
    return ClaimRecord(
        claim_id="claim-rq2-containment",
        text=text,
        scope=ClaimScope.RESULT,
        status=ClaimStatus.SUPPORTED,
        evidence_ids=("ev-capture",),
        analysis_ids=("A-RQ2_cooperative_containment_stability",),
        citation_ids=(CITATION_KEY,),
    )


def _novelty_claim(evidence_ids: tuple[str, ...] = ("ev-capture",)) -> ClaimRecord:
    return ClaimRecord(
        claim_id="claim-novelty-osm",
        text="This study evaluates cooperative containment on extracted OSM road networks.",
        scope=ClaimScope.NOVELTY,
        status=ClaimStatus.SUPPORTED,
        evidence_ids=evidence_ids,
        analysis_ids=("A-RQ2_cooperative_containment_stability",),
        citation_ids=(CITATION_KEY,),
        comparison_axes=("map_source_realism",),
    )


def _field_readiness_claim() -> ClaimRecord:
    return ClaimRecord(
        claim_id="claim-field-readiness",
        text=(
            "Whether this policy would perform in an operational pursuit is not established "
            "by this study."
        ),
        scope=ClaimScope.FIELD_READINESS,
        status=ClaimStatus.UNSUPPORTED,
        evidence_ids=("ev-capture",),
    )


@dataclass(frozen=True, slots=True)
class _Parts:
    """Every real object a passing bundle is assembled from."""

    protocols: ProtocolStore
    runs: RunManifestStore
    sealed: ProtocolRecord
    manifest: RunManifest
    statistics: PairedComparisonResult
    null_statistics: PairedComparisonResult


@pytest.fixture(scope="module")
def parts(tmp_path_factory: pytest.TempPathFactory) -> _Parts:
    tmp_path: Path = tmp_path_factory.mktemp("claim-gate")
    protocols = ProtocolStore(tmp_path / "protocols")
    draft = protocols.create(
        questions=default_research_questions(), specifications=_specifications()
    )
    sealed = protocols.seal(draft.protocol_id, signer=SIGNER)
    runs = RunManifestStore(tmp_path / "runs")
    manifest = _sealed_manifest(runs, sealed.protocol_hash, run_id="run-primary")
    return _Parts(
        protocols=protocols,
        runs=runs,
        sealed=sealed,
        manifest=manifest,
        statistics=_paired_result(),
        null_statistics=_paired_result(effect=False),
    )


def _confirmatory_event(parts: _Parts, analysis_id: str):
    """Register the pre-registered analysis unchanged, which classifies it confirmatory."""
    declared = next(
        item for item in _analyses(default_research_questions()) if item.analysis_id == analysis_id
    )
    path = parts.protocols.root / parts.sealed.protocol_id / "analyses" / f"{analysis_id}.json"
    path.unlink(missing_ok=True)
    return parts.protocols.register_analysis(parts.sealed.protocol_id, declared)


def _bundle(parts: _Parts, claim: ClaimRecord | None = None, **overrides) -> ClaimDependencyBundle:
    """A fully-satisfied bundle: every mandatory dependency present and passing."""
    claim = claim if claim is not None else _result_claim()
    fields = dict(
        claim=claim,
        protocol=parts.sealed,
        analysis_event=(
            _confirmatory_event(parts, claim.analysis_ids[0]) if claim.analysis_ids else None
        ),
        selection_contexts=(
            SelectionContext(
                context_id="ctx-checkpoint-selection",
                purpose=SelectionPurpose.CHECKPOINT_SELECTION,
                handles=_selection_handles(),
            ),
        ),
        evidence=_evidence(),
        citation_ledger=_citation_ledger(),
        body_citation_keys=(CITATION_KEY,),
        run_manifests=(parts.manifest,),
        source_hashes=reconciled_source_hashes((parts.manifest,)),
        statistics=parts.statistics,
        asserts_superiority=True,
        execution=_execution_ledger(parts.sealed.protocol_hash),
        condition_ids=(CONDITION_ID,),
        future_work=_future_work_registry(),
    )
    fields.update(overrides)
    return ClaimDependencyBundle(**fields)


# ---------------------------------------------------------------------------
# Requirement 17.1, 19.10: field readiness never moves, whatever the number says
# ---------------------------------------------------------------------------


def test_field_readiness_is_unsupported_at_any_capture_rate(parts: _Parts) -> None:
    """The single load-bearing invariant: performance is not an input to this verdict."""
    claim = _field_readiness_claim()
    near_perfect = _bundle(
        parts,
        claim,
        execution=_execution_ledger(parts.sealed.protocol_hash, capture_rate=0.99),
        asserts_superiority=False,
    )
    near_useless = _bundle(
        parts,
        claim,
        execution=_execution_ledger(parts.sealed.protocol_hash, capture_rate=0.10),
        asserts_superiority=False,
    )
    assert near_perfect.execution.record_for(CONDITION_ID).result_value == 0.99
    assert near_useless.execution.record_for(CONDITION_ID).result_value == 0.10

    high = inspect_claim_eligibility(near_perfect)
    low = inspect_claim_eligibility(near_useless)

    assert high.status is ClaimStatus.UNSUPPORTED
    assert low.status is ClaimStatus.UNSUPPORTED
    assert high.eligible and low.eligible, "a transparently unsupported claim stays exportable"
    assert high.errors == () and low.errors == ()
    assert high.content_hash == low.content_hash, (
        "the two verdicts must be byte-identical: no measurement may reach the report"
    )


def test_field_readiness_may_be_exported_but_never_asserted(parts: _Parts) -> None:
    """The gate blocks a stronger assertion, not the transparent unsupported report itself."""
    claim = _field_readiness_claim()
    reported = inspect_claim_eligibility(_bundle(parts, claim, asserts_superiority=False))
    assert reported.eligible
    assert reported.status is ClaimStatus.UNSUPPORTED

    asserted = inspect_claim_eligibility(
        _bundle(
            parts,
            claim,
            execution=_execution_ledger(parts.sealed.protocol_hash, capture_rate=1.0),
            asserts_superiority=True,
        )
    )
    assert not asserted.eligible
    assert asserted.error_codes == ("FIELD_READINESS_ASSERTION_BLOCKED",)
    assert asserted.status is ClaimStatus.UNSUPPORTED


def test_a_field_readiness_claim_cannot_be_constructed_as_supported() -> None:
    with pytest.raises(ResearchValidationError) as excinfo:
        ClaimRecord(
            claim_id="claim-field-readiness",
            text="The policy is ready for operational use.",
            scope=ClaimScope.FIELD_READINESS,
            status=ClaimStatus.SUPPORTED,
            evidence_ids=("ev-capture",),
        )
    assert excinfo.value.code == "FIELD_READINESS_UNSUPPORTED"


# ---------------------------------------------------------------------------
# Requirement 19.7, 19.9: every mandatory dependency blocks export on its own
# ---------------------------------------------------------------------------


def _unsealed_protocol(parts: _Parts) -> ProtocolRecord:
    return parts.protocols.create(
        questions=default_research_questions(), specifications=_specifications()
    )


def _exploratory_event(parts: _Parts):
    """An analysis absent from the sealed protocol is demoted to exploratory."""
    unplanned = PlannedAnalysis(
        analysis_id="A-unplanned-subgroup",
        question_id="RQ2_cooperative_containment_stability",
        classification=AnalysisClassification.CONFIRMATORY,
        estimand="paired difference in capture rate for dead-end-heavy maps only",
        metric_id="capture_rate",
        comparison="post-hoc subgroup",
    )
    path = (
        parts.protocols.root
        / parts.sealed.protocol_id
        / "analyses"
        / f"{unplanned.analysis_id}.json"
    )
    path.unlink(missing_ok=True)
    return parts.protocols.register_analysis(parts.sealed.protocol_id, unplanned)


def _tampered_source_hashes(parts: _Parts) -> dict[str, str]:
    hashes = reconciled_source_hashes((parts.manifest,))
    hashes["run-primary"] = _hash("a number that was never produced by this run")
    return hashes


def _draft_manifest(parts: _Parts) -> RunManifest:
    return parts.runs.create(
        protocol_hash=parts.sealed.protocol_hash,
        condition_hash=_hash("condition::cooperative_containment"),
        provenance=_provenance(),
        run_id="run-unsealed",
    )


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
    ("evidence_type_mismatch", "EVIDENCE_TYPE_SCOPE_MISMATCH"),
)


def _broken_bundle(parts: _Parts, name: str) -> ClaimDependencyBundle:
    if name == "unsealed_protocol":
        return _bundle(parts, protocol=_unsealed_protocol(parts))
    if name == "exploratory_analysis":
        return _bundle(parts, analysis_event=_exploratory_event(parts))
    if name == "test_split_leak":
        return _bundle(
            parts,
            selection_contexts=(
                SelectionContext(
                    context_id="ctx-leaky-checkpoint-selection",
                    purpose=SelectionPurpose.CHECKPOINT_SELECTION,
                    handles=_selection_handles()
                    + (DataHandle(SplitScope.TEST, HandleKind.METRIC, "held-out-capture-rate"),),
                ),
            ),
        )
    if name == "unverified_citation":
        return _bundle(parts, citation_ledger=_citation_ledger(verified=False))
    if name == "source_hash_mismatch":
        return _bundle(parts, source_hashes=_tampered_source_hashes(parts))
    if name == "unsealed_manifest":
        draft = _draft_manifest(parts)
        return _bundle(
            parts,
            run_manifests=(draft,),
            source_hashes={draft.run_id: str(draft.run.content_hash)},
        )
    if name == "statistics_ineligible":
        return _bundle(
            parts,
            statistics=dataclasses.replace(parts.statistics, confirmatory_eligible=False),
        )
    if name == "practical_gate_failed":
        return _bundle(parts, statistics=parts.null_statistics)
    if name == "condition_not_run":
        return _bundle(
            parts,
            execution=_execution_ledger(
                parts.sealed.protocol_hash,
                measured=False,
                execution_status=ExecutionStatus.NOT_RUN,
            ),
        )
    if name == "success_only_view":
        ledger = _execution_ledger(parts.sealed.protocol_hash)
        return _bundle(
            parts, execution=ledger, population_view=success_only_views(ledger).success_only
        )
    if name == "future_work_evidence":
        return _bundle(parts, _novelty_claim(evidence_ids=("ev-capture", FUTURE_WORK_ARTIFACT)))
    if name == "overclaiming_text":
        return _bundle(
            parts,
            _result_claim(text="Cooperative containment delivers a real-world capture rate gain."),
        )
    if name == "evidence_type_mismatch":
        return _bundle(
            parts,
            dataclasses.replace(
                _result_claim(), evidence_ids=(FUTURE_WORK_ARTIFACT,), content_hash=None
            ),
            future_work=None,
        )
    raise AssertionError(f"unhandled break {name!r}")


@pytest.mark.parametrize("name,expected_code", BREAKS, ids=[item[0] for item in BREAKS])
def test_each_mandatory_dependency_blocks_export_on_its_own(
    parts: _Parts, name: str, expected_code: str
) -> None:
    """One broken dependency is reported alone -- no cascade masking the others."""
    assert inspect_claim_eligibility(_bundle(parts)).eligible, "the baseline bundle must pass"

    report = inspect_claim_eligibility(_broken_bundle(parts, name))

    assert not report.eligible
    assert report.error_codes == (expected_code,), (
        f"breaking {name} must name exactly that dependency, not a cascade"
    )


def test_a_novelty_claim_needs_the_future_work_registry_to_be_screened(parts: _Parts) -> None:
    report = inspect_claim_eligibility(_bundle(parts, _novelty_claim(), future_work=None))
    assert report.error_codes == ("MISSING_FUTURE_WORK_REGISTRY",)


def test_missing_optional_dependencies_are_each_reported(parts: _Parts) -> None:
    """Several simultaneous failures all survive into one report."""
    report = inspect_claim_eligibility(
        _bundle(parts, citation_ledger=None, statistics=None, execution=None)
    )
    assert set(report.error_codes) == {
        "MISSING_CITATION_LEDGER",
        "MISSING_STATISTICAL_BACKING",
        "MISSING_EXECUTION_LEDGER",
    }


def test_a_body_reference_without_a_ledger_entry_fails_reconciliation(parts: _Parts) -> None:
    report = inspect_claim_eligibility(
        _bundle(parts, body_citation_keys=(CITATION_KEY, "ghost2024"))
    )
    assert report.error_codes == ("BODY_REFERENCE_MISSING",)
    assert report.errors[0].actual == ["ghost2024"]


def test_a_ledger_entry_nobody_cites_is_an_orphan(parts: _Parts) -> None:
    report = inspect_claim_eligibility(_bundle(parts, body_citation_keys=()))
    assert report.error_codes == ("BODY_REFERENCE_ORPHAN",)
    assert report.errors[0].actual == [CITATION_KEY]


# ---------------------------------------------------------------------------
# Requirement 4.6-4.7: an unfavourable outcome is still a completed outcome
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("capture_rate", [0.99, 0.10, None])
def test_a_completed_condition_stays_claimable_whatever_it_measured(
    parts: _Parts, capture_rate: float | None
) -> None:
    """Requirement 4.7: null and unfavourable results stay completed, not dropped."""
    report = inspect_claim_eligibility(
        _bundle(
            parts, execution=_execution_ledger(parts.sealed.protocol_hash, capture_rate=capture_rate)
        )
    )
    assert report.eligible
    assert report.errors == ()


def test_an_exploratory_sample_size_condition_cannot_back_a_claim(parts: _Parts) -> None:
    report = inspect_claim_eligibility(
        _bundle(
            parts,
            execution=_execution_ledger(
                parts.sealed.protocol_hash, sample_size_status=SampleSizeStatus.EXPLORATORY
            ),
        )
    )
    assert report.error_codes == ("EXPLORATORY_CONDITION_SAMPLE_SIZE",)


# ---------------------------------------------------------------------------
# The passing path and the raising wrapper
# ---------------------------------------------------------------------------


def test_a_fully_satisfied_result_claim_is_eligible_with_no_errors(parts: _Parts) -> None:
    report = inspect_claim_eligibility(_bundle(parts))
    assert report.eligible
    assert report.errors == ()
    assert report.error_codes == ()
    assert report.scope is ClaimScope.RESULT
    assert report.status is ClaimStatus.SUPPORTED
    assert report.claim_hash == str(_result_claim().content_hash)
    assert parts.statistics.gate.superiority, "the superiority assertion must be truly satisfied"


def test_a_fully_satisfied_novelty_claim_is_eligible(parts: _Parts) -> None:
    report = inspect_claim_eligibility(_bundle(parts, _novelty_claim()))
    assert report.eligible
    assert report.errors == ()


def test_require_claim_eligibility_returns_the_report_when_eligible(parts: _Parts) -> None:
    report = require_claim_eligibility(_bundle(parts))
    assert isinstance(report, ClaimGateReport)
    assert report.eligible


def test_require_claim_eligibility_raises_carrying_the_whole_report(parts: _Parts) -> None:
    broken = _broken_bundle(parts, "unsealed_protocol")
    with pytest.raises(ClaimGateError) as excinfo:
        require_claim_eligibility(broken)
    error = excinfo.value
    assert error.code == "PROTOCOL_NOT_SEALED"
    assert not error.report.eligible
    assert error.report.error_codes == ("PROTOCOL_NOT_SEALED",)
    assert error.record.details["gate_report_hash"] == str(error.report.content_hash)


# ---------------------------------------------------------------------------
# Requirement 4.7: the gate never bypasses construction-time validation
# ---------------------------------------------------------------------------


def test_priority_language_still_fails_at_claim_construction() -> None:
    with pytest.raises(ResearchValidationError) as excinfo:
        ClaimRecord(
            claim_id="claim-novelty-priority",
            text="This is the 최초 study to evaluate pursuit on extracted OSM road networks.",
            scope=ClaimScope.NOVELTY,
            status=ClaimStatus.SUPPORTED,
            evidence_ids=("ev-capture",),
            comparison_axes=("map_source_realism",),
        )
    assert excinfo.value.code == "FORBIDDEN_PRIORITY_CLAIM"

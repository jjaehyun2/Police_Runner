"""Task 12.5 regressions for the claim-gated final paper release.

Exercises pursuit_evasion_rl/research/experiments/paper_release.py against a
real, tiny CPU-trained-and-evaluated condition (reusing
experiments.main_study's real pipeline for statistics and a sealed run
manifest): a fully reconciled release exports a real PaperArtifactBundle
with reproducible content hashes, and a missing dependency (an unreconciled
citation) fails closed with both the gate and export reports preserved.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pursuit_evasion_rl.osm_demo.coarsening import coarsen_raw_graph, prepare_model_network
from pursuit_evasion_rl.osm_demo.fixtures import daejeon
from pursuit_evasion_rl.research.budget import ConditionResourceEstimate, ResourceCeiling, SampleSizeStatus, decide_sample_size
from pursuit_evasion_rl.research.domain import AnalysisClassification, EvidenceRecord, EvidenceType, ExecutionStatus, MapScenario, ScreeningDecision
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.execution import ConditionExecutionLedger
from pursuit_evasion_rl.research.experiments.main_study import (
    MainStudyConditionSpec,
    run_main_study_condition,
)
from pursuit_evasion_rl.research.experiments.paper_release import (
    require_paper_release_admission,
    run_paper_release,
)
from pursuit_evasion_rl.research.maps.splits import DataHandle, HandleKind, SplitScope, TuningDataView
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
    NonSubstitutionStatement,
    PreFieldValidationCategory,
    PreFieldValidationEntry,
    PreFieldValidationReport,
    PreFieldValidationStatus,
    RiskCategory,
    RiskDisclosure,
    RiskStatement,
    ScopeEthicsBundle,
    ScopeSection,
    ScopeSectionKind,
    SimulationLimitationDisclosure,
)
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
from pursuit_evasion_rl.research.quality import GateAttestation, PropertyCoverageReport, PropertyFileCheck, REQUIRED_PROPERTY_IDS, SuiteRunReport
from pursuit_evasion_rl.research.runs.manifest import RunManifestStore, RunProvenance
from pursuit_evasion_rl.research.statistics.paired import BootstrapPlan, CorrectionMethod, EffectDirection, MissingDataPolicy, PracticalThreshold
from pursuit_evasion_rl.research.training.trainer import TrainerConfig
from pursuit_evasion_rl.research.variants.factory import full_condition_matrix
from pursuit_evasion_rl.research.variants.placement import PlacementCurriculumConfig, PlacementStyle

pytestmark = pytest.mark.offline

SIGNER = "principal-investigator"
UTC = "2026-08-01T00:00:00+00:00"
QUERY = "multi-agent reinforcement learning vehicle pursuit road network"
CITATION_KEY = "yang2023"
FUTURE_WORK_ARTIFACT = "cctv-fusion-demo"


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class LowestLegalActionPolicy:
    policy_id = "lowest-legal-action-v1"

    def act(self, observation) -> int:
        return min(index for index, legal in enumerate(observation.action_mask) if legal)


def _network():
    return prepare_model_network(coarsen_raw_graph(daejeon())).network


def _tuning_view() -> TuningDataView:
    return TuningDataView(
        train=(DataHandle(SplitScope.TRAIN, HandleKind.MAP, "release-train"),),
        validation=(DataHandle(SplitScope.VALIDATION, HandleKind.MAP, "release-validation"),),
    )


_SPECS = full_condition_matrix()[:1]
CONDITION_ID = _SPECS[0].condition.condition_id


def _metrics():
    return (
        MetricDeclaration(metric_id="capture_rate", symbol="P_cap", unit="probability", direction=MetricDirection.HIGHER_IS_BETTER, role=MetricRole.PRIMARY, formula="captured episodes divided by planned episodes"),
        metric_declaration("blocked_exit_fraction", MetricRole.PRIMARY),
    )


def _resource_spec() -> ResourceSpec:
    return ResourceSpec(
        ceiling=ResourceCeiling(resource_ceiling_id="ceiling-release", measured_at_utc="2026-07-01T00:00:00Z", accelerator_hours=10.0, wall_clock_hours=10.0),
        estimates=(ConditionResourceEstimate(condition_id=CONDITION_ID, accelerator_hours_per_seed=0.01, wall_clock_hours_per_seed=0.01, env_steps_per_seed=30),),
    )


def _sealed_protocol(tmp_path: Path):
    questions = default_research_questions()
    metrics = _metrics()
    resource = _resource_spec()
    analyses = tuple(
        PlannedAnalysis(analysis_id=f"A-{q.question_id}", question_id=q.question_id, classification=AnalysisClassification.CONFIRMATORY, estimand=f"paired difference in {q.primary_outcome}", metric_id=q.primary_outcome, comparison=q.directional_inequality)
        for q in questions
    )
    specifications = ProtocolSpecifications(
        hypotheses=tuple(HypothesisSpec(hypothesis_id=f"H-{q.question_id}", question_id=q.question_id, statement=q.directional_inequality, null_statement=f"no difference in {q.primary_outcome}") for q in questions),
        conditions=(ConditionRef(condition_id=CONDITION_ID, condition_hash=_hash("condition::default"), axis="baseline", arm="default"),),
        split=SplitSpec(
            metric_crs="EPSG:5186", buffer_m=50.0,
            polygon_hashes={"train": "1" * 64, "validation": "2" * 64, "test": "3" * 64},
            network_hashes={"train": "4" * 64, "validation": "5" * 64, "test": "6" * 64},
            cross_city_city_ids=("busan", "seoul"), split_protocol_hash="7" * 64, split_validation_report_hash="8" * 64,
            held_out_evaluation_rule="apply the validation-selected policy unchanged to the test split, once",
        ),
        sample=SampleSpec(plan=decide_sample_size(resource.ceiling, resource.estimates), selection_handles=(DataHandle(SplitScope.TRAIN, HandleKind.MAP, "release-train"), DataHandle(SplitScope.VALIDATION, HandleKind.MAP, "release-validation"))),
        resource=resource, metrics=metrics,
        statistics=StatisticsSpec(bootstrap=BootstrapPlan(), correction=CorrectionMethod.HOLM, primary_test="exact_mcnemar", multiplicity_family="the four pre-registered primary outcomes"),
        thresholds=practical_thresholds_for(questions, metrics),
        tolerance=ToleranceSpec(rtol=1e-5, atol=1e-7, applies_to=("resume_equivalence",), rationale="float32 accumulation differs across resume boundaries on the same device"),
        exclusion=ExclusionSpec(
            outcome_mapping={"valid": "use the observed episode outcome", "missing": "score the pre-registered worst outcome for both policies", "failed": "score the pre-registered worst outcome for both policies", "interrupted": "score the pre-registered worst outcome for both policies"},
            excludable_conditions=("simulator process crash reproduced twice",), primary_policy=MissingDataPolicy.PRE_REGISTERED_WORST_CASE, sensitivity_policy=MissingDataPolicy.COMPLETE_CASE,
            planned_case_accounting_rule="every planned episode case is reported as valid, missing, failed or interrupted",
        ),
        stop=StopSpec(rules=(StopRule(rule_id="validation-plateau", criterion="validation capture rate does not improve for 20 evaluations", action="halt the condition and record the reason", evaluated_on=SplitScope.VALIDATION),)),
        search=SearchSpec(search_date_utc="2026-07-01T00:00:00Z", query=QUERY, sources=("Scopus", "IEEE Xplore", "arXiv"), date_range_start="2015-01-01", date_range_end="2026-06-30", languages=("en", "ko"), inclusion_criteria=("multi-pursuer pursuit on a graph or road network",), exclusion_criteria=("continuous open-plane pursuit without a road network",)),
        analyses=analyses,
    )
    store = ProtocolStore(tmp_path / "protocols")
    draft = store.create(questions=questions, specifications=specifications)
    sealed = store.seal(draft.protocol_id, signer=SIGNER)
    analysis_event = store.register_analysis(sealed.protocol_id, analyses[0])
    return sealed, analysis_event


def _clean_attestation() -> GateAttestation:
    checks = tuple(PropertyFileCheck(property_id=pid, file_path=f"tests/research/properties/test_property_{pid}.py", has_offline_marker=True, max_examples=100) for pid in REQUIRED_PROPERTY_IDS)
    coverage = PropertyCoverageReport(checks=checks, deferred_property_ids=(18,))
    suite = SuiteRunReport(target="tests/research", total=1, failures=0, errors=0, skipped=0, skipped_test_ids=(), return_code=0, junit_xml_path="synthetic.xml")
    return GateAttestation(coverage=coverage, suite_runs=(suite,), code_hash_at_attestation="synthetic", generated_at_utc=UTC)


def _provenance() -> RunProvenance:
    return RunProvenance(code_hash="a" * 64, dirty_tree=False, dependency_hash="b" * 64, runtime="python3.11+torch2.8.0+cpu", device="cpu", map_hash="c" * 64, split="train", seed=5, input_hashes={"map": "c" * 64})


def _citation_ledger(*, verified: bool = True) -> CitationLedger:
    search_protocol = SearchProtocolSpec(
        search_date_utc=UTC, queries=(QUERY,),
        sources=(
            SearchSource(source_id="openalex", name="OpenAlex", kind=SearchSourceKind.ACADEMIC_INDEX, operator="OurResearch", endpoint="https://api.openalex.org/works"),
            SearchSource(source_id="scopus", name="Scopus", kind=SearchSourceKind.ACADEMIC_INDEX, operator="Elsevier", endpoint="https://api.elsevier.com/content/search/scopus"),
            SearchSource(source_id="arxiv", name="arXiv", kind=SearchSourceKind.PREPRINT_SERVER, operator="Cornell University", endpoint="https://export.arxiv.org/api/query"),
        ),
        period_start="2015-01-01", period_end="2026-07-31", languages=("en", "ko"),
        inclusion_criteria=("road-network pursuit or interception of a fleeing vehicle",), exclusion_criteria=("no evaluation on a road network",),
    )
    ledger = CitationLedger(search_protocol)
    work_id = "work::yang-progression-cognition"
    ledger.add_candidate(CitationCandidate(
        candidate_id=f"{CITATION_KEY}-c1", canonical_work_id=work_id, source_id="openalex", query=QUERY,
        executed_at_utc=UTC, result_rank=1, record_identifier=f"record::{CITATION_KEY}", result_content_hash=_hash(f"search-result::{CITATION_KEY}"),
    ))
    ledger.screen(f"{CITATION_KEY}-c1", ScreeningDecision.INCLUDED, "road-network pursuit with a learned policy", screened_at_utc=UTC)
    ledger.register_citation(CitationRecord(
        citation_key=CITATION_KEY, canonical_work_id=work_id, title="Progression Cognition Reinforcement Learning for Multi-Vehicle Pursuit",
        authors=("Yang, X.",), year=2023, candidate_ids=ledger.candidate_ids_for_work(work_id),
        proposal_cross_check=ProposalCrossCheck.NOT_FROM_PROPOSAL,
        metadata_cross_check=(MetadataCrossCheck(identifier="10.1007/s41109-024-00689-1", identifier_kind=PersistentIdentifierKind.DOI, metadata_source="Crossref", checked_at_utc=UTC, author_match=True, year_match=True, title_match=True) if verified else None),
        source_location=(SourceLocation(original_text_content_hash=_hash(CITATION_KEY), section="4 Experiments", table="Table 2") if verified else None),
    ))
    return ledger


def _scope() -> ScopeEthicsBundle:
    future_work = FutureWorkRegistry(
        primary_result_root="artifacts/primary", future_work_root="artifacts/future_work",
        artifacts=(FutureWorkArtifact(artifact_id=FUTURE_WORK_ARTIFACT, system=FutureWorkSystem.CCTV, description="illustrative CCTV demo, not evaluated", artifact_path="artifacts/future_work/cctv/demo.mp4"),),
    )
    return ScopeEthicsBundle(
        limitations=SimulationLimitationDisclosure(
            traffic_and_vehicle_dynamics="Vehicle dynamics and ambient traffic are not modelled.",
            sensor_error="Sensor position error and dropout are assumed zero.",
            communication_latency_and_loss="Communication latency and packet loss are not modelled.",
            human_behavior="Driver and commander human behaviour is not modelled.",
            legal_and_operational_constraints="Legal and operational constraints are not modelled.",
        ),
        non_substitution=NonSubstitutionStatement(),
        risks=RiskDisclosure(statements=tuple(RiskStatement(category=category, description=f"{category.value} risk statement") for category in RiskCategory)),
        pre_field_validation=PreFieldValidationReport(entries=tuple(PreFieldValidationEntry(category=category, validation_id=f"pfv-{category.value}", status=PreFieldValidationStatus.NOT_PERFORMED) for category in PreFieldValidationCategory)),
        future_work=future_work,
        sections=(
            ScopeSection(section_id="sec-scope-current", kind=ScopeSectionKind.CURRENT_RESEARCH_SCOPE, title="Current research scope", body="Limited to simulated multi-officer pursuit policy evaluation on road graphs."),
            ScopeSection(section_id="sec-scope-future", kind=ScopeSectionKind.FUTURE_WORK, title="Future work", body="The following systems are not implemented in this study and remain future work.", systems=tuple(FutureWorkSystem)),
        ),
    )


def _sections() -> dict:
    return {
        "title": "Cooperative containment for multi-officer pursuit on extracted road networks",
        "abstract": "We evaluate a cooperative containment policy on road graphs across five seeds.",
        "introduction": "Vehicle pursuit on a road network is a graph-structured coordination problem.",
        "related_work": "Prior graph-pursuit work is compared across eight differentiation axes.",
        "problem_definition": "Six officers pursue one goal-directed evader on a directed road graph.",
        "methods": "A masked multi-agent policy is optimized against the audited reward set.",
        "experiments": "Every planned condition is executed under one pre-registered sample size.",
        "results": "The paired capture-rate difference and its interval are reported per seed.",
        "discussion": "The measured advantage is bounded by what this simulation represents.",
        "validity_limitations": "Internal, external, construct and statistical conclusion validity threats are enumerated separately.",
        "ethics": "Automation bias, surveillance expansion and misuse risks are disclosed.",
        "reproducibility": "Seeds, content hashes and the sealed protocol are published with limits noted.",
        "conclusion": "Cooperative containment improves paired capture rate within this simulation.",
        "references": "Yang, X. (2023). Progression Cognition Reinforcement Learning.",
    }


def _real_run(tmp_path: Path, sealed, analysis_event):
    """A real tiny train+paired-evaluate condition, reused as the release's source data."""
    network = _network()
    spec = MainStudyConditionSpec(
        condition_id=CONDITION_ID, scenario=MapScenario.INTERIOR_CONTAINED, train_network=network, validation_network=network,
        tuning_data=_tuning_view(), config=TrainerConfig(updates=1, episodes_per_update=1, max_steps=10, hidden_dims=(4,)),
        placement_config=PlacementCurriculumConfig(style=PlacementStyle.GLOBAL),
        baseline_policy_factory=lambda _n: LowestLegalActionPolicy(), episodes_per_seed=3,
    )
    runs = RunManifestStore(tmp_path / "runs")

    class _TokenLike:
        protocol = sealed

    result = run_main_study_condition(
        _TokenLike(), spec, seeds=1, runs=runs, output_root=str(tmp_path / "artifacts"), provenance_factory=_provenance,
        bootstrap_plan=BootstrapPlan(n_bootstrap=200, rng_seed=11, rationale="release test"),
        threshold=PracticalThreshold(metric="capture_rate", unit="probability", minimum_effect=0.05, direction=EffectDirection.GREATER_IS_BETTER),
    )
    manifest = runs.read(result.seed_outcomes[0].run_id)
    ledger = ConditionExecutionLedger(
        protocol_hash=sealed.protocol_hash, sample_size_plan=decide_sample_size(_resource_spec().ceiling, _resource_spec().estimates),
        planned_condition_ids=(CONDITION_ID,), records=(result.execution_record,),
    )
    evidence = (
        EvidenceRecord(record_id="ev-capture", evidence_type=EvidenceType.EXPERIMENT_OBSERVATION, producer="paired-evaluator", created_at_utc=UTC, method="paired episode evaluation", source=f"{manifest.run_id} metrics", extracted_value=result.execution_record.result.value, verification="metrics artifact hash matches the sealed manifest"),
    )
    return manifest, result.statistics, ledger, evidence, analysis_event.analysis_id


def test_a_fully_reconciled_release_exports_a_real_reproducible_bundle(tmp_path: Path) -> None:
    sealed, analysis_event = _sealed_protocol(tmp_path)
    token = require_paper_release_admission(quality_attestation=_clean_attestation(), protocol=sealed)
    manifest, statistics, ledger, evidence, analysis_id = _real_run(tmp_path, sealed, analysis_event)

    report = run_paper_release(
        token, bundle_id="paper-osm-pursuit-release-test", run_manifests=(manifest,), statistics=statistics,
        execution_ledger=ledger, planned_conditions=_SPECS, citation_ledger=_citation_ledger(), citation_key=CITATION_KEY,
        evidence=evidence, scope=_scope(), sections=_sections(), condition_id=CONDITION_ID, analysis_id=analysis_id,
        config_hash=_hash("config::release-test"),
        # This sanity-scale real run (3 paired episodes) cannot itself reach
        # statistical/practical significance -- reporting it, not asserting
        # superiority over it, is the honest claim (Requirement 4.7).
        asserts_superiority=False,
    )

    assert report.eligible
    assert report.gate_report.eligible
    assert report.export_report.eligible
    assert report.bundle is not None
    assert report.bundle.tables[0].table_hash
    assert report.bundle.figures[0].figure_hash

    from pursuit_evasion_rl.research.paper.tables import verify_cell_replay
    from pursuit_evasion_rl.research.paper.figures import verify_panel_replay

    for cell in report.bundle.tables[0].cells:
        assert verify_cell_replay(cell) == ()
    for panel in report.bundle.figures[0].panels:
        assert verify_panel_replay(panel) == ()

    # Every cell/panel's own provenance sidecar is what "reproducible" means
    # here (Requirement 14.11-14.13): each recomputes its registered output
    # hash from its own recorded inputs, verified above.


def test_an_unverified_citation_fails_the_release_closed_with_both_reports_preserved(tmp_path: Path) -> None:
    sealed, analysis_event = _sealed_protocol(tmp_path)
    token = require_paper_release_admission(quality_attestation=_clean_attestation(), protocol=sealed)
    manifest, statistics, ledger, evidence, analysis_id = _real_run(tmp_path, sealed, analysis_event)

    report = run_paper_release(
        token, bundle_id="paper-osm-pursuit-release-broken", run_manifests=(manifest,), statistics=statistics,
        execution_ledger=ledger, planned_conditions=_SPECS, citation_ledger=_citation_ledger(verified=False), citation_key=CITATION_KEY,
        evidence=evidence, scope=_scope(), sections=_sections(), condition_id=CONDITION_ID, analysis_id=analysis_id,
        config_hash=_hash("config::release-broken"),
    )

    assert not report.eligible
    assert not report.gate_report.eligible
    assert "UNVERIFIED_CITATION" in report.gate_report.error_codes
    assert report.bundle is None

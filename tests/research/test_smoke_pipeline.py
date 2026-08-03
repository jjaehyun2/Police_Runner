"""Task 10.4: one offline, no-network run of the whole research pipeline, start to finish.

Protocol freeze -> analysis registration -> sealed run manifest -> paired
statistics -> condition execution ledger -> citation ledger -> claim gate ->
result table/figure -> condition matrix -> Paper_Artifact_Set export.  Every
object here is the real production type; nothing is mocked.  If any link in
this chain silently drifted out of sync with another (a renamed field, a
changed constructor signature, a gate that stopped composing correctly), this
is the one test whose failure says so before Task 12 ever spends real compute.
"""

from __future__ import annotations

import hashlib

import pytest

from pursuit_evasion_rl.research.budget import (
    ConditionResourceEstimate,
    ResourceCeiling,
    SampleSizeStatus,
    decide_sample_size,
)
from pursuit_evasion_rl.research.claims.gate import (
    ClaimDependencyBundle,
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
from pursuit_evasion_rl.research.execution import ConditionExecutionLedger, ConditionExecutionRecord, MeasuredResult
from pursuit_evasion_rl.research.maps.splits import DataHandle, HandleKind, SplitScope
from pursuit_evasion_rl.research.metrics.behavior import MetricDirection
from pursuit_evasion_rl.research.paper.bundle import (
    PAPER_SECTIONS,
    REQUIRED_NOTATION_KINDS,
    PaperArtifactBundle,
    PaperSection,
    SymbolDefinition,
    export_paper_bundle,
)
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
from pursuit_evasion_rl.research.paper.figures import (
    REQUIRED_PANEL_KINDS,
    FigurePanelKind,
    ResultFigure,
    TrajectoryCandidate,
    TrajectoryOutcome,
    build_figure_panel,
    freeze_trajectory_selection_rule,
    select_representative_trajectory,
)
from pursuit_evasion_rl.research.paper.related_work import RelatedWorkMatrix
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
from pursuit_evasion_rl.research.paper.tables import build_condition_matrix_table, build_result_table
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
from pursuit_evasion_rl.research.runs.manifest import RunManifestStore, RunProvenance
from pursuit_evasion_rl.research.statistics.paired import (
    BootstrapPlan,
    CorrectionMethod,
    EffectDirection,
    MissingDataPolicy,
    PairedCase,
    PracticalThreshold,
    compare_paired_binary,
)
from pursuit_evasion_rl.research.variants.factory import full_condition_matrix

pytestmark = [pytest.mark.smoke, pytest.mark.offline]

SIGNER = "principal-investigator"
UTC = "2026-07-31T00:00:00+00:00"
QUERY = "multi-agent reinforcement learning vehicle pursuit road network"
CITATION_KEY = "yang2023"
FUTURE_WORK_ARTIFACT = "cctv-fusion-demo"

#: The real default Condition from the variant factory, so the condition matrix
#: built at the end of the pipeline is describing an arm that actually exists in
#: the pre-registered matrix rather than a smoke-test-only stand-in.
_SPECS = full_condition_matrix()[:1]
CONDITION_ID = _SPECS[0].condition.condition_id


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def test_the_full_offline_pipeline_runs_start_to_finish_and_exports_a_paper_bundle(tmp_path) -> None:
    # --- protocol: draft -> seal -> confirmatory analysis registration -----
    questions = default_research_questions()
    metrics = (
        MetricDeclaration(
            metric_id="capture_rate", symbol="P_cap", unit="probability",
            direction=MetricDirection.HIGHER_IS_BETTER, role=MetricRole.PRIMARY,
            formula="captured episodes divided by planned episodes",
        ),
        metric_declaration("blocked_exit_fraction", MetricRole.PRIMARY),
    )
    resource = ResourceSpec(
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
    analyses = tuple(
        PlannedAnalysis(
            analysis_id=f"A-{question.question_id}", question_id=question.question_id,
            classification=AnalysisClassification.CONFIRMATORY,
            estimand=f"paired difference in {question.primary_outcome}",
            metric_id=question.primary_outcome, comparison=question.directional_inequality,
        )
        for question in questions
    )
    specifications = ProtocolSpecifications(
        hypotheses=tuple(
            HypothesisSpec(
                hypothesis_id=f"H-{question.question_id}", question_id=question.question_id,
                statement=question.directional_inequality, null_statement=f"no difference in {question.primary_outcome}",
            )
            for question in questions
        ),
        conditions=(ConditionRef(condition_id=CONDITION_ID, condition_hash=_hash("condition::default_interior"), axis="baseline", arm="default"),),
        split=SplitSpec(
            metric_crs="EPSG:5186", buffer_m=50.0,
            polygon_hashes={"train": "1" * 64, "validation": "2" * 64, "test": "3" * 64},
            network_hashes={"train": "4" * 64, "validation": "5" * 64, "test": "6" * 64},
            cross_city_city_ids=("busan", "seoul"), split_protocol_hash="7" * 64, split_validation_report_hash="8" * 64,
            held_out_evaluation_rule="apply the validation-selected policy unchanged to the test split, once",
        ),
        sample=SampleSpec(
            plan=decide_sample_size(resource.ceiling, resource.estimates),
            selection_handles=(DataHandle(SplitScope.TRAIN, HandleKind.MAP, "daejeon-train"), DataHandle(SplitScope.VALIDATION, HandleKind.MAP, "daejeon-validation")),
        ),
        resource=resource, metrics=metrics,
        statistics=StatisticsSpec(bootstrap=BootstrapPlan(), correction=CorrectionMethod.HOLM, primary_test="exact_mcnemar", multiplicity_family="the four pre-registered primary outcomes"),
        thresholds=practical_thresholds_for(questions, metrics),
        tolerance=ToleranceSpec(rtol=1e-5, atol=1e-7, applies_to=("resume_equivalence",), rationale="float32 accumulation differs across resume boundaries on the same device"),
        exclusion=ExclusionSpec(
            outcome_mapping={
                "valid": "use the observed episode outcome",
                "missing": "score the pre-registered worst outcome for both policies",
                "failed": "score the pre-registered worst outcome for both policies",
                "interrupted": "score the pre-registered worst outcome for both policies",
            },
            excludable_conditions=("simulator process crash reproduced twice",),
            primary_policy=MissingDataPolicy.PRE_REGISTERED_WORST_CASE, sensitivity_policy=MissingDataPolicy.COMPLETE_CASE,
            planned_case_accounting_rule="every planned episode case is reported as valid, missing, failed or interrupted",
        ),
        stop=StopSpec(rules=(StopRule(rule_id="validation-plateau", criterion="validation capture rate does not improve for 20 evaluations", action="halt the condition and record the reason", evaluated_on=SplitScope.VALIDATION),)),
        search=SearchSpec(
            search_date_utc="2026-07-01T00:00:00Z", query=QUERY, sources=("Scopus", "IEEE Xplore", "arXiv"),
            date_range_start="2015-01-01", date_range_end="2026-06-30", languages=("en", "ko"),
            inclusion_criteria=("multi-pursuer pursuit on a graph or road network",), exclusion_criteria=("continuous open-plane pursuit without a road network",),
        ),
        analyses=analyses,
    )
    protocols = ProtocolStore(tmp_path / "protocols")
    draft = protocols.create(questions=questions, specifications=specifications)
    sealed = protocols.seal(draft.protocol_id, signer=SIGNER)
    analysis_event = protocols.register_analysis(sealed.protocol_id, analyses[0])
    assert analysis_event.is_confirmatory

    # --- run manifest: draft -> seal ---------------------------------------
    runs = RunManifestStore(tmp_path / "runs")
    provenance = RunProvenance(
        code_hash=_hash("code"), dirty_tree=False, dependency_hash=_hash("deps"),
        runtime="python3.11+torch2.8.0+cpu", device="cpu", map_hash=_hash("daejeon-train"),
        split="train", seed=7, input_hashes={"map": _hash("daejeon-train")},
    )
    runs.create(protocol_hash=sealed.protocol_hash, condition_hash=_hash("condition::default_interior"), provenance=provenance, run_id="run-primary")
    manifest = runs.seal(
        "run-primary", execution_status=ExecutionStatus.COMPLETED, status_reason="all planned seeds finished",
        artifact_hashes={"metrics": _hash("metrics::run-primary")}, result={"capture_rate": 0.62},
    )
    assert manifest.is_sealed

    # --- paired statistics ---------------------------------------------------
    cases = []
    for seed in range(5):
        for index in range(100):
            outcomes = (True, True) if index < 70 else ((True, False) if index < 90 else (False, False))
            cases.append(PairedCase(case_id=f"case-{seed}-{index}", training_seed=seed, outcome_a=outcomes[0], outcome_b=outcomes[1]))
    statistics = compare_paired_binary(
        tuple(cases), comparison_id="cooperative-vs-independent", metric="capture_rate", unit="probability",
        plan=BootstrapPlan(n_bootstrap=200, rng_seed=11, rationale="smoke-test runtime ceiling"),
        threshold=PracticalThreshold(metric="capture_rate", unit="probability", minimum_effect=0.05, direction=EffectDirection.GREATER_IS_BETTER),
    )
    assert statistics.confirmatory_eligible
    assert statistics.gate is not None and statistics.gate.superiority

    # --- condition execution ledger ------------------------------------------
    execution = ConditionExecutionLedger(
        protocol_hash=sealed.protocol_hash, sample_size_plan=decide_sample_size(resource.ceiling, resource.estimates),
        planned_condition_ids=(CONDITION_ID,),
        records=(
            ConditionExecutionRecord(
                condition_id=CONDITION_ID, protocol_hash=sealed.protocol_hash, execution_status=ExecutionStatus.COMPLETED,
                status_reason="all planned seeds finished", sample_size_status=SampleSizeStatus.DEFAULT,
                artifact_hashes={"metrics": _hash("metrics::run-primary")}, result=MeasuredResult(value=0.62), run_ids=("run-primary",),
            ),
        ),
    )
    assert execution.conserved

    # --- citations -------------------------------------------------------------
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
    citation_ledger = CitationLedger(search_protocol)
    work_id = "work::yang-progression-cognition"
    citation_ledger.add_candidate(CitationCandidate(
        candidate_id=f"{CITATION_KEY}-c1", canonical_work_id=work_id, source_id="openalex", query=QUERY,
        executed_at_utc=UTC, result_rank=1, record_identifier=f"record::{CITATION_KEY}", result_content_hash=_hash(f"search-result::{CITATION_KEY}"),
    ))
    citation_ledger.screen(f"{CITATION_KEY}-c1", ScreeningDecision.INCLUDED, "road-network pursuit with a learned policy", screened_at_utc=UTC)
    citation_ledger.register_citation(CitationRecord(
        citation_key=CITATION_KEY, canonical_work_id=work_id, title="Progression Cognition Reinforcement Learning for Multi-Vehicle Pursuit",
        authors=("Yang, X.",), year=2023, candidate_ids=citation_ledger.candidate_ids_for_work(work_id),
        proposal_cross_check=ProposalCrossCheck.NOT_FROM_PROPOSAL,
        metadata_cross_check=MetadataCrossCheck(identifier="10.1007/s41109-024-00689-1", identifier_kind=PersistentIdentifierKind.DOI, metadata_source="Crossref", checked_at_utc=UTC, author_match=True, year_match=True, title_match=True),
        source_location=SourceLocation(original_text_content_hash=_hash(CITATION_KEY), section="4 Experiments", table="Table 2"),
    ))

    # --- future-work isolation ---------------------------------------------
    future_work = FutureWorkRegistry(
        primary_result_root="artifacts/primary", future_work_root="artifacts/future_work",
        artifacts=(FutureWorkArtifact(artifact_id=FUTURE_WORK_ARTIFACT, system=FutureWorkSystem.CCTV, description="illustrative CCTV demo, not evaluated", artifact_path="artifacts/future_work/cctv/demo.mp4"),),
    )

    # --- claim gate ---------------------------------------------------------
    claim = ClaimRecord(
        claim_id="claim-rq2-containment", text="Cooperative containment raises the paired capture rate.",
        scope=ClaimScope.RESULT, status=ClaimStatus.SUPPORTED, evidence_ids=("ev-capture",),
        analysis_ids=(analyses[0].analysis_id,), citation_ids=(CITATION_KEY,),
    )
    bundle = ClaimDependencyBundle(
        claim=claim, protocol=sealed, analysis_event=analysis_event,
        selection_contexts=(SelectionContext(context_id="ctx-checkpoint-selection", purpose=SelectionPurpose.CHECKPOINT_SELECTION, handles=specifications.sample.selection_handles),),
        evidence=(EvidenceRecord(record_id="ev-capture", evidence_type=EvidenceType.EXPERIMENT_OBSERVATION, producer="paired-evaluator", created_at_utc=UTC, method="paired episode evaluation over five training seeds", source="run-primary metrics", extracted_value=0.62, verification="metrics artifact hash matches the sealed manifest"),),
        citation_ledger=citation_ledger, body_citation_keys=(CITATION_KEY,), run_manifests=(manifest,),
        source_hashes=reconciled_source_hashes((manifest,)), statistics=statistics, asserts_superiority=True,
        execution=execution, condition_ids=(CONDITION_ID,), future_work=future_work,
    )
    gate = require_claim_eligibility(bundle)
    assert gate.eligible

    # --- result table + figure -----------------------------------------------
    table = build_result_table(
        table_id="table-1-capture-rate", caption="Paired capture rate, proposed versus independent pursuit.",
        result=statistics, gate=gate, run_id="run-primary", config_hash=_hash("config::smoke"),
        input_hashes={"metrics": _hash("metrics::run-primary")},
    )
    selection_rule = freeze_trajectory_selection_rule(
        sealed, rule_id="rule-representative-trajectory", metric="time_to_capture_s",
        tie_break=("containment_area_m2", "total_distance_m"), higher_is_better=False, declared_at_utc=UTC,
    )

    def _candidates(prefix: str) -> tuple[TrajectoryCandidate, ...]:
        return tuple(
            TrajectoryCandidate(
                trajectory_id=f"{prefix}-{index}", run_id="run-primary",
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
        panels.append(
            build_figure_panel(
                figure_id="figure-2-results", panel_key=kind.value, kind=kind, payload=payload,
                gate=gate, run_id="run-primary", config_hash=_hash("config::smoke"),
                input_hashes={"metrics": _hash("metrics::run-primary")}, selection=selection,
            )
        )
    panels = tuple(panels)
    figure = ResultFigure(figure_id="figure-2-results", caption="Split, learning, seed variation, behavior and representative trajectories.", panels=panels)

    # --- condition matrix -----------------------------------------------------
    specs = _SPECS
    condition_matrix = build_condition_matrix_table(table_id="table-2-condition-matrix", caption="Execution status of the default condition.", specs=specs, ledger=execution)

    # --- scope/ethics bundle ---------------------------------------------------
    scope = ScopeEthicsBundle(
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

    # --- paper bundle export ---------------------------------------------------
    body = {
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
    sections = tuple(PaperSection(section_id=section_id, title=section_id.replace("_", " ").title(), body=body[section_id]) for section_id in PAPER_SECTIONS)
    related_work = RelatedWorkMatrix(citation_ledger)
    related_work.add_row(CITATION_KEY)
    notation = tuple(SymbolDefinition(kind=kind, symbol=f"s_{kind.value}", definition=f"definition of {kind.value}", equation=f"{kind.value} = f(x)") for kind in REQUIRED_NOTATION_KINDS)

    paper = PaperArtifactBundle(
        bundle_id="paper-osm-pursuit-smoke", sections=sections, notation=notation, protocol=sealed,
        related_work=related_work, citation_ledger=citation_ledger, body_citation_keys=(CITATION_KEY,),
        tables=(table,), figures=(figure,), condition_matrix=condition_matrix, planned_conditions=specs,
        run_manifests=(manifest,), scope=scope,
    )
    report = export_paper_bundle(paper)
    assert report.eligible
    assert report.errors == ()

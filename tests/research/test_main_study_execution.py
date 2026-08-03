"""Task 12.2 regressions for the multi-seed main-study orchestrator.

Exercises pursuit_evasion_rl/research/experiments/main_study.py against a
real, tiny CPU trainer and a real paired-evaluation replay: admission is
refused ahead of a completed pilot, and once admitted, every seed trains,
its validation-selected (not merely final) checkpoint is paired-evaluated
against a frozen baseline over identical sealed EpisodeCases, and the
resulting binary outcomes drive a real compare_paired_binary call whose
result lands in a conserved ConditionExecutionLedger.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pursuit_evasion_rl.osm_demo.coarsening import coarsen_raw_graph, prepare_model_network
from pursuit_evasion_rl.osm_demo.fixtures import daejeon
from pursuit_evasion_rl.research.budget import ConditionResourceEstimate, ResourceCeiling, SampleSizePlan, SampleSizeStatus, decide_sample_size
from pursuit_evasion_rl.research.domain import AnalysisClassification, ExecutionStatus, MapScenario
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.experiments.main_study import (
    LEGACY_BEST_V2_PATH,
    BestV2Role,
    MainStudyAdmissionToken,
    MainStudyConditionSpec,
    require_main_study_admission,
    run_main_study,
    train_and_evaluate_seed,
)
from pursuit_evasion_rl.research.experiments.pilot import PilotConditionOutcome, PilotReport
from pursuit_evasion_rl.research.maps.splits import DataHandle, HandleKind, SplitScope, TuningDataView
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
from pursuit_evasion_rl.research.quality import GateAttestation, PropertyCoverageReport, PropertyFileCheck, REQUIRED_PROPERTY_IDS, SuiteRunReport
from pursuit_evasion_rl.research.runs.manifest import RunManifestStore, RunProvenance
from pursuit_evasion_rl.research.statistics.paired import BootstrapPlan, CorrectionMethod, EffectDirection, MissingDataPolicy, PracticalThreshold
from pursuit_evasion_rl.research.training.trainer import TrainerConfig
from pursuit_evasion_rl.research.variants.placement import PlacementCurriculumConfig, PlacementStyle

pytestmark = pytest.mark.offline

SIGNER = "principal-investigator"
CONDITION_ID = "main_study_sanity_interior"


class LowestLegalActionPolicy:
    """Deterministic FrozenPolicy stand-in, matching the reference used by test_paired_evaluator.py."""

    policy_id = "lowest-legal-action-v1"

    def act(self, observation) -> int:
        return min(index for index, legal in enumerate(observation.action_mask) if legal)


def _baseline_factory(_network):
    return LowestLegalActionPolicy()


def _tiny_network():
    return prepare_model_network(coarsen_raw_graph(daejeon())).network


def _tiny_config() -> TrainerConfig:
    return TrainerConfig(updates=2, episodes_per_update=1, max_steps=15, hidden_dims=(4,))


def _tuning_view() -> TuningDataView:
    return TuningDataView(
        train=(DataHandle(SplitScope.TRAIN, HandleKind.MAP, "main-study-train"),),
        validation=(DataHandle(SplitScope.VALIDATION, HandleKind.MAP, "main-study-validation"),),
    )


def _metrics():
    return (
        MetricDeclaration(
            metric_id="capture_rate", symbol="P_cap", unit="probability",
            direction=MetricDirection.HIGHER_IS_BETTER, role=MetricRole.PRIMARY,
            formula="captured episodes divided by planned episodes",
        ),
        metric_declaration("blocked_exit_fraction", MetricRole.PRIMARY),
    )


def _resource_spec() -> ResourceSpec:
    return ResourceSpec(
        ceiling=ResourceCeiling(resource_ceiling_id="ceiling-main-study", measured_at_utc="2026-07-01T00:00:00Z", accelerator_hours=10.0, wall_clock_hours=10.0),
        estimates=(ConditionResourceEstimate(condition_id=CONDITION_ID, accelerator_hours_per_seed=0.01, wall_clock_hours_per_seed=0.01, env_steps_per_seed=30),),
    )


def _sealed_protocol(tmp_path: Path):
    questions = default_research_questions()
    metrics = _metrics()
    resource = _resource_spec()
    analyses = tuple(
        PlannedAnalysis(
            analysis_id=f"A-{q.question_id}", question_id=q.question_id, classification=AnalysisClassification.CONFIRMATORY,
            estimand=f"paired difference in {q.primary_outcome}", metric_id=q.primary_outcome, comparison=q.directional_inequality,
        )
        for q in questions
    )
    specifications = ProtocolSpecifications(
        hypotheses=tuple(HypothesisSpec(hypothesis_id=f"H-{q.question_id}", question_id=q.question_id, statement=q.directional_inequality, null_statement=f"no difference in {q.primary_outcome}") for q in questions),
        conditions=(ConditionRef(condition_id=CONDITION_ID, condition_hash="9" * 64, axis="baseline", arm="default"),),
        split=SplitSpec(
            metric_crs="EPSG:5186", buffer_m=50.0,
            polygon_hashes={"train": "1" * 64, "validation": "2" * 64, "test": "3" * 64},
            network_hashes={"train": "4" * 64, "validation": "5" * 64, "test": "6" * 64},
            cross_city_city_ids=("busan", "seoul"), split_protocol_hash="7" * 64, split_validation_report_hash="8" * 64,
            held_out_evaluation_rule="apply the validation-selected policy unchanged to the test split, once",
        ),
        sample=SampleSpec(plan=decide_sample_size(resource.ceiling, resource.estimates), selection_handles=(DataHandle(SplitScope.TRAIN, HandleKind.MAP, "main-study-train"), DataHandle(SplitScope.VALIDATION, HandleKind.MAP, "main-study-validation"))),
        resource=resource, metrics=metrics,
        statistics=StatisticsSpec(bootstrap=BootstrapPlan(), correction=CorrectionMethod.HOLM, primary_test="exact_mcnemar", multiplicity_family="the four pre-registered primary outcomes"),
        thresholds=practical_thresholds_for(questions, metrics),
        tolerance=ToleranceSpec(rtol=1e-5, atol=1e-7, applies_to=("resume_equivalence",), rationale="float32 accumulation differs across resume boundaries on the same device"),
        exclusion=ExclusionSpec(
            outcome_mapping={"valid": "use the observed episode outcome", "missing": "score the pre-registered worst outcome for both policies", "failed": "score the pre-registered worst outcome for both policies", "interrupted": "score the pre-registered worst outcome for both policies"},
            excludable_conditions=("simulator process crash reproduced twice",),
            primary_policy=MissingDataPolicy.PRE_REGISTERED_WORST_CASE, sensitivity_policy=MissingDataPolicy.COMPLETE_CASE,
            planned_case_accounting_rule="every planned episode case is reported as valid, missing, failed or interrupted",
        ),
        stop=StopSpec(rules=(StopRule(rule_id="validation-plateau", criterion="validation capture rate does not improve for 20 evaluations", action="halt the condition and record the reason", evaluated_on=SplitScope.VALIDATION),)),
        search=SearchSpec(search_date_utc="2026-07-01T00:00:00Z", query="multi-agent reinforcement learning vehicle pursuit road network", sources=("Scopus", "IEEE Xplore", "arXiv"), date_range_start="2015-01-01", date_range_end="2026-06-30", languages=("en", "ko"), inclusion_criteria=("multi-pursuer pursuit on a graph or road network",), exclusion_criteria=("continuous open-plane pursuit without a road network",)),
        analyses=analyses,
    )
    store = ProtocolStore(tmp_path / "protocols")
    draft = store.create(questions=questions, specifications=specifications)
    return store.seal(draft.protocol_id, signer=SIGNER), resource


def _clean_attestation() -> GateAttestation:
    checks = tuple(
        PropertyFileCheck(property_id=pid, file_path=f"tests/research/properties/test_property_{pid}.py", has_offline_marker=True, max_examples=100)
        for pid in REQUIRED_PROPERTY_IDS
    )
    coverage = PropertyCoverageReport(checks=checks, deferred_property_ids=(18,))
    suite = SuiteRunReport(target="tests/research", total=1, failures=0, errors=0, skipped=0, skipped_test_ids=(), return_code=0, junit_xml_path="synthetic.xml")
    return GateAttestation(coverage=coverage, suite_runs=(suite,), code_hash_at_attestation="synthetic", generated_at_utc="2026-07-31T00:00:00Z")


def _completed_pilot_report() -> PilotReport:
    return PilotReport(
        admission_hash="a" * 64,
        outcomes=(PilotConditionOutcome(condition_id="pilot_sanity_interior", execution_status=ExecutionStatus.COMPLETED, status_reason="pilot ran fine", run_id="pilot-run-1"),),
    )


def _incomplete_pilot_report() -> PilotReport:
    return PilotReport(
        admission_hash="a" * 64,
        outcomes=(PilotConditionOutcome(condition_id="pilot_sanity_interior", execution_status=ExecutionStatus.FAILED, status_reason="pilot crashed", run_id="pilot-run-1"),),
    )


def _provenance() -> RunProvenance:
    return RunProvenance(code_hash="a" * 64, dirty_tree=False, dependency_hash="b" * 64, runtime="python3.11+torch2.8.0+cpu", device="cpu", map_hash="c" * 64, split="train", seed=5, input_hashes={"map": "c" * 64})


def _spec(*, episodes_per_seed: int = 2) -> MainStudyConditionSpec:
    network = _tiny_network()
    return MainStudyConditionSpec(
        condition_id=CONDITION_ID, scenario=MapScenario.INTERIOR_CONTAINED, train_network=network, validation_network=network,
        tuning_data=_tuning_view(), config=_tiny_config(), placement_config=PlacementCurriculumConfig(style=PlacementStyle.GLOBAL),
        baseline_policy_factory=_baseline_factory, episodes_per_seed=episodes_per_seed,
    )


def test_admission_requires_a_fully_completed_pilot_report(tmp_path: Path) -> None:
    sealed, _resource = _sealed_protocol(tmp_path)
    attestation = _clean_attestation()

    with pytest.raises(ResearchValidationError) as excinfo:
        require_main_study_admission(quality_attestation=attestation, protocol=sealed, pilot_report=_incomplete_pilot_report())
    assert excinfo.value.code == "PILOT_NOT_COMPLETED"

    token = require_main_study_admission(quality_attestation=attestation, protocol=sealed, pilot_report=_completed_pilot_report())
    assert isinstance(token, MainStudyAdmissionToken)


def test_a_real_seed_trains_and_paired_evaluates_against_a_frozen_baseline(tmp_path: Path) -> None:
    sealed, _resource = _sealed_protocol(tmp_path)
    token = require_main_study_admission(quality_attestation=_clean_attestation(), protocol=sealed, pilot_report=_completed_pilot_report())
    spec = _spec()
    runs = RunManifestStore(tmp_path / "runs")

    outcome = train_and_evaluate_seed(token, spec, 11, runs=runs, output_root=str(tmp_path / "artifacts"), provenance_factory=_provenance)

    assert outcome.execution_status is ExecutionStatus.COMPLETED
    assert len(outcome.paired_cases) == spec.episodes_per_seed
    assert all(case.training_seed == 11 for case in outcome.paired_cases)
    manifest = runs.read(outcome.run_id)
    assert manifest.is_sealed and manifest.run.execution_status is ExecutionStatus.COMPLETED


def test_run_main_study_produces_a_conserved_ledger_and_real_paired_statistics(tmp_path: Path) -> None:
    sealed, resource = _sealed_protocol(tmp_path)
    token = require_main_study_admission(quality_attestation=_clean_attestation(), protocol=sealed, pilot_report=_completed_pilot_report())
    spec = _spec(episodes_per_seed=3)
    runs = RunManifestStore(tmp_path / "runs")
    # Keep the real proof small: an explicit exploratory-status plan at 2
    # seeds, matching Task 12.1's pilot-scale discipline rather than the
    # pre-registered 5/500 default (which decide_sample_size would otherwise
    # select given this test's generous synthetic ceiling).
    small_plan = SampleSizePlan(
        status=SampleSizeStatus.EXPLORATORY, seeds_per_condition=2, episodes_per_condition=2,
        resource_ceiling_id=resource.ceiling.resource_ceiling_id,
        reduction_inputs={"reason": "bounded sanity-scale proof run"},
        power_limitation="explicitly reduced below the confirmatory floor for a bounded sanity-scale proof run",
    )

    report = run_main_study(
        token, (spec,), sample_size_plan=small_plan, runs=runs, output_root=str(tmp_path / "artifacts"),
        provenance_factory=_provenance, bootstrap_plan=BootstrapPlan(n_bootstrap=200, rng_seed=11, rationale="test runtime ceiling"),
        threshold=PracticalThreshold(metric="capture_rate", unit="probability", minimum_effect=0.05, direction=EffectDirection.GREATER_IS_BETTER),
    )

    assert report.ledger.conserved
    assert report.ledger.planned_count == 1
    assert len(report.condition_results) == 1
    result = report.condition_results[0]
    assert result.execution_record.execution_status is ExecutionStatus.COMPLETED
    assert result.execution_record.is_measured
    assert len(result.seed_outcomes) == 2
    assert result.statistics is not None
    assert result.statistics.primary.n_pairs == 2 * spec.episodes_per_seed
    assert result.statistics.primary.n_seeds == 2


def test_a_training_failure_in_one_seed_does_not_abort_the_condition(tmp_path: Path) -> None:
    """A pre-occupied run directory forces a real ResearchTrainer failure for one seed only."""
    sealed, resource = _sealed_protocol(tmp_path)
    token = require_main_study_admission(quality_attestation=_clean_attestation(), protocol=sealed, pilot_report=_completed_pilot_report())
    spec = _spec(episodes_per_seed=2)
    output_root = tmp_path / "artifacts"

    class _FixedRunManifestStore(RunManifestStore):
        def create(self, **kwargs):
            kwargs["run_id"] = "collision-run"
            return super().create(**kwargs)

    (output_root / CONDITION_ID / "0" / "collision-run").mkdir(parents=True)
    runs = _FixedRunManifestStore(tmp_path / "runs")

    outcome = train_and_evaluate_seed(token, spec, 0, runs=runs, output_root=str(output_root), provenance_factory=_provenance)

    assert outcome.execution_status is ExecutionStatus.FAILED
    assert outcome.error_artifact_hash is not None
    assert outcome.paired_cases == ()

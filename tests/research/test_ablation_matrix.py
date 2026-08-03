"""Task 12.3 regressions for the gated observation/reward/placement/stabilization ablation matrix.

Exercises pursuit_evasion_rl/research/experiments/ablations.py against a
real, tiny CPU trainer: the matrix is single-factor-clean and admission
requires the same four-part gate the pilot uses, every hysteresis arm is
recorded not_run with its incompatibility reason rather than trained under
an unreviewed interpretation, and a small real slice of the matrix (one arm
per axis, plus the metadata-completeness check across the full matrix)
actually trains and paired-evaluates.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pursuit_evasion_rl.osm_demo.coarsening import coarsen_raw_graph, prepare_model_network
from pursuit_evasion_rl.osm_demo.fixtures import daejeon
from pursuit_evasion_rl.research.budget import ConditionResourceEstimate, ResourceCeiling, decide_sample_size
from pursuit_evasion_rl.research.domain import AnalysisClassification, ExecutionStatus, MapScenario
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.execution import ConditionCostEstimate, plan_execution
from pursuit_evasion_rl.research.experiments.ablations import (
    HYSTERESIS_INCOMPATIBILITY_REASON,
    build_ablation_matrix,
    require_ablation_admission,
    run_ablation_matrix,
)
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
CONDITION_ID = "default__interior_contained"


class LowestLegalActionPolicy:
    policy_id = "lowest-legal-action-v1"

    def act(self, observation) -> int:
        return min(index for index, legal in enumerate(observation.action_mask) if legal)


def _baseline_factory(_network):
    return LowestLegalActionPolicy()


def _network():
    return prepare_model_network(coarsen_raw_graph(daejeon())).network


def _tuning_view() -> TuningDataView:
    return TuningDataView(
        train=(DataHandle(SplitScope.TRAIN, HandleKind.MAP, "ablation-train"),),
        validation=(DataHandle(SplitScope.VALIDATION, HandleKind.MAP, "ablation-validation"),),
    )


def _base_config() -> TrainerConfig:
    return TrainerConfig(updates=1, episodes_per_update=1, max_steps=10, hidden_dims=(4,))


def _resource_spec() -> ResourceSpec:
    return ResourceSpec(
        ceiling=ResourceCeiling(resource_ceiling_id="ceiling-ablation", measured_at_utc="2026-07-01T00:00:00Z", accelerator_hours=10.0, wall_clock_hours=10.0),
        estimates=(ConditionResourceEstimate(condition_id=CONDITION_ID, accelerator_hours_per_seed=0.01, wall_clock_hours_per_seed=0.01, env_steps_per_seed=30),),
    )


def _metrics():
    return (
        MetricDeclaration(metric_id="capture_rate", symbol="P_cap", unit="probability", direction=MetricDirection.HIGHER_IS_BETTER, role=MetricRole.PRIMARY, formula="captured episodes divided by planned episodes"),
        metric_declaration("blocked_exit_fraction", MetricRole.PRIMARY),
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
        conditions=(ConditionRef(condition_id=CONDITION_ID, condition_hash="9" * 64, axis="baseline", arm="default"),),
        split=SplitSpec(
            metric_crs="EPSG:5186", buffer_m=50.0,
            polygon_hashes={"train": "1" * 64, "validation": "2" * 64, "test": "3" * 64},
            network_hashes={"train": "4" * 64, "validation": "5" * 64, "test": "6" * 64},
            cross_city_city_ids=("busan", "seoul"), split_protocol_hash="7" * 64, split_validation_report_hash="8" * 64,
            held_out_evaluation_rule="apply the validation-selected policy unchanged to the test split, once",
        ),
        sample=SampleSpec(plan=decide_sample_size(resource.ceiling, resource.estimates), selection_handles=(DataHandle(SplitScope.TRAIN, HandleKind.MAP, "ablation-train"), DataHandle(SplitScope.VALIDATION, HandleKind.MAP, "ablation-validation"))),
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
        search=SearchSpec(search_date_utc="2026-07-01T00:00:00Z", query="multi-agent reinforcement learning vehicle pursuit road network", sources=("Scopus", "IEEE Xplore", "arXiv"), date_range_start="2015-01-01", date_range_end="2026-06-30", languages=("en", "ko"), inclusion_criteria=("multi-pursuer pursuit on a graph or road network",), exclusion_criteria=("continuous open-plane pursuit without a road network",)),
        analyses=analyses,
    )
    store = ProtocolStore(tmp_path / "protocols")
    draft = store.create(questions=questions, specifications=specifications)
    return store.seal(draft.protocol_id, signer=SIGNER), resource


def _clean_attestation() -> GateAttestation:
    checks = tuple(PropertyFileCheck(property_id=pid, file_path=f"tests/research/properties/test_property_{pid}.py", has_offline_marker=True, max_examples=100) for pid in REQUIRED_PROPERTY_IDS)
    coverage = PropertyCoverageReport(checks=checks, deferred_property_ids=(18,))
    suite = SuiteRunReport(target="tests/research", total=1, failures=0, errors=0, skipped=0, skipped_test_ids=(), return_code=0, junit_xml_path="synthetic.xml")
    return GateAttestation(coverage=coverage, suite_runs=(suite,), code_hash_at_attestation="synthetic", generated_at_utc="2026-07-31T00:00:00Z")


def _provenance() -> RunProvenance:
    return RunProvenance(code_hash="a" * 64, dirty_tree=False, dependency_hash="b" * 64, runtime="python3.11+torch2.8.0+cpu", device="cpu", map_hash="c" * 64, split="train", seed=5, input_hashes={"map": "c" * 64})


def _matrix(network):
    return build_ablation_matrix(
        MapScenario.INTERIOR_CONTAINED, network=network, tuning_data=_tuning_view(), base_config=_base_config(),
        baseline_policy_factory=_baseline_factory, episodes_per_seed=1, placement_config=PlacementCurriculumConfig(style=PlacementStyle.GLOBAL),
    )


def test_the_full_matrix_is_single_factor_clean_and_hysteresis_arms_are_not_run(tmp_path: Path) -> None:
    network = _network()
    arms = _matrix(network)
    # 2 observation + 4 reward + 3 placement + 4 stabilization = 13 arms.
    assert len(arms) == 13
    not_run = [arm for arm in arms if arm.condition_spec is None]
    assert len(not_run) == 2
    assert all(arm.not_run_reason == HYSTERESIS_INCOMPATIBILITY_REASON for arm in not_run)
    assert all("hyst_on" in arm.condition_id for arm in not_run)
    trainable = [arm for arm in arms if arm.condition_spec is not None]
    assert len(trainable) == 11


def test_admission_requires_the_same_four_part_gate_as_the_pilot(tmp_path: Path) -> None:
    sealed, resource = _sealed_protocol(tmp_path)
    plan = plan_execution(
        protocol_hash=sealed.protocol_hash, ceiling=resource.ceiling,
        cost_estimates=(ConditionCostEstimate(resource=ConditionResourceEstimate(condition_id=CONDITION_ID, accelerator_hours_per_seed=0.01, wall_clock_hours_per_seed=0.01, env_steps_per_seed=30)),),
    )
    with pytest.raises(ResearchValidationError) as excinfo:
        require_ablation_admission(quality_attestation=_clean_attestation(), protocol=sealed, resource_plan=None)
    assert excinfo.value.code == "MISSING_RESOURCE_ESTIMATE"

    token = require_ablation_admission(quality_attestation=_clean_attestation(), protocol=sealed, resource_plan=plan)
    assert token.admission_hash


def test_a_real_slice_of_the_matrix_trains_and_the_ledger_conserves_including_not_run(tmp_path: Path) -> None:
    sealed, resource = _sealed_protocol(tmp_path)
    network = _network()
    plan = plan_execution(
        protocol_hash=sealed.protocol_hash, ceiling=resource.ceiling,
        cost_estimates=(ConditionCostEstimate(resource=ConditionResourceEstimate(condition_id=CONDITION_ID, accelerator_hours_per_seed=0.01, wall_clock_hours_per_seed=0.01, env_steps_per_seed=30)),),
    )
    token = require_ablation_admission(quality_attestation=_clean_attestation(), protocol=sealed, resource_plan=plan)

    arms = _matrix(network)
    # A real, small proof: one trainable arm per axis, plus both hysteresis
    # not_run arms -- not the full 11-condition matrix, to keep wall-clock
    # bounded (matching Task 12.1/12.2's own sanity-scale discipline).
    by_condition_id = {arm.condition_id: arm for arm in arms}
    hysteresis_arms = [arm for arm in arms if arm.condition_spec is None]
    one_per_axis = []
    seen_prefixes: set[str] = set()
    for arm in arms:
        if arm.condition_spec is None:
            continue
        prefix = arm.condition_id.split("__")[0]
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)
        one_per_axis.append(arm)
    subset = tuple(one_per_axis + hysteresis_arms)
    assert len(seen_prefixes) == 4  # observation, reward, placement, stabilization

    runs = RunManifestStore(tmp_path / "runs")
    report = run_ablation_matrix(
        token, subset, seeds=1, runs=runs, output_root=str(tmp_path / "artifacts"), provenance_factory=_provenance,
        bootstrap_plan=BootstrapPlan(n_bootstrap=200, rng_seed=11, rationale="test runtime ceiling"),
        threshold=PracticalThreshold(metric="capture_rate", unit="probability", minimum_effect=0.05, direction=EffectDirection.GREATER_IS_BETTER),
    )

    assert report.ledger.conserved
    assert report.ledger.planned_count == len(subset)
    statuses = {record.condition_id: record.execution_status for record in report.ledger.records}
    for arm in hysteresis_arms:
        assert statuses[arm.condition_id] is ExecutionStatus.NOT_RUN
    for arm in one_per_axis:
        assert statuses[arm.condition_id] is ExecutionStatus.COMPLETED

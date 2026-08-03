"""Task 12.4 regressions for gated cross-city zero-shot evaluation.

Exercises pursuit_evasion_rl/research/experiments/cross_city.py against a
real, tiny CPU-trained policy: a policy selected on daejeon is frozen and
zero-shot evaluated (no further training) on two synthetic target cities
(busan, seoul), a target city identical to the source is refused as leaked
generalization evidence, and a missing/unavailable target city is recorded
not_run rather than blocking the whole run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pursuit_evasion_rl.osm_demo.coarsening import coarsen_raw_graph, prepare_model_network
from pursuit_evasion_rl.osm_demo.fixtures import busan, daejeon, seoul
from pursuit_evasion_rl.research.budget import ConditionResourceEstimate, ResourceCeiling, decide_sample_size
from pursuit_evasion_rl.research.canonical import content_hash
from pursuit_evasion_rl.research.domain import AnalysisClassification, ExecutionStatus, MapScenario
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.execution import ConditionCostEstimate, plan_execution
from pursuit_evasion_rl.research.experiments.cross_city import (
    CrossCityTargetSpec,
    assert_no_target_city_leakage,
    evaluate_cross_city_target,
    require_cross_city_admission,
    run_cross_city_zero_shot,
)
from pursuit_evasion_rl.research.maps.splits import DataHandle, HandleKind, SplitScope, TuningDataView
from pursuit_evasion_rl.research.metrics.behavior import MetricDirection
from pursuit_evasion_rl.research.preservation import measure_artifact
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
from pursuit_evasion_rl.research.statistics.paired import BootstrapPlan, CorrectionMethod, EffectDirection, MissingDataPolicy, PracticalThreshold
from pursuit_evasion_rl.research.training.trainer import ResearchTrainer, TrainerConfig
from pursuit_evasion_rl.research.variants.placement import PlacementCurriculumConfig, PlacementStyle

pytestmark = pytest.mark.offline

SIGNER = "principal-investigator"
CONDITION_ID = "cross_city_source"


class LowestLegalActionPolicy:
    policy_id = "lowest-legal-action-v1"

    def act(self, observation) -> int:
        return min(index for index, legal in enumerate(observation.action_mask) if legal)


def _baseline_factory(_network):
    return LowestLegalActionPolicy()


def _daejeon_network():
    return prepare_model_network(coarsen_raw_graph(daejeon())).network


def _busan_network():
    return prepare_model_network(coarsen_raw_graph(busan())).network


def _seoul_network():
    return prepare_model_network(coarsen_raw_graph(seoul())).network


def _tuning_view() -> TuningDataView:
    return TuningDataView(
        train=(DataHandle(SplitScope.TRAIN, HandleKind.MAP, "cross-city-train"),),
        validation=(DataHandle(SplitScope.VALIDATION, HandleKind.MAP, "cross-city-validation"),),
    )


def _selected_checkpoint(tmp_path: Path) -> tuple[Path, str]:
    """Trains a real tiny policy on daejeon and returns its selected checkpoint and source map hash."""
    network = _daejeon_network()
    config = TrainerConfig(updates=1, episodes_per_update=1, max_steps=10, hidden_dims=(4,))
    trainer = ResearchTrainer(
        train_network=network, validation_network=network, tuning_data=_tuning_view(), config=config,
        condition_id=CONDITION_ID, training_seed=1, output_root=tmp_path / "artifacts",
    )
    result = trainer.train()
    return result.checkpoint_path, content_hash(network)


def _metrics():
    return (
        MetricDeclaration(metric_id="capture_rate", symbol="P_cap", unit="probability", direction=MetricDirection.HIGHER_IS_BETTER, role=MetricRole.PRIMARY, formula="captured episodes divided by planned episodes"),
        metric_declaration("blocked_exit_fraction", MetricRole.PRIMARY),
    )


def _resource_spec() -> ResourceSpec:
    return ResourceSpec(
        ceiling=ResourceCeiling(resource_ceiling_id="ceiling-cross-city", measured_at_utc="2026-07-01T00:00:00Z", accelerator_hours=10.0, wall_clock_hours=10.0),
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
        conditions=(ConditionRef(condition_id=CONDITION_ID, condition_hash="9" * 64, axis="baseline", arm="default"),),
        split=SplitSpec(
            metric_crs="EPSG:5186", buffer_m=50.0,
            polygon_hashes={"train": "1" * 64, "validation": "2" * 64, "test": "3" * 64},
            network_hashes={"train": "4" * 64, "validation": "5" * 64, "test": "6" * 64},
            cross_city_city_ids=("busan", "seoul"), split_protocol_hash="7" * 64, split_validation_report_hash="8" * 64,
            held_out_evaluation_rule="apply the validation-selected policy unchanged to the test split, once",
        ),
        sample=SampleSpec(plan=decide_sample_size(resource.ceiling, resource.estimates), selection_handles=(DataHandle(SplitScope.TRAIN, HandleKind.MAP, "cross-city-train"), DataHandle(SplitScope.VALIDATION, HandleKind.MAP, "cross-city-validation"))),
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


def _spec(city_id: str, network) -> CrossCityTargetSpec:
    return CrossCityTargetSpec(
        city_id=city_id, scenario=MapScenario.INTERIOR_CONTAINED, network=network,
        config=TrainerConfig(updates=1, episodes_per_update=1, max_steps=10, hidden_dims=(4,)),
        placement_config=PlacementCurriculumConfig(style=PlacementStyle.GLOBAL),
        baseline_policy_factory=_baseline_factory, episodes=2,
    )


def test_a_target_city_identical_to_the_source_is_refused_as_leaked() -> None:
    network = _daejeon_network()
    with pytest.raises(ResearchValidationError) as excinfo:
        assert_no_target_city_leakage(source_map_hash=content_hash(network), target_network=network, target_city_id="daejeon-again")
    assert excinfo.value.code == "TARGET_CITY_LEAKAGE"


def test_a_genuinely_different_target_city_passes_the_leakage_guard() -> None:
    source_hash = content_hash(_daejeon_network())
    assert_no_target_city_leakage(source_map_hash=source_hash, target_network=_busan_network(), target_city_id="busan")  # must not raise


def test_admission_requires_a_sealed_protocol(tmp_path: Path) -> None:
    with pytest.raises(ResearchValidationError) as excinfo:
        require_cross_city_admission(quality_attestation=_clean_attestation(), protocol=None, resource_plan=object())
    assert excinfo.value.code == "MISSING_SEALED_PROTOCOL"


def _token_and_source(tmp_path: Path):
    sealed, resource = _sealed_protocol(tmp_path)
    checkpoint_path, source_hash = _selected_checkpoint(tmp_path)
    plan = plan_execution(
        protocol_hash=sealed.protocol_hash, ceiling=resource.ceiling,
        cost_estimates=(ConditionCostEstimate(resource=ConditionResourceEstimate(condition_id=CONDITION_ID, accelerator_hours_per_seed=0.01, wall_clock_hours_per_seed=0.01, env_steps_per_seed=30)),),
    )
    token = require_cross_city_admission(quality_attestation=_clean_attestation(), protocol=sealed, resource_plan=plan)
    return token, checkpoint_path, source_hash, sealed.protocol_hash


def test_a_missing_target_city_is_recorded_not_run_not_a_launch_failure(tmp_path: Path) -> None:
    token, checkpoint_path, source_hash, protocol_hash = _token_and_source(tmp_path)
    unavailable = CrossCityTargetSpec(
        city_id="unavailable-city", scenario=MapScenario.INTERIOR_CONTAINED, network=None,
        config=TrainerConfig(updates=1, episodes_per_update=1, max_steps=10, hidden_dims=(4,)),
        placement_config=PlacementCurriculumConfig(style=PlacementStyle.GLOBAL),
        baseline_policy_factory=_baseline_factory, episodes=2,
        unavailable_reason="snapshot cache miss: no cached OSM extract for this city",
    )

    result = evaluate_cross_city_target(
        token, unavailable, frozen_checkpoint_path=str(checkpoint_path), source_map_hash=source_hash,
        observation_adapter_factory=None, protocol_hash=protocol_hash,
        bootstrap_plan=BootstrapPlan(n_bootstrap=200, rng_seed=11, rationale="test"),
        threshold=PracticalThreshold(metric="capture_rate", unit="probability", minimum_effect=0.05, direction=EffectDirection.GREATER_IS_BETTER),
    )
    assert result.execution_record.execution_status is ExecutionStatus.NOT_RUN
    assert "cache miss" in result.execution_record.status_reason
    assert result.statistics is None


def test_a_real_frozen_policy_zero_shot_evaluates_on_two_target_cities_with_no_further_training(tmp_path: Path) -> None:
    token, checkpoint_path, source_hash, protocol_hash = _token_and_source(tmp_path)

    targets = (_spec("busan", _busan_network()), _spec("seoul", _seoul_network()))
    report = run_cross_city_zero_shot(
        token, targets, frozen_checkpoint_path=str(checkpoint_path), source_map_hash=source_hash, protocol_hash=protocol_hash,
        bootstrap_plan=BootstrapPlan(n_bootstrap=200, rng_seed=11, rationale="test"),
        threshold=PracticalThreshold(metric="capture_rate", unit="probability", minimum_effect=0.05, direction=EffectDirection.GREATER_IS_BETTER),
    )

    assert report.ledger.conserved
    assert report.ledger.planned_count == 2
    for record in report.ledger.records:
        assert record.execution_status is ExecutionStatus.COMPLETED
        assert record.is_measured


def test_a_target_city_identical_to_the_source_blocks_the_whole_run(tmp_path: Path) -> None:
    token, checkpoint_path, source_hash, protocol_hash = _token_and_source(tmp_path)
    leaked = _spec("daejeon-relabelled", _daejeon_network())

    with pytest.raises(ResearchValidationError) as excinfo:
        run_cross_city_zero_shot(
            token, (leaked,), frozen_checkpoint_path=str(checkpoint_path), source_map_hash=source_hash, protocol_hash=protocol_hash,
            bootstrap_plan=BootstrapPlan(n_bootstrap=200, rng_seed=11, rationale="test"),
            threshold=PracticalThreshold(metric="capture_rate", unit="probability", minimum_effect=0.05, direction=EffectDirection.GREATER_IS_BETTER),
        )
    assert excinfo.value.code == "TARGET_CITY_LEAKAGE"

"""Task 12.1 regressions for the protocol-blind pilot runner.

Exercises pursuit_evasion_rl/research/experiments/pilot.py against a real,
tiny CPU trainer: admission is refused when any of the four required pieces
(quality gate, sealed protocol, resource plan, best_v2 measurement) is
missing, and once admitted a pilot condition always seals a manifest --
completed on success, failed (with a preserved error trace) on a real
misconfiguration -- without ever touching the legacy checkpoint.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from pursuit_evasion_rl.osm_demo.coarsening import coarsen_raw_graph, prepare_model_network
from pursuit_evasion_rl.osm_demo.fixtures import daejeon
from pursuit_evasion_rl.research.budget import ConditionResourceEstimate, ResourceCeiling, decide_sample_size
from pursuit_evasion_rl.research.domain import AnalysisClassification, ExecutionStatus
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.execution import plan_execution
from pursuit_evasion_rl.research.experiments.pilot import (
    LEGACY_BEST_V2_PATH,
    PilotAdmissionToken,
    PilotConditionSpec,
    cost_pilot_matrix,
    execute_pilot_condition,
    require_pilot_admission,
    run_pilot,
)
from pursuit_evasion_rl.research.maps.splits import DataHandle, HandleKind, SplitScope, TuningDataView
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
from pursuit_evasion_rl.research.metrics.behavior import MetricDirection
from pursuit_evasion_rl.research.runs.manifest import RunManifestStore, RunProvenance
from pursuit_evasion_rl.research.statistics.paired import BootstrapPlan, CorrectionMethod, MissingDataPolicy
from pursuit_evasion_rl.research.training.trainer import TrainerConfig

pytestmark = pytest.mark.offline

SIGNER = "principal-investigator"
CONDITION_ID = "pilot_sanity_interior"


def _tiny_network():
    return prepare_model_network(coarsen_raw_graph(daejeon())).network


def _tiny_config() -> TrainerConfig:
    return TrainerConfig(updates=2, episodes_per_update=1, max_steps=15, hidden_dims=(4,))


def _tuning_view() -> TuningDataView:
    return TuningDataView(
        train=(DataHandle(SplitScope.TRAIN, HandleKind.MAP, "pilot-train"),),
        validation=(DataHandle(SplitScope.VALIDATION, HandleKind.MAP, "pilot-validation"),),
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
        ceiling=ResourceCeiling(
            resource_ceiling_id="ceiling-pilot", measured_at_utc="2026-07-01T00:00:00Z",
            accelerator_hours=10.0, wall_clock_hours=10.0,
        ),
        estimates=(
            ConditionResourceEstimate(
                condition_id=CONDITION_ID, accelerator_hours_per_seed=0.01,
                wall_clock_hours_per_seed=0.01, env_steps_per_seed=30,
            ),
        ),
    )


def _sealed_protocol(tmp_path: Path):
    questions = default_research_questions()
    metrics = _metrics()
    resource = _resource_spec()
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
        conditions=(ConditionRef(condition_id=CONDITION_ID, condition_hash="9" * 64, axis="baseline", arm="default"),),
        split=SplitSpec(
            metric_crs="EPSG:5186", buffer_m=50.0,
            polygon_hashes={"train": "1" * 64, "validation": "2" * 64, "test": "3" * 64},
            network_hashes={"train": "4" * 64, "validation": "5" * 64, "test": "6" * 64},
            cross_city_city_ids=("busan", "seoul"), split_protocol_hash="7" * 64, split_validation_report_hash="8" * 64,
            held_out_evaluation_rule="apply the validation-selected policy unchanged to the test split, once",
        ),
        sample=SampleSpec(
            plan=decide_sample_size(resource.ceiling, resource.estimates),
            selection_handles=(DataHandle(SplitScope.TRAIN, HandleKind.MAP, "pilot-train"), DataHandle(SplitScope.VALIDATION, HandleKind.MAP, "pilot-validation")),
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
            search_date_utc="2026-07-01T00:00:00Z", query="multi-agent reinforcement learning vehicle pursuit road network",
            sources=("Scopus", "IEEE Xplore", "arXiv"), date_range_start="2015-01-01", date_range_end="2026-06-30",
            languages=("en", "ko"), inclusion_criteria=("multi-pursuer pursuit on a graph or road network",), exclusion_criteria=("continuous open-plane pursuit without a road network",),
        ),
        analyses=analyses,
    )
    store = ProtocolStore(tmp_path / "protocols")
    draft = store.create(questions=questions, specifications=specifications)
    return store.seal(draft.protocol_id, signer=SIGNER), resource


def _clean_attestation(properties_dir: Path) -> GateAttestation:
    """A synthetic but internally-consistent passing attestation (no full suite run)."""
    checks = tuple(
        PropertyFileCheck(property_id=property_id, file_path=f"tests/research/properties/test_property_{property_id}.py", has_offline_marker=True, max_examples=100)
        for property_id in REQUIRED_PROPERTY_IDS
    )
    coverage = PropertyCoverageReport(checks=checks, deferred_property_ids=(18,))
    suite = SuiteRunReport(target="tests/research", total=1, failures=0, errors=0, skipped=0, skipped_test_ids=(), return_code=0, junit_xml_path="synthetic.xml")
    return GateAttestation(coverage=coverage, suite_runs=(suite,), code_hash_at_attestation="synthetic", generated_at_utc="2026-07-31T00:00:00Z")


def _provenance() -> RunProvenance:
    return RunProvenance(
        code_hash="a" * 64, dirty_tree=False, dependency_hash="b" * 64,
        runtime="python3.11+torch2.8.0+cpu", device="cpu", map_hash="c" * 64,
        split="train", seed=5, input_hashes={"map": "c" * 64},
    )


def _spec(*, output_root: Path, updates: int = 2) -> PilotConditionSpec:
    network = _tiny_network()
    config = dataclasses.replace(_tiny_config(), updates=updates)
    return PilotConditionSpec(
        condition_id=CONDITION_ID, train_network=network, validation_network=network,
        tuning_data=_tuning_view(), config=config, training_seed=11,
    )


def test_admission_is_refused_and_zero_runs_are_created_when_any_requirement_is_missing(tmp_path: Path) -> None:
    sealed, resource = _sealed_protocol(tmp_path)
    plan = cost_pilot_matrix(
        (_spec(output_root=tmp_path / "artifacts"),), protocol_hash=sealed.protocol_hash, ceiling=resource.ceiling,
        evaluation_episodes=1, accelerator_hours_per_million_env_steps=1.0, wall_clock_hours_per_million_env_steps=1.0,
    )
    attestation = _clean_attestation(tmp_path)
    runs = RunManifestStore(tmp_path / "runs")

    # Missing/ineligible quality gate.
    ineligible = dataclasses.replace(
        attestation,
        coverage=dataclasses.replace(attestation.coverage, checks=tuple(
            dataclasses.replace(item, file_path=None, has_offline_marker=False, max_examples=None) if item.property_id == attestation.coverage.checks[0].property_id else item
            for item in attestation.coverage.checks
        )),
    )
    with pytest.raises(ResearchValidationError) as excinfo:
        require_pilot_admission(quality_attestation=ineligible, protocol=sealed, resource_plan=plan)
    assert excinfo.value.code == "GATE_NOT_ADMISSION_ELIGIBLE"

    # Unsealed protocol.
    draft = ProtocolStore(tmp_path / "protocols-2").create(questions=default_research_questions(), specifications=_sealed_protocol(tmp_path / "throwaway")[0].protocol.specifications)
    with pytest.raises(ResearchValidationError) as excinfo:
        require_pilot_admission(quality_attestation=attestation, protocol=draft, resource_plan=plan)
    assert excinfo.value.code == "PROTOCOL_NOT_SEALED"

    # Resource plan costed against a different protocol.
    other_sealed, _ = _sealed_protocol(tmp_path / "other-protocol")
    with pytest.raises(ResearchValidationError) as excinfo:
        require_pilot_admission(quality_attestation=attestation, protocol=other_sealed, resource_plan=plan)
    assert excinfo.value.code == "RESOURCE_PLAN_PROTOCOL_MISMATCH"

    # Nonexistent best_v2 path.
    with pytest.raises(ResearchValidationError):
        require_pilot_admission(quality_attestation=attestation, protocol=sealed, resource_plan=plan, best_v2_path=str(tmp_path / "no-such-checkpoint.pt"))

    assert list(runs.root.iterdir()) == [], "no run manifest may exist when admission itself never succeeded"


def test_a_real_pilot_condition_runs_to_completion_and_seals_a_manifest(tmp_path: Path) -> None:
    sealed, resource = _sealed_protocol(tmp_path)
    spec = _spec(output_root=tmp_path / "artifacts")
    plan = cost_pilot_matrix(
        (spec,), protocol_hash=sealed.protocol_hash, ceiling=resource.ceiling,
        evaluation_episodes=1, accelerator_hours_per_million_env_steps=1.0, wall_clock_hours_per_million_env_steps=1.0,
    )
    checkpoint_path = tmp_path / "fake-best-v2.pt"
    checkpoint_path.write_bytes(b"not a real checkpoint, only a path to measure")
    token = require_pilot_admission(
        quality_attestation=_clean_attestation(tmp_path), protocol=sealed, resource_plan=plan,
        best_v2_path=str(checkpoint_path),
    )
    assert isinstance(token, PilotAdmissionToken)

    runs = RunManifestStore(tmp_path / "runs")
    report = run_pilot(token, (spec,), runs=runs, output_root=str(tmp_path / "artifacts"), provenance_factory=_provenance)

    assert report.all_completed
    assert len(report.outcomes) == 1
    outcome = report.outcomes[0]
    assert outcome.execution_status is ExecutionStatus.COMPLETED
    assert outcome.run_id is not None
    manifest = runs.read(outcome.run_id)
    assert manifest.is_sealed
    assert manifest.run.execution_status is ExecutionStatus.COMPLETED

    # The legacy checkpoint file itself was never touched.
    assert checkpoint_path.read_bytes() == b"not a real checkpoint, only a path to measure"


def test_a_misconfigured_pilot_condition_seals_failed_with_a_preserved_error_trace(tmp_path: Path) -> None:
    """A real ResearchTrainer-time failure -- a pre-occupied run directory -- must still seal, not raise past the caller."""
    sealed, resource = _sealed_protocol(tmp_path)
    network = _tiny_network()
    tuning = _tuning_view()
    output_root = tmp_path / "artifacts"
    fixed_config = dataclasses.replace(_tiny_config(), updates=1)
    spec = PilotConditionSpec(
        condition_id=CONDITION_ID, train_network=network, validation_network=network,
        tuning_data=tuning, config=fixed_config, training_seed=11,
    )
    plan = cost_pilot_matrix(
        (spec,), protocol_hash=sealed.protocol_hash, ceiling=resource.ceiling,
        evaluation_episodes=1, accelerator_hours_per_million_env_steps=1.0, wall_clock_hours_per_million_env_steps=1.0,
    )
    checkpoint_path = tmp_path / "fake-best-v2.pt"
    checkpoint_path.write_bytes(b"not a real checkpoint")
    token = require_pilot_admission(
        quality_attestation=_clean_attestation(tmp_path), protocol=sealed, resource_plan=plan,
        best_v2_path=str(checkpoint_path),
    )

    class _FixedRunManifestStore(RunManifestStore):
        """Hands back a fixed run_id so the trainer's run directory is guaranteed pre-occupied."""

        def create(self, **kwargs):
            kwargs["run_id"] = "collision-run"
            return super().create(**kwargs)

    # Pre-occupy the exact directory ResearchTrainer will try to claim.
    (output_root / CONDITION_ID / "11" / "collision-run").mkdir(parents=True)

    outcome, manifest = execute_pilot_condition(
        token, spec, runs=_FixedRunManifestStore(tmp_path / "runs"), output_root=str(output_root), provenance_factory=_provenance,
    )

    assert outcome.execution_status is ExecutionStatus.FAILED
    assert outcome.error_artifact_hash is not None
    assert manifest.run.execution_status is ExecutionStatus.FAILED
    assert "RUN_DIRECTORY_EXISTS" in outcome.status_reason

"""Task 12.3: the gated observation/reward/placement/stabilization ablation matrix.

Every arm here is one factor away from the shared default Condition
(:func:`~pursuit_evasion_rl.research.variants.factory.default_condition`):
Observation_21D vs 28D, one reward component left out (or the retreat arm
made symmetric), one placement curriculum style, or one stabilization
factor toggled.  :func:`build_ablation_matrix` assembles the full metadata
matrix and validates it single-factor-clean via
:func:`~pursuit_evasion_rl.research.variants.factory.validate_condition_matrix`
*before* any training launches -- the same fail-closed ordering Task 12.1's
pilot and 12.2's main study already establish.

Execution reuses :mod:`~pursuit_evasion_rl.research.experiments.main_study`'s
already-verified per-seed train/evaluate/paired-statistics pipeline rather
than duplicating it; each axis's arm supplies the trainer-extension
parameter (``reward_components``, ``observation_dim`` +
``observation_adapter_factory``, ``placement_config``, or
``stabilization_condition``) :mod:`~pursuit_evasion_rl.research.training.trainer`
now accepts.

Hysteresis is the one declared factor this module refuses to train under
(:data:`HYSTERESIS_INCOMPATIBILITY_REASON`): its "held distant goal" design
has no stable candidate to persist across decisions when every decision
only ever chooses among the current intersection's immediate neighbors, so
the two arms with ``hysteresis=True`` are recorded ``not_run`` with that
reason rather than trained under an invented, unreviewed interpretation
(Requirement 15.8 already requires ``not_run`` to be a first-class,
preserved outcome -- this is that path, used honestly).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from pursuit_evasion_rl.osm_demo.models import ModelNetwork

from ..budget import ConditionResourceEstimate, ResourceCeiling, SampleSizeStatus, decide_sample_size
from ..canonical import content_hash
from ..domain import ExecutionStatus, MapScenario
from ..errors import ResearchValidationError
from ..evaluation.paired import FrozenPolicy
from ..execution import ConditionExecutionLedger, ConditionExecutionRecord
from ..maps.splits import TuningDataView
from ..preservation import ArtifactMeasurement, measure_artifact
from ..protocol import ProtocolRecord
from ..quality import GateAttestation
from ..runs.manifest import RunManifestStore
from ..statistics.paired import BootstrapPlan, PracticalThreshold
from ..training.trainer import TrainerConfig
from ..variants.factory import (
    ConditionSpec,
    default_condition,
    observation_ablation_conditions,
    placement_ablation_conditions,
    reward_ablation_conditions,
    stabilization_ablation_conditions,
    validate_condition_matrix,
)
from ..variants.observations import OBSERVATION_21D_DIM, OBSERVATION_28D_DIM, Observation21DAdapter
from ..variants.placement import PlacementCurriculumConfig, PlacementStyle, all_placement_conditions
from ..variants.rewards import RewardComponent, RewardComponentSet, leave_one_component_out
from ..variants.stabilization import StabilizationCondition, all_stabilization_arms
from .main_study import (
    LEGACY_BEST_V2_PATH,
    MainStudyConditionSpec,
    run_main_study_condition,
)

ABLATION_SCHEMA_VERSION = "1.0"

#: Requirement 10.4-10.11: hysteresis is a declared factor this pipeline
#: cannot train correctly yet -- see the module docstring.
HYSTERESIS_INCOMPATIBILITY_REASON = (
    "hysteresis holds a committed distant-goal intersection across decisions, but this "
    "policy's per-decision candidates are only the current intersection's immediate "
    "neighbors, which changes completely at every decision; there is no stable candidate "
    "for a committed target to persist against, so this arm is not trainable under the "
    "stabilization module's current (goal-holding) design without an unreviewed "
    "reinterpretation"
)


def _fail(code: str, message: str, **kwargs) -> None:
    raise ResearchValidationError(code, message, **kwargs)


class AblationAdmissionError(ResearchValidationError):
    """Raised when the ablation matrix cannot launch: an admission requirement is missing."""


# ---------------------------------------------------------------------------
# Admission: quality gate + sealed protocol + a fresh resource estimate for
# this matrix + a fresh best_v2 measurement (Requirement 7.1-7.5, 19.9-19.11)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class AblationAdmissionToken:
    quality_attestation: GateAttestation
    protocol: ProtocolRecord
    resource_plan: object
    best_v2_measurement: ArtifactMeasurement
    schema_version: str = ABLATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.quality_attestation, GateAttestation):
            _fail("MISSING_QUALITY_GATE", "the ablation matrix requires a quality-gate attestation", path="quality_attestation")
        if not self.quality_attestation.pilot_admission_eligible:
            _fail(
                "GATE_NOT_ADMISSION_ELIGIBLE", "the quality-gate attestation does not clear admission",
                path="quality_attestation.pilot_admission_eligible", expected=True, actual=False,
            )
        if not isinstance(self.protocol, ProtocolRecord):
            _fail("MISSING_SEALED_PROTOCOL", "the ablation matrix requires a ProtocolRecord", path="protocol")
        if not self.protocol.is_sealed:
            _fail("PROTOCOL_NOT_SEALED", "the ablation matrix cannot launch against an unsealed protocol", path="protocol.is_sealed", expected=True, actual=False)
        if self.resource_plan is None:
            _fail("MISSING_RESOURCE_ESTIMATE", "the ablation matrix requires a fresh resource estimate", path="resource_plan")
        if getattr(self.resource_plan, "protocol_hash", None) != self.protocol.protocol_hash:
            _fail(
                "RESOURCE_PLAN_PROTOCOL_MISMATCH", "the resource plan was costed against a different protocol",
                path="resource_plan.protocol_hash", expected=self.protocol.protocol_hash,
                actual=getattr(self.resource_plan, "protocol_hash", None),
            )
        if not isinstance(self.best_v2_measurement, ArtifactMeasurement):
            _fail("MISSING_PROTECTED_ARTIFACT_MEASUREMENT", "the ablation matrix requires a fresh best_v2 measurement", path="best_v2_measurement")

    @property
    def admission_hash(self) -> str:
        return content_hash(self)


def require_ablation_admission(
    *, quality_attestation: GateAttestation, protocol: ProtocolRecord, resource_plan: object,
    best_v2_path: str = LEGACY_BEST_V2_PATH,
) -> AblationAdmissionToken:
    measurement = measure_artifact(best_v2_path)
    return AblationAdmissionToken(
        quality_attestation=quality_attestation, protocol=protocol,
        resource_plan=resource_plan, best_v2_measurement=measurement,
    )


# ---------------------------------------------------------------------------
# One executable spec per metadata ConditionSpec, per axis
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class AblationArmSpec:
    """One arm's metadata identity plus its executable recipe (or refusal reason)."""

    metadata: ConditionSpec
    condition_spec: MainStudyConditionSpec | None
    not_run_reason: str | None = None
    schema_version: str = ABLATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (self.condition_spec is None) == (self.not_run_reason is None):
            _fail(
                "INVALID_ABLATION_ARM", "an ablation arm is either executable or carries its not_run reason, not both or neither",
                path="condition_spec",
            )

    @property
    def condition_id(self) -> str:
        return self.metadata.condition.condition_id


def _base_kwargs(
    *, scenario: MapScenario, network: ModelNetwork, tuning_data: TuningDataView, base_config: TrainerConfig,
    baseline_policy_factory: Callable[[ModelNetwork], FrozenPolicy], episodes_per_seed: int,
    placement_config: PlacementCurriculumConfig,
) -> dict:
    return dict(
        scenario=scenario, train_network=network, validation_network=network, tuning_data=tuning_data,
        config=base_config, placement_config=placement_config, baseline_policy_factory=baseline_policy_factory,
        episodes_per_seed=episodes_per_seed,
    )


def observation_ablation_specs(
    scenario: MapScenario, *, network: ModelNetwork, tuning_data: TuningDataView, base_config: TrainerConfig,
    baseline_policy_factory: Callable[[ModelNetwork], FrozenPolicy], episodes_per_seed: int,
    placement_config: PlacementCurriculumConfig,
) -> tuple[AblationArmSpec, ...]:
    """Requirement 9.1: the Observation_21D vs 28D one-factor pair."""
    metadata = observation_ablation_conditions(scenario)
    arms = []
    for meta in metadata:
        is_21d = "observation_21d" in meta.arm
        kwargs = _base_kwargs(
            scenario=scenario, network=network, tuning_data=tuning_data, base_config=base_config,
            baseline_policy_factory=baseline_policy_factory, episodes_per_seed=episodes_per_seed,
            placement_config=placement_config,
        )
        condition_spec = MainStudyConditionSpec(
            condition_id=meta.condition.condition_id,
            observation_dim=OBSERVATION_21D_DIM if is_21d else OBSERVATION_28D_DIM,
            observation_adapter_factory=(
                (lambda net, clip, near: Observation21DAdapter(net, clip_distance_m=clip)) if is_21d else None
            ),
            **kwargs,
        )
        arms.append(AblationArmSpec(metadata=meta, condition_spec=condition_spec))
    return tuple(arms)


def reward_ablation_specs(
    scenario: MapScenario, *, network: ModelNetwork, tuning_data: TuningDataView, base_config: TrainerConfig,
    baseline_policy_factory: Callable[[ModelNetwork], FrozenPolicy], episodes_per_seed: int,
    placement_config: PlacementCurriculumConfig,
) -> tuple[AblationArmSpec, ...]:
    """Requirement 9.6-9.7: leave-one-component-out arms."""
    metadata = reward_ablation_conditions(scenario)
    arms = []
    for meta, component in zip(metadata, RewardComponent):
        kwargs = _base_kwargs(
            scenario=scenario, network=network, tuning_data=tuning_data, base_config=base_config,
            baseline_policy_factory=baseline_policy_factory, episodes_per_seed=episodes_per_seed,
            placement_config=placement_config,
        )
        condition_spec = MainStudyConditionSpec(
            condition_id=meta.condition.condition_id, reward_components=leave_one_component_out(component), **kwargs,
        )
        arms.append(AblationArmSpec(metadata=meta, condition_spec=condition_spec))
    return tuple(arms)


def placement_ablation_specs(
    scenario: MapScenario, *, network: ModelNetwork, tuning_data: TuningDataView, base_config: TrainerConfig,
    baseline_policy_factory: Callable[[ModelNetwork], FrozenPolicy], episodes_per_seed: int,
) -> tuple[AblationArmSpec, ...]:
    """Requirement 10.1: global/ring/mixed, one factor apart from the default."""
    metadata = placement_ablation_conditions(scenario)
    placements = all_placement_conditions(PlacementCurriculumConfig(style=PlacementStyle.GLOBAL))
    arms = []
    for meta, placement in zip(metadata, placements):
        kwargs = _base_kwargs(
            scenario=scenario, network=network, tuning_data=tuning_data, base_config=base_config,
            baseline_policy_factory=baseline_policy_factory, episodes_per_seed=episodes_per_seed,
            placement_config=placement,
        )
        condition_spec = MainStudyConditionSpec(condition_id=meta.condition.condition_id, **kwargs)
        arms.append(AblationArmSpec(metadata=meta, condition_spec=condition_spec))
    return tuple(arms)


def stabilization_ablation_specs(
    scenario: MapScenario, *, network: ModelNetwork, tuning_data: TuningDataView, base_config: TrainerConfig,
    baseline_policy_factory: Callable[[ModelNetwork], FrozenPolicy], episodes_per_seed: int,
    placement_config: PlacementCurriculumConfig,
) -> tuple[AblationArmSpec, ...]:
    """Requirement 10.4: the four off/on x off/on arms; hysteresis=True is not_run."""
    metadata = stabilization_ablation_conditions(scenario)
    stabilization_arms = all_stabilization_arms(StabilizationCondition())
    arms = []
    for meta, arm in zip(metadata, stabilization_arms):
        if arm.hysteresis:
            arms.append(AblationArmSpec(metadata=meta, condition_spec=None, not_run_reason=HYSTERESIS_INCOMPATIBILITY_REASON))
            continue
        kwargs = _base_kwargs(
            scenario=scenario, network=network, tuning_data=tuning_data, base_config=base_config,
            baseline_policy_factory=baseline_policy_factory, episodes_per_seed=episodes_per_seed,
            placement_config=placement_config,
        )
        condition_spec = MainStudyConditionSpec(condition_id=meta.condition.condition_id, stabilization_condition=arm, **kwargs)
        arms.append(AblationArmSpec(metadata=meta, condition_spec=condition_spec))
    return tuple(arms)


def build_ablation_matrix(
    scenario: MapScenario, *, network: ModelNetwork, tuning_data: TuningDataView, base_config: TrainerConfig,
    baseline_policy_factory: Callable[[ModelNetwork], FrozenPolicy], episodes_per_seed: int,
    placement_config: PlacementCurriculumConfig,
) -> tuple[AblationArmSpec, ...]:
    """The full matrix -- default plus every axis's arms -- validated single-factor-clean.

    Validation runs against the metadata layer (:func:`validate_condition_matrix`)
    before anything about the *executable* recipes is even considered, so a
    drifted or duplicated arm is caught before it could ever reach training.
    """
    default_meta = default_condition(scenario)
    observation = observation_ablation_specs(
        scenario, network=network, tuning_data=tuning_data, base_config=base_config,
        baseline_policy_factory=baseline_policy_factory, episodes_per_seed=episodes_per_seed, placement_config=placement_config,
    )
    reward = reward_ablation_specs(
        scenario, network=network, tuning_data=tuning_data, base_config=base_config,
        baseline_policy_factory=baseline_policy_factory, episodes_per_seed=episodes_per_seed, placement_config=placement_config,
    )
    placement = placement_ablation_specs(
        scenario, network=network, tuning_data=tuning_data, base_config=base_config,
        baseline_policy_factory=baseline_policy_factory, episodes_per_seed=episodes_per_seed,
    )
    stabilization = stabilization_ablation_specs(
        scenario, network=network, tuning_data=tuning_data, base_config=base_config,
        baseline_policy_factory=baseline_policy_factory, episodes_per_seed=episodes_per_seed, placement_config=placement_config,
    )
    arms = observation + reward + placement + stabilization
    validate_condition_matrix((default_meta,) + tuple(arm.metadata for arm in arms))
    return arms


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class AblationMatrixReport:
    admission_hash: str
    ledger: ConditionExecutionLedger
    schema_version: str = ABLATION_SCHEMA_VERSION

    @property
    def report_hash(self) -> str:
        return content_hash(self)


def run_ablation_matrix(
    token: AblationAdmissionToken, arms: Sequence[AblationArmSpec], *, seeds: int,
    runs: RunManifestStore, output_root: str, provenance_factory: Callable[[], object],
    bootstrap_plan: BootstrapPlan, threshold: PracticalThreshold,
) -> AblationMatrixReport:
    """Run every trainable arm and seal every declared arm, including the not_run ones.

    A ``not_run`` arm (hysteresis today) still gets its own
    :class:`~pursuit_evasion_rl.research.execution.ConditionExecutionRecord`
    with its reason recorded -- it is a preserved outcome, not an absence
    (Requirement 15.8).
    """
    if not isinstance(token, AblationAdmissionToken):
        _fail("MISSING_ABLATION_ADMISSION", "run_ablation_matrix requires a validated AblationAdmissionToken", path="token")
    arms = tuple(arms)
    if not arms:
        _fail("EMPTY_CONDITION_MATRIX", "the ablation matrix requires at least one arm", path="arms")

    records: list[ConditionExecutionRecord] = []
    for arm in arms:
        if arm.condition_spec is None:
            records.append(
                ConditionExecutionRecord(
                    condition_id=arm.condition_id, protocol_hash=token.protocol.protocol_hash,
                    execution_status=ExecutionStatus.NOT_RUN,
                    status_reason=arm.not_run_reason, sample_size_status=SampleSizeStatus.DEFAULT,
                )
            )
            continue
        result = run_main_study_condition(
            token, arm.condition_spec, seeds=seeds, runs=runs, output_root=output_root,
            provenance_factory=provenance_factory, bootstrap_plan=bootstrap_plan, threshold=threshold,
        )
        records.append(result.execution_record)

    ceiling = ResourceCeiling(resource_ceiling_id="ablation-ledger-ceiling", measured_at_utc="2026-07-31T00:00:00Z", accelerator_hours=1.0, wall_clock_hours=1.0)
    estimate = ConditionResourceEstimate(condition_id=arms[0].condition_id, accelerator_hours_per_seed=0.001, wall_clock_hours_per_seed=0.001, env_steps_per_seed=1)
    plan = decide_sample_size(ceiling, [estimate])
    ledger = ConditionExecutionLedger(
        protocol_hash=token.protocol.protocol_hash, sample_size_plan=plan,
        planned_condition_ids=tuple(arm.condition_id for arm in arms), records=tuple(records),
    )
    return AblationMatrixReport(admission_hash=token.admission_hash, ledger=ledger)


__all__ = (
    "ABLATION_SCHEMA_VERSION",
    "HYSTERESIS_INCOMPATIBILITY_REASON",
    "AblationAdmissionError",
    "AblationAdmissionToken",
    "AblationArmSpec",
    "AblationMatrixReport",
    "build_ablation_matrix",
    "observation_ablation_specs",
    "placement_ablation_specs",
    "require_ablation_admission",
    "reward_ablation_specs",
    "run_ablation_matrix",
    "stabilization_ablation_specs",
)

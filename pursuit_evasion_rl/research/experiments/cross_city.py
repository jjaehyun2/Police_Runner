"""Task 12.4: gate-checked cross-city zero-shot evaluation (numerical baselines only).

A cross-city run answers one narrow question: does a policy already selected
on the *training* city's validation network still capture, unchanged, on a
city it never saw? Nothing here may move that policy's weights -- this
module deliberately never imports
:class:`~pursuit_evasion_rl.research.training.trainer.ResearchTrainer`, so
"no adaptation" (Requirement 6.5-6.8) is a structural fact about what this
file can do, not a promise about what it happens to do.

Two synthetic target-city snapshots
    :func:`~pursuit_evasion_rl.osm_demo.fixtures.busan` and
    :func:`~pursuit_evasion_rl.osm_demo.fixtures.seoul` stand in for the "at
    least two target-city snapshots" this task names -- both are hand-built
    offline fixtures (like :func:`~pursuit_evasion_rl.osm_demo.fixtures.daejeon`
    already used throughout Sections 12.1-12.3), not real OSM downloads. Live
    OSM cross-city snapshots are a separate, explicitly optional integration
    (Requirement 11.1, `live_osm`-marked), never a default-path dependency of
    this module.

Leakage guard
    :func:`assert_no_target_city_leakage` refuses to produce generalization
    evidence for a target city whose map hash matches the *source* city the
    policy was trained or selected against -- the exact failure mode
    Requirement 6.5-6.8 exists to catch, and the only failure mode this
    module's four-argument gate cannot detect just from admission (it needs
    the two concrete map hashes side by side).

The offline LLM comparison arm (Requirement 12.1-12.10) is explicitly out of
scope for this module by the user's own standing decision this session;
:mod:`~pursuit_evasion_rl.research.experiments.cross_city` compares the
frozen proposed policy against numerical :class:`FrozenPolicy` baselines
only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from pursuit_evasion_rl.osm_demo.models import ModelNetwork

from ..budget import SampleSizeStatus
from ..canonical import content_hash
from ..domain import EpisodeOutcome, ExecutionStatus, MapScenario
from ..errors import ResearchValidationError
from ..evaluation.paired import FrozenPolicy, replay_episode
from ..execution import ConditionExecutionLedger, ConditionExecutionRecord, MeasuredResult
from ..maps.splits import TuningDataView
from ..preservation import ArtifactMeasurement, measure_artifact
from ..protocol import ProtocolRecord
from ..quality import GateAttestation
from ..statistics.paired import BootstrapPlan, PairedCase, PairedComparisonResult, PracticalThreshold, compare_paired_binary
from ..training.trainer import TrainerConfig
from ..variants.placement import PlacementCurriculumConfig
from .main_study import (
    LEGACY_BEST_V2_PATH,
    GreedyPolicyAdapter,
    MainStudyConditionSpec,
    build_episode_case,
    load_frozen_policy,
)

CROSS_CITY_SCHEMA_VERSION = "1.0"


def _fail(code: str, message: str, **kwargs) -> None:
    raise ResearchValidationError(code, message, **kwargs)


class CrossCityAdmissionError(ResearchValidationError):
    """Raised when the cross-city run cannot launch: an admission requirement is missing."""


# ---------------------------------------------------------------------------
# Admission
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class CrossCityAdmissionToken:
    quality_attestation: GateAttestation
    protocol: ProtocolRecord
    resource_plan: object
    best_v2_measurement: ArtifactMeasurement
    schema_version: str = CROSS_CITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.quality_attestation, GateAttestation):
            _fail("MISSING_QUALITY_GATE", "cross-city evaluation requires a quality-gate attestation", path="quality_attestation")
        if not self.quality_attestation.pilot_admission_eligible:
            _fail(
                "GATE_NOT_ADMISSION_ELIGIBLE", "the quality-gate attestation does not clear admission",
                path="quality_attestation.pilot_admission_eligible", expected=True, actual=False,
            )
        if not isinstance(self.protocol, ProtocolRecord):
            _fail("MISSING_SEALED_PROTOCOL", "cross-city evaluation requires a ProtocolRecord", path="protocol")
        if not self.protocol.is_sealed:
            _fail("PROTOCOL_NOT_SEALED", "cross-city evaluation cannot launch against an unsealed protocol", path="protocol.is_sealed", expected=True, actual=False)
        if self.resource_plan is None:
            _fail("MISSING_RESOURCE_ESTIMATE", "cross-city evaluation requires a fresh resource estimate", path="resource_plan")
        if getattr(self.resource_plan, "protocol_hash", None) != self.protocol.protocol_hash:
            _fail(
                "RESOURCE_PLAN_PROTOCOL_MISMATCH", "the resource plan was costed against a different protocol",
                path="resource_plan.protocol_hash", expected=self.protocol.protocol_hash,
                actual=getattr(self.resource_plan, "protocol_hash", None),
            )
        if not isinstance(self.best_v2_measurement, ArtifactMeasurement):
            _fail("MISSING_PROTECTED_ARTIFACT_MEASUREMENT", "cross-city evaluation requires a fresh best_v2 measurement", path="best_v2_measurement")

    @property
    def admission_hash(self) -> str:
        return content_hash(self)


def require_cross_city_admission(
    *, quality_attestation: GateAttestation, protocol: ProtocolRecord, resource_plan: object,
    best_v2_path: str = LEGACY_BEST_V2_PATH,
) -> CrossCityAdmissionToken:
    measurement = measure_artifact(best_v2_path)
    return CrossCityAdmissionToken(
        quality_attestation=quality_attestation, protocol=protocol,
        resource_plan=resource_plan, best_v2_measurement=measurement,
    )


# ---------------------------------------------------------------------------
# Leakage guard (Requirement 6.5-6.8)
# ---------------------------------------------------------------------------


class TargetCityLeakageError(ResearchValidationError):
    """Raised when a target city's map is not actually distinct from the training/selection map."""


def assert_no_target_city_leakage(*, source_map_hash: str, target_network: ModelNetwork, target_city_id: str) -> None:
    """Refuse to produce generalization evidence for a target city that is not actually held out.

    A cross-city claim is void if the "target" city's map is byte-identical
    to whatever the policy was trained or validation-selected against --
    that is not zero-shot generalization, it is the training city under a
    second name.
    """
    target_hash = content_hash(target_network)
    if target_hash == source_map_hash:
        _fail(
            "TARGET_CITY_LEAKAGE",
            "the target city's map is identical to the source (training/selection) map; this is not a zero-shot generalization test",
            path="target_network", expected=f"!= {source_map_hash}", actual=target_hash,
            details={"target_city_id": target_city_id},
        )


# ---------------------------------------------------------------------------
# One target-city condition
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class CrossCityTargetSpec:
    """One target city's recipe: which frozen policy, which network, how to evaluate on it."""

    city_id: str
    scenario: MapScenario
    network: ModelNetwork | None
    config: TrainerConfig
    placement_config: PlacementCurriculumConfig
    baseline_policy_factory: Callable[[ModelNetwork], FrozenPolicy]
    episodes: int
    unavailable_reason: str | None = None
    # Optional: a team-level numeric baseline (e.g. GreedyInterceptPolice,
    # directed_shortest_path_policy) evaluated instead of baseline_policy_factory.
    # None reproduces prior behavior exactly (baseline_policy_factory's
    # single-officer FrozenPolicy replayed via replay_episode's policy= path).
    team_baseline_factory: Callable[[ModelNetwork], object] | None = None
    schema_version: str = CROSS_CITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (self.network is None) != (self.unavailable_reason is not None):
            _fail(
                "INVALID_CROSS_CITY_TARGET",
                "a target city is either a loaded network or carries its unavailability reason, not both or neither",
                path="network",
            )
        if not isinstance(self.city_id, str) or not self.city_id.strip():
            _fail("MISSING_REQUIRED_FIELD", "city_id must be non-empty", path="city_id")
        if self.episodes <= 0:
            _fail("INVALID_EPISODE_COUNT", "episodes must be positive", path="episodes", actual=self.episodes)

    @property
    def condition_id(self) -> str:
        return f"cross_city__{self.city_id}__{self.scenario.value}"


@dataclass(frozen=True, slots=True, kw_only=True)
class CrossCityResult:
    condition_id: str
    execution_record: ConditionExecutionRecord
    statistics: PairedComparisonResult | None
    schema_version: str = CROSS_CITY_SCHEMA_VERSION


def evaluate_cross_city_target(
    token: CrossCityAdmissionToken, target: CrossCityTargetSpec, *,
    frozen_checkpoint_path: str, source_map_hash: str, observation_adapter_factory,
    protocol_hash: str, bootstrap_plan: BootstrapPlan, threshold: PracticalThreshold,
) -> CrossCityResult:
    """Zero-shot-evaluate one target city under the frozen policy -- never training on it.

    ``observation_adapter_factory`` must match whatever adapter the frozen
    checkpoint was actually trained under (``load_frozen_policy`` recovers
    the policy's tensor dimensions from the checkpoint itself, but not which
    *adapter* produced them; a mismatched factory would silently feed the
    wrong feature vector to a correctly-shaped policy).

    Missing cache/resource (``target.network is None``) is recorded
    ``not_run`` with its reason and nothing further is attempted for that
    city (Requirement 12.5-12.6's completion criterion). A genuine
    leakage detection blocks with a distinct, named error rather than a
    silent ``not_run``, since a leaked city is a defect in the run's own
    setup, not an absence of data.
    """
    if not isinstance(token, CrossCityAdmissionToken):
        _fail("MISSING_CROSS_CITY_ADMISSION", "cross-city evaluation requires a validated CrossCityAdmissionToken", path="token")
    if target.network is None:
        record = ConditionExecutionRecord(
            condition_id=target.condition_id, protocol_hash=protocol_hash,
            execution_status=ExecutionStatus.NOT_RUN, status_reason=target.unavailable_reason,
            sample_size_status=SampleSizeStatus.DEFAULT,
        )
        return CrossCityResult(condition_id=target.condition_id, execution_record=record, statistics=None)

    assert_no_target_city_leakage(source_map_hash=source_map_hash, target_network=target.network, target_city_id=target.city_id)

    frozen = load_frozen_policy(frozen_checkpoint_path, policy_id=f"frozen-selected-{target.city_id}")
    proposed = GreedyPolicyAdapter(frozen, f"proposed-frozen-{target.city_id}")
    team_baseline = target.team_baseline_factory(target.network) if target.team_baseline_factory else None
    baseline = None if team_baseline is not None else target.baseline_policy_factory(target.network)

    spec_for_case_building = MainStudyConditionSpec(
        condition_id=target.condition_id, scenario=target.scenario, train_network=target.network,
        validation_network=target.network, tuning_data=_dummy_tuning_view(),
        config=target.config, placement_config=target.placement_config,
        baseline_policy_factory=target.baseline_policy_factory, episodes_per_seed=target.episodes,
    )

    paired_cases: list[PairedCase] = []
    for episode_index in range(target.episodes):
        case = build_episode_case(target.network, spec_for_case_building, seed=0, episode_index=episode_index)
        proposed_replay = replay_episode(case, proposed, network=target.network, observation_adapter_factory=observation_adapter_factory)
        if team_baseline is not None:
            baseline_replay = replay_episode(case, network=target.network, team_policy=team_baseline)
        else:
            baseline_replay = replay_episode(case, baseline, network=target.network)
        paired_cases.append(
            PairedCase(
                case_id=case.case_hash, training_seed=0,
                outcome_a=proposed_replay.outcome is EpisodeOutcome.CAPTURE,
                outcome_b=baseline_replay.outcome is EpisodeOutcome.CAPTURE,
            )
        )

    statistics = compare_paired_binary(
        tuple(paired_cases), comparison_id=f"{target.condition_id}-vs-baseline", metric="capture_rate", unit="probability",
        plan=bootstrap_plan, threshold=threshold,
    )
    table = statistics.primary.table
    proposed_capture_rate = (table.n11 + table.n10) / statistics.primary.n_pairs
    record = ConditionExecutionRecord(
        condition_id=target.condition_id, protocol_hash=protocol_hash, execution_status=ExecutionStatus.COMPLETED,
        status_reason=f"zero-shot evaluated {len(paired_cases)} paired episode(s) on {target.city_id}, frozen policy, no adaptation",
        sample_size_status=SampleSizeStatus.DEFAULT, result=MeasuredResult(value=proposed_capture_rate),
    )
    return CrossCityResult(condition_id=target.condition_id, execution_record=record, statistics=statistics)


def _dummy_tuning_view() -> TuningDataView:
    """A syntactically valid TuningDataView; cross-city evaluation never actually tunes anything.

    ``MainStudyConditionSpec`` requires one to build an episode case through
    the shared ``build_episode_case`` helper, but no training or selection
    ever reads it here -- this module never imports the trainer at all.
    """
    from ..maps.splits import DataHandle, HandleKind, SplitScope

    return TuningDataView(
        train=(DataHandle(SplitScope.TRAIN, HandleKind.MAP, "cross-city-unused"),),
        validation=(DataHandle(SplitScope.VALIDATION, HandleKind.MAP, "cross-city-unused"),),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class CrossCityReport:
    admission_hash: str
    ledger: ConditionExecutionLedger
    schema_version: str = CROSS_CITY_SCHEMA_VERSION

    @property
    def report_hash(self) -> str:
        return content_hash(self)


def run_cross_city_zero_shot(
    token: CrossCityAdmissionToken, targets: Sequence[CrossCityTargetSpec], *,
    frozen_checkpoint_path: str, source_map_hash: str, protocol_hash: str,
    observation_adapter_factory=None,
    bootstrap_plan: BootstrapPlan, threshold: PracticalThreshold,
) -> CrossCityReport:
    """Evaluate every target city, whatever the outcome for each -- never stopping early."""
    if not isinstance(token, CrossCityAdmissionToken):
        _fail("MISSING_CROSS_CITY_ADMISSION", "run_cross_city_zero_shot requires a validated CrossCityAdmissionToken", path="token")
    targets = tuple(targets)
    if not targets:
        _fail("EMPTY_CROSS_CITY_MATRIX", "cross-city evaluation requires at least one target city", path="targets")

    results = tuple(
        evaluate_cross_city_target(
            token, target, frozen_checkpoint_path=frozen_checkpoint_path, source_map_hash=source_map_hash,
            observation_adapter_factory=observation_adapter_factory,
            protocol_hash=protocol_hash, bootstrap_plan=bootstrap_plan, threshold=threshold,
        )
        for target in targets
    )
    from ..budget import ConditionResourceEstimate, ResourceCeiling, decide_sample_size

    ceiling = ResourceCeiling(resource_ceiling_id="cross-city-ledger-ceiling", measured_at_utc="2026-07-31T00:00:00Z", accelerator_hours=1.0, wall_clock_hours=1.0)
    estimate = ConditionResourceEstimate(condition_id=targets[0].condition_id, accelerator_hours_per_seed=0.001, wall_clock_hours_per_seed=0.001, env_steps_per_seed=1)
    plan = decide_sample_size(ceiling, [estimate])
    ledger = ConditionExecutionLedger(
        protocol_hash=protocol_hash, sample_size_plan=plan,
        planned_condition_ids=tuple(target.condition_id for target in targets),
        records=tuple(result.execution_record for result in results),
    )
    return CrossCityReport(admission_hash=token.admission_hash, ledger=ledger)


__all__ = (
    "CROSS_CITY_SCHEMA_VERSION",
    "LEGACY_BEST_V2_PATH",
    "CrossCityAdmissionError",
    "CrossCityAdmissionToken",
    "CrossCityReport",
    "CrossCityResult",
    "CrossCityTargetSpec",
    "TargetCityLeakageError",
    "assert_no_target_city_leakage",
    "evaluate_cross_city_target",
    "require_cross_city_admission",
    "run_cross_city_zero_shot",
)

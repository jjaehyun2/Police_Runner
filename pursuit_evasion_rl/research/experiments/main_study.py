"""Task 12.2: the multi-seed main-study orchestrator.

A main-study condition trains :attr:`~pursuit_evasion_rl.research.budget.SampleSizePlan.seeds_per_condition`
independent seeds -- gradients only ever come from the *train* network, and
checkpoint selection only ever reads the *validation* network's capture rate,
mirroring :class:`~pursuit_evasion_rl.research.training.trainer.ResearchTrainer`'s
own leakage-safe contract one layer up.  Each seed's *selected* (validation-best,
not merely final) checkpoint is then paired-evaluated against a frozen baseline
policy over :class:`~pursuit_evasion_rl.research.evaluation.paired.EpisodeCase`
replays, and the resulting binary outcomes feed
:func:`~pursuit_evasion_rl.research.statistics.paired.compare_paired_binary`.

``checkpoints/osm_mappo/best_v2.pt`` may enter this pipeline in exactly two
ways, both read-only: as :data:`BestV2Role.FROZEN_BASELINE` (a paired-evaluation
opponent, never trained further) or :data:`BestV2Role.EXPLICIT_INITIALIZATION`
(its weights seed a fresh trainer's policy, which then trains normally). No
code path in this module ever opens ``best_v2.pt`` for writing; the write-side
guarantee is :meth:`ResearchTrainer._validate_output_path`'s existing protected-
path refusal, exercised the same way Task 12.1's pilot already exercised it.

Admission (:func:`require_main_study_admission`) extends the pilot's four-part
gate with one more requirement: the pilot itself must have completed --
:attr:`~pursuit_evasion_rl.research.experiments.pilot.PilotReport.all_completed`
-- so the main study can never launch ahead of the integration check it exists
to precede.

Every planned condition always ends up ``completed``, ``failed``, or
``not_run`` in the returned :class:`~pursuit_evasion_rl.research.execution.ConditionExecutionLedger`
(Requirement 15.8-15.9); a success-only aggregate is never constructed here at
all, matching the ledger's own :func:`~pursuit_evasion_rl.research.execution.success_only_views`
discipline.
"""

from __future__ import annotations

from dataclasses import dataclass, replace as _dataclass_replace
from enum import Enum
from pathlib import Path
import time
import traceback
from typing import Any, Callable, Sequence

import torch

from pursuit_evasion_rl.osm_demo.models import EpisodeConfig, EpisodeOutcome, ModelNetwork

from ..budget import SampleSizePlan, SampleSizeStatus
from ..canonical import content_hash
from ..domain import DataKind, ExecutionStatus, MapScenario
from ..errors import ResearchValidationError
from ..evaluation.paired import (
    EpisodeCase,
    FrozenPolicy,
    FugitiveRNGSpec,
    OfficerObservation,
    TerminationConfig,
    replay_episode,
    seal_episode_case,
    scenario_outcome_domain,
)
from ..execution import ConditionExecutionLedger, ConditionExecutionRecord, MeasuredResult
from ..experiments.pilot import PilotReport
from ..maps.splits import TuningDataView
from ..policies.masked_mappo import ResearchMaskedMAPPO
from ..preservation import ArtifactMeasurement, measure_artifact
from ..protocol import ProtocolRecord
from ..quality import GateAttestation
from ..runs.manifest import RunManifestStore
from ..statistics.paired import BootstrapPlan, PairedCase, PairedComparisonResult, PracticalThreshold, compare_paired_binary
from ..training.trainer import ResearchTrainer, TrainerConfig, TrainingResult
from ..variants.placement import PlacementCurriculumConfig, generate_placement

MAIN_STUDY_SCHEMA_VERSION = "1.0"
LEGACY_BEST_V2_PATH = "checkpoints/osm_mappo/best_v2.pt"


def _fail(code: str, message: str, **kwargs: Any) -> None:
    raise ResearchValidationError(code, message, **kwargs)


class MainStudyAdmissionError(ResearchValidationError):
    """Raised when a main-study launch is refused before any condition starts."""


# ---------------------------------------------------------------------------
# Admission: pilot's four requirements, plus the pilot itself having completed
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class MainStudyAdmissionToken:
    quality_attestation: GateAttestation
    protocol: ProtocolRecord
    pilot_report: PilotReport
    best_v2_measurement: ArtifactMeasurement
    schema_version: str = MAIN_STUDY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.quality_attestation, GateAttestation):
            _fail("MISSING_QUALITY_GATE", "the main study requires a quality-gate attestation", path="quality_attestation")
        if not self.quality_attestation.pilot_admission_eligible:
            _fail(
                "GATE_NOT_ADMISSION_ELIGIBLE", "the quality-gate attestation does not clear admission",
                path="quality_attestation.pilot_admission_eligible", expected=True, actual=False,
            )
        if not isinstance(self.protocol, ProtocolRecord):
            _fail("MISSING_SEALED_PROTOCOL", "the main study requires a ProtocolRecord", path="protocol")
        if not self.protocol.is_sealed:
            _fail("PROTOCOL_NOT_SEALED", "the main study cannot launch against an unsealed protocol", path="protocol.is_sealed", expected=True, actual=False)
        if not isinstance(self.pilot_report, PilotReport):
            _fail("MISSING_PILOT_REPORT", "the main study requires the completed Task 12.1 pilot report", path="pilot_report")
        if not self.pilot_report.all_completed:
            _fail(
                "PILOT_NOT_COMPLETED", "the main study cannot launch ahead of a fully-completed pilot",
                path="pilot_report.all_completed", expected=True, actual=False,
            )
        if not isinstance(self.best_v2_measurement, ArtifactMeasurement):
            _fail("MISSING_PROTECTED_ARTIFACT_MEASUREMENT", "the main study requires a fresh best_v2 measurement", path="best_v2_measurement")

    @property
    def admission_hash(self) -> str:
        return content_hash(self)


def require_main_study_admission(
    *,
    quality_attestation: GateAttestation,
    protocol: ProtocolRecord,
    pilot_report: PilotReport,
    best_v2_path: str = LEGACY_BEST_V2_PATH,
) -> MainStudyAdmissionToken:
    measurement = measure_artifact(best_v2_path)
    return MainStudyAdmissionToken(
        quality_attestation=quality_attestation, protocol=protocol,
        pilot_report=pilot_report, best_v2_measurement=measurement,
    )


# ---------------------------------------------------------------------------
# best_v2: the only two read-only roles it may play
# ---------------------------------------------------------------------------


class BestV2Role(str, Enum):
    FROZEN_BASELINE = "frozen_baseline"
    EXPLICIT_INITIALIZATION = "explicit_initialization"


def load_frozen_policy(checkpoint_path: str | Path, *, policy_id: str) -> ResearchMaskedMAPPO:
    """Reconstruct the policy a checkpoint recorded -- read-only, never the training instance."""
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    policy = ResearchMaskedMAPPO(
        actor_obs_dim=payload["actor_obs_dim"], critic_context_dim=payload["critic_context_dim"],
        action_dim=payload["action_dim"], num_officers=payload["num_officers"],
        hidden_dims=tuple(payload["hidden_dims"]), device="cpu",
    )
    policy.load_state_dict(payload["model"])
    return policy


@dataclass(frozen=True, slots=True)
class GreedyPolicyAdapter:
    """A deterministic, argmax-over-masked-logits :class:`FrozenPolicy` wrapper.

    Deliberately bypasses :meth:`ResearchMaskedMAPPO.sample`'s stochastic
    categorical draw: paired evaluation needs a policy that is a pure function
    of its observation (Requirement 8.1's replay-determinism), and greedy
    argmax is the simplest rule that satisfies that without inventing a new
    RNG-reseeding contract.
    """

    _policy: ResearchMaskedMAPPO
    _policy_id: str

    @property
    def policy_id(self) -> str:
        return self._policy_id

    def act(self, observation: OfficerObservation) -> int:
        obs = torch.tensor(observation.features, dtype=torch.float32).unsqueeze(0)
        officer_ids = torch.tensor([observation.officer_id], dtype=torch.long)
        with torch.no_grad():
            logits = self._policy.actor_logits(obs, officer_ids)[0]
        mask = torch.tensor(observation.action_mask, dtype=torch.bool)
        masked = torch.where(mask, logits, torch.full_like(logits, float("-inf")))
        return int(torch.argmax(masked).item())


# ---------------------------------------------------------------------------
# The condition matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class MainStudyConditionSpec:
    """One planned Condition's full recipe: how to train it and what to compare it against."""

    condition_id: str
    scenario: MapScenario
    train_network: ModelNetwork
    validation_network: ModelNetwork
    tuning_data: TuningDataView
    config: TrainerConfig
    placement_config: PlacementCurriculumConfig
    baseline_policy_factory: Callable[[ModelNetwork], FrozenPolicy]
    episodes_per_seed: int
    best_v2_role: BestV2Role | None = None
    best_v2_path: str = LEGACY_BEST_V2_PATH
    # The three ablation axes 12.2 did not yet need: each None reproduces the
    # trainer's own prior default exactly (see ResearchTrainer.__init__).
    reward_components: object | None = None
    observation_dim: int | None = None
    observation_adapter_factory: object | None = None
    stabilization_condition: object | None = None
    # Defaults to the synthetic-fixture label every prior caller relied on;
    # callers backing conditions with a real ActualOSMSpec network must pass
    # DataKind.ACTUAL_OSM_MAP explicitly so episode provenance stays truthful.
    data_kind: DataKind = DataKind.SYNTHETIC_FIXTURE
    schema_version: str = MAIN_STUDY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.condition_id, str) or not self.condition_id.strip():
            _fail("MISSING_REQUIRED_FIELD", "condition_id must be non-empty", path="condition_id")
        if not isinstance(self.tuning_data, TuningDataView):
            _fail("INVALID_TUNING_VIEW", "a condition spec requires a leakage-safe TuningDataView", path="tuning_data")
        if not isinstance(self.config, TrainerConfig):
            _fail("INVALID_TRAINER_CONFIGURATION", "config must be a TrainerConfig", path="config")
        if self.episodes_per_seed <= 0:
            _fail("INVALID_EPISODE_COUNT", "episodes_per_seed must be positive", path="episodes_per_seed", actual=self.episodes_per_seed)
        if not isinstance(self.data_kind, DataKind):
            _fail("INVALID_DATA_KIND", "data_kind must be a DataKind", path="data_kind", actual=self.data_kind)


def build_episode_case(
    network: ModelNetwork, spec: MainStudyConditionSpec, *, seed: int, episode_index: int,
) -> EpisodeCase:
    """One deterministic, sealed episode both policies will be replayed against identically."""
    stream_id = f"{spec.condition_id}-seed{seed}-ep{episode_index}"
    fugitive_rng_seed = abs(hash((spec.condition_id, seed, episode_index))) % (2**31)
    fugitive_rng = FugitiveRNGSpec(stream_id=stream_id, seed=fugitive_rng_seed)
    placement = generate_placement(network, _dataclass_replace(spec.placement_config, placement_seed=fugitive_rng_seed))
    environment = EpisodeConfig(
        dt_s=1.0, police_speed_mps=spec.config.police_speed_mps, fugitive_speed_mps=spec.config.fugitive_speed_mps,
        capture_radius_m=spec.config.capture_radius_m, max_steps=spec.config.max_steps,
    )
    termination = TerminationConfig(
        outcome_domain=scenario_outcome_domain(spec.scenario), timeout_step_limit=spec.config.max_steps,
    )
    return seal_episode_case(
        network=network, data_kind=spec.data_kind, scenario=spec.scenario,
        police=placement.police, fugitive=placement.fugitive, fugitive_rng=fugitive_rng,
        environment=environment, termination=termination,
    )


# ---------------------------------------------------------------------------
# Per-seed training and paired evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class SeedOutcome:
    condition_id: str
    training_seed: int
    execution_status: ExecutionStatus
    status_reason: str
    run_id: str | None
    paired_cases: tuple[PairedCase, ...] = ()
    error_artifact_hash: str | None = None
    schema_version: str = MAIN_STUDY_SCHEMA_VERSION


def _initial_policy(spec: MainStudyConditionSpec) -> ResearchMaskedMAPPO | None:
    if spec.best_v2_role is not BestV2Role.EXPLICIT_INITIALIZATION:
        return None
    return load_frozen_policy(spec.best_v2_path, policy_id="best_v2_initialization")


def train_and_evaluate_seed(
    token: MainStudyAdmissionToken, spec: MainStudyConditionSpec, seed: int,
    *, runs: RunManifestStore, output_root: str, provenance_factory: Callable[[], Any],
    progress_callback: Callable[[Any], None] | None = None,
) -> SeedOutcome:
    """Train one seed, then paired-evaluate its selected checkpoint against the baseline.

    A real training or replay failure seals ``failed`` and returns immediately;
    it never propagates past this function, mirroring
    :func:`~pursuit_evasion_rl.research.experiments.pilot.execute_pilot_condition`.
    """
    # Structural, not nominal: this function's own contract is only ever
    # ``token.protocol.protocol_hash``. Requiring the concrete
    # MainStudyAdmissionToken type would block reuse by any other admission
    # gate (e.g. the ablation matrix's own AblationAdmissionToken) whose
    # extra requirements are checked at that gate's own construction time.
    if not isinstance(getattr(token, "protocol", None), ProtocolRecord):
        _fail("MISSING_MAIN_STUDY_ADMISSION", "a seed cannot train without an admission token carrying a sealed protocol", path="token")
    draft = runs.create(
        protocol_hash=token.protocol.protocol_hash,
        condition_hash=content_hash({"condition_id": spec.condition_id, "main_study": True}),
        provenance=provenance_factory(),
    )
    started = time.monotonic()
    try:
        trainer = ResearchTrainer(
            train_network=spec.train_network, validation_network=spec.validation_network,
            tuning_data=spec.tuning_data, config=spec.config, condition_id=spec.condition_id,
            training_seed=seed, output_root=output_root, run_id=draft.run_id, policy=_initial_policy(spec),
            reward_components=spec.reward_components, observation_dim=spec.observation_dim,
            observation_adapter_factory=spec.observation_adapter_factory,
            placement_config=spec.placement_config, stabilization_condition=spec.stabilization_condition,
        )
        result: TrainingResult = trainer.train(progress_callback=progress_callback)
        selected = load_frozen_policy(result.checkpoint_path, policy_id=f"{spec.condition_id}-seed{seed}")
        proposed = GreedyPolicyAdapter(selected, f"{spec.condition_id}-proposed")

        baseline_network = spec.validation_network
        baseline = spec.baseline_policy_factory(baseline_network)
        if not isinstance(baseline, FrozenPolicy) and not (hasattr(baseline, "policy_id") and hasattr(baseline, "act")):
            _fail("INVALID_BASELINE_POLICY", "baseline_policy_factory must return a FrozenPolicy", path="baseline_policy_factory")

        paired_cases: list[PairedCase] = []
        for episode_index in range(spec.episodes_per_seed):
            case = build_episode_case(baseline_network, spec, seed=seed, episode_index=episode_index)
            proposed_replay = replay_episode(
                case, proposed, network=baseline_network, observation_adapter_factory=spec.observation_adapter_factory,
            )
            baseline_replay = replay_episode(case, baseline, network=baseline_network)
            paired_cases.append(
                PairedCase(
                    case_id=case.case_hash, training_seed=seed,
                    outcome_a=proposed_replay.outcome is EpisodeOutcome.CAPTURE,
                    outcome_b=baseline_replay.outcome is EpisodeOutcome.CAPTURE,
                )
            )
    except ResearchValidationError as exc:
        trace_hash = content_hash({"error": "".join(traceback.format_exception_only(type(exc), exc))})
        sealed = runs.seal(
            draft.run_id, execution_status=ExecutionStatus.FAILED,
            status_reason=f"{exc.code}: {exc.args[0] if exc.args else exc}",
            artifact_hashes={"error_trace": trace_hash},
        )
        return SeedOutcome(
            condition_id=spec.condition_id, training_seed=seed, execution_status=ExecutionStatus.FAILED,
            status_reason=sealed.run.status_reason or "", run_id=draft.run_id, error_artifact_hash=trace_hash,
        )

    elapsed = time.monotonic() - started
    sealed = runs.seal(
        draft.run_id, execution_status=ExecutionStatus.COMPLETED,
        status_reason=f"seed {seed} trained {len(result.history)} update(s) and paired-evaluated {len(paired_cases)} episode(s) in {elapsed:.1f}s",
        artifact_hashes={"checkpoint": result.manifest_path.name},
    )
    return SeedOutcome(
        condition_id=spec.condition_id, training_seed=seed, execution_status=ExecutionStatus.COMPLETED,
        status_reason=sealed.run.status_reason or "", run_id=draft.run_id, paired_cases=tuple(paired_cases),
    )


# ---------------------------------------------------------------------------
# Condition-level and study-level orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class ConditionStudyResult:
    condition_id: str
    execution_record: ConditionExecutionRecord
    seed_outcomes: tuple[SeedOutcome, ...]
    statistics: PairedComparisonResult | None
    schema_version: str = MAIN_STUDY_SCHEMA_VERSION


def run_main_study_condition(
    token: MainStudyAdmissionToken, spec: MainStudyConditionSpec, *, seeds: int,
    runs: RunManifestStore, output_root: str, provenance_factory: Callable[[], Any],
    bootstrap_plan: BootstrapPlan, threshold: PracticalThreshold,
) -> ConditionStudyResult:
    """Train every planned seed for one Condition and fold the results into one record.

    Every seed always contributes a sealed manifest, whatever its outcome;
    ``execution_record`` reports the Condition ``failed`` only when *every*
    seed failed, and ``completed`` with a null result when at least one seed
    completed but produced no comparable paired case -- a partial failure is
    never silently treated as a full success (Requirement 4.7, 15.8).
    """
    outcomes = tuple(
        train_and_evaluate_seed(token, spec, seed, runs=runs, output_root=output_root, provenance_factory=provenance_factory)
        for seed in range(seeds)
    )
    return pool_seed_outcomes(token, spec, outcomes, bootstrap_plan=bootstrap_plan, threshold=threshold)


def pool_seed_outcomes(
    token: MainStudyAdmissionToken, spec: MainStudyConditionSpec, outcomes: Sequence[SeedOutcome],
    *, bootstrap_plan: BootstrapPlan, threshold: PracticalThreshold,
) -> ConditionStudyResult:
    """Fold already-computed per-seed outcomes into one Condition record.

    This is the pooling half of :func:`run_main_study_condition`, split out so
    seeds trained as independent parallel processes (each producing its own
    :class:`SeedOutcome`) can be combined afterward without re-running
    training sequentially in a single process. Given the same outcomes, this
    produces a byte-identical :class:`ConditionStudyResult` to the in-process path.
    """
    outcomes = tuple(outcomes)
    run_ids = tuple(item.run_id for item in outcomes if item.run_id)
    completed = tuple(item for item in outcomes if item.execution_status is ExecutionStatus.COMPLETED)
    all_cases = tuple(case for item in completed for case in item.paired_cases)

    if not completed:
        record = ConditionExecutionRecord(
            condition_id=spec.condition_id, protocol_hash=token.protocol.protocol_hash,
            execution_status=ExecutionStatus.FAILED,
            status_reason=f"all {len(outcomes)} planned seed(s) failed",
            sample_size_status=SampleSizeStatus.DEFAULT,
            run_ids=run_ids,
        )
        return ConditionStudyResult(condition_id=spec.condition_id, execution_record=record, seed_outcomes=outcomes, statistics=None)

    if not all_cases:
        record = ConditionExecutionRecord(
            condition_id=spec.condition_id, protocol_hash=token.protocol.protocol_hash,
            execution_status=ExecutionStatus.COMPLETED,
            status_reason=f"{len(completed)}/{len(outcomes)} seed(s) completed but produced no comparable episode",
            sample_size_status=SampleSizeStatus.DEFAULT, result=MeasuredResult(value=None), run_ids=run_ids,
        )
        return ConditionStudyResult(condition_id=spec.condition_id, execution_record=record, seed_outcomes=outcomes, statistics=None)

    statistics = compare_paired_binary(
        all_cases, comparison_id=f"{spec.condition_id}-vs-baseline", metric="capture_rate", unit="probability",
        plan=bootstrap_plan, threshold=threshold,
    )
    # The proposed policy's own capture rate: pairs where it captured, whether
    # or not the baseline also did (table.n11 + table.n10), over all pairs.
    table = statistics.primary.table
    proposed_capture_rate = (table.n11 + table.n10) / statistics.primary.n_pairs
    record = ConditionExecutionRecord(
        condition_id=spec.condition_id, protocol_hash=token.protocol.protocol_hash,
        execution_status=ExecutionStatus.COMPLETED,
        status_reason=f"{len(completed)}/{len(outcomes)} seed(s) completed, {len(all_cases)} paired episode(s)",
        sample_size_status=SampleSizeStatus.DEFAULT,
        result=MeasuredResult(value=proposed_capture_rate),
        run_ids=run_ids,
    )
    return ConditionStudyResult(condition_id=spec.condition_id, execution_record=record, seed_outcomes=outcomes, statistics=statistics)


@dataclass(frozen=True, slots=True, kw_only=True)
class MainStudyReport:
    admission_hash: str
    ledger: ConditionExecutionLedger
    condition_results: tuple[ConditionStudyResult, ...]
    schema_version: str = MAIN_STUDY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "condition_results", tuple(self.condition_results))

    @property
    def report_hash(self) -> str:
        return content_hash(self)


def run_main_study(
    token: MainStudyAdmissionToken, specs: Sequence[MainStudyConditionSpec], *,
    sample_size_plan: SampleSizePlan, runs: RunManifestStore, output_root: str, provenance_factory: Callable[[], Any],
    bootstrap_plan: BootstrapPlan, threshold: PracticalThreshold,
) -> MainStudyReport:
    """Run every planned Condition and assemble the conserved execution ledger.

    Every planned Condition gets a record regardless of the others' outcomes --
    the study never stops early on one Condition's failure, matching Task
    12.1's pilot discipline one layer up the execution hierarchy.

    ``sample_size_plan`` is the caller's own pre-launch
    :class:`~pursuit_evasion_rl.research.budget.SampleSizePlan` decision (the
    5/500 -> 3/100 -> exploratory rule); every Condition trains exactly
    ``sample_size_plan.seeds_per_condition`` seeds, and the same plan is what
    the returned ledger records, so the ledger can never silently diverge from
    the sample size that actually governed this run.
    """
    if not isinstance(token, MainStudyAdmissionToken):
        _fail("MISSING_MAIN_STUDY_ADMISSION", "run_main_study requires a validated MainStudyAdmissionToken", path="token")
    specs = tuple(specs)
    if not specs:
        _fail("EMPTY_CONDITION_MATRIX", "the main study requires at least one planned condition", path="specs")

    results = tuple(
        run_main_study_condition(
            token, spec, seeds=sample_size_plan.seeds_per_condition, runs=runs, output_root=output_root,
            provenance_factory=provenance_factory, bootstrap_plan=bootstrap_plan, threshold=threshold,
        )
        for spec in specs
    )
    ledger = ConditionExecutionLedger(
        protocol_hash=token.protocol.protocol_hash, sample_size_plan=sample_size_plan,
        planned_condition_ids=tuple(spec.condition_id for spec in specs),
        records=tuple(result.execution_record for result in results),
    )
    return MainStudyReport(admission_hash=token.admission_hash, ledger=ledger, condition_results=results)


__all__ = (
    "LEGACY_BEST_V2_PATH",
    "MAIN_STUDY_SCHEMA_VERSION",
    "BestV2Role",
    "ConditionStudyResult",
    "GreedyPolicyAdapter",
    "MainStudyAdmissionError",
    "MainStudyAdmissionToken",
    "MainStudyConditionSpec",
    "MainStudyReport",
    "SeedOutcome",
    "build_episode_case",
    "load_frozen_policy",
    "pool_seed_outcomes",
    "require_main_study_admission",
    "run_main_study",
    "run_main_study_condition",
    "train_and_evaluate_seed",
)

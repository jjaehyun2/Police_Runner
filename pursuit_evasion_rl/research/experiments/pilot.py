"""Task 12.1: the protocol-blind pilot runner.

A pilot exists to catch integration defects -- a broken config, a network the
trainer cannot place six officers on, a checkpoint path collision -- before any
condition in the main study spends real compute. It is deliberately blind to
outcome: nothing here reads a pilot's capture rate and nothing here adjusts a
Practical_Threshold, a sample-size decision, or any other confirmatory
parameter. A pilot's only two effects are (a) a sealed
:class:`~pursuit_evasion_rl.research.runs.manifest.RunManifest` per condition,
whatever its status, and (b) the fail-closed refusal to launch at all when its
admission requirements are not met.

Admission (:func:`require_pilot_admission`) requires, together, in one
:class:`PilotAdmissionToken`:

* a quality-gate attestation whose ``pilot_admission_eligible`` is true
  (Task 10.1's gate -- every formal property covered, the offline suite green);
* a sealed :class:`~pursuit_evasion_rl.research.protocol.ProtocolRecord`;
* a pre-launch :class:`~pursuit_evasion_rl.research.execution.ExecutionPlan`
  costing the sanity matrix against a stated resource ceiling;
* a freshly re-measured hash of the legacy ``checkpoints/osm_mappo/best_v2.pt``
  artifact, captured at admission time rather than accepted as a caller-typed
  value.  This module never writes to that path -- the trainer's own
  ``_validate_output_path`` protected-path check is the write-side guarantee
  -- and does not itself compare the measurement against a prior recorded
  hash; a caller that already holds a preserved-copy record from
  :mod:`~pursuit_evasion_rl.research.preservation` compares this measurement
  against it separately.

Missing any one of the four is zero launches, not a partial run
(Requirement 2.3-2.6, 7.1-7.5, 14.1-15.9, 19.9-19.11).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
import traceback
from typing import Any, Sequence

from ..canonical import content_hash
from ..domain import ExecutionStatus
from ..errors import ResearchValidationError
from ..execution import ExecutionPlan, estimate_condition_cost, plan_execution
from ..preservation import ArtifactMeasurement, measure_artifact
from ..protocol import ProtocolRecord
from ..quality import GateAttestation
from ..runs.manifest import RunManifest, RunManifestStore
from ..training.trainer import ResearchTrainer, TrainerConfig, TrainingResult
from ..maps.splits import TuningDataView

PILOT_SCHEMA_VERSION = "1.0"

#: Task 12.1's own protected path, restated here rather than imported from
#: ``training.trainer`` so a caller cannot satisfy this module's admission
#: check while pointing the trainer somewhere else entirely.
LEGACY_BEST_V2_PATH = "checkpoints/osm_mappo/best_v2.pt"


def _fail(code: str, message: str, **kwargs: Any) -> None:
    raise ResearchValidationError(code, message, **kwargs)


class PilotAdmissionError(ResearchValidationError):
    """Raised when a pilot cannot launch: one of the four admission requirements is missing."""


@dataclass(frozen=True, slots=True, kw_only=True)
class PilotAdmissionToken:
    """Proof every Task 12.1 launch precondition was checked before any run started.

    Constructing this object *is* the admission decision: every field is
    validated in ``__post_init__``, so a token in hand is the only thing
    :func:`execute_pilot_condition` will accept.
    """

    quality_attestation: GateAttestation
    protocol: ProtocolRecord
    resource_plan: ExecutionPlan
    best_v2_measurement: ArtifactMeasurement
    schema_version: str = PILOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.quality_attestation, GateAttestation):
            _fail("MISSING_QUALITY_GATE", "a pilot requires a quality-gate attestation", path="quality_attestation")
        if not self.quality_attestation.pilot_admission_eligible:
            _fail(
                "GATE_NOT_ADMISSION_ELIGIBLE",
                "the quality-gate attestation does not clear pilot admission",
                path="quality_attestation.pilot_admission_eligible",
                expected=True, actual=False,
            )
        if not isinstance(self.protocol, ProtocolRecord):
            _fail("MISSING_SEALED_PROTOCOL", "a pilot requires a ProtocolRecord", path="protocol")
        if not self.protocol.is_sealed:
            _fail(
                "PROTOCOL_NOT_SEALED", "a pilot cannot launch against an unsealed protocol",
                path="protocol.is_sealed", expected=True, actual=False,
            )
        if not isinstance(self.resource_plan, ExecutionPlan):
            _fail("MISSING_RESOURCE_ESTIMATE", "a pilot requires a pre-launch ExecutionPlan", path="resource_plan")
        if self.resource_plan.protocol_hash != self.protocol.protocol_hash:
            _fail(
                "RESOURCE_PLAN_PROTOCOL_MISMATCH",
                "the resource plan was costed against a different protocol",
                path="resource_plan.protocol_hash",
                expected=self.protocol.protocol_hash, actual=self.resource_plan.protocol_hash,
            )
        if not isinstance(self.best_v2_measurement, ArtifactMeasurement):
            _fail("MISSING_PROTECTED_ARTIFACT_MEASUREMENT", "a pilot requires a fresh best_v2 measurement", path="best_v2_measurement")

    @property
    def admission_hash(self) -> str:
        return content_hash(self)


def require_pilot_admission(
    *,
    quality_attestation: GateAttestation,
    protocol: ProtocolRecord,
    resource_plan: ExecutionPlan,
    best_v2_path: str = LEGACY_BEST_V2_PATH,
) -> PilotAdmissionToken:
    """Assemble and validate the admission token, or raise before anything launches.

    ``best_v2_path`` is re-measured here, freshly, rather than accepting a
    caller-supplied hash: the whole point of Requirement 2.3-2.6 is that this
    check cannot be satisfied by a stale or hand-typed value.
    """
    measurement = measure_artifact(best_v2_path)
    return PilotAdmissionToken(
        quality_attestation=quality_attestation, protocol=protocol,
        resource_plan=resource_plan, best_v2_measurement=measurement,
    )


# ---------------------------------------------------------------------------
# The sanity matrix and its execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class PilotConditionSpec:
    """One condition of the small, train/validation-only sanity matrix."""

    condition_id: str
    train_network: Any
    validation_network: Any
    tuning_data: TuningDataView
    config: TrainerConfig
    training_seed: int
    schema_version: str = PILOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.condition_id, str) or not self.condition_id.strip():
            _fail("MISSING_REQUIRED_FIELD", "condition_id must be non-empty", path="condition_id")
        if not isinstance(self.tuning_data, TuningDataView):
            _fail("INVALID_TUNING_VIEW", "a pilot condition requires a leakage-safe TuningDataView", path="tuning_data")
        if not isinstance(self.config, TrainerConfig):
            _fail("INVALID_TRAINER_CONFIGURATION", "config must be a TrainerConfig", path="config")


def cost_pilot_matrix(
    specs: Sequence[PilotConditionSpec],
    *,
    protocol_hash: str,
    ceiling,
    evaluation_episodes: int,
    accelerator_hours_per_million_env_steps: float,
    wall_clock_hours_per_million_env_steps: float,
) -> ExecutionPlan:
    """Price the pilot's own tiny matrix -- Requirement 7.1-7.5's estimate-before-launch."""
    estimates = tuple(
        estimate_condition_cost(
            condition_id=spec.condition_id, schedule=spec.config, evaluation_episodes=evaluation_episodes,
            accelerator_hours_per_million_env_steps=accelerator_hours_per_million_env_steps,
            wall_clock_hours_per_million_env_steps=wall_clock_hours_per_million_env_steps,
        )
        for spec in specs
    )
    return plan_execution(protocol_hash=protocol_hash, ceiling=ceiling, cost_estimates=estimates)


@dataclass(frozen=True, slots=True, kw_only=True)
class PilotConditionOutcome:
    """One condition's terminal pilot state -- never presupposing success."""

    condition_id: str
    execution_status: ExecutionStatus
    status_reason: str
    run_id: str | None
    resource_actual: dict[str, float] = field(default_factory=dict)
    error_artifact_hash: str | None = None
    schema_version: str = PILOT_SCHEMA_VERSION


def execute_pilot_condition(
    token: PilotAdmissionToken,
    spec: PilotConditionSpec,
    *,
    runs: RunManifestStore,
    output_root: str,
    provenance_factory,
) -> tuple[PilotConditionOutcome, RunManifest]:
    """Run exactly one pilot condition and seal its manifest, whatever the outcome.

    ``provenance_factory`` is a zero-argument callable returning a fresh
    :class:`RunProvenance` for this condition (code/dependency hashes, device,
    map hash) -- supplied by the caller because those identities are
    environment facts this module has no business fabricating.
    """
    if not isinstance(token, PilotAdmissionToken):
        _fail("MISSING_PILOT_ADMISSION", "a pilot condition cannot execute without an admission token", path="token")

    draft = runs.create(
        protocol_hash=token.protocol.protocol_hash,
        condition_hash=content_hash({"condition_id": spec.condition_id, "pilot": True}),
        provenance=provenance_factory(),
    )
    started = time.monotonic()
    try:
        trainer = ResearchTrainer(
            train_network=spec.train_network, validation_network=spec.validation_network,
            tuning_data=spec.tuning_data, config=spec.config, condition_id=spec.condition_id,
            training_seed=spec.training_seed, output_root=output_root, run_id=draft.run_id,
        )
        result: TrainingResult = trainer.train()
    except ResearchValidationError as exc:
        elapsed = time.monotonic() - started
        trace_hash = content_hash({"error": "".join(traceback.format_exception_only(type(exc), exc))})
        sealed = runs.seal(
            draft.run_id, execution_status=ExecutionStatus.FAILED,
            status_reason=f"{exc.code}: {exc.args[0] if exc.args else exc}",
            artifact_hashes={"error_trace": trace_hash},
        )
        outcome = PilotConditionOutcome(
            condition_id=spec.condition_id, execution_status=ExecutionStatus.FAILED,
            status_reason=sealed.run.status_reason or "", run_id=draft.run_id,
            resource_actual={"wall_clock_hours": elapsed / 3600.0},
            error_artifact_hash=trace_hash,
        )
        return outcome, sealed

    elapsed = time.monotonic() - started
    sealed = runs.seal(
        draft.run_id, execution_status=ExecutionStatus.COMPLETED,
        status_reason=f"pilot condition ran {len(result.history)} update(s) to completion",
        artifact_hashes={"checkpoint": result.manifest_path.name},
    )
    outcome = PilotConditionOutcome(
        condition_id=spec.condition_id, execution_status=ExecutionStatus.COMPLETED,
        status_reason=sealed.run.status_reason or "", run_id=draft.run_id,
        resource_actual={"wall_clock_hours": elapsed / 3600.0},
    )
    return outcome, sealed


@dataclass(frozen=True, slots=True, kw_only=True)
class PilotReport:
    """Every pilot condition's terminal outcome, content-addressed as a whole."""

    admission_hash: str
    outcomes: tuple[PilotConditionOutcome, ...]
    schema_version: str = PILOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcomes", tuple(self.outcomes))
        if not self.outcomes:
            _fail("EMPTY_PILOT_MATRIX", "a pilot report requires at least one condition outcome", path="outcomes")

    @property
    def all_completed(self) -> bool:
        return all(item.execution_status is ExecutionStatus.COMPLETED for item in self.outcomes)

    @property
    def report_hash(self) -> str:
        return content_hash(self)


def run_pilot(
    token: PilotAdmissionToken,
    specs: Sequence[PilotConditionSpec],
    *,
    runs: RunManifestStore,
    output_root: str,
    provenance_factory,
) -> PilotReport:
    """Execute every condition in the sanity matrix and seal a manifest for each.

    A pilot never stops early on a failure: every planned condition gets its
    own sealed outcome, exactly like the main study's execution ledger later
    requires (Requirement 15.8-15.9) -- a pilot is where that discipline is
    first exercised for real.
    """
    if not isinstance(token, PilotAdmissionToken):
        _fail("MISSING_PILOT_ADMISSION", "run_pilot requires a validated PilotAdmissionToken", path="token")
    specs = tuple(specs)
    if not specs:
        _fail("EMPTY_PILOT_MATRIX", "a pilot requires at least one condition in its sanity matrix", path="specs")
    outcomes = []
    for spec in specs:
        outcome, _manifest = execute_pilot_condition(
            token, spec, runs=runs, output_root=output_root, provenance_factory=provenance_factory,
        )
        outcomes.append(outcome)
    return PilotReport(admission_hash=token.admission_hash, outcomes=tuple(outcomes))


__all__ = (
    "LEGACY_BEST_V2_PATH",
    "PILOT_SCHEMA_VERSION",
    "PilotAdmissionError",
    "PilotAdmissionToken",
    "PilotConditionOutcome",
    "PilotConditionSpec",
    "PilotReport",
    "cost_pilot_matrix",
    "execute_pilot_condition",
    "require_pilot_admission",
    "run_pilot",
)

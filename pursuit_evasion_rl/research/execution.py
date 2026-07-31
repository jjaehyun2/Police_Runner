"""Resource estimation and the Condition execution ledger (Requirements 4.6-4.7, 7.1-7.8, 8.7, 15.8-15.9).

Two layers live here, both strictly *above* existing modules rather than
duplicating them:

Resource estimation
    :mod:`pursuit_evasion_rl.research.budget` already meters accelerator-hours
    and wall-clock-hours per Training_Seed and already owns the pre-registered
    5 seeds / 500 episodes -> reduced 3 / 100 -> exploratory decision
    (:func:`~pursuit_evasion_rl.research.budget.decide_sample_size`).  This
    module derives those per-seed hours from an environment-step count and adds
    the one cost dimension the budget model has no slot for -- LLM call cost --
    then hands the resulting
    :class:`~pursuit_evasion_rl.research.budget.ConditionResourceEstimate`
    straight back to ``decide_sample_size``.  The sample-size rule is never
    re-implemented here.

Condition execution ledger
    :class:`~pursuit_evasion_rl.research.runs.manifest.RunManifest` records one
    run.  A Condition is executed as several seed-replicate runs, so the ledger
    is one level up: exactly one
    :class:`ConditionExecutionRecord` per planned Condition, carrying one
    :class:`~pursuit_evasion_rl.research.domain.ExecutionStatus` chosen through
    :func:`~pursuit_evasion_rl.research.domain.exactly_one`, an always-required
    reason, and the artifact hashes and measured result -- which survive
    ``failed`` and ``not_run`` exactly as they survive ``completed``.

The two disciplines the ledger exists to enforce:

* **A null or unfavorable result is not a failure** (Requirement 4.7).  A
  Condition that ran and produced no measurable primary outcome stays
  ``completed`` with ``result=MeasuredResult(None)``.  That is deliberately a
  different value from ``result=None``, which means *never measured*.
* **A success-only aggregate can never stand alone** (Requirement 15.9).
  :func:`success_only_views` is the only way to obtain one and it returns the
  subset already paired with the full-population view and already classified
  ``exploratory``; the pairing cannot be constructed with any other
  classification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Protocol, Sequence

from pursuit_evasion_rl.research.budget import (
    ConditionResourceEstimate,
    ResourceCeiling,
    SampleSizePlan,
    SampleSizeStatus,
    decide_sample_size,
)
from pursuit_evasion_rl.research.canonical import canonical_data, content_hash
from pursuit_evasion_rl.research.domain import (
    AnalysisClassification,
    ExecutionStatus,
    exactly_one,
)
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.variants.factory import ConditionSpec

EXECUTION_SCHEMA_VERSION = "1.0"

# Per-seed hours are quoted per million environment steps; this is only a unit,
# not a claim about throughput, and the caller supplies the measured rate.
ENV_STEPS_PER_RATE_UNIT = 1_000_000


def _fail(code: str, message: str, **kwargs) -> None:
    raise ResearchValidationError(code, message, **kwargs)


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("MISSING_REQUIRED_FIELD", f"{name} must be a non-empty string", path=name, actual=value)
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail("INVALID_RESOURCE_VALUE", f"{name} must be a positive integer", path=name, actual=value)
    return value


def _nonnegative_finite(value: Any, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        _fail("INVALID_RESOURCE_VALUE", f"{name} must be finite and nonnegative", path=name, actual=value)
    return numeric


def _positive_finite(value: Any, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        _fail("INVALID_RESOURCE_VALUE", f"{name} must be finite and positive", path=name, actual=value)
    return numeric


def _freeze_hash_map(value: Mapping[str, str], name: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) or not isinstance(item, str) or not key.strip() or not item.strip()
        for key, item in value.items()
    ):
        _fail("INVALID_HASH_MAP", f"{name} must map non-empty string keys to non-empty string hashes", path=name)
    return MappingProxyType(dict(value))


# ---------------------------------------------------------------------------
# Resource estimation
# ---------------------------------------------------------------------------


class TrainingSchedule(Protocol):
    """The three fields of a ``TrainerConfig`` that set the environment-step count.

    Typed structurally so this module never imports the trainer (and therefore
    never pulls in torch) merely to price a Condition.
    """

    updates: int
    episodes_per_update: int
    max_steps: int


@dataclass(frozen=True, slots=True)
class LlmCostEstimate:
    """Per-Training_Seed LLM call cost for a Condition that consults one.

    Conditions with no LLM in the loop carry ``llm_cost=None`` on their
    :class:`ConditionCostEstimate` -- an explicit "this Condition uses no LLM",
    distinct from a zero-valued estimate meaning "it uses one, priced at zero".
    """

    provider: str
    calls_per_seed: int
    usd_per_call: float
    schema_version: str = EXECUTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _required_text(self.provider, "provider")
        calls = self.calls_per_seed
        if isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
            _fail("INVALID_RESOURCE_VALUE", "calls_per_seed must be a nonnegative integer", path="calls_per_seed", actual=calls)
        object.__setattr__(self, "usd_per_call", _nonnegative_finite(self.usd_per_call, "usd_per_call"))

    @property
    def usd_per_seed(self) -> float:
        return self.calls_per_seed * self.usd_per_call

    @property
    def config_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class ConditionCostEstimate:
    """A :class:`ConditionResourceEstimate` plus the optional LLM cost dimension.

    The accelerator/wall-clock/environment-step figures stay in the budget
    module's own dataclass so :func:`decide_sample_size` consumes them
    unchanged; this wrapper only carries the extra dimension.
    """

    resource: ConditionResourceEstimate
    llm_cost: LlmCostEstimate | None = None
    schema_version: str = EXECUTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.resource, ConditionResourceEstimate):
            _fail("INVALID_COST_ESTIMATE", "resource must be a ConditionResourceEstimate", path="resource")
        if self.llm_cost is not None and not isinstance(self.llm_cost, LlmCostEstimate):
            _fail("INVALID_COST_ESTIMATE", "llm_cost must be an LlmCostEstimate or None", path="llm_cost")

    @property
    def condition_id(self) -> str:
        return self.resource.condition_id

    @property
    def env_steps_per_seed(self) -> int:
        return self.resource.env_steps_per_seed

    @property
    def accelerator_hours_per_seed(self) -> float:
        return self.resource.accelerator_hours_per_seed

    @property
    def wall_clock_hours_per_seed(self) -> float:
        return self.resource.wall_clock_hours_per_seed

    @property
    def llm_usd_per_seed(self) -> float | None:
        """``None`` means this Condition consults no LLM, not "cost unknown"."""
        return None if self.llm_cost is None else self.llm_cost.usd_per_seed

    @property
    def config_hash(self) -> str:
        return content_hash(self)


def estimate_environment_steps(schedule: TrainingSchedule, *, evaluation_episodes: int) -> int:
    """Environment steps one Training_Seed of a Condition costs, training plus evaluation.

    ``max_steps`` bounds an episode, so this is an upper bound: an episode that
    terminates early costs fewer steps.  Planning deliberately uses the bound so
    a Resource_Ceiling decision is never optimistic.
    """
    updates = _positive_int(getattr(schedule, "updates", None), "updates")
    episodes_per_update = _positive_int(getattr(schedule, "episodes_per_update", None), "episodes_per_update")
    max_steps = _positive_int(getattr(schedule, "max_steps", None), "max_steps")
    _positive_int(evaluation_episodes, "evaluation_episodes")
    return (updates * episodes_per_update + evaluation_episodes) * max_steps


def estimate_condition_cost(
    *,
    condition_id: str,
    schedule: TrainingSchedule,
    evaluation_episodes: int,
    accelerator_hours_per_million_env_steps: float,
    wall_clock_hours_per_million_env_steps: float,
    llm_cost: LlmCostEstimate | None = None,
) -> ConditionCostEstimate:
    """Price one Condition's Training_Seed from its schedule and measured throughput.

    ``evaluation_episodes`` is a *planning* input (the pre-registered default
    episode count), not a result: the estimate must be fixed before any
    Condition runs so :func:`decide_sample_size` stays a pure function of the
    ceiling and the estimates (Requirement 7.3-7.4).
    """
    _required_text(condition_id, "condition_id")
    env_steps = estimate_environment_steps(schedule, evaluation_episodes=evaluation_episodes)
    accelerator_rate = _positive_finite(
        accelerator_hours_per_million_env_steps, "accelerator_hours_per_million_env_steps"
    )
    wall_clock_rate = _positive_finite(
        wall_clock_hours_per_million_env_steps, "wall_clock_hours_per_million_env_steps"
    )
    scale = env_steps / ENV_STEPS_PER_RATE_UNIT
    return ConditionCostEstimate(
        resource=ConditionResourceEstimate(
            condition_id=condition_id,
            accelerator_hours_per_seed=scale * accelerator_rate,
            wall_clock_hours_per_seed=scale * wall_clock_rate,
            env_steps_per_seed=env_steps,
        ),
        llm_cost=llm_cost,
    )


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """The pre-result execution decision for a whole Condition matrix."""

    protocol_hash: str
    sample_size_plan: SampleSizePlan
    cost_estimates: tuple[ConditionCostEstimate, ...]
    schema_version: str = EXECUTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _required_text(self.protocol_hash, "protocol_hash")
        estimates = tuple(self.cost_estimates)
        if not estimates:
            _fail("EMPTY_CONDITION_ESTIMATES", "an execution plan requires at least one condition cost estimate")
        ids = [estimate.condition_id for estimate in estimates]
        if len(ids) != len(set(ids)):
            _fail(
                "DUPLICATE_CONDITION_ID", "condition cost estimates must be unique per condition",
                path="cost_estimates", actual=sorted({item for item in ids if ids.count(item) > 1}),
            )
        object.__setattr__(self, "cost_estimates", estimates)

    @property
    def planned_condition_ids(self) -> tuple[str, ...]:
        return tuple(estimate.condition_id for estimate in self.cost_estimates)

    @property
    def total_env_steps(self) -> int:
        return sum(
            estimate.env_steps_per_seed for estimate in self.cost_estimates
        ) * self.sample_size_plan.seeds_per_condition

    @property
    def total_llm_usd(self) -> float | None:
        """Total LLM spend, or ``None`` when no Condition in the matrix uses one."""
        priced = [estimate.llm_usd_per_seed for estimate in self.cost_estimates if estimate.llm_cost is not None]
        if not priced:
            return None
        return sum(priced) * self.sample_size_plan.seeds_per_condition

    @property
    def is_confirmatory_eligible(self) -> bool:
        return self.sample_size_plan.is_confirmatory_eligible

    @property
    def plan_hash(self) -> str:
        return content_hash(self)


def plan_execution(
    *,
    protocol_hash: str,
    ceiling: ResourceCeiling,
    cost_estimates: Sequence[ConditionCostEstimate],
) -> ExecutionPlan:
    """Choose the matrix-wide sample size the ceiling affords, before any result exists.

    The 5/500 -> 3/100 -> exploratory rule itself belongs to
    :func:`~pursuit_evasion_rl.research.budget.decide_sample_size`; this only
    supplies it the per-Condition estimates and binds the outcome to a
    ``protocol_hash``.
    """
    _required_text(protocol_hash, "protocol_hash")
    estimates = tuple(cost_estimates)
    if not estimates:
        _fail("EMPTY_CONDITION_ESTIMATES", "at least one condition cost estimate is required")
    plan = decide_sample_size(ceiling, [estimate.resource for estimate in estimates])
    return ExecutionPlan(protocol_hash=protocol_hash, sample_size_plan=plan, cost_estimates=estimates)


def cost_estimates_for_matrix(
    specs: Sequence[ConditionSpec],
    *,
    schedule: TrainingSchedule,
    evaluation_episodes: int,
    accelerator_hours_per_million_env_steps: float,
    wall_clock_hours_per_million_env_steps: float,
    llm_cost: LlmCostEstimate | None = None,
) -> tuple[ConditionCostEstimate, ...]:
    """Price every Condition in a matrix under one shared schedule and throughput."""
    if not specs:
        _fail("EMPTY_CONDITION_MATRIX", "a condition matrix must not be empty")
    return tuple(
        estimate_condition_cost(
            condition_id=spec.condition.condition_id,
            schedule=schedule,
            evaluation_episodes=evaluation_episodes,
            accelerator_hours_per_million_env_steps=accelerator_hours_per_million_env_steps,
            wall_clock_hours_per_million_env_steps=wall_clock_hours_per_million_env_steps,
            llm_cost=llm_cost,
        )
        for spec in specs
    )


# ---------------------------------------------------------------------------
# Condition execution ledger
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MeasuredResult:
    """A primary outcome that *was* measured, whose value may itself be null.

    ``MeasuredResult(None)`` -- the Condition ran and produced no measurable
    primary outcome -- is a different fact from a record's ``result=None``,
    which means the outcome was never measured at all.  Requirement 4.7 forbids
    collapsing the former into a failure, so the two are kept distinguishable in
    the type rather than by convention.
    """

    value: Any = None
    schema_version: str = EXECUTION_SCHEMA_VERSION

    @property
    def is_null(self) -> bool:
        return self.value is None


@dataclass(frozen=True, slots=True)
class ConditionExecutionRecord:
    """One planned Condition's terminal execution state (Requirements 4.6-4.7, 15.8).

    ``artifact_hashes`` and ``result`` are retained for every status: a failed
    or interrupted Condition keeps its partial-log or error-trace hash so the
    failure stays auditable rather than becoming an absence.
    """

    condition_id: str
    protocol_hash: str
    execution_status: ExecutionStatus
    status_reason: str
    sample_size_status: SampleSizeStatus
    artifact_hashes: Mapping[str, str] = field(default_factory=dict)
    result: MeasuredResult | None = None
    run_ids: tuple[str, ...] = ()
    schema_version: str = EXECUTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _required_text(self.condition_id, "condition_id")
        _required_text(self.protocol_hash, "protocol_hash")
        # Requirement 4.6: exactly one of completed/failed/not_run, always with a reason.
        object.__setattr__(
            self, "execution_status", exactly_one(self.execution_status, ExecutionStatus, path="execution_status")
        )
        object.__setattr__(
            self,
            "sample_size_status",
            exactly_one(self.sample_size_status, SampleSizeStatus, path="sample_size_status"),
        )
        _required_text(self.status_reason, "status_reason")
        object.__setattr__(self, "artifact_hashes", _freeze_hash_map(self.artifact_hashes, "artifact_hashes"))
        if self.result is not None and not isinstance(self.result, MeasuredResult):
            _fail(
                "INVALID_EXECUTION_RESULT",
                "result must be a MeasuredResult (possibly wrapping None) or None for never-measured",
                path="result", actual=type(self.result).__name__,
            )
        if self.execution_status is ExecutionStatus.COMPLETED and self.result is None:
            _fail(
                "MISSING_COMPLETED_RESULT",
                "a completed Condition must record its measured outcome, even when null or unfavorable",
                path="result", actual=None,
            )
        object.__setattr__(self, "run_ids", tuple(str(item) for item in self.run_ids))

    @property
    def is_measured(self) -> bool:
        return self.result is not None

    @property
    def result_value(self) -> Any:
        """The measured value; raises when nothing was ever measured."""
        if self.result is None:
            _fail(
                "RESULT_NEVER_MEASURED", "this Condition has no measured outcome to read",
                path="result", actual=self.condition_id,
            )
        return self.result.value

    @property
    def is_confirmatory_eligible(self) -> bool:
        return self.sample_size_status is not SampleSizeStatus.EXPLORATORY

    @property
    def record_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class ConditionExecutionLedger:
    """Conserved planned-vs-executed accounting for one Condition matrix.

    Every planned Condition needs its own record, including one that never ran
    -- that is what ``not_run`` is for -- so a missing row is an error rather
    than a silently smaller denominator, mirroring
    :func:`~pursuit_evasion_rl.research.metrics.physical.account_intention_to_evaluate`
    at Condition granularity.
    """

    protocol_hash: str
    sample_size_plan: SampleSizePlan
    planned_condition_ids: tuple[str, ...]
    records: tuple[ConditionExecutionRecord, ...]
    schema_version: str = EXECUTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _required_text(self.protocol_hash, "protocol_hash")
        planned = tuple(str(item) for item in self.planned_condition_ids)
        if not planned:
            _fail("EMPTY_CONDITION_MATRIX", "a ledger requires at least one planned condition", path="planned_condition_ids")
        if len(set(planned)) != len(planned):
            _fail(
                "DUPLICATE_CONDITION_ID", "planned condition identifiers must be unique",
                path="planned_condition_ids", actual=sorted({item for item in planned if planned.count(item) > 1}),
            )
        records = tuple(self.records)
        planned_set = set(planned)
        seen: set[str] = set()
        for record in records:
            if not isinstance(record, ConditionExecutionRecord):
                _fail("INVALID_LEDGER_RECORD", "records must be ConditionExecutionRecord instances", path="records")
            if record.protocol_hash != self.protocol_hash:
                _fail(
                    "PROTOCOL_HASH_MISMATCH",
                    "a ledger record was executed under a different frozen protocol",
                    path="records.protocol_hash",
                    expected=self.protocol_hash, actual=record.protocol_hash,
                )
            if record.condition_id not in planned_set:
                _fail(
                    "UNPLANNED_CONDITION", "an executed condition was never planned",
                    path="records.condition_id", actual=record.condition_id,
                )
            if record.condition_id in seen:
                _fail(
                    "DUPLICATE_CONDITION_RECORD", "a condition was recorded more than once",
                    path="records.condition_id", actual=record.condition_id,
                )
            seen.add(record.condition_id)
        missing = sorted(planned_set - seen)
        if missing:
            _fail(
                "MISSING_CONDITION_RECORD",
                "every planned condition requires a record; a condition that never ran must be recorded not_run",
                path="records", expected=len(planned), actual=missing,
            )
        object.__setattr__(self, "planned_condition_ids", planned)
        object.__setattr__(self, "records", records)
        if not self.conserved:
            _fail(
                "LEDGER_CONSERVATION_VIOLATION",
                "completed + failed + not_run must equal the planned condition count",
                path="records", expected=len(planned),
                actual=self.completed_count + self.failed_count + self.not_run_count,
            )

    def _by_status(self, status: ExecutionStatus) -> tuple[ConditionExecutionRecord, ...]:
        return tuple(record for record in self.records if record.execution_status is status)

    @property
    def completed_records(self) -> tuple[ConditionExecutionRecord, ...]:
        return self._by_status(ExecutionStatus.COMPLETED)

    @property
    def failed_records(self) -> tuple[ConditionExecutionRecord, ...]:
        return self._by_status(ExecutionStatus.FAILED)

    @property
    def not_run_records(self) -> tuple[ConditionExecutionRecord, ...]:
        return self._by_status(ExecutionStatus.NOT_RUN)

    @property
    def planned_count(self) -> int:
        return len(self.planned_condition_ids)

    @property
    def completed_count(self) -> int:
        return len(self.completed_records)

    @property
    def failed_count(self) -> int:
        return len(self.failed_records)

    @property
    def not_run_count(self) -> int:
        return len(self.not_run_records)

    @property
    def conserved(self) -> bool:
        return self.completed_count + self.failed_count + self.not_run_count == self.planned_count

    @property
    def exploratory_condition_ids(self) -> tuple[str, ...]:
        return tuple(
            record.condition_id
            for record in self.records
            if record.sample_size_status is SampleSizeStatus.EXPLORATORY
        )

    @property
    def null_result_condition_ids(self) -> tuple[str, ...]:
        """Completed Conditions whose measured primary outcome was null (still completed)."""
        return tuple(
            record.condition_id
            for record in self.completed_records
            if record.result is not None and record.result.is_null
        )

    def record_for(self, condition_id: str) -> ConditionExecutionRecord:
        for record in self.records:
            if record.condition_id == condition_id:
                return record
        _fail(
            "UNKNOWN_CONDITION", "no execution record exists for this condition",
            path="condition_id", actual=condition_id,
        )

    def assert_protocol_hash(self, expected_protocol_hash: str) -> None:
        """Cross-check the ledger against the identity of the frozen protocol.

        The ledger stores the hash opaquely; a caller holding the real sealed
        protocol calls this to confirm the executed matrix is the pre-registered
        one.
        """
        _required_text(expected_protocol_hash, "expected_protocol_hash")
        if expected_protocol_hash != self.protocol_hash:
            _fail(
                "PROTOCOL_HASH_MISMATCH",
                "the ledger was produced under a different frozen protocol",
                path="protocol_hash", expected=expected_protocol_hash, actual=self.protocol_hash,
            )

    def claim_input(self) -> dict[str, Any]:
        """Canonical claim-gate input retaining every planned Condition and its reason.

        Failures, interruptions, never-run Conditions and null outcomes all
        appear here; nothing is filtered by favourability (Requirements 15.8,
        8.7).
        """
        return canonical_data(
            {
                "schema_version": self.schema_version,
                "protocol_hash": self.protocol_hash,
                "sample_size_plan_hash": self.sample_size_plan.plan_hash,
                "sample_size_status": self.sample_size_plan.status,
                "planned_count": self.planned_count,
                "completed_count": self.completed_count,
                "failed_count": self.failed_count,
                "not_run_count": self.not_run_count,
                "conserved": self.conserved,
                "exploratory_condition_ids": self.exploratory_condition_ids,
                "null_result_condition_ids": self.null_result_condition_ids,
                "records": tuple(
                    {
                        "condition_id": record.condition_id,
                        "execution_status": record.execution_status,
                        "status_reason": record.status_reason,
                        "sample_size_status": record.sample_size_status,
                        "artifact_hashes": dict(record.artifact_hashes),
                        "measured": record.is_measured,
                        "result": record.result,
                        "run_ids": record.run_ids,
                        "record_hash": record.record_hash,
                    }
                    for record in self.records
                ),
            }
        )

    @property
    def ledger_hash(self) -> str:
        return content_hash(self)


def build_condition_execution_ledger(
    plan: ExecutionPlan, records: Iterable[ConditionExecutionRecord]
) -> ConditionExecutionLedger:
    """Assemble a ledger over exactly the Conditions an :class:`ExecutionPlan` planned."""
    return ConditionExecutionLedger(
        protocol_hash=plan.protocol_hash,
        sample_size_plan=plan.sample_size_plan,
        planned_condition_ids=plan.planned_condition_ids,
        records=tuple(records),
    )


# ---------------------------------------------------------------------------
# Requirement 15.9: a success-only view never replaces the full population
# ---------------------------------------------------------------------------


FULL_POPULATION = "full_population"
SUCCESS_ONLY = "success_only"


@dataclass(frozen=True, slots=True)
class PopulationView:
    """A labelled subset of a ledger's records, always naming its own coverage."""

    coverage: str
    records: tuple[ConditionExecutionRecord, ...]
    schema_version: str = EXECUTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.coverage not in (FULL_POPULATION, SUCCESS_ONLY):
            _fail(
                "INVALID_VIEW_COVERAGE", "coverage must name the full population or the success-only subset",
                path="coverage", expected=[FULL_POPULATION, SUCCESS_ONLY], actual=self.coverage,
            )
        object.__setattr__(self, "records", tuple(self.records))

    @property
    def condition_ids(self) -> tuple[str, ...]:
        return tuple(record.condition_id for record in self.records)

    @property
    def count(self) -> int:
        return len(self.records)

    @property
    def view_hash(self) -> str:
        return content_hash(self)


@dataclass(frozen=True, slots=True)
class PairedPopulationViews:
    """A success-only subset that physically cannot travel without its full population.

    Requirement 15.9 makes a success-only aggregate exploratory regardless of
    how it was produced, so the classification is validated rather than chosen:
    constructing this pairing as ``confirmatory`` fails.
    """

    full_population: PopulationView
    success_only: PopulationView
    exploratory_reason: str
    classification: AnalysisClassification = AnalysisClassification.EXPLORATORY
    schema_version: str = EXECUTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("full_population", "success_only"):
            if not isinstance(getattr(self, name), PopulationView):
                _fail("INVALID_POPULATION_VIEW", f"{name} must be a PopulationView", path=name)
        if self.full_population.coverage != FULL_POPULATION:
            _fail(
                "INVALID_VIEW_COVERAGE", "the paired full-population view must cover the full population",
                path="full_population.coverage", expected=FULL_POPULATION, actual=self.full_population.coverage,
            )
        if self.success_only.coverage != SUCCESS_ONLY:
            _fail(
                "INVALID_VIEW_COVERAGE", "the paired subset view must be labelled success-only",
                path="success_only.coverage", expected=SUCCESS_ONLY, actual=self.success_only.coverage,
            )
        classification = exactly_one(self.classification, AnalysisClassification, path="classification")
        if classification is not AnalysisClassification.EXPLORATORY:
            _fail(
                "SUCCESS_ONLY_AGGREGATE_NOT_CONFIRMATORY",
                "an aggregate over successful seeds only is exploratory and cannot be reclassified",
                path="classification",
                expected=AnalysisClassification.EXPLORATORY.value, actual=classification.value,
            )
        object.__setattr__(self, "classification", classification)
        _required_text(self.exploratory_reason, "exploratory_reason")
        subset = set(self.success_only.condition_ids)
        if not subset.issubset(set(self.full_population.condition_ids)):
            _fail(
                "SUCCESS_ONLY_NOT_A_SUBSET",
                "the success-only view contains conditions absent from the full population view",
                path="success_only", actual=sorted(subset - set(self.full_population.condition_ids)),
            )


def full_population_view(ledger: ConditionExecutionLedger) -> PopulationView:
    """Every planned Condition, failures and null outcomes included."""
    return PopulationView(coverage=FULL_POPULATION, records=ledger.records)


def success_only_views(ledger: ConditionExecutionLedger) -> PairedPopulationViews:
    """The completed-only subset, returned only as a pair with the full population.

    This is the sole entry point to a success-only aggregate, and it hands back
    the full-population view alongside it under a fixed ``exploratory``
    classification (Requirement 15.9).
    """
    return PairedPopulationViews(
        full_population=full_population_view(ledger),
        success_only=PopulationView(coverage=SUCCESS_ONLY, records=ledger.completed_records),
        exploratory_reason=(
            f"aggregate restricted to the {ledger.completed_count} completed of "
            f"{ledger.planned_count} pre-registered conditions; "
            f"{ledger.failed_count} failed and {ledger.not_run_count} never ran"
        ),
    )


__all__ = (
    "ENV_STEPS_PER_RATE_UNIT",
    "EXECUTION_SCHEMA_VERSION",
    "FULL_POPULATION",
    "SUCCESS_ONLY",
    "ConditionCostEstimate",
    "ConditionExecutionLedger",
    "ConditionExecutionRecord",
    "ExecutionPlan",
    "LlmCostEstimate",
    "MeasuredResult",
    "PairedPopulationViews",
    "PopulationView",
    "TrainingSchedule",
    "build_condition_execution_ledger",
    "cost_estimates_for_matrix",
    "estimate_condition_cost",
    "estimate_environment_steps",
    "full_population_view",
    "plan_execution",
    "success_only_views",
)

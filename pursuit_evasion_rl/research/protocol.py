"""Research_Protocol freeze, post-freeze fork, and leakage guard (Requirements 3.1, 4.1-4.4, 6.1-6.7, 15.1-15.7).

A protocol is created ``draft`` and stays freely editable while draft.  Sealing
validates every required schema slot, records the freeze timestamp, signer and
content hash, and makes the protocol immutable: any later change must
:meth:`fork <ProtocolStore.fork>` a new protocol identity carrying a
``parent_hash`` back to the sealed original (Requirement 15.1-15.2).

Registering an analysis against a sealed protocol produces an
:class:`AnalysisLineageEvent`.  Only an analysis declared -- unchanged -- in the
*original* sealed protocol registers as ``confirmatory``; anything unplanned,
changed, or registered against a forked branch is demoted to ``exploratory``
with both the before and after protocol hashes preserved (Requirement 15.3-15.4).

The leakage guard is the protocol-level third layer over
:mod:`~pursuit_evasion_rl.research.maps.splits` (typed tuning views) and
:mod:`~pursuit_evasion_rl.research.evaluation.paired` (separate evaluation
lanes): a :class:`SelectionContext` that references any handle outside
train/validation cannot be turned into a :class:`TuningDataView`, and the
refusal is recorded as a :class:`ContaminationRecord` that propagates to every
descendant run so a later claim gate can exclude them all (Requirement
15.5-15.6).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

from .budget import ConditionResourceEstimate, ResourceCeiling, SampleSizePlan, SampleSizeStatus
from .canonical import canonical_data, content_hash
from .domain import (
    AnalysisClassification,
    Condition,
    PersistedModel,
    ProtocolState,
    ResearchProtocol,
    ResearchQuestion,
    exactly_one,
)
from .errors import ErrorRecord, ResearchValidationError
from .maps.splits import (
    DataHandle,
    HandleKind,
    SplitProtocol,
    SplitScope,
    SplitValidationReport,
    TuningDataView,
)
from .metrics.behavior import METRIC_REGISTRY, MetricDirection
from .runs.manifest import RunManifestStore
from .statistics.paired import (
    BootstrapPlan,
    CaseStatus,
    CorrectionMethod,
    EffectDirection,
    MissingDataPolicy,
    PracticalThreshold,
)

PROTOCOL_SCHEMA_VERSION = "1.0"

#: Scopes a selection, tuning or early-stopping path may ever observe (Requirement 6.6, 15.5).
SELECTION_SCOPES = frozenset({SplitScope.TRAIN, SplitScope.VALIDATION})

#: Scopes that contaminate any selection path they reach (Requirement 15.5-15.6).
HELD_OUT_SCOPES = frozenset({SplitScope.TEST, SplitScope.CROSS_CITY})


def _fail(code: str, message: str, **kwargs) -> None:
    raise ResearchValidationError(code, message, **kwargs)


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("MISSING_REQUIRED_FIELD", f"{name} must be a non-empty string", path=name, actual=value)
    return value


def _required_texts(values: Iterable[Any], name: str) -> tuple[str, ...]:
    items = tuple(values)
    if not items:
        _fail("MISSING_REQUIRED_FIELD", f"{name} must not be empty", path=name)
    return tuple(_required_text(item, name) for item in items)


def _positive_finite(value: Any, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        _fail("INVALID_PROTOCOL_VALUE", f"{name} must be positive and finite", path=name, actual=value)
    return numeric


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def generate_protocol_id() -> str:
    """A unique, sortable protocol identifier."""
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Requirement 4.1: the four fixed research questions of this study
# ---------------------------------------------------------------------------

RQ_SPATIAL_GENERALIZATION = "RQ1_osm_spatial_generalization"
RQ_CONTAINMENT_STABILITY = "RQ2_cooperative_containment_stability"
RQ_COMPONENT_CONTRIBUTION = "RQ3_observation_reward_placement_contribution"
RQ_OFFLINE_LLM_VALUE = "RQ4_offline_llm_value"

REQUIRED_QUESTION_IDS: tuple[str, ...] = (
    RQ_SPATIAL_GENERALIZATION,
    RQ_CONTAINMENT_STABILITY,
    RQ_COMPONENT_CONTRIBUTION,
    RQ_OFFLINE_LLM_VALUE,
)


def default_research_questions() -> tuple[ResearchQuestion, ...]:
    """The four pre-registered questions, each with exactly one primary outcome.

    Every question carries one Analysis_Classification, one primary outcome, a
    directional inequality naming its comparator, and a Practical_Threshold with
    a unit (Requirement 4.1-4.4).
    """
    return (
        ResearchQuestion(
            question_id=RQ_SPATIAL_GENERALIZATION,
            analysis_class=AnalysisClassification.CONFIRMATORY,
            primary_outcome="capture_rate",
            directional_inequality=(
                "capture_rate(held_out_osm_split) - capture_rate(in_region_validation) >= -0.05"
            ),
            practical_threshold=0.05,
            threshold_unit="probability",
            decision_rule=(
                "supported when the paired bootstrap interval for the held-out minus in-region "
                "difference lies entirely above -0.05 probability"
            ),
        ),
        ResearchQuestion(
            question_id=RQ_CONTAINMENT_STABILITY,
            analysis_class=AnalysisClassification.CONFIRMATORY,
            primary_outcome="blocked_exit_fraction",
            directional_inequality=(
                "blocked_exit_fraction(cooperative_containment) - "
                "blocked_exit_fraction(independent_pursuit_baseline) >= 0.05"
            ),
            practical_threshold=0.05,
            threshold_unit="1",
            decision_rule=(
                "supported when the paired bootstrap interval excludes zero and its lower bound "
                "exceeds the 0.05 practical threshold"
            ),
        ),
        ResearchQuestion(
            question_id=RQ_COMPONENT_CONTRIBUTION,
            analysis_class=AnalysisClassification.CONFIRMATORY,
            primary_outcome="capture_rate",
            directional_inequality=(
                "capture_rate(full_observation_reward_placement) - capture_rate(single_axis_ablation) >= 0.05"
            ),
            practical_threshold=0.05,
            threshold_unit="probability",
            decision_rule=(
                "an axis contributes when its ablation's multiplicity-adjusted paired interval "
                "lies entirely below -0.05 probability"
            ),
        ),
        ResearchQuestion(
            question_id=RQ_OFFLINE_LLM_VALUE,
            analysis_class=AnalysisClassification.CONFIRMATORY,
            primary_outcome="capture_rate",
            directional_inequality=(
                "capture_rate(offline_llm_assisted) - capture_rate(no_llm_control) >= 0.05"
            ),
            practical_threshold=0.05,
            threshold_unit="probability",
            decision_rule=(
                "supported only when the offline LLM artifact hash is fixed before training and "
                "the paired interval lower bound exceeds 0.05 probability"
            ),
        ),
    )


def validate_questions(questions: Sequence[ResearchQuestion]) -> None:
    """Every fixed question is present and fully specified (Requirement 4.1-4.4)."""
    by_id = {question.question_id: question for question in questions}
    missing = [item for item in REQUIRED_QUESTION_IDS if item not in by_id]
    if missing:
        _fail(
            "MISSING_RESEARCH_QUESTION",
            "the protocol must identify all four pre-registered research questions",
            path="questions",
            expected=list(REQUIRED_QUESTION_IDS),
            actual=sorted(by_id),
        )
    for question in questions:
        exactly_one(
            question.analysis_class,
            AnalysisClassification,
            path=f"questions.{question.question_id}.analysis_class",
        )
        threshold = question.practical_threshold
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            _fail(
                "MISSING_PRACTICAL_THRESHOLD",
                "every research question requires a numeric Practical_Threshold",
                path=f"questions.{question.question_id}.practical_threshold",
                actual=threshold,
            )
        if not math.isfinite(float(threshold)) or float(threshold) <= 0.0:
            _fail(
                "INVALID_PRACTICAL_THRESHOLD",
                "a Practical_Threshold must be a positive, finite magnitude",
                path=f"questions.{question.question_id}.practical_threshold",
                actual=threshold,
            )


# ---------------------------------------------------------------------------
# Specification slots (Requirement 3.1, 6.1-6.7, 15.7)
# ---------------------------------------------------------------------------


class MetricRole(str, Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"


class SelectionPurpose(str, Enum):
    """Every path Requirement 15.5 forbids held-out data from reaching."""

    TUNING = "tuning"
    EARLY_STOPPING = "early_stopping"
    REWARD_DESIGN = "reward_design"
    LLM_INPUT = "llm_input"
    BASELINE_SELECTION = "baseline_selection"
    CHECKPOINT_SELECTION = "checkpoint_selection"


@dataclass(frozen=True, slots=True)
class HypothesisSpec:
    """One directional hypothesis bound to exactly one research question."""

    hypothesis_id: str
    question_id: str
    statement: str
    null_statement: str
    schema_version: str = PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("hypothesis_id", "question_id", "statement", "null_statement"):
            _required_text(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class ConditionRef:
    """A Condition identity referenced by hash, never redefined here."""

    condition_id: str
    condition_hash: str
    axis: str
    arm: str
    schema_version: str = PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("condition_id", "condition_hash", "axis", "arm"):
            _required_text(getattr(self, name), name)

    @classmethod
    def from_condition(cls, condition: Condition, *, axis: str, arm: str) -> "ConditionRef":
        if not isinstance(condition, Condition):
            _fail("INVALID_CONDITION", "a domain Condition is required", path="condition")
        return cls(
            condition_id=condition.condition_id,
            condition_hash=str(condition.content_hash),
            axis=axis,
            arm=arm,
        )


def condition_refs(specs: Iterable[Any]) -> tuple[ConditionRef, ...]:
    """Reference every ``ConditionSpec`` of the variant factory matrix by hash."""
    return tuple(
        ConditionRef.from_condition(spec.condition, axis=spec.axis, arm=spec.arm) for spec in specs
    )


@dataclass(frozen=True, slots=True)
class SplitSpec:
    """Frozen spatial split identities and leakage boundary (Requirement 6.1-6.7)."""

    metric_crs: str
    buffer_m: float
    polygon_hashes: Mapping[str, str]
    network_hashes: Mapping[str, str]
    cross_city_city_ids: tuple[str, ...]
    split_protocol_hash: str
    split_validation_report_hash: str
    held_out_evaluation_rule: str
    schema_version: str = PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _required_text(self.metric_crs, "metric_crs")
        _required_text(self.split_protocol_hash, "split_protocol_hash")
        _required_text(self.split_validation_report_hash, "split_validation_report_hash")
        _required_text(self.held_out_evaluation_rule, "held_out_evaluation_rule")
        object.__setattr__(self, "buffer_m", _positive_finite(self.buffer_m, "buffer_m"))
        expected = tuple(scope.value for scope in (SplitScope.TRAIN, SplitScope.VALIDATION, SplitScope.TEST))
        for name in ("polygon_hashes", "network_hashes"):
            mapping = dict(getattr(self, name))
            if tuple(sorted(mapping)) != tuple(sorted(expected)):
                _fail(
                    "MISSING_SPLIT_HASH",
                    f"{name} must record exactly the train, validation and test splits",
                    path=name,
                    expected=sorted(expected),
                    actual=sorted(mapping),
                )
            for scope, value in mapping.items():
                _required_text(value, f"{name}.{scope}")
            if len(set(mapping.values())) != len(mapping):
                _fail(
                    "DUPLICATE_SPLIT_HASH",
                    f"{name} must differ across train, validation and test",
                    path=name,
                    actual=sorted(mapping.values()),
                )
            object.__setattr__(self, name, dict(sorted(mapping.items())))
        cities = _required_texts(self.cross_city_city_ids, "cross_city_city_ids")
        if len(set(cities)) < 2:
            _fail(
                "INSUFFICIENT_ZERO_SHOT_CITIES",
                "at least two distinct zero-shot cities must be pre-registered",
                path="cross_city_city_ids",
                expected=">=2",
                actual=sorted(set(cities)),
            )
        object.__setattr__(self, "cross_city_city_ids", tuple(sorted(set(cities))))

    @classmethod
    def from_validation_report(
        cls,
        report: SplitValidationReport,
        split_protocol: SplitProtocol,
        *,
        cross_city_city_ids: Sequence[str],
        held_out_evaluation_rule: str,
    ) -> "SplitSpec":
        """Freeze the geometry and network content hashes an eligible split report proved."""
        if not isinstance(report, SplitValidationReport):
            _fail("INVALID_SPLIT_REPORT", "a SplitValidationReport is required", path="report")
        if not isinstance(split_protocol, SplitProtocol):
            _fail("INVALID_SPLIT_PROTOCOL", "a SplitProtocol is required", path="split_protocol")
        if not report.eligible_for_generalization:
            _fail(
                "SPLIT_NOT_GENERALIZATION_ELIGIBLE",
                "only a passing split report may be frozen into a protocol",
                path="report",
                actual=report.report_hash,
            )
        return cls(
            metric_crs=split_protocol.metric_crs,
            buffer_m=split_protocol.buffer_m,
            polygon_hashes=dict(report.polygon_hashes),
            network_hashes=dict(report.network_hashes),
            cross_city_city_ids=tuple(cross_city_city_ids),
            split_protocol_hash=str(split_protocol.content_hash),
            split_validation_report_hash=report.report_hash,
            held_out_evaluation_rule=held_out_evaluation_rule,
        )


@dataclass(frozen=True, slots=True)
class SampleSpec:
    """The pre-registered sample size plus the only handles selection may observe.

    ``selection_handles`` is validated by constructing a
    :class:`~pursuit_evasion_rl.research.maps.splits.TuningDataView`, so the
    protocol's own sample specification structurally cannot name a test or
    cross-city handle (Requirement 6.6, 15.5).
    """

    plan: SampleSizePlan
    selection_handles: tuple[DataHandle[SplitScope], ...]
    schema_version: str = PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.plan, SampleSizePlan):
            _fail("INVALID_SAMPLE_PLAN", "a SampleSizePlan is required", path="plan")
        handles = tuple(self.selection_handles)
        if not handles:
            _fail("MISSING_REQUIRED_FIELD", "selection_handles must not be empty", path="selection_handles")
        leaked = tuple(handle for handle in handles if handle.scope in HELD_OUT_SCOPES)
        if leaked:
            _fail(
                "PROTOCOL_SELECTION_LEAKAGE",
                "the sample specification must not reference test or cross-city handles",
                path="selection_handles",
                expected=[],
                actual=[handle.identifier for handle in leaked],
            )
        TuningDataView(
            train=tuple(handle for handle in handles if handle.scope is SplitScope.TRAIN),
            validation=tuple(handle for handle in handles if handle.scope is SplitScope.VALIDATION),
        )
        object.__setattr__(self, "selection_handles", handles)


@dataclass(frozen=True, slots=True)
class ResourceSpec:
    """The measured Resource_Ceiling and the per-condition estimates it was applied to."""

    ceiling: ResourceCeiling
    estimates: tuple[ConditionResourceEstimate, ...]
    schema_version: str = PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.ceiling, ResourceCeiling):
            _fail("INVALID_RESOURCE_CEILING", "a ResourceCeiling is required", path="ceiling")
        estimates = tuple(self.estimates)
        if not estimates:
            _fail("MISSING_REQUIRED_FIELD", "estimates must not be empty", path="estimates")
        if any(not isinstance(item, ConditionResourceEstimate) for item in estimates):
            _fail(
                "INVALID_RESOURCE_ESTIMATE",
                "estimates must be ConditionResourceEstimate values",
                path="estimates",
            )
        object.__setattr__(self, "estimates", estimates)


@dataclass(frozen=True, slots=True)
class MetricDeclaration:
    """Which metric is primary or secondary -- never a redefinition of its math.

    A metric already registered in :data:`METRIC_REGISTRY` must be declared with
    that registry's own symbol, unit and direction, so the protocol can select
    metrics without forking their definitions.
    """

    metric_id: str
    symbol: str
    unit: str
    direction: MetricDirection
    role: MetricRole
    formula: str
    schema_version: str = PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("metric_id", "symbol", "unit", "formula"):
            _required_text(getattr(self, name), name)
        object.__setattr__(self, "direction", exactly_one(self.direction, MetricDirection, path="direction"))
        object.__setattr__(self, "role", exactly_one(self.role, MetricRole, path="role"))
        registered = METRIC_REGISTRY.get(self.metric_id)
        if registered is not None:
            actual = (self.symbol, self.unit, self.direction)
            expected = (registered.symbol, registered.unit, registered.direction)
            if actual != expected:
                _fail(
                    "METRIC_REDEFINITION",
                    "a registered metric must be declared with its registry semantics",
                    path="metric_id",
                    expected=[registered.symbol, registered.unit, registered.direction.value],
                    actual=[self.symbol, self.unit, self.direction.value],
                )

    @property
    def effect_direction(self) -> EffectDirection:
        return (
            EffectDirection.GREATER_IS_BETTER
            if self.direction is MetricDirection.HIGHER_IS_BETTER
            else EffectDirection.LESS_IS_BETTER
        )


def metric_declaration(metric_id: str, role: MetricRole) -> MetricDeclaration:
    """Declare a registry metric's role without restating its formula."""
    registered = METRIC_REGISTRY.get(metric_id)
    if registered is None:
        _fail(
            "UNKNOWN_METRIC",
            "only a registered metric can be declared from the registry",
            path="metric_id",
            actual=metric_id,
        )
    return MetricDeclaration(
        metric_id=metric_id,
        symbol=registered.symbol,
        unit=registered.unit,
        direction=registered.direction,
        role=role,
        formula=registered.formula,
    )


@dataclass(frozen=True, slots=True)
class StatisticsSpec:
    """The pre-registered estimation and multiplicity plan."""

    bootstrap: BootstrapPlan
    correction: CorrectionMethod
    primary_test: str
    multiplicity_family: str
    schema_version: str = PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.bootstrap, BootstrapPlan):
            _fail("INVALID_BOOTSTRAP_PLAN", "a BootstrapPlan is required", path="bootstrap")
        object.__setattr__(
            self, "correction", exactly_one(self.correction, CorrectionMethod, path="correction")
        )
        for name in ("primary_test", "multiplicity_family"):
            _required_text(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class ToleranceSpec:
    """Numerical equivalence tolerances for resume and interruption comparisons."""

    rtol: float
    atol: float
    applies_to: tuple[str, ...]
    rationale: str
    schema_version: str = PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "rtol", _positive_finite(self.rtol, "rtol"))
        object.__setattr__(self, "atol", _positive_finite(self.atol, "atol"))
        object.__setattr__(self, "applies_to", _required_texts(self.applies_to, "applies_to"))
        _required_text(self.rationale, "rationale")


@dataclass(frozen=True, slots=True)
class ExclusionSpec:
    """Failure, missing and interruption handling fixed in advance (Requirement 15.7)."""

    outcome_mapping: Mapping[str, str]
    excludable_conditions: tuple[str, ...]
    primary_policy: MissingDataPolicy
    sensitivity_policy: MissingDataPolicy
    planned_case_accounting_rule: str
    schema_version: str = PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        mapping = dict(self.outcome_mapping)
        expected = tuple(sorted(status.value for status in CaseStatus))
        if tuple(sorted(mapping)) != expected:
            _fail(
                "INCOMPLETE_OUTCOME_MAPPING",
                "every case status requires one outcome mapping rule",
                path="outcome_mapping",
                expected=list(expected),
                actual=sorted(mapping),
            )
        for status, rule in mapping.items():
            _required_text(rule, f"outcome_mapping.{status}")
        object.__setattr__(self, "outcome_mapping", dict(sorted(mapping.items())))
        object.__setattr__(
            self, "excludable_conditions", _required_texts(self.excludable_conditions, "excludable_conditions")
        )
        object.__setattr__(
            self, "primary_policy", exactly_one(self.primary_policy, MissingDataPolicy, path="primary_policy")
        )
        object.__setattr__(
            self,
            "sensitivity_policy",
            exactly_one(self.sensitivity_policy, MissingDataPolicy, path="sensitivity_policy"),
        )
        if self.primary_policy is self.sensitivity_policy:
            _fail(
                "MISSING_SENSITIVITY_ANALYSIS",
                "the sensitivity policy must differ from the primary missing-data policy",
                path="sensitivity_policy",
                actual=self.sensitivity_policy.value,
            )
        _required_text(self.planned_case_accounting_rule, "planned_case_accounting_rule")


@dataclass(frozen=True, slots=True)
class StopRule:
    """One halting rule, evaluated only on data a selection path may observe."""

    rule_id: str
    criterion: str
    action: str
    evaluated_on: SplitScope
    schema_version: str = PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("rule_id", "criterion", "action"):
            _required_text(getattr(self, name), name)
        scope = exactly_one(self.evaluated_on, SplitScope, path="evaluated_on")
        if scope not in SELECTION_SCOPES:
            _fail(
                "STOP_RULE_LEAKAGE",
                "a stop rule may not be evaluated on test or cross-city data",
                path="evaluated_on",
                expected=sorted(item.value for item in SELECTION_SCOPES),
                actual=scope.value,
            )
        object.__setattr__(self, "evaluated_on", scope)


@dataclass(frozen=True, slots=True)
class StopSpec:
    rules: tuple[StopRule, ...]
    schema_version: str = PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        rules = tuple(self.rules)
        if not rules:
            _fail("MISSING_REQUIRED_FIELD", "at least one stop rule is required", path="rules")
        if any(not isinstance(rule, StopRule) for rule in rules):
            _fail("INVALID_STOP_RULE", "rules must be StopRule values", path="rules")
        identifiers = [rule.rule_id for rule in rules]
        if len(set(identifiers)) != len(identifiers):
            _fail("DUPLICATE_IDENTIFIER", "stop rule identifiers must be unique", path="rules")
        object.__setattr__(self, "rules", rules)


@dataclass(frozen=True, slots=True)
class SearchSpec:
    """The literature search fixed before any confirmatory result is read (Requirement 3.1)."""

    search_date_utc: str
    query: str
    sources: tuple[str, ...]
    date_range_start: str
    date_range_end: str
    languages: tuple[str, ...]
    inclusion_criteria: tuple[str, ...]
    exclusion_criteria: tuple[str, ...]
    schema_version: str = PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("search_date_utc", "query", "date_range_start", "date_range_end"):
            _required_text(getattr(self, name), name)
        for name in ("sources", "languages", "inclusion_criteria", "exclusion_criteria"):
            object.__setattr__(self, name, _required_texts(getattr(self, name), name))
        if self.date_range_end < self.date_range_start:
            _fail(
                "INVALID_SEARCH_RANGE",
                "the search date range must not end before it starts",
                path="date_range_end",
                expected=f">= {self.date_range_start}",
                actual=self.date_range_end,
            )


@dataclass(frozen=True, slots=True)
class PlannedAnalysis:
    """One analysis declared in the protocol, carrying exactly one classification."""

    analysis_id: str
    question_id: str
    classification: AnalysisClassification
    estimand: str
    metric_id: str
    comparison: str
    schema_version: str = PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("analysis_id", "question_id", "estimand", "metric_id", "comparison"):
            _required_text(getattr(self, name), name)
        object.__setattr__(
            self,
            "classification",
            exactly_one(self.classification, AnalysisClassification, path="classification"),
        )

    @property
    def declaration_hash(self) -> str:
        return content_hash(self)


REQUIRED_SPECIFICATION_SLOTS: tuple[str, ...] = (
    "hypotheses",
    "conditions",
    "split",
    "sample",
    "resource",
    "metrics",
    "statistics",
    "thresholds",
    "tolerance",
    "exclusion",
    "stop",
    "search",
    "analyses",
)


@dataclass(frozen=True, slots=True)
class ProtocolSpecifications:
    """Every required specification slot of a sealable protocol."""

    hypotheses: tuple[HypothesisSpec, ...]
    conditions: tuple[ConditionRef, ...]
    split: SplitSpec
    sample: SampleSpec
    resource: ResourceSpec
    metrics: tuple[MetricDeclaration, ...]
    statistics: StatisticsSpec
    thresholds: tuple[PracticalThreshold, ...]
    tolerance: ToleranceSpec
    exclusion: ExclusionSpec
    stop: StopSpec
    search: SearchSpec
    analyses: tuple[PlannedAnalysis, ...]
    schema_version: str = PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name, item_type in (
            ("hypotheses", HypothesisSpec),
            ("conditions", ConditionRef),
            ("metrics", MetricDeclaration),
            ("thresholds", PracticalThreshold),
            ("analyses", PlannedAnalysis),
        ):
            items = tuple(getattr(self, name))
            if not items:
                _fail("MISSING_REQUIRED_FIELD", f"{name} must not be empty", path=name)
            if any(not isinstance(item, item_type) for item in items):
                _fail(
                    "INVALID_SPECIFICATION_ENTRY",
                    f"{name} must contain {item_type.__name__} values",
                    path=name,
                )
            object.__setattr__(self, name, items)
        analysis_ids = [item.analysis_id for item in self.analyses]
        if len(set(analysis_ids)) != len(analysis_ids):
            _fail("DUPLICATE_IDENTIFIER", "analysis identifiers must be unique", path="analyses")
        if len({item.metric for item in self.thresholds}) != len(self.thresholds):
            _fail("DUPLICATE_IDENTIFIER", "each metric has at most one threshold", path="thresholds")

    def as_mapping(self) -> dict[str, Any]:
        """Canonical JSON data for :attr:`ResearchProtocol.specifications`."""
        return canonical_data(self)


def practical_thresholds_for(
    questions: Sequence[ResearchQuestion], metrics: Sequence[MetricDeclaration]
) -> tuple[PracticalThreshold, ...]:
    """Derive one threshold per primary outcome from the questions themselves."""
    by_metric = {metric.metric_id: metric for metric in metrics}
    thresholds: dict[str, PracticalThreshold] = {}
    for question in questions:
        metric = by_metric.get(question.primary_outcome)
        if metric is None:
            _fail(
                "UNDECLARED_PRIMARY_OUTCOME",
                "a question's primary outcome must be a declared metric",
                path=f"questions.{question.question_id}.primary_outcome",
                actual=question.primary_outcome,
            )
        thresholds.setdefault(
            question.primary_outcome,
            PracticalThreshold(
                metric=question.primary_outcome,
                unit=question.threshold_unit,
                minimum_effect=float(question.practical_threshold),
                direction=metric.effect_direction,
            ),
        )
    return tuple(thresholds[key] for key in sorted(thresholds))


# ---------------------------------------------------------------------------
# Structural validation of a stored specification mapping
# ---------------------------------------------------------------------------


def _slot(specifications: Mapping[str, Any], name: str) -> Any:
    try:
        return specifications[name]
    except (KeyError, TypeError):
        _fail(
            "MISSING_SPECIFICATION_SLOT",
            f"the protocol requires a {name!r} specification",
            path=f"specifications.{name}",
            expected=list(REQUIRED_SPECIFICATION_SLOTS),
        )


def validate_specification_slots(specifications: Mapping[str, Any]) -> None:
    """Every required slot exists and carries content (Requirement 15.1)."""
    if not isinstance(specifications, Mapping):
        _fail("INVALID_SPECIFICATIONS", "specifications must be a mapping", path="specifications")
    for name in REQUIRED_SPECIFICATION_SLOTS:
        value = _slot(specifications, name)
        if value is None or (isinstance(value, (str, bytes, Sequence, Mapping)) and len(value) == 0):
            _fail(
                "EMPTY_SPECIFICATION_SLOT",
                f"the {name!r} specification must not be empty",
                path=f"specifications.{name}",
            )


def _entries(value: Any, path: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _fail("INVALID_SPECIFICATION_ENTRY", f"{path} must be a sequence", path=path)
    if any(not isinstance(item, Mapping) for item in value):
        _fail("INVALID_SPECIFICATION_ENTRY", f"{path} entries must be mappings", path=path)
    return tuple(value)


def validate_protocol_content(
    questions: Sequence[ResearchQuestion], specifications: Mapping[str, Any]
) -> None:
    """Cross-check questions against the declared metrics, thresholds and analyses."""
    validate_questions(questions)
    validate_specification_slots(specifications)

    metrics = _entries(_slot(specifications, "metrics"), "specifications.metrics")
    thresholds = _entries(_slot(specifications, "thresholds"), "specifications.thresholds")
    analyses = _entries(_slot(specifications, "analyses"), "specifications.analyses")
    hypotheses = _entries(_slot(specifications, "hypotheses"), "specifications.hypotheses")

    primary_metrics = {
        item["metric_id"]: item for item in metrics if item.get("role") == MetricRole.PRIMARY.value
    }
    thresholds_by_metric = {item["metric"]: item for item in thresholds}
    question_ids = {question.question_id for question in questions}

    for question in questions:
        outcome = question.primary_outcome
        metric = primary_metrics.get(outcome)
        if metric is None:
            _fail(
                "UNDECLARED_PRIMARY_OUTCOME",
                "a question's primary outcome must be declared as a primary metric",
                path=f"questions.{question.question_id}.primary_outcome",
                expected=sorted(primary_metrics),
                actual=outcome,
            )
        threshold = thresholds_by_metric.get(outcome)
        if threshold is None:
            _fail(
                "MISSING_PRACTICAL_THRESHOLD",
                "each primary outcome requires a registered Practical_Threshold",
                path="specifications.thresholds",
                actual=outcome,
            )
        if threshold["unit"] != question.threshold_unit:
            _fail(
                "THRESHOLD_UNIT_MISMATCH",
                "the threshold unit must match the question's declared unit",
                path=f"specifications.thresholds.{outcome}.unit",
                expected=question.threshold_unit,
                actual=threshold["unit"],
            )
        if float(threshold["minimum_effect"]) != float(question.practical_threshold):
            _fail(
                "THRESHOLD_VALUE_MISMATCH",
                "the threshold magnitude must match the question's Practical_Threshold",
                path=f"specifications.thresholds.{outcome}.minimum_effect",
                expected=float(question.practical_threshold),
                actual=float(threshold["minimum_effect"]),
            )
        expected_direction = (
            EffectDirection.GREATER_IS_BETTER.value
            if metric["direction"] == MetricDirection.HIGHER_IS_BETTER.value
            else EffectDirection.LESS_IS_BETTER.value
        )
        if threshold["direction"] != expected_direction:
            _fail(
                "THRESHOLD_DIRECTION_MISMATCH",
                "the threshold direction must follow the metric's registered direction",
                path=f"specifications.thresholds.{outcome}.direction",
                expected=expected_direction,
                actual=threshold["direction"],
            )

    for name, entries in (("analyses", analyses), ("hypotheses", hypotheses)):
        covered: set[str] = set()
        for entry in entries:
            question_id = entry["question_id"]
            if question_id not in question_ids:
                _fail(
                    "UNKNOWN_ANALYSIS_QUESTION",
                    f"a {name[:-1]} references an undeclared research question",
                    path=f"specifications.{name}",
                    expected=sorted(question_ids),
                    actual=question_id,
                )
            covered.add(question_id)
        uncovered = sorted(question_ids - covered)
        if uncovered:
            _fail(
                "MISSING_PLANNED_ANALYSIS" if name == "analyses" else "MISSING_HYPOTHESIS",
                f"every research question requires at least one declared {name[:-1]}",
                path=f"specifications.{name}",
                actual=uncovered,
            )

    for entry in analyses:
        exactly_one(
            entry["classification"],
            AnalysisClassification,
            path=f"specifications.analyses.{entry['analysis_id']}.classification",
        )

    sample = _slot(specifications, "sample")
    resource = _slot(specifications, "resource")
    plan_ceiling = sample["plan"]["resource_ceiling_id"]
    declared_ceiling = resource["ceiling"]["resource_ceiling_id"]
    if plan_ceiling != declared_ceiling:
        _fail(
            "RESOURCE_CEILING_MISMATCH",
            "the sample plan must be derived from the declared Resource_Ceiling",
            path="specifications.sample.plan.resource_ceiling_id",
            expected=declared_ceiling,
            actual=plan_ceiling,
        )
    for handle in _entries(sample["selection_handles"], "specifications.sample.selection_handles"):
        if handle["scope"] not in {scope.value for scope in SELECTION_SCOPES}:
            _fail(
                "PROTOCOL_SELECTION_LEAKAGE",
                "the sample specification must not reference test or cross-city handles",
                path="specifications.sample.selection_handles",
                expected=sorted(scope.value for scope in SELECTION_SCOPES),
                actual=handle["scope"],
            )


# ---------------------------------------------------------------------------
# Freeze and post-freeze fork (Requirement 15.1-15.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProtocolRecord:
    """A stored protocol plus the lineage the store tracks around it."""

    protocol: ResearchProtocol
    created_at_utc: str
    parent_id: str | None = None
    schema_version: str = PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, ResearchProtocol):
            _fail("INVALID_PROTOCOL_RECORD", "protocol must be a ResearchProtocol", path="protocol")
        _required_text(self.created_at_utc, "created_at_utc")

    @property
    def protocol_id(self) -> str:
        return self.protocol.protocol_id

    @property
    def protocol_hash(self) -> str:
        return str(self.protocol.content_hash)

    @property
    def is_sealed(self) -> bool:
        return self.protocol.state is ProtocolState.SEALED

    @property
    def is_branch(self) -> bool:
        """A forked protocol version, created after its parent was sealed."""
        return self.protocol.parent_hash is not None


@dataclass(frozen=True, slots=True)
class AnalysisLineageEvent(PersistedModel):
    """The recorded decision of one analysis registration (Requirement 15.3-15.4)."""

    analysis_id: str
    protocol_id: str
    protocol_hash_at_declaration: str
    protocol_hash_at_registration: str
    classification: AnalysisClassification
    reason: str
    registered_at_utc: str
    declared_analysis_hash: str | None = None
    registered_analysis_hash: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "analysis_id",
            "protocol_id",
            "protocol_hash_at_declaration",
            "protocol_hash_at_registration",
            "reason",
            "registered_at_utc",
        ):
            _required_text(getattr(self, name), name)
        object.__setattr__(
            self,
            "classification",
            exactly_one(self.classification, AnalysisClassification, path="classification"),
        )
        super(AnalysisLineageEvent, self).__post_init__()

    @property
    def is_confirmatory(self) -> bool:
        return self.classification is AnalysisClassification.CONFIRMATORY


def _question_to_dict(question: ResearchQuestion) -> dict[str, Any]:
    return canonical_data(question)


def _question_from_dict(payload: Mapping[str, Any]) -> ResearchQuestion:
    return ResearchQuestion(
        question_id=payload["question_id"],
        analysis_class=AnalysisClassification(payload["analysis_class"]),
        primary_outcome=payload["primary_outcome"],
        directional_inequality=payload["directional_inequality"],
        practical_threshold=payload["practical_threshold"],
        threshold_unit=payload["threshold_unit"],
        decision_rule=payload["decision_rule"],
        schema_version=payload["schema_version"],
        content_hash=payload["content_hash"],
    )


def _record_to_dict(record: ProtocolRecord) -> dict[str, Any]:
    protocol = record.protocol
    return {
        "schema_version": record.schema_version,
        "protocol_hash": record.protocol_hash,
        "created_at_utc": record.created_at_utc,
        "parent_id": record.parent_id,
        "protocol": {
            "schema_version": protocol.schema_version,
            "content_hash": protocol.content_hash,
            "protocol_id": protocol.protocol_id,
            "state": protocol.state.value,
            "questions": [_question_to_dict(item) for item in protocol.questions],
            "specifications": canonical_data(protocol.specifications),
            "frozen_at_utc": protocol.frozen_at_utc,
            "signer": protocol.signer,
            "parent_hash": protocol.parent_hash,
        },
    }


def _record_from_dict(payload: Mapping[str, Any]) -> ProtocolRecord:
    body = payload["protocol"]
    protocol = ResearchProtocol(
        protocol_id=body["protocol_id"],
        state=ProtocolState(body["state"]),
        questions=tuple(_question_from_dict(item) for item in body["questions"]),
        specifications=body["specifications"],
        frozen_at_utc=body["frozen_at_utc"],
        signer=body["signer"],
        parent_hash=body["parent_hash"],
        schema_version=body["schema_version"],
        content_hash=body["content_hash"],
    )
    return ProtocolRecord(
        protocol=protocol,
        created_at_utc=payload["created_at_utc"],
        parent_id=payload["parent_id"],
        schema_version=payload["schema_version"],
    )


class ProtocolStore:
    """A directory of one-protocol-per-identity JSON files, atomically written."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, protocol_id: str) -> Path:
        return self.root / protocol_id / "protocol.json"

    def _analysis_path(self, protocol_id: str, analysis_id: str) -> Path:
        return self.root / protocol_id / "analyses" / f"{analysis_id}.json"

    @staticmethod
    def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(path)

    def exists(self, protocol_id: str) -> bool:
        return self._path(protocol_id).is_file()

    def read(self, protocol_id: str) -> ProtocolRecord:
        path = self._path(protocol_id)
        if not path.is_file():
            _fail(
                "PROTOCOL_NOT_FOUND",
                f"no protocol exists for {protocol_id!r}",
                path="protocol_id",
                actual=protocol_id,
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        record = _record_from_dict(payload)
        if record.protocol_hash != payload["protocol_hash"]:
            _fail(
                "PROTOCOL_HASH_MISMATCH",
                "the stored protocol hash does not match its content",
                path="protocol_hash",
                expected=record.protocol_hash,
                actual=payload["protocol_hash"],
            )
        return record

    def create(
        self,
        *,
        questions: Sequence[ResearchQuestion],
        specifications: ProtocolSpecifications | Mapping[str, Any],
        protocol_id: str | None = None,
        parent_id: str | None = None,
        parent_hash: str | None = None,
    ) -> ProtocolRecord:
        """Create a ``draft`` protocol; drafts may still be incomplete."""
        protocol_id = protocol_id or generate_protocol_id()
        if self.exists(protocol_id):
            _fail(
                "PROTOCOL_ID_COLLISION",
                f"a protocol already exists for {protocol_id!r}",
                path="protocol_id",
                actual=protocol_id,
            )
        payload = (
            specifications.as_mapping()
            if isinstance(specifications, ProtocolSpecifications)
            else dict(specifications)
        )
        protocol = ResearchProtocol(
            protocol_id=protocol_id,
            state=ProtocolState.DRAFT,
            questions=tuple(questions),
            specifications=payload,
            parent_hash=parent_hash,
        )
        record = ProtocolRecord(protocol=protocol, created_at_utc=_utc_now(), parent_id=parent_id)
        self._write_atomic(self._path(protocol_id), _record_to_dict(record))
        return record

    def update_draft(
        self,
        protocol_id: str,
        *,
        questions: Sequence[ResearchQuestion] | None = None,
        specifications: ProtocolSpecifications | Mapping[str, Any] | None = None,
    ) -> ProtocolRecord:
        """Edit a still-``draft`` protocol; a sealed one must be forked instead."""
        current = self.read(protocol_id)
        if current.is_sealed:
            _fail(
                "SEALED_PROTOCOL_MUTATION_BLOCKED",
                "a sealed protocol cannot be changed in place; fork a new protocol branch instead",
                path="protocol_id",
                actual=protocol_id,
            )
        payload = current.protocol.specifications
        if specifications is not None:
            payload = (
                specifications.as_mapping()
                if isinstance(specifications, ProtocolSpecifications)
                else dict(specifications)
            )
        updated_protocol = ResearchProtocol(
            protocol_id=current.protocol.protocol_id,
            state=ProtocolState.DRAFT,
            questions=tuple(questions) if questions is not None else current.protocol.questions,
            specifications=payload,
            parent_hash=current.protocol.parent_hash,
        )
        updated = replace(current, protocol=updated_protocol)
        self._write_atomic(self._path(protocol_id), _record_to_dict(updated))
        return updated

    def seal(
        self, protocol_id: str, *, signer: str, frozen_at_utc: str | None = None
    ) -> ProtocolRecord:
        """Validate every required slot and freeze the protocol (Requirement 15.1)."""
        current = self.read(protocol_id)
        if current.is_sealed:
            _fail(
                "SEALED_PROTOCOL_MUTATION_BLOCKED",
                "a sealed protocol cannot be re-sealed; fork a new protocol branch instead",
                path="protocol_id",
                actual=protocol_id,
            )
        _required_text(signer, "signer")
        validate_protocol_content(current.protocol.questions, current.protocol.specifications)
        sealed_protocol = ResearchProtocol(
            protocol_id=current.protocol.protocol_id,
            state=ProtocolState.SEALED,
            questions=current.protocol.questions,
            specifications=current.protocol.specifications,
            frozen_at_utc=frozen_at_utc or _utc_now(),
            signer=signer,
            parent_hash=current.protocol.parent_hash,
        )
        sealed = replace(current, protocol=sealed_protocol)
        self._write_atomic(self._path(protocol_id), _record_to_dict(sealed))
        return sealed

    def fork(
        self,
        protocol_id: str,
        *,
        questions: Sequence[ResearchQuestion] | None = None,
        specifications: ProtocolSpecifications | Mapping[str, Any] | None = None,
        new_protocol_id: str | None = None,
    ) -> ProtocolRecord:
        """Branch a sealed protocol into a new draft identity (Requirement 15.2).

        The sealed original is left untouched; the child records its parent's
        content hash so every later analysis can see it came after the freeze.
        """
        parent = self.read(protocol_id)
        if not parent.is_sealed:
            _fail(
                "CANNOT_FORK_DRAFT_PROTOCOL",
                "only a sealed protocol needs a branch; a draft can still be edited in place",
                path="protocol_id",
                actual=protocol_id,
            )
        return self.create(
            questions=questions if questions is not None else parent.protocol.questions,
            specifications=(
                specifications if specifications is not None else parent.protocol.specifications
            ),
            protocol_id=new_protocol_id,
            parent_id=protocol_id,
            parent_hash=parent.protocol_hash,
        )

    def lineage(self, protocol_id: str) -> tuple[ProtocolRecord, ...]:
        """The chain from the original sealed protocol down to ``protocol_id``."""
        chain: list[ProtocolRecord] = []
        current: str | None = protocol_id
        seen: set[str] = set()
        while current is not None:
            if current in seen:
                _fail(
                    "PROTOCOL_LINEAGE_CYCLE",
                    "protocol lineage contains a cycle",
                    path="protocol_id",
                    actual=current,
                )
            seen.add(current)
            record = self.read(current)
            chain.append(record)
            current = record.parent_id
        return tuple(reversed(chain))

    def register_analysis(
        self, protocol_id: str, analysis: PlannedAnalysis
    ) -> AnalysisLineageEvent:
        """Classify one analysis against the sealed protocol (Requirement 15.3-15.4).

        ``confirmatory`` requires the analysis to be present, unchanged, and
        already classified confirmatory in the *original* sealed protocol.  An
        unplanned analysis, a changed one, or any analysis registered against a
        post-freeze branch is demoted to ``exploratory``, and the before and
        after protocol hashes are preserved either way.
        """
        if not isinstance(analysis, PlannedAnalysis):
            _fail("INVALID_ANALYSIS", "a PlannedAnalysis is required", path="analysis")
        record = self.read(protocol_id)
        if not record.is_sealed:
            _fail(
                "ANALYSIS_REGISTRATION_REQUIRES_SEALED_PROTOCOL",
                "an analysis can only be registered against a sealed protocol",
                path="protocol_id",
                actual=protocol_id,
            )
        path = self._analysis_path(protocol_id, analysis.analysis_id)
        if path.is_file():
            _fail(
                "DUPLICATE_ANALYSIS_REGISTRATION",
                "an analysis is registered against a protocol exactly once",
                path="analysis_id",
                actual=analysis.analysis_id,
            )

        declared = self._declared_analysis(record, analysis.analysis_id)
        registered_hash = analysis.declaration_hash
        declared_hash = (
            content_hash(_declared_to_analysis(declared)) if declared is not None else None
        )
        origin_hash = (
            record.protocol.parent_hash if record.is_branch else record.protocol_hash
        )

        if record.is_branch:
            classification = AnalysisClassification.EXPLORATORY
            reason = "registered against a protocol branch created after the original freeze"
        elif declared is None:
            classification = AnalysisClassification.EXPLORATORY
            reason = "the analysis is absent from the sealed protocol"
        elif declared_hash != registered_hash:
            classification = AnalysisClassification.EXPLORATORY
            reason = "the analysis differs from the version declared in the sealed protocol"
        elif AnalysisClassification(declared["classification"]) is AnalysisClassification.EXPLORATORY:
            classification = AnalysisClassification.EXPLORATORY
            reason = "the analysis was pre-registered as exploratory"
        else:
            classification = AnalysisClassification.CONFIRMATORY
            reason = "the analysis was declared unchanged in the sealed protocol"

        event = AnalysisLineageEvent(
            analysis_id=analysis.analysis_id,
            protocol_id=protocol_id,
            protocol_hash_at_declaration=str(origin_hash),
            protocol_hash_at_registration=record.protocol_hash,
            classification=classification,
            reason=reason,
            registered_at_utc=_utc_now(),
            declared_analysis_hash=declared_hash,
            registered_analysis_hash=registered_hash,
        )
        self._write_atomic(path, canonical_data(event))
        return event

    @staticmethod
    def _declared_analysis(
        record: ProtocolRecord, analysis_id: str
    ) -> Mapping[str, Any] | None:
        declared = record.protocol.specifications.get("analyses", ())
        for entry in declared:
            if isinstance(entry, Mapping) and entry.get("analysis_id") == analysis_id:
                return entry
        return None

    def analysis_events(self, protocol_id: str) -> tuple[AnalysisLineageEvent, ...]:
        directory = self.root / protocol_id / "analyses"
        if not directory.is_dir():
            return ()
        events = []
        for path in sorted(directory.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            events.append(
                AnalysisLineageEvent(
                    analysis_id=payload["analysis_id"],
                    protocol_id=payload["protocol_id"],
                    protocol_hash_at_declaration=payload["protocol_hash_at_declaration"],
                    protocol_hash_at_registration=payload["protocol_hash_at_registration"],
                    classification=AnalysisClassification(payload["classification"]),
                    reason=payload["reason"],
                    registered_at_utc=payload["registered_at_utc"],
                    declared_analysis_hash=payload["declared_analysis_hash"],
                    registered_analysis_hash=payload["registered_analysis_hash"],
                    schema_version=payload["schema_version"],
                    content_hash=payload["content_hash"],
                )
            )
        return tuple(events)


def _declared_to_analysis(declared: Mapping[str, Any]) -> PlannedAnalysis:
    return PlannedAnalysis(
        analysis_id=declared["analysis_id"],
        question_id=declared["question_id"],
        classification=AnalysisClassification(declared["classification"]),
        estimand=declared["estimand"],
        metric_id=declared["metric_id"],
        comparison=declared["comparison"],
        schema_version=declared["schema_version"],
    )


# ---------------------------------------------------------------------------
# Leakage guard and contamination propagation (Requirement 15.5-15.6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SelectionContext:
    """Everything one selection-facing decision is allowed to look at.

    Construction deliberately accepts held-out handles so a contaminated context
    can be detected and recorded; :func:`guard_selection_context` is what refuses
    it.
    """

    context_id: str
    purpose: SelectionPurpose
    handles: tuple[DataHandle[SplitScope], ...]
    run_id: str | None = None
    schema_version: str = PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _required_text(self.context_id, "context_id")
        object.__setattr__(self, "purpose", exactly_one(self.purpose, SelectionPurpose, path="purpose"))
        handles = tuple(self.handles)
        if any(not isinstance(handle, DataHandle) for handle in handles):
            _fail(
                "INVALID_SELECTION_HANDLE",
                "a selection context holds typed DataHandle values",
                path="handles",
            )
        object.__setattr__(self, "handles", handles)

    @property
    def leaked_handles(self) -> tuple[DataHandle[SplitScope], ...]:
        return tuple(handle for handle in self.handles if handle.scope in HELD_OUT_SCOPES)


@dataclass(frozen=True, slots=True)
class ContaminationRecord(PersistedModel):
    """A leaked handle, the context it reached, and every run it invalidates.

    A later claim gate consumes ``excluded_run_ids`` to drop the contaminated run
    and all of its descendants from confirmatory evidence (Requirement 15.6).
    """

    record_id: str
    context_id: str
    purpose: SelectionPurpose
    leaked_handles: tuple[DataHandle[SplitScope], ...]
    detected_at_utc: str
    reason: str
    excluded_run_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("record_id", "context_id", "detected_at_utc", "reason"):
            _required_text(getattr(self, name), name)
        object.__setattr__(self, "purpose", exactly_one(self.purpose, SelectionPurpose, path="purpose"))
        handles = tuple(self.leaked_handles)
        if not handles:
            _fail(
                "EMPTY_CONTAMINATION_RECORD",
                "a contamination record requires at least one leaked handle",
                path="leaked_handles",
            )
        object.__setattr__(self, "leaked_handles", handles)
        object.__setattr__(self, "excluded_run_ids", tuple(sorted(set(self.excluded_run_ids))))
        super(ContaminationRecord, self).__post_init__()

    @property
    def leaked_scopes(self) -> tuple[SplitScope, ...]:
        return tuple(sorted({handle.scope for handle in self.leaked_handles}, key=lambda item: item.value))

    @property
    def references_test_data(self) -> bool:
        return any(handle.scope is SplitScope.TEST for handle in self.leaked_handles)

    @property
    def references_target_city(self) -> bool:
        return any(handle.scope is SplitScope.CROSS_CITY for handle in self.leaked_handles)


class SelectionLeakageError(ResearchValidationError):
    """Raised when a selection path references held-out or cross-city data."""

    def __init__(self, contamination: ContaminationRecord) -> None:
        super().__init__(
            "SELECTION_PATH_LEAKAGE",
            "test and cross-city handles are forbidden in selection paths",
            path=f"selection.{contamination.purpose.value}",
            expected=[],
            actual=[handle.identifier for handle in contamination.leaked_handles],
            details={"contamination_record_hash": str(contamination.content_hash)},
        )
        self.contamination = contamination


def inspect_selection_context(context: SelectionContext) -> ContaminationRecord | None:
    """Return a contamination record if the context reaches held-out data, else ``None``."""
    if not isinstance(context, SelectionContext):
        _fail("INVALID_SELECTION_CONTEXT", "a SelectionContext is required", path="context")
    leaked = context.leaked_handles
    if not leaked:
        return None
    scopes = sorted({handle.scope.value for handle in leaked})
    return ContaminationRecord(
        record_id=f"contamination-{context.context_id}",
        context_id=context.context_id,
        purpose=context.purpose,
        leaked_handles=leaked,
        detected_at_utc=_utc_now(),
        reason=(
            f"{', '.join(scopes)} handles reached the {context.purpose.value} path, "
            "which may observe train and validation data only"
        ),
        excluded_run_ids=(context.run_id,) if context.run_id else (),
    )


def guard_selection_context(context: SelectionContext) -> TuningDataView:
    """Return the tuning capability for a clean context, or refuse the leaked one.

    The return type is the same typed view the trainer already consumes, so a
    caller that passes this gate holds no representation of held-out data at all.
    """
    contamination = inspect_selection_context(context)
    if contamination is not None:
        raise SelectionLeakageError(contamination)
    return TuningDataView(
        train=tuple(handle for handle in context.handles if handle.scope is SplitScope.TRAIN),
        validation=tuple(handle for handle in context.handles if handle.scope is SplitScope.VALIDATION),
    )


def contaminated_run_lineage(store: RunManifestStore, origin_run_id: str) -> tuple[str, ...]:
    """Every run whose lineage passes through ``origin_run_id``, itself included."""
    if not isinstance(store, RunManifestStore):
        _fail("INVALID_RUN_STORE", "a RunManifestStore is required", path="store")
    if not store.exists(origin_run_id):
        _fail(
            "RUN_NOT_FOUND",
            f"no manifest exists for run {origin_run_id!r}",
            path="origin_run_id",
            actual=origin_run_id,
        )
    descendants = [
        entry.name
        for entry in sorted(store.root.iterdir())
        if (entry / "manifest.json").is_file()
        and any(item.run_id == origin_run_id for item in store.lineage(entry.name))
    ]
    return tuple(descendants)


def propagate_contamination(
    record: ContaminationRecord, store: RunManifestStore, *, origin_run_id: str
) -> ContaminationRecord:
    """Extend a contamination record to every descendant of the tainted run."""
    if not isinstance(record, ContaminationRecord):
        _fail("INVALID_CONTAMINATION_RECORD", "a ContaminationRecord is required", path="record")
    excluded = set(record.excluded_run_ids) | set(contaminated_run_lineage(store, origin_run_id))
    return ContaminationRecord(
        record_id=record.record_id,
        context_id=record.context_id,
        purpose=record.purpose,
        leaked_handles=record.leaked_handles,
        detected_at_utc=record.detected_at_utc,
        reason=record.reason,
        excluded_run_ids=tuple(sorted(excluded)),
    )


# ---------------------------------------------------------------------------
# Confirmatory export gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ConfirmatoryGateReport(PersistedModel):
    """Whether a protocol and analysis may back a confirmatory claim."""

    protocol_id: str
    protocol_hash: str
    analysis_id: str | None
    eligible: bool
    errors: tuple[ErrorRecord, ...] = ()

    def __post_init__(self) -> None:
        _required_text(self.protocol_id, "protocol_id")
        _required_text(self.protocol_hash, "protocol_hash")
        errors = tuple(self.errors)
        if self.eligible == bool(errors):
            _fail(
                "INVALID_GATE_REPORT",
                "eligible reports carry no errors and rejected reports carry errors",
                path="errors",
            )
        object.__setattr__(self, "errors", errors)
        super(ConfirmatoryGateReport, self).__post_init__()

    @property
    def error_codes(self) -> tuple[str, ...]:
        return tuple(error.code for error in self.errors)


class ConfirmatoryGateError(ResearchValidationError):
    def __init__(self, report: ConfirmatoryGateReport) -> None:
        first = report.errors[0]
        super().__init__(
            first.code,
            first.message,
            path=first.path,
            expected=first.expected,
            actual=first.actual,
            details={**dict(first.details), "gate_report_hash": str(report.content_hash)},
        )
        self.report = report


def _error(code: str, message: str, **values: Any) -> ErrorRecord:
    return ErrorRecord(code=code, message=message, **values)


def inspect_confirmatory_eligibility(
    record: ProtocolRecord,
    *,
    analysis_event: AnalysisLineageEvent | None = None,
    selection_contexts: Sequence[SelectionContext] = (),
    contamination: Sequence[ContaminationRecord] = (),
) -> ConfirmatoryGateReport:
    """Fail-closed check that confirmatory evidence may be exported.

    Rejects an unsealed protocol, an exploratory analysis, an unresolved
    test-data dependency in any selection path, target-city adaptation, an
    exploratory-power sample plan, and any known contaminated lineage.
    """
    if not isinstance(record, ProtocolRecord):
        _fail("INVALID_PROTOCOL_RECORD", "a ProtocolRecord is required", path="record")
    errors: list[ErrorRecord] = []

    if not record.is_sealed:
        errors.append(
            _error(
                "PROTOCOL_NOT_SEALED",
                "confirmatory evidence requires a sealed protocol",
                path="protocol.state",
                expected=ProtocolState.SEALED.value,
                actual=record.protocol.state.value,
            )
        )
    else:
        try:
            validate_protocol_content(record.protocol.questions, record.protocol.specifications)
        except ResearchValidationError as exc:
            errors.append(exc.record)

    if analysis_event is not None and not analysis_event.is_confirmatory:
        errors.append(
            _error(
                "ANALYSIS_NOT_CONFIRMATORY",
                "the analysis was demoted to exploratory and cannot back a confirmatory claim",
                path="analysis.classification",
                expected=AnalysisClassification.CONFIRMATORY.value,
                actual=analysis_event.classification.value,
                details={
                    "analysis_id": analysis_event.analysis_id,
                    "protocol_hash_at_declaration": analysis_event.protocol_hash_at_declaration,
                    "protocol_hash_at_registration": analysis_event.protocol_hash_at_registration,
                    "reason": analysis_event.reason,
                },
            )
        )

    records = [item for item in contamination]
    for context in selection_contexts:
        detected = inspect_selection_context(context)
        if detected is not None:
            records.append(detected)
    for item in records:
        if item.references_test_data:
            errors.append(
                _error(
                    "TEST_DATA_DEPENDENCY",
                    "a selection path depends on held-out test data",
                    path=f"selection.{item.purpose.value}",
                    expected=[],
                    actual=[
                        handle.identifier
                        for handle in item.leaked_handles
                        if handle.scope is SplitScope.TEST
                    ],
                    details={"contamination_record_hash": str(item.content_hash)},
                )
            )
        if item.references_target_city:
            errors.append(
                _error(
                    "TARGET_CITY_ADAPTATION",
                    "a selection path adapts to a zero-shot target city",
                    path=f"selection.{item.purpose.value}",
                    expected=[],
                    actual=[
                        handle.identifier
                        for handle in item.leaked_handles
                        if handle.scope is SplitScope.CROSS_CITY
                    ],
                    details={"contamination_record_hash": str(item.content_hash)},
                )
            )
        if item.excluded_run_ids:
            errors.append(
                _error(
                    "CONTAMINATED_LINEAGE",
                    "the contaminated run and its descendants are excluded from confirmatory evidence",
                    path="runs",
                    expected=[],
                    actual=list(item.excluded_run_ids),
                    details={"contamination_record_hash": str(item.content_hash)},
                )
            )

    if record.is_sealed:
        sample = record.protocol.specifications.get("sample")
        status = sample.get("plan", {}).get("status") if isinstance(sample, Mapping) else None
        if status == SampleSizeStatus.EXPLORATORY.value:
            errors.append(
                _error(
                    "EXPLORATORY_SAMPLE_SIZE",
                    "the pre-registered sample size falls below the confirmatory floor",
                    path="specifications.sample.plan.status",
                    expected=[SampleSizeStatus.DEFAULT.value, SampleSizeStatus.REDUCED.value],
                    actual=status,
                )
            )

    return ConfirmatoryGateReport(
        protocol_id=record.protocol_id,
        protocol_hash=record.protocol_hash,
        analysis_id=analysis_event.analysis_id if analysis_event is not None else None,
        eligible=not errors,
        errors=tuple(errors),
    )


def require_confirmatory_eligibility(
    record: ProtocolRecord,
    *,
    analysis_event: AnalysisLineageEvent | None = None,
    selection_contexts: Sequence[SelectionContext] = (),
    contamination: Sequence[ContaminationRecord] = (),
) -> ConfirmatoryGateReport:
    report = inspect_confirmatory_eligibility(
        record,
        analysis_event=analysis_event,
        selection_contexts=selection_contexts,
        contamination=contamination,
    )
    if not report.eligible:
        raise ConfirmatoryGateError(report)
    return report


__all__ = (
    "HELD_OUT_SCOPES",
    "PROTOCOL_SCHEMA_VERSION",
    "REQUIRED_QUESTION_IDS",
    "REQUIRED_SPECIFICATION_SLOTS",
    "RQ_COMPONENT_CONTRIBUTION",
    "RQ_CONTAINMENT_STABILITY",
    "RQ_OFFLINE_LLM_VALUE",
    "RQ_SPATIAL_GENERALIZATION",
    "SELECTION_SCOPES",
    "AnalysisLineageEvent",
    "ConditionRef",
    "ConfirmatoryGateError",
    "ConfirmatoryGateReport",
    "ContaminationRecord",
    "ExclusionSpec",
    "HypothesisSpec",
    "MetricDeclaration",
    "MetricRole",
    "PlannedAnalysis",
    "ProtocolRecord",
    "ProtocolSpecifications",
    "ProtocolStore",
    "ResourceSpec",
    "SampleSpec",
    "SearchSpec",
    "SelectionContext",
    "SelectionLeakageError",
    "SelectionPurpose",
    "SplitSpec",
    "StatisticsSpec",
    "StopRule",
    "StopSpec",
    "ToleranceSpec",
    "condition_refs",
    "contaminated_run_lineage",
    "default_research_questions",
    "generate_protocol_id",
    "guard_selection_context",
    "inspect_confirmatory_eligibility",
    "inspect_selection_context",
    "metric_declaration",
    "practical_thresholds_for",
    "propagate_contamination",
    "require_confirmatory_eligibility",
    "validate_protocol_content",
    "validate_questions",
    "validate_specification_slots",
)

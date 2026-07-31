"""Typed, immutable research records with canonical content identities."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Sequence, TypeVar

from .canonical import canonical_data, content_hash
from .errors import ResearchValidationError

RESEARCH_SCHEMA_VERSION = "1.0"
SUPPORTED_SCHEMA_VERSIONS = frozenset({RESEARCH_SCHEMA_VERSION})
E = TypeVar("E", bound=Enum)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


def _required(value: str, path: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ResearchValidationError("MISSING_REQUIRED_FIELD", f"{path} must be non-empty", path=path)


def validate_schema_version(version: str) -> None:
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ResearchValidationError(
            "UNSUPPORTED_SCHEMA_VERSION",
            f"Unsupported research schema version {version!r}",
            path="schema_version",
            expected=sorted(SUPPORTED_SCHEMA_VERSIONS),
            actual=version,
        )


def exactly_one(
    value: E | str | Sequence[E | str] | Mapping[E | str, bool] | None,
    enum_type: type[E],
    *,
    path: str,
    provenance_consistent: bool = True,
) -> E:
    """Return one enum member and reject missing, duplicate, unknown, or inconsistent input."""
    if not provenance_consistent:
        raise ResearchValidationError(
            "PROVENANCE_MISMATCH", f"{path} conflicts with provenance", path=path
        )
    if value is None:
        selected: list[E | str] = []
    elif isinstance(value, Mapping):
        selected = [item for item, enabled in value.items() if enabled]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        selected = list(value)
    else:
        selected = [value]
    if not selected:
        raise ResearchValidationError(
            "MISSING_CLASSIFICATION", f"{path} requires exactly one value", path=path
        )
    if len(selected) != 1:
        raise ResearchValidationError(
            "DUPLICATE_CLASSIFICATION",
            f"{path} requires exactly one value",
            path=path,
            expected=1,
            actual=len(selected),
        )
    try:
        return selected[0] if isinstance(selected[0], enum_type) else enum_type(selected[0])
    except (TypeError, ValueError) as exc:
        raise ResearchValidationError(
            "INVALID_CLASSIFICATION",
            f"{path} is not an allowed {enum_type.__name__}",
            path=path,
            expected=[item.value for item in enum_type],
            actual=selected[0],
        ) from exc


class EvidenceType(str, Enum):
    IMPLEMENTATION_EXISTENCE = "implementation_existence"
    TEST_RESULT = "test_result"
    EXPERIMENT_OBSERVATION = "experiment_observation"
    PAPER_CLAIM = "paper_claim"


class PriorResultStatus(str, Enum):
    VERIFIED = "verified"
    PRELIMINARY = "preliminary"
    INVALID = "invalid"


class TestStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    NOT_TESTED = "not_tested"


class AnalysisClassification(str, Enum):
    CONFIRMATORY = "confirmatory"
    EXPLORATORY = "exploratory"


class ScreeningDecision(str, Enum):
    INCLUDED = "included"
    EXCLUDED = "excluded"


class DataKind(str, Enum):
    ACTUAL_OSM_MAP = "actual_osm_map"
    SYNTHETIC_FIXTURE = "synthetic_fixture"


class MapScenario(str, Enum):
    INTERIOR_CONTAINED = "interior_contained"
    BOUNDARY_ESCAPE = "boundary_escape"


class EpisodeOutcome(str, Enum):
    CAPTURE = "capture"
    ESCAPE = "escape"
    TIMEOUT = "timeout"


class ExecutionStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    NOT_RUN = "not_run"


class ProtocolState(str, Enum):
    DRAFT = "draft"
    SEALED = "sealed"


class ManifestState(str, Enum):
    DRAFT = "draft"
    SEALED = "sealed"


class ClaimScope(str, Enum):
    IMPLEMENTATION = "implementation"
    RESULT = "result"
    NOVELTY = "novelty"
    FIELD_READINESS = "field_readiness"


class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    FALSIFIED = "falsified"
    INCONCLUSIVE = "inconclusive"
    PRELIMINARY = "preliminary"
    INVALID = "invalid"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True, kw_only=True)
class PersistedModel:
    schema_version: str = RESEARCH_SCHEMA_VERSION
    content_hash: str | None = None

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version)
        calculated = content_hash(self)
        if self.content_hash is not None and self.content_hash != calculated:
            raise ResearchValidationError(
                "CONTENT_HASH_MISMATCH",
                "content_hash does not match canonical model content",
                path="content_hash",
                expected=calculated,
                actual=self.content_hash,
            )
        object.__setattr__(self, "content_hash", calculated)

    def as_dict(self) -> dict[str, Any]:
        return canonical_data(self)

    def verify_hash(self) -> bool:
        return self.content_hash == content_hash(self)


@dataclass(frozen=True, slots=True)
class EvidenceRecord(PersistedModel):
    record_id: str
    evidence_type: EvidenceType
    producer: str
    created_at_utc: str
    method: str
    source: str
    extracted_value: Any
    verification: str
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("record_id", "producer", "created_at_utc", "method", "source", "verification"):
            _required(getattr(self, name), name)
        object.__setattr__(self, "evidence_type", exactly_one(self.evidence_type, EvidenceType, path="evidence_type"))
        object.__setattr__(self, "extracted_value", _freeze(self.extracted_value))
        object.__setattr__(self, "limitations", tuple(str(item) for item in self.limitations))
        super(EvidenceRecord, self).__post_init__()


@dataclass(frozen=True, slots=True)
class ResearchQuestion(PersistedModel):
    question_id: str
    analysis_class: AnalysisClassification
    primary_outcome: str
    directional_inequality: str
    practical_threshold: float
    threshold_unit: str
    decision_rule: str

    def __post_init__(self) -> None:
        for name in (
            "question_id", "primary_outcome", "directional_inequality", "threshold_unit", "decision_rule"
        ):
            _required(getattr(self, name), name)
        object.__setattr__(
            self,
            "analysis_class",
            exactly_one(self.analysis_class, AnalysisClassification, path="analysis_class"),
        )
        super(ResearchQuestion, self).__post_init__()


@dataclass(frozen=True, slots=True)
class ResearchProtocol(PersistedModel):
    protocol_id: str
    state: ProtocolState
    questions: tuple[ResearchQuestion, ...]
    specifications: Mapping[str, Any]
    frozen_at_utc: str | None = None
    signer: str | None = None
    parent_hash: str | None = None

    def __post_init__(self) -> None:
        _required(self.protocol_id, "protocol_id")
        object.__setattr__(self, "state", exactly_one(self.state, ProtocolState, path="state"))
        object.__setattr__(self, "questions", tuple(self.questions))
        if not self.questions:
            raise ResearchValidationError(
                "MISSING_REQUIRED_FIELD", "questions must not be empty", path="questions"
            )
        question_ids = [item.question_id for item in self.questions]
        if len(question_ids) != len(set(question_ids)):
            raise ResearchValidationError(
                "DUPLICATE_IDENTIFIER", "question identifiers must be unique", path="questions"
            )
        object.__setattr__(self, "specifications", _freeze(self.specifications))
        if self.state is ProtocolState.SEALED:
            _required(self.frozen_at_utc or "", "frozen_at_utc")
            _required(self.signer or "", "signer")
        super(ResearchProtocol, self).__post_init__()


@dataclass(frozen=True, slots=True)
class Condition(PersistedModel):
    condition_id: str
    policy: str
    observation: str
    reward: str
    placement: str
    uturn: str
    hysteresis: str
    scenario: MapScenario
    evader: str
    budget: Mapping[str, Any]
    llm_artifact_hash: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "condition_id", "policy", "observation", "reward", "placement", "uturn", "hysteresis", "evader"
        ):
            _required(getattr(self, name), name)
        object.__setattr__(self, "scenario", exactly_one(self.scenario, MapScenario, path="scenario"))
        object.__setattr__(self, "budget", _freeze(self.budget))
        super(Condition, self).__post_init__()


@dataclass(frozen=True, slots=True)
class MapRecord(PersistedModel):
    map_id: str
    data_kind: DataKind
    scenario: MapScenario
    network_hash: str
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        _required(self.map_id, "map_id")
        _required(self.network_hash, "network_hash")
        object.__setattr__(self, "data_kind", exactly_one(self.data_kind, DataKind, path="data_kind"))
        object.__setattr__(self, "scenario", exactly_one(self.scenario, MapScenario, path="scenario"))
        object.__setattr__(self, "provenance", _freeze(self.provenance))
        if not self.provenance:
            raise ResearchValidationError(
                "MISSING_REQUIRED_FIELD", "map provenance must not be empty", path="provenance"
            )
        super(MapRecord, self).__post_init__()


@dataclass(frozen=True, slots=True)
class RunRecord(PersistedModel):
    run_id: str
    protocol_hash: str
    condition_hash: str
    execution_status: ExecutionStatus | None
    status_reason: str | None
    manifest_state: ManifestState = ManifestState.DRAFT
    parent_id: str | None = None
    artifact_hashes: Mapping[str, str] = field(default_factory=dict)
    result: Any = None

    def __post_init__(self) -> None:
        for name in ("run_id", "protocol_hash", "condition_hash"):
            _required(getattr(self, name), name)
        object.__setattr__(
            self, "manifest_state", exactly_one(self.manifest_state, ManifestState, path="manifest_state")
        )
        if self.execution_status is not None:
            object.__setattr__(
                self,
                "execution_status",
                exactly_one(self.execution_status, ExecutionStatus, path="execution_status"),
            )
            _required(self.status_reason or "", "status_reason")
        elif self.manifest_state is ManifestState.SEALED:
            raise ResearchValidationError(
                "MISSING_CLASSIFICATION",
                "sealed runs require one terminal execution status",
                path="execution_status",
            )
        object.__setattr__(self, "artifact_hashes", _freeze(self.artifact_hashes))
        object.__setattr__(self, "result", _freeze(self.result))
        super(RunRecord, self).__post_init__()


@dataclass(frozen=True, slots=True)
class ClaimRecord(PersistedModel):
    claim_id: str
    text: str
    scope: ClaimScope
    status: ClaimStatus
    evidence_ids: tuple[str, ...]
    analysis_ids: tuple[str, ...] = ()
    citation_ids: tuple[str, ...] = ()
    comparison_axes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _required(self.claim_id, "claim_id")
        _required(self.text, "text")
        object.__setattr__(self, "scope", exactly_one(self.scope, ClaimScope, path="scope"))
        object.__setattr__(self, "status", exactly_one(self.status, ClaimStatus, path="status"))
        for name in ("evidence_ids", "analysis_ids", "citation_ids", "comparison_axes"):
            object.__setattr__(self, name, tuple(str(item) for item in getattr(self, name)))
        if self.scope is ClaimScope.NOVELTY:
            normalized = self.text.casefold()
            forbidden = ("최초", "유일", " first ", " only ")
            padded = f" {normalized} "
            if any(term in padded for term in forbidden):
                raise ResearchValidationError(
                    "FORBIDDEN_PRIORITY_CLAIM",
                    "Novelty claims must not use priority or exclusivity language",
                    path="text",
                )
            if not self.evidence_ids or not self.comparison_axes:
                raise ResearchValidationError(
                    "INCOMPLETE_NOVELTY_EVIDENCE",
                    "Novelty claims require evidence and a direct comparison axis",
                    path="evidence_ids",
                )
        if self.scope is ClaimScope.FIELD_READINESS and self.status is not ClaimStatus.UNSUPPORTED:
            raise ResearchValidationError(
                "FIELD_READINESS_UNSUPPORTED",
                "Field readiness must remain unsupported",
                path="status",
            )
        super(ClaimRecord, self).__post_init__()


__all__ = (
    "RESEARCH_SCHEMA_VERSION",
    "AnalysisClassification",
    "ClaimRecord",
    "ClaimScope",
    "ClaimStatus",
    "Condition",
    "DataKind",
    "EpisodeOutcome",
    "EvidenceRecord",
    "EvidenceType",
    "ExecutionStatus",
    "ManifestState",
    "MapRecord",
    "MapScenario",
    "PersistedModel",
    "PriorResultStatus",
    "ProtocolState",
    "ResearchProtocol",
    "ResearchQuestion",
    "RunRecord",
    "ScreeningDecision",
    "TestStatus",
    "exactly_one",
    "validate_schema_version",
)

"""Reality-limits, ethics, and Future_Work_System scope schema (Requirements 17, 18).

This module is the *schema and check* layer that a later Paper_Claim_Gate calls.
It deliberately does not re-derive what :mod:`pursuit_evasion_rl.research.domain`
already guarantees: a :class:`~pursuit_evasion_rl.research.domain.ClaimRecord`
with ``scope=ClaimScope.FIELD_READINESS`` already refuses any status other than
``unsupported`` (Requirement 17.1).  What is added here is everything that lives
*above* a single claim record:

* the mandatory disclosure text that must accompany any performance narration
  (17.2) and the non-substitution statement (17.3);
* a denylist check that refuses to let a Practical_Threshold-exceeding result be
  restated as field safety, crime reduction, or real capture-rate gain (17.4);
* exactly five risk categories (17.5) and exactly six independently reported
  pre-field validation categories (17.6-17.8), with no aggregate "ready" shortcut;
* the four Future_Work_Systems (18.1-18.4), each artifact carrying the
  ``future_work_demo`` label and stored outside the primary result tree (18.6);
* :func:`assert_no_future_work_dependency`, which blocks a future-work artifact
  from backing a Novelty_Claim, a success criterion, a Field_Readiness ground, or
  a primary result (18.5, 18.8);
* two structurally distinct scope sections so the manuscript places current work
  and future work apart (18.7).

CCTV, ANPR, the command dashboard, and drone integration are *not* implemented
anywhere in this repository.  They exist here only as classification labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Iterable, Mapping, Sequence

from ..canonical import content_hash
from ..domain import ClaimRecord, ClaimScope, ClaimStatus, exactly_one
from ..errors import ResearchValidationError

SCOPE_SCHEMA_VERSION = "1.0"


def _require_text(value: object, path: str, code: str = "MISSING_REQUIRED_FIELD") -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResearchValidationError(code, f"{path} must be a non-empty statement", path=path)
    return value


# ---------------------------------------------------------------------------
# Requirement 17.2 -- unmodeled aspects that must accompany any performance result
# ---------------------------------------------------------------------------

SIMULATION_LIMITATION_FIELDS: tuple[str, ...] = (
    "traffic_and_vehicle_dynamics",
    "sensor_error",
    "communication_latency_and_loss",
    "human_behavior",
    "legal_and_operational_constraints",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class SimulationLimitationDisclosure:
    """The five unmodeled aspects, each as prose rather than a boolean flag.

    A flag would let a bundle satisfy Requirement 17.2 without telling the reader
    *what* is unmodeled, so every field must carry text that says something.
    """

    traffic_and_vehicle_dynamics: str
    sensor_error: str
    communication_latency_and_loss: str
    human_behavior: str
    legal_and_operational_constraints: str
    schema_version: str = SCOPE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in SIMULATION_LIMITATION_FIELDS:
            _require_text(getattr(self, name), name, "MISSING_SIMULATION_LIMITATION")


# ---------------------------------------------------------------------------
# Requirement 17.3 -- the system does not replace real police decisions
# ---------------------------------------------------------------------------

DEFAULT_NON_SUBSTITUTION_STATEMENT = (
    "본 Research_System은 연구용 시뮬레이션이며, 실제 경찰 지휘 및 추격 의사결정을 "
    "대체하지 않는다. This research simulation does not replace real police command "
    "or pursuit decisions."
)


@dataclass(frozen=True, slots=True, kw_only=True)
class NonSubstitutionStatement:
    statement: str = DEFAULT_NON_SUBSTITUTION_STATEMENT
    schema_version: str = SCOPE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.statement, "statement", "MISSING_NON_SUBSTITUTION_STATEMENT")


# ---------------------------------------------------------------------------
# Requirement 17.4 -- simulation results are never restated as field outcomes
# ---------------------------------------------------------------------------

#: Phrases that convert a simulation number into a field-safety, crime-reduction,
#: or real-capture-rate claim.  Matching is casefolded substring matching, which
#: is deliberately blunt: this is a tripwire for reviewers, not an NLP system.
#: Callers extend it per venue through the ``extra_patterns`` argument rather
#: than editing this tuple.
FIELD_READINESS_DERIVATION_PATTERNS: tuple[str, ...] = (
    "현장 안전",
    "현장안전",
    "실제 검거율",
    "검거율 향상",
    "검거율 개선",
    "범죄 감소",
    "범죄율 감소",
    "치안 개선",
    "실전 배치",
    "현장 배치",
    "field safety",
    "field-safety",
    "real-world safety",
    "crime reduction",
    "crime rate reduction",
    "reduce crime",
    "real-world capture rate",
    "real world capture rate",
    "real-world arrest rate",
    "actual arrest rate",
    "field-ready",
    "field ready",
    "deployment-ready",
    "ready for deployment",
    "operational readiness",
)


def find_field_readiness_derivations(
    text: str, *, extra_patterns: Sequence[str] = ()
) -> tuple[str, ...]:
    """Return every banned derivation phrase present in ``text``, in pattern order."""
    haystack = text.casefold()
    patterns = tuple(FIELD_READINESS_DERIVATION_PATTERNS) + tuple(extra_patterns)
    return tuple(pattern for pattern in patterns if pattern.casefold() in haystack)


def assert_no_field_readiness_derivation(
    text: str, *, path: str = "text", extra_patterns: Sequence[str] = ()
) -> None:
    """Reject text that converts a simulation result into a field-outcome claim.

    Apply this to result narration and to claim text -- not to the limitation or
    validation disclosures, which legitimately discuss field concepts in order to
    say they are *out* of scope.
    """
    matched = find_field_readiness_derivations(text, extra_patterns=extra_patterns)
    if matched:
        raise ResearchValidationError(
            "FIELD_READINESS_DERIVATION_BLOCKED",
            "simulation results must not be restated as field safety, crime reduction, "
            "or real capture-rate improvement",
            path=path,
            actual=list(matched),
        )


# ---------------------------------------------------------------------------
# Requirement 17.5 -- exactly five risk categories
# ---------------------------------------------------------------------------


class RiskCategory(str, Enum):
    EXCESSIVE_PURSUIT_INDUCEMENT = "excessive_pursuit_inducement"
    REGIONAL_BIAS = "regional_bias"
    SURVEILLANCE_EXPANSION = "surveillance_expansion"
    AUTOMATION_BIAS = "automation_bias"
    POLICY_MISUSE = "policy_misuse"


REQUIRED_RISK_CATEGORIES: frozenset[RiskCategory] = frozenset(RiskCategory)


@dataclass(frozen=True, slots=True, kw_only=True)
class RiskStatement:
    category: RiskCategory
    description: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "category", exactly_one(self.category, RiskCategory, path="category")
        )
        _require_text(self.description, "description", "MISSING_RISK_DESCRIPTION")


@dataclass(frozen=True, slots=True, kw_only=True)
class RiskDisclosure:
    """All five risk categories, each with its own description (Requirement 17.5)."""

    statements: tuple[RiskStatement, ...]
    schema_version: str = SCOPE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        statements = tuple(self.statements)
        object.__setattr__(self, "statements", statements)
        seen = [item.category for item in statements]
        duplicates = sorted({item.value for item in seen if seen.count(item) > 1})
        if duplicates:
            raise ResearchValidationError(
                "DUPLICATE_RISK_CATEGORY",
                "each risk category is recorded exactly once",
                path="statements",
                actual=duplicates,
            )
        missing = sorted(item.value for item in REQUIRED_RISK_CATEGORIES - set(seen))
        if missing:
            raise ResearchValidationError(
                "MISSING_RISK_CATEGORY",
                "risk reporting requires exactly the five named risk categories",
                path="statements",
                expected=sorted(item.value for item in REQUIRED_RISK_CATEGORIES),
                actual=missing,
            )

    @classmethod
    def from_mapping(cls, descriptions: Mapping[RiskCategory | str, str]) -> RiskDisclosure:
        return cls(
            statements=tuple(
                RiskStatement(category=RiskCategory(key), description=value)
                for key, value in descriptions.items()
            )
        )

    def description_of(self, category: RiskCategory | str) -> str:
        wanted = RiskCategory(category)
        for statement in self.statements:
            if statement.category is wanted:
                return statement.description
        raise ResearchValidationError(
            "MISSING_RISK_CATEGORY", f"{wanted.value} is not disclosed", path="statements"
        )


# ---------------------------------------------------------------------------
# Requirement 17.6-17.8 -- exactly six independently reported validation categories
# ---------------------------------------------------------------------------


class PreFieldValidationCategory(str, Enum):
    TRAFFIC_MICROSIMULATION = "traffic_microsimulation"
    SENSOR_UNCERTAINTY_ASSESSMENT = "sensor_uncertainty_assessment"
    HUMAN_IN_THE_LOOP_EVALUATION = "human_in_the_loop_evaluation"
    SAFETY_VERIFICATION = "safety_verification"
    LEGAL_REVIEW = "legal_review"
    CONTROLLED_PILOT = "controlled_pilot"


class PreFieldValidationStatus(str, Enum):
    NOT_PERFORMED = "not_performed"
    FAILED = "failed"
    PASSED = "passed"


REQUIRED_PRE_FIELD_VALIDATION_CATEGORIES: frozenset[PreFieldValidationCategory] = frozenset(
    PreFieldValidationCategory
)


@dataclass(frozen=True, slots=True, kw_only=True)
class PreFieldValidationEntry:
    """One category's own identifier and exactly one status (Requirement 17.7)."""

    category: PreFieldValidationCategory
    validation_id: str
    status: PreFieldValidationStatus
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "category",
            exactly_one(self.category, PreFieldValidationCategory, path="category"),
        )
        _require_text(self.validation_id, "validation_id", "MISSING_VALIDATION_IDENTIFIER")
        object.__setattr__(
            self, "status", exactly_one(self.status, PreFieldValidationStatus, path="status")
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class PreFieldValidationReport:
    """All six pre-field validation categories, reported independently.

    Requirement 17.8 forbids one category's result from standing in for another,
    so this type exposes no aggregate: there is no ``all_passed``, no passed
    count, and no truthiness. Readiness is only ever readable one category at a
    time through :meth:`status_of`, and :func:`assert_no_cross_category_substitution`
    rejects any attempt to cite one category's identifier under another.
    """

    entries: tuple[PreFieldValidationEntry, ...]
    schema_version: str = SCOPE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        object.__setattr__(self, "entries", entries)
        seen = [item.category for item in entries]
        duplicates = sorted({item.value for item in seen if seen.count(item) > 1})
        if duplicates:
            raise ResearchValidationError(
                "DUPLICATE_PRE_FIELD_VALIDATION_CATEGORY",
                "each pre-field validation category is recorded exactly once",
                path="entries",
                actual=duplicates,
            )
        missing = sorted(
            item.value for item in REQUIRED_PRE_FIELD_VALIDATION_CATEGORIES - set(seen)
        )
        if missing:
            raise ResearchValidationError(
                "MISSING_PRE_FIELD_VALIDATION_CATEGORY",
                "pre-field validation requires exactly the six named categories",
                path="entries",
                expected=sorted(
                    item.value for item in REQUIRED_PRE_FIELD_VALIDATION_CATEGORIES
                ),
                actual=missing,
            )
        identifiers = [item.validation_id for item in entries]
        if len(identifiers) != len(set(identifiers)):
            raise ResearchValidationError(
                "DUPLICATE_IDENTIFIER",
                "each pre-field validation category requires its own identifier",
                path="entries",
                actual=sorted(identifiers),
            )

    def entry_of(self, category: PreFieldValidationCategory | str) -> PreFieldValidationEntry:
        wanted = PreFieldValidationCategory(category)
        for entry in self.entries:
            if entry.category is wanted:
                return entry
        raise ResearchValidationError(
            "MISSING_PRE_FIELD_VALIDATION_CATEGORY",
            f"{wanted.value} is not reported",
            path="entries",
        )

    def status_of(self, category: PreFieldValidationCategory | str) -> PreFieldValidationStatus:
        """That one category's own status -- never a proxy for any other category."""
        return self.entry_of(category).status


def assert_no_cross_category_substitution(
    report: PreFieldValidationReport,
    *,
    category: PreFieldValidationCategory | str,
    cited_validation_ids: Iterable[str],
    path: str = "cited_validation_ids",
) -> None:
    """Reject citing one category's validation identifier as another's grounds (17.8)."""
    wanted = PreFieldValidationCategory(category)
    owner = {entry.validation_id: entry.category for entry in report.entries}
    foreign = sorted(
        f"{identifier}->{owner[identifier].value}"
        for identifier in cited_validation_ids
        if identifier in owner and owner[identifier] is not wanted
    )
    if foreign:
        raise ResearchValidationError(
            "CROSS_CATEGORY_VALIDATION_SUBSTITUTION",
            "one pre-field validation category cannot ground another category's status",
            path=path,
            expected=wanted.value,
            actual=foreign,
        )


# ---------------------------------------------------------------------------
# Requirement 18.1-18.6 -- Future_Work_Systems, labelled and stored separately
# ---------------------------------------------------------------------------


class FutureWorkSystem(str, Enum):
    """The four systems that are classified as future work only, never implemented."""

    CCTV = "cctv"
    ANPR = "anpr"
    REALTIME_COMMAND_DASHBOARD = "realtime_command_dashboard"
    DRONE_HETEROGENEOUS_AGENT = "drone_heterogeneous_agent"


FUTURE_WORK_SYSTEMS: frozenset[FutureWorkSystem] = frozenset(FutureWorkSystem)

#: Requirement 18.6: the exact label every future-work demo artifact must carry.
FUTURE_WORK_DEMO_LABEL = "future_work_demo"


@dataclass(frozen=True, slots=True, kw_only=True)
class FutureWorkArtifact:
    artifact_id: str
    system: FutureWorkSystem
    description: str
    artifact_path: str
    label: str = FUTURE_WORK_DEMO_LABEL
    schema_version: str = SCOPE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.artifact_id, "artifact_id")
        _require_text(self.description, "description")
        _require_text(self.artifact_path, "artifact_path")
        object.__setattr__(
            self, "system", exactly_one(self.system, FutureWorkSystem, path="system")
        )
        if self.label != FUTURE_WORK_DEMO_LABEL:
            raise ResearchValidationError(
                "INVALID_FUTURE_WORK_LABEL",
                f"future-work demos must be labelled {FUTURE_WORK_DEMO_LABEL!r}",
                path="label",
                expected=FUTURE_WORK_DEMO_LABEL,
                actual=self.label,
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class FutureWorkRegistry:
    """Future-work artifacts kept under a path root disjoint from primary results.

    The two roots must not nest in either direction, which is what makes
    "separate path" (Requirement 18.6) a checkable property rather than a naming
    convention.
    """

    primary_result_root: str
    future_work_root: str
    artifacts: tuple[FutureWorkArtifact, ...] = ()
    schema_version: str = SCOPE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.primary_result_root, "primary_result_root")
        _require_text(self.future_work_root, "future_work_root")
        primary = PurePosixPath(self.primary_result_root)
        future = PurePosixPath(self.future_work_root)
        if future.is_relative_to(primary) or primary.is_relative_to(future):
            raise ResearchValidationError(
                "FUTURE_WORK_PATH_NOT_SEPARATED",
                "future-work artifacts must not share a tree with primary results",
                path="future_work_root",
                expected=f"disjoint from {primary}",
                actual=str(future),
            )
        artifacts = tuple(self.artifacts)
        object.__setattr__(self, "artifacts", artifacts)
        identifiers = [item.artifact_id for item in artifacts]
        if len(identifiers) != len(set(identifiers)):
            raise ResearchValidationError(
                "DUPLICATE_IDENTIFIER",
                "future-work artifact identifiers must be unique",
                path="artifacts",
                actual=sorted(identifiers),
            )
        for artifact in artifacts:
            location = PurePosixPath(artifact.artifact_path)
            if location.is_relative_to(primary) or not location.is_relative_to(future):
                raise ResearchValidationError(
                    "FUTURE_WORK_PATH_NOT_SEPARATED",
                    f"{artifact.artifact_id} must be stored under the future-work root "
                    "and outside the primary result root",
                    path="artifact_path",
                    expected=str(future),
                    actual=str(location),
                )

    @property
    def artifact_ids(self) -> frozenset[str]:
        return frozenset(item.artifact_id for item in self.artifacts)

    def system_of(self, artifact_id: str) -> FutureWorkSystem | None:
        for artifact in self.artifacts:
            if artifact.artifact_id == artifact_id:
                return artifact.system
        return None


# ---------------------------------------------------------------------------
# Requirement 18.5, 18.8 -- future work never grounds a core claim
# ---------------------------------------------------------------------------


class ProhibitedConnection(str, Enum):
    """The connection kinds Requirements 18.5 and 18.8 forbid future work from taking."""

    NOVELTY_EVIDENCE = "novelty_evidence"
    SUCCESS_CRITERION = "success_criterion"
    FIELD_READINESS_GROUND = "field_readiness_ground"
    PRIMARY_RESULT = "primary_result"


_SCOPE_CONNECTION: Mapping[ClaimScope, ProhibitedConnection] = {
    ClaimScope.NOVELTY: ProhibitedConnection.NOVELTY_EVIDENCE,
    ClaimScope.RESULT: ProhibitedConnection.PRIMARY_RESULT,
    ClaimScope.FIELD_READINESS: ProhibitedConnection.FIELD_READINESS_GROUND,
}


def assert_no_future_work_dependency(
    references: ClaimRecord | Iterable[str],
    registry: FutureWorkRegistry,
    *,
    connection: ProhibitedConnection | str | None = None,
    path: str = "evidence_ids",
) -> None:
    """Block any core claim or success criterion that depends on a future-work artifact.

    ``references`` is either a :class:`ClaimRecord` -- whose evidence, analysis,
    and comparison-axis references are all checked and whose scope selects the
    connection kind -- or a bare iterable of identifiers, in which case
    ``connection`` names which prohibited link is being screened.

    An ``implementation``-scope claim is not screened: Requirement 18.6 permits a
    labelled future-work demo to exist and to be described as such.  What is
    forbidden is that demo grounding novelty, success, field readiness, or a
    primary result.
    """
    if isinstance(references, ClaimRecord):
        if connection is None:
            resolved = _SCOPE_CONNECTION.get(references.scope)
            if resolved is None:
                return
        else:
            resolved = ProhibitedConnection(connection)
        identifiers: tuple[str, ...] = (
            references.evidence_ids + references.analysis_ids + references.comparison_axes
        )
        subject = references.claim_id
    else:
        if connection is None:
            raise ResearchValidationError(
                "MISSING_CLASSIFICATION",
                "connection must name which prohibited link is being screened",
                path="connection",
                expected=[item.value for item in ProhibitedConnection],
            )
        resolved = ProhibitedConnection(connection)
        identifiers = tuple(str(item) for item in references)
        subject = resolved.value

    offending = sorted(identifier for identifier in identifiers if identifier in registry.artifact_ids)
    if offending:
        raise ResearchValidationError(
            "FUTURE_WORK_DEPENDENCY_BLOCKED",
            f"{resolved.value} must not depend on a {FUTURE_WORK_DEMO_LABEL} artifact",
            path=path,
            expected=subject,
            actual=offending,
            details={
                "connection": resolved.value,
                "systems": sorted(
                    {registry.system_of(item).value for item in offending}  # type: ignore[union-attr]
                ),
            },
        )


# ---------------------------------------------------------------------------
# Requirement 18.7 -- current scope and future work live in different sections
# ---------------------------------------------------------------------------


class ScopeSectionKind(str, Enum):
    CURRENT_RESEARCH_SCOPE = "current_research_scope"
    FUTURE_WORK = "future_work"


@dataclass(frozen=True, slots=True, kw_only=True)
class ScopeSection:
    section_id: str
    kind: ScopeSectionKind
    title: str
    body: str
    systems: tuple[FutureWorkSystem, ...] = ()
    schema_version: str = SCOPE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.section_id, "section_id")
        _require_text(self.title, "title")
        _require_text(self.body, "body")
        object.__setattr__(self, "kind", exactly_one(self.kind, ScopeSectionKind, path="kind"))
        object.__setattr__(
            self, "systems", tuple(FutureWorkSystem(item) for item in self.systems)
        )
        if self.kind is ScopeSectionKind.CURRENT_RESEARCH_SCOPE and self.systems:
            raise ResearchValidationError(
                "FUTURE_WORK_SYSTEM_IN_CURRENT_SCOPE",
                "future-work systems must not appear in the current research scope section",
                path="systems",
                actual=[item.value for item in self.systems],
            )
        if self.kind is ScopeSectionKind.FUTURE_WORK and not self.systems:
            raise ResearchValidationError(
                "MISSING_REQUIRED_FIELD",
                "the future-work section must name the systems it defers",
                path="systems",
            )


def assert_distinct_scope_sections(sections: Sequence[ScopeSection]) -> None:
    """Require exactly one current-scope section and one future-work section (18.7)."""
    kinds = [section.kind for section in sections]
    for kind in ScopeSectionKind:
        if kinds.count(kind) != 1:
            raise ResearchValidationError(
                "SCOPE_SECTION_NOT_SEPARATED",
                "current research scope and future work require exactly one section each",
                path="sections",
                expected=kind.value,
                actual=kinds.count(kind),
            )
    identifiers = [section.section_id for section in sections]
    if len(identifiers) != len(set(identifiers)):
        raise ResearchValidationError(
            "DUPLICATE_IDENTIFIER",
            "scope section identifiers must be unique",
            path="sections",
            actual=sorted(identifiers),
        )


# ---------------------------------------------------------------------------
# The assembled bundle a paper generator and claim gate consume
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class ScopeEthicsBundle:
    limitations: SimulationLimitationDisclosure
    non_substitution: NonSubstitutionStatement
    risks: RiskDisclosure
    pre_field_validation: PreFieldValidationReport
    future_work: FutureWorkRegistry
    sections: tuple[ScopeSection, ...]
    schema_version: str = SCOPE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "sections", tuple(self.sections))
        assert_distinct_scope_sections(self.sections)

    @property
    def bundle_hash(self) -> str:
        return content_hash(self)

    def field_readiness_claim(
        self,
        *,
        claim_id: str,
        text: str,
        evidence_ids: Sequence[str] = (),
        observed_capture_rate: float | None = None,
    ) -> ClaimRecord:
        """Build the Field_Readiness_Claim, always ``unsupported`` (Requirement 17.1).

        ``observed_capture_rate`` is accepted only so callers do not have to hide
        their numbers; it never reaches the status, which is why the same claim
        comes back unsupported for a 0.99 capture rate and a 0.01 one.  The text
        is screened for field-outcome derivations first (Requirement 17.4).
        """
        if observed_capture_rate is not None:
            float(observed_capture_rate)
        assert_no_field_readiness_derivation(text, path="text")
        assert_no_future_work_dependency(
            tuple(evidence_ids),
            self.future_work,
            connection=ProhibitedConnection.FIELD_READINESS_GROUND,
        )
        return ClaimRecord(
            claim_id=claim_id,
            text=text,
            scope=ClaimScope.FIELD_READINESS,
            status=ClaimStatus.UNSUPPORTED,
            evidence_ids=tuple(evidence_ids),
        )


__all__ = (
    "DEFAULT_NON_SUBSTITUTION_STATEMENT",
    "FIELD_READINESS_DERIVATION_PATTERNS",
    "FUTURE_WORK_DEMO_LABEL",
    "FUTURE_WORK_SYSTEMS",
    "REQUIRED_PRE_FIELD_VALIDATION_CATEGORIES",
    "REQUIRED_RISK_CATEGORIES",
    "SCOPE_SCHEMA_VERSION",
    "SIMULATION_LIMITATION_FIELDS",
    "FutureWorkArtifact",
    "FutureWorkRegistry",
    "FutureWorkSystem",
    "NonSubstitutionStatement",
    "PreFieldValidationCategory",
    "PreFieldValidationEntry",
    "PreFieldValidationReport",
    "PreFieldValidationStatus",
    "ProhibitedConnection",
    "RiskCategory",
    "RiskDisclosure",
    "RiskStatement",
    "ScopeEthicsBundle",
    "ScopeSection",
    "ScopeSectionKind",
    "SimulationLimitationDisclosure",
    "assert_distinct_scope_sections",
    "assert_no_cross_category_substitution",
    "assert_no_field_readiness_derivation",
    "assert_no_future_work_dependency",
    "find_field_readiness_derivations",
)

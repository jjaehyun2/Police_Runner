"""The fail-closed Paper_Claim_Gate (Requirements 4.1, 4.4-4.7, 17.1, 17.4, 19.7-19.10).

One :class:`ClaimDependencyBundle` gathers everything a single
:class:`~pursuit_evasion_rl.research.domain.ClaimRecord` rests on -- its
sealed protocol and analysis lineage, its selection contexts, its evidence,
its citations, the sealed run manifests its numbers came from, its paired
statistics, and its condition execution ledger.  :func:`inspect_claim_eligibility`
runs *every* dependency check and returns one :class:`ClaimGateReport` naming
each failure, so a single missing dependency can never hide another
(Requirement 19.7, 19.9).

None of the checks are new law.  Each one delegates to the module that already
owns that rule -- :func:`~pursuit_evasion_rl.research.protocol.inspect_confirmatory_eligibility`
for seal/classification/leakage, :mod:`~pursuit_evasion_rl.research.paper.citations`
for verification and body reconciliation,
:mod:`~pursuit_evasion_rl.research.paper.scope` for the field-outcome and
future-work bans -- and the gate composes their verdicts.  Claims themselves
are only ever accepted as real ``ClaimRecord`` values, so the construction-time
bans on priority language (Requirement 4.7) and on a supported
``Field_Readiness_Claim`` (Requirement 17.1) are exercised rather than bypassed.

Typical wiring::

    bundle = ClaimDependencyBundle(
        claim=claim,                              # a real ClaimRecord
        protocol=protocol_store.read(protocol_id),
        analysis_event=protocol_store.register_analysis(protocol_id, analysis),
        selection_contexts=(tuning_context,),
        evidence=(observation_record,),
        citation_ledger=ledger,
        body_citation_keys=tuple(ledger.citation_keys()),
        run_manifests=(sealed_manifest,),
        source_hashes=reconciled_source_hashes((sealed_manifest,)),
        statistics=paired_result,
        asserts_superiority=True,
        execution=execution_ledger,
        condition_ids=("cooperative_containment",),
        future_work=future_work_registry,
    )
    report = require_claim_eligibility(bundle)

``Field_Readiness`` is handled by :func:`_gate_status`, which pins the affirmed
status to ``unsupported`` before reading anything else.  The distinction that
matters is narrow: a claim that transparently *reports* field readiness as
unsupported is exportable, while any attempt to make it assert something
stronger is refused (Requirement 17.1, 19.10).  Because the report records the
verdict and never the measurement, two bundles differing only in their observed
performance produce byte-identical reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ..domain import (
    ClaimRecord,
    ClaimScope,
    ClaimStatus,
    EvidenceRecord,
    EvidenceType,
    ExecutionStatus,
    PersistedModel,
    exactly_one,
)
from ..errors import ErrorRecord, ResearchValidationError
from ..execution import FULL_POPULATION, ConditionExecutionLedger, PopulationView
from ..paper.citations import CitationLedger, reconcile_body_citations
from ..paper.scope import (
    FutureWorkRegistry,
    assert_no_field_readiness_derivation,
    assert_no_future_work_dependency,
)
from ..protocol import (
    AnalysisLineageEvent,
    ContaminationRecord,
    ProtocolRecord,
    SelectionContext,
    inspect_confirmatory_eligibility,
)
from ..runs.manifest import RunManifest
from ..statistics.paired import PairedComparisonResult

#: Scopes whose claims assert a positive finding, and therefore owe the full
#: supporting-evidence battery of Requirement 19.7.  ``field_readiness`` is
#: absent by design: it asserts nothing, so it has nothing to support.
ASSERTING_SCOPES = frozenset(
    {ClaimScope.IMPLEMENTATION, ClaimScope.RESULT, ClaimScope.NOVELTY}
)

#: Which Evidence_Types can carry which claim scope (Requirement 4.1, 19.7).
#: A field-readiness claim is unsupported by construction, so no evidence type
#: qualifies it either way.
CLAIM_SCOPE_EVIDENCE_TYPES: Mapping[ClaimScope, frozenset[EvidenceType]] = {
    ClaimScope.IMPLEMENTATION: frozenset(
        {EvidenceType.IMPLEMENTATION_EXISTENCE, EvidenceType.TEST_RESULT}
    ),
    ClaimScope.RESULT: frozenset(
        {EvidenceType.EXPERIMENT_OBSERVATION, EvidenceType.TEST_RESULT}
    ),
    ClaimScope.NOVELTY: frozenset(
        {
            EvidenceType.EXPERIMENT_OBSERVATION,
            EvidenceType.IMPLEMENTATION_EXISTENCE,
            EvidenceType.PAPER_CLAIM,
        }
    ),
    ClaimScope.FIELD_READINESS: frozenset(),
}


def _fail(code: str, message: str, **kwargs: Any) -> None:
    raise ResearchValidationError(code, message, **kwargs)


def _error(code: str, message: str, **values: Any) -> ErrorRecord:
    return ErrorRecord(code=code, message=message, **values)


def reconciled_source_hashes(manifests: Iterable[RunManifest]) -> dict[str, str]:
    """The ``{run_id: Content_Hash}`` map a claim cites as the source of its numbers.

    Building the map from the manifests themselves is the honest case; the gate
    exists for the dishonest one, where a reported number is attributed to a run
    whose sealed content no longer hashes to what the manuscript claims.
    """
    return {manifest.run_id: str(manifest.run.content_hash) for manifest in manifests}


@dataclass(frozen=True, slots=True, kw_only=True)
class ClaimDependencyBundle:
    """Every mandatory dependency of one claim, gathered for a single gate pass."""

    claim: ClaimRecord
    protocol: ProtocolRecord
    analysis_event: AnalysisLineageEvent | None = None
    selection_contexts: tuple[SelectionContext, ...] = ()
    contamination: tuple[ContaminationRecord, ...] = ()
    evidence: tuple[EvidenceRecord, ...] = ()
    citation_ledger: CitationLedger | None = None
    body_citation_keys: tuple[str, ...] = ()
    run_manifests: tuple[RunManifest, ...] = ()
    source_hashes: Mapping[str, str] = field(default_factory=dict)
    statistics: PairedComparisonResult | None = None
    asserts_superiority: bool = False
    execution: ConditionExecutionLedger | None = None
    condition_ids: tuple[str, ...] = ()
    population_view: PopulationView | None = None
    future_work: FutureWorkRegistry | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.claim, ClaimRecord):
            _fail("INVALID_CLAIM_RECORD", "a domain ClaimRecord is required", path="claim")
        if not isinstance(self.protocol, ProtocolRecord):
            _fail("INVALID_PROTOCOL_RECORD", "a ProtocolRecord is required", path="protocol")
        for name in ("selection_contexts", "contamination", "evidence", "run_manifests"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ("body_citation_keys", "condition_ids"):
            object.__setattr__(self, name, tuple(str(item) for item in getattr(self, name)))
        object.__setattr__(self, "source_hashes", dict(self.source_hashes))
        if not isinstance(self.asserts_superiority, bool):
            _fail(
                "INVALID_CLAIM_ASSERTION",
                "asserts_superiority must be a boolean",
                path="asserts_superiority",
                actual=self.asserts_superiority,
            )

    @property
    def asserts_findings(self) -> bool:
        return self.claim.scope in ASSERTING_SCOPES


@dataclass(frozen=True, slots=True)
class ClaimGateReport(PersistedModel):
    """Whether one claim may enter a confirmatory Paper_Artifact_Set.

    ``status`` is the status the gate affirms, which for a ``field_readiness``
    claim is ``unsupported`` whatever its evidence reports.  No measurement
    reaches this record, so its content hash is a function of the claim and its
    failures alone.
    """

    claim_id: str
    claim_hash: str
    scope: ClaimScope
    status: ClaimStatus
    eligible: bool
    errors: tuple[ErrorRecord, ...] = ()

    def __post_init__(self) -> None:
        for name in ("claim_id", "claim_hash"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                _fail(
                    "MISSING_REQUIRED_FIELD",
                    f"{name} must be a non-empty string",
                    path=name,
                    actual=value,
                )
        object.__setattr__(self, "scope", exactly_one(self.scope, ClaimScope, path="scope"))
        object.__setattr__(self, "status", exactly_one(self.status, ClaimStatus, path="status"))
        errors = tuple(self.errors)
        if self.eligible == bool(errors):
            _fail(
                "INVALID_GATE_REPORT",
                "eligible reports carry no errors and rejected reports carry errors",
                path="errors",
            )
        object.__setattr__(self, "errors", errors)
        super(ClaimGateReport, self).__post_init__()

    @property
    def error_codes(self) -> tuple[str, ...]:
        return tuple(error.code for error in self.errors)


class ClaimGateError(ResearchValidationError):
    """Raised by :func:`require_claim_eligibility`, carrying the preserved report."""

    def __init__(self, report: ClaimGateReport) -> None:
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


def _gate_status(claim: ClaimRecord) -> ClaimStatus:
    """The status the gate affirms, pinned for field readiness (Requirement 17.1, 19.10).

    Performance is not a parameter here, which is the whole point: no capture
    rate, however high, can move a ``Field_Readiness_Claim`` off ``unsupported``.
    """
    if claim.scope is ClaimScope.FIELD_READINESS:
        return ClaimStatus.UNSUPPORTED
    return claim.status


def _check_evidence(bundle: ClaimDependencyBundle) -> tuple[ErrorRecord, ...]:
    """Requirement 4.1, 19.7: every claim resolves to Evidence_Types its scope allows."""
    claim = bundle.claim
    if not bundle.asserts_findings:
        return ()
    errors: list[ErrorRecord] = []
    if not claim.evidence_ids:
        errors.append(
            _error(
                "MISSING_EVIDENCE_RECORD",
                "an asserting claim requires at least one Evidence_Record",
                path="claim.evidence_ids",
                actual=claim.claim_id,
            )
        )
    allowed = CLAIM_SCOPE_EVIDENCE_TYPES[claim.scope]
    resolved = {record.record_id: record for record in bundle.evidence}
    for identifier in claim.evidence_ids:
        record = resolved.get(identifier)
        if record is None:
            errors.append(
                _error(
                    "UNRESOLVED_EVIDENCE_REFERENCE",
                    "a claim references an Evidence_Record absent from its bundle",
                    path="claim.evidence_ids",
                    expected=sorted(resolved),
                    actual=identifier,
                )
            )
        elif record.evidence_type not in allowed:
            errors.append(
                _error(
                    "EVIDENCE_TYPE_SCOPE_MISMATCH",
                    "the Evidence_Type cannot carry a claim of this scope",
                    path="evidence.evidence_type",
                    expected=sorted(item.value for item in allowed),
                    actual=record.evidence_type.value,
                    details={"record_id": record.record_id, "scope": claim.scope.value},
                )
            )
    return tuple(errors)


def _check_protocol_and_classification(
    bundle: ClaimDependencyBundle,
) -> tuple[ErrorRecord, ...]:
    """Requirement 19.7: seal, Analysis_Classification and split leakage, via protocol.py."""
    if not bundle.asserts_findings:
        return ()
    report = inspect_confirmatory_eligibility(
        bundle.protocol,
        analysis_event=bundle.analysis_event,
        selection_contexts=bundle.selection_contexts,
        contamination=bundle.contamination,
    )
    return report.errors


def _check_citations(bundle: ClaimDependencyBundle) -> tuple[ErrorRecord, ...]:
    """Requirement 19.7: verified citations and a body-reference ledger with zero orphans."""
    claim = bundle.claim
    if not bundle.asserts_findings:
        return ()
    ledger = bundle.citation_ledger
    if ledger is None:
        return (
            _error(
                "MISSING_CITATION_LEDGER",
                "claim citations cannot be verified without a citation ledger",
                path="citation_ledger",
                actual=claim.claim_id,
            ),
        )
    errors: list[ErrorRecord] = []
    registered = {record.citation_key: record for record in ledger.citations()}
    for key in claim.citation_ids:
        record = registered.get(key)
        if record is None:
            errors.append(
                _error(
                    "UNRESOLVED_CITATION_REFERENCE",
                    "a claim cites a key absent from the citation ledger",
                    path="claim.citation_ids",
                    expected=sorted(registered),
                    actual=key,
                )
            )
        elif not record.is_verified:
            errors.append(
                _error(
                    "UNVERIFIED_CITATION",
                    "a claim may only cite a verified citation",
                    path="citation.verification_status",
                    expected="verified",
                    actual=record.verification_status.value,
                    details={
                        "citation_key": key,
                        "unverified_reasons": list(record.unverified_reasons),
                    },
                )
            )
    reconciliation = reconcile_body_citations(ledger, bundle.body_citation_keys)
    if reconciliation.missing_keys:
        errors.append(
            _error(
                "BODY_REFERENCE_MISSING",
                "a body reference resolves to no ledger entry",
                path="body_citation_keys",
                expected=[],
                actual=list(reconciliation.missing_keys),
            )
        )
    if reconciliation.orphan_keys:
        errors.append(
            _error(
                "BODY_REFERENCE_ORPHAN",
                "a ledger entry is never referenced from the body",
                path="citation_ledger",
                expected=[],
                actual=list(reconciliation.orphan_keys),
            )
        )
    return tuple(errors)


def _check_manifests(bundle: ClaimDependencyBundle) -> tuple[ErrorRecord, ...]:
    """Requirement 19.7-19.8: each cited number resolves to a sealed manifest of that hash."""
    if not bundle.asserts_findings:
        return ()
    if not bundle.source_hashes:
        return (
            _error(
                "MISSING_SOURCE_HASH_RECONCILIATION",
                "an asserting claim must attribute its numbers to at least one sealed run",
                path="source_hashes",
                actual=bundle.claim.claim_id,
            ),
        )
    errors: list[ErrorRecord] = []
    manifests = {manifest.run_id: manifest for manifest in bundle.run_manifests}
    for run_id, expected in sorted(bundle.source_hashes.items()):
        manifest = manifests.get(run_id)
        if manifest is None:
            errors.append(
                _error(
                    "UNRESOLVED_RUN_MANIFEST",
                    "a cited source run has no manifest in the bundle",
                    path="source_hashes",
                    expected=sorted(manifests),
                    actual=run_id,
                )
            )
        elif not manifest.is_sealed:
            errors.append(
                _error(
                    "RUN_MANIFEST_NOT_SEALED",
                    "a claim may only cite a sealed run manifest",
                    path="run_manifests",
                    expected="sealed",
                    actual=run_id,
                )
            )
        elif str(manifest.run.content_hash) != expected:
            errors.append(
                _error(
                    "SOURCE_HASH_MISMATCH",
                    "the cited source hash does not match the sealed run's content",
                    path="source_hashes",
                    expected=expected,
                    actual=str(manifest.run.content_hash),
                    details={"run_id": run_id},
                )
            )
    return tuple(errors)


def _check_statistics(bundle: ClaimDependencyBundle) -> tuple[ErrorRecord, ...]:
    """Requirement 4.5, 19.7: corrected, confirmatory-eligible statistics behind the claim."""
    claim = bundle.claim
    if not bundle.asserts_findings:
        return ()
    result = bundle.statistics
    if result is None:
        if claim.scope is ClaimScope.RESULT:
            return (
                _error(
                    "MISSING_STATISTICAL_BACKING",
                    "a result claim requires a paired comparison behind it",
                    path="statistics",
                    actual=claim.claim_id,
                ),
            )
        return ()
    errors: list[ErrorRecord] = []
    if not result.confirmatory_eligible:
        errors.append(
            _error(
                "STATISTICS_NOT_CONFIRMATORY_ELIGIBLE",
                "the paired comparison did not reach the pre-registered confirmatory sample size",
                path="statistics.confirmatory_eligible",
                expected=True,
                actual=False,
                details={"comparison_id": result.comparison_id, "result_hash": result.result_hash},
            )
        )
    if bundle.asserts_superiority:
        if result.gate is None:
            errors.append(
                _error(
                    "MISSING_PRACTICAL_GATE",
                    "a superiority claim requires a Practical_Threshold decision",
                    path="statistics.gate",
                    actual=result.comparison_id,
                )
            )
        elif not result.gate.superiority:
            errors.append(
                _error(
                    "PRACTICAL_GATE_NOT_PASSED",
                    "the comparison did not meet both statistical and practical significance",
                    path="statistics.gate.superiority",
                    expected=True,
                    actual=False,
                    details={
                        "comparison_id": result.comparison_id,
                        "reasons": list(result.gate.reasons),
                    },
                )
            )
    return tuple(errors)


def _check_execution(bundle: ClaimDependencyBundle) -> tuple[ErrorRecord, ...]:
    """Requirement 4.6-4.7, 19.7-19.8: a valid, complete, full-population Execution_Status.

    Favourability is deliberately not read.  A completed Condition whose measured
    outcome is null or unfavourable stays completed and stays claimable; what is
    refused is a Condition that never produced a measurement, one running below
    the confirmatory sample size, and a success-only aggregate standing in for
    the planned population.
    """
    if not bundle.asserts_findings:
        return ()
    ledger = bundle.execution
    if ledger is None:
        return (
            _error(
                "MISSING_EXECUTION_LEDGER",
                "an asserting claim requires the Condition execution ledger behind it",
                path="execution",
                actual=bundle.claim.claim_id,
            ),
        )
    errors: list[ErrorRecord] = []
    if not bundle.condition_ids:
        errors.append(
            _error(
                "MISSING_CONDITION_REFERENCE",
                "an asserting claim must name the Conditions it rests on",
                path="condition_ids",
                expected=list(ledger.planned_condition_ids),
                actual=[],
            )
        )
    for condition_id in bundle.condition_ids:
        try:
            record = ledger.record_for(condition_id)
        except ResearchValidationError as exc:
            errors.append(exc.record)
            continue
        if record.execution_status is not ExecutionStatus.COMPLETED:
            errors.append(
                _error(
                    "CONDITION_NOT_COMPLETED",
                    "a claim cannot rest on a Condition that produced no measurement",
                    path="execution.execution_status",
                    expected=ExecutionStatus.COMPLETED.value,
                    actual=record.execution_status.value,
                    details={
                        "condition_id": condition_id,
                        "status_reason": record.status_reason,
                    },
                )
            )
        if not record.is_confirmatory_eligible:
            errors.append(
                _error(
                    "EXPLORATORY_CONDITION_SAMPLE_SIZE",
                    "the Condition ran below the confirmatory sample size",
                    path="execution.sample_size_status",
                    actual=record.sample_size_status.value,
                    details={"condition_id": condition_id},
                )
            )
    view = bundle.population_view
    if view is not None and view.coverage != FULL_POPULATION:
        errors.append(
            _error(
                "SUCCESS_ONLY_POPULATION_VIEW",
                "a success-only aggregate cannot stand in for the planned population",
                path="population_view.coverage",
                expected=FULL_POPULATION,
                actual=view.coverage,
            )
        )
    return tuple(errors)


def _check_future_work(bundle: ClaimDependencyBundle) -> tuple[ErrorRecord, ...]:
    """Requirement 18.5, 18.8: no core claim rests on a labelled future-work demo."""
    claim = bundle.claim
    registry = bundle.future_work
    if registry is None:
        if claim.scope is ClaimScope.NOVELTY:
            return (
                _error(
                    "MISSING_FUTURE_WORK_REGISTRY",
                    "novelty evidence must be screened against the future-work registry",
                    path="future_work",
                    actual=claim.claim_id,
                ),
            )
        return ()
    try:
        assert_no_future_work_dependency(claim, registry)
    except ResearchValidationError as exc:
        return (exc.record,)
    return ()


def _check_field_readiness(bundle: ClaimDependencyBundle) -> tuple[ErrorRecord, ...]:
    """Requirement 17.1, 17.4, 19.10: no field-outcome derivation, no stronger assertion.

    The language ban applies to every scope, because Requirement 17.4 is about
    what a *result* may be restated as, not only about claims already labelled
    field readiness.
    """
    claim = bundle.claim
    errors: list[ErrorRecord] = []
    try:
        assert_no_field_readiness_derivation(claim.text, path="claim.text")
    except ResearchValidationError as exc:
        errors.append(exc.record)
    if claim.scope is ClaimScope.FIELD_READINESS and bundle.asserts_superiority:
        errors.append(
            _error(
                "FIELD_READINESS_ASSERTION_BLOCKED",
                "a Field_Readiness_Claim stays unsupported and can never assert superiority",
                path="asserts_superiority",
                expected=False,
                actual=True,
                details={"claim_id": claim.claim_id, "status": ClaimStatus.UNSUPPORTED.value},
            )
        )
    return tuple(errors)


#: Every mandatory dependency check, run unconditionally so that one failure
#: never conceals another (Requirement 19.7, 19.9).
_CHECKS = (
    _check_evidence,
    _check_protocol_and_classification,
    _check_citations,
    _check_manifests,
    _check_statistics,
    _check_execution,
    _check_future_work,
    _check_field_readiness,
)


def inspect_claim_eligibility(bundle: ClaimDependencyBundle) -> ClaimGateReport:
    """Run every mandatory dependency check and preserve all failures at once."""
    if not isinstance(bundle, ClaimDependencyBundle):
        _fail("INVALID_CLAIM_BUNDLE", "a ClaimDependencyBundle is required", path="bundle")
    errors: list[ErrorRecord] = []
    for check in _CHECKS:
        errors.extend(check(bundle))
    return ClaimGateReport(
        claim_id=bundle.claim.claim_id,
        claim_hash=str(bundle.claim.content_hash),
        scope=bundle.claim.scope,
        status=_gate_status(bundle.claim),
        eligible=not errors,
        errors=tuple(errors),
    )


def require_claim_eligibility(bundle: ClaimDependencyBundle) -> ClaimGateReport:
    """Return the report, or raise :class:`ClaimGateError` carrying it (Requirement 19.9)."""
    report = inspect_claim_eligibility(bundle)
    if not report.eligible:
        raise ClaimGateError(report)
    return report


__all__ = (
    "ASSERTING_SCOPES",
    "CLAIM_SCOPE_EVIDENCE_TYPES",
    "ClaimDependencyBundle",
    "ClaimGateError",
    "ClaimGateReport",
    "inspect_claim_eligibility",
    "reconciled_source_hashes",
    "require_claim_eligibility",
)

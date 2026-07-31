"""Search protocol, citation ledger, and verification gates for related work."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import re
from typing import Iterable, Mapping, Sequence

from ..domain import PersistedModel, ScreeningDecision, exactly_one
from ..errors import ResearchValidationError

MIN_INDEPENDENT_SOURCES = 3
MIN_ACADEMIC_INDEX_SOURCES = 2

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _required(value: str, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResearchValidationError(
            "MISSING_REQUIRED_FIELD", f"{path} must be non-empty", path=path
        )
    return value


def _sha256(value: str, path: str) -> str:
    _required(value, path)
    if _SHA256.fullmatch(value) is None:
        raise ResearchValidationError(
            "INVALID_CONTENT_HASH",
            f"{path} must be a lowercase SHA-256",
            path=path,
            expected="64 lowercase hexadecimal characters",
            actual=value,
        )
    return value


def _utc(value: str, path: str) -> str:
    _required(value, path)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ResearchValidationError(
            "INVALID_UTC_TIMESTAMP",
            f"{path} must be an ISO-8601 UTC timestamp",
            path=path,
            actual=value,
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ResearchValidationError(
            "INVALID_UTC_TIMESTAMP",
            f"{path} must include the UTC offset",
            path=path,
            actual=value,
        )
    return value


def _texts(values: Iterable[str], path: str) -> tuple[str, ...]:
    normalized = tuple(str(item).strip() for item in values)
    if not normalized or any(not item for item in normalized):
        raise ResearchValidationError(
            "MISSING_REQUIRED_FIELD", f"{path} must contain non-empty entries", path=path
        )
    if len(set(normalized)) != len(normalized):
        raise ResearchValidationError(
            "DUPLICATE_IDENTIFIER", f"{path} must not repeat entries", path=path
        )
    return normalized


class SearchSourceKind(str, Enum):
    """How a search source is operated, so academic indices can be counted."""

    ACADEMIC_INDEX = "academic_index"
    PREPRINT_SERVER = "preprint_server"
    PUBLISHER_PORTAL = "publisher_portal"
    INSTITUTIONAL_REPOSITORY = "institutional_repository"
    GENERAL_WEB = "general_web"


class PersistentIdentifierKind(str, Enum):
    DOI = "doi"
    HANDLE = "handle"
    ARXIV = "arxiv"
    PUBLISHER_PERMANENT_URL = "publisher_permanent_url"


class ProposalCrossCheck(str, Enum):
    """State of the comparison against the original proposal document's own text."""

    NOT_FROM_PROPOSAL = "not_from_proposal"
    PROPOSAL_TEXT_VERIFIED = "proposal_text_verified"
    PROPOSAL_TEXT_UNVERIFIED = "proposal_text_unverified"


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"


class ScreeningState(str, Enum):
    """Resolved screening state of one candidate, including malformed states."""

    INCLUDED = "included"
    EXCLUDED = "excluded"
    MISSING = "missing"
    DUPLICATE = "duplicate"
    INCONSISTENT = "inconsistent"


class NoveltyEvidenceExclusion(str, Enum):
    """Why a citation may not back a Novelty_Claim."""

    SCREENING_MISSING = "screening_missing"
    SCREENING_DUPLICATE = "screening_duplicate"
    SCREENING_INCONSISTENT = "screening_inconsistent"
    SCREENED_EXCLUDED = "screened_excluded"
    UNVERIFIED = "unverified"


_SCREENING_EXCLUSIONS = {
    ScreeningState.MISSING: NoveltyEvidenceExclusion.SCREENING_MISSING,
    ScreeningState.DUPLICATE: NoveltyEvidenceExclusion.SCREENING_DUPLICATE,
    ScreeningState.INCONSISTENT: NoveltyEvidenceExclusion.SCREENING_INCONSISTENT,
    ScreeningState.EXCLUDED: NoveltyEvidenceExclusion.SCREENED_EXCLUDED,
}


@dataclass(frozen=True, slots=True)
class SearchSource(PersistedModel):
    """One search source; `operator` carries the mutual-independence identity."""

    source_id: str
    name: str
    kind: SearchSourceKind
    operator: str
    endpoint: str

    def __post_init__(self) -> None:
        for name in ("source_id", "name", "operator", "endpoint"):
            _required(getattr(self, name), name)
        object.__setattr__(self, "kind", exactly_one(self.kind, SearchSourceKind, path="kind"))
        super(SearchSource, self).__post_init__()


@dataclass(frozen=True, slots=True)
class SearchProtocolSpec(PersistedModel):
    """Frozen search plan recorded before the first confirmatory result is read."""

    search_date_utc: str
    queries: tuple[str, ...]
    sources: tuple[SearchSource, ...]
    period_start: str
    period_end: str
    languages: tuple[str, ...]
    inclusion_criteria: tuple[str, ...]
    exclusion_criteria: tuple[str, ...]

    def __post_init__(self) -> None:
        _utc(self.search_date_utc, "search_date_utc")
        for name in ("queries", "languages", "inclusion_criteria", "exclusion_criteria"):
            object.__setattr__(self, name, _texts(getattr(self, name), name))
        _required(self.period_start, "period_start")
        _required(self.period_end, "period_end")
        if self.period_start > self.period_end:
            raise ResearchValidationError(
                "INVALID_SEARCH_PERIOD",
                "period_start must not follow period_end",
                path="period_start",
                expected=f"<= {self.period_end}",
                actual=self.period_start,
            )
        object.__setattr__(self, "sources", tuple(self.sources))
        source_ids = [item.source_id for item in self.sources]
        if len(set(source_ids)) != len(source_ids):
            raise ResearchValidationError(
                "DUPLICATE_IDENTIFIER", "source identifiers must be unique", path="sources"
            )
        operators = {item.operator for item in self.sources}
        if len(operators) < MIN_INDEPENDENT_SOURCES:
            raise ResearchValidationError(
                "INSUFFICIENT_SEARCH_SOURCES",
                "related-work search requires mutually independent sources",
                path="sources",
                expected=MIN_INDEPENDENT_SOURCES,
                actual=len(operators),
            )
        academic = {
            item.operator for item in self.sources if item.kind is SearchSourceKind.ACADEMIC_INDEX
        }
        if len(academic) < MIN_ACADEMIC_INDEX_SOURCES:
            raise ResearchValidationError(
                "INSUFFICIENT_ACADEMIC_INDEX_SOURCES",
                "related-work search requires independent academic indices",
                path="sources",
                expected=MIN_ACADEMIC_INDEX_SOURCES,
                actual=len(academic),
            )
        super(SearchProtocolSpec, self).__post_init__()

    def source_ids(self) -> frozenset[str]:
        return frozenset(item.source_id for item in self.sources)


@dataclass(frozen=True, slots=True)
class CitationCandidate(PersistedModel):
    """One raw search hit, recorded before deduplication and screening."""

    candidate_id: str
    canonical_work_id: str
    source_id: str
    query: str
    executed_at_utc: str
    result_rank: int
    record_identifier: str
    result_content_hash: str

    def __post_init__(self) -> None:
        for name in (
            "candidate_id",
            "canonical_work_id",
            "source_id",
            "query",
            "record_identifier",
        ):
            _required(getattr(self, name), name)
        _utc(self.executed_at_utc, "executed_at_utc")
        _sha256(self.result_content_hash, "result_content_hash")
        if not isinstance(self.result_rank, int) or isinstance(self.result_rank, bool):
            raise ResearchValidationError(
                "INVALID_RESULT_RANK", "result_rank must be an integer", path="result_rank"
            )
        if self.result_rank < 1:
            raise ResearchValidationError(
                "INVALID_RESULT_RANK",
                "result_rank must be a 1-based search position",
                path="result_rank",
                expected=">= 1",
                actual=self.result_rank,
            )
        super(CitationCandidate, self).__post_init__()


@dataclass(frozen=True, slots=True)
class ScreeningRecord(PersistedModel):
    candidate_id: str
    decision: ScreeningDecision
    reason: str
    screened_at_utc: str

    def __post_init__(self) -> None:
        _required(self.candidate_id, "candidate_id")
        _required(self.reason, "reason")
        _utc(self.screened_at_utc, "screened_at_utc")
        object.__setattr__(
            self, "decision", exactly_one(self.decision, ScreeningDecision, path="decision")
        )
        super(ScreeningRecord, self).__post_init__()


@dataclass(frozen=True, slots=True)
class MetadataCrossCheck(PersistedModel):
    """Official-metadata comparison against author, year, and title."""

    identifier: str
    identifier_kind: PersistentIdentifierKind
    metadata_source: str
    checked_at_utc: str
    author_match: bool
    year_match: bool
    title_match: bool

    def __post_init__(self) -> None:
        _required(self.identifier, "identifier")
        _required(self.metadata_source, "metadata_source")
        _utc(self.checked_at_utc, "checked_at_utc")
        object.__setattr__(
            self,
            "identifier_kind",
            exactly_one(self.identifier_kind, PersistentIdentifierKind, path="identifier_kind"),
        )
        for name in ("author_match", "year_match", "title_match"):
            if not isinstance(getattr(self, name), bool):
                raise ResearchValidationError(
                    "INVALID_MATCH_FLAG", f"{name} must be a boolean", path=name
                )
        super(MetadataCrossCheck, self).__post_init__()

    @property
    def passed(self) -> bool:
        return self.author_match and self.year_match and self.title_match


@dataclass(frozen=True, slots=True)
class SourceLocation(PersistedModel):
    """Where in the original text a cited fact was read, plus that text's hash."""

    original_text_content_hash: str
    page: str | None = None
    section: str | None = None
    table: str | None = None

    def __post_init__(self) -> None:
        _sha256(self.original_text_content_hash, "original_text_content_hash")
        if not any(
            isinstance(getattr(self, name), str) and getattr(self, name).strip()
            for name in ("page", "section", "table")
        ):
            raise ResearchValidationError(
                "MISSING_SOURCE_LOCATION",
                "source location requires a page, section, or table",
                path="page",
            )
        super(SourceLocation, self).__post_init__()


@dataclass(frozen=True, slots=True)
class CitationRecord(PersistedModel):
    """A canonical work after screening, with a derived verification status."""

    citation_key: str
    canonical_work_id: str
    title: str
    authors: tuple[str, ...]
    year: int
    candidate_ids: tuple[str, ...]
    proposal_cross_check: ProposalCrossCheck
    metadata_cross_check: MetadataCrossCheck | None = None
    source_location: SourceLocation | None = None
    proposal_reference_label: str | None = None
    verification_status: VerificationStatus = field(init=False, default=VerificationStatus.UNVERIFIED)
    unverified_reasons: tuple[str, ...] = field(init=False, default=())

    def __post_init__(self) -> None:
        for name in ("citation_key", "canonical_work_id", "title"):
            _required(getattr(self, name), name)
        object.__setattr__(self, "authors", _texts(self.authors, "authors"))
        object.__setattr__(self, "candidate_ids", _texts(self.candidate_ids, "candidate_ids"))
        if not isinstance(self.year, int) or isinstance(self.year, bool):
            raise ResearchValidationError("INVALID_YEAR", "year must be an integer", path="year")
        object.__setattr__(
            self,
            "proposal_cross_check",
            exactly_one(self.proposal_cross_check, ProposalCrossCheck, path="proposal_cross_check"),
        )
        from_proposal = self.proposal_cross_check is not ProposalCrossCheck.NOT_FROM_PROPOSAL
        if from_proposal:
            _required(self.proposal_reference_label or "", "proposal_reference_label")
        elif self.proposal_reference_label is not None:
            raise ResearchValidationError(
                "PROVENANCE_MISMATCH",
                "proposal_reference_label requires a proposal cross-check state",
                path="proposal_reference_label",
                actual=self.proposal_reference_label,
            )

        reasons: list[str] = []
        if self.metadata_cross_check is None:
            reasons.append("metadata_cross_check_missing")
        elif not self.metadata_cross_check.passed:
            reasons.append("metadata_cross_check_failed")
        if self.source_location is None:
            reasons.append("source_location_missing")
        if self.proposal_cross_check is ProposalCrossCheck.PROPOSAL_TEXT_UNVERIFIED:
            reasons.append("proposal_text_unverified")
        object.__setattr__(self, "unverified_reasons", tuple(reasons))
        object.__setattr__(
            self,
            "verification_status",
            VerificationStatus.UNVERIFIED if reasons else VerificationStatus.VERIFIED,
        )
        super(CitationRecord, self).__post_init__()

    @property
    def is_verified(self) -> bool:
        return self.verification_status is VerificationStatus.VERIFIED


@dataclass(frozen=True, slots=True)
class BodyReconciliation(PersistedModel):
    """Missing and orphan citation keys between a manuscript body and the ledger."""

    missing_keys: tuple[str, ...]
    orphan_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("missing_keys", "orphan_keys"):
            object.__setattr__(self, name, tuple(sorted({str(item) for item in getattr(self, name)})))
        super(BodyReconciliation, self).__post_init__()

    @property
    def missing_count(self) -> int:
        return len(self.missing_keys)

    @property
    def orphan_count(self) -> int:
        return len(self.orphan_keys)

    @property
    def is_clean(self) -> bool:
        return not self.missing_keys and not self.orphan_keys


class CitationLedger:
    """Fail-closed ledger from search hits to verified, novelty-eligible citations."""

    def __init__(self, protocol: SearchProtocolSpec) -> None:
        if not isinstance(protocol, SearchProtocolSpec):
            raise ResearchValidationError(
                "MISSING_SEARCH_PROTOCOL",
                "a citation ledger requires a frozen search protocol",
                path="protocol",
            )
        self._protocol = protocol
        self._candidates: dict[str, CitationCandidate] = {}
        self._canonical: dict[str, list[str]] = {}
        self._screenings: dict[str, list[ScreeningRecord]] = {}
        self._citations: dict[str, CitationRecord] = {}

    @property
    def protocol(self) -> SearchProtocolSpec:
        return self._protocol

    def add_candidate(self, candidate: CitationCandidate) -> CitationCandidate:
        if candidate.candidate_id in self._candidates:
            raise ResearchValidationError(
                "DUPLICATE_IDENTIFIER",
                "candidate identifiers must be unique",
                path="candidate_id",
                actual=candidate.candidate_id,
            )
        if candidate.source_id not in self._protocol.source_ids():
            raise ResearchValidationError(
                "UNKNOWN_SEARCH_SOURCE",
                "candidate source is absent from the search protocol",
                path="source_id",
                expected=sorted(self._protocol.source_ids()),
                actual=candidate.source_id,
            )
        if candidate.query not in self._protocol.queries:
            raise ResearchValidationError(
                "UNKNOWN_SEARCH_QUERY",
                "candidate query is absent from the search protocol",
                path="query",
                expected=list(self._protocol.queries),
                actual=candidate.query,
            )
        self._candidates[candidate.candidate_id] = candidate
        self._canonical.setdefault(candidate.canonical_work_id, []).append(candidate.candidate_id)
        return candidate

    def candidates(self) -> tuple[CitationCandidate, ...]:
        return tuple(self._candidates.values())

    def candidate_ids_for_work(self, canonical_work_id: str) -> tuple[str, ...]:
        return tuple(self._canonical.get(canonical_work_id, ()))

    def duplicate_links(self) -> Mapping[str, tuple[str, ...]]:
        """Canonical work identifiers that more than one candidate record resolved to."""
        return {
            work_id: tuple(members)
            for work_id, members in self._canonical.items()
            if len(members) > 1
        }

    def screen(
        self,
        candidate_id: str,
        decision: ScreeningDecision | str,
        reason: str,
        *,
        screened_at_utc: str,
    ) -> ScreeningRecord:
        """Record exactly one decision plus a reason; repeat calls are kept as conflicts."""
        if candidate_id not in self._candidates:
            raise ResearchValidationError(
                "UNKNOWN_CANDIDATE",
                "screening requires a registered candidate",
                path="candidate_id",
                actual=candidate_id,
            )
        record = ScreeningRecord(
            candidate_id=candidate_id,
            decision=decision,
            reason=reason,
            screened_at_utc=screened_at_utc,
        )
        self._screenings.setdefault(candidate_id, []).append(record)
        return record

    def screening_state(self, candidate_id: str) -> ScreeningState:
        records = self._screenings.get(candidate_id, [])
        if not records:
            return ScreeningState.MISSING
        decisions = {item.decision for item in records}
        if len(decisions) > 1:
            return ScreeningState.INCONSISTENT
        if len(records) > 1:
            return ScreeningState.DUPLICATE
        return ScreeningState(records[0].decision.value)

    def register_citation(self, record: CitationRecord) -> CitationRecord:
        if record.citation_key in self._citations:
            raise ResearchValidationError(
                "DUPLICATE_IDENTIFIER",
                "citation keys must be unique",
                path="citation_key",
                actual=record.citation_key,
            )
        for candidate_id in record.candidate_ids:
            candidate = self._candidates.get(candidate_id)
            if candidate is None:
                raise ResearchValidationError(
                    "UNKNOWN_CANDIDATE",
                    "citation references an unregistered candidate",
                    path="candidate_ids",
                    actual=candidate_id,
                )
            if candidate.canonical_work_id != record.canonical_work_id:
                raise ResearchValidationError(
                    "PROVENANCE_MISMATCH",
                    "citation candidates must share one canonical work identifier",
                    path="canonical_work_id",
                    expected=record.canonical_work_id,
                    actual=candidate.canonical_work_id,
                )
        linked = set(self._canonical.get(record.canonical_work_id, ()))
        if linked != set(record.candidate_ids):
            raise ResearchValidationError(
                "UNLINKED_DUPLICATE_CANDIDATE",
                "citation must link every candidate of its canonical work",
                path="candidate_ids",
                expected=sorted(linked),
                actual=sorted(record.candidate_ids),
            )
        self._citations[record.citation_key] = record
        return record

    def citations(self) -> tuple[CitationRecord, ...]:
        return tuple(self._citations.values())

    def citation_keys(self) -> frozenset[str]:
        return frozenset(self._citations)

    def novelty_exclusion(self, citation_key: str) -> NoveltyEvidenceExclusion | None:
        """Return why a citation cannot back a Novelty_Claim, or None when eligible."""
        record = self._citations.get(citation_key)
        if record is None:
            raise ResearchValidationError(
                "UNKNOWN_CITATION",
                "citation is absent from the ledger",
                path="citation_key",
                actual=citation_key,
            )
        states = {self.screening_state(item) for item in record.candidate_ids}
        for state in (
            ScreeningState.INCONSISTENT,
            ScreeningState.MISSING,
            ScreeningState.DUPLICATE,
        ):
            if state in states:
                return _SCREENING_EXCLUSIONS[state]
        if states != {ScreeningState.INCLUDED}:
            return (
                NoveltyEvidenceExclusion.SCREENING_INCONSISTENT
                if len(states) > 1
                else NoveltyEvidenceExclusion.SCREENED_EXCLUDED
            )
        if not record.is_verified:
            return NoveltyEvidenceExclusion.UNVERIFIED
        return None

    def novelty_exclusions(self) -> Mapping[str, NoveltyEvidenceExclusion]:
        found = ((key, self.novelty_exclusion(key)) for key in self._citations)
        return {key: reason for key, reason in found if reason is not None}

    def novelty_eligible_citations(self) -> tuple[CitationRecord, ...]:
        return tuple(
            record
            for key, record in self._citations.items()
            if self.novelty_exclusion(key) is None
        )

    def is_novelty_eligible(self, citation_key: str) -> bool:
        return self.novelty_exclusion(citation_key) is None


def reconcile_body_citations(
    ledger: CitationLedger, body_citation_keys: Sequence[str] | Iterable[str]
) -> BodyReconciliation:
    """Compare in-text citation keys against the ledger for missing and orphan entries."""
    cited = {str(item).strip() for item in body_citation_keys}
    if any(not item for item in cited):
        raise ResearchValidationError(
            "MISSING_REQUIRED_FIELD",
            "body citation keys must be non-empty",
            path="body_citation_keys",
        )
    registered = set(ledger.citation_keys())
    return BodyReconciliation(
        missing_keys=tuple(cited - registered),
        orphan_keys=tuple(registered - cited),
    )


__all__ = (
    "MIN_ACADEMIC_INDEX_SOURCES",
    "MIN_INDEPENDENT_SOURCES",
    "BodyReconciliation",
    "CitationCandidate",
    "CitationLedger",
    "CitationRecord",
    "MetadataCrossCheck",
    "NoveltyEvidenceExclusion",
    "PersistentIdentifierKind",
    "ProposalCrossCheck",
    "ScreeningRecord",
    "ScreeningState",
    "SearchProtocolSpec",
    "SearchSource",
    "SearchSourceKind",
    "SourceLocation",
    "VerificationStatus",
    "reconcile_body_citations",
)

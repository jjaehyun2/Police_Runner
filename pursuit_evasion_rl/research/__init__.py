"""Fail-closed, content-addressed research domain foundation."""

from .audit import AuditReport, AssetAuditor, EvidenceRegistry, audit_repository
from .canonical import canonical_json, canonical_loads, content_hash, sha256_bytes
from .domain import (
    ClaimRecord,
    Condition,
    EpisodeOutcome,
    EvidenceRecord,
    MapRecord,
    ResearchProtocol,
    RunRecord,
    exactly_one,
)
from .errors import ErrorRecord, ResearchValidationError
from .migration import (
    LegacyCheckpointConversion,
    LegacyConfigConversion,
    assert_research_import_boundary,
    convert_legacy_checkpoint,
    convert_legacy_cli_arguments,
    dispatch_legacy_script,
    load_legacy_checkpoint,
)
from .preservation import (
    ArtifactMeasurement,
    PreservedArtifact,
    PreservationGateReport,
    ProtectedPathGuard,
    RunPaths,
    preserve_artifact,
    verify_before_run,
)

__all__ = (
    "ArtifactMeasurement",
    "AssetAuditor",
    "AuditReport",
    "EvidenceRegistry",
    "audit_repository",
    "ClaimRecord",
    "Condition",
    "EpisodeOutcome",
    "ErrorRecord",
    "EvidenceRecord",
    "LegacyCheckpointConversion",
    "LegacyConfigConversion",
    "MapRecord",
    "PreservedArtifact",
    "PreservationGateReport",
    "ProtectedPathGuard",
    "ResearchProtocol",
    "ResearchValidationError",
    "RunPaths",
    "RunRecord",
    "canonical_json",
    "canonical_loads",
    "content_hash",
    "convert_legacy_checkpoint",
    "convert_legacy_cli_arguments",
    "dispatch_legacy_script",
    "exactly_one",
    "assert_research_import_boundary",
    "load_legacy_checkpoint",
    "preserve_artifact",
    "sha256_bytes",
    "verify_before_run",
)

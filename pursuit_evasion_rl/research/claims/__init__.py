"""Fail-closed claim export gating."""

from .gate import (
    ASSERTING_SCOPES,
    CLAIM_SCOPE_EVIDENCE_TYPES,
    ClaimDependencyBundle,
    ClaimGateError,
    ClaimGateReport,
    inspect_claim_eligibility,
    reconciled_source_hashes,
    require_claim_eligibility,
)

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

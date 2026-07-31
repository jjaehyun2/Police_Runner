"""Evidence-backed claim registry with non-overstatement transition guards."""

from __future__ import annotations

from enum import Enum
from typing import TypeAlias

from .canonical import content_hash
from .experiments import (
    OSM_OBSERVATION_PROFILE,
    EvidenceState,
    ExperimentEvidence,
    ExperimentPlan,
    ImplementationEvidence,
)
from .models import ClaimRecord, ClaimStatus, DomainValidationError, RunMode


class ClaimKind(str, Enum):
    FEATURE = "feature"
    OSM_PERFORMANCE = "osm-performance"
    OSM_GENERALIZATION = "osm-generalization"
    SAFETY_EFFECT = "safety-effect"


Evidence: TypeAlias = ExperimentEvidence | ImplementationEvidence


class ClaimRegistry:
    """In-memory append-only registry for plans, evidence, and guarded claims."""

    def __init__(self) -> None:
        self._plans: dict[str, ExperimentPlan] = {}
        self._evidence: dict[str, Evidence] = {}
        self._claims: dict[str, ClaimRecord] = {}
        self._claim_kinds: dict[str, ClaimKind] = {}

    def register_plan(self, plan: ExperimentPlan) -> ExperimentPlan:
        existing = self._plans.get(plan.plan_id)
        if existing is not None and content_hash(existing) != content_hash(plan):
            raise DomainValidationError(
                "IMMUTABLE_PLAN_CONFLICT", f"Plan {plan.plan_id!r} is already frozen with different content"
            )
        self._plans[plan.plan_id] = plan
        return plan

    def register_evidence(self, evidence: Evidence) -> Evidence:
        existing = self._evidence.get(evidence.evidence_id)
        if existing is not None and content_hash(existing) != content_hash(evidence):
            raise DomainValidationError(
                "IMMUTABLE_EVIDENCE_CONFLICT",
                f"Evidence {evidence.evidence_id!r} is already registered with different content",
            )
        self._evidence[evidence.evidence_id] = evidence
        return evidence

    def register_claim(
        self,
        name: str,
        kind: ClaimKind,
        *,
        reason: str,
        updated_at: str,
        synthetic_grid_only: bool = False,
    ) -> ClaimRecord:
        if not name.strip() or not reason.strip() or not updated_at.strip():
            raise DomainValidationError("MISSING_CLAIM_FIELD", "Claim name, reason, and updated_at are required")
        if name in self._claims:
            if self._claim_kinds[name] != kind:
                raise DomainValidationError("IMMUTABLE_CLAIM_KIND", "A claim kind cannot be changed")
            return self._claims[name]
        status = (
            ClaimStatus.UNSUPPORTED
            if synthetic_grid_only and kind != ClaimKind.FEATURE
            else ClaimStatus.EXPERIMENTAL
        )
        record = ClaimRecord(name, status, (), reason, updated_at)
        self._claims[name] = record
        self._claim_kinds[name] = kind
        return record

    def get(self, name: str) -> ClaimRecord:
        try:
            return self._claims[name]
        except KeyError as exc:
            raise DomainValidationError("CLAIM_NOT_FOUND", f"Unknown claim {name!r}") from exc

    def transition(
        self,
        name: str,
        target: ClaimStatus,
        *,
        evidence_ids: tuple[str, ...] = (),
        reason: str,
        updated_at: str,
    ) -> ClaimRecord:
        current = self.get(name)
        kind = self._claim_kinds[name]
        if current.status == ClaimStatus.VERIFIED and target != ClaimStatus.VERIFIED:
            raise DomainValidationError("VERIFIED_CLAIM_DOWNGRADE", "Verified claims cannot be silently downgraded")
        if not reason.strip() or not updated_at.strip():
            raise DomainValidationError("MISSING_CLAIM_FIELD", "Transition reason and updated_at are required")
        evidence = self._resolve_evidence(evidence_ids)
        if target in {ClaimStatus.IMPLEMENTED, ClaimStatus.VERIFIED}:
            if kind == ClaimKind.FEATURE:
                self._guard_feature_evidence(evidence)
            elif target == ClaimStatus.IMPLEMENTED:
                raise DomainValidationError(
                    "INVALID_PERFORMANCE_STATUS", "OSM result claims are experimental, unsupported, or verified"
                )
            else:
                self._guard_osm_evidence(evidence)
        record = ClaimRecord(name, target, tuple(evidence_ids), reason, updated_at)
        self._claims[name] = record
        return record

    def _resolve_evidence(self, evidence_ids: tuple[str, ...]) -> tuple[Evidence, ...]:
        if len(set(evidence_ids)) != len(evidence_ids):
            raise DomainValidationError("DUPLICATE_EVIDENCE", "Evidence identifiers must be unique")
        missing = [item for item in evidence_ids if item not in self._evidence]
        if missing:
            raise DomainValidationError("EVIDENCE_NOT_FOUND", f"Unknown evidence identifiers: {missing}")
        return tuple(self._evidence[item] for item in evidence_ids)

    @staticmethod
    def _guard_feature_evidence(evidence: tuple[Evidence, ...]) -> None:
        if not evidence or any(
            not isinstance(item, ImplementationEvidence)
            or not item.passed
            or not item.reproducible
            or not item.artifact_hashes
            for item in evidence
        ):
            raise DomainValidationError(
                "AUTOMATED_CHECK_REQUIRED",
                "Implemented/verified feature status requires passed reproducible automated-check evidence",
            )

    def _guard_osm_evidence(self, evidence: tuple[Evidence, ...]) -> None:
        if not evidence:
            raise DomainValidationError(
                "VERIFICATION_EVIDENCE_REQUIRED", "Verified OSM claims require registered evidence"
            )
        failures: dict[str, tuple[str, ...]] = {}
        for item in evidence:
            if not isinstance(item, ExperimentEvidence):
                failures[item.evidence_id] = ("not-osm-experiment-evidence",)
                continue
            reasons = self.osm_verification_failures(item)
            if reasons:
                failures[item.evidence_id] = reasons
        if failures:
            raise DomainValidationError(
                "INELIGIBLE_OSM_EVIDENCE",
                f"Evidence cannot verify OSM performance: {failures}",
                actual=failures,
            )

    def osm_verification_failures(self, evidence: ExperimentEvidence) -> tuple[str, ...]:
        """Return stable reasons why evidence must be excluded from verified OSM aggregates."""
        run = evidence.run_manifest
        reasons: list[str] = []
        if run.mode == RunMode.LEGACY_DIRECT:
            reasons.append("legacy-direct-excluded")
        elif run.mode != RunMode.OSM_EVALUATION:
            reasons.append("not-osm-evaluation")
        if evidence.state != EvidenceState.COMPLETE:
            reasons.append(f"evidence-{evidence.state.value}")
        if not evidence.success_criteria_met:
            reasons.append("success-criteria-not-met")
        plan = self._plans.get(run.plan_id or "")
        if plan is None:
            reasons.append("missing-registered-plan")
        else:
            if evidence.plan_hash != plan.frozen_config_hash:
                reasons.append("plan-hash-mismatch")
            if run.checkpoint_hash not in plan.checkpoint_candidates:
                reasons.append("checkpoint-not-pre-registered")
            if tuple(run.seeds) != plan.seeds or evidence.episode_count != plan.episode_count:
                reasons.append("seed-or-episode-plan-mismatch")
            if evidence.confidence_interval_method != plan.confidence_interval_method:
                reasons.append("confidence-method-mismatch")
            if evidence.episode_count < plan.minimum_evaluation_episodes:
                reasons.append("preliminary-episode-count")
            evaluation_hashes = frozenset(evidence.evaluation_region_hashes)
            training_hashes = frozenset(evidence.training_region_hashes)
            if evaluation_hashes != plan.region_splits.test_hashes:
                reasons.append("unseen-test-regions-incomplete")
            if training_hashes != plan.region_splits.train_hashes:
                reasons.append("training-regions-mismatch")
            if evaluation_hashes & training_hashes:
                reasons.append("training-test-region-overlap")
        if run.observation_profile != OSM_OBSERVATION_PROFILE:
            reasons.append("semantic-profile-not-osm")
        if not run.checkpoint_hash:
            reasons.append("missing-checkpoint-hash")
        if not run.network_hash or run.network_hash not in evidence.evaluation_region_hashes:
            reasons.append("missing-evaluation-network-hash")
        if not run.run_config:
            reasons.append("missing-run-config")
        if not run.seeds or len(run.seeds) != evidence.episode_count:
            reasons.append("missing-seeds-or-episode-count")
        if not evidence.confidence_interval_method:
            reasons.append("missing-confidence-method")
        if not evidence.artifact_hashes or any(
            not name or not digest for name, digest in evidence.artifact_hashes.items()
        ):
            reasons.append("missing-result-artifacts")
        reproducibility = evidence.reproducibility
        if reproducibility is None or not reproducibility.complete:
            reasons.append("unreproducible-artifacts")
        else:
            if reproducibility.checkpoint_hash != run.checkpoint_hash:
                reasons.append("reproducibility-checkpoint-mismatch")
            if frozenset(reproducibility.network_hashes) != frozenset(evidence.evaluation_region_hashes):
                reasons.append("reproducibility-network-mismatch")
            if any(
                reproducibility.input_artifact_hashes.get(name) != digest
                for name, digest in evidence.artifact_hashes.items()
            ):
                reasons.append("reproducibility-artifact-mismatch")
        return tuple(dict.fromkeys(reasons))

    def verified_osm_evidence(self) -> tuple[ExperimentEvidence, ...]:
        """Return only evidence admissible for verified OSM performance reporting."""
        return tuple(
            item
            for item in self._evidence.values()
            if isinstance(item, ExperimentEvidence) and not self.osm_verification_failures(item)
        )

    @property
    def claims(self) -> tuple[ClaimRecord, ...]:
        return tuple(self._claims.values())


__all__ = ("ClaimKind", "ClaimRegistry")

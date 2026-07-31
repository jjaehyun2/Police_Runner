"""Immutable experiment plans, geographic splits, run evidence, and reproducibility."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from .canonical import content_hash
from .models import (
    DOMAIN_SCHEMA_VERSION,
    POLICE_COUNT,
    DomainValidationError,
    RunManifest,
    RunMode,
    _finite,
    _freeze_field,
    validate_schema_version,
)

OSM_OBSERVATION_PROFILE = "osm_topology_v1"
# Backward-compatible name used by early task 1.4 drafts.
OSM_TOPOLOGY_PROFILE = OSM_OBSERVATION_PROFILE


def _require_text(value: str | None, name: str) -> str:
    if value is None or not str(value).strip():
        raise DomainValidationError(
            "MISSING_EXPERIMENT_FIELD", f"{name} must be nonempty", path=name
        )
    return str(value)


def _as_hash_set(values: frozenset[str] | tuple[str, ...], name: str) -> frozenset[str]:
    result = frozenset(_require_text(value, name) for value in values)
    if not result:
        raise DomainValidationError(
            "EMPTY_REGION_SPLIT", f"{name} must contain at least one network hash", path=name
        )
    return result


@dataclass(frozen=True, slots=True)
class RegionSplits:
    """Pairwise-disjoint network identities for train, validation, and test."""

    train: frozenset[str]
    validation: frozenset[str]
    test: frozenset[str]


    def __post_init__(self) -> None:
        for name in ("train", "validation", "test"):
            object.__setattr__(self, name, _as_hash_set(getattr(self, name), name))
        overlaps = {
            "train_validation": sorted(self.train & self.validation),
            "train_test": sorted(self.train & self.test),
            "validation_test": sorted(self.validation & self.test),
        }
        conflicts = {name: values for name, values in overlaps.items() if values}
        if conflicts:
            raise DomainValidationError(
                "OVERLAPPING_REGION_SPLITS",
                "Training, validation, and test network hashes must be pairwise disjoint",
                actual=conflicts,
            )

    @property
    def train_hashes(self) -> frozenset[str]:
        return self.train

    @property
    def validation_hashes(self) -> frozenset[str]:
        return self.validation

    @property
    def test_hashes(self) -> frozenset[str]:
        return self.test

    def hashes_for(self, partition: str) -> frozenset[str]:
        if partition not in {"train", "validation", "test"}:
            raise DomainValidationError(
                "UNKNOWN_REGION_SPLIT", f"Unknown region split {partition!r}"
            )
        return getattr(self, partition)


def _plan_payload(
    *,
    region_splits: RegionSplits,
    checkpoint_candidates: tuple[str, ...],
    fugitive_rules: Mapping[str, Any],
    police_count: int,
    capture_radius_m: float,
    max_steps: int,
    episode_count: int,
    seeds: tuple[int, ...],
    success_criteria: Mapping[str, Any],
    evaluation_config: Mapping[str, Any],
    baseline_policy: str,
    confidence_interval_method: str,
    minimum_evaluation_episodes: int,
) -> dict[str, Any]:
    return {
        "region_splits": region_splits,
        "checkpoint_candidates": checkpoint_candidates,
        "fugitive_rules": fugitive_rules,
        "police_count": police_count,
        "capture_radius_m": capture_radius_m,
        "max_steps": max_steps,
        "episode_count": episode_count,
        "seeds": seeds,
        "success_criteria": success_criteria,
        "evaluation_config": evaluation_config,
        "baseline_policy": baseline_policy,
        "confidence_interval_method": confidence_interval_method,
        "minimum_evaluation_episodes": minimum_evaluation_episodes,
    }


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    """A content-frozen plan containing every required OSM evaluation input."""

    plan_id: str
    frozen_config_hash: str
    region_splits: RegionSplits
    checkpoint_candidates: tuple[str, ...]
    fugitive_rules: Mapping[str, Any]
    capture_radius_m: float
    max_steps: int
    episode_count: int
    seeds: tuple[int, ...]
    success_criteria: Mapping[str, Any]
    evaluation_config: Mapping[str, Any]
    police_count: int = POLICE_COUNT
    baseline_policy: str = "baseline-police-v1"
    confidence_interval_method: str = "wilson"
    minimum_evaluation_episodes: int = 1
    schema_version: str = DOMAIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version)
        for name in (
            "plan_id",
            "frozen_config_hash",
            "baseline_policy",
            "confidence_interval_method",
        ):
            _require_text(getattr(self, name), name)
        candidates = tuple(
            _require_text(value, "checkpoint_candidates")
            for value in self.checkpoint_candidates
        )
        seeds = tuple(self.seeds)
        object.__setattr__(self, "checkpoint_candidates", candidates)
        object.__setattr__(self, "seeds", seeds)
        for name in ("fugitive_rules", "success_criteria", "evaluation_config"):
            _freeze_field(self, name)
        radius = _finite(self.capture_radius_m, "capture_radius_m")
        object.__setattr__(self, "capture_radius_m", radius)
        if self.police_count != POLICE_COUNT:
            raise DomainValidationError(
                "INVALID_POLICE_COUNT",
                f"Experiment plans require exactly {POLICE_COUNT} police",
                expected=POLICE_COUNT,
                actual=self.police_count,
            )
        if radius <= 0 or self.max_steps <= 0 or self.episode_count <= 0:
            raise DomainValidationError(
                "INVALID_EXPERIMENT_PLAN",
                "Radius, steps, and episode count must be positive",
            )
        if self.minimum_evaluation_episodes <= 0:
            raise DomainValidationError(
                "INVALID_MINIMUM_EPISODES", "Minimum evaluation episodes must be positive"
            )
        if self.episode_count < self.minimum_evaluation_episodes:
            raise DomainValidationError(
                "PRELIMINARY_EXPERIMENT_PLAN",
                "Planned episode count must meet the pre-registered evaluation minimum",
                expected=self.minimum_evaluation_episodes,
                actual=self.episode_count,
            )
        if not candidates or not self.fugitive_rules or not self.success_criteria or not self.evaluation_config:
            raise DomainValidationError(
                "INCOMPLETE_EXPERIMENT_PLAN",
                "Checkpoint candidates, fugitive rules, success criteria, and evaluation config are required",
            )
        if len(seeds) != self.episode_count:
            raise DomainValidationError(
                "SEED_COUNT_MISMATCH",
                "One deterministic seed is required per planned episode",
                expected=self.episode_count,
                actual=len(seeds),
            )
        expected = content_hash(self.identity_payload())
        if self.frozen_config_hash != expected:
            raise DomainValidationError(
                "PLAN_HASH_MISMATCH",
                "Experiment plan content does not match its frozen configuration hash",
                expected=expected,
                actual=self.frozen_config_hash,
            )

    def identity_payload(self) -> dict[str, Any]:
        return _plan_payload(
            region_splits=self.region_splits,
            checkpoint_candidates=self.checkpoint_candidates,
            fugitive_rules=self.fugitive_rules,
            police_count=self.police_count,
            capture_radius_m=self.capture_radius_m,
            max_steps=self.max_steps,
            episode_count=self.episode_count,
            seeds=self.seeds,
            success_criteria=self.success_criteria,
            evaluation_config=self.evaluation_config,
            baseline_policy=self.baseline_policy,
            confidence_interval_method=self.confidence_interval_method,
            minimum_evaluation_episodes=self.minimum_evaluation_episodes,
        )

    @classmethod
    def create(
        cls,
        *,
        plan_id: str,
        region_splits: RegionSplits,
        checkpoint_candidates: tuple[str, ...],
        fugitive_rules: Mapping[str, Any],
        capture_radius_m: float,
        max_steps: int,
        episode_count: int,
        seeds: tuple[int, ...],
        success_criteria: Mapping[str, Any],
        evaluation_config: Mapping[str, Any],
        police_count: int = POLICE_COUNT,
        baseline_policy: str = "baseline-police-v1",
        confidence_interval_method: str = "wilson",
        minimum_evaluation_episodes: int = 1,
    ) -> "ExperimentPlan":
        payload = _plan_payload(
            region_splits=region_splits,
            checkpoint_candidates=tuple(checkpoint_candidates),
            fugitive_rules=fugitive_rules,
            police_count=police_count,
            capture_radius_m=capture_radius_m,
            max_steps=max_steps,
            episode_count=episode_count,
            seeds=tuple(seeds),
            success_criteria=success_criteria,
            evaluation_config=evaluation_config,
            baseline_policy=baseline_policy,
            confidence_interval_method=confidence_interval_method,
            minimum_evaluation_episodes=minimum_evaluation_episodes,
        )
        return cls(
            plan_id=plan_id,
            frozen_config_hash=content_hash(payload),
            **payload,
        )


class EvidenceState(str, Enum):
    COMPLETE = "complete"
    PRELIMINARY = "preliminary"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ImplementationEvidence:
    """Automated-check evidence used only for implementation claims."""

    evidence_id: str
    checks: tuple[str, ...]
    artifact_hashes: Mapping[str, str]
    passed: bool
    reproducible: bool
    schema_version: str = DOMAIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version)
        _require_text(self.evidence_id, "evidence_id")
        checks = tuple(_require_text(value, "checks") for value in self.checks)
        object.__setattr__(self, "checks", checks)
        _freeze_field(self, "artifact_hashes")
        if not checks:
            raise DomainValidationError(
                "MISSING_AUTOMATED_CHECKS", "Implementation evidence requires automated checks"
            )
        if any(not name or not digest for name, digest in self.artifact_hashes.items()):
            raise DomainValidationError(
                "INVALID_ARTIFACT_HASH", "Artifact names and hashes must be nonempty"
            )


@dataclass(frozen=True, slots=True)
class ReproducibilityRecord:
    """Inputs needed to independently reproduce an experiment artifact set."""

    checkpoint_hash: str | None
    network_hashes: frozenset[str]
    run_config_hash: str | None
    code_hash: str | None
    input_artifact_hashes: Mapping[str, str] = field(default_factory=dict)
    schema_version: str = DOMAIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version)
        object.__setattr__(
            self,
            "network_hashes",
            frozenset(str(value) for value in self.network_hashes),
        )
        _freeze_field(self, "input_artifact_hashes")

    @property
    def complete(self) -> bool:
        required_text = (self.checkpoint_hash, self.run_config_hash, self.code_hash)
        return (
            all(value is not None and str(value).strip() for value in required_text)
            and bool(self.network_hashes)
            and all(value.strip() for value in self.network_hashes)
            and bool(self.input_artifact_hashes)
            and all(
                bool(name.strip()) and bool(str(digest).strip())
                for name, digest in self.input_artifact_hashes.items()
            )
        )


@dataclass(frozen=True, slots=True)
class ExperimentEvidence:
    """Immutable result evidence; eligibility is decided by the claim registry."""

    evidence_id: str
    run_manifest: RunManifest
    plan_hash: str
    state: EvidenceState
    episode_count: int
    training_region_hashes: frozenset[str]
    evaluation_region_hashes: frozenset[str]
    success_criteria_met: bool
    confidence_interval_method: str
    artifact_hashes: Mapping[str, str] = field(default_factory=dict)
    reproducibility: ReproducibilityRecord | None = None
    schema_version: str = DOMAIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version)
        _require_text(self.evidence_id, "evidence_id")
        _require_text(self.plan_hash, "plan_hash")
        _require_text(self.confidence_interval_method, "confidence_interval_method")
        if not isinstance(self.run_manifest, RunManifest):
            raise DomainValidationError(
                "INVALID_RUN_EVIDENCE", "Experiment evidence requires an immutable RunManifest"
            )
        if not isinstance(self.state, EvidenceState):
            try:
                object.__setattr__(self, "state", EvidenceState(self.state))
            except ValueError as exc:
                raise DomainValidationError(
                    "INVALID_EVIDENCE_STATE", f"Unknown evidence state {self.state!r}"
                ) from exc
        if self.episode_count <= 0:
            raise DomainValidationError(
                "INVALID_EPISODE_COUNT", "Experiment evidence episode count must be positive"
            )
        for name in ("training_region_hashes", "evaluation_region_hashes"):
            object.__setattr__(self, name, _as_hash_set(getattr(self, name), name))
        _freeze_field(self, "artifact_hashes")
        if self.success_criteria_met and self.state is not EvidenceState.COMPLETE:
            raise DomainValidationError(
                "INCOMPLETE_SUCCESS_EVIDENCE",
                "Only complete evidence may report that success criteria were met",
            )


@dataclass(frozen=True, slots=True)
class TrainingRunLineage:
    """Mode-specific lineage for an OSM fine-tune or from-scratch run."""

    run_manifest: RunManifest
    region_splits: RegionSplits
    parent_checkpoint_hash: str | None = None
    completed_test_region_hashes: frozenset[str] = field(default_factory=frozenset)
    checkpoint_manifest_hash: str | None = None
    schema_version: str = DOMAIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version)
        run = self.run_manifest
        if run.mode not in {RunMode.OSM_FINETUNE, RunMode.OSM_FROM_SCRATCH}:
            raise DomainValidationError(
                "INVALID_TRAINING_MODE",
                "Training lineage is only valid for osm-finetune or osm-from-scratch runs",
            )
        if run.observation_profile != OSM_OBSERVATION_PROFILE:
            raise DomainValidationError(
                "INVALID_TRAINING_PROFILE",
                "OSM training lineage must preserve osm_topology_v1",
            )
        if run.network_hash not in self.region_splits.train:
            raise DomainValidationError(
                "TRAINING_NETWORK_OUTSIDE_SPLIT",
                "The training run network must belong to the registered training split",
            )
        if run.mode is RunMode.OSM_FINETUNE and not self.parent_checkpoint_hash:
            raise DomainValidationError(
                "MISSING_PARENT_CHECKPOINT",
                "Fine-tune lineage requires a parent checkpoint hash",
            )
        if run.mode is RunMode.OSM_FROM_SCRATCH and self.parent_checkpoint_hash is not None:
            raise DomainValidationError(
                "UNEXPECTED_PARENT_CHECKPOINT",
                "From-scratch lineage cannot reference parent weights",
            )
        completed = frozenset(
            _require_text(value, "completed_test_region_hashes")
            for value in self.completed_test_region_hashes
        )
        object.__setattr__(self, "completed_test_region_hashes", completed)
        if completed and completed != self.region_splits.test:
            raise DomainValidationError(
                "INCOMPLETE_UNSEEN_TEST_EVIDENCE",
                "Completed unseen-test evidence must cover the entire registered test split",
                expected=sorted(self.region_splits.test),
                actual=sorted(completed),
            )
        if self.checkpoint_manifest_hash and not completed:
            raise DomainValidationError(
                "MISSING_UNSEEN_TEST_EVIDENCE",
                "A completed checkpoint manifest requires unseen-test evidence",
            )

    @property
    def has_complete_unseen_test_evidence(self) -> bool:
        return bool(
            self.checkpoint_manifest_hash
            and self.completed_test_region_hashes == self.region_splits.test
        )


__all__ = (
    "EvidenceState",
    "ExperimentEvidence",
    "ExperimentPlan",
    "ImplementationEvidence",
    "OSM_OBSERVATION_PROFILE",
    "OSM_TOPOLOGY_PROFILE",
    "RegionSplits",
    "ReproducibilityRecord",
    "TrainingRunLineage",
)

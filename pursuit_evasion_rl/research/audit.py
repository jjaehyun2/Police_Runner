"""Fail-closed audit of legacy assets and pre-protocol results.

This module records existence, test execution, experiment observations, and paper
claims as distinct evidence types.  It never treats one type as a substitute for
another.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import re
import subprocess
import sys
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .canonical import content_hash, sha256_bytes
from .domain import (
    DataKind,
    EvidenceRecord,
    EvidenceType,
    PersistedModel,
    PriorResultStatus,
    TestStatus,
    exactly_one,
)
from .errors import ResearchValidationError

UNKNOWN = "unknown"


def _required(value: str, path: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ResearchValidationError(
            "MISSING_REQUIRED_FIELD", f"{path} must be non-empty", path=path
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class AssetCategory(str, Enum):
    ENVIRONMENT = "environment"
    OBSERVATION = "observation"
    POLICY = "policy"
    TRAINING = "training"
    EVALUATION = "evaluation"
    METRICS = "metrics"
    CHECKPOINT = "checkpoint"


class ClaimKind(str, Enum):
    IMPLEMENTATION = "implementation"
    TEST = "test"
    PRIOR_RESULT = "prior_result"
    PAPER = "paper"


_EXPECTED_EVIDENCE = {
    ClaimKind.IMPLEMENTATION: EvidenceType.IMPLEMENTATION_EXISTENCE,
    ClaimKind.TEST: EvidenceType.TEST_RESULT,
    ClaimKind.PRIOR_RESULT: EvidenceType.EXPERIMENT_OBSERVATION,
    ClaimKind.PAPER: EvidenceType.PAPER_CLAIM,
}


@dataclass(frozen=True, slots=True)
class AssetSpec:
    category: AssetCategory
    relative_path: str
    test_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TestExecution(PersistedModel):
    command: tuple[str, ...]
    return_code: int
    output_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "command", tuple(str(item) for item in self.command))
        if not self.command:
            raise ResearchValidationError(
                "MISSING_REQUIRED_FIELD", "test command must not be empty", path="command"
            )
        _required(self.output_hash, "output_hash")
        super(TestExecution, self).__post_init__()


@dataclass(frozen=True, slots=True)
class ImplementationInventory(PersistedModel):
    category: AssetCategory
    asset_path: str
    code_revision: str
    test_status: TestStatus
    reason: str
    exists: bool
    test_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", exactly_one(self.category, AssetCategory, path="category"))
        object.__setattr__(
            self, "test_status", exactly_one(self.test_status, TestStatus, path="test_status")
        )
        for name in ("asset_path", "code_revision", "reason"):
            _required(getattr(self, name), name)
        object.__setattr__(self, "test_paths", tuple(str(item) for item in self.test_paths))
        super(ImplementationInventory, self).__post_init__()


@dataclass(frozen=True, slots=True)
class KnownDefect(PersistedModel):
    code: str
    description: str
    result_changing: bool

    def __post_init__(self) -> None:
        _required(self.code, "code")
        _required(self.description, "description")
        super(KnownDefect, self).__post_init__()

    @property
    def invalidates_result(self) -> bool:
        normalized = re.sub(r"[^a-z0-9]+", "_", self.code.casefold()).strip("_")
        return self.result_changing or normalized in {
            "escape_bug",
            "escape_detection_bug",
            "pre_fix_escape_bug",
        }


PRIOR_PROVENANCE_FIELDS = (
    "generator_path",
    "code_revision",
    "data_kind",
    "map_hash",
    "condition",
    "training_seed",
    "checkpoint_time",
    "episodes",
    "created_at_utc",
    "method",
)


@dataclass(frozen=True, slots=True)
class PriorResultInput:
    result_id: str
    generator_path: str | None
    code_revision: str | None
    data_kind: DataKind | str | None
    map_hash: str | None
    condition: str | None
    training_seed: int | str | None
    checkpoint_time: str | None
    episodes: int | str | None
    created_at_utc: str | None
    method: str | None
    source: str
    observed_value: Any
    known_defects: tuple[KnownDefect, ...] = ()
    unknown_reasons: Mapping[str, str] = field(default_factory=dict)
    independent_training_seeds: int | str = UNKNOWN
    evaluation_episodes_per_condition: int | str = UNKNOWN
    has_ci95: bool | str = UNKNOWN
    has_spatially_disjoint_split: bool | str = UNKNOWN
    same_training_distribution_moving_window: bool = False
    force_preliminary: bool = False


@dataclass(frozen=True, slots=True)
class PriorResultRecord(PersistedModel):
    result_id: str
    generator_path: str
    code_revision: str
    data_kind: str
    map_hash: str
    condition: str
    training_seed: int | str
    checkpoint_time: str
    episodes: int | str
    created_at_utc: str
    method: str
    source: str
    observed_value: Any
    known_defects: tuple[KnownDefect, ...]
    unknown_reasons: Mapping[str, str]
    independent_training_seeds: int | str
    evaluation_episodes_per_condition: int | str
    has_ci95: bool | str
    has_spatially_disjoint_split: bool | str
    same_training_distribution_moving_window: bool
    fixed_preliminary: bool
    status: PriorResultStatus
    classification_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("result_id", "generator_path", "code_revision", "data_kind", "map_hash",
                     "condition", "checkpoint_time", "created_at_utc", "method", "source"):
            _required(getattr(self, name), name)
        object.__setattr__(
            self, "status", exactly_one(self.status, PriorResultStatus, path="prior_result_status")
        )
        object.__setattr__(self, "known_defects", tuple(self.known_defects))
        object.__setattr__(
            self, "unknown_reasons", MappingProxyType(dict(self.unknown_reasons))
        )
        object.__setattr__(
            self, "classification_reasons", tuple(str(item) for item in self.classification_reasons)
        )
        if self.observed_value is None or (
            isinstance(self.observed_value, Mapping) and not self.observed_value
        ):
            raise ResearchValidationError(
                "INCOMPLETE_PRIOR_RESULT",
                "prior result observed_value must be present and non-empty",
                path="observed_value",
            )
        _validate_unknown_reasons(self)
        expected_status = _expected_prior_status(self)
        if self.status is not expected_status:
            raise ResearchValidationError(
                "PRIOR_RESULT_STATUS_MISMATCH",
                "prior result status violates defect precedence or verification gates",
                path="prior_result_status",
                expected=expected_status.value,
                actual=self.status.value,
            )
        super(PriorResultRecord, self).__post_init__()


@dataclass(frozen=True, slots=True)
class ClaimRegisterEntry(PersistedModel):
    claim_id: str
    claim_kind: ClaimKind
    statement: str
    status: PriorResultStatus | str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _required(self.claim_id, "claim_id")
        _required(self.statement, "statement")
        object.__setattr__(
            self, "claim_kind", exactly_one(self.claim_kind, ClaimKind, path="claim_kind")
        )
        if self.claim_kind is ClaimKind.PRIOR_RESULT:
            object.__setattr__(
                self,
                "status",
                exactly_one(self.status, PriorResultStatus, path="prior_result_status"),
            )
        else:
            _required(str(self.status), "status")
        object.__setattr__(self, "evidence_ids", tuple(str(item) for item in self.evidence_ids))
        if not self.evidence_ids:
            raise ResearchValidationError(
                "MISSING_REQUIRED_FIELD", "claims require evidence", path="evidence_ids"
            )
        super(ClaimRegisterEntry, self).__post_init__()


def _unknown(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().casefold() == UNKNOWN)


def _normalize_unknown(value: Any) -> Any:
    return UNKNOWN if _unknown(value) else value


def _validate_unknown_reasons(record: PriorResultRecord) -> None:
    reasons = dict(record.unknown_reasons)
    for field_name in PRIOR_PROVENANCE_FIELDS:
        value = getattr(record, field_name)
        if _unknown(value) and not reasons.get(field_name, "").strip():
            raise ResearchValidationError(
                "UNKNOWN_REASON_REQUIRED",
                f"unknown provenance field {field_name} requires a field-specific reason",
                path=f"unknown_reasons.{field_name}",
            )
    for field_name, reason in reasons.items():
        if field_name not in PRIOR_PROVENANCE_FIELDS:
            raise ResearchValidationError(
                "UNKNOWN_REASON_FIELD_MISMATCH",
                "unknown reason names a non-provenance field",
                path=f"unknown_reasons.{field_name}",
            )
        if not _unknown(getattr(record, field_name)):
            raise ResearchValidationError(
                "UNKNOWN_REASON_FIELD_MISMATCH",
                "unknown reason is present for a known provenance field",
                path=f"unknown_reasons.{field_name}",
            )
        _required(reason, f"unknown_reasons.{field_name}")


def _expected_prior_status(record: PriorResultRecord) -> PriorResultStatus:
    if any(defect.invalidates_result for defect in record.known_defects):
        return PriorResultStatus.INVALID
    seeds = _positive_int_or_unknown(
        record.independent_training_seeds, "independent_training_seeds"
    )
    episodes = _positive_int_or_unknown(
        record.evaluation_episodes_per_condition, "evaluation_episodes_per_condition"
    )
    has_unknown_provenance = any(
        _unknown(getattr(record, field_name)) for field_name in PRIOR_PROVENANCE_FIELDS
    )
    if (
        record.fixed_preliminary
        or record.same_training_distribution_moving_window
        or seeds is None
        or seeds < 3
        or episodes is None
        or episodes < 100
        or record.has_ci95 is not True
        or record.has_spatially_disjoint_split is not True
        or has_unknown_provenance
    ):
        return PriorResultStatus.PRELIMINARY
    return PriorResultStatus.VERIFIED


def _positive_int_or_unknown(value: int | str, field_name: str) -> int | None:
    if _unknown(value):
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResearchValidationError(
            "INVALID_PROVENANCE_VALUE",
            f"{field_name} must be a non-negative integer or unknown",
            path=field_name,
            actual=value,
        )
    return value


def classify_prior_result(raw: PriorResultInput) -> tuple[PriorResultStatus, tuple[str, ...]]:
    """Classify with result-changing defects taking unconditional precedence."""
    invalidating = tuple(defect.code for defect in raw.known_defects if defect.invalidates_result)
    if invalidating:
        return PriorResultStatus.INVALID, (
            "known result-changing defect takes precedence: " + ", ".join(invalidating),
        )

    seeds = _positive_int_or_unknown(raw.independent_training_seeds, "independent_training_seeds")
    episodes = _positive_int_or_unknown(
        raw.evaluation_episodes_per_condition, "evaluation_episodes_per_condition"
    )
    preliminary: list[str] = []
    if raw.force_preliminary:
        preliminary.append("result is explicitly fixed as preliminary")
    if raw.same_training_distribution_moving_window:
        preliminary.append("selected moving window from the same training distribution")
    if seeds is None or seeds < 3:
        preliminary.append("fewer than 3 confirmed independent training seeds")
    if episodes is None or episodes < 100:
        preliminary.append("fewer than 100 confirmed evaluation episodes per condition")
    if raw.has_ci95 is not True:
        preliminary.append("95% confidence interval is absent or unknown")
    if raw.has_spatially_disjoint_split is not True:
        preliminary.append("spatially disjoint split evaluation is absent or unknown")
    if any(_unknown(getattr(raw, name)) for name in PRIOR_PROVENANCE_FIELDS):
        preliminary.append("required provenance contains unknown fields")
    if preliminary:
        return PriorResultStatus.PRELIMINARY, tuple(dict.fromkeys(preliminary))
    return PriorResultStatus.VERIFIED, ("all provenance and verification gates passed",)


def build_prior_result(raw: PriorResultInput) -> PriorResultRecord:
    status, reasons = classify_prior_result(raw)
    data_kind = raw.data_kind.value if isinstance(raw.data_kind, DataKind) else raw.data_kind
    return PriorResultRecord(
        result_id=raw.result_id,
        generator_path=_normalize_unknown(raw.generator_path),
        code_revision=_normalize_unknown(raw.code_revision),
        data_kind=_normalize_unknown(data_kind),
        map_hash=_normalize_unknown(raw.map_hash),
        condition=_normalize_unknown(raw.condition),
        training_seed=_normalize_unknown(raw.training_seed),
        checkpoint_time=_normalize_unknown(raw.checkpoint_time),
        episodes=_normalize_unknown(raw.episodes),
        created_at_utc=_normalize_unknown(raw.created_at_utc),
        method=_normalize_unknown(raw.method),
        source=raw.source,
        observed_value=raw.observed_value,
        known_defects=tuple(raw.known_defects),
        unknown_reasons=raw.unknown_reasons,
        independent_training_seeds=raw.independent_training_seeds,
        evaluation_episodes_per_condition=raw.evaluation_episodes_per_condition,
        has_ci95=raw.has_ci95,
        has_spatially_disjoint_split=raw.has_spatially_disjoint_split,
        same_training_distribution_moving_window=raw.same_training_distribution_moving_window,
        fixed_preliminary=raw.force_preliminary,
        status=status,
        classification_reasons=reasons,
    )


class EvidenceRegistry:
    """Registers complete evidence and enforces exact evidence-type links."""

    def __init__(self) -> None:
        self._records: dict[str, EvidenceRecord] = {}

    @staticmethod
    def _validate_complete(record: EvidenceRecord) -> None:
        if not record.verify_hash():
            raise ResearchValidationError(
                "CONTENT_HASH_MISMATCH", "evidence hash verification failed", path="content_hash"
            )
        if record.extracted_value is None or (
            isinstance(record.extracted_value, Mapping) and not record.extracted_value
        ):
            raise ResearchValidationError(
                "INCOMPLETE_EVIDENCE_RECORD",
                "evidence extracted_value must be present and non-empty",
                path="extracted_value",
            )
        try:
            parsed = datetime.fromisoformat(record.created_at_utc.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ResearchValidationError(
                "INCOMPLETE_EVIDENCE_RECORD",
                "evidence creation time must be ISO-8601",
                path="created_at_utc",
            ) from exc
        if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
            raise ResearchValidationError(
                "INCOMPLETE_EVIDENCE_RECORD",
                "evidence creation time must be expressed in UTC",
                path="created_at_utc",
            )

    def register(self, record: EvidenceRecord) -> EvidenceRecord:
        self._validate_complete(record)
        if record.record_id in self._records:
            raise ResearchValidationError(
                "DUPLICATE_IDENTIFIER", "evidence record_id must be unique", path="record_id"
            )
        self._records[record.record_id] = record
        return record

    def require_type(
        self, evidence_ids: Sequence[str], expected_type: EvidenceType
    ) -> tuple[EvidenceRecord, ...]:
        records: list[EvidenceRecord] = []
        for evidence_id in evidence_ids:
            if evidence_id not in self._records:
                raise ResearchValidationError(
                    "MISSING_EVIDENCE", "claim references unknown evidence", path="evidence_ids",
                    actual=evidence_id,
                )
            record = self._records[evidence_id]
            if record.evidence_type is not expected_type:
                raise ResearchValidationError(
                    "EVIDENCE_TYPE_SUBSTITUTION",
                    f"{record.evidence_type.value} cannot substitute for {expected_type.value}",
                    path="evidence_ids",
                    expected=expected_type.value,
                    actual=record.evidence_type.value,
                )
            records.append(record)
        return tuple(records)

    @property
    def records(self) -> tuple[EvidenceRecord, ...]:
        return tuple(self._records.values())


class ClaimRegister:
    def __init__(self, evidence: EvidenceRegistry) -> None:
        self._evidence = evidence
        self._entries: dict[str, ClaimRegisterEntry] = {}

    def register(self, entry: ClaimRegisterEntry) -> ClaimRegisterEntry:
        if entry.claim_id in self._entries:
            raise ResearchValidationError(
                "DUPLICATE_IDENTIFIER", "claim_id must be unique", path="claim_id"
            )
        self._evidence.require_type(entry.evidence_ids, _EXPECTED_EVIDENCE[entry.claim_kind])
        self._entries[entry.claim_id] = entry
        return entry

    @property
    def entries(self) -> tuple[ClaimRegisterEntry, ...]:
        return tuple(self._entries.values())


DEFAULT_ASSETS: tuple[AssetSpec, ...] = (
    AssetSpec(AssetCategory.ENVIRONMENT, "pursuit_evasion_rl/osm_demo/environment.py",
              ("tests/test_osm_demo_environment.py",)),
    AssetSpec(AssetCategory.OBSERVATION, "pursuit_evasion_rl/osm_demo/observations.py",
              ("tests/test_osm_demo_observations.py",)),
    AssetSpec(AssetCategory.OBSERVATION, "osm_obs_aug.py"),
    AssetSpec(AssetCategory.POLICY, "pursuit_evasion_rl/osm_demo/policies.py",
              ("tests/test_osm_demo_policies.py",)),
    AssetSpec(AssetCategory.POLICY, "demo_pursuit.py",
              ("tests/test_osm_demo_baseline_policy.py",)),
    AssetSpec(AssetCategory.TRAINING, "pursuit_evasion_rl/training/algorithms.py",
              ("tests/test_osm_demo_training.py",)),
    AssetSpec(AssetCategory.TRAINING, "train_osm_pursuit.py"),
    AssetSpec(AssetCategory.EVALUATION, "pursuit_evasion_rl/osm_demo/evaluation.py",
              ("tests/test_osm_demo_evaluation.py",)),
    AssetSpec(AssetCategory.EVALUATION, "eval_trained.py"),
    AssetSpec(AssetCategory.METRICS, "pursuit_evasion_rl/osm_demo/metrics.py",
              ("tests/test_osm_demo_metrics.py",)),
    AssetSpec(AssetCategory.CHECKPOINT, "pursuit_evasion_rl/osm_demo/checkpoint.py",
              ("tests/test_osm_demo_checkpoint.py",)),
    AssetSpec(AssetCategory.CHECKPOINT, "checkpoints/osm_mappo/best_v2.pt"),
)


def code_revision(root: Path) -> str:
    """Return commit plus a digest of dirty paths, without claiming a clean revision."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.encode("utf-8")
    except (OSError, subprocess.CalledProcessError):
        return UNKNOWN
    return f"{commit}+clean" if not dirty else f"{commit}+dirty:{sha256_bytes(dirty)[:16]}"


def run_asset_tests(
    root: Path, assets: Sequence[AssetSpec] = DEFAULT_ASSETS
) -> Mapping[str, TestExecution]:
    """Run each declared targeted test once and return results keyed by asset path."""
    by_suite: dict[tuple[str, ...], TestExecution] = {}
    results: dict[str, TestExecution] = {}
    for asset in assets:
        if not asset.test_paths or not all((root / item).is_file() for item in asset.test_paths):
            continue
        if asset.test_paths not in by_suite:
            command = (sys.executable, "-m", "pytest", *asset.test_paths, "-q")
            completed = subprocess.run(command, cwd=root, capture_output=True)
            output = completed.stdout + completed.stderr
            by_suite[asset.test_paths] = TestExecution(
                command=command,
                return_code=completed.returncode,
                output_hash=sha256_bytes(output),
            )
        results[asset.relative_path] = by_suite[asset.test_paths]
    return MappingProxyType(results)


def _inventory_status(
    root: Path, asset: AssetSpec, test_results: Mapping[str, TestExecution]
) -> tuple[TestStatus, str]:
    if not (root / asset.relative_path).is_file():
        return TestStatus.NOT_TESTED, "asset file is missing; no test result can apply"
    if not asset.test_paths:
        return TestStatus.NOT_TESTED, "no asset-specific automated test is registered"
    missing_tests = [item for item in asset.test_paths if not (root / item).is_file()]
    if missing_tests:
        return TestStatus.NOT_TESTED, "registered test file is missing: " + ", ".join(missing_tests)
    execution = test_results.get(asset.relative_path)
    if execution is None:
        return TestStatus.NOT_TESTED, "registered automated test was not executed"
    if execution.return_code == 0:
        return TestStatus.PASSED, "targeted automated test passed: " + ", ".join(asset.test_paths)
    return TestStatus.FAILED, "targeted automated test failed: " + ", ".join(asset.test_paths)


class AssetAuditor:
    def __init__(self, root: Path, assets: Sequence[AssetSpec] = DEFAULT_ASSETS) -> None:
        self.root = root.resolve()
        self.assets = tuple(assets)

    def inventory(
        self,
        revision: str | None = None,
        test_results: Mapping[str, TestExecution] | None = None,
    ) -> tuple[ImplementationInventory, ...]:
        revision = revision or code_revision(self.root)
        results = test_results or {}
        records = []
        for asset in self.assets:
            status, reason = _inventory_status(self.root, asset, results)
            records.append(
                ImplementationInventory(
                    category=asset.category,
                    asset_path=asset.relative_path,
                    code_revision=revision,
                    test_status=status,
                    reason=reason,
                    exists=(self.root / asset.relative_path).is_file(),
                    test_paths=asset.test_paths,
                )
            )
        return tuple(records)


def inventory_evidence(
    inventory: Sequence[ImplementationInventory],
    test_results: Mapping[str, TestExecution],
    *,
    created_at_utc: str,
) -> tuple[tuple[EvidenceRecord, ...], tuple[ClaimRegisterEntry, ...]]:
    registry = EvidenceRegistry()
    claims = ClaimRegister(registry)
    for index, item in enumerate(inventory):
        implementation_id = f"implementation-{index:03d}"
        registry.register(EvidenceRecord(
            record_id=implementation_id,
            evidence_type=EvidenceType.IMPLEMENTATION_EXISTENCE,
            producer="research.asset-auditor",
            created_at_utc=created_at_utc,
            method="filesystem existence inventory",
            source=item.asset_path,
            extracted_value={
                "exists": item.exists,
                "category": item.category.value,
                "code_revision": item.code_revision,
                "test_status": item.test_status.value,
            },
            verification="filesystem_checked",
            limitations=("existence does not establish test success or performance",),
        ))
        claims.register(ClaimRegisterEntry(
            claim_id=f"claim-{implementation_id}",
            claim_kind=ClaimKind.IMPLEMENTATION,
            statement=f"Implementation asset existence: {item.asset_path}",
            status="exists" if item.exists else "missing",
            evidence_ids=(implementation_id,),
        ))
        execution = test_results.get(item.asset_path)
        if execution is None:
            continue
        test_id = f"test-{index:03d}"
        registry.register(EvidenceRecord(
            record_id=test_id,
            evidence_type=EvidenceType.TEST_RESULT,
            producer="pytest",
            created_at_utc=created_at_utc,
            method="targeted pytest subprocess",
            source=", ".join(item.test_paths),
            extracted_value={
                "status": item.test_status.value,
                "return_code": execution.return_code,
                "command": execution.command,
                "output_hash": execution.output_hash,
            },
            verification="executed",
            limitations=("test result applies only to the named automated tests",),
        ))
        claims.register(ClaimRegisterEntry(
            claim_id=f"claim-{test_id}",
            claim_kind=ClaimKind.TEST,
            statement=f"Automated test status for {item.asset_path}",
            status=item.test_status.value,
            evidence_ids=(test_id,),
        ))
    return registry.records, claims.entries


_CAPTURE_RE = re.compile(
    r"ep\s+(?P<episode>\d+)\s+\|\s+capture_rate\(last(?P<window>\d+)\)="
    r"(?P<rate>\d+(?:\.\d+)?)"
)


def prior_from_training_log(
    root: Path,
    relative_log_path: str = "checkpoints/osm_mappo/training.log",
    *,
    known_defects: Sequence[KnownDefect] = (),
) -> PriorResultInput:
    """Extract the selected peak moving-window observation without upgrading its provenance."""
    path = root / relative_log_path
    if not path.is_file():
        raise ResearchValidationError(
            "MISSING_PRIOR_RESULT_SOURCE",
            "legacy training log does not exist",
            path=relative_log_path,
        )
    raw_bytes = path.read_bytes()
    text = raw_bytes.decode("utf-8")
    observations = []
    for match in _CAPTURE_RE.finditer(text):
        observations.append(
            (float(match.group("rate")), int(match.group("episode")), int(match.group("window")))
        )
    if not observations:
        raise ResearchValidationError(
            "PRIOR_RESULT_PARSE_FAILED",
            "training log contains no capture_rate moving-window observation",
            path=relative_log_path,
        )
    rate, episode, window = max(observations, key=lambda item: (item[0], -item[1]))
    parameter_lines = sorted({
        line.split("] ", 1)[-1]
        for line in text.splitlines()
        if "params:" in line or "placement:" in line or "net:" in line
    })
    condition = content_hash({"legacy_log_configuration": parameter_lines})
    return PriorResultInput(
        result_id=f"legacy-osm-mappo-peak-last{window}",
        generator_path="train_osm_pursuit.py",
        code_revision=UNKNOWN,
        data_kind=UNKNOWN,
        map_hash=UNKNOWN,
        condition=condition,
        training_seed=UNKNOWN,
        checkpoint_time=f"episode_{episode}_best_checkpoint_event",
        episodes=window,
        created_at_utc=UNKNOWN,
        method="capture_rate(lastN) peak parser over legacy training.log",
        source=relative_log_path,
        observed_value={
            "metric": "capture_rate",
            "value": rate,
            "window_episodes": window,
            "selection": "maximum reported moving window",
            "source_content_hash": sha256_bytes(raw_bytes),
        },
        known_defects=tuple(known_defects),
        unknown_reasons={
            "code_revision": "the legacy log does not record the generating commit or dirty tree",
            "data_kind": "the log says OSM but contains no auditable map provenance",
            "map_hash": "the legacy log does not record the map content hash",
            "training_seed": "episode seeds do not establish an independent training seed",
            "created_at_utc": "log timestamps omit date and timezone",
        },
        independent_training_seeds=UNKNOWN,
        evaluation_episodes_per_condition=UNKNOWN,
        has_ci95=False,
        has_spatially_disjoint_split=False,
        same_training_distribution_moving_window=True,
        force_preliminary=(rate == 0.99 and window == 100),
    )


def register_prior_result(
    raw: PriorResultInput,
    evidence: EvidenceRegistry,
    claims: ClaimRegister,
    *,
    audited_at_utc: str,
) -> PriorResultRecord:
    result = build_prior_result(raw)
    evidence_id = f"prior-observation-{result.result_id}"
    evidence.register(EvidenceRecord(
        record_id=evidence_id,
        evidence_type=EvidenceType.EXPERIMENT_OBSERVATION,
        producer="research.prior-result-auditor",
        created_at_utc=audited_at_utc,
        method=result.method,
        source=result.source,
        extracted_value=result.as_dict(),
        verification="provenance_audited",
        limitations=tuple(result.classification_reasons),
    ))
    claims.register(ClaimRegisterEntry(
        claim_id=f"claim-{result.result_id}",
        claim_kind=ClaimKind.PRIOR_RESULT,
        statement="Legacy training-distribution moving-window capture-rate observation",
        status=result.status,
        evidence_ids=(evidence_id,),
    ))
    return result


@dataclass(frozen=True, slots=True)
class AuditReport(PersistedModel):
    code_revision: str
    audited_at_utc: str
    inventory: tuple[ImplementationInventory, ...]
    prior_results: tuple[PriorResultRecord, ...]
    evidence_records: tuple[EvidenceRecord, ...]
    claim_register: tuple[ClaimRegisterEntry, ...]
    completeness_passed: bool

    def __post_init__(self) -> None:
        _required(self.code_revision, "code_revision")
        _required(self.audited_at_utc, "audited_at_utc")
        for name in ("inventory", "prior_results", "evidence_records", "claim_register"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.inventory or not self.prior_results:
            raise ResearchValidationError(
                "INCOMPLETE_AUDIT_REPORT",
                "audit report requires inventory and prior results",
            )
        if not self.completeness_passed:
            raise ResearchValidationError(
                "AUDIT_COMPLETENESS_FAILED",
                "incomplete audit reports cannot be registered",
                path="completeness_passed",
            )
        super(AuditReport, self).__post_init__()


def audit_repository(
    root: Path,
    *,
    execute_tests: bool = True,
    assets: Sequence[AssetSpec] = DEFAULT_ASSETS,
    prior_defects: Sequence[KnownDefect] = (),
) -> AuditReport:
    root = root.resolve()
    audited_at = utc_now()
    revision = code_revision(root)
    test_results = run_asset_tests(root, assets) if execute_tests else MappingProxyType({})
    inventory = AssetAuditor(root, assets).inventory(revision, test_results)
    initial_evidence, initial_claims = inventory_evidence(
        inventory, test_results, created_at_utc=audited_at
    )
    evidence = EvidenceRegistry()
    for record in initial_evidence:
        evidence.register(record)
    claims = ClaimRegister(evidence)
    for entry in initial_claims:
        claims.register(entry)
    prior = register_prior_result(
        prior_from_training_log(root, known_defects=prior_defects),
        evidence,
        claims,
        audited_at_utc=audited_at,
    )
    return AuditReport(
        code_revision=revision,
        audited_at_utc=audited_at,
        inventory=inventory,
        prior_results=(prior,),
        evidence_records=evidence.records,
        claim_register=claims.entries,
        completeness_passed=True,
    )


__all__ = (
    "AssetAuditor",
    "AssetCategory",
    "AssetSpec",
    "AuditReport",
    "ClaimKind",
    "ClaimRegister",
    "ClaimRegisterEntry",
    "DEFAULT_ASSETS",
    "EvidenceRegistry",
    "ImplementationInventory",
    "KnownDefect",
    "PriorResultInput",
    "PriorResultRecord",
    "TestExecution",
    "audit_repository",
    "build_prior_result",
    "classify_prior_result",
    "code_revision",
    "inventory_evidence",
    "prior_from_training_log",
    "register_prior_result",
    "run_asset_tests",
)

"""Fail-closed preservation and path gates for the legacy ``best_v2`` artifact."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Iterable

from .domain import PersistedModel
from .errors import ResearchValidationError

# This is a prior design observation, never an expected value for a successful gate.
DESIGN_AUDIT_OBSERVATION_SHA256 = (
    "391527bb6914ca6c6e0d138472570e927330ba50354c87190e1647c22c016cd9"
)
_ALLOWED_MUTATIONS = frozenset({"write", "move", "delete"})
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _fail(code: str, message: str, *, path: Path | str | None = None, **details: object) -> None:
    raise ResearchValidationError(
        code,
        message,
        path=os.fspath(path) if path is not None else None,
        details=details,
    )


def _absolute(path: Path | str) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def resolve_real_path(path: Path | str, *, must_exist: bool = False) -> Path:
    """Resolve symlinks and existing parents, including for not-yet-created outputs."""
    candidate = _absolute(path)
    try:
        return candidate.resolve(strict=must_exist)
    except (OSError, RuntimeError) as exc:
        _fail("PATH_RESOLUTION_FAILED", "Path cannot be resolved safely", path=candidate, error=str(exc))


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.normpath(os.fspath(path)))


def paths_overlap(first: Path | str, second: Path | str) -> bool:
    """Return whether either resolved path is equal to or contains the other."""
    left = resolve_real_path(first)
    right = resolve_real_path(second)
    try:
        common = os.path.commonpath((_path_key(left), _path_key(right)))
    except ValueError:
        return False
    return common in {_path_key(left), _path_key(right)}


def _validate_sha256(value: str, field_name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        _fail("INVALID_SHA256", f"{field_name} must be a lowercase SHA-256 digest", path=field_name)


@dataclass(frozen=True, slots=True)
class ArtifactMeasurement(PersistedModel):
    requested_path: str
    real_path: str
    sha256: str
    byte_size: int
    measured_at_utc: str

    def __post_init__(self) -> None:
        if not self.requested_path or not self.real_path or not self.measured_at_utc:
            _fail("MISSING_ARTIFACT_MEASUREMENT", "Artifact measurement fields must be non-empty")
        _validate_sha256(self.sha256, "sha256")
        if self.byte_size < 0:
            _fail("INVALID_ARTIFACT_SIZE", "Artifact byte size cannot be negative", path="byte_size")
        super(ArtifactMeasurement, self).__post_init__()


@dataclass(frozen=True, slots=True)
class PreservedArtifact(PersistedModel):
    original: ArtifactMeasurement
    preserved_copy: ArtifactMeasurement
    preserved_copy_readonly: bool
    prior_audit_sha256: str
    prior_audit_matches_current: bool

    def __post_init__(self) -> None:
        _validate_sha256(self.prior_audit_sha256, "prior_audit_sha256")
        if self.original.sha256 != self.preserved_copy.sha256:
            _fail("PRESERVED_COPY_MISMATCH", "Preserved copy is not byte-identical to the original")
        if self.prior_audit_matches_current != (self.original.sha256 == self.prior_audit_sha256):
            _fail(
                "AUDIT_OBSERVATION_MISMATCH",
                "Prior audit comparison flag does not match the measured original hash",
            )
        if self.original.byte_size != self.preserved_copy.byte_size:
            _fail("PRESERVED_COPY_MISMATCH", "Preserved copy size differs from the original")
        if paths_overlap(self.original.real_path, self.preserved_copy.real_path):
            _fail("PROTECTED_PATH_OVERLAP", "Original and preserved copy paths must be separate")
        if not self.preserved_copy_readonly:
            _fail("PRESERVED_COPY_NOT_READONLY", "Preserved copy must be sealed read-only")
        super(PreservedArtifact, self).__post_init__()


@dataclass(frozen=True, slots=True)
class RunPaths:
    project_root: Path
    condition: str
    seed: int | str
    run_id: str
    input_paths: tuple[Path, ...] = ()
    output_paths: tuple[Path, ...] = ()
    temporary_paths: tuple[Path, ...] = ()
    checkpoint_outputs: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class PreservationGateReport(PersistedModel):
    artifact_record_hash: str
    original_sha256: str
    preserved_sha256: str
    byte_size: int
    verified_at_utc: str
    checked_path_count: int


def measure_artifact(path: Path | str) -> ArtifactMeasurement:
    requested = _absolute(path)
    real = resolve_real_path(requested, must_exist=True)
    if not real.is_file():
        _fail("ARTIFACT_NOT_REGULAR_FILE", "Protected artifact must be a regular file", path=requested)
    digest = hashlib.sha256()
    try:
        with real.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        size = real.stat().st_size
    except OSError as exc:
        _fail("ARTIFACT_READ_FAILED", "Protected artifact could not be measured", path=requested, error=str(exc))
    return ArtifactMeasurement(
        requested_path=os.fspath(requested),
        real_path=os.fspath(real),
        sha256=digest.hexdigest(),
        byte_size=size,
        measured_at_utc=_utc_now(),
    )


def is_readonly(path: Path | str) -> bool:
    real = resolve_real_path(path, must_exist=True)
    try:
        return not bool(real.stat().st_mode & _WRITE_BITS)
    except OSError as exc:
        _fail("ARTIFACT_STAT_FAILED", "Read-only state cannot be verified", path=real, error=str(exc))


def _same_measurement(first: ArtifactMeasurement, second: ArtifactMeasurement) -> bool:
    return first.sha256 == second.sha256 and first.byte_size == second.byte_size


def preserve_artifact(
    source: Path | str,
    destination: Path | str,
    *,
    prior_audit_sha256: str = DESIGN_AUDIT_OBSERVATION_SHA256,
) -> PreservedArtifact:
    """Create a new byte-identical copy without ever replacing an existing destination."""
    _validate_sha256(prior_audit_sha256, "prior_audit_sha256")
    source_before = measure_artifact(source)
    destination_path = _absolute(destination)
    destination_real = resolve_real_path(destination_path)
    if paths_overlap(source_before.real_path, destination_real):
        _fail("PROTECTED_PATH_OVERLAP", "Preservation destination overlaps the original", path=destination)
    if destination_path.exists() or destination_path.is_symlink():
        _fail(
            "PRESERVATION_DESTINATION_EXISTS",
            "Existing preservation destinations are never overwritten",
            path=destination_path,
        )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    destination_created = False
    completed = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination_path.parent, prefix=f".{destination_path.name}.",
            suffix=".tmp", delete=False,
        ) as output:
            temporary_name = output.name
            with Path(source_before.real_path).open("rb") as input_stream:
                for chunk in iter(lambda: input_stream.read(1024 * 1024), b""):
                    output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        temporary = Path(temporary_name)
        source_after = measure_artifact(source)
        staged = measure_artifact(temporary)
        if not _same_measurement(source_before, source_after) or not _same_measurement(source_before, staged):
            _fail("SOURCE_CHANGED_DURING_PRESERVATION", "Source identity changed while it was copied")
        try:
            os.link(temporary, destination_path)
            destination_created = True
        except OSError as exc:
            _fail(
                "PRESERVATION_SEAL_FAILED",
                "Could not atomically seal a non-overwriting preservation copy",
                path=destination_path,
                error=str(exc),
            )
        temporary.unlink()
        temporary_name = None
        try:
            current_mode = destination_path.stat().st_mode
            destination_path.chmod(current_mode & ~_WRITE_BITS)
        except OSError as exc:
            _fail(
                "PRESERVATION_SEAL_FAILED",
                "Could not set the preservation copy read-only",
                path=destination_path,
                error=str(exc),
            )
        preserved = measure_artifact(destination_path)
        readonly = is_readonly(destination_path)
        if not _same_measurement(source_before, preserved):
            _fail("PRESERVED_COPY_MISMATCH", "Sealed copy differs from the source", path=destination_path)
        if not readonly:
            _fail("PRESERVED_COPY_NOT_READONLY", "Sealed copy remains writable", path=destination_path)
        result = PreservedArtifact(
            original=source_before,
            preserved_copy=preserved,
            preserved_copy_readonly=readonly,
            prior_audit_sha256=prior_audit_sha256,
            prior_audit_matches_current=source_before.sha256 == prior_audit_sha256,
        )
        completed = True
        return result
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass
        if destination_created and not completed:
            try:
                destination_path.chmod(destination_path.stat().st_mode | stat.S_IWUSR)
                destination_path.unlink(missing_ok=True)
            except OSError:
                pass


def _verify_identity(label: str, registered: ArtifactMeasurement) -> ArtifactMeasurement:
    current = measure_artifact(registered.requested_path)
    if _path_key(Path(current.real_path)) != _path_key(Path(registered.real_path)):
        _fail(
            "PROTECTED_REALPATH_CHANGED",
            f"{label} real path changed after registration",
            path=registered.requested_path,
            registered=registered.real_path,
            current=current.real_path,
        )
    if not _same_measurement(current, registered):
        _fail(
            f"{label.upper()}_INTEGRITY_MISMATCH",
            f"{label} hash or byte size no longer matches its registered identity",
            path=registered.requested_path,
            expected_sha256=registered.sha256,
            actual_sha256=current.sha256,
            expected_size=registered.byte_size,
            actual_size=current.byte_size,
        )
    return current


def _safe_component(value: int | str, name: str) -> str:
    component = str(value)
    if component in {".", ".."} or not _SAFE_COMPONENT.fullmatch(component):
        _fail("INVALID_CHECKPOINT_COMPONENT", f"Unsafe checkpoint {name}", path=name, value=component)
    return component


def checkpoint_run_directory(
    project_root: Path | str, condition: str, seed: int | str, run_id: str
) -> Path:
    root = resolve_real_path(project_root)
    return root.joinpath(
        "artifacts", "research", "checkpoints",
        _safe_component(condition, "condition"),
        _safe_component(seed, "seed"),
        _safe_component(run_id, "run_id"),
    )


def validate_checkpoint_output(
    path: Path | str,
    *,
    project_root: Path | str,
    condition: str,
    seed: int | str,
    run_id: str,
) -> Path:
    allowed_requested = checkpoint_run_directory(project_root, condition, seed, run_id)
    candidate_requested = _absolute(path)
    allowed = resolve_real_path(allowed_requested)
    candidate = resolve_real_path(candidate_requested)
    try:
        candidate_requested.relative_to(allowed_requested)
        candidate.relative_to(allowed)
    except ValueError:
        if candidate_requested != allowed_requested or candidate != allowed:
            _fail(
                "CHECKPOINT_OUTPUT_OUTSIDE_RUN_DIRECTORY",
                "New checkpoints are restricted to the condition/seed/run directory",
                path=candidate_requested,
                allowed_directory=os.fspath(allowed_requested),
                resolved_path=os.fspath(candidate),
            )
    return candidate


def _assert_no_protected_overlap(
    paths: Iterable[Path | str], protected: tuple[Path, Path], category: str
) -> int:
    checked = 0
    for candidate in paths:
        resolved = resolve_real_path(candidate)
        checked += 1
        for protected_path in protected:
            if paths_overlap(resolved, protected_path):
                _fail(
                    "RUN_PATH_OVERLAPS_PROTECTED_ARTIFACT",
                    f"Run {category} path overlaps a protected artifact",
                    path=candidate,
                    category=category,
                    resolved_path=os.fspath(resolved),
                    protected_path=os.fspath(protected_path),
                )
    return checked


def verify_before_run(artifact: PreservedArtifact, paths: RunPaths) -> PreservationGateReport:
    """Re-measure identities and validate every path before any run output is created."""
    original = _verify_identity("original", artifact.original)
    preserved = _verify_identity("preserved", artifact.preserved_copy)
    if not is_readonly(artifact.preserved_copy.requested_path):
        _fail(
            "PRESERVED_COPY_NOT_READONLY",
            "Preserved copy lost its read-only seal",
            path=artifact.preserved_copy.requested_path,
        )
    protected = (Path(original.real_path), Path(preserved.real_path))
    if paths_overlap(*protected):
        _fail("PROTECTED_PATH_OVERLAP", "Original and preserved paths are no longer separate")
    checked = 0
    checked += _assert_no_protected_overlap(paths.input_paths, protected, "input")
    checked += _assert_no_protected_overlap(paths.output_paths, protected, "output")
    checked += _assert_no_protected_overlap(paths.temporary_paths, protected, "temporary")
    checked += _assert_no_protected_overlap(paths.checkpoint_outputs, protected, "checkpoint")
    for output in paths.checkpoint_outputs:
        validate_checkpoint_output(
            output,
            project_root=paths.project_root,
            condition=paths.condition,
            seed=paths.seed,
            run_id=paths.run_id,
        )
    return PreservationGateReport(
        artifact_record_hash=artifact.content_hash or "",
        original_sha256=original.sha256,
        preserved_sha256=preserved.sha256,
        byte_size=original.byte_size,
        verified_at_utc=_utc_now(),
        checked_path_count=checked,
    )


class ProtectedPathGuard:
    """Guard write/move/delete requests made by a research run."""

    def __init__(self, artifact: PreservedArtifact) -> None:
        self._protected = (
            Path(artifact.original.real_path),
            Path(artifact.preserved_copy.real_path),
        )

    def assert_allowed(self, operation: str, *paths: Path | str) -> None:
        if operation not in _ALLOWED_MUTATIONS:
            _fail(
                "UNKNOWN_MUTATION_OPERATION",
                "Unknown filesystem mutation is denied by default",
                operation=operation,
            )
        if not paths:
            _fail("MISSING_MUTATION_PATH", "Filesystem mutation requires at least one path")
        for candidate in paths:
            resolved = resolve_real_path(candidate)
            for protected in self._protected:
                if paths_overlap(resolved, protected):
                    _fail(
                        "PROTECTED_ARTIFACT_MUTATION_BLOCKED",
                        f"{operation} is blocked for protected artifact paths",
                        path=candidate,
                        operation=operation,
                        resolved_path=os.fspath(resolved),
                        protected_path=os.fspath(protected),
                    )

    def assert_write_allowed(self, path: Path | str) -> None:
        self.assert_allowed("write", path)

    def assert_move_allowed(self, source: Path | str, destination: Path | str) -> None:
        self.assert_allowed("move", source, destination)

    def assert_delete_allowed(self, path: Path | str) -> None:
        self.assert_allowed("delete", path)


__all__ = (
    "DESIGN_AUDIT_OBSERVATION_SHA256",
    "ArtifactMeasurement",
    "PreservedArtifact",
    "PreservationGateReport",
    "ProtectedPathGuard",
    "RunPaths",
    "checkpoint_run_directory",
    "is_readonly",
    "measure_artifact",
    "paths_overlap",
    "preserve_artifact",
    "resolve_real_path",
    "validate_checkpoint_output",
    "verify_before_run",
)

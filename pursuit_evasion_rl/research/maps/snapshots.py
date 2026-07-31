"""Offline-first immutable raw/network map snapshot adapters.

The default importer reads only repository cache or test-fixture paths.  It has
no transport dependency: a missing or unsealed snapshot becomes a structured
``not_run`` result rather than an attempted external request.  Live OSM
acquisition is isolated behind :class:`ExternalOSMFetchAdapter` and requires an
explicit opt-in on every invocation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping

import yaml

from pursuit_evasion_rl.osm_demo.models import (
    DOMAIN_SCHEMA_VERSION,
    Intersection,
    ModelNetwork,
    Segment,
)

from ..canonical import canonical_data, canonical_json, content_hash, sha256_bytes
from ..domain import ExecutionStatus, PersistedModel
from ..errors import ResearchValidationError
from .registry import ActualOSMProvenance

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
SNAPSHOT_SCHEMA_VERSION = "1.0"
DEFAULT_ALLOWED_ROOTS = (Path("cache"), Path("tests/fixtures"))


class SnapshotRole(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    ZERO_SHOT = "zero_shot"


def _required(value: Any, path: str) -> Any:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ResearchValidationError(
            "MISSING_REQUIRED_FIELD", f"{path} must be recorded", path=path
        )
    return value


def _hash_or_none(value: Any, path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ResearchValidationError(
            "INVALID_CONTENT_HASH", f"{path} must be a lowercase SHA-256 or null",
            path=path, actual=value,
        )
    return value


def _utc(value: str, path: str) -> str:
    _required(value, path)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ResearchValidationError(
            "INVALID_UTC_TIMESTAMP", f"{path} must be ISO-8601 UTC", path=path,
            actual=value,
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ResearchValidationError(
            "INVALID_UTC_TIMESTAMP", f"{path} must include UTC offset", path=path,
            actual=value,
        )
    return value


def _role(value: str | SnapshotRole) -> SnapshotRole:
    try:
        return value if isinstance(value, SnapshotRole) else SnapshotRole(value)
    except ValueError as exc:
        raise ResearchValidationError(
            "INVALID_SNAPSHOT_ROLE", "role must be train, validation, test, or zero_shot",
            path="role", actual=value,
        ) from exc


@dataclass(frozen=True, slots=True)
class SnapshotSpec(PersistedModel):
    map_id: str
    city: str
    role: SnapshotRole
    query: str
    query_geometry: Mapping[str, Any]
    source: str
    service: str
    service_version: str
    acquired_at_utc: str
    preprocessing_version: str
    metric_crs: str
    raw_path: str
    network_path: str
    raw_file_hash: str | None
    network_file_hash: str | None
    network_content_hash: str | None
    polygon_content_hash: str

    def __post_init__(self) -> None:
        for name in (
            "map_id", "city", "query", "source", "service", "service_version",
            "acquired_at_utc", "preprocessing_version", "metric_crs", "raw_path",
            "network_path", "polygon_content_hash",
        ):
            _required(getattr(self, name), name)
        object.__setattr__(self, "role", _role(self.role))
        _utc(self.acquired_at_utc, "acquired_at_utc")
        if not isinstance(self.query_geometry, Mapping) or not self.query_geometry:
            raise ResearchValidationError(
                "MISSING_REQUIRED_FIELD", "query_geometry must be a non-empty mapping",
                path="query_geometry",
            )
        if any(token in self.metric_crs.casefold().replace(" ", "") for token in ("4326", "crs84")):
            raise ResearchValidationError(
                "NON_METRIC_CRS", "metric_crs must be projected", path="metric_crs",
                actual=self.metric_crs,
            )
        for name in (
            "raw_file_hash", "network_file_hash", "network_content_hash",
            "polygon_content_hash",
        ):
            object.__setattr__(self, name, _hash_or_none(getattr(self, name), name))
        geometry = canonical_data(dict(self.query_geometry))
        if content_hash(geometry) != self.polygon_content_hash:
            raise ResearchValidationError(
                "POLYGON_HASH_MISMATCH", "polygon_content_hash does not match query_geometry",
                path="polygon_content_hash", expected=content_hash(geometry),
                actual=self.polygon_content_hash,
            )
        object.__setattr__(self, "query_geometry", geometry)
        super(SnapshotSpec, self).__post_init__()

    @property
    def sealed(self) -> bool:
        return all((self.raw_file_hash, self.network_file_hash, self.network_content_hash))

    @property
    def provenance_gaps(self) -> tuple[str, ...]:
        """Return metadata fields that cannot support an Actual OSM label."""
        gaps: list[str] = []
        for name in ("source", "service", "service_version", "preprocessing_version"):
            value = getattr(self, name).strip().casefold()
            if any(marker in value for marker in ("pending", "unknown", "unavailable")):
                gaps.append(name)
        acquired = datetime.fromisoformat(self.acquired_at_utc.replace("Z", "+00:00"))
        if acquired == datetime(1970, 1, 1, tzinfo=timezone.utc):
            gaps.append("acquired_at_utc")
        return tuple(gaps)


@dataclass(frozen=True, slots=True)
class ImportedSnapshot(PersistedModel):
    map_id: str
    city: str
    role: SnapshotRole
    raw_file_hash: str
    network_file_hash: str
    network_content_hash: str
    spec_content_hash: str
    network: ModelNetwork
    actual_osm_provenance: ActualOSMProvenance

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _role(self.role))
        for name in ("raw_file_hash", "network_file_hash", "network_content_hash", "spec_content_hash"):
            _hash_or_none(getattr(self, name), name)
        if self.actual_osm_provenance.network_content_hash != self.network_content_hash:
            raise ResearchValidationError(
                "PROVENANCE_MISMATCH", "registry provenance must identify the imported network",
                path="actual_osm_provenance.network_content_hash",
                expected=self.network_content_hash,
                actual=self.actual_osm_provenance.network_content_hash,
            )
        super(ImportedSnapshot, self).__post_init__()


@dataclass(frozen=True, slots=True)
class SnapshotImportResult(PersistedModel):
    map_id: str
    status: ExecutionStatus
    reason_code: str
    reason: str
    snapshot: ImportedSnapshot | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _required(self.map_id, "map_id")
        _required(self.reason_code, "reason_code")
        _required(self.reason, "reason")
        if not isinstance(self.status, ExecutionStatus):
            object.__setattr__(self, "status", ExecutionStatus(self.status))
        if self.status is ExecutionStatus.COMPLETED and self.snapshot is None:
            raise ResearchValidationError(
                "MISSING_REQUIRED_FIELD", "completed imports require a snapshot",
                path="snapshot",
            )
        if self.status is not ExecutionStatus.COMPLETED and self.snapshot is not None:
            raise ResearchValidationError(
                "STATUS_PAYLOAD_MISMATCH", "non-completed imports cannot expose snapshot data",
                path="snapshot",
            )
        object.__setattr__(self, "details", canonical_data(dict(self.details)))
        super(SnapshotImportResult, self).__post_init__()


def _points(values: Any) -> tuple[tuple[float, float], ...]:
    return tuple((float(point[0]), float(point[1])) for point in values)


def _network_from_data(data: Mapping[str, Any]) -> ModelNetwork:
    try:
        intersections = tuple(
            Intersection(
                id=item["id"],
                position_xy=(float(item["position_xy"][0]), float(item["position_xy"][1])),
                source_signature=item["source_signature"],
                boundary_kind=item.get("boundary_kind"),
                virtual=bool(item.get("virtual", False)),
                outgoing_segment_ids=tuple(item.get("outgoing_segment_ids", ())),
                incoming_segment_ids=tuple(item.get("incoming_segment_ids", ())),
            )
            for item in data["intersections"]
        )
        segments = tuple(
            Segment(
                id=item["id"], start_id=item["start_id"], end_id=item["end_id"],
                length_m=item["length_m"], geometry_xy=_points(item["geometry_xy"]),
                source_edge_refs=tuple(item.get("source_edge_refs", ())),
                attributes=dict(item.get("attributes", {})),
                virtual=bool(item.get("virtual", False)),
                crosses_boundary=bool(item.get("crosses_boundary", False)),
            )
            for item in data["segments"]
        )
        return ModelNetwork(
            intersections=intersections, segments=segments,
            schema_version=data.get("schema_version", DOMAIN_SCHEMA_VERSION),
        )
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ResearchValidationError(
            "INVALID_NETWORK_SNAPSHOT", "network snapshot failed schema validation",
            actual=str(exc),
        ) from exc


def load_snapshot_spec(path: str | Path) -> SnapshotSpec:
    config_path = Path(path)
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ResearchValidationError(
            "SNAPSHOT_CONFIG_MISSING", "snapshot config does not exist",
            path=str(config_path),
        ) from exc
    except yaml.YAMLError as exc:
        raise ResearchValidationError(
            "INVALID_SNAPSHOT_CONFIG", "snapshot config is not valid YAML",
            path=str(config_path), actual=str(exc),
        ) from exc
    if not isinstance(data, Mapping):
        raise ResearchValidationError(
            "INVALID_SNAPSHOT_CONFIG", "snapshot config root must be a mapping",
            path=str(config_path),
        )
    if data.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ResearchValidationError(
            "UNSUPPORTED_SCHEMA_VERSION", "unsupported snapshot config schema",
            path="schema_version", expected=SNAPSHOT_SCHEMA_VERSION,
            actual=data.get("schema_version"),
        )
    fields = dict(data)
    fields.pop("schema_version", None)
    return SnapshotSpec(**fields)


def load_snapshot_registry(path: str | Path, *, repository_root: str | Path) -> tuple[SnapshotSpec, ...]:
    """Load the committed registry without permitting path traversal or loose entries."""
    registry_path = Path(path)
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ResearchValidationError(
            "INVALID_SNAPSHOT_REGISTRY", "snapshot registry cannot be read",
            path=str(registry_path), actual=str(exc),
        ) from exc
    if not isinstance(data, Mapping) or data.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ResearchValidationError(
            "INVALID_SNAPSHOT_REGISTRY", "registry has an unsupported schema",
            path="schema_version", expected=SNAPSHOT_SCHEMA_VERSION,
            actual=data.get("schema_version") if isinstance(data, Mapping) else None,
        )
    entries = data.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ResearchValidationError(
            "INVALID_SNAPSHOT_REGISTRY", "registry entries must be a non-empty list",
            path="entries",
        )
    root = Path(repository_root).resolve()
    config_root = (root / "configs/research/maps").resolve()
    config_paths: list[Path] = []
    for index, item in enumerate(entries):
        if not isinstance(item, Mapping) or set(item) != {"config_path"}:
            raise ResearchValidationError(
                "INVALID_SNAPSHOT_REGISTRY_ENTRY",
                "each registry entry must contain only config_path",
                path=f"entries[{index}]", actual=item,
            )
        relative = item["config_path"]
        if not isinstance(relative, str) or not re.fullmatch(
            r"configs/research/maps/[^/\\]+\.yaml", relative
        ):
            raise ResearchValidationError(
                "INVALID_SNAPSHOT_REGISTRY_ENTRY",
                "config_path must name a direct YAML child of configs/research/maps",
                path=f"entries[{index}].config_path", actual=relative,
            )
        resolved = (root / relative).resolve()
        if resolved.parent != config_root:
            raise ResearchValidationError(
                "SNAPSHOT_CONFIG_PATH_ESCAPE", "config_path escapes the map config root",
                path=f"entries[{index}].config_path", actual=relative,
            )
        config_paths.append(resolved)
    if len(config_paths) != len(set(config_paths)):
        raise ResearchValidationError(
            "DUPLICATE_IDENTIFIER", "registry config paths must be unique", path="entries"
        )
    specs = tuple(load_snapshot_spec(config_path) for config_path in config_paths)
    ids = [spec.map_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise ResearchValidationError(
            "DUPLICATE_IDENTIFIER", "snapshot map_id values must be unique", path="entries"
        )
    daejeon = [spec for spec in specs if spec.city.casefold() == "daejeon"]
    required = {SnapshotRole.TRAIN, SnapshotRole.VALIDATION, SnapshotRole.TEST}
    daejeon_roles = [spec.role for spec in daejeon]
    if set(daejeon_roles) != required or len(daejeon_roles) != len(required):
        raise ResearchValidationError(
            "INCOMPLETE_DAEJEON_SPLIT",
            "Daejeon requires exactly one train, validation, and test snapshot",
            expected=sorted(role.value for role in required),
            actual=sorted(role.value for role in daejeon_roles),
        )
    zero_cities = {
        spec.city.casefold() for spec in specs
        if spec.role is SnapshotRole.ZERO_SHOT and spec.city.casefold() != "daejeon"
    }
    if len(zero_cities) < 2:
        raise ResearchValidationError(
            "INSUFFICIENT_ZERO_SHOT_CITIES", "at least two non-training zero-shot cities are required",
            expected=2, actual=len(zero_cities),
        )
    return specs


class OfflineSnapshotStore:
    """Verified importer that cannot perform network I/O by construction."""

    def __init__(
        self, repository_root: str | Path, *,
        allowed_roots: tuple[str | Path, ...] = DEFAULT_ALLOWED_ROOTS,
    ) -> None:
        self.root = Path(repository_root).resolve()
        self.allowed_roots = tuple((self.root / Path(item)).resolve() for item in allowed_roots)

    def _resolve(self, relative: str) -> Path:
        candidate = (self.root / relative).resolve()
        if not any(candidate == base or base in candidate.parents for base in self.allowed_roots):
            raise ResearchValidationError(
                "SNAPSHOT_PATH_OUTSIDE_OFFLINE_ROOT",
                "default snapshot imports are restricted to cache/ or test fixtures",
                path=relative,
            )
        return candidate

    def import_snapshot(self, spec: SnapshotSpec) -> SnapshotImportResult:
        raw_path = self._resolve(spec.raw_path)
        network_path = self._resolve(spec.network_path)
        missing = [str(path.relative_to(self.root)) for path in (raw_path, network_path) if not path.is_file()]
        if missing:
            return SnapshotImportResult(
                map_id=spec.map_id, status=ExecutionStatus.NOT_RUN,
                reason_code="SNAPSHOT_CACHE_MISS",
                reason="required immutable snapshot files are absent; no external request was attempted",
                details={"missing_paths": missing, "external_requests": 0},
            )
        if not spec.sealed:
            return SnapshotImportResult(
                map_id=spec.map_id, status=ExecutionStatus.NOT_RUN,
                reason_code="SNAPSHOT_UNSEALED",
                reason="snapshot hashes are not sealed; no external request was attempted",
                details={"external_requests": 0},
            )
        if spec.provenance_gaps:
            return SnapshotImportResult(
                map_id=spec.map_id, status=ExecutionStatus.NOT_RUN,
                reason_code="SNAPSHOT_PROVENANCE_INCOMPLETE",
                reason="snapshot metadata cannot support an Actual OSM label; no external request was attempted",
                details={
                    "incomplete_fields": list(spec.provenance_gaps),
                    "external_requests": 0,
                },
            )
        raw_bytes, network_bytes = raw_path.read_bytes(), network_path.read_bytes()
        actual_raw, actual_file = sha256_bytes(raw_bytes), sha256_bytes(network_bytes)
        mismatches = {}
        if actual_raw != spec.raw_file_hash:
            mismatches["raw_file_hash"] = {"expected": spec.raw_file_hash, "actual": actual_raw}
        if actual_file != spec.network_file_hash:
            mismatches["network_file_hash"] = {"expected": spec.network_file_hash, "actual": actual_file}
        if mismatches:
            return SnapshotImportResult(
                map_id=spec.map_id, status=ExecutionStatus.FAILED,
                reason_code="SNAPSHOT_HASH_MISMATCH", reason="snapshot byte hash verification failed",
                details={"mismatches": mismatches, "external_requests": 0},
            )
        try:
            json.loads(raw_bytes.decode("utf-8"))
            network_data = json.loads(network_bytes.decode("utf-8"))
            network = _network_from_data(network_data)
        except (UnicodeDecodeError, json.JSONDecodeError, ResearchValidationError) as exc:
            return SnapshotImportResult(
                map_id=spec.map_id, status=ExecutionStatus.FAILED,
                reason_code="SNAPSHOT_SCHEMA_INVALID", reason="snapshot JSON/schema validation failed",
                details={"error": str(exc), "external_requests": 0},
            )
        actual_content = content_hash(network)
        if actual_content != spec.network_content_hash:
            return SnapshotImportResult(
                map_id=spec.map_id, status=ExecutionStatus.FAILED,
                reason_code="SNAPSHOT_HASH_MISMATCH", reason="network semantic content hash verification failed",
                details={
                    "mismatches": {"network_content_hash": {
                        "expected": spec.network_content_hash, "actual": actual_content,
                    }},
                    "external_requests": 0,
                },
            )
        source_edge_ids = tuple(sorted({
            edge for segment in network.segments for edge in segment.source_edge_refs
        }))
        try:
            registry_provenance = ActualOSMProvenance(
                query_geometry=spec.query_geometry,
                source=(
                    f"{spec.source}; service={spec.service}; "
                    f"service_version={spec.service_version}; query={spec.query}"
                ),
                acquired_at_utc=spec.acquired_at_utc,
                raw_content_hash=actual_raw,
                preprocessing_version=spec.preprocessing_version,
                metric_crs=spec.metric_crs,
                network_content_hash=actual_content,
                source_edge_ids=source_edge_ids,
            )
        except ResearchValidationError as exc:
            return SnapshotImportResult(
                map_id=spec.map_id, status=ExecutionStatus.FAILED,
                reason_code="SNAPSHOT_PROVENANCE_INVALID",
                reason="verified bytes cannot satisfy the existing map registry provenance contract",
                details={"error": exc.as_dict(), "external_requests": 0},
            )
        imported = ImportedSnapshot(
            map_id=spec.map_id, city=spec.city, role=spec.role,
            raw_file_hash=actual_raw, network_file_hash=actual_file,
            network_content_hash=actual_content, spec_content_hash=spec.content_hash,
            network=network, actual_osm_provenance=registry_provenance,
        )
        return SnapshotImportResult(
            map_id=spec.map_id, status=ExecutionStatus.COMPLETED,
            reason_code="SNAPSHOT_VERIFIED", reason="immutable offline snapshot verified",
            snapshot=imported, details={"external_requests": 0},
        )


@dataclass(frozen=True, slots=True)
class ExternalFetchRequest(PersistedModel):
    integration_run_id: str
    query: str
    query_geometry: Mapping[str, Any]
    service: str
    service_version: str
    preprocessing_version: str
    query_geometry_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for name in (
            "integration_run_id", "query", "service", "service_version",
            "preprocessing_version",
        ):
            _required(getattr(self, name), name)
        if not isinstance(self.query_geometry, Mapping) or not self.query_geometry:
            raise ResearchValidationError(
                "MISSING_REQUIRED_FIELD", "query_geometry must be a non-empty mapping",
                path="query_geometry",
            )
        geometry = canonical_data(dict(self.query_geometry))
        object.__setattr__(self, "query_geometry", geometry)
        object.__setattr__(self, "query_geometry_hash", content_hash(geometry))
        super(ExternalFetchRequest, self).__post_init__()


@dataclass(frozen=True, slots=True)
class ExternalFetchProvenance(PersistedModel):
    integration_run_id: str
    query: str
    query_geometry: Mapping[str, Any]
    query_geometry_hash: str
    service: str
    service_version: str
    preprocessing_version: str
    executed_at_utc: str
    input_content_hash: str
    raw_output_hash: str
    network_output_hash: str
    network_content_hash: str
    output_content_hash: str

    def __post_init__(self) -> None:
        for name in (
            "integration_run_id", "query", "service", "service_version",
            "preprocessing_version",
        ):
            _required(getattr(self, name), name)
        _utc(self.executed_at_utc, "executed_at_utc")
        geometry = canonical_data(dict(self.query_geometry))
        if content_hash(geometry) != self.query_geometry_hash:
            raise ResearchValidationError(
                "POLYGON_HASH_MISMATCH", "query_geometry_hash does not match query_geometry",
                path="query_geometry_hash", expected=content_hash(geometry),
                actual=self.query_geometry_hash,
            )
        object.__setattr__(self, "query_geometry", geometry)
        for name in (
            "query_geometry_hash", "input_content_hash", "raw_output_hash",
            "network_output_hash", "network_content_hash", "output_content_hash",
        ):
            _hash_or_none(getattr(self, name), name)
        expected_input = ExternalFetchRequest(
            integration_run_id=self.integration_run_id,
            query=self.query,
            query_geometry=geometry,
            service=self.service,
            service_version=self.service_version,
            preprocessing_version=self.preprocessing_version,
        ).content_hash
        if self.input_content_hash != expected_input:
            raise ResearchValidationError(
                "FETCH_INPUT_HASH_MISMATCH",
                "input_content_hash does not identify the external fetch request",
                path="input_content_hash", expected=expected_input,
                actual=self.input_content_hash,
            )
        expected_output = content_hash({
            "raw_output_hash": self.raw_output_hash,
            "network_output_hash": self.network_output_hash,
            "network_content_hash": self.network_content_hash,
        })
        if self.output_content_hash != expected_output:
            raise ResearchValidationError(
                "FETCH_OUTPUT_HASH_MISMATCH",
                "output_content_hash does not identify all external fetch outputs",
                path="output_content_hash", expected=expected_output,
                actual=self.output_content_hash,
            )
        super(ExternalFetchProvenance, self).__post_init__()


class ExternalOSMFetchAdapter:
    """Explicit integration-only transport boundary; never used by offline imports."""

    def __init__(
        self, transport: Callable[[ExternalFetchRequest], tuple[bytes, bytes]]
    ) -> None:
        self._transport = transport

    def fetch(
        self, request: ExternalFetchRequest, *, opt_in: bool = False,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> tuple[bytes, bytes, ExternalFetchProvenance]:
        if not opt_in:
            raise ResearchValidationError(
                "EXTERNAL_FETCH_OPT_IN_REQUIRED",
                "external OSM integration requires explicit opt_in=True",
                path="opt_in", expected=True, actual=False,
            )
        raw_bytes, network_bytes = self._transport(request)
        if not isinstance(raw_bytes, bytes) or not isinstance(network_bytes, bytes):
            raise ResearchValidationError(
                "INVALID_EXTERNAL_FETCH_OUTPUT", "transport must return raw and network bytes"
            )
        try:
            json.loads(raw_bytes.decode("utf-8"))
            network = _network_from_data(json.loads(network_bytes.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError, ResearchValidationError) as exc:
            raise ResearchValidationError(
                "INVALID_EXTERNAL_FETCH_OUTPUT",
                "external transport returned invalid raw or network JSON",
                actual=str(exc),
            ) from exc
        raw_hash = sha256_bytes(raw_bytes)
        network_file_hash = sha256_bytes(network_bytes)
        network_semantic_hash = content_hash(network)
        output_hash = content_hash({
            "raw_output_hash": raw_hash,
            "network_output_hash": network_file_hash,
            "network_content_hash": network_semantic_hash,
        })
        executed_at = clock()
        if executed_at.tzinfo is None or executed_at.utcoffset() is None:
            raise ResearchValidationError(
                "INVALID_UTC_TIMESTAMP",
                "external fetch clock must return a timezone-aware datetime",
                path="clock",
            )
        provenance = ExternalFetchProvenance(
            integration_run_id=request.integration_run_id,
            query=request.query, query_geometry=request.query_geometry,
            query_geometry_hash=request.query_geometry_hash,
            service=request.service, service_version=request.service_version,
            preprocessing_version=request.preprocessing_version,
            executed_at_utc=executed_at.astimezone(timezone.utc).isoformat(),
            input_content_hash=request.content_hash,
            raw_output_hash=raw_hash,
            network_output_hash=network_file_hash,
            network_content_hash=network_semantic_hash,
            output_content_hash=output_hash,
        )
        return raw_bytes, network_bytes, provenance


def write_immutable_snapshot(path: str | Path, payload: bytes) -> str:
    """Publish integration output once; refusing overwrite preserves identity."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise ResearchValidationError(
            "IMMUTABLE_SNAPSHOT_EXISTS", "snapshot output already exists",
            path=str(destination),
        ) from exc
    return sha256_bytes(payload)


__all__ = (
    "DEFAULT_ALLOWED_ROOTS", "ExternalFetchProvenance", "ExternalFetchRequest",
    "ExternalOSMFetchAdapter", "ImportedSnapshot", "OfflineSnapshotStore",
    "SNAPSHOT_SCHEMA_VERSION", "SnapshotImportResult", "SnapshotRole", "SnapshotSpec",
    "load_snapshot_registry", "load_snapshot_spec", "write_immutable_snapshot",
)

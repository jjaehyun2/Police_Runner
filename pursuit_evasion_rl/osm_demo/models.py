"""Immutable, versioned domain models for the OSM road-pursuit demo."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Sequence

DOMAIN_SCHEMA_VERSION = "1.0"
CONFIG_VERSION = "1.0"
SUPPORTED_SCHEMA_VERSIONS = frozenset({DOMAIN_SCHEMA_VERSION})
SUPPORTED_CONFIG_VERSIONS = frozenset({CONFIG_VERSION})
POLICE_COUNT = 6
EARTH_RADIUS_KM = 6371.0088


class DomainValidationError(ValueError):
    """A stable, machine-readable domain validation failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        path: str | None = None,
        expected: Any = None,
        actual: Any = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.path = path
        self.expected = expected
        self.actual = actual

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "path": self.path,
            "expected": self.expected,
            "actual": self.actual,
        }


def validate_schema_version(
    version: str,
    *,
    path: str = "schema_version",
    supported: frozenset[str] = SUPPORTED_SCHEMA_VERSIONS,
) -> None:
    """Reject unsupported persisted/input schemas with the supported list."""
    if version not in supported:
        supported_list = sorted(supported)
        raise DomainValidationError(
            "UNSUPPORTED_SCHEMA_VERSION",
            f"Unsupported schema version {version!r}; supported versions: {supported_list}",
            path=path,
            expected=supported_list,
            actual=version,
        )


def validate_config_version(
    version: str,
    *,
    path: str = "config_version",
    supported: frozenset[str] = SUPPORTED_CONFIG_VERSIONS,
) -> None:
    """Reject unsupported domain configuration versions with the supported list."""
    if version not in supported:
        supported_list = sorted(supported)
        raise DomainValidationError(
            "UNSUPPORTED_CONFIG_VERSION",
            f"Unsupported config version {version!r}; supported versions: {supported_list}",
            path=path,
            expected=supported_list,
            actual=version,
        )


def _finite(value: float, path: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise DomainValidationError(
            "NON_FINITE_NUMBER", f"{path} must be a finite number", path=path, actual=value
        ) from exc
    if not math.isfinite(numeric):
        raise DomainValidationError(
            "NON_FINITE_NUMBER", f"{path} must be a finite number", path=path, actual=value
        )
    return numeric


def _freeze(value: Any) -> Any:
    """Recursively detach mutable containers from immutable domain values."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


def _freeze_field(instance: Any, name: str) -> None:
    object.__setattr__(instance, name, _freeze(getattr(instance, name)))


def _machine_value(value: Any) -> Any:
    """Convert frozen domain values to JSON-compatible primitives."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _machine_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_machine_value(item) for item in value]
    return value


def _validate_point(point: Sequence[float], path: str) -> tuple[float, float]:
    if len(point) != 2:
        raise DomainValidationError(
            "INVALID_POSITION", f"{path} must contain exactly two coordinates", path=path, actual=point
        )
    return (_finite(point[0], f"{path}[0]"), _finite(point[1], f"{path}[1]"))


@dataclass(frozen=True, slots=True)
class BoundedArea:
    name: str
    north: float
    south: float
    east: float
    west: float
    max_area_km2: float
    config_version: str = CONFIG_VERSION
    schema_version: str = DOMAIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version)
        validate_config_version(self.config_version)
        if not self.name.strip():
            raise DomainValidationError("EMPTY_NAME", "Bounded area name must not be empty", path="name")
        north = _finite(self.north, "north")
        south = _finite(self.south, "south")
        east = _finite(self.east, "east")
        west = _finite(self.west, "west")
        maximum = _finite(self.max_area_km2, "max_area_km2")
        for name, value, lower, upper in (
            ("north", north, -90.0, 90.0),
            ("south", south, -90.0, 90.0),
            ("east", east, -180.0, 180.0),
            ("west", west, -180.0, 180.0),
        ):
            if not lower <= value <= upper:
                raise DomainValidationError(
                    "COORDINATE_OUT_OF_RANGE",
                    f"{name} must be in [{lower}, {upper}]",
                    path=name,
                    expected=[lower, upper],
                    actual=value,
                )
        if north <= south:
            raise DomainValidationError(
                "INVALID_BBOX_ORDER", "north must be greater than south", path="north", actual=north
            )
        if east <= west:
            raise DomainValidationError(
                "INVALID_BBOX_ORDER", "east must be greater than west", path="east", actual=east
            )
        if maximum <= 0.0:
            raise DomainValidationError(
                "INVALID_AREA_LIMIT", "max_area_km2 must be positive", path="max_area_km2", actual=maximum
            )
        for field_name, value in (("north", north), ("south", south), ("east", east), ("west", west), ("max_area_km2", maximum)):
            object.__setattr__(self, field_name, value)
        if self.area_km2 > maximum:
            raise DomainValidationError(
                "BBOX_AREA_EXCEEDED",
                f"Bounded area is {self.area_km2:.6f} km²; maximum allowed is {maximum:.6f} km²",
                path="max_area_km2",
                expected=maximum,
                actual=self.area_km2,
            )

    @property
    def area_km2(self) -> float:
        north_rad = math.radians(self.north)
        south_rad = math.radians(self.south)
        longitude_span = math.radians(self.east - self.west)
        return EARTH_RADIUS_KM**2 * longitude_span * abs(math.sin(north_rad) - math.sin(south_rad))


@dataclass(frozen=True, slots=True)
class RawNode:
    source_id: str
    lat: float
    lon: float
    x_m: float
    y_m: float
    tags: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.source_id):
            raise DomainValidationError("EMPTY_SOURCE_ID", "Raw node source_id must not be empty")
        for name in ("lat", "lon", "x_m", "y_m"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if not -90.0 <= self.lat <= 90.0 or not -180.0 <= self.lon <= 180.0:
            raise DomainValidationError("COORDINATE_OUT_OF_RANGE", "Raw node latitude/longitude is invalid")
        _freeze_field(self, "tags")


@dataclass(frozen=True, slots=True)
class RawEdge:
    source_u: str
    source_v: str
    source_key: str
    length_m: float
    geometry_xy: tuple[tuple[float, float], ...]
    way_refs: tuple[str, ...] = ()
    oneway: bool = False
    road_class: str = ""
    tags: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        length = _finite(self.length_m, "length_m")
        if length < 0.0:
            raise DomainValidationError("NEGATIVE_LENGTH", "Raw edge length must be nonnegative")
        object.__setattr__(self, "length_m", length)
        geometry = tuple(_validate_point(point, "geometry_xy") for point in self.geometry_xy)
        if len(geometry) < 2:
            raise DomainValidationError("INVALID_GEOMETRY", "Raw edge geometry requires at least two points")
        object.__setattr__(self, "geometry_xy", geometry)
        object.__setattr__(self, "way_refs", tuple(str(item) for item in self.way_refs))
        _freeze_field(self, "tags")


@dataclass(frozen=True, slots=True)
class RawOSMGraph:
    nodes: tuple[RawNode, ...]
    edges: tuple[RawEdge, ...]
    schema_version: str = DOMAIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version)
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "edges", tuple(self.edges))
        if not self.nodes or not self.edges:
            raise DomainValidationError("EMPTY_NETWORK", "Raw OSM graph requires nodes and directed edges")
        node_ids = [node.source_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise DomainValidationError("DUPLICATE_ID", "Raw node source IDs must be unique")
        known = set(node_ids)
        if any(edge.source_u not in known or edge.source_v not in known for edge in self.edges):
            raise DomainValidationError("INVALID_EDGE_ENDPOINT", "Raw edge references an unknown source node")


@dataclass(frozen=True, slots=True)
class Intersection:
    id: int
    position_xy: tuple[float, float]
    source_signature: str
    boundary_kind: str | None = None
    virtual: bool = False
    outgoing_segment_ids: tuple[int, ...] = ()
    incoming_segment_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.id < 0:
            raise DomainValidationError("INVALID_CANONICAL_ID", "Intersection ID must be nonnegative")
        object.__setattr__(self, "position_xy", _validate_point(self.position_xy, "position_xy"))
        if not self.source_signature:
            raise DomainValidationError("EMPTY_SIGNATURE", "Intersection source signature must not be empty")
        object.__setattr__(self, "outgoing_segment_ids", tuple(self.outgoing_segment_ids))
        object.__setattr__(self, "incoming_segment_ids", tuple(self.incoming_segment_ids))


@dataclass(frozen=True, slots=True)
class Segment:
    id: int
    start_id: int
    end_id: int
    length_m: float
    geometry_xy: tuple[tuple[float, float], ...]
    source_edge_refs: tuple[str, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)
    virtual: bool = False
    crosses_boundary: bool = False

    def __post_init__(self) -> None:
        if min(self.id, self.start_id, self.end_id) < 0:
            raise DomainValidationError("INVALID_CANONICAL_ID", "Segment and endpoint IDs must be nonnegative")
        length = _finite(self.length_m, "length_m")
        if length < 0.0:
            raise DomainValidationError("NEGATIVE_LENGTH", "Segment length must be nonnegative")
        object.__setattr__(self, "length_m", length)
        geometry = tuple(_validate_point(point, "geometry_xy") for point in self.geometry_xy)
        if len(geometry) < 2:
            raise DomainValidationError("INVALID_GEOMETRY", "Segment geometry requires at least two points")
        object.__setattr__(self, "geometry_xy", geometry)
        object.__setattr__(self, "source_edge_refs", tuple(str(item) for item in self.source_edge_refs))
        _freeze_field(self, "attributes")


@dataclass(frozen=True, slots=True)
class ModelNetwork:
    intersections: tuple[Intersection, ...]
    segments: tuple[Segment, ...]
    schema_version: str = DOMAIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version)
        object.__setattr__(self, "intersections", tuple(self.intersections))
        object.__setattr__(self, "segments", tuple(self.segments))
        if not self.intersections or not self.segments:
            raise DomainValidationError(
                "EMPTY_NETWORK", "Model network requires at least one intersection and one segment"
            )
        intersection_ids = [item.id for item in self.intersections]
        segment_ids = [item.id for item in self.segments]
        if intersection_ids != list(range(len(intersection_ids))):
            raise DomainValidationError(
                "NONCONTIGUOUS_CANONICAL_IDS", "Intersection IDs must be unique, sorted, and contiguous"
            )
        if segment_ids != list(range(len(segment_ids))):
            raise DomainValidationError(
                "NONCONTIGUOUS_CANONICAL_IDS", "Segment IDs must be unique, sorted, and contiguous"
            )
        known_intersections = set(intersection_ids)
        known_segments = set(segment_ids)
        for segment in self.segments:
            if segment.start_id not in known_intersections or segment.end_id not in known_intersections:
                raise DomainValidationError(
                    "INVALID_SEGMENT_ENDPOINT",
                    f"Segment {segment.id} references an unknown intersection",
                    actual=[segment.start_id, segment.end_id],
                )
        for intersection in self.intersections:
            referenced_segments = set(intersection.outgoing_segment_ids) | set(intersection.incoming_segment_ids)
            if not referenced_segments <= known_segments:
                raise DomainValidationError(
                    "INVALID_SEGMENT_REFERENCE",
                    f"Intersection {intersection.id} references an unknown segment",
                )
            expected_outgoing = {item.id for item in self.segments if item.start_id == intersection.id}
            expected_incoming = {item.id for item in self.segments if item.end_id == intersection.id}
            if set(intersection.outgoing_segment_ids) != expected_outgoing or set(intersection.incoming_segment_ids) != expected_incoming:
                raise DomainValidationError(
                    "INCONSISTENT_ADJACENCY", f"Intersection {intersection.id} adjacency does not match segments"
                )

    @property
    def intersection_ids(self) -> frozenset[int]:
        return frozenset(item.id for item in self.intersections)

    @property
    def segment_ids(self) -> frozenset[int]:
        return frozenset(item.id for item in self.segments)

    @property
    def drivable_position_count(self) -> int:
        connected = {
            item.id
            for item in self.intersections
            if item.outgoing_segment_ids or item.incoming_segment_ids
        }
        physical_segments = sum(not item.virtual for item in self.segments)
        return len(connected) + physical_segments


@dataclass(frozen=True, slots=True)
class MappingManifest:
    coarsener_version: str
    raw_node_to_intersection: Mapping[str, int | None]
    segment_sources: Mapping[str, tuple[str, ...]]
    virtual_mappings: Mapping[str, Any]
    canonical_rules: str
    action_ordering: str
    schema_version: str = DOMAIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version)
        for name in ("raw_node_to_intersection", "segment_sources", "virtual_mappings"):
            _freeze_field(self, name)


@dataclass(frozen=True, slots=True)
class NetworkMetadata:
    area: BoundedArea
    source: str
    acquired_at: str
    network_type: str
    metric_crs: str
    schema_versions: Mapping[str, str]
    settings_hash: str
    artifact_hashes: Mapping[str, str] = field(default_factory=dict)
    statistics: Mapping[str, Any] = field(default_factory=dict)
    reachability_digest: str | None = None
    validation: Mapping[str, Any] = field(default_factory=dict)
    operation_status: str = "READY"
    schema_version: str = DOMAIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version)
        if self.network_type != "drive":
            raise DomainValidationError("INVALID_NETWORK_TYPE", "OSM network_type must be 'drive'")
        for artifact, version in self.schema_versions.items():
            validate_schema_version(version, path=f"schema_versions.{artifact}")
        for name in ("schema_versions", "artifact_hashes", "statistics", "validation"):
            _freeze_field(self, name)


@dataclass(frozen=True, slots=True)
class ArtifactDescriptor:
    name: str
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class BundleManifest:
    cache_key: str
    artifacts: tuple[ArtifactDescriptor, ...]
    attribution: str
    committed_at: str
    schema_version: str = DOMAIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version)
        object.__setattr__(self, "artifacts", tuple(self.artifacts))


class ClaimStatus(str, Enum):
    IMPLEMENTED = "implemented"
    VERIFIED = "verified"
    EXPERIMENTAL = "experimental"
    UNSUPPORTED = "unsupported"


class CheckStatus(str, Enum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ObservationContract:
    profile_id: str
    fields: tuple[str, ...]
    dtype: str
    padding_value: float
    normalization: Mapping[str, Any]
    action_ordering: str
    field_sizes: tuple[int, ...] = ()
    schema_version: str = DOMAIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version)
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "field_sizes", tuple(self.field_sizes))
        if self.field_sizes and (
            len(self.field_sizes) != len(self.fields)
            or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in self.field_sizes)
        ):
            raise DomainValidationError(
                "INVALID_OBSERVATION_FIELD_SIZES",
                "Observation field sizes must be positive integers aligned with fields",
                expected=len(self.fields),
                actual=self.field_sizes,
            )
        object.__setattr__(self, "padding_value", _finite(self.padding_value, "padding_value"))
        _freeze_field(self, "normalization")


@dataclass(frozen=True, slots=True)
class ArchitectureSpec:
    observation_dim: int
    hidden_dims: tuple[int, ...]
    action_dim: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden_dims", tuple(self.hidden_dims))
        if self.observation_dim <= 0 or self.action_dim <= 0 or any(item <= 0 for item in self.hidden_dims):
            raise DomainValidationError("INVALID_ARCHITECTURE", "Architecture dimensions must be positive")


@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    checkpoint_hash: str
    architecture: ArchitectureSpec
    police_count: int
    observation_contract: ObservationContract
    training_networks: tuple[str, ...]
    split_manifest_hash: str | None
    lineage: Mapping[str, Any]
    code_version: str
    claim_status: ClaimStatus = ClaimStatus.EXPERIMENTAL
    schema_version: str = DOMAIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version)
        if self.police_count != POLICE_COUNT:
            raise DomainValidationError(
                "INVALID_POLICE_COUNT",
                f"Checkpoint manifest must declare exactly {POLICE_COUNT} police",
                expected=POLICE_COUNT,
                actual=self.police_count,
            )
        object.__setattr__(self, "training_networks", tuple(self.training_networks))
        _freeze_field(self, "lineage")


@dataclass(frozen=True, slots=True)
class InferredCheckpointContract:
    actor_key: str
    layer_shapes: tuple[tuple[int, ...], ...]
    observation_dim: int | None
    hidden_dims: tuple[int, ...]
    action_dim: int | None
    inferable: frozenset[str]
    unknown: frozenset[str]
    schema_version: str = DOMAIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version)
        object.__setattr__(self, "layer_shapes", tuple(tuple(shape) for shape in self.layer_shapes))
        object.__setattr__(self, "hidden_dims", tuple(self.hidden_dims))
        object.__setattr__(self, "inferable", frozenset(self.inferable))
        object.__setattr__(self, "unknown", frozenset(self.unknown))


@dataclass(frozen=True, slots=True)
class CompatibilityCheck:
    code: str
    status: CheckStatus
    expected: Any
    actual: Any
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "expected", _freeze(self.expected))
        object.__setattr__(self, "actual", _freeze(self.actual))

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "status": self.status.value,
            "expected": _machine_value(self.expected),
            "actual": _machine_value(self.actual),
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    report_id: str
    checkpoint_hash: str
    network_hash: str
    structure: tuple[CompatibilityCheck, ...]
    semantics: tuple[CompatibilityCheck, ...]
    network: tuple[CompatibilityCheck, ...]
    execution: tuple[CompatibilityCheck, ...]
    overall: CheckStatus
    schema_version: str = DOMAIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version)
        for name in ("structure", "semantics", "network", "execution"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def as_dict(self) -> dict[str, Any]:
        sections = {
            name: [check.as_dict() for check in getattr(self, name)]
            for name in ("structure", "semantics", "network", "execution")
        }
        mismatches = [
            {"section": name, **check.as_dict()}
            for name in sections
            for check in getattr(self, name)
            if check.status in {CheckStatus.FAIL, CheckStatus.WARNING}
        ]
        blocking_sections = (self.structure, self.network, self.execution)
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "checkpoint_hash": self.checkpoint_hash,
            "network_hash": self.network_hash,
            "overall": self.overall.value,
            "inference_blocked": any(
                check.status is CheckStatus.FAIL
                for section in blocking_sections
                for check in section
            ),
            "verified_osm_inference_allowed": all(
                check.status is CheckStatus.PASS
                for section in (self.structure, self.semantics, self.network, self.execution)
                for check in section
            ),
            "sections": sections,
            "mismatches": mismatches,
        }


@dataclass(frozen=True, slots=True)
class VehiclePlacement:
    segment_id: int | None = None
    progress: float = 0.0
    intersection_id: int | None = None

    def __post_init__(self) -> None:
        if (self.segment_id is None) == (self.intersection_id is None):
            raise DomainValidationError(
                "INVALID_POSITION",
                "Vehicle position must identify exactly one segment or intersection",
            )
        progress = _finite(self.progress, "progress")
        if self.segment_id is not None and not 0.0 <= progress <= 1.0:
            raise DomainValidationError(
                "INVALID_POSITION", "Segment progress must be in [0, 1]", path="progress", actual=progress
            )
        if self.segment_id is None and progress != 0.0:
            raise DomainValidationError(
                "INVALID_POSITION", "Intersection positions must use progress 0", path="progress", actual=progress
            )
        if self.segment_id is not None and self.segment_id < 0:
            raise DomainValidationError("INVALID_POSITION", "Segment ID must be nonnegative")
        if self.intersection_id is not None and self.intersection_id < 0:
            raise DomainValidationError("INVALID_POSITION", "Intersection ID must be nonnegative")
        object.__setattr__(self, "progress", progress)

    @property
    def identity(self) -> tuple[str, int, float]:
        if self.segment_id is not None:
            return ("segment", self.segment_id, self.progress)
        return ("intersection", int(self.intersection_id), 0.0)


@dataclass(frozen=True, slots=True)
class EpisodeConfig:
    dt_s: float
    police_speed_mps: float
    fugitive_speed_mps: float
    capture_radius_m: float
    max_steps: int
    deterministic: bool = True
    schema_version: str = DOMAIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version)
        for name in ("dt_s", "police_speed_mps", "fugitive_speed_mps", "capture_radius_m"):
            value = _finite(getattr(self, name), name)
            if value <= 0.0:
                raise DomainValidationError("INVALID_EPISODE_CONFIG", f"{name} must be positive", path=name)
            object.__setattr__(self, name, value)
        if self.max_steps <= 0:
            raise DomainValidationError("INVALID_EPISODE_CONFIG", "max_steps must be positive")


class EpisodeOutcome(str, Enum):
    CAPTURE = "capture"
    ESCAPE = "escape"
    TIMEOUT = "timeout"


@dataclass(frozen=True, slots=True)
class EpisodeState:
    step: int
    simulated_s: float
    police: tuple[VehiclePlacement, ...]
    fugitive: VehiclePlacement
    incoming_headings: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "police", tuple(self.police))
        if len(self.police) != POLICE_COUNT:
            raise DomainValidationError(
                "INVALID_POLICE_COUNT",
                f"Episode state requires exactly {POLICE_COUNT} police positions",
                expected=POLICE_COUNT,
                actual=len(self.police),
            )
        if self.step < 0 or _finite(self.simulated_s, "simulated_s") < 0.0:
            raise DomainValidationError("INVALID_EPISODE_STATE", "Step and simulated time must be nonnegative")
        for agent, heading in self.incoming_headings.items():
            _finite(heading, f"incoming_headings.{agent}")
        _freeze_field(self, "incoming_headings")


@dataclass(frozen=True, slots=True)
class EpisodeTransition:
    step: int
    actions: tuple[int, ...]
    state: EpisodeState
    events: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(self, "events", tuple(self.events))


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    run_id: str
    episode_id: str
    seed: int
    initial_state: EpisodeState
    transitions: tuple[EpisodeTransition, ...]
    outcome: EpisodeOutcome
    terminal_priority: str
    hashes: Mapping[str, str]
    schema_version: str = DOMAIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version)
        object.__setattr__(self, "transitions", tuple(self.transitions))
        _freeze_field(self, "hashes")


@dataclass(frozen=True, slots=True)
class EpisodeBatchRequest:
    network: ModelNetwork
    checkpoint_ref: str
    checkpoint_manifest: CheckpointManifest
    config: EpisodeConfig
    episode_count: int
    seeds: tuple[int, ...]
    police: tuple[VehiclePlacement, ...] | None = None
    fugitive: VehiclePlacement | None = None
    schema_version: str = DOMAIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version)
        object.__setattr__(self, "seeds", tuple(self.seeds))
        if self.episode_count <= 0:
            raise DomainValidationError("INVALID_EPISODE_COUNT", "episode_count must be positive")
        if len(self.seeds) != self.episode_count:
            raise DomainValidationError(
                "SEED_COUNT_MISMATCH",
                "One deterministic seed is required per episode",
                expected=self.episode_count,
                actual=len(self.seeds),
            )
        if not self.checkpoint_ref:
            raise DomainValidationError("MISSING_CHECKPOINT", "checkpoint_ref must not be empty")
        if (self.police is None) != (self.fugitive is None):
            raise DomainValidationError(
                "INCOMPLETE_PLACEMENT", "Explicit placement requires police and fugitive positions"
            )
        if self.police is not None:
            object.__setattr__(self, "police", tuple(self.police))
            validate_vehicle_placements(self.network, self.police, self.fugitive)


@dataclass(frozen=True, slots=True)
class LatencySample:
    value_ms: float
    status: str
    device: str
    warmup: bool

    def __post_init__(self) -> None:
        value = _finite(self.value_ms, "value_ms")
        if value < 0.0 and not (value == -1.0 and self.status == "not_measured"):
            raise DomainValidationError("INVALID_LATENCY", "Latency must be nonnegative or -1/not_measured")
        object.__setattr__(self, "value_ms", value)


@dataclass(frozen=True, slots=True)
class MetricsSummary:
    outcomes: Mapping[str, int]
    rates: Mapping[str, float]
    confidence_intervals: Mapping[str, tuple[float, float]]
    episode_lengths: tuple[int, ...]
    latency: Mapping[str, Any]
    preliminary: bool = True
    schema_version: str = DOMAIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version)
        expected = {item.value for item in EpisodeOutcome}
        if set(self.outcomes) != expected or any(value < 0 for value in self.outcomes.values()):
            raise DomainValidationError("INVALID_OUTCOME_COUNTS", "Metrics require nonnegative counts for all outcomes")
        if sum(self.outcomes.values()) != len(self.episode_lengths):
            raise DomainValidationError("OUTCOME_COUNT_MISMATCH", "Outcome counts must equal completed episodes")
        for name in ("outcomes", "rates", "confidence_intervals", "latency"):
            _freeze_field(self, name)
        object.__setattr__(self, "episode_lengths", tuple(self.episode_lengths))


@dataclass(frozen=True, slots=True)
class ClaimRecord:
    name: str
    status: ClaimStatus
    evidence_ids: tuple[str, ...]
    reason: str
    updated_at: str
    schema_version: str = DOMAIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version)
        if not self.name.strip() or not self.reason.strip() or not self.updated_at.strip():
            raise DomainValidationError(
                "MISSING_CLAIM_FIELD", "Claim name, reason, and updated_at are required"
            )
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise DomainValidationError(
                "DUPLICATE_EVIDENCE", "Claim evidence identifiers must be unique"
            )


class RunMode(str, Enum):
    LEGACY_DIRECT = "legacy-direct"
    OSM_FINETUNE = "osm-finetune"
    OSM_FROM_SCRATCH = "osm-from-scratch"
    OSM_EVALUATION = "osm-evaluation"
    BASELINE = "baseline"


@dataclass(frozen=True, slots=True)
class RunLineage:
    """Immutable training/evaluation provenance with disjoint geographic splits."""

    split_manifest_hash: str
    train_network_hashes: frozenset[str]
    validation_network_hashes: frozenset[str]
    test_network_hashes: frozenset[str]
    code_version: str
    parent_checkpoint_hash: str | None = None
    parent_run_id: str | None = None
    schema_version: str = DOMAIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version)
        for name in ("split_manifest_hash", "code_version"):
            if not str(getattr(self, name)).strip():
                raise DomainValidationError(
                    "INCOMPLETE_RUN_LINEAGE", f"{name} must be nonempty", path=name
                )
        for name in ("train_network_hashes", "validation_network_hashes", "test_network_hashes"):
            values = frozenset(str(value) for value in getattr(self, name))
            if not values or any(not value.strip() for value in values):
                raise DomainValidationError(
                    "INCOMPLETE_REGION_SPLITS", f"{name} must contain nonempty network hashes", path=name
                )
            object.__setattr__(self, name, values)
        overlaps = {
            "train_validation": sorted(self.train_network_hashes & self.validation_network_hashes),
            "train_test": sorted(self.train_network_hashes & self.test_network_hashes),
            "validation_test": sorted(self.validation_network_hashes & self.test_network_hashes),
        }
        conflicts = {name: values for name, values in overlaps.items() if values}
        if conflicts:
            raise DomainValidationError(
                "OVERLAPPING_REGION_SPLITS",
                "Run lineage train, validation, and test hashes must be pairwise disjoint",
                actual=conflicts,
            )
        for name in ("parent_checkpoint_hash", "parent_run_id"):
            value = getattr(self, name)
            if value is not None and not str(value).strip():
                raise DomainValidationError(
                    "INCOMPLETE_RUN_LINEAGE", f"{name} cannot be blank", path=name
                )


@dataclass(frozen=True, slots=True)
class RunManifest:
    run_id: str
    mode: RunMode
    network_hash: str
    observation_profile: str
    seeds: tuple[int, ...]
    policy_kind: str
    claim_status: ClaimStatus
    checkpoint_hash: str | None = None
    plan_id: str | None = None
    artifacts: Mapping[str, str] = field(default_factory=dict)
    run_config: Mapping[str, Any] = field(default_factory=dict)
    lineage: RunLineage | None = None
    schema_version: str = DOMAIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version)
        object.__setattr__(self, "seeds", tuple(self.seeds))
        _freeze_field(self, "artifacts")
        _freeze_field(self, "run_config")
        required = {
            "run_id": self.run_id,
            "network_hash": self.network_hash,
            "observation_profile": self.observation_profile,
            "policy_kind": self.policy_kind,
        }
        empty = [name for name, value in required.items() if not str(value).strip()]
        if empty:
            raise DomainValidationError(
                "INCOMPLETE_RUN_MANIFEST", f"Run manifest fields must be nonempty: {empty}"
            )
        if not self.seeds:
            raise DomainValidationError("MISSING_SEEDS", "Run manifest requires deterministic seeds")
        if self.mode is RunMode.LEGACY_DIRECT:
            if self.claim_status is not ClaimStatus.EXPERIMENTAL:
                raise DomainValidationError(
                    "LEGACY_EVIDENCE_EXCLUDED",
                    "Legacy-direct runs must remain experimental and cannot be verified OSM evidence",
                )
            if not self.checkpoint_hash:
                raise DomainValidationError(
                    "MISSING_CHECKPOINT", "Legacy-direct runs require a checkpoint hash"
                )
        if self.mode in {RunMode.OSM_FINETUNE, RunMode.OSM_FROM_SCRATCH}:
            if self.claim_status is not ClaimStatus.EXPERIMENTAL:
                raise DomainValidationError(
                    "TRAINING_RUN_NOT_EVIDENCE",
                    "Training completion alone remains experimental; use a planned OSM evaluation",
                )
            if not self.checkpoint_hash:
                raise DomainValidationError(
                    "MISSING_CHECKPOINT", "OSM training runs require an output checkpoint hash"
                )
            if self.observation_profile != "osm_topology_v1":
                raise DomainValidationError(
                    "INVALID_TRAINING_PROFILE", "OSM training requires osm_topology_v1"
                )
            if self.lineage is None:
                raise DomainValidationError(
                    "MISSING_RUN_LINEAGE", "OSM training runs require split and code lineage"
                )
            if self.mode is RunMode.OSM_FINETUNE and not self.lineage.parent_checkpoint_hash:
                raise DomainValidationError(
                    "MISSING_PARENT_CHECKPOINT", "Fine-tune runs require a parent checkpoint hash"
                )
            if self.mode is RunMode.OSM_FROM_SCRATCH and self.lineage.parent_checkpoint_hash is not None:
                raise DomainValidationError(
                    "UNEXPECTED_PARENT_CHECKPOINT", "From-scratch runs cannot use parent weights"
                )
        if self.mode is RunMode.OSM_EVALUATION:
            if self.claim_status is not ClaimStatus.EXPERIMENTAL:
                raise DomainValidationError(
                    "EVALUATION_NOT_AUTOVERIFIED",
                    "Evaluation runs remain experimental until evidence-backed claim promotion",
                )
            if not self.plan_id:
                raise DomainValidationError(
                    "MISSING_EXPERIMENT_PLAN", "OSM evaluation runs require a pre-registered plan"
                )
            if not self.checkpoint_hash:
                raise DomainValidationError(
                    "MISSING_CHECKPOINT", "OSM evaluation runs require a checkpoint hash"
                )
            if self.observation_profile != "osm_topology_v1":
                raise DomainValidationError(
                    "INVALID_EVALUATION_PROFILE", "OSM evaluation requires osm_topology_v1"
                )
            if self.lineage is None:
                raise DomainValidationError(
                    "MISSING_RUN_LINEAGE", "OSM evaluation runs require training/test split lineage"
                )
        if self.mode is RunMode.BASELINE and self.checkpoint_hash is not None:
            raise DomainValidationError(
                "UNEXPECTED_CHECKPOINT", "Baseline runs must not claim a learned checkpoint"
            )


@dataclass(frozen=True, slots=True)
class ExportFile:
    relative_path: str
    sha256: str
    size: int
    media_type: str


@dataclass(frozen=True, slots=True)
class ExportManifest:
    export_id: str
    input_hashes: Mapping[str, str]
    files: tuple[ExportFile, ...]
    dimensions: tuple[int, int]
    code_version: str
    claims: tuple[ClaimRecord, ...]
    attribution: str
    warnings: tuple[str, ...] = ()
    schema_version: str = DOMAIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version)
        _freeze_field(self, "input_hashes")
        object.__setattr__(self, "files", tuple(self.files))
        object.__setattr__(self, "claims", tuple(self.claims))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "dimensions", tuple(self.dimensions))


def validate_vehicle_placement(network: ModelNetwork, placement: VehiclePlacement, path: str) -> None:
    if placement.segment_id is not None:
        segment = next(
            (item for item in network.segments if item.id == placement.segment_id),
            None,
        )
        if segment is None:
            raise DomainValidationError(
                "INVALID_VEHICLE_POSITION",
                f"{path} references unknown segment {placement.segment_id}",
                path=f"{path}.segment_id",
                actual=placement.segment_id,
            )
        if segment.virtual:
            raise DomainValidationError(
                "NON_DRIVABLE_VEHICLE_POSITION",
                f"{path} references virtual segment {placement.segment_id}",
                path=f"{path}.segment_id",
                actual=placement.segment_id,
            )
    if placement.intersection_id is not None:
        intersection = next(
            (item for item in network.intersections if item.id == placement.intersection_id),
            None,
        )
        if intersection is None:
            raise DomainValidationError(
                "INVALID_VEHICLE_POSITION",
                f"{path} references unknown intersection {placement.intersection_id}",
                path=f"{path}.intersection_id",
                actual=placement.intersection_id,
            )
        if intersection.virtual or not (
            intersection.outgoing_segment_ids or intersection.incoming_segment_ids
        ):
            raise DomainValidationError(
                "NON_DRIVABLE_VEHICLE_POSITION",
                f"{path} references a non-drivable intersection {placement.intersection_id}",
                path=f"{path}.intersection_id",
                actual=placement.intersection_id,
            )


def validate_vehicle_placements(
    network: ModelNetwork,
    police: Sequence[VehiclePlacement],
    fugitive: VehiclePlacement | None,
    *,
    require_distinct: bool = True,
) -> None:
    """Validate the complete six-police/one-fugitive placement contract."""
    if len(police) != POLICE_COUNT:
        raise DomainValidationError(
            "INVALID_POLICE_COUNT",
            f"Exactly {POLICE_COUNT} police positions are required",
            path="police",
            expected=POLICE_COUNT,
            actual=len(police),
        )
    if fugitive is None:
        raise DomainValidationError("MISSING_FUGITIVE", "One fugitive position is required", path="fugitive")
    if network.drivable_position_count < POLICE_COUNT + 1:
        raise DomainValidationError(
            "INSUFFICIENT_DRIVABLE_POSITIONS",
            "Model network cannot place six police and one fugitive on distinct drivable positions",
            expected=POLICE_COUNT + 1,
            actual=network.drivable_position_count,
        )
    all_placements = tuple(police) + (fugitive,)
    for index, placement in enumerate(police):
        validate_vehicle_placement(network, placement, f"police[{index}]")
    validate_vehicle_placement(network, fugitive, "fugitive")
    if require_distinct and len({placement.identity for placement in all_placements}) != len(all_placements):
        raise DomainValidationError(
            "DUPLICATE_VEHICLE_POSITION", "Police and fugitive positions must be distinct"
        )

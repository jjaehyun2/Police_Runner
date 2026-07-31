"""Content-addressed spatial split construction and fail-closed leakage gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import re
from typing import Generic, Iterable, Literal, Sequence, TypeAlias, TypeVar

from shapely.geometry import LineString, Point, Polygon

from pursuit_evasion_rl.osm_demo.models import ModelNetwork

from ..canonical import content_hash
from ..domain import DataKind, PersistedModel, exactly_one
from ..errors import ErrorRecord, ResearchValidationError
from .registry import ActualOSMProvenance, RegisteredMap

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SplitScope(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    CROSS_CITY = "cross_city"


class HandleKind(str, Enum):
    MAP = "map"
    EPISODE = "episode"
    METRIC = "metric"


def _required(value: str, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResearchValidationError("MISSING_REQUIRED_FIELD", f"{path} must be non-empty", path=path)
    return value


def _hash(value: str, path: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ResearchValidationError(
            "INVALID_CONTENT_HASH", f"{path} must be a lowercase SHA-256", path=path, actual=value
        )
    return value


def _point(value: Sequence[float], path: str) -> tuple[float, float]:
    if len(value) != 2:
        raise ResearchValidationError("INVALID_GEOMETRY", f"{path} must have two coordinates", path=path)
    coordinates = (float(value[0]), float(value[1]))
    if not all(math.isfinite(item) for item in coordinates):
        raise ResearchValidationError("NON_FINITE_NUMBER", f"{path} must be finite", path=path)
    return coordinates


def _metric_crs(value: str) -> str:
    normalized = _required(value, "metric_crs").replace(" ", "").casefold()
    if any(token in normalized for token in ("epsg:4326", "crs84", "wgs84")):
        raise ResearchValidationError("NON_METRIC_CRS", "metric_crs must be projected and metric", path="metric_crs")
    return value


@dataclass(frozen=True, slots=True)
class SplitProtocol(PersistedModel):
    metric_crs: str
    buffer_m: float

    def __post_init__(self) -> None:
        _metric_crs(self.metric_crs)
        distance = float(self.buffer_m)
        if not math.isfinite(distance) or distance <= 0.0:
            raise ResearchValidationError(
                "INVALID_SPLIT_BUFFER", "buffer_m must be finite and greater than zero",
                path="buffer_m", expected="> 0", actual=self.buffer_m,
            )
        object.__setattr__(self, "buffer_m", distance)
        super(SplitProtocol, self).__post_init__()


@dataclass(frozen=True, slots=True)
class MetricPolygon(PersistedModel):
    exterior: tuple[tuple[float, float], ...]
    holes: tuple[tuple[tuple[float, float], ...], ...] = ()
    geometry_hash: str = field(init=False)

    def __post_init__(self) -> None:
        exterior = tuple(_point(point, "exterior") for point in self.exterior)
        holes = tuple(tuple(_point(point, "holes") for point in ring) for ring in self.holes)
        geometry = Polygon(exterior, holes)
        if not geometry.is_valid or geometry.is_empty or geometry.area <= 0.0:
            raise ResearchValidationError("INVALID_POLYGON", "polygon must be valid and have positive area")
        normalized = geometry.normalize()
        canonical_exterior = tuple((float(x), float(y)) for x, y in normalized.exterior.coords)
        canonical_holes = tuple(
            tuple((float(x), float(y)) for x, y in ring.coords) for ring in normalized.interiors
        )
        object.__setattr__(self, "exterior", canonical_exterior)
        object.__setattr__(self, "holes", canonical_holes)
        object.__setattr__(
            self, "geometry_hash", content_hash({"exterior": canonical_exterior, "holes": canonical_holes})
        )
        super(MetricPolygon, self).__post_init__()

    def geometry(self) -> Polygon:
        return Polygon(self.exterior, self.holes)


@dataclass(frozen=True, slots=True)
class SplitNode:
    node_id: str
    position_xy: tuple[float, float]

    def __post_init__(self) -> None:
        _required(self.node_id, "node_id")
        object.__setattr__(self, "position_xy", _point(self.position_xy, "position_xy"))


@dataclass(frozen=True, slots=True)
class SplitEdge:
    edge_id: str
    geometry_xy: tuple[tuple[float, float], ...]
    source_edge_signatures: tuple[str, ...]
    virtual: bool = False

    def __post_init__(self) -> None:
        _required(self.edge_id, "edge_id")
        geometry = tuple(_point(point, "geometry_xy") for point in self.geometry_xy)
        if len(geometry) < 2:
            raise ResearchValidationError("INVALID_GEOMETRY", "edge geometry needs at least two points")
        signatures = tuple(sorted(str(item).strip() for item in self.source_edge_signatures))
        if any(not item for item in signatures) or len(signatures) != len(set(signatures)):
            raise ResearchValidationError(
                "INVALID_SOURCE_EDGE_SIGNATURE", "source edge signatures must be unique and non-empty"
            )
        if not signatures and not self.virtual:
            raise ResearchValidationError(
                "INVALID_SOURCE_EDGE_SIGNATURE", "physical edges require a source edge signature"
            )
        object.__setattr__(self, "geometry_xy", geometry)
        object.__setattr__(self, "source_edge_signatures", signatures)


@dataclass(frozen=True, slots=True)
class NetworkSlice(PersistedModel):
    map_id: str
    city_id: str
    metric_crs: str
    nodes: tuple[SplitNode, ...]
    edges: tuple[SplitEdge, ...]
    network_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _required(self.map_id, "map_id")
        _required(self.city_id, "city_id")
        _metric_crs(self.metric_crs)
        nodes, edges = tuple(self.nodes), tuple(self.edges)
        if not nodes or not edges:
            raise ResearchValidationError("EMPTY_NETWORK_SLICE", "a split network needs nodes and edges")
        if len({node.node_id for node in nodes}) != len(nodes) or len({edge.edge_id for edge in edges}) != len(edges):
            raise ResearchValidationError("DUPLICATE_IDENTIFIER", "split node and edge identifiers must be unique")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(
            self, "network_content_hash",
            content_hash(
                {
                    "map_id": self.map_id,
                    "city_id": self.city_id,
                    "metric_crs": self.metric_crs,
                    "nodes": nodes,
                    "edges": edges,
                }
            ),
        )
        super(NetworkSlice, self).__post_init__()

    @property
    def source_edge_signatures(self) -> frozenset[str]:
        return frozenset(signature for edge in self.edges for signature in edge.source_edge_signatures)


@dataclass(frozen=True, slots=True)
class SpatialPartition(PersistedModel):
    scope: SplitScope
    registered_map: RegisteredMap
    polygon: MetricPolygon
    network: NetworkSlice
    polygon_hash: str
    network_hash: str
    excluded_node_ids: tuple[str, ...] = ()
    excluded_edge_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        scope = exactly_one(self.scope, SplitScope, path="scope")
        if scope is SplitScope.CROSS_CITY:
            raise ResearchValidationError("INVALID_SPLIT_SCOPE", "cross-city maps are not polygon split partitions")
        if not isinstance(self.registered_map, RegisteredMap):
            raise ResearchValidationError(
                "SPLIT_REGISTRY_RECORD_REQUIRED",
                "spatial partitions require a task 2.1 RegisteredMap",
                path="registered_map",
            )
        if (
            self.registered_map.data_kind is not DataKind.ACTUAL_OSM_MAP
            or not isinstance(self.registered_map.provenance, ActualOSMProvenance)
        ):
            raise ResearchValidationError(
                "SPLIT_REQUIRES_ACTUAL_OSM",
                "OSM generalization partitions require registered Actual OSM provenance",
                path="registered_map.data_kind",
                actual=self.registered_map.data_kind.value,
            )
        if self.network.map_id != self.registered_map.map_id:
            raise ResearchValidationError(
                "SPLIT_REGISTRY_MISMATCH",
                "network slice map_id must match its registered map",
                path="network.map_id",
                expected=self.registered_map.map_id,
                actual=self.network.map_id,
            )
        unknown_sources = sorted(
            self.network.source_edge_signatures
            - set(self.registered_map.provenance.source_edge_ids)
        )
        if unknown_sources:
            raise ResearchValidationError(
                "SPLIT_SOURCE_EDGE_PROVENANCE_MISMATCH",
                "split source-edge signatures must belong to the registered OSM lineage",
                path="network.source_edge_signatures",
                expected="subset of registered_map.provenance.source_edge_ids",
                actual=unknown_sources,
            )
        object.__setattr__(self, "scope", scope)
        _hash(self.polygon_hash, "polygon_hash")
        _hash(self.network_hash, "network_hash")
        excluded_nodes = tuple(self.excluded_node_ids)
        excluded_edges = tuple(self.excluded_edge_ids)
        if len(excluded_nodes) != len(set(excluded_nodes)) or len(excluded_edges) != len(set(excluded_edges)):
            raise ResearchValidationError(
                "DUPLICATE_IDENTIFIER", "excluded split element identifiers must be unique"
            )
        included_nodes = {node.node_id for node in self.network.nodes}
        included_edges = {edge.edge_id for edge in self.network.edges}
        if included_nodes.intersection(excluded_nodes) or included_edges.intersection(excluded_edges):
            raise ResearchValidationError(
                "EXCLUDED_ELEMENT_PRESENT", "excluded elements must not remain in the split network"
            )
        object.__setattr__(self, "excluded_node_ids", tuple(sorted(excluded_nodes)))
        object.__setattr__(self, "excluded_edge_ids", tuple(sorted(excluded_edges)))
        super(SpatialPartition, self).__post_init__()


def build_spatial_partition(
    *,
    scope: SplitScope,
    polygon: MetricPolygon,
    network: ModelNetwork,
    registered_map: RegisteredMap,
    city_id: str,
    protocol: SplitProtocol,
) -> SpatialPartition:
    """Build one registered OSM partition, excluding all buffer-touching elements."""
    if not isinstance(network, ModelNetwork):
        raise ResearchValidationError("INVALID_NETWORK", "network must be a ModelNetwork", path="network")
    if not isinstance(registered_map, RegisteredMap):
        raise ResearchValidationError(
            "SPLIT_REGISTRY_RECORD_REQUIRED",
            "build_spatial_partition requires a task 2.1 RegisteredMap",
            path="registered_map",
        )
    if (
        registered_map.data_kind is not DataKind.ACTUAL_OSM_MAP
        or not isinstance(registered_map.provenance, ActualOSMProvenance)
    ):
        raise ResearchValidationError(
            "SPLIT_REQUIRES_ACTUAL_OSM",
            "OSM generalization partitions require registered Actual OSM provenance",
            path="registered_map.data_kind",
            actual=registered_map.data_kind.value,
        )
    actual_network_hash = content_hash(network)
    if actual_network_hash != registered_map.provenance.network_content_hash:
        raise ResearchValidationError(
            "SPLIT_REGISTRY_NETWORK_MISMATCH",
            "network bytes do not match the registered OSM network provenance",
            path="network",
            expected=registered_map.provenance.network_content_hash,
            actual=actual_network_hash,
        )
    if registered_map.provenance.metric_crs != protocol.metric_crs:
        raise ResearchValidationError(
            "SPLIT_CRS_MISMATCH",
            "registered map CRS must match the split protocol metric CRS",
            path="registered_map.provenance.metric_crs",
            expected=protocol.metric_crs,
            actual=registered_map.provenance.metric_crs,
        )
    scope = exactly_one(scope, SplitScope, path="scope")
    if scope is SplitScope.CROSS_CITY:
        raise ResearchValidationError("INVALID_SPLIT_SCOPE", "cross-city maps are not polygon split partitions")

    geometry = polygon.geometry()
    boundary_buffer = geometry.boundary.buffer(protocol.buffer_m)
    retained_node_ids = {
        node.id
        for node in network.intersections
        if geometry.covers(Point(node.position_xy))
        and not boundary_buffer.intersects(Point(node.position_xy))
    }
    retained_segments = tuple(
        segment
        for segment in network.segments
        if segment.start_id in retained_node_ids
        and segment.end_id in retained_node_ids
        and geometry.covers(LineString(segment.geometry_xy))
        and not boundary_buffer.intersects(LineString(segment.geometry_xy))
    )
    retained_segment_endpoints = {
        endpoint for segment in retained_segments for endpoint in (segment.start_id, segment.end_id)
    }
    retained_nodes = tuple(
        SplitNode(str(node.id), node.position_xy)
        for node in network.intersections
        if node.id in retained_segment_endpoints
    )
    retained_edges = tuple(
        SplitEdge(
            edge_id=str(segment.id),
            geometry_xy=segment.geometry_xy,
            source_edge_signatures=segment.source_edge_refs,
            virtual=segment.virtual,
        )
        for segment in retained_segments
    )
    if not retained_nodes or not retained_edges:
        raise ResearchValidationError(
            "EMPTY_NETWORK_SLICE",
            "buffer exclusion removed all connected map elements",
            path=scope.value,
        )
    network_slice = NetworkSlice(
        map_id=registered_map.map_id,
        city_id=city_id,
        metric_crs=protocol.metric_crs,
        nodes=retained_nodes,
        edges=retained_edges,
    )
    retained_node_labels = {node.node_id for node in retained_nodes}
    retained_edge_labels = {edge.edge_id for edge in retained_edges}
    return SpatialPartition(
        scope=scope,
        registered_map=registered_map,
        polygon=polygon,
        network=network_slice,
        polygon_hash=polygon.geometry_hash,
        network_hash=network_slice.network_content_hash,
        excluded_node_ids=tuple(
            str(node.id) for node in network.intersections if str(node.id) not in retained_node_labels
        ),
        excluded_edge_ids=tuple(
            str(edge.id) for edge in network.segments if str(edge.id) not in retained_edge_labels
        ),
    )


@dataclass(frozen=True, slots=True)
class CrossCityMapRef(PersistedModel):
    city_id: str
    registered_map: RegisteredMap
    frozen_policy_hash: str

    def __post_init__(self) -> None:
        _required(self.city_id, "city_id")
        _hash(self.frozen_policy_hash, "frozen_policy_hash")
        if not isinstance(self.registered_map, RegisteredMap):
            raise ResearchValidationError(
                "SPLIT_REGISTRY_RECORD_REQUIRED",
                "cross-city references require a task 2.1 RegisteredMap",
                path="registered_map",
            )
        if (
            self.registered_map.data_kind is not DataKind.ACTUAL_OSM_MAP
            or not isinstance(self.registered_map.provenance, ActualOSMProvenance)
        ):
            raise ResearchValidationError(
                "ZERO_SHOT_REQUIRES_ACTUAL_OSM", "cross-city targets must be Actual OSM maps"
            )
        super(CrossCityMapRef, self).__post_init__()

    @property
    def map_id(self) -> str:
        return self.registered_map.map_id

    @property
    def map_hash(self) -> str:
        return str(self.registered_map.content_hash)

    @property
    def data_kind(self) -> DataKind:
        return self.registered_map.data_kind


@dataclass(frozen=True, slots=True)
class SpatialSplit(PersistedModel):
    train: SpatialPartition
    validation: SpatialPartition
    test: SpatialPartition
    cross_city: tuple[CrossCityMapRef, ...]

    def __post_init__(self) -> None:
        expected = (("train", SplitScope.TRAIN), ("validation", SplitScope.VALIDATION), ("test", SplitScope.TEST))
        for name, scope in expected:
            if getattr(self, name).scope is not scope:
                raise ResearchValidationError(
                    "SPLIT_SCOPE_MISMATCH", f"{name} partition has the wrong scope",
                    path=name, expected=scope.value, actual=getattr(self, name).scope.value,
                )
        object.__setattr__(self, "cross_city", tuple(self.cross_city))
        super(SpatialSplit, self).__post_init__()


@dataclass(frozen=True, slots=True)
class SplitValidationReport(PersistedModel):
    split_hash: str
    protocol_hash: str
    eligible_for_generalization: bool
    polygon_hashes: tuple[tuple[str, str], ...]
    network_hashes: tuple[tuple[str, str], ...]
    errors: tuple[ErrorRecord, ...] = ()

    def __post_init__(self) -> None:
        _hash(self.split_hash, "split_hash")
        _hash(self.protocol_hash, "protocol_hash")
        errors = tuple(self.errors)
        if self.eligible_for_generalization == bool(errors):
            raise ResearchValidationError(
                "INVALID_GATE_REPORT", "eligible reports have no errors and failed reports have errors"
            )
        object.__setattr__(self, "errors", errors)
        super(SplitValidationReport, self).__post_init__()

    @property
    def report_hash(self) -> str:
        return str(self.content_hash)


class SplitValidationError(ResearchValidationError):
    def __init__(self, report: SplitValidationReport) -> None:
        first = report.errors[0]
        super().__init__(
            first.code, first.message, path=first.path, expected=first.expected,
            actual=first.actual, details={**dict(first.details), "validation_report_hash": report.report_hash},
        )
        self.report = report


def _error(code: str, message: str, **values: object) -> ErrorRecord:
    return ErrorRecord(code=code, message=message, **values)


def _element_errors(partition: SpatialPartition, buffer_m: float) -> list[ErrorRecord]:
    polygon = partition.polygon.geometry()
    boundary_buffer = polygon.boundary.buffer(buffer_m)
    violating_nodes = sorted(
        node.node_id for node in partition.network.nodes
        if not polygon.covers(Point(node.position_xy)) or boundary_buffer.intersects(Point(node.position_xy))
    )
    violating_edges = sorted(
        edge.edge_id for edge in partition.network.edges
        if not polygon.covers(LineString(edge.geometry_xy))
        or boundary_buffer.intersects(LineString(edge.geometry_xy))
    )
    if not violating_nodes and not violating_edges:
        return []
    return [_error(
        "BUFFER_ZONE_VIOLATION", "split contains node or edge crossing the protocol buffer",
        path=partition.scope.value,
        expected={"buffer_crossing_nodes": 0, "buffer_crossing_edges": 0},
        actual={"node_ids": violating_nodes, "edge_ids": violating_edges},
    )]


def inspect_spatial_split(split: SpatialSplit, protocol: SplitProtocol) -> SplitValidationReport:
    """Return a hash-preserving report; invalid splits are never generalization eligible."""
    errors: list[ErrorRecord] = []
    partitions = (split.train, split.validation, split.test)
    for left_index, left in enumerate(partitions):
        for right in partitions[left_index + 1:]:
            overlap = left.polygon.geometry().intersection(right.polygon.geometry()).area
            if overlap > 0.0:
                errors.append(_error(
                    "POLYGON_OVERLAP", "split polygons have positive intersection area",
                    path=f"{left.scope.value}:{right.scope.value}", expected=0.0, actual=overlap,
                ))
    for partition in partitions:
        if partition.polygon_hash != partition.polygon.geometry_hash:
            errors.append(_error(
                "POLYGON_HASH_MISMATCH", "declared polygon hash does not match geometry",
                path=partition.scope.value, expected=partition.polygon.geometry_hash,
                actual=partition.polygon_hash,
            ))
        if partition.network_hash != partition.network.network_content_hash:
            errors.append(_error(
                "NETWORK_HASH_MISMATCH", "declared network hash does not match split elements",
                path=partition.scope.value, expected=partition.network.network_content_hash,
                actual=partition.network_hash,
            ))
        if partition.network.metric_crs != protocol.metric_crs:
            errors.append(_error(
                "SPLIT_CRS_MISMATCH", "split network CRS does not match the protocol metric CRS",
                path=partition.scope.value, expected=protocol.metric_crs,
                actual=partition.network.metric_crs,
            ))
        errors.extend(_element_errors(partition, protocol.buffer_m))

    for left_index, left in enumerate(partitions):
        for right in partitions[left_index + 1:]:
            shared = sorted(left.network.source_edge_signatures & right.network.source_edge_signatures)
            if shared:
                errors.append(_error(
                    "SHARED_SOURCE_EDGE", "split networks share OSM source-edge signatures",
                    path=f"{left.scope.value}:{right.scope.value}", expected=0, actual=shared,
                ))
    split_cities = {
        split.train.network.city_id,
        split.validation.network.city_id,
        split.test.network.city_id,
    }
    target_cities = {target.city_id for target in split.cross_city}
    target_map_ids = [target.map_id for target in split.cross_city]
    if len(target_cities) < 2:
        errors.append(_error(
            "INSUFFICIENT_ZERO_SHOT_CITIES", "at least two distinct zero-shot cities are required",
            path="cross_city", expected=">=2", actual=sorted(target_cities),
        ))
    overlap_cities = sorted(split_cities & target_cities)
    if overlap_cities:
        errors.append(_error(
            "ZERO_SHOT_TRAIN_CITY_LEAKAGE",
            "zero-shot cities must not occur in train, validation, or test split data",
            path="cross_city", expected=[], actual=overlap_cities,
        ))
    if len(target_map_ids) != len(set(target_map_ids)):
        errors.append(_error(
            "DUPLICATE_ZERO_SHOT_MAP", "zero-shot map references must be unique",
            path="cross_city", expected=len(target_map_ids), actual=len(set(target_map_ids)),
        ))
    policies = {target.frozen_policy_hash for target in split.cross_city}
    if len(policies) > 1:
        errors.append(_error(
            "ZERO_SHOT_POLICY_DRIFT", "all zero-shot cities must use one unchanged policy",
            path="cross_city", expected=1, actual=len(policies),
        ))
    return SplitValidationReport(
        split_hash=str(split.content_hash), protocol_hash=str(protocol.content_hash),
        eligible_for_generalization=not errors,
        polygon_hashes=tuple((item.scope.value, item.polygon_hash) for item in partitions),
        network_hashes=tuple((item.scope.value, item.network_hash) for item in partitions),
        errors=tuple(errors),
    )


def validate_spatial_split(split: SpatialSplit, protocol: SplitProtocol) -> SplitValidationReport:
    report = inspect_spatial_split(split, protocol)
    if not report.eligible_for_generalization:
        raise SplitValidationError(report)
    return report


def generalization_evidence(reports: Iterable[SplitValidationReport]) -> tuple[SplitValidationReport, ...]:
    """Exclude every failed split report rather than silently treating it as evidence."""
    return tuple(report for report in reports if report.eligible_for_generalization and not report.errors)


ScopeT = TypeVar("ScopeT", bound=SplitScope)


@dataclass(frozen=True, slots=True)
class DataHandle(Generic[ScopeT]):
    scope: ScopeT
    kind: HandleKind
    identifier: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", exactly_one(self.scope, SplitScope, path="scope"))
        object.__setattr__(self, "kind", exactly_one(self.kind, HandleKind, path="kind"))
        _required(self.identifier, "identifier")


TrainDataHandle: TypeAlias = DataHandle[Literal[SplitScope.TRAIN]]
ValidationDataHandle: TypeAlias = DataHandle[Literal[SplitScope.VALIDATION]]


class TuningDataView:
    """Capability object whose public and typed surface contains no held-out handles."""

    __slots__ = ("_train", "_validation")

    def __init__(
        self,
        *,
        train: Iterable[TrainDataHandle],
        validation: Iterable[ValidationDataHandle],
    ) -> None:
        train_handles, validation_handles = tuple(train), tuple(validation)
        handles = train_handles + validation_handles
        if any(not isinstance(handle, DataHandle) for handle in handles):
            raise ResearchValidationError(
                "TUNING_HANDLE_TYPE_MISMATCH",
                "tuning views accept only typed DataHandle values",
            )
        self._reject_leakage(handles)
        if any(handle.scope is not SplitScope.TRAIN for handle in train_handles):
            raise ResearchValidationError(
                "TUNING_SCOPE_MISMATCH", "train view may contain only train handles", path="train"
            )
        if any(handle.scope is not SplitScope.VALIDATION for handle in validation_handles):
            raise ResearchValidationError(
                "TUNING_SCOPE_MISMATCH", "validation view may contain only validation handles",
                path="validation",
            )
        self._train, self._validation = train_handles, validation_handles

    @classmethod
    def from_split(cls, split: SpatialSplit) -> "TuningDataView":
        """Derive map capabilities without ever copying test or cross-city references."""
        return cls(
            train=(DataHandle(SplitScope.TRAIN, HandleKind.MAP, split.train.network.map_id),),
            validation=(
                DataHandle(
                    SplitScope.VALIDATION,
                    HandleKind.MAP,
                    split.validation.network.map_id,
                ),
            ),
        )

    @staticmethod
    def _reject_leakage(handles: Iterable[DataHandle[SplitScope]]) -> None:
        leaked = [
            handle.identifier
            for handle in handles
            if handle.scope in {SplitScope.TEST, SplitScope.CROSS_CITY}
        ]
        if leaked:
            raise ResearchValidationError(
                "TUNING_DATA_LEAKAGE",
                "test and cross-city handles are forbidden in selection paths",
                expected=[],
                actual=leaked,
            )

    @property
    def train(self) -> tuple[TrainDataHandle, ...]:
        return self._train

    @property
    def validation(self) -> tuple[ValidationDataHandle, ...]:
        return self._validation


__all__ = (
    "CrossCityMapRef",
    "DataHandle",
    "HandleKind",
    "MetricPolygon",
    "NetworkSlice",
    "SpatialPartition",
    "SpatialSplit",
    "SplitEdge",
    "SplitNode",
    "SplitProtocol",
    "SplitScope",
    "SplitValidationError",
    "SplitValidationReport",
    "TrainDataHandle",
    "TuningDataView",
    "ValidationDataHandle",
    "build_spatial_partition",
    "generalization_evidence",
    "inspect_spatial_split",
    "validate_spatial_split",
)
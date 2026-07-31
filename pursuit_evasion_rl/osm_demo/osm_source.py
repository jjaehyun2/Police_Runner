"""Bounded, provenance-preserving OpenStreetMap acquisition and conversion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Any, Callable, Mapping, Protocol

import networkx as nx
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point, box

from .models import (
    DOMAIN_SCHEMA_VERSION,
    BoundedArea,
    DomainValidationError,
    NetworkMetadata,
    RawEdge,
    RawNode,
    RawOSMGraph,
)

CONVERTER_VERSION = "osm-source-v1"
OSM_ATTRIBUTION = "OpenStreetMap contributors via OSMnx"


@dataclass(frozen=True, slots=True)
class RawGraphAcquisition:
    """A converted raw graph together with its acquisition provenance."""

    graph: RawOSMGraph
    metadata: NetworkMetadata


class OSMSource(Protocol):
    """Injectable source contract; implementations need not use a network."""

    def fetch(self, area: BoundedArea, network_type: str = "drive") -> RawOSMGraph: ...


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _validate_network_type(network_type: str) -> None:
    if network_type != "drive":
        raise DomainValidationError(
            "INVALID_NETWORK_TYPE",
            "OSM raw acquisition supports only network_type 'drive'",
            path="network_type",
            expected="drive",
            actual=network_type,
        )

def _json_value(value: Any) -> Any:
    """Convert common OSM attribute values to deterministic JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_value(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "wkt"):
        return str(value.wkt)
    return str(value)


def _digest(value: Any) -> str:
    payload = json.dumps(
        _json_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _raw_graph_hash(graph: RawOSMGraph) -> str:
    return _digest(
        {
            "schema_version": graph.schema_version,
            "nodes": [
                {
                    "source_id": node.source_id,
                    "lat": node.lat,
                    "lon": node.lon,
                    "x_m": node.x_m,
                    "y_m": node.y_m,
                    "tags": node.tags,
                }
                for node in graph.nodes
            ],
            "edges": [
                {
                    "source_u": edge.source_u,
                    "source_v": edge.source_v,
                    "source_key": edge.source_key,
                    "length_m": edge.length_m,
                    "geometry_xy": edge.geometry_xy,
                    "way_refs": edge.way_refs,
                    "oneway": edge.oneway,
                    "road_class": edge.road_class,
                    "tags": edge.tags,
                }
                for edge in graph.edges
            ],
        }
    )


def _source_id(value: Any) -> str:
    return str(value)


def _line_parts(geometry: Any) -> list[LineString]:
    if isinstance(geometry, LineString):
        return [geometry] if len(geometry.coords) >= 2 else []
    if isinstance(geometry, MultiLineString):
        return [part for part in geometry.geoms if len(part.coords) >= 2]
    if isinstance(geometry, GeometryCollection):
        parts: list[LineString] = []
        for item in geometry.geoms:
            parts.extend(_line_parts(item))
        return parts
    return []


def _edge_line(graph: nx.MultiDiGraph, u: Any, v: Any, data: Mapping[str, Any]) -> LineString:
    start = (float(graph.nodes[u]["x"]), float(graph.nodes[u]["y"]))
    end = (float(graph.nodes[v]["x"]), float(graph.nodes[v]["y"]))
    geometry = data.get("geometry")
    if isinstance(geometry, LineString) and len(geometry.coords) >= 2:
        coordinates = [(float(x), float(y)) for x, y, *_ in geometry.coords]
        return LineString(_orient_coordinates(coordinates, start, end))
    return LineString([start, end])


def _ordered_clipped_parts(source: LineString, clipped: Any) -> list[LineString]:
    ordered: list[tuple[float, LineString]] = []
    for part in _line_parts(clipped):
        coords = list(part.coords)
        start_distance = source.project(Point(coords[0]))
        end_distance = source.project(Point(coords[-1]))
        if end_distance < start_distance:
            coords.reverse()
            start_distance, end_distance = end_distance, start_distance
        ordered.append((start_distance, LineString(coords)))
    return [part for _, part in sorted(ordered, key=lambda item: item[0])]


def _same_point(first: tuple[float, float], second: tuple[float, float], tolerance: float = 1e-10) -> bool:
    return abs(first[0] - second[0]) <= tolerance and abs(first[1] - second[1]) <= tolerance

def _clip_to_area(graph: nx.MultiDiGraph, area: BoundedArea) -> tuple[nx.MultiDiGraph, bool]:
    """Clip directed edge geometry to an explicit bbox, creating boundary nodes."""
    if not graph.is_directed():
        raise DomainValidationError("UNDIRECTED_OSM_GRAPH", "OSM source graph must be directed")
    boundary = box(area.west, area.south, area.east, area.north)
    clipped = nx.MultiDiGraph()
    clipped.graph.update(graph.graph)
    clipped.graph["crs"] = "EPSG:4326"
    clipped.graph["simplified"] = True
    truncated = False

    original_positions: dict[Any, tuple[float, float]] = {}
    for node_id, data in graph.nodes(data=True):
        try:
            position = (float(data["x"]), float(data["y"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise DomainValidationError(
                "MISSING_NODE_COORDINATE", f"OSM node {node_id!r} lacks finite x/y coordinates"
            ) from exc
        if not all(math.isfinite(value) for value in position):
            raise DomainValidationError(
                "MISSING_NODE_COORDINATE", f"OSM node {node_id!r} lacks finite x/y coordinates"
            )
        original_positions[node_id] = position
        if not boundary.covers(Point(position)):
            truncated = True

    def endpoint_id(
        point: tuple[float, float], original_id: Any, original_position: tuple[float, float]
    ) -> Any:
        if _same_point(point, original_position) and boundary.covers(Point(original_position)):
            return original_id
        return f"__bbox_clip__:{point[0]:.12f}:{point[1]:.12f}"

    for u, v, key, data in graph.edges(keys=True, data=True):
        source_line = _edge_line(graph, u, v, data)
        if not boundary.covers(source_line):
            truncated = True
        parts = _ordered_clipped_parts(source_line, source_line.intersection(boundary))
        for part_index, part in enumerate(parts):
            coords = [(float(x), float(y)) for x, y, *_ in part.coords]
            if len(coords) < 2 or part.length <= 0.0:
                continue
            start_id = endpoint_id(coords[0], u, original_positions[u])
            end_id = endpoint_id(coords[-1], v, original_positions[v])
            for clipped_id, coordinate, original_id in (
                (start_id, coords[0], u),
                (end_id, coords[-1], v),
            ):
                if clipped_id in graph and clipped_id == original_id:
                    node_data = dict(graph.nodes[original_id])
                else:
                    node_data = {"boundary_clipped": True}
                node_data.update({"x": coordinate[0], "y": coordinate[1]})
                clipped.add_node(clipped_id, **node_data)
            edge_data = dict(data)
            edge_data["geometry"] = part
            edge_data["source_u"] = _source_id(u)
            edge_data["source_v"] = _source_id(v)
            edge_data["source_key"] = _source_id(key)
            edge_key: Any = key if len(parts) == 1 else f"{key}:clip:{part_index}"
            clipped.add_edge(start_id, end_id, key=edge_key, **edge_data)

    if clipped.number_of_edges() == 0:
        raise DomainValidationError(
            "EMPTY_NETWORK", "No drivable OSM edges intersect the requested bounded area"
        )
    return clipped, truncated


def _orient_coordinates(
    coordinates: list[tuple[float, float]], start: tuple[float, float], end: tuple[float, float]
) -> tuple[tuple[float, float], ...]:
    forward = math.dist(coordinates[0], start) + math.dist(coordinates[-1], end)
    reverse = math.dist(coordinates[-1], start) + math.dist(coordinates[0], end)
    if reverse < forward:
        coordinates.reverse()
    return tuple(coordinates)


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(value)
    return (value,)


def _oneway(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"yes", "true", "1", "-1", "reversible"}
    return bool(value)

def convert_osm_graph(
    graph: nx.MultiDiGraph,
    area: BoundedArea,
    *,
    network_type: str = "drive",
    source: str = OSM_ATTRIBUTION,
    acquired_at: datetime | str | None = None,
    projector: Callable[[nx.MultiDiGraph], nx.MultiDiGraph] | None = None,
) -> RawGraphAcquisition:
    """Clip and convert a geographic directed graph into immutable raw models."""
    _validate_network_type(network_type)
    clipped, truncated = _clip_to_area(graph, area)
    geographic_positions = {
        node_id: (float(data["y"]), float(data["x"]))
        for node_id, data in clipped.nodes(data=True)
    }
    if projector is None:
        try:
            import osmnx as ox
        except ImportError as exc:
            raise DomainValidationError(
                "OSMNX_UNAVAILABLE", "OSMnx is required to project geographic OSM graphs"
            ) from exc
        projector = ox.project_graph
    projected = projector(clipped)
    crs = projected.graph.get("crs")
    try:
        from pyproj import CRS

        projected_crs = CRS.from_user_input(crs)
    except (ImportError, TypeError, ValueError) as exc:
        raise DomainValidationError(
            "NON_METRIC_CRS", "OSM graph projection did not provide a recognized metric CRS"
        ) from exc
    if projected_crs.is_geographic or not projected_crs.axis_info or any(
        axis.unit_name.lower() not in {"metre", "meter"} for axis in projected_crs.axis_info
    ):
        raise DomainValidationError("NON_METRIC_CRS", "OSM graph projection did not produce a metric CRS")
    metric_crs = projected_crs.to_string()

    raw_nodes: list[RawNode] = []
    for node_id, data in projected.nodes(data=True):
        lat, lon = geographic_positions[node_id]
        tags = {key: _json_value(value) for key, value in data.items() if key not in {"x", "y", "geometry"}}
        raw_nodes.append(
            RawNode(
                source_id=_source_id(node_id),
                lat=lat,
                lon=lon,
                x_m=float(data["x"]),
                y_m=float(data["y"]),
                tags=tags,
            )
        )
    raw_nodes.sort(key=lambda item: item.source_id)
    node_by_id = {item.source_id: item for item in raw_nodes}

    raw_edges: list[RawEdge] = []
    for u, v, key, data in projected.edges(keys=True, data=True):
        source_u = _source_id(u)
        source_v = _source_id(v)
        start = (node_by_id[source_u].x_m, node_by_id[source_u].y_m)
        end = (node_by_id[source_v].x_m, node_by_id[source_v].y_m)
        geometry = data.get("geometry")
        if isinstance(geometry, LineString):
            points = [(float(x), float(y)) for x, y, *_ in geometry.coords]
        else:
            points = [start, end]
        geometry_xy = _orient_coordinates(points, start, end)
        metric_length = LineString(geometry_xy).length
        highway = _as_tuple(data.get("highway"))
        source_key = _source_id(key)
        original_u = _source_id(data.get("source_u", u))
        original_v = _source_id(data.get("source_v", v))
        original_key = _source_id(data.get("source_key", key))
        tags = {
            attribute: _json_value(value)
            for attribute, value in data.items()
            if attribute not in {"geometry", "source_u", "source_v", "source_key"}
        }
        tags["source_edge"] = {"u": original_u, "v": original_v, "key": original_key}
        raw_edges.append(
            RawEdge(
                source_u=source_u,
                source_v=source_v,
                source_key=source_key,
                length_m=float(metric_length),
                geometry_xy=geometry_xy,
                way_refs=tuple(_source_id(item) for item in _as_tuple(data.get("osmid"))),
                oneway=_oneway(data.get("oneway", False)),
                road_class="|".join(sorted(_source_id(item) for item in highway)),
                tags=tags,
            )
        )
    raw_edges.sort(key=lambda item: (item.source_u, item.source_v, item.source_key, item.geometry_xy))
    raw_graph = RawOSMGraph(nodes=tuple(raw_nodes), edges=tuple(raw_edges))

    if acquired_at is None:
        acquired_text = _utc_now().isoformat()
    elif isinstance(acquired_at, datetime):
        acquired_text = acquired_at.isoformat()
    else:
        acquired_text = str(acquired_at)
    raw_hash = _raw_graph_hash(raw_graph)
    metadata = NetworkMetadata(
        area=area,
        source=source,
        acquired_at=acquired_text,
        network_type=network_type,
        metric_crs=metric_crs,
        schema_versions={"raw_osm": DOMAIN_SCHEMA_VERSION},
        settings_hash=_digest({"converter_version": CONVERTER_VERSION, "network_type": network_type}),
        artifact_hashes={"raw_osm": raw_hash},
        statistics={"raw_nodes": len(raw_nodes), "raw_directed_edges": len(raw_edges)},
        validation={"explicit_bbox": True, "clipped_to_bbox": truncated},
        operation_status="TRUNCATED" if truncated else "READY",
    )
    return RawGraphAcquisition(graph=raw_graph, metadata=metadata)

class OSMnxSource:
    """Live source that performs only an explicit bounded-box OSM query."""

    def __init__(
        self,
        *,
        osmnx_module: Any | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._osmnx = osmnx_module
        self._clock = clock

    def _client(self) -> Any:
        if self._osmnx is None:
            try:
                import osmnx as ox
            except ImportError as exc:
                raise DomainValidationError(
                    "OSMNX_UNAVAILABLE", "Live OSM acquisition requires the pinned OSMnx dependency"
                ) from exc
            self._osmnx = ox
        return self._osmnx

    def fetch_with_metadata(
        self, area: BoundedArea, network_type: str = "drive"
    ) -> RawGraphAcquisition:
        _validate_network_type(network_type)
        client = self._client()
        bbox = (area.west, area.south, area.east, area.north)
        graph = client.graph_from_bbox(
            bbox,
            network_type=network_type,
            simplify=True,
            retain_all=True,
            truncate_by_edge=True,
        )
        return convert_osm_graph(
            graph,
            area,
            network_type=network_type,
            acquired_at=self._clock(),
            projector=client.project_graph,
        )

    def fetch(self, area: BoundedArea, network_type: str = "drive") -> RawOSMGraph:
        return self.fetch_with_metadata(area, network_type).graph


class FixtureOSMSource:
    """Offline source backed by injected immutable raw graph fixtures."""

    def __init__(
        self,
        fixtures: RawOSMGraph | Mapping[str, RawOSMGraph],
        *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if isinstance(fixtures, RawOSMGraph):
            self._fixtures: RawOSMGraph | Mapping[str, RawOSMGraph] = fixtures
        else:
            self._fixtures = dict(fixtures)
        self._clock = clock

    def _select(self, area: BoundedArea) -> RawOSMGraph:
        if isinstance(self._fixtures, RawOSMGraph):
            return self._fixtures
        try:
            return self._fixtures[area.name]
        except KeyError as exc:
            raise DomainValidationError(
                "FIXTURE_NOT_FOUND", f"No offline OSM fixture exists for bounded area {area.name!r}"
            ) from exc

    def fetch_with_metadata(
        self, area: BoundedArea, network_type: str = "drive"
    ) -> RawGraphAcquisition:
        _validate_network_type(network_type)
        graph = self._select(area)
        raw_hash = _raw_graph_hash(graph)
        timestamp = self._clock().isoformat()
        metadata = NetworkMetadata(
            area=area,
            source="injected offline OSM fixture",
            acquired_at=timestamp,
            network_type=network_type,
            metric_crs="fixture-provided-metric-crs",
            schema_versions={"raw_osm": graph.schema_version},
            settings_hash=_digest({"converter_version": CONVERTER_VERSION, "network_type": network_type}),
            artifact_hashes={"raw_osm": raw_hash},
            statistics={"raw_nodes": len(graph.nodes), "raw_directed_edges": len(graph.edges)},
            validation={"explicit_bbox": True, "clipped_to_bbox": False, "offline": True},
            operation_status="READY",
        )
        return RawGraphAcquisition(graph=graph, metadata=metadata)

    def fetch(self, area: BoundedArea, network_type: str = "drive") -> RawOSMGraph:
        return self.fetch_with_metadata(area, network_type).graph


__all__ = (
    "CONVERTER_VERSION",
    "FixtureOSMSource",
    "OSMSource",
    "OSMnxSource",
    "RawGraphAcquisition",
    "convert_osm_graph",
)

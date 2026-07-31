"""Deterministic decision-node coarsening for directed OSM road graphs."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .canonical import FLOAT_QUANTIZATION_DECIMALS, canonical_json, content_hash
from .models import (
    DomainValidationError, Intersection, MappingManifest, ModelNetwork,
    RawEdge, RawNode, RawOSMGraph, Segment,
)

COARSENER_VERSION = "decision-chain-v1"
CANONICAL_RULES = (
    "quantized coordinates, normalized decision attributes and local directed topology; "
    "lexicographic signatures; raw source IDs never break ties"
)
ACTION_ORDERING = (
    "signed turn from incoming heading, then absolute bearing, length and segment content hash; "
    "north is the reference when no incoming heading exists"
)
_TRANSITION_TAGS = (
    "access", "bridge", "junction", "lanes", "maxspeed", "motor_vehicle",
    "service", "speed_kph", "surface", "tunnel", "vehicle",
)
_IDENTITY_TAGS = frozenset({"id", "osmid", "osm_id", "source", "source_edge", "source_id"})


@dataclass(frozen=True, slots=True)
class CoarseningSettings:
    coordinate_decimals: int = FLOAT_QUANTIZATION_DECIMALS
    geometry_tolerance_m: float = 1e-4
    maximum_turn_degrees: float = 45.0
    transition_tags: tuple[str, ...] = _TRANSITION_TAGS

    def __post_init__(self) -> None:
        if isinstance(self.coordinate_decimals, bool) or self.coordinate_decimals < 0:
            raise DomainValidationError("INVALID_COARSENING_SETTINGS", "coordinate_decimals must be nonnegative")
        if self.geometry_tolerance_m < 0 or not 0 <= self.maximum_turn_degrees <= 180:
            raise DomainValidationError("INVALID_COARSENING_SETTINGS", "invalid coarsening tolerance")
        object.__setattr__(self, "transition_tags", tuple(sorted(set(self.transition_tags))))


@dataclass(frozen=True, slots=True)
class CoarseningResult:
    network: ModelNetwork
    mapping: MappingManifest

    @property
    def manifest(self) -> MappingManifest:
        return self.mapping


@dataclass(frozen=True, slots=True)
class _Candidate:
    start_signature: str
    end_signature: str
    edge_indexes: tuple[int, ...]
    raw_nodes: tuple[str, ...]
    length_m: float
    geometry_xy: tuple[tuple[float, float], ...]
    attributes: Mapping[str, Any]
    crosses_boundary: bool
    identity: bytes


def _point(point: Sequence[float], decimals: int) -> tuple[float, float]:
    return round(float(point[0]), decimals), round(float(point[1]), decimals)


def _normalized_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalized_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return tuple(sorted((_normalized_value(item) for item in value), key=canonical_json))
    return value


def _attributes(edge: RawEdge, settings: CoarseningSettings) -> dict[str, Any]:
    result: dict[str, Any] = {"road_class": edge.road_class, "oneway": edge.oneway}
    for key in settings.transition_tags:
        if key in edge.tags:
            result[key] = _normalized_value(edge.tags[key])
    return result


def _edge_payload(edge: RawEdge, settings: CoarseningSettings) -> dict[str, Any]:
    return {
        "length_m": round(edge.length_m, settings.coordinate_decimals),
        "geometry_xy": tuple(_point(point, settings.coordinate_decimals) for point in edge.geometry_xy),
        "attributes": _attributes(edge, settings),
    }


def _edge_signature(edge: RawEdge, settings: CoarseningSettings) -> str:
    return content_hash(_edge_payload(edge, settings), float_decimals=settings.coordinate_decimals)


def _raw_edge_ref(edge: RawEdge) -> str:
    return canonical_json({
        "source_u": edge.source_u,
        "source_v": edge.source_v,
        "source_key": edge.source_key,
        "way_refs": edge.way_refs,
    }).decode("utf-8")


def _boundary_kind(node: RawNode) -> str | None:
    if node.tags.get("boundary_kind"):
        return str(node.tags["boundary_kind"])
    return "bbox" if node.tags.get("boundary_clipped") else None


def _heading(geometry: Sequence[tuple[float, float]], from_start: bool) -> float:
    pairs = ((0, index) for index in range(1, len(geometry))) if from_start else (
        (index, len(geometry) - 1) for index in range(len(geometry) - 2, -1, -1)
    )
    for first, second in pairs:
        dx, dy = geometry[second][0] - geometry[first][0], geometry[second][1] - geometry[first][1]
        if dx or dy:
            return math.atan2(dy, dx)
    raise DomainValidationError("INVALID_GEOMETRY", "Road geometry has no directed extent")


def _turn_degrees(incoming: RawEdge, outgoing: RawEdge) -> float:
    first, second = _heading(incoming.geometry_xy, False), _heading(outgoing.geometry_xy, True)
    return abs(math.degrees(math.atan2(math.sin(second - first), math.cos(second - first))))


def _decision_reasons(
    node_id: str,
    incoming: Mapping[str, list[int]],
    outgoing: Mapping[str, list[int]],
    edges: Sequence[RawEdge],
    boundary: str | None,
    settings: CoarseningSettings,
) -> frozenset[str]:
    ins, outs = [edges[index] for index in incoming[node_id]], [edges[index] for index in outgoing[node_id]]
    neighbors = {edge.source_u for edge in ins if edge.source_u != node_id} | {
        edge.source_v for edge in outs if edge.source_v != node_id
    }
    reasons: set[str] = set()
    if boundary:
        reasons.add("boundary")
    if any(edge.source_u == edge.source_v for edge in ins + outs):
        reasons.add("self_loop")
    if len(neighbors) <= 1:
        reasons.add("dead_end")
    elif len(neighbors) > 2:
        reasons.add("branch_merge")
    else:
        for edge_in in ins:
            continuations = [edge_out for edge_out in outs if edge_out.source_v != edge_in.source_u]
            if len(continuations) != 1:
                reasons.add("branch_merge")
                continue
            edge_out = continuations[0]
            if _attributes(edge_in, settings) != _attributes(edge_out, settings):
                reasons.add("attribute_transition")
            if _turn_degrees(edge_in, edge_out) > settings.maximum_turn_degrees:
                reasons.add("geometry_transition")
        for edge_out in outs:
            predecessors = [edge_in for edge_in in ins if edge_in.source_u != edge_out.source_v]
            if len(predecessors) != 1:
                reasons.add("branch_merge")
    return frozenset(reasons)


def _node_payload(
    node: RawNode,
    reasons: frozenset[str],
    boundary: str | None,
    incoming: Sequence[int],
    outgoing: Sequence[int],
    edges: Sequence[RawEdge],
    settings: CoarseningSettings,
) -> dict[str, Any]:
    tags = {
        str(key): _normalized_value(value) for key, value in node.tags.items()
        if str(key).lower() not in _IDENTITY_TAGS and key != "boundary_clipped"
    }
    def descriptors(indexes: Sequence[int], direction: str) -> list[dict[str, Any]]:
        values = [{"direction": direction, "edge": _edge_payload(edges[index], settings)} for index in indexes]
        return sorted(values, key=canonical_json)
    return {
        "position_xy": _point((node.x_m, node.y_m), settings.coordinate_decimals),
        "reasons": tuple(sorted(reasons)),
        "boundary_kind": boundary,
        "tags": tags,
        "incoming": descriptors(incoming, "in"),
        "outgoing": descriptors(outgoing, "out"),
    }


def _orient_geometry(edge: RawEdge, nodes: Mapping[str, RawNode], tolerance: float) -> tuple[tuple[float, float], ...]:
    geometry = tuple(edge.geometry_xy)
    start, end = (nodes[edge.source_u].x_m, nodes[edge.source_u].y_m), (nodes[edge.source_v].x_m, nodes[edge.source_v].y_m)
    forward = math.dist(geometry[0], start) + math.dist(geometry[-1], end)
    reverse = math.dist(geometry[-1], start) + math.dist(geometry[0], end)
    if reverse < forward:
        geometry, forward = tuple(reversed(geometry)), reverse
    if forward > tolerance * 2:
        raise DomainValidationError("INVALID_EDGE_GEOMETRY", "Edge geometry does not match directed endpoints", actual=_raw_edge_ref(edge))
    return geometry


def _chain_geometry(path: Sequence[RawEdge], nodes: Mapping[str, RawNode], tolerance: float) -> tuple[tuple[float, float], ...]:
    result: list[tuple[float, float]] = []
    for edge in path:
        geometry = _orient_geometry(edge, nodes, tolerance)
        if not result:
            result.extend(geometry)
        elif math.dist(result[-1], geometry[0]) <= tolerance:
            result.extend(geometry[1:])
        else:
            raise DomainValidationError("NONCONTIGUOUS_SOURCE_CHAIN", "Coarsened source geometry is not contiguous")
    return tuple(result)


def _segment_attributes(path: Sequence[RawEdge], settings: CoarseningSettings) -> dict[str, Any]:
    return {
        "road_class": path[0].road_class,
        "oneway": path[0].oneway,
        "transition_attributes": tuple(_attributes(edge, settings) for edge in path),
    }


def _candidate_identity(candidate: Mapping[str, Any], settings: CoarseningSettings) -> bytes:
    return canonical_json(candidate, float_decimals=settings.coordinate_decimals)


def _action_key(segment: Segment, reference: float) -> tuple[float, float, float, float, str]:
    # Virtual continuation segments are colocated (zero directed extent) and can
    # not yield a bearing.  They always occupy the final action slot after the
    # physical exits, ordered among themselves by content hash for determinism.
    digest = content_hash({"geometry": segment.geometry_xy, "attributes": segment.attributes})
    if segment.virtual:
        return 1.0, 0.0, 0.0, segment.length_m, digest
    bearing = _heading(segment.geometry_xy, True)
    turn = math.atan2(math.sin(bearing - reference), math.cos(bearing - reference))
    return 0.0, turn, bearing % (2 * math.pi), segment.length_m, digest


def ordered_outgoing_segment_ids(
    network: ModelNetwork, intersection_id: int, incoming_heading: float | None = None
) -> tuple[int, ...]:
    if intersection_id not in network.intersection_ids:
        raise DomainValidationError("INVALID_CANONICAL_ID", f"Unknown intersection {intersection_id}")
    reference = math.pi / 2 if incoming_heading is None else float(incoming_heading)
    if not math.isfinite(reference):
        raise DomainValidationError("NON_FINITE_NUMBER", "incoming_heading must be finite")
    outgoing = [segment for segment in network.segments if segment.start_id == intersection_id]
    return tuple(segment.id for segment in sorted(outgoing, key=lambda item: _action_key(item, reference)))


class GraphCoarsener:
    def __init__(self, settings: CoarseningSettings | None = None) -> None:
        self.settings = settings or CoarseningSettings()

    def coarsen(self, graph: RawOSMGraph) -> CoarseningResult:
        settings = self.settings
        nodes = {node.source_id: node for node in graph.nodes}

        # Raw fixtures and third-party converters are not required by the domain
        # model to store a line string in source->destination order.  Normalize
        # that representation before decision detection and signature creation so
        # direction, turn transitions, and canonical identities all agree.
        edges = tuple(
            RawEdge(
                source_u=edge.source_u,
                source_v=edge.source_v,
                source_key=edge.source_key,
                length_m=edge.length_m,
                geometry_xy=_orient_geometry(edge, nodes, settings.geometry_tolerance_m),
                way_refs=edge.way_refs,
                oneway=edge.oneway,
                road_class=edge.road_class,
                tags=edge.tags,
            )
            for edge in graph.edges
        )
        source_keys = [(edge.source_u, edge.source_v, edge.source_key) for edge in edges]
        seen_source_keys: set[tuple[str, str, str]] = set()
        duplicate_source_keys: set[tuple[str, str, str]] = set()
        for source_key in source_keys:
            if source_key in seen_source_keys:
                duplicate_source_keys.add(source_key)
            seen_source_keys.add(source_key)
        if duplicate_source_keys:
            raise DomainValidationError(
                "AMBIGUOUS_SOURCE_EDGE_REFERENCE",
                "Raw directed edges must have unique source endpoint/key references for complete mapping",
                actual=[list(item) for item in sorted(duplicate_source_keys)],
            )

        incoming = {source_id: [] for source_id in nodes}
        outgoing = {source_id: [] for source_id in nodes}
        for index, edge in enumerate(edges):
            outgoing[edge.source_u].append(index)
            incoming[edge.source_v].append(index)
        boundaries = {source_id: _boundary_kind(node) for source_id, node in nodes.items()}
        reasons = {
            source_id: _decision_reasons(source_id, incoming, outgoing, edges, boundaries[source_id], settings)
            for source_id in nodes
        }
        decision_ids = {source_id for source_id, value in reasons.items() if value}

        # Every weak component needs a decision anchor.  A closed compatible
        # directed ring has no natural branch/dead-end decision; anchoring only
        # when the *whole graph* lacks decisions leaves rings in disconnected
        # components unvisited.  Component discovery may use source IDs as
        # storage keys, but anchor choice is exclusively canonical payload based.
        unseen = set(nodes)
        while unseen:
            seed = next(iter(unseen))
            component: set[str] = set()
            pending = [seed]
            while pending:
                source_id = pending.pop()
                if source_id in component:
                    continue
                component.add(source_id)
                unseen.discard(source_id)
                incident = incoming[source_id] + outgoing[source_id]
                for edge_index in incident:
                    source_edge = edges[edge_index]
                    for neighbor in (source_edge.source_u, source_edge.source_v):
                        if neighbor not in component:
                            pending.append(neighbor)
            if component & decision_ids:
                continue

            anchor_reason = frozenset({"cycle_anchor"})
            anchor_payloads = {
                source_id: canonical_json(
                    _node_payload(
                        nodes[source_id],
                        anchor_reason,
                        boundaries[source_id],
                        incoming[source_id],
                        outgoing[source_id],
                        edges,
                        settings,
                    ),
                    float_decimals=settings.coordinate_decimals,
                )
                for source_id in component
            }
            anchors = sorted(component, key=lambda source_id: anchor_payloads[source_id])
            if len(anchors) > 1 and anchor_payloads[anchors[0]] == anchor_payloads[anchors[1]]:
                self._collision("cycle anchor", anchors[:2])
            anchor = anchors[0]
            reasons[anchor] = anchor_reason
            decision_ids.add(anchor)

        payloads = {
            source_id: canonical_json(_node_payload(nodes[source_id], reasons[source_id], boundaries[source_id], incoming[source_id], outgoing[source_id], edges, settings), float_decimals=settings.coordinate_decimals)
            for source_id in decision_ids
        }
        signatures = {source_id: content_hash(payload.decode("utf-8")) for source_id, payload in payloads.items()}
        owners: dict[str, list[str]] = {}
        for source_id, signature in signatures.items():
            owners.setdefault(signature, []).append(source_id)
        for signature, source_ids in owners.items():
            if len(source_ids) > 1:
                self._collision("intersection", source_ids, signature)
        ordered_decisions = sorted(decision_ids, key=lambda source_id: signatures[source_id])
        ids = {source_id: index for index, source_id in enumerate(ordered_decisions)}

        visited: set[int] = set()
        candidates: list[_Candidate] = []
        for start in ordered_decisions:
            for first in sorted(outgoing[start], key=lambda index: canonical_json(_edge_payload(edges[index], settings))):
                if first in visited:
                    continue
                indexes, raw_nodes = [first], [start, edges[first].source_v]
                visited.add(first)
                previous, current = start, edges[first].source_v
                while current not in decision_ids:
                    continuations = [index for index in outgoing[current] if edges[index].source_v != previous]
                    if len(continuations) != 1:
                        raise DomainValidationError("AMBIGUOUS_SOURCE_CHAIN", "Non-decision node has no unique directed continuation", actual=len(continuations))
                    next_index = continuations[0]
                    if next_index in visited:
                        raise DomainValidationError("AMBIGUOUS_SOURCE_CHAIN", "Source chain revisited an edge")
                    visited.add(next_index)
                    indexes.append(next_index)
                    previous, current = current, edges[next_index].source_v
                    raw_nodes.append(current)
                path = tuple(edges[index] for index in indexes)
                geometry, attributes = _chain_geometry(path, nodes, settings.geometry_tolerance_m), _segment_attributes(path, settings)
                length = sum(edge.length_m for edge in path)
                crosses = any(boundaries[source_id] is not None for source_id in raw_nodes)
                identity_payload = {
                    "start_signature": signatures[start], "end_signature": signatures[current],
                    "geometry_hash": content_hash(geometry, float_decimals=settings.coordinate_decimals),
                    "attribute_hash": content_hash(attributes, float_decimals=settings.coordinate_decimals),
                    "length_m": round(length, settings.coordinate_decimals), "crosses_boundary": crosses,
                }
                candidates.append(_Candidate(signatures[start], signatures[current], tuple(indexes), tuple(raw_nodes), length, geometry, attributes, crosses, _candidate_identity(identity_payload, settings)))
        if len(visited) != len(edges):
            raise DomainValidationError("UNCOARSENED_SOURCE_EDGE", "Not every directed edge belongs to a canonical chain", expected=len(edges), actual=len(visited))

        groups: dict[bytes, list[_Candidate]] = {}
        for candidate in candidates:
            groups.setdefault(candidate.identity, []).append(candidate)
        ordered_groups = sorted(groups.values(), key=lambda group: (
            group[0].start_signature, group[0].end_signature,
            content_hash(group[0].geometry_xy), content_hash(group[0].attributes), group[0].identity,
        ))
        signature_ids = {signatures[source_id]: ids[source_id] for source_id in ordered_decisions}
        segments: list[Segment] = []
        segment_sources: dict[str, tuple[str, ...]] = {}
        collapsed: dict[str, list[int]] = {source_id: [] for source_id in nodes if source_id not in decision_ids}

        for segment_id, group in enumerate(ordered_groups):
            representative = group[0]
            chains = sorted(tuple(_raw_edge_ref(edges[index]) for index in item.edge_indexes) for item in group)
            segment_sources[str(segment_id)] = tuple(ref for chain in chains for ref in chain)
            source_signatures = tuple(_edge_signature(edges[index], settings) for index in representative.edge_indexes)
            segments.append(Segment(
                segment_id, signature_ids[representative.start_signature], signature_ids[representative.end_signature],
                representative.length_m, representative.geometry_xy, source_signatures,
                representative.attributes, False, representative.crosses_boundary,
            ))
            for item in group:
                for source_id in item.raw_nodes[1:-1]:
                    collapsed.setdefault(source_id, []).append(segment_id)

        intersections = []
        for source_id in ordered_decisions:
            intersection_id = ids[source_id]
            outgoing_segments = [segment for segment in segments if segment.start_id == intersection_id]
            outgoing_ids = tuple(segment.id for segment in sorted(outgoing_segments, key=lambda item: _action_key(item, math.pi / 2)))
            incoming_ids = tuple(segment.id for segment in segments if segment.end_id == intersection_id)
            intersections.append(Intersection(
                intersection_id, (nodes[source_id].x_m, nodes[source_id].y_m), signatures[source_id],
                boundaries[source_id], False, outgoing_ids, incoming_ids,
            ))
        network = ModelNetwork(tuple(intersections), tuple(segments))
        mapping = MappingManifest(
            COARSENER_VERSION,
            {source_id: ids.get(source_id) for source_id in sorted(nodes)},
            segment_sources,
            {"collapsed_raw_node_to_segments": {
                source_id: tuple(sorted(set(segment_ids))) for source_id, segment_ids in sorted(collapsed.items())
            }},
            CANONICAL_RULES,
            ACTION_ORDERING,
        )
        return CoarseningResult(network, mapping)

    @staticmethod
    def _collision(kind: str, source_ids: Sequence[str], signature: str | None = None) -> None:
        ordered = sorted(source_ids)
        raise DomainValidationError(
            "CANONICAL_COLLISION",
            f"Ambiguous {kind} canonical signature; raw IDs cannot break the tie",
            actual={"source_ids": ordered, "signature": signature},
        )


def coarsen_raw_graph(
    graph: RawOSMGraph, *, settings: CoarseningSettings | None = None
) -> CoarseningResult:
    return GraphCoarsener(settings).coarsen(graph)


# ---------------------------------------------------------------------------
# Degree splitting, topology validation and network statistics
# ---------------------------------------------------------------------------

MAX_OUT_DEGREE = 5
DEGREE_SPLITTER_VERSION = "colocated-virtual-chain-v1"
_VIRTUAL_ATTRIBUTES = {"virtual": True, "role": "degree_split_continuation"}


def _episode_start_reference() -> float:
    # Canonical action ordering uses north as the reference heading at episode
    # start (see coarsening ACTION_ORDERING and design section 3).
    return math.pi / 2


def _outgoing_ids(network: ModelNetwork) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {item.id: [] for item in network.intersections}
    for segment in network.segments:
        result[segment.start_id].append(segment.id)
    return result


def _order_outgoing(segment_ids: Sequence[int], segments: Mapping[int, Segment]) -> list[int]:
    reference = _episode_start_reference()
    return sorted(segment_ids, key=lambda seg_id: _action_key(segments[seg_id], reference))


@dataclass(frozen=True, slots=True)
class NetworkStatistics:
    intersection_count: int
    segment_count: int
    physical_segment_count: int
    virtual_segment_count: int
    virtual_intersection_count: int
    max_out_degree: int
    weak_component_count: int
    boundary_intersection_count: int
    unreachable_segment_count: int
    reachability_digest: str
    validation_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "intersection_count": self.intersection_count,
            "segment_count": self.segment_count,
            "physical_segment_count": self.physical_segment_count,
            "virtual_segment_count": self.virtual_segment_count,
            "virtual_intersection_count": self.virtual_intersection_count,
            "max_out_degree": self.max_out_degree,
            "weak_component_count": self.weak_component_count,
            "boundary_intersection_count": self.boundary_intersection_count,
            "unreachable_segment_count": self.unreachable_segment_count,
            "reachability_digest": self.reachability_digest,
            "validation_reasons": list(self.validation_reasons),
        }


@dataclass(frozen=True, slots=True)
class PreparedNetwork:
    network: ModelNetwork
    mapping: MappingManifest
    statistics: NetworkStatistics


def _boundary_reachable_signatures(network: ModelNetwork) -> dict[str, frozenset[str]]:
    """Map every boundary intersection signature to the set of boundary
    signatures reachable from it along directed segments.

    Signatures rather than canonical IDs are used so the comparison is stable
    across degree splitting, which renumbers/introduces intersections but must
    never change which real boundaries can reach which real boundaries.
    """
    successors: dict[int, list[int]] = {item.id: [] for item in network.intersections}
    for segment in network.segments:
        successors[segment.start_id].append(segment.end_id)
    boundary_signature = {
        item.id: item.source_signature
        for item in network.intersections
        if item.boundary_kind is not None
    }
    result: dict[str, frozenset[str]] = {}
    for source_id, signature in boundary_signature.items():
        reached: set[int] = set()
        pending = [source_id]
        while pending:
            current = pending.pop()
            for nxt in successors[current]:
                if nxt not in reached:
                    reached.add(nxt)
                    pending.append(nxt)
        reachable_boundaries = frozenset(
            boundary_signature[node] for node in reached if node in boundary_signature
        )
        # A duplicate boundary signature would already be rejected upstream as a
        # canonical collision, so assignment here is unambiguous.
        result[signature] = reachable_boundaries
    return result


def _weak_component_count(network: ModelNetwork) -> int:
    parent = {item.id: item.id for item in network.intersections}

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    for segment in network.segments:
        left, right = find(segment.start_id), find(segment.end_id)
        if left != right:
            parent[right] = left
    return len({find(item.id) for item in network.intersections})


def _unreachable_segment_count(network: ModelNetwork) -> int:
    """Count segments whose start intersection cannot be reached from any
    boundary intersection along directed segments.

    Boundary intersections are the network's entry/exit points; when none
    exist there is no defined entry and every segment is treated as reachable.
    """
    boundaries = [item.id for item in network.intersections if item.boundary_kind is not None]
    if not boundaries:
        return 0
    successors: dict[int, list[int]] = {item.id: [] for item in network.intersections}
    for segment in network.segments:
        successors[segment.start_id].append(segment.end_id)
    reachable: set[int] = set(boundaries)
    pending = list(boundaries)
    while pending:
        current = pending.pop()
        for nxt in successors[current]:
            if nxt not in reachable:
                reachable.add(nxt)
                pending.append(nxt)
    return sum(1 for segment in network.segments if segment.start_id not in reachable)


def compute_statistics(
    network: ModelNetwork, *, extra_reasons: Sequence[str] = ()
) -> NetworkStatistics:
    """Compute Model_Network structure statistics (Requirements 6.1, 6.6)."""
    outgoing = _outgoing_ids(network)
    max_out_degree = max((len(items) for items in outgoing.values()), default=0)
    physical_segments = sum(not segment.virtual for segment in network.segments)
    virtual_segments = sum(segment.virtual for segment in network.segments)
    virtual_intersections = sum(item.virtual for item in network.intersections)
    boundary_count = sum(item.boundary_kind is not None for item in network.intersections)
    weak_components = _weak_component_count(network)
    unreachable = _unreachable_segment_count(network)
    reachability_digest = content_hash({
        signature: tuple(sorted(reachable))
        for signature, reachable in sorted(_boundary_reachable_signatures(network).items())
    })

    reasons: list[str] = list(extra_reasons)
    reasons.append(
        "max_out_degree_within_limit"
        if max_out_degree <= MAX_OUT_DEGREE
        else "max_out_degree_exceeds_limit"
    )
    reasons.append("single_weak_component" if weak_components <= 1 else "multiple_weak_components")
    if boundary_count:
        reasons.append("has_boundary_intersections")
    if unreachable:
        reasons.append("unreachable_segments_present")
    if virtual_intersections:
        reasons.append("degree_split_applied")

    return NetworkStatistics(
        intersection_count=len(network.intersections),
        segment_count=len(network.segments),
        physical_segment_count=physical_segments,
        virtual_segment_count=virtual_segments,
        virtual_intersection_count=virtual_intersections,
        max_out_degree=max_out_degree,
        weak_component_count=weak_components,
        boundary_intersection_count=boundary_count,
        unreachable_segment_count=unreachable,
        reachability_digest=reachability_digest,
        validation_reasons=tuple(reasons),
    )


class DegreeSplitter:
    """Deterministically bound out-degree to at most five without losing exits.

    An intersection whose out-degree exceeds :data:`MAX_OUT_DEGREE` is expanded
    into an acyclic chain of colocated virtual intersections.  Each node in the
    chain exposes at most four physical exits plus one zero-length virtual
    continuation to the next node; the final node carries up to five physical
    exits and no continuation.  The chain is strictly forward, so it can never
    create a cycle, and its length is a hop cap for the runtime microstep loop.
    """

    def __init__(self, max_out_degree: int = MAX_OUT_DEGREE) -> None:
        if max_out_degree < 2:
            raise DomainValidationError(
                "INVALID_DEGREE_LIMIT", "max_out_degree must allow at least one exit and one continuation"
            )
        self.max_out_degree = max_out_degree

    def split(self, result: CoarseningResult) -> CoarseningResult:
        network = result.network
        segments_by_id = {segment.id: segment for segment in network.segments}
        intersections_by_id = {item.id: item for item in network.intersections}
        outgoing = _outgoing_ids(network)
        over_degree = {
            intersection_id: ordered
            for intersection_id, ids in outgoing.items()
            if len(ids) > self.max_out_degree
            for ordered in (_order_outgoing(ids, segments_by_id),)
        }
        if not over_degree:
            return result

        next_intersection_id = len(network.intersections)
        next_segment_id = len(network.segments)
        virtual_intersections: list[Intersection] = []
        continuation_segments: list[Segment] = []
        start_override: dict[int, int] = {}
        virtual_intersection_to_parent: dict[str, int] = {}
        virtual_segment_to_parent: dict[str, int] = {}
        parent_to_chain: dict[str, tuple[int, ...]] = {}
        hop_cap = 0

        for parent_id in sorted(over_degree):
            exits = over_degree[parent_id]
            parent = intersections_by_id[parent_id]
            chunks = self._chunk(exits)
            hop_cap = max(hop_cap, len(chunks))
            node_ids = [parent_id]
            for chain_index in range(1, len(chunks)):
                virtual_id = next_intersection_id
                next_intersection_id += 1
                node_ids.append(virtual_id)
                virtual_intersections.append(
                    Intersection(
                        virtual_id,
                        parent.position_xy,
                        content_hash({
                            "virtual_parent_signature": parent.source_signature,
                            "chain_index": chain_index,
                        }),
                        None,
                        True,
                    )
                )
                virtual_intersection_to_parent[str(virtual_id)] = parent_id
            parent_to_chain[str(parent_id)] = tuple(node_ids[1:])

            for chain_index, (hosted, needs_continuation) in enumerate(chunks):
                node_id = node_ids[chain_index]
                if node_id != parent_id:
                    for segment_id in hosted:
                        start_override[segment_id] = node_id
                if needs_continuation:
                    continuation_id = next_segment_id
                    next_segment_id += 1
                    continuation_segments.append(
                        Segment(
                            continuation_id,
                            node_id,
                            node_ids[chain_index + 1],
                            0.0,
                            (parent.position_xy, parent.position_xy),
                            (),
                            dict(_VIRTUAL_ATTRIBUTES),
                            True,
                            False,
                        )
                    )
                    virtual_segment_to_parent[str(continuation_id)] = parent_id

        rebuilt_segments: list[Segment] = []
        for segment in network.segments:
            new_start = start_override.get(segment.id, segment.start_id)
            if new_start == segment.start_id:
                rebuilt_segments.append(segment)
            else:
                rebuilt_segments.append(
                    Segment(
                        segment.id, new_start, segment.end_id, segment.length_m,
                        segment.geometry_xy, segment.source_edge_refs, segment.attributes,
                        segment.virtual, segment.crosses_boundary,
                    )
                )
        rebuilt_segments.extend(continuation_segments)
        segment_lookup = {segment.id: segment for segment in rebuilt_segments}
        outgoing_after = _outgoing_ids_from(rebuilt_segments, network.intersections, virtual_intersections)

        intersections: list[Intersection] = []
        for item in list(network.intersections) + virtual_intersections:
            outgoing_ids = tuple(_order_outgoing(outgoing_after[item.id], segment_lookup))
            incoming_ids = tuple(
                sorted(segment.id for segment in rebuilt_segments if segment.end_id == item.id)
            )
            intersections.append(
                Intersection(
                    item.id, item.position_xy, item.source_signature,
                    item.boundary_kind, item.virtual, outgoing_ids, incoming_ids,
                )
            )

        split_network = ModelNetwork(tuple(intersections), tuple(rebuilt_segments))
        physical_exit_ids = frozenset(segment.id for segment in network.segments if not segment.virtual)
        self._verify_split(split_network, physical_exit_ids, continuation_segments)

        virtual_mappings = {
            key: value for key, value in result.mapping.virtual_mappings.items()
        }
        virtual_mappings["degree_split"] = {
            "splitter_version": DEGREE_SPLITTER_VERSION,
            "max_out_degree": self.max_out_degree,
            "hop_cap": hop_cap,
            "parent_intersection_to_virtual_chain": parent_to_chain,
            "virtual_intersection_to_parent": virtual_intersection_to_parent,
            "virtual_segment_to_parent": virtual_segment_to_parent,
        }
        mapping = MappingManifest(
            result.mapping.coarsener_version,
            result.mapping.raw_node_to_intersection,
            result.mapping.segment_sources,
            virtual_mappings,
            result.mapping.canonical_rules,
            result.mapping.action_ordering,
        )
        return CoarseningResult(split_network, mapping)

    def _chunk(self, exits: Sequence[int]) -> list[tuple[tuple[int, ...], bool]]:
        chunks: list[tuple[tuple[int, ...], bool]] = []
        remaining = list(exits)
        while remaining:
            if len(remaining) <= self.max_out_degree:
                chunks.append((tuple(remaining), False))
                remaining = []
            else:
                head = remaining[: self.max_out_degree - 1]
                remaining = remaining[self.max_out_degree - 1 :]
                chunks.append((tuple(head), True))
        return chunks

    def _verify_split(
        self,
        split_network: ModelNetwork,
        physical_exit_ids: frozenset[int],
        continuation_segments: Sequence[Segment],
    ) -> None:
        outgoing = _outgoing_ids(split_network)
        max_degree = max((len(items) for items in outgoing.values()), default=0)
        if max_degree > self.max_out_degree:
            raise DomainValidationError(
                "MAX_OUT_DEGREE_UNRESOLVED",
                "Degree splitting could not bound out-degree without truncating exits",
                expected=self.max_out_degree,
                actual=max_degree,
            )

        preserved = frozenset(segment.id for segment in split_network.segments if not segment.virtual)
        if preserved != physical_exit_ids:
            raise DomainValidationError(
                "EXIT_NOT_PRESERVED",
                "Degree splitting must preserve every physical exit exactly once",
                expected=len(physical_exit_ids),
                actual=len(preserved),
            )

        if _has_virtual_cycle(split_network, continuation_segments):
            raise DomainValidationError(
                "DEGREE_SPLIT_CYCLE", "Degree splitting produced a cyclic virtual routing chain"
            )


def _outgoing_ids_from(
    segments: Sequence[Segment],
    physical: Sequence[Intersection],
    virtual: Sequence[Intersection],
) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {item.id: [] for item in list(physical) + list(virtual)}
    for segment in segments:
        result[segment.start_id].append(segment.id)
    return result


def _has_virtual_cycle(network: ModelNetwork, continuation_segments: Sequence[Segment]) -> bool:
    """Detect any directed cycle reachable through virtual continuation edges."""
    if not continuation_segments:
        return False
    successors: dict[int, list[int]] = {item.id: [] for item in network.intersections}
    for segment in network.segments:
        successors[segment.start_id].append(segment.end_id)
    virtual_starts = {segment.start_id for segment in continuation_segments}
    color: dict[int, int] = {}

    def visits(start: int) -> bool:
        stack = [(start, iter(successors[start]))]
        color[start] = 1
        while stack:
            node, children = stack[-1]
            advanced = False
            for child in children:
                state = color.get(child, 0)
                if state == 1:
                    return True
                if state == 0:
                    color[child] = 1
                    stack.append((child, iter(successors[child])))
                    advanced = True
                    break
            if not advanced:
                color[node] = 2
                stack.pop()
        return False

    for start in virtual_starts:
        if color.get(start, 0) == 0 and visits(start):
            return True
    return False


class NetworkValidator:
    """Validate topology preservation and compute Model_Network statistics."""

    def validate(self, before: ModelNetwork, after: ModelNetwork) -> NetworkStatistics:
        before_reach = _boundary_reachable_signatures(before)
        after_reach = _boundary_reachable_signatures(after)
        if before_reach != after_reach:
            changed = sorted(
                signature
                for signature in set(before_reach) | set(after_reach)
                if before_reach.get(signature) != after_reach.get(signature)
            )
            raise DomainValidationError(
                "BOUNDARY_REACHABILITY_CHANGED",
                "Directed boundary reachability changed during transformation",
                actual=changed,
            )
        reasons = ("boundary_reachability_preserved",)
        return compute_statistics(after, extra_reasons=reasons)


def prepare_model_network(
    result: CoarseningResult, *, splitter: DegreeSplitter | None = None
) -> PreparedNetwork:
    """Bound out-degree, verify boundary reachability and gather statistics.

    Raises a classified :class:`DomainValidationError` instead of returning a
    partial graph when exits cannot be preserved, a virtual cycle is created,
    or directed boundary reachability changes (Requirements 5.9, 6.2, 6.3).
    """
    splitter = splitter or DegreeSplitter()
    split = splitter.split(result)
    statistics = NetworkValidator().validate(result.network, split.network)
    return PreparedNetwork(split.network, split.mapping, statistics)


__all__ = (
    "ACTION_ORDERING", "CANONICAL_RULES", "COARSENER_VERSION",
    "DEGREE_SPLITTER_VERSION", "MAX_OUT_DEGREE",
    "CoarseningResult", "CoarseningSettings", "GraphCoarsener",
    "DegreeSplitter", "NetworkStatistics", "NetworkValidator", "PreparedNetwork",
    "coarsen_raw_graph", "compute_statistics", "ordered_outgoing_segment_ids",
    "prepare_model_network",
)

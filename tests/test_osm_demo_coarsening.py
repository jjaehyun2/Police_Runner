"""Focused offline tests for deterministic directed decision-node coarsening."""
from __future__ import annotations

import json
import pytest

from pursuit_evasion_rl.osm_demo.canonical import canonical_json
from pursuit_evasion_rl.osm_demo.coarsening import (
    COARSENER_VERSION, coarsen_raw_graph, ordered_outgoing_segment_ids,
)
from pursuit_evasion_rl.osm_demo.models import DomainValidationError, RawEdge, RawNode, RawOSMGraph

pytestmark = pytest.mark.offline


def node(source_id: str, x: float, y: float = 0.0, **tags) -> RawNode:
    return RawNode(source_id, 36.35, 127.40, x, y, tags)


def edge(
    source_u: str,
    source_v: str,
    key: str,
    positions: dict[str, tuple[float, float]],
    *,
    road_class: str = "residential",
    oneway: bool = True,
    way_ref: str | None = None,
    **tags,
) -> RawEdge:
    start, end = positions[source_u], positions[source_v]
    return RawEdge(
        source_u, source_v, key,
        length_m=math_dist(start, end),
        geometry_xy=(start, end),
        way_refs=(way_ref or key,),
        oneway=oneway,
        road_class=road_class,
        tags=tags,
    )


def math_dist(first: tuple[float, float], second: tuple[float, float]) -> float:
    return ((second[0] - first[0]) ** 2 + (second[1] - first[1]) ** 2) ** 0.5


def test_one_way_chain_collapses_with_geometry_length_and_full_mapping():
    positions = {"a": (0.0, 0.0), "middle": (10.0, 0.0), "z": (25.0, 0.0)}
    graph = RawOSMGraph(
        nodes=tuple(node(source_id, *position) for source_id, position in positions.items()),
        edges=(
            edge("a", "middle", "edge-1", positions, way_ref="way-10"),
            edge("middle", "z", "edge-2", positions, way_ref="way-11"),
        ),
    )

    result = coarsen_raw_graph(graph)

    assert len(result.network.intersections) == 2
    assert len(result.network.segments) == 1
    segment = result.network.segments[0]
    assert segment.length_m == pytest.approx(25.0)
    assert segment.geometry_xy == ((0.0, 0.0), (10.0, 0.0), (25.0, 0.0))
    assert len(segment.source_edge_refs) == 2
    assert result.mapping.coarsener_version == COARSENER_VERSION
    assert result.mapping.raw_node_to_intersection["middle"] is None
    refs = [json.loads(item) for item in result.mapping.segment_sources["0"]]
    assert [(item["source_u"], item["source_v"]) for item in refs] == [
        ("a", "middle"), ("middle", "z"),
    ]
    assert [item["way_refs"] for item in refs] == [["way-10"], ["way-11"]]
    assert result.mapping.virtual_mappings["collapsed_raw_node_to_segments"]["middle"] == (0,)


def test_branch_merge_boundary_and_attribute_transition_nodes_are_retained():
    positions = {
        "branch-in": (0, 0), "branch": (10, 0), "branch-a": (20, 5), "branch-b": (20, -5),
        "merge-a": (0, 20), "merge-b": (0, 30), "merge": (10, 25), "merge-out": (20, 25),
        "attr-a": (0, 40), "attr": (10, 40), "attr-z": (20, 40),
        "bound-a": (0, 50), "bound": (10, 50), "bound-z": (20, 50),
    }
    nodes = tuple(node(source_id, *position, **({"boundary_clipped": True} if source_id == "bound" else {}))
                  for source_id, position in positions.items())
    edges = (
        edge("branch-in", "branch", "b0", positions), edge("branch", "branch-a", "b1", positions),
        edge("branch", "branch-b", "b2", positions), edge("merge-a", "merge", "m0", positions),
        edge("merge-b", "merge", "m1", positions), edge("merge", "merge-out", "m2", positions),
        edge("attr-a", "attr", "a0", positions),
        edge("attr", "attr-z", "a1", positions, road_class="primary"),
        edge("bound-a", "bound", "d0", positions), edge("bound", "bound-z", "d1", positions),
    )

    result = coarsen_raw_graph(RawOSMGraph(nodes=nodes, edges=edges))

    for source_id in ("branch", "merge", "attr", "bound"):
        assert result.mapping.raw_node_to_intersection[source_id] is not None
    boundary_id = result.mapping.raw_node_to_intersection["bound"]
    assert result.network.intersections[boundary_id].boundary_kind == "bbox"


def relabelled_chain(prefix: str, reverse_storage: bool, way_prefix: str) -> RawOSMGraph:
    ids = [f"{prefix}-left", f"{prefix}-middle", f"{prefix}-right"]
    positions = {ids[0]: (0.0, 0.0), ids[1]: (10.0, 0.0), ids[2]: (20.0, 0.0)}
    nodes = [node(source_id, *positions[source_id]) for source_id in ids]
    edges = [
        edge(ids[0], ids[1], f"{prefix}-key-1", positions, way_ref=f"{way_prefix}-1"),
        edge(ids[1], ids[2], f"{prefix}-key-2", positions, way_ref=f"{way_prefix}-2"),
    ]
    if reverse_storage:
        nodes.reverse()
        edges.reverse()
    return RawOSMGraph(tuple(nodes), tuple(edges))


def test_canonical_network_ignores_raw_ids_way_refs_and_iteration_order():
    first = coarsen_raw_graph(relabelled_chain("raw-a", False, "way-a"))
    second = coarsen_raw_graph(relabelled_chain("raw-b", True, "way-b"))

    assert canonical_json(first.network) == canonical_json(second.network)
    assert [item.id for item in first.network.intersections] == [0, 1]
    assert [item.id for item in first.network.segments] == [0]
    assert first.mapping.segment_sources != second.mapping.segment_sources


def test_bidirectional_roads_remain_distinct_directed_segments():
    positions = {"left": (0.0, 0.0), "middle": (10.0, 0.0), "right": (20.0, 0.0)}
    graph = RawOSMGraph(
        nodes=tuple(node(source_id, *position) for source_id, position in positions.items()),
        edges=(
            edge("left", "middle", "lm", positions, oneway=False),
            edge("middle", "right", "mr", positions, oneway=False),
            edge("right", "middle", "rm", positions, oneway=False),
            edge("middle", "left", "ml", positions, oneway=False),
        ),
    )

    result = coarsen_raw_graph(graph)

    assert len(result.network.intersections) == 2
    assert len(result.network.segments) == 2
    assert {item.geometry_xy for item in result.network.segments} == {
        ((0.0, 0.0), (10.0, 0.0), (20.0, 0.0)),
        ((20.0, 0.0), (10.0, 0.0), (0.0, 0.0)),
    }
    assert result.mapping.raw_node_to_intersection["middle"] is None


def test_action_slots_have_stable_geometric_order():
    positions = {
        "south": (0, -10), "center": (0, 0), "east": (10, 0),
        "north": (0, 10), "west": (-10, 0),
    }
    graph = RawOSMGraph(
        nodes=tuple(node(source_id, *position) for source_id, position in positions.items()),
        edges=(
            edge("south", "center", "in", positions), edge("center", "west", "west", positions),
            edge("center", "north", "north", positions), edge("center", "east", "east", positions),
        ),
    )
    result = coarsen_raw_graph(graph)
    center_id = result.mapping.raw_node_to_intersection["center"]

    ordered = ordered_outgoing_segment_ids(result.network, center_id)
    destinations = [result.network.segments[item].geometry_xy[-1] for item in ordered]

    assert destinations == [(10.0, 0.0), (0.0, 10.0), (-10.0, 0.0)]
    assert result.network.intersections[center_id].outgoing_segment_ids == ordered


def test_ambiguous_intersection_signature_is_rejected_without_raw_id_tiebreaking():
    positions = {"road-a": (10.0, 0.0), "road-b": (20.0, 0.0)}
    graph = RawOSMGraph(
        nodes=(node("road-a", 10), node("road-b", 20), node("ambiguous-1", 0, 50), node("ambiguous-2", 0, 50)),
        edges=(edge("road-a", "road-b", "road", positions),),
    )

    with pytest.raises(DomainValidationError) as raised:
        coarsen_raw_graph(graph)

    assert raised.value.code == "CANONICAL_COLLISION"
    assert raised.value.actual["source_ids"] == ["ambiguous-1", "ambiguous-2"]


def test_repeated_processing_has_identical_network_and_manifest():
    graph = relabelled_chain("same", True, "way")

    first, second = coarsen_raw_graph(graph), coarsen_raw_graph(graph)

    assert canonical_json(first.network) == canonical_json(second.network)
    assert canonical_json(first.mapping) == canonical_json(second.mapping)


def test_disconnected_compatible_ring_gets_canonical_anchor_and_complete_mapping():
    """A ring still coarsens when another component already has decisions."""
    from pursuit_evasion_rl.osm_demo.coarsening import CoarseningSettings

    positions = {
        "chain-a": (-30.0, 0.0),
        "chain-z": (-20.0, 0.0),
        "ring-0": (10.0, 0.0),
        "ring-1": (7.071067812, 7.071067812),
        "ring-2": (0.0, 10.0),
        "ring-3": (-7.071067812, 7.071067812),
        "ring-4": (-10.0, 0.0),
        "ring-5": (-7.071067812, -7.071067812),
        "ring-6": (0.0, -10.0),
        "ring-7": (7.071067812, -7.071067812),
    }
    ring_ids = tuple(f"ring-{index}" for index in range(8))
    graph_edges = [edge("chain-a", "chain-z", "chain", positions)]
    graph_edges.extend(
        edge(source_id, ring_ids[(index + 1) % len(ring_ids)], f"ring-edge-{index}", positions)
        for index, source_id in enumerate(ring_ids)
    )
    graph = RawOSMGraph(
        nodes=tuple(node(source_id, *position) for source_id, position in positions.items()),
        edges=tuple(graph_edges),
    )

    result = coarsen_raw_graph(
        graph,
        settings=CoarseningSettings(maximum_turn_degrees=180.0),
    )

    assert len(result.network.intersections) == 3
    assert len(result.network.segments) == 2
    ring_segment = next(segment for segment in result.network.segments if segment.start_id == segment.end_id)
    assert len(ring_segment.source_edge_refs) == 8
    mapped_refs = [
        json.loads(reference)
        for references in result.mapping.segment_sources.values()
        for reference in references
    ]
    assert len(mapped_refs) == len(graph.edges)
    assert {
        (reference["source_u"], reference["source_v"], reference["source_key"])
        for reference in mapped_refs
    } == {(item.source_u, item.source_v, item.source_key) for item in graph.edges}
    collapsed = result.mapping.virtual_mappings["collapsed_raw_node_to_segments"]
    assert all(
        result.mapping.raw_node_to_intersection[source_id] is not None
        or collapsed[source_id]
        for source_id in positions
    )


def test_reversed_stored_geometry_is_oriented_to_directed_source_chain():
    positions = {"start": (0.0, 0.0), "middle": (10.0, 0.0), "end": (20.0, 0.0)}
    forward_edges = (
        edge("start", "middle", "first", positions),
        edge("middle", "end", "second", positions),
    )
    reversed_edges = tuple(
        RawEdge(
            source_u=item.source_u,
            source_v=item.source_v,
            source_key=item.source_key,
            length_m=item.length_m,
            geometry_xy=tuple(reversed(item.geometry_xy)),
            way_refs=item.way_refs,
            oneway=item.oneway,
            road_class=item.road_class,
            tags=item.tags,
        )
        for item in forward_edges
    )
    nodes = tuple(node(source_id, *position) for source_id, position in positions.items())

    normal = coarsen_raw_graph(RawOSMGraph(nodes, forward_edges))
    reversed_storage = coarsen_raw_graph(RawOSMGraph(tuple(reversed(nodes)), tuple(reversed(reversed_edges))))

    assert canonical_json(normal.network) == canonical_json(reversed_storage.network)
    assert reversed_storage.network.segments[0].geometry_xy == (
        (0.0, 0.0), (10.0, 0.0), (20.0, 0.0)
    )


def test_exact_parallel_segments_deduplicate_but_manifest_maps_every_source_edge():
    positions = {"start": (0.0, 0.0), "end": (10.0, 0.0)}
    graph = RawOSMGraph(
        nodes=tuple(node(source_id, *position) for source_id, position in positions.items()),
        edges=(
            edge("start", "end", "parallel-a", positions, way_ref="way-a"),
            edge("start", "end", "parallel-b", positions, way_ref="way-b"),
        ),
    )

    result = coarsen_raw_graph(graph)

    assert len(result.network.segments) == 1
    mapped = [json.loads(item) for item in result.mapping.segment_sources["0"]]
    assert {(item["source_key"], tuple(item["way_refs"])) for item in mapped} == {
        ("parallel-a", ("way-a",)),
        ("parallel-b", ("way-b",)),
    }


def test_duplicate_raw_endpoint_key_is_rejected_as_an_ambiguous_mapping():
    positions = {"start": (0.0, 0.0), "end": (10.0, 0.0)}
    duplicate = edge("start", "end", "same-key", positions)
    graph = RawOSMGraph(
        nodes=tuple(node(source_id, *position) for source_id, position in positions.items()),
        edges=(duplicate, duplicate),
    )

    with pytest.raises(DomainValidationError) as raised:
        coarsen_raw_graph(graph)

    assert raised.value.code == "AMBIGUOUS_SOURCE_EDGE_REFERENCE"
    assert raised.value.actual == [["start", "end", "same-key"]]

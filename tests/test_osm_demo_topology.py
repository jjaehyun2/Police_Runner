"""Focused offline tests for degree splitting, topology validation and stats.

Covers Requirements 5.9, 6.1-6.3 and 6.6 without external OSM access.
"""
from __future__ import annotations

import math

import pytest

from pursuit_evasion_rl.osm_demo.coarsening import (
    MAX_OUT_DEGREE,
    CoarseningResult,
    DegreeSplitter,
    NetworkValidator,
    compute_statistics,
    coarsen_raw_graph,
    ordered_outgoing_segment_ids,
    prepare_model_network,
)
from pursuit_evasion_rl.osm_demo.models import (
    DomainValidationError,
    Intersection,
    MappingManifest,
    ModelNetwork,
    RawEdge,
    RawNode,
    RawOSMGraph,
    Segment,
)

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _node(source_id: str, x: float, y: float, **tags) -> RawNode:
    return RawNode(source_id, 36.35, 127.40, x, y, tags)


def _edge(u: str, v: str, key: str, positions, *, oneway: bool = True) -> RawEdge:
    start, end = positions[u], positions[v]
    length = math.dist(start, end)
    return RawEdge(u, v, key, length, (start, end), (key,), oneway, "residential", {})


def _radial_graph(exit_count: int) -> RawOSMGraph:
    """A single center intersection with ``exit_count`` outgoing dead-end roads
    plus one incoming road, so the center's out-degree equals ``exit_count``."""
    positions = {"center": (0.0, 0.0), "feed": (0.0, -100.0)}
    edges = [_edge("feed", "center", "feed-edge", positions)]
    for index in range(exit_count):
        angle = (2.0 * math.pi * index) / exit_count + 0.15
        name = f"exit-{index}"
        positions[name] = (round(50.0 * math.cos(angle), 6), round(50.0 * math.sin(angle), 6))
        edges.append(_edge("center", name, f"exit-edge-{index}", positions))
    nodes = tuple(_node(source_id, *position) for source_id, position in positions.items())
    return RawOSMGraph(nodes=nodes, edges=tuple(edges))


def _center_id(result: CoarseningResult) -> int:
    return result.mapping.raw_node_to_intersection["center"]


# ---------------------------------------------------------------------------
# degree splitting
# ---------------------------------------------------------------------------

def test_low_degree_network_is_returned_unchanged():
    result = coarsen_raw_graph(_radial_graph(4))
    split = DegreeSplitter().split(result)
    assert split is result  # no virtual routing needed


def test_degree_six_intersection_splits_without_truncating_exits():
    result = coarsen_raw_graph(_radial_graph(6))
    center = _center_id(result)
    assert len(result.network.intersections[center].outgoing_segment_ids) == 6

    split = DegreeSplitter().split(result)

    # Every physical exit is preserved exactly once.
    physical = [segment for segment in split.network.segments if not segment.virtual]
    assert len(physical) == len(result.network.segments)

    # Out-degree is now bounded by five everywhere.
    outgoing = {item.id: item.outgoing_segment_ids for item in split.network.intersections}
    assert max(len(ids) for ids in outgoing.values()) <= MAX_OUT_DEGREE

    # Exactly one colocated virtual intersection with a zero-length continuation.
    virtual_intersections = [item for item in split.network.intersections if item.virtual]
    virtual_segments = [segment for segment in split.network.segments if segment.virtual]
    assert len(virtual_intersections) == 1
    assert len(virtual_segments) == 1
    continuation = virtual_segments[0]
    assert continuation.length_m == 0.0
    center_position = split.network.intersections[center].position_xy
    assert virtual_intersections[0].position_xy == center_position
    assert continuation.geometry_xy == (center_position, center_position)

    # Mapping records the parent linkage and a hop cap for runtime microsteps.
    degree_split = split.mapping.virtual_mappings["degree_split"]
    assert degree_split["hop_cap"] == 2
    assert str(virtual_intersections[0].id) in degree_split["virtual_intersection_to_parent"]
    assert degree_split["virtual_intersection_to_parent"][str(virtual_intersections[0].id)] == center


def test_all_six_exit_destinations_remain_reachable_from_center():
    result = coarsen_raw_graph(_radial_graph(6))
    center = _center_id(result)
    original_targets = {
        result.network.segments[seg_id].end_id
        for seg_id in result.network.intersections[center].outgoing_segment_ids
    }

    split = DegreeSplitter().split(result)

    successors: dict[int, list[int]] = {item.id: [] for item in split.network.intersections}
    for segment in split.network.segments:
        successors[segment.start_id].append(segment.end_id)
    reached: set[int] = set()
    pending = [center]
    while pending:
        current = pending.pop()
        for nxt in successors[current]:
            if nxt not in reached:
                reached.add(nxt)
                pending.append(nxt)
    assert original_targets <= reached


def test_large_degree_split_chains_multiple_virtual_nodes_and_bounds_degree():
    result = coarsen_raw_graph(_radial_graph(11))
    split = DegreeSplitter().split(result)

    outgoing = {item.id: item.outgoing_segment_ids for item in split.network.intersections}
    assert max(len(ids) for ids in outgoing.values()) <= MAX_OUT_DEGREE
    virtual_intersections = [item for item in split.network.intersections if item.virtual]
    # 11 exits -> chunks of 4, 4, 3 -> two virtual nodes after the parent.
    assert len(virtual_intersections) == 2
    assert split.mapping.virtual_mappings["degree_split"]["hop_cap"] == 3
    # ModelNetwork construction already enforced contiguous IDs and adjacency.
    assert [item.id for item in split.network.intersections] == list(
        range(len(split.network.intersections))
    )


def test_degree_split_is_deterministic_and_idempotent():
    first = DegreeSplitter().split(coarsen_raw_graph(_radial_graph(7)))
    second = DegreeSplitter().split(coarsen_raw_graph(_radial_graph(7)))
    from pursuit_evasion_rl.osm_demo.canonical import canonical_json

    assert canonical_json(first.network) == canonical_json(second.network)
    assert canonical_json(first.mapping) == canonical_json(second.mapping)

    # Splitting an already-bounded network changes nothing further.
    again = DegreeSplitter().split(first)
    assert canonical_json(again.network) == canonical_json(first.network)


def test_continuation_slot_orders_after_physical_exits():
    result = coarsen_raw_graph(_radial_graph(6))
    center = _center_id(result)
    split = DegreeSplitter().split(result)

    ordered = ordered_outgoing_segment_ids(split.network, center)
    virtual_flags = [split.network.segments[seg_id].virtual for seg_id in ordered]
    # The zero-length continuation is always the final action slot.
    assert virtual_flags[-1] is True
    assert not any(virtual_flags[:-1])


# ---------------------------------------------------------------------------
# statistics and topology validation
# ---------------------------------------------------------------------------

def test_statistics_match_reference_counts():
    result = coarsen_raw_graph(_radial_graph(6))
    prepared = prepare_model_network(result)
    stats = prepared.statistics
    network = prepared.network

    assert stats.intersection_count == len(network.intersections)
    assert stats.segment_count == len(network.segments)
    assert stats.max_out_degree == max(
        sum(segment.start_id == item.id for segment in network.segments)
        for item in network.intersections
    )
    assert stats.max_out_degree <= MAX_OUT_DEGREE
    assert stats.virtual_intersection_count == sum(item.virtual for item in network.intersections)
    assert stats.virtual_segment_count == sum(segment.virtual for segment in network.segments)
    assert "boundary_reachability_preserved" in stats.validation_reasons
    assert "degree_split_applied" in stats.validation_reasons


def test_boundary_statistics_and_reachability_digest():
    positions = {"gate": (0.0, 0.0), "inner": (10.0, 0.0), "exit": (20.0, 0.0)}
    nodes = (
        _node("gate", 0.0, 0.0, boundary_clipped=True),
        _node("inner", 10.0, 0.0),
        _node("exit", 20.0, 0.0, boundary_clipped=True),
    )
    edges = (
        _edge("gate", "inner", "g", positions),
        _edge("inner", "exit", "e", positions),
    )
    result = coarsen_raw_graph(RawOSMGraph(nodes=nodes, edges=edges))
    stats = compute_statistics(result.network)
    assert stats.boundary_intersection_count == 2
    assert stats.weak_component_count == 1
    assert stats.unreachable_segment_count == 0
    assert stats.reachability_digest  # nonempty digest recorded for metadata


def test_validator_accepts_reachability_preserving_split():
    result = coarsen_raw_graph(_radial_graph(6))
    split = DegreeSplitter().split(result)
    stats = NetworkValidator().validate(result.network, split.network)
    assert stats.max_out_degree <= MAX_OUT_DEGREE


def test_validator_rejects_changed_boundary_reachability():
    # Build a minimal valid network, then drop the segment that connects the
    # two boundaries so the "after" graph loses a reachable boundary.
    intersections = (
        Intersection(0, (0.0, 0.0), "sig-a", "bbox", False, (0,), ()),
        Intersection(1, (10.0, 0.0), "sig-b", "bbox", False, (), (0,)),
    )
    segment = Segment(0, 0, 1, 10.0, ((0.0, 0.0), (10.0, 0.0)))
    before = ModelNetwork(intersections, (segment,))

    # "after" reroutes the segment so boundary 0 can no longer reach boundary 1.
    detour = Intersection(2, (5.0, 5.0), "sig-c", None, False, (), (0,))
    rerouted = Segment(0, 0, 2, 10.0, ((0.0, 0.0), (5.0, 5.0)))
    after = ModelNetwork(
        (
            Intersection(0, (0.0, 0.0), "sig-a", "bbox", False, (0,), ()),
            Intersection(1, (10.0, 0.0), "sig-b", "bbox", False, (), ()),
            detour,
        ),
        (rerouted,),
    )

    with pytest.raises(DomainValidationError) as raised:
        NetworkValidator().validate(before, after)
    assert raised.value.code == "BOUNDARY_REACHABILITY_CHANGED"


def test_prepare_model_network_reports_max_degree_within_limit_for_simple_graph():
    positions = {"a": (0.0, 0.0), "b": (10.0, 0.0), "c": (20.0, 0.0)}
    graph = RawOSMGraph(
        nodes=tuple(_node(source_id, *position) for source_id, position in positions.items()),
        edges=(_edge("a", "b", "ab", positions), _edge("b", "c", "bc", positions)),
    )
    prepared = prepare_model_network(coarsen_raw_graph(graph))
    assert prepared.statistics.virtual_intersection_count == 0
    assert "max_out_degree_within_limit" in prepared.statistics.validation_reasons

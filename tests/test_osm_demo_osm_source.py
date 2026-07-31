"""Offline tests for bounded OSM acquisition and raw graph conversion."""

from datetime import datetime, timezone

import networkx as nx
import osmnx as ox
import pytest
from shapely.geometry import LineString

from pursuit_evasion_rl.osm_demo.models import (
    BoundedArea,
    DomainValidationError,
    RawEdge,
    RawNode,
    RawOSMGraph,
)
from pursuit_evasion_rl.osm_demo.osm_source import (
    FixtureOSMSource,
    OSMnxSource,
    convert_osm_graph,
)


def area() -> BoundedArea:
    return BoundedArea(
        name="offline-test",
        north=36.351,
        south=36.349,
        east=127.402,
        west=127.398,
        max_area_km2=2.0,
    )


def geographic_graph(*, oversized: bool = False) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(crs="EPSG:4326", simplified=True)
    graph.add_node(10, x=127.399, y=36.350, highway="traffic_signals")
    end_lon = 127.404 if oversized else 127.401
    graph.add_node(20, x=end_lon, y=36.350, name="east")
    graph.add_edge(
        10,
        20,
        key=7,
        osmid=[1001, 1002],
        highway="primary",
        oneway="yes",
        maxspeed="50",
        geometry=LineString([(127.399, 36.350), (127.400, 36.3502), (end_lon, 36.350)]),
    )
    if not oversized:
        graph.add_edge(
            20,
            10,
            key=8,
            osmid=1003,
            highway="secondary",
            oneway=False,
            geometry=LineString([(127.401, 36.350), (127.399, 36.350)]),
        )
    return graph

class RecordingOSMnx:
    def __init__(self, graph: nx.MultiDiGraph) -> None:
        self.graph = graph
        self.calls: list[tuple[tuple[float, float, float, float], dict]] = []

    def graph_from_bbox(self, bbox, **kwargs):
        self.calls.append((bbox, kwargs))
        return self.graph.copy()

    @staticmethod
    def project_graph(graph):
        return ox.project_graph(graph)


def test_live_source_uses_only_explicit_bbox_and_preserves_directed_provenance():
    bounded = area()
    client = RecordingOSMnx(geographic_graph())
    source = OSMnxSource(
        osmnx_module=client,
        clock=lambda: datetime(2025, 1, 2, 3, 4, tzinfo=timezone.utc),
    )

    result = source.fetch_with_metadata(bounded)

    assert client.calls == [
        (
            (bounded.west, bounded.south, bounded.east, bounded.north),
            {
                "network_type": "drive",
                "simplify": True,
                "retain_all": True,
                "truncate_by_edge": True,
            },
        )
    ]
    assert [(edge.source_u, edge.source_v) for edge in result.graph.edges] == [("10", "20"), ("20", "10")]
    forward = result.graph.edges[0]
    assert forward.way_refs == ("1001", "1002")
    assert forward.oneway is True
    assert forward.road_class == "primary"
    assert forward.tags["maxspeed"] == "50"
    assert forward.tags["source_edge"] == {"u": "10", "v": "20", "key": "7"}
    assert forward.length_m > 100.0
    assert forward.geometry_xy[0] != forward.geometry_xy[-1]
    assert result.metadata.operation_status == "READY"
    assert result.metadata.area == bounded
    assert result.metadata.network_type == "drive"
    assert result.metadata.source.startswith("OpenStreetMap contributors")
    assert result.metadata.acquired_at.startswith("2025-01-02T03:04")
    assert len(result.metadata.artifact_hashes["raw_osm"]) == 64
    assert "4326" not in result.metadata.metric_crs


def test_oversized_supplied_graph_is_clipped_and_marked_truncated():
    bounded = area()

    result = convert_osm_graph(geographic_graph(oversized=True), bounded)

    assert result.metadata.operation_status == "TRUNCATED"
    assert result.metadata.validation["clipped_to_bbox"] is True
    assert all(bounded.west <= node.lon <= bounded.east for node in result.graph.nodes)
    assert all(bounded.south <= node.lat <= bounded.north for node in result.graph.nodes)
    assert any(node.tags.get("boundary_clipped") for node in result.graph.nodes)
    assert result.graph.edges[0].tags["source_edge"] == {"u": "10", "v": "20", "key": "7"}
    assert result.graph.edges[0].source_v.startswith("__bbox_clip__")


def test_reversed_osm_geometry_is_oriented_in_the_directed_edge_direction():
    graph = geographic_graph()
    graph.edges[10, 20, 7]["geometry"] = LineString(
        reversed(graph.edges[10, 20, 7]["geometry"].coords)
    )

    result = convert_osm_graph(graph, area())

    forward = next(edge for edge in result.graph.edges if edge.tags["source_edge"]["key"] == "7")
    nodes = {node.source_id: node for node in result.graph.nodes}
    assert forward.source_u == "10"
    assert forward.source_v == "20"
    assert forward.geometry_xy[0] == pytest.approx((nodes["10"].x_m, nodes["10"].y_m))
    assert forward.geometry_xy[-1] == pytest.approx((nodes["20"].x_m, nodes["20"].y_m))


def test_conversion_rejects_a_geographic_projector_result():
    def non_metric_projector(graph):
        result = graph.copy()
        result.graph["crs"] = "EPSG:4269"
        return result

    with pytest.raises(DomainValidationError) as raised:
        convert_osm_graph(geographic_graph(), area(), projector=non_metric_projector)

    assert raised.value.code == "NON_METRIC_CRS"


def fixture_graph() -> RawOSMGraph:
    return RawOSMGraph(
        nodes=(
            RawNode("a", 36.35, 127.40, 0.0, 0.0, {"name": "A"}),
            RawNode("b", 36.35, 127.401, 90.0, 0.0, {"name": "B"}),
        ),
        edges=(
            RawEdge(
                "a",
                "b",
                "0",
                90.0,
                ((0.0, 0.0), (90.0, 0.0)),
                way_refs=("42",),
                oneway=True,
                road_class="residential",
            ),
        ),
    )


def test_fixture_source_is_offline_injected_and_produces_equivalent_metadata():
    bounded = area()
    expected = fixture_graph()
    source = FixtureOSMSource(
        {bounded.name: expected},
        clock=lambda: datetime(2025, 2, 3, tzinfo=timezone.utc),
    )

    result = source.fetch_with_metadata(bounded)

    assert source.fetch(bounded) is expected
    assert result.graph is expected
    assert result.metadata.area == bounded
    assert result.metadata.source == "injected offline OSM fixture"
    assert result.metadata.validation["offline"] is True
    assert result.metadata.operation_status == "READY"
    assert len(result.metadata.artifact_hashes["raw_osm"]) == 64


def test_custom_area_uses_same_validation_and_drive_only_contract():
    source = FixtureOSMSource(fixture_graph())

    with pytest.raises(DomainValidationError) as raised:
        source.fetch(area(), "walk")

    assert raised.value.code == "INVALID_NETWORK_TYPE"
    assert raised.value.expected == "drive"


def test_missing_named_fixture_is_classified_without_falling_back_to_network():
    source = FixtureOSMSource({"different-area": fixture_graph()})

    with pytest.raises(DomainValidationError) as raised:
        source.fetch(area())

    assert raised.value.code == "FIXTURE_NOT_FOUND"

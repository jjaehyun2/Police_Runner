"""Property 6 coverage for the spatial split generalization gate."""

from __future__ import annotations

from dataclasses import dataclass

from hypothesis import given, settings, strategies as st
import pytest

from pursuit_evasion_rl.research.domain import DataKind, EpisodeOutcome, MapScenario
from pursuit_evasion_rl.research.maps.registry import ActualOSMProvenance, RegisteredMap
from pursuit_evasion_rl.research.maps.splits import (
    CrossCityMapRef,
    MetricPolygon,
    NetworkSlice,
    SpatialPartition,
    SpatialSplit,
    SplitEdge,
    SplitNode,
    SplitProtocol,
    SplitScope,
    SplitValidationError,
    generalization_evidence,
    inspect_spatial_split,
    validate_spatial_split,
)


positive_buffers = st.floats(
    min_value=0.5,
    max_value=4.0,
    allow_nan=False,
    allow_infinity=False,
)


@st.composite
def source_edge_sets(draw):
    """Generate three non-empty source-edge sets, optionally sharing one lineage."""
    counts = draw(st.tuples(*(st.integers(1, 3) for _ in range(3))))
    shared = draw(st.booleans())
    sets = [
        {f"{scope}/osm-edge-{index}" for index in range(count)}
        for scope, count in zip(("train", "validation", "test"), counts)
    ]
    if shared:
        sets[0].add("shared/osm-edge")
        sets[1].add("shared/osm-edge")
    return tuple(frozenset(items) for items in sets), not shared


@dataclass(frozen=True)
class SpatialCase:
    polygons: tuple[MetricPolygon, MetricPolygon, MetricPolygon]
    buffer_m: float
    source_sets: tuple[frozenset[str], frozenset[str], frozenset[str]]
    polygon_non_overlap: bool
    buffer_clear: bool
    source_disjoint: bool


@st.composite
def small_polygon_cases(draw):
    """Generate small rectangles with either zero or positive pairwise overlap area."""
    width = draw(st.integers(min_value=20, max_value=50))
    gap = draw(st.integers(min_value=1, max_value=10))
    non_overlapping = draw(st.booleans())
    validation_x = width + gap if non_overlapping else width - draw(st.integers(1, width // 2))
    test_x = validation_x + width + gap

    def rectangle(x: float) -> MetricPolygon:
        return MetricPolygon(exterior=((x, 0.0), (x + width, 0.0),
                                       (x + width, width), (x, width)))

    return (rectangle(0.0), rectangle(float(validation_x)), rectangle(float(test_x))), non_overlapping


@st.composite
def spatial_split_cases(draw):
    polygons, polygon_non_overlap = draw(small_polygon_cases())
    buffer_m = draw(positive_buffers)
    source_sets, source_disjoint = draw(source_edge_sets())
    return SpatialCase(
        polygons=polygons,
        buffer_m=buffer_m,
        source_sets=source_sets,
        polygon_non_overlap=polygon_non_overlap,
        buffer_clear=draw(st.booleans()),
        source_disjoint=source_disjoint,
    )


def _registered_actual(map_id: str, source_edges: frozenset[str]) -> RegisteredMap:
    provenance = ActualOSMProvenance(
        query_geometry={"type": "Polygon", "map_id": map_id},
        source="OpenStreetMap/frozen-test-snapshot",
        acquired_at_utc="2026-01-01T00:00:00Z",
        raw_content_hash="a" * 64,
        preprocessing_version="property-06-v1",
        metric_crs="EPSG:5179",
        network_content_hash="b" * 64,
        source_edge_ids=tuple(sorted(source_edges)),
    )
    return RegisteredMap(
        map_id=map_id,
        data_kind=DataKind.ACTUAL_OSM_MAP,
        scenario=MapScenario.INTERIOR_CONTAINED,
        network_hash="b" * 64,
        boundary_arc_ids=(),
        outcome_domain=frozenset({EpisodeOutcome.CAPTURE, EpisodeOutcome.TIMEOUT}),
        provenance=provenance,
    )


def _partition(
    scope: SplitScope,
    polygon: MetricPolygon,
    buffer_m: float,
    source_edges: frozenset[str],
    buffer_clear: bool,
) -> SpatialPartition:
    min_x, min_y, max_x, max_y = polygon.geometry().bounds
    y = (min_y + max_y) / 2.0
    safe_left, safe_right = min_x + 2 * buffer_m, max_x - 2 * buffer_m
    nodes = [SplitNode("safe-left", (safe_left, y)), SplitNode("safe-right", (safe_right, y))]
    edges = [SplitEdge("safe", ((safe_left, y), (safe_right, y)), tuple(sorted(source_edges)))]
    if not buffer_clear and scope is SplitScope.TRAIN:
        bad_x = min_x + buffer_m / 2.0
        nodes.append(SplitNode("buffer-node", (bad_x, y)))
        edges.append(SplitEdge("buffer-edge", ((bad_x, y), (safe_left, y)), (min(source_edges),)))
    network = NetworkSlice(
        map_id=f"{scope.value}-map",
        city_id="Daejeon",
        metric_crs="EPSG:5179",
        nodes=tuple(nodes),
        edges=tuple(edges),
    )
    return SpatialPartition(
        scope=scope,
        registered_map=_registered_actual(network.map_id, source_edges),
        polygon=polygon,
        network=network,
        polygon_hash=polygon.geometry_hash,
        network_hash=network.network_content_hash,
    )


def _cross_city(city: str) -> CrossCityMapRef:
    registered = _registered_actual(f"{city.lower()}-map", frozenset({f"{city}/osm-edge"}))
    return CrossCityMapRef(city, registered, "f" * 64)


def _build(case: SpatialCase) -> tuple[SpatialSplit, SplitProtocol]:
    protocol = SplitProtocol(metric_crs="EPSG:5179", buffer_m=case.buffer_m)
    scopes = (SplitScope.TRAIN, SplitScope.VALIDATION, SplitScope.TEST)
    partitions = tuple(
        _partition(scope, polygon, case.buffer_m, sources, case.buffer_clear)
        for scope, polygon, sources in zip(scopes, case.polygons, case.source_sets)
    )
    return SpatialSplit(
        train=partitions[0],
        validation=partitions[1],
        test=partitions[2],
        cross_city=(_cross_city("Seoul"), _cross_city("Busan")),
    ), protocol


@settings(max_examples=100, deadline=None)
@given(case=spatial_split_cases())
def test_property_06_spatial_splits_are_geometrically_and_genealogically_disjoint(case):
    """Property 6: Spatial splits are geometrically and genealogically disjoint.

    **Validates: Requirements 6.2, 6.8, 19.3**
    """
    split, protocol = _build(case)
    report = inspect_spatial_split(split, protocol)
    expected = case.polygon_non_overlap and case.buffer_clear and case.source_disjoint
    error_codes = {error.code for error in report.errors}

    assert protocol.buffer_m > 0
    assert report.eligible_for_generalization is expected
    assert (report.errors == ()) is expected
    assert generalization_evidence((report,)) == ((report,) if expected else ())

    if not case.polygon_non_overlap:
        assert "POLYGON_OVERLAP" in error_codes
    if not case.buffer_clear:
        assert "BUFFER_ZONE_VIOLATION" in error_codes
    if not case.source_disjoint:
        assert "SHARED_SOURCE_EDGE" in error_codes

    if expected:
        assert validate_spatial_split(split, protocol) == report
    else:
        with pytest.raises(SplitValidationError) as raised:
            validate_spatial_split(split, protocol)
        assert raised.value.report == report

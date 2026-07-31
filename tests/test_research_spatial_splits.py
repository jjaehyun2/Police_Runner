"""Task 2.2 unit tests for spatial split and tuning leakage gates."""

from dataclasses import replace
from inspect import signature

import pytest

from pursuit_evasion_rl.osm_demo.models import Intersection, ModelNetwork, Segment
from pursuit_evasion_rl.research.canonical import content_hash
from pursuit_evasion_rl.research.domain import DataKind, MapScenario
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.maps.registry import (
    ActualOSMProvenance,
    ActualOSMSpec,
    FixtureSpec,
    MapRegistry,
    SyntheticFixtureProvenance,
)
from pursuit_evasion_rl.research.maps.splits import (
    CrossCityMapRef,
    DataHandle,
    HandleKind,
    MetricPolygon,
    NetworkSlice,
    SpatialPartition,
    SpatialSplit,
    SplitEdge,
    SplitNode,
    SplitProtocol,
    SplitScope,
    SplitValidationError,
    TuningDataView,
    build_spatial_partition,
    generalization_evidence,
    inspect_spatial_split,
    validate_spatial_split,
)


def _polygon(offset: float) -> MetricPolygon:
    return MetricPolygon(
        exterior=((offset, 0.0), (offset + 100.0, 0.0),
                  (offset + 100.0, 100.0), (offset, 100.0)),
    )


def _network(offset: float, source: str) -> ModelNetwork:
    points = ((offset + 5.0, 50.0), (offset + 20.0, 50.0), (offset + 80.0, 50.0))
    return ModelNetwork(
        intersections=(
            Intersection(0, points[0], f"{source}/node/0", outgoing_segment_ids=(0,)),
            Intersection(1, points[1], f"{source}/node/1", outgoing_segment_ids=(1,), incoming_segment_ids=(0,)),
            Intersection(2, points[2], f"{source}/node/2", incoming_segment_ids=(1,)),
        ),
        segments=(
            Segment(0, 0, 1, 15.0, (points[0], points[1]), (f"{source}/edge/buffer",)),
            Segment(1, 1, 2, 60.0, (points[1], points[2]), (f"{source}/edge/interior",)),
        ),
    )


def _registered_actual(network: ModelNetwork, map_id: str):
    source_edges = tuple(
        edge_ref for segment in network.segments for edge_ref in segment.source_edge_refs
    )
    provenance = ActualOSMProvenance(
        query_geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
        source="OpenStreetMap/Overpass",
        acquired_at_utc="2026-01-01T00:00:00Z",
        raw_content_hash="a" * 64,
        preprocessing_version="split-test-v1",
        metric_crs="EPSG:5179",
        network_content_hash=content_hash(network),
        source_edge_ids=source_edges,
    )
    return MapRegistry().register_actual(
        ActualOSMSpec(
            map_id=map_id,
            scenario=MapScenario.INTERIOR_CONTAINED,
            network=network,
            provenance=provenance,
        )
    )


def _partition(scope: SplitScope, offset: float, source: str, protocol: SplitProtocol) -> SpatialPartition:
    network = _network(offset, source)
    registered_map = _registered_actual(network, f"{scope.value}-map")
    return build_spatial_partition(
        scope=scope, polygon=_polygon(offset), network=network,
        registered_map=registered_map, city_id="Daejeon", protocol=protocol,
    )


def _cross_city(city: str, map_id: str, source: str) -> CrossCityMapRef:
    network = _network(0.0, source)
    return CrossCityMapRef(
        city_id=city,
        registered_map=_registered_actual(network, map_id),
        frozen_policy_hash="f" * 64,
    )


def _cross_cities() -> tuple[CrossCityMapRef, ...]:
    return (
        _cross_city("Seoul", "seoul-map", "seoul"),
        _cross_city("Busan", "busan-map", "busan"),
    )


def _split(*, validation_offset: float = 120.0, shared_source: bool = False) -> tuple[SpatialSplit, SplitProtocol]:
    protocol = SplitProtocol(metric_crs="EPSG:5179", buffer_m=10.0)
    return SpatialSplit(
        train=_partition(SplitScope.TRAIN, 0.0, "train", protocol),
        validation=_partition(
            SplitScope.VALIDATION, validation_offset,
            "train" if shared_source else "validation", protocol,
        ),
        test=_partition(SplitScope.TEST, 240.0, "test", protocol),
        cross_city=_cross_cities(),
    ), protocol


def test_builder_excludes_every_node_and_edge_touching_the_protocol_buffer():
    split, protocol = _split()

    for partition in (split.train, split.validation, split.test):
        assert partition.excluded_node_ids == ("0",)
        assert partition.excluded_edge_ids == ("0",)
        assert {node.node_id for node in partition.network.nodes} == {"1", "2"}
        assert {edge.edge_id for edge in partition.network.edges} == {"1"}
        assert partition.polygon_hash == partition.polygon.geometry_hash
        assert partition.network_hash == partition.network.network_content_hash

    report = validate_spatial_split(split, protocol)
    assert report.eligible_for_generalization
    assert report.errors == ()
    assert len(report.report_hash) == 64
    assert report.verify_hash()


def test_polygon_overlap_alone_fails_with_exact_structured_code():
    split, protocol = _split(validation_offset=50.0)

    report = inspect_spatial_split(split, protocol)
    assert {error.code for error in report.errors} == {"POLYGON_OVERLAP"}
    with pytest.raises(SplitValidationError) as raised:
        validate_spatial_split(split, protocol)
    assert raised.value.code == "POLYGON_OVERLAP"
    assert raised.value.report.report_hash == raised.value.record.details["validation_report_hash"]


def test_buffer_violation_alone_fails_with_exact_structured_code():
    split, protocol = _split()
    train = split.train
    violating_network = NetworkSlice(
        map_id=train.network.map_id, city_id=train.network.city_id,
        metric_crs=train.network.metric_crs,
        nodes=train.network.nodes + (SplitNode("bad", (5.0, 50.0)),),
        edges=train.network.edges + (
            SplitEdge(
                "bad", ((5.0, 50.0), (20.0, 50.0)), ("train/edge/buffer",)
            ),
        ),
    )
    violating_train = replace(
        train, network=violating_network,
        network_hash=violating_network.network_content_hash, content_hash=None,
    )
    bad_split = replace(split, train=violating_train, content_hash=None)

    report = inspect_spatial_split(bad_split, protocol)
    assert {error.code for error in report.errors} == {"BUFFER_ZONE_VIOLATION"}
    assert report.errors[0].actual == {"node_ids": ["bad"], "edge_ids": ["bad"]}
    with pytest.raises(SplitValidationError) as raised:
        validate_spatial_split(bad_split, protocol)
    assert raised.value.code == "BUFFER_ZONE_VIOLATION"


def test_shared_source_edge_alone_fails_with_exact_structured_code():
    split, protocol = _split(shared_source=True)

    report = inspect_spatial_split(split, protocol)
    assert {error.code for error in report.errors} == {"SHARED_SOURCE_EDGE"}
    assert report.errors[0].actual == ["train/edge/interior"]
    with pytest.raises(SplitValidationError) as raised:
        validate_spatial_split(split, protocol)
    assert raised.value.code == "SHARED_SOURCE_EDGE"


def test_polygon_network_hash_and_metric_crs_are_verified():
    split, protocol = _split()
    bad_polygon = replace(split.train, polygon_hash="0" * 64, content_hash=None)
    bad_network = replace(split.validation, network_hash="1" * 64, content_hash=None)
    wrong_crs_slice = replace(split.test.network, metric_crs="EPSG:5186", content_hash=None)
    wrong_crs = replace(
        split.test, network=wrong_crs_slice,
        network_hash=wrong_crs_slice.network_content_hash, content_hash=None,
    )
    bad_split = replace(
        split, train=bad_polygon, validation=bad_network, test=wrong_crs, content_hash=None,
    )

    assert {error.code for error in inspect_spatial_split(bad_split, protocol).errors} == {
        "POLYGON_HASH_MISMATCH", "NETWORK_HASH_MISMATCH", "SPLIT_CRS_MISMATCH",
    }


def test_failed_reports_are_excluded_from_generalization_evidence():
    valid_split, protocol = _split()
    invalid_split, _ = _split(shared_source=True)
    valid = inspect_spatial_split(valid_split, protocol)
    invalid = inspect_spatial_split(invalid_split, protocol)

    assert generalization_evidence((invalid, valid)) == (valid,)
    assert not invalid.eligible_for_generalization


def test_zero_shot_requires_two_non_training_cities_and_one_frozen_policy():
    split, protocol = _split()
    one_city = replace(split, cross_city=split.cross_city[:1], content_hash=None)
    training_city = replace(
        split.cross_city[1], city_id="Daejeon", content_hash=None,
    )
    leaked = replace(split, cross_city=(split.cross_city[0], training_city), content_hash=None)
    drifted_policy = replace(
        split.cross_city[1], frozen_policy_hash="e" * 64, content_hash=None,
    )
    drifted = replace(split, cross_city=(split.cross_city[0], drifted_policy), content_hash=None)

    assert "INSUFFICIENT_ZERO_SHOT_CITIES" in {
        error.code for error in inspect_spatial_split(one_city, protocol).errors
    }
    assert "ZERO_SHOT_TRAIN_CITY_LEAKAGE" in {
        error.code for error in inspect_spatial_split(leaked, protocol).errors
    }
    assert "ZERO_SHOT_POLICY_DRIFT" in {
        error.code for error in inspect_spatial_split(drifted, protocol).errors
    }


@pytest.mark.parametrize("scope", [SplitScope.TEST, SplitScope.CROSS_CITY])
def test_tuning_runtime_boundary_rejects_held_out_handles(scope):
    train = DataHandle(SplitScope.TRAIN, HandleKind.MAP, "train")
    leaked = DataHandle(scope, HandleKind.METRIC, "held-out")

    with pytest.raises(ResearchValidationError) as raised:
        TuningDataView(train=(train, leaked), validation=())
    assert raised.value.code == "TUNING_DATA_LEAKAGE"


def test_tuning_api_surface_contains_only_typed_train_and_validation_views():
    split, _ = _split()
    view = TuningDataView.from_split(split)

    assert tuple(signature(TuningDataView).parameters) == ("train", "validation")
    assert [handle.identifier for handle in view.train] == ["train-map"]
    assert [handle.identifier for handle in view.validation] == ["validation-map"]
    assert not hasattr(view, "test")
    assert not hasattr(view, "cross_city")


def test_split_protocol_requires_positive_metric_buffer():
    with pytest.raises(ResearchValidationError) as raised:
        SplitProtocol(metric_crs="EPSG:5179", buffer_m=0.0)
    assert raised.value.code == "INVALID_SPLIT_BUFFER"


def test_partition_builder_reuses_registry_and_rejects_network_identity_drift():
    protocol = SplitProtocol(metric_crs="EPSG:5179", buffer_m=10.0)
    registered_network = _network(0.0, "registered")
    registered_map = _registered_actual(registered_network, "registered-map")

    with pytest.raises(ResearchValidationError) as raised:
        build_spatial_partition(
            scope=SplitScope.TRAIN,
            polygon=_polygon(120.0),
            network=_network(120.0, "different"),
            registered_map=registered_map,
            city_id="Daejeon",
            protocol=protocol,
        )
    assert raised.value.code == "SPLIT_REGISTRY_NETWORK_MISMATCH"


def test_cross_city_reference_requires_registered_actual_osm():
    network = _network(0.0, "fixture")
    provenance = SyntheticFixtureProvenance(
        generator_path="tests.make_split_fixture",
        generator_version="1.0",
        parameters={"city": "Seoul"},
        seed=1,
        fixture_content_hash=content_hash(network),
        purpose="reject fixture as zero-shot OSM evidence",
    )
    registered_fixture = MapRegistry().register_fixture(
        FixtureSpec(
            map_id="fixture-map",
            scenario=MapScenario.INTERIOR_CONTAINED,
            network=network,
            provenance=provenance,
        )
    )

    with pytest.raises(ResearchValidationError) as raised:
        CrossCityMapRef("Seoul", registered_fixture, "f" * 64)
    assert raised.value.code == "ZERO_SHOT_REQUIRES_ACTUAL_OSM"
    assert registered_fixture.data_kind is DataKind.SYNTHETIC_FIXTURE

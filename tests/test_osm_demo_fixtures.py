"""Offline tests for committed raw/model fixtures (Requirements 4.3-4.7, 13.10-13.12)."""
from __future__ import annotations

import pytest

from pursuit_evasion_rl.osm_demo import fixtures
from pursuit_evasion_rl.osm_demo.canonical import canonical_json, content_hash
from pursuit_evasion_rl.osm_demo.coarsening import (
    MAX_OUT_DEGREE,
    coarsen_raw_graph,
    prepare_model_network,
)
from pursuit_evasion_rl.osm_demo.models import RawOSMGraph
from pursuit_evasion_rl.osm_demo.presets import DAEJEON_DRIVE_PRESET

pytestmark = pytest.mark.offline

ALL_FIXTURE_NAMES = (
    "one_way",
    "bidirectional",
    "parallel_edge",
    "degree_six",
    "boundary_crossing",
    "daejeon",
)


def test_registry_exposes_every_named_fixture():
    assert set(fixtures.OFFLINE_FIXTURES) == set(ALL_FIXTURE_NAMES)
    assert set(fixtures.FIXTURE_RAW_HASHES) == set(ALL_FIXTURE_NAMES)


@pytest.mark.parametrize("name", ALL_FIXTURE_NAMES)
def test_fixture_matches_committed_hash_and_is_a_raw_graph(name):
    graph = fixtures.build_fixture(name)
    assert isinstance(graph, RawOSMGraph)
    # A committed hash guards against silent fixture drift (Req 13.10-13.12).
    assert content_hash(graph) == fixtures.fixture_raw_hash(name)


@pytest.mark.parametrize("name", ALL_FIXTURE_NAMES)
def test_every_fixture_coarsens_and_prepares_without_external_service(name):
    graph = fixtures.build_fixture(name)
    prepared = prepare_model_network(coarsen_raw_graph(graph))

    assert prepared.network.intersections
    assert prepared.network.segments
    # Degree splitting guarantees the model out-degree limit holds everywhere.
    assert prepared.statistics.max_out_degree <= MAX_OUT_DEGREE


@pytest.mark.parametrize("name", ALL_FIXTURE_NAMES)
def test_fixture_factories_are_deterministic_and_independent(name):
    first = fixtures.build_fixture(name)
    second = fixtures.build_fixture(name)
    assert first is not second
    assert canonical_json(first) == canonical_json(second)


def test_one_way_chain_collapses_to_single_segment():
    prepared = prepare_model_network(coarsen_raw_graph(fixtures.one_way()))
    assert len(prepared.network.intersections) == 2
    assert len(prepared.network.segments) == 1


def test_bidirectional_keeps_two_directed_segments():
    prepared = prepare_model_network(coarsen_raw_graph(fixtures.bidirectional()))
    assert len(prepared.network.segments) == 2


def test_parallel_edges_deduplicate_but_map_every_source_edge():
    result = coarsen_raw_graph(fixtures.parallel_edge())
    assert len(result.network.segments) == 1
    mapped = result.mapping.segment_sources["0"]
    assert len(mapped) == 2


def test_degree_six_hub_requires_deterministic_degree_splitting():
    result = coarsen_raw_graph(fixtures.degree_six())
    # The raw hub exceeds the model out-degree limit before splitting.
    hub_out_degree = max(
        len(item.outgoing_segment_ids) for item in result.network.intersections
    )
    assert hub_out_degree > MAX_OUT_DEGREE

    prepared = prepare_model_network(result)
    assert prepared.statistics.max_out_degree <= MAX_OUT_DEGREE
    assert prepared.statistics.virtual_intersection_count >= 1


def test_boundary_crossing_fixture_has_two_boundary_intersections():
    prepared = prepare_model_network(coarsen_raw_graph(fixtures.boundary_crossing()))
    boundary_intersections = [
        item for item in prepared.network.intersections if item.boundary_kind is not None
    ]
    assert len(boundary_intersections) == 2
    assert any(segment.crosses_boundary for segment in prepared.network.segments)


def test_daejeon_fixture_source_serves_the_preset_area_offline():
    source = fixtures.daejeon_fixture_source()
    acquisition = source.fetch_with_metadata(DAEJEON_DRIVE_PRESET.area)

    assert acquisition.metadata.operation_status == "READY"
    assert acquisition.metadata.validation["offline"] is True
    prepared = prepare_model_network(coarsen_raw_graph(acquisition.graph))
    assert len(prepared.network.intersections) == 9
    assert prepared.statistics.max_out_degree <= MAX_OUT_DEGREE

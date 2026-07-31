"""Task 2.4 offline integration tests for map registry and snapshots.

Validates the integration boundaries behind Requirements 5.1–5.10, 6.1,
19.2, and 19.11–19.12 (Correctness Properties 5 and 6).
"""

from datetime import datetime, timezone
import socket

import pytest

from pursuit_evasion_rl.osm_demo.models import Intersection, ModelNetwork, Segment
from pursuit_evasion_rl.research.canonical import canonical_json, content_hash, sha256_bytes
from pursuit_evasion_rl.research.domain import (
    DataKind,
    EpisodeOutcome,
    ExecutionStatus,
    MapScenario,
)
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.maps.registry import (
    ActualOSMSpec,
    EpisodeResult,
    FixtureSpec,
    MapRegistry,
    SyntheticFixtureProvenance,
)
from pursuit_evasion_rl.research.maps.snapshots import (
    ExternalFetchRequest,
    ExternalOSMFetchAdapter,
    OfflineSnapshotStore,
    SnapshotRole,
    SnapshotSpec,
)

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def no_external_network(monkeypatch):
    """Make an accidental socket connection fail before leaving the process."""
    calls = []

    def blocked(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("external network access is forbidden in offline tests")

    monkeypatch.setattr(socket, "create_connection", blocked)
    yield calls
    assert calls == []


def _network(*, boundary: bool, edge_ref: str) -> ModelNetwork:
    return ModelNetwork(
        intersections=(
            Intersection(
                0, (0.0, 0.0), f"{edge_ref}/start",
                outgoing_segment_ids=(0,),
            ),
            Intersection(
                1, (10.0, 0.0), f"{edge_ref}/end",
                boundary_kind="bbox" if boundary else None,
                incoming_segment_ids=(0,),
            ),
        ),
        segments=(
            Segment(
                0, 0, 1, 10.0, ((0.0, 0.0), (10.0, 0.0)),
                source_edge_refs=(edge_ref,), crosses_boundary=boundary,
            ),
        ),
    )


def _cached_spec(
    tmp_path, *, map_id: str, role: SnapshotRole, network: ModelNetwork
) -> SnapshotSpec:
    raw_bytes = canonical_json({"map_id": map_id, "source": "cached OSM"})
    network_bytes = canonical_json(network)
    raw_path = f"cache/research/maps/{map_id}/raw_osm.json"
    network_path = f"cache/research/maps/{map_id}/model_network.json"
    (tmp_path / raw_path).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / raw_path).write_bytes(raw_bytes)
    (tmp_path / network_path).write_bytes(network_bytes)
    geometry = {
        "type": "Polygon",
        "coordinates": [[[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 0.0]]],
    }
    return SnapshotSpec(
        map_id=map_id,
        city="Daejeon",
        role=role,
        query=f"sealed query for {map_id}",
        query_geometry=geometry,
        source="OpenStreetMap contributors",
        service="Overpass API",
        service_version="0.7.62",
        acquired_at_utc="2026-01-02T00:00:00Z",
        preprocessing_version="integration-coarsener-v1",
        metric_crs="EPSG:5179",
        raw_path=raw_path,
        network_path=network_path,
        raw_file_hash=sha256_bytes(raw_bytes),
        network_file_hash=sha256_bytes(network_bytes),
        network_content_hash=content_hash(network),
        polygon_content_hash=content_hash(geometry),
    )


def _fixture_provenance(network: ModelNetwork, purpose: str):
    return SyntheticFixtureProvenance(
        generator_path="tests.research.test_map_registry_integration._network",
        generator_version="1.0",
        parameters={"boundary": any(s.crosses_boundary for s in network.segments)},
        seed=24,
        fixture_content_hash=content_hash(network),
        purpose=purpose,
    )


@pytest.fixture
def registered_maps(tmp_path):
    registry = MapRegistry()
    store = OfflineSnapshotStore(tmp_path)
    records = {}

    actual_inputs = (
        ("actual-interior", SnapshotRole.TRAIN, MapScenario.INTERIOR_CONTAINED, False),
        ("actual-boundary", SnapshotRole.TEST, MapScenario.BOUNDARY_ESCAPE, True),
    )
    for map_id, role, scenario, boundary in actual_inputs:
        network = _network(boundary=boundary, edge_ref=f"osm-way/{map_id}/0")
        spec = _cached_spec(tmp_path, map_id=map_id, role=role, network=network)
        imported = store.import_snapshot(spec)
        assert imported.status is ExecutionStatus.COMPLETED
        assert imported.details == {"external_requests": 0}
        assert imported.snapshot.spec_content_hash == spec.content_hash
        assert imported.snapshot.network_content_hash == spec.network_content_hash
        assert spec.polygon_content_hash == content_hash(spec.query_geometry)
        records[map_id] = registry.register_actual(
            ActualOSMSpec(
                map_id=map_id,
                scenario=scenario,
                network=imported.snapshot.network,
                provenance=imported.snapshot.actual_osm_provenance,
            )
        )

    fixture_inputs = (
        ("fixture-interior", MapScenario.INTERIOR_CONTAINED, False),
        ("fixture-boundary", MapScenario.BOUNDARY_ESCAPE, True),
    )
    for map_id, scenario, boundary in fixture_inputs:
        network = _network(boundary=boundary, edge_ref=f"fixture-edge/{map_id}/0")
        records[map_id] = registry.register_fixture(
            FixtureSpec(
                map_id=map_id,
                scenario=scenario,
                network=network,
                provenance=_fixture_provenance(network, "offline integration control"),
            )
        )
    return registry, records


def _episode(record, episode_id: str, outcome: EpisodeOutcome) -> EpisodeResult:
    return EpisodeResult(
        episode_id=episode_id,
        map_id=record.map_id,
        map_hash=record.content_hash,
        data_kind=record.data_kind,
        scenario=record.scenario,
        outcome=outcome,
    )


def test_cached_actual_and_fixture_results_form_four_separate_tables(registered_maps):
    registry, maps = registered_maps
    episodes = (
        _episode(maps["actual-interior"], "ai-capture", EpisodeOutcome.CAPTURE),
        _episode(maps["actual-boundary"], "ab-escape", EpisodeOutcome.ESCAPE),
        _episode(maps["fixture-interior"], "fi-timeout", EpisodeOutcome.TIMEOUT),
        _episode(maps["fixture-boundary"], "fb-capture", EpisodeOutcome.CAPTURE),
    )

    tables = registry.aggregate(episodes)
    by_stratum = {(table.data_kind, table.scenario): table for table in tables}
    assert set(by_stratum) == {
        (DataKind.ACTUAL_OSM_MAP, MapScenario.INTERIOR_CONTAINED),
        (DataKind.ACTUAL_OSM_MAP, MapScenario.BOUNDARY_ESCAPE),
        (DataKind.SYNTHETIC_FIXTURE, MapScenario.INTERIOR_CONTAINED),
        (DataKind.SYNTHETIC_FIXTURE, MapScenario.BOUNDARY_ESCAPE),
    }
    assert by_stratum[
        DataKind.ACTUAL_OSM_MAP, MapScenario.INTERIOR_CONTAINED
    ].outcome_counts == {"capture": 1, "timeout": 0}
    assert by_stratum[
        DataKind.SYNTHETIC_FIXTURE, MapScenario.BOUNDARY_ESCAPE
    ].outcome_counts == {"capture": 1, "escape": 0, "timeout": 0}


def test_single_table_aggregate_rejects_kind_or_scenario_merges(registered_maps):
    registry, maps = registered_maps
    actual_interior = _episode(
        maps["actual-interior"], "actual-interior", EpisodeOutcome.CAPTURE
    )
    fixture_interior = _episode(
        maps["fixture-interior"], "fixture-interior", EpisodeOutcome.TIMEOUT
    )
    actual_boundary = _episode(
        maps["actual-boundary"], "actual-boundary", EpisodeOutcome.ESCAPE
    )

    for mixed in (
        (actual_interior, fixture_interior),
        (actual_interior, actual_boundary),
    ):
        with pytest.raises(ResearchValidationError) as raised:
            registry.aggregate_single_stratum(mixed)
        assert raised.value.code == "MIXED_AGGREGATE_STRATA"
        assert raised.value.expected == "exactly one data-kind×scenario stratum"
        assert len(raised.value.actual) == 2

    table = registry.aggregate_single_stratum((actual_interior,))
    assert table.data_kind is DataKind.ACTUAL_OSM_MAP
    assert table.scenario is MapScenario.INTERIOR_CONTAINED


def test_optional_external_adapter_is_opt_in_and_records_complete_provenance():
    network = _network(boundary=False, edge_ref="osm-way/optional/0")
    raw_bytes = canonical_json({"source": "deterministic local transport"})
    network_bytes = canonical_json(network)
    transport_calls = []
    adapter = ExternalOSMFetchAdapter(
        lambda request: transport_calls.append(request) or (raw_bytes, network_bytes)
    )
    geometry = {
        "type": "Polygon",
        "coordinates": [[[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 0.0]]],
    }
    request = ExternalFetchRequest(
        integration_run_id="optional-osm-integration-24",
        query="sealed optional integration query",
        query_geometry=geometry,
        service="Overpass API",
        service_version="0.7.62",
        preprocessing_version="integration-coarsener-v1",
    )

    with pytest.raises(ResearchValidationError) as blocked:
        adapter.fetch(request, opt_in=False)
    assert blocked.value.code == "EXTERNAL_FETCH_OPT_IN_REQUIRED"
    assert transport_calls == []

    _, _, provenance = adapter.fetch(
        request,
        opt_in=True,
        clock=lambda: datetime(2026, 1, 3, tzinfo=timezone.utc),
    )
    assert transport_calls == [request]
    assert provenance.integration_run_id == "optional-osm-integration-24"
    assert provenance.service == "Overpass API"
    assert provenance.service_version == "0.7.62"
    assert provenance.executed_at_utc == "2026-01-03T00:00:00+00:00"
    assert provenance.input_content_hash == request.content_hash
    assert provenance.raw_output_hash == sha256_bytes(raw_bytes)
    assert provenance.network_output_hash == sha256_bytes(network_bytes)
    assert provenance.network_content_hash == content_hash(network)
    assert provenance.output_content_hash == content_hash({
        "raw_output_hash": provenance.raw_output_hash,
        "network_output_hash": provenance.network_output_hash,
        "network_content_hash": provenance.network_content_hash,
    })

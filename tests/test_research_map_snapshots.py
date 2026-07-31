"""Task 2.3 unit tests for offline immutable map snapshots."""

from datetime import datetime, timezone
import socket

import pytest
import yaml

from pursuit_evasion_rl.osm_demo.models import Intersection, ModelNetwork, Segment
from pursuit_evasion_rl.research.canonical import canonical_json, content_hash, sha256_bytes
from pursuit_evasion_rl.research.domain import ExecutionStatus
from pursuit_evasion_rl.research.errors import ResearchValidationError
from pursuit_evasion_rl.research.maps.snapshots import (
    ExternalFetchProvenance,
    ExternalFetchRequest,
    ExternalOSMFetchAdapter,
    OfflineSnapshotStore,
    SnapshotRole,
    SnapshotSpec,
    load_snapshot_registry,
    load_snapshot_spec,
    write_immutable_snapshot,
)
from pursuit_evasion_rl.research.cli.fetch_osm_snapshot import build_parser, run_integration

pytestmark = pytest.mark.offline


def _network() -> ModelNetwork:
    return ModelNetwork(
        intersections=(
            Intersection(0, (0.0, 0.0), "osm-node/1", outgoing_segment_ids=(0,)),
            Intersection(1, (10.0, 0.0), "osm-node/2", incoming_segment_ids=(0,)),
        ),
        segments=(
            Segment(
                0, 0, 1, 10.0, ((0.0, 0.0), (10.0, 0.0)),
                source_edge_refs=("osm-way/10/0",),
            ),
        ),
    )


def _spec(raw_bytes: bytes, network_bytes: bytes, **changes) -> SnapshotSpec:
    geometry = {
        "type": "Polygon",
        "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]],
    }
    values = {
        "map_id": "daejeon-train", "city": "Daejeon", "role": SnapshotRole.TRAIN,
        "query": "fixed query", "query_geometry": geometry,
        "source": "OpenStreetMap contributors", "service": "Overpass API",
        "service_version": "0.7.62", "acquired_at_utc": "2026-01-01T00:00:00Z",
        "preprocessing_version": "snapshot-test-v1", "metric_crs": "EPSG:5179",
        "raw_path": "cache/maps/raw.json", "network_path": "cache/maps/network.json",
        "raw_file_hash": sha256_bytes(raw_bytes),
        "network_file_hash": sha256_bytes(network_bytes),
        "network_content_hash": content_hash(_network()),
        "polygon_content_hash": content_hash(geometry),
    }
    values.update(changes)
    return SnapshotSpec(**values)


def test_cached_snapshot_import_is_reproducible_with_network_blocked(tmp_path, monkeypatch):
    raw_bytes = b'{"osm":"fixed"}'
    network_bytes = canonical_json(_network())
    raw_path = tmp_path / "cache/maps/raw.json"
    network_path = tmp_path / "cache/maps/network.json"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(raw_bytes)
    network_path.write_bytes(network_bytes)

    calls = []
    def blocked(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("network access is forbidden in offline snapshot import")
    monkeypatch.setattr(socket, "create_connection", blocked)

    store = OfflineSnapshotStore(tmp_path)
    first = store.import_snapshot(_spec(raw_bytes, network_bytes))
    second = store.import_snapshot(_spec(raw_bytes, network_bytes))

    assert first.status is ExecutionStatus.COMPLETED
    assert first.reason_code == "SNAPSHOT_VERIFIED"
    assert first.content_hash == second.content_hash
    assert first.snapshot.network_content_hash == content_hash(_network())
    assert first.snapshot.actual_osm_provenance.network_content_hash == content_hash(_network())
    assert first.snapshot.actual_osm_provenance.source_edge_ids == ("osm-way/10/0",)
    assert first.details["external_requests"] == 0
    assert calls == []


def test_cache_miss_is_structured_not_run_and_never_calls_transport(tmp_path):
    raw_bytes = b'{"osm":"fixed"}'
    network_bytes = canonical_json(_network())
    result = OfflineSnapshotStore(tmp_path).import_snapshot(_spec(raw_bytes, network_bytes))

    assert result.status is ExecutionStatus.NOT_RUN
    assert result.reason_code == "SNAPSHOT_CACHE_MISS"
    assert result.snapshot is None
    assert result.details["external_requests"] == 0
    assert result.details["missing_paths"] == [
        "cache\\maps\\raw.json" if __import__("os").name == "nt" else "cache/maps/raw.json",
        "cache\\maps\\network.json" if __import__("os").name == "nt" else "cache/maps/network.json",
    ]


def test_unsealed_existing_snapshot_is_not_run_and_tamper_is_failed(tmp_path):
    raw_bytes = b'{"osm":"fixed"}'
    network_bytes = canonical_json(_network())
    raw_path = tmp_path / "cache/maps/raw.json"
    network_path = tmp_path / "cache/maps/network.json"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(raw_bytes)
    network_path.write_bytes(network_bytes)
    store = OfflineSnapshotStore(tmp_path)

    unsealed = store.import_snapshot(_spec(raw_bytes, network_bytes, network_file_hash=None))
    assert unsealed.status is ExecutionStatus.NOT_RUN
    assert unsealed.reason_code == "SNAPSHOT_UNSEALED"

    raw_path.write_bytes(raw_bytes + b" ")
    tampered = store.import_snapshot(_spec(raw_bytes, network_bytes))
    assert tampered.status is ExecutionStatus.FAILED
    assert tampered.reason_code == "SNAPSHOT_HASH_MISMATCH"
    assert "raw_file_hash" in tampered.details["mismatches"]


def test_complete_hashes_do_not_promote_placeholder_provenance_to_actual_osm(tmp_path):
    raw_bytes = b'{"osm":"fixed"}'
    network_bytes = canonical_json(_network())
    raw_path = tmp_path / "cache/maps/raw.json"
    network_path = tmp_path / "cache/maps/network.json"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes(raw_bytes)
    network_path.write_bytes(network_bytes)

    result = OfflineSnapshotStore(tmp_path).import_snapshot(
        _spec(raw_bytes, network_bytes, preprocessing_version="pending-integration")
    )

    assert result.status is ExecutionStatus.NOT_RUN
    assert result.reason_code == "SNAPSHOT_PROVENANCE_INCOMPLETE"
    assert result.snapshot is None
    assert result.details == {
        "external_requests": 0,
        "incomplete_fields": ["preprocessing_version"],
    }


def test_default_store_rejects_paths_outside_cache_or_fixture(tmp_path):
    raw_bytes = b"{}"
    network_bytes = canonical_json(_network())
    with pytest.raises(ResearchValidationError) as raised:
        OfflineSnapshotStore(tmp_path).import_snapshot(
            _spec(raw_bytes, network_bytes, raw_path="artifacts/raw.json")
        )
    assert raised.value.code == "SNAPSHOT_PATH_OUTSIDE_OFFLINE_ROOT"


def test_yaml_loader_and_committed_registry_require_daejeon_and_two_zero_shot_cities():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    spec = load_snapshot_spec(root / "configs/research/maps/daejeon_train.yaml")
    specs = load_snapshot_registry(
        root / "artifacts/research/maps/registry.json", repository_root=root
    )

    assert spec.role is SnapshotRole.TRAIN
    assert {item.role for item in specs if item.city == "Daejeon"} == {
        SnapshotRole.TRAIN, SnapshotRole.VALIDATION, SnapshotRole.TEST
    }
    assert {item.city for item in specs if item.role is SnapshotRole.ZERO_SHOT} == {
        "Seoul", "Busan"
    }


def test_yaml_loader_rejects_polygon_hash_drift(tmp_path):
    raw_bytes = b"{}"
    network_bytes = canonical_json(_network())
    data = _spec(raw_bytes, network_bytes).as_dict()
    data.pop("content_hash")
    data["query_geometry"]["coordinates"][0][0][0] = 99.0
    config = tmp_path / "drift.yaml"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ResearchValidationError) as raised:
        load_snapshot_spec(config)
    assert raised.value.code == "POLYGON_HASH_MISMATCH"


def test_external_fetch_adapter_requires_opt_in_and_records_full_provenance():
    calls = []
    raw_bytes = b'{"raw":true}'
    network_bytes = canonical_json(_network())
    adapter = ExternalOSMFetchAdapter(
        lambda request: calls.append(request) or (raw_bytes, network_bytes)
    )
    geometry = {
        "type": "Polygon",
        "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]],
    }
    request = ExternalFetchRequest(
        integration_run_id="osm-integration-001", query="fixed query",
        query_geometry=geometry, service="Overpass API", service_version="0.7.62",
        preprocessing_version="coarsener-1+splitter-1",
    )

    with pytest.raises(ResearchValidationError) as blocked:
        adapter.fetch(request)
    assert blocked.value.code == "EXTERNAL_FETCH_OPT_IN_REQUIRED"
    assert calls == []

    _, _, provenance = adapter.fetch(
        request, opt_in=True,
        clock=lambda: datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    assert calls == [request]
    assert provenance.integration_run_id == "osm-integration-001"
    assert provenance.query == request.query
    assert provenance.query_geometry == geometry
    assert provenance.query_geometry_hash == content_hash(geometry)
    assert provenance.service == "Overpass API"
    assert provenance.service_version == "0.7.62"
    assert provenance.preprocessing_version == "coarsener-1+splitter-1"
    assert provenance.executed_at_utc == "2026-01-02T00:00:00+00:00"
    assert provenance.input_content_hash == request.content_hash
    assert provenance.raw_output_hash == sha256_bytes(raw_bytes)
    assert provenance.network_output_hash == sha256_bytes(network_bytes)
    assert provenance.network_content_hash == content_hash(_network())
    assert provenance.output_content_hash == content_hash({
        "raw_output_hash": provenance.raw_output_hash,
        "network_output_hash": provenance.network_output_hash,
        "network_content_hash": provenance.network_content_hash,
    })

    tampered = provenance.as_dict()
    tampered.pop("content_hash")
    tampered["output_content_hash"] = "0" * 64
    with pytest.raises(ResearchValidationError) as raised:
        ExternalFetchProvenance(**tampered)
    assert raised.value.code == "FETCH_OUTPUT_HASH_MISMATCH"


def test_immutable_snapshot_writer_refuses_overwrite(tmp_path):
    destination = tmp_path / "snapshot.json"
    digest = write_immutable_snapshot(destination, b"first")
    assert digest == sha256_bytes(b"first")
    with pytest.raises(ResearchValidationError) as raised:
        write_immutable_snapshot(destination, b"second")
    assert raised.value.code == "IMMUTABLE_SNAPSHOT_EXISTS"
    assert destination.read_bytes() == b"first"


def test_explicit_integration_command_refuses_network_without_opt_in(tmp_path):
    raw_bytes = b'{"raw":true}'
    network_bytes = canonical_json(_network())
    spec = _spec(raw_bytes, network_bytes)
    config_data = spec.as_dict()
    config_data.pop("content_hash")
    config_path = tmp_path / "source.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    args = build_parser().parse_args([
        str(config_path), "--root", str(tmp_path), "--run-id", "integration-no-opt-in",
        "--service-version", "0.7.62", "--sealed-config-output",
        "configs/research/maps/fetched.yaml",
    ])
    calls = []

    with pytest.raises(ResearchValidationError) as raised:
        run_integration(
            args,
            transport=lambda request: calls.append(request) or (raw_bytes, network_bytes),
        )

    assert raised.value.code == "EXTERNAL_FETCH_OPT_IN_REQUIRED"
    assert calls == []
    assert not (tmp_path / "cache/maps/raw.json").exists()
    assert not (tmp_path / "cache/maps/network.json").exists()

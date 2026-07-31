"""Focused offline tests for immutable atomic cache bundles and verified loading."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from pursuit_evasion_rl.osm_demo.cache import (
    BUNDLE_MANIFEST_ARTIFACT,
    MODEL_NETWORK_ARTIFACT,
    RAW_GRAPH_ARTIFACT,
    CacheError,
    CacheStore,
)
from pursuit_evasion_rl.osm_demo.canonical import canonical_json
from pursuit_evasion_rl.osm_demo.coarsening import (
    COARSENER_VERSION,
    coarsen_raw_graph,
    prepare_model_network,
)
from pursuit_evasion_rl.osm_demo.models import (
    BoundedArea,
    NetworkMetadata,
    RawEdge,
    RawNode,
    RawOSMGraph,
)

pytestmark = pytest.mark.offline

PREPROCESSING_HASH = "settings-hash-abc123"


def _node(source_id: str, x: float, y: float = 0.0, **tags) -> RawNode:
    return RawNode(source_id, 36.355, 127.405, x, y, tags)


def _edge(source_u: str, source_v: str, key: str, positions, **tags) -> RawEdge:
    start, end = positions[source_u], positions[source_v]
    length = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
    return RawEdge(
        source_u,
        source_v,
        key,
        length_m=length,
        geometry_xy=(start, end),
        way_refs=(key,),
        oneway=True,
        road_class="residential",
        tags=tags,
    )


def _raw_graph() -> RawOSMGraph:
    positions = {
        "a": (0.0, 0.0),
        "mid": (10.0, 0.0),
        "b": (25.0, 0.0),
        "c": (25.0, 15.0),
        "d": (40.0, 0.0),
    }
    nodes = tuple(_node(source_id, *position) for source_id, position in positions.items())
    edges = (
        _edge("a", "mid", "e1", positions),
        _edge("mid", "b", "e2", positions),
        _edge("b", "c", "e3", positions),
        _edge("b", "d", "e4", positions),
    )
    return RawOSMGraph(nodes=nodes, edges=edges)


def _bounded_area() -> BoundedArea:
    return BoundedArea(
        name="test-area",
        north=36.36,
        south=36.35,
        east=127.41,
        west=127.40,
        max_area_km2=50.0,
    )


def _metadata(area: BoundedArea) -> NetworkMetadata:
    return NetworkMetadata(
        area=area,
        source="OpenStreetMap contributors via OSMnx",
        acquired_at=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc).isoformat(),
        network_type="drive",
        metric_crs="EPSG:32652",
        schema_versions={"raw_osm": "1.0", "model_network": "1.0"},
        settings_hash=PREPROCESSING_HASH,
        statistics={"raw_nodes": 5},
        validation={"explicit_bbox": True},
    )


def _prepared():
    graph = _raw_graph()
    prepared = prepare_model_network(coarsen_raw_graph(graph))
    area = _bounded_area()
    return graph, prepared, _metadata(area)


def _store(tmp_path, clock=None) -> CacheStore:
    if clock is None:
        return CacheStore(tmp_path)
    return CacheStore(tmp_path, clock=clock)


def _save(store: CacheStore):
    graph, prepared, metadata = _prepared()
    return store.save(
        raw_graph=graph,
        network=prepared.network,
        mapping=prepared.mapping,
        metadata=metadata,
        coarsener_version=COARSENER_VERSION,
        preprocessing_settings_hash=PREPROCESSING_HASH,
    )


def test_save_then_verified_load_round_trips_equivalent_content(tmp_path):
    store = _store(tmp_path)
    saved = _save(store)

    loaded = store.load(saved.cache_key)

    assert canonical_json(loaded.network) == canonical_json(saved.network)
    assert canonical_json(loaded.mapping) == canonical_json(saved.mapping)
    assert canonical_json(loaded.raw_graph) == canonical_json(saved.raw_graph)
    assert loaded.metadata.area.name == "test-area"
    assert loaded.metadata.acquired_at == saved.metadata.acquired_at
    # Attribution and creation/commit time are preserved for exports (Req 4.7).
    provenance = loaded.provenance()
    assert provenance["attribution"] == "OpenStreetMap contributors via OSMnx"
    assert provenance["acquired_at"] == saved.metadata.acquired_at
    assert provenance["committed_at"] == saved.manifest.committed_at


def test_reloading_same_key_reuses_the_committed_entry(tmp_path):
    store = _store(tmp_path)
    saved = _save(store)

    assert store.has_valid_entry(saved.cache_key)
    again = _save(store)
    assert again.cache_key == saved.cache_key
    assert canonical_json(again.network) == canonical_json(saved.network)


def test_mutated_artifact_is_detected_and_quarantined(tmp_path):
    store = _store(tmp_path)
    saved = _save(store)
    entry = tmp_path / saved.cache_key
    artifact = entry / MODEL_NETWORK_ARTIFACT
    tampered = artifact.read_bytes().replace(b'"length_m"', b'"length_x"', 1)
    assert tampered != artifact.read_bytes()
    artifact.write_bytes(tampered)

    with pytest.raises(CacheError) as raised:
        store.load(saved.cache_key)
    assert raised.value.code == "CACHE_CORRUPT"
    assert raised.value.cache_key == saved.cache_key

    # Logically quarantined: never returned again even without re-reading.
    with pytest.raises(CacheError) as requarantined:
        store.load(saved.cache_key)
    assert requarantined.value.code == "CACHE_QUARANTINED"


def test_missing_entry_reports_actionable_offline_miss(tmp_path):
    store = _store(tmp_path)
    _, prepared, metadata = _prepared()
    key = store.cache_key(
        metadata.area,
        coarsener_version=COARSENER_VERSION,
        preprocessing_settings_hash=PREPROCESSING_HASH,
    )

    with pytest.raises(CacheError) as raised:
        store.load_offline(key)

    error = raised.value
    assert error.code == "CACHE_MISS"
    assert error.cache_key == key
    assert error.actual["required_cache_key"] == key
    assert "prepar" in error.actual["preparation"].lower()


def test_absent_commit_marker_is_treated_as_a_miss_not_a_corruption(tmp_path):
    store = _store(tmp_path)
    saved = _save(store)
    (tmp_path / saved.cache_key / BUNDLE_MANIFEST_ARTIFACT).unlink()

    with pytest.raises(CacheError) as raised:
        store.load(saved.cache_key)
    assert raised.value.code == "CACHE_MISS"


def test_metadata_records_content_hashes_matching_stored_artifacts(tmp_path):
    store = _store(tmp_path)
    saved = _save(store)

    from pursuit_evasion_rl.osm_demo.canonical import content_hash

    assert saved.metadata.artifact_hashes["raw_osm"] == content_hash(saved.raw_graph)
    assert saved.metadata.artifact_hashes["model_network"] == content_hash(saved.network)
    assert saved.metadata.artifact_hashes["mapping_manifest"] == content_hash(saved.mapping)
    recorded = {descriptor.name for descriptor in saved.manifest.artifacts}
    assert RAW_GRAPH_ARTIFACT in recorded and MODEL_NETWORK_ARTIFACT in recorded


# ---------------------------------------------------------------------------
# Legacy single-file cache migration boundary (Requirements 4.3-4.7, 13.10-13.12)
# ---------------------------------------------------------------------------


def test_require_migration_detects_legacy_single_file_cache(tmp_path):
    store = _store(tmp_path)
    # The legacy cache wrote a single Overpass JSON file at the cache root.
    (tmp_path / "d24b5791524514f8321ccf2b23a1cc28f8a60d69.json").write_text(
        '{"version": 0.6, "elements": []}', encoding="utf-8"
    )

    assert store.legacy_entries() == (
        "d24b5791524514f8321ccf2b23a1cc28f8a60d69.json",
    )
    with pytest.raises(CacheError) as raised:
        store.require_migration()
    assert raised.value.code == "CACHE_MIGRATION_REQUIRED"
    assert (
        "d24b5791524514f8321ccf2b23a1cc28f8a60d69.json"
        in raised.value.actual["legacy_files"]
    )


def test_require_migration_passes_when_only_immutable_bundles_exist(tmp_path):
    store = _store(tmp_path)
    _save(store)
    # Bundles are directories, never bare root-level JSON files.
    assert store.legacy_entries() == ()
    store.require_migration()  # must not raise


def test_load_refuses_to_silently_reuse_legacy_single_file_for_key(tmp_path):
    store = _store(tmp_path)
    _, prepared, metadata = _prepared()
    key = store.cache_key(
        metadata.area,
        coarsener_version=COARSENER_VERSION,
        preprocessing_settings_hash=PREPROCESSING_HASH,
    )
    # Legacy single-file entry sitting exactly where this key would resolve.
    (tmp_path / f"{key}.json").write_text('{"legacy": true}', encoding="utf-8")

    with pytest.raises(CacheError) as raised:
        store.load(key)
    assert raised.value.code == "CACHE_MIGRATION_REQUIRED"
    assert raised.value.cache_key == key

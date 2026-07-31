"""Focused offline tests for the dependency-injected OSM demo services.

These tests exercise the application-layer use cases wired in
:mod:`pursuit_evasion_rl.osm_demo.service` (Requirements 12.1-12.3) using only
offline fixtures and the deterministic baseline policy, so no external OSM
service, checkpoint or GPU is required.
"""

from __future__ import annotations

import pytest

from pursuit_evasion_rl.osm_demo.fixtures import daejeon_fixture_source
from pursuit_evasion_rl.osm_demo.cache import CacheStore
from pursuit_evasion_rl.osm_demo.models import (
    DomainValidationError,
    EpisodeConfig,
    RunMode,
    VehiclePlacement,
)
from pursuit_evasion_rl.osm_demo.presets import DAEJEON_DRIVE_PRESET
from pursuit_evasion_rl.osm_demo.service import (
    DemoService,
    FileNetworkRepository,
    FileRunRepository,
    InMemoryNetworkRepository,
    InMemoryRunRepository,
    PreparedNetworkRecord,
    RunResult,
)

pytestmark = pytest.mark.offline


def _service(tmp_path, *, file_backed: bool = False) -> DemoService:
    cache_store = CacheStore(tmp_path / "cache")
    if file_backed:
        return DemoService(
            cache_store=cache_store,
            networks=FileNetworkRepository(tmp_path / "store"),
            runs=FileRunRepository(tmp_path / "store"),
        )
    return DemoService(cache_store=cache_store)


def _prepare(service: DemoService) -> PreparedNetworkRecord:
    return service.prepare_online(
        DAEJEON_DRIVE_PRESET.area,
        source=daejeon_fixture_source(),
        network_type="drive",
    )


def _placements(record: PreparedNetworkRecord):
    ids = sorted(record.network.intersection_ids)
    police = [VehiclePlacement(intersection_id=i) for i in ids[:6]]
    fugitive = VehiclePlacement(intersection_id=ids[6])
    return police, fugitive


def _config() -> EpisodeConfig:
    return EpisodeConfig(
        dt_s=1.0,
        police_speed_mps=15.0,
        fugitive_speed_mps=13.0,
        capture_radius_m=25.0,
        max_steps=20,
    )


def test_prepare_online_registers_validated_network(tmp_path):
    service = _service(tmp_path)
    record = _prepare(service)

    assert record.network_id
    assert record.network_hash
    assert record.statistics.max_out_degree <= 5
    # The prepared network is retrievable and its summary is API-safe (no paths).
    fetched = service.get_network(record.network_id)
    assert fetched.network_hash == record.network_hash
    summary = record.summary()
    assert summary["network_id"] == record.network_id
    assert "statistics" in summary


def test_prepare_offline_matches_online_identity(tmp_path):
    service = _service(tmp_path)
    online = _prepare(service)

    # The cache key (network_id) is timestamp-free, so an offline reload of the
    # same bundle yields the identical network identity.
    offline = service.prepare_offline(online.network_id)
    assert offline.network_id == online.network_id
    assert offline.network_hash == online.network_hash


def test_get_network_missing_raises_not_found(tmp_path):
    service = _service(tmp_path)
    with pytest.raises(DomainValidationError) as excinfo:
        service.get_network("does-not-exist")
    assert excinfo.value.code == "NETWORK_NOT_FOUND"


def test_recommend_actions_with_baseline_returns_six_atomic(tmp_path):
    service = _service(tmp_path)
    record = _prepare(service)
    policy = service.build_baseline_policy(record.network_id)
    police, fugitive = _placements(record)

    result = service.recommend_actions(
        record.network_id,
        policy,
        police=police,
        fugitive=fugitive,
        step=0,
        max_steps=20,
    )

    assert len(result.recommendations) == 6
    assert all(rec.valid for rec in result.recommendations)
    # The non-inference baseline reports a not_measured latency row.
    assert result.latency["status"] == "not_measured"
    assert result.experimental is False


def test_run_episodes_returns_addressable_result(tmp_path):
    service = _service(tmp_path)
    record = _prepare(service)
    policy = service.build_baseline_policy(record.network_id)

    result = service.run_episodes(
        record.network_id,
        _config(),
        policy,
        run_seed=7,
        episode_count=3,
        mode=RunMode.BASELINE,
    )

    assert isinstance(result, RunResult)
    assert result.run_id
    # Requirement 12.3: run id, input hashes, deterministic seed list, location.
    assert result.manifest.artifacts["network"] == record.network_hash
    assert result.manifest.seeds[0] == 7
    assert len(result.manifest.seeds) == 1 + 3
    assert result.result_location

    entry = service.get_run(result.run_id)
    assert len(entry.records) == 3
    metrics = service.get_run_metrics(result.run_id)
    assert sum(metrics.outcomes.values()) == 3


def test_run_id_is_deterministic_for_same_request(tmp_path):
    service = _service(tmp_path)
    record = _prepare(service)
    policy = service.build_baseline_policy(record.network_id)

    first = service.run_episodes(
        record.network_id, _config(), policy, run_seed=11, episode_count=2
    )
    second = service.run_episodes(
        record.network_id, _config(), policy, run_seed=11, episode_count=2
    )
    assert first.run_id == second.run_id


def test_get_run_missing_raises_not_found(tmp_path):
    service = _service(tmp_path)
    with pytest.raises(DomainValidationError) as excinfo:
        service.get_run("missing-run")
    assert excinfo.value.code == "RUN_NOT_FOUND"


def test_file_backed_repositories_survive_cold_reload(tmp_path):
    service = _service(tmp_path, file_backed=True)
    record = _prepare(service)
    policy = service.build_baseline_policy(record.network_id)
    result = service.run_episodes(
        record.network_id, _config(), policy, run_seed=5, episode_count=2
    )

    # A brand-new service instance backed by the same directory reloads the
    # persisted artifacts without any in-memory state.
    reloaded = DemoService(
        cache_store=CacheStore(tmp_path / "cache"),
        networks=FileNetworkRepository(tmp_path / "store"),
        runs=FileRunRepository(tmp_path / "store"),
    )
    fetched_network = reloaded.get_network(record.network_id)
    assert fetched_network.network_hash == record.network_hash
    fetched_run = reloaded.get_run(result.run_id)
    assert len(fetched_run.records) == 2
    assert reloaded.get_network(record.network_id).network_id == record.network_id


def test_in_memory_default_repositories_are_wired(tmp_path):
    service = _service(tmp_path)
    assert isinstance(service.networks, InMemoryNetworkRepository)
    assert isinstance(service.runs, InMemoryRunRepository)

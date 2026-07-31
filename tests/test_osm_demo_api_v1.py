"""Focused offline tests for the FastAPI v1 transport layer.

These tests drive the ``/api/v1`` routes through FastAPI's ``TestClient`` with
an offline fixture OSM source and the deterministic baseline policy, so no
external OSM service, checkpoint or GPU is required (Requirements 12.1-12.3).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pursuit_evasion_rl.osm_demo.api_v1 import API_PREFIX, create_app
from pursuit_evasion_rl.osm_demo.cache import CacheStore
from pursuit_evasion_rl.osm_demo.fixtures import daejeon_fixture_source
from pursuit_evasion_rl.osm_demo.presets import DAEJEON_DRIVE_PRESET
from pursuit_evasion_rl.osm_demo.service import DemoService

pytestmark = pytest.mark.offline


@pytest.fixture()
def client(tmp_path) -> TestClient:
    service = DemoService(cache_store=CacheStore(tmp_path / "cache"))
    app = create_app(
        service,
        osm_source=daejeon_fixture_source(),
        export_root=tmp_path / "exports",
    )
    return TestClient(app)


def _osm_request_body() -> dict:
    area = DAEJEON_DRIVE_PRESET.area
    return {
        "name": area.name,
        "north": area.north,
        "south": area.south,
        "east": area.east,
        "west": area.west,
        "max_area_km2": area.max_area_km2,
        "config_version": area.config_version,
        "network_type": "drive",
    }


def _prepare_network(client: TestClient) -> str:
    response = client.post(f"{API_PREFIX}/networks/osm", json=_osm_request_body())
    assert response.status_code == 201, response.text
    return response.json()["network_id"]


def _placements(client: TestClient, network_id: str):
    response = client.get(f"{API_PREFIX}/networks/{network_id}")
    assert response.status_code == 200
    # The 3x3 Daejeon fixture always yields at least seven drivable intersections.
    police = [{"intersection_id": i} for i in range(6)]
    fugitive = {"intersection_id": 6}
    return police, fugitive


def test_health_reports_registry_and_schema(client):
    response = client.get(f"{API_PREFIX}/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["osm_source_available"] is True
    assert body["registries"]["networks"] == 0


def test_prepare_and_fetch_network_returns_api_safe_summary(client):
    network_id = _prepare_network(client)

    fetched = client.get(f"{API_PREFIX}/networks/{network_id}")
    assert fetched.status_code == 200
    summary = fetched.json()
    assert summary["network_id"] == network_id
    assert "statistics" in summary
    # Responses never leak absolute local paths.
    assert "\\" not in str(summary) and "/tmp" not in str(summary).lower()


def test_offline_cache_load_matches_online_identity(client):
    network_id = _prepare_network(client)
    response = client.post(f"{API_PREFIX}/networks/cache", json={"cache_key": network_id})
    assert response.status_code == 201, response.text
    assert response.json()["network_id"] == network_id


def test_recommend_returns_six_complete_recommendations_and_latency(client):
    network_id = _prepare_network(client)
    police, fugitive = _placements(client, network_id)

    response = client.post(
        f"{API_PREFIX}/actions/recommend",
        json={
            "network_id": network_id,
            "police": police,
            "fugitive": fugitive,
            "step": 0,
            "max_steps": 20,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["recommendations"]) == 6
    assert all(rec["valid"] for rec in body["recommendations"])
    # The non-inference baseline reports a not_measured latency row.
    assert body["latency"]["status"] == "not_measured"


def test_run_returns_accepted_id_hashes_seeds_and_result_url(client):
    network_id = _prepare_network(client)
    config = {
        "dt_s": 1.0,
        "police_speed_mps": 15.0,
        "fugitive_speed_mps": 13.0,
        "capture_radius_m": 25.0,
        "max_steps": 20,
    }
    response = client.post(
        f"{API_PREFIX}/runs",
        json={
            "network_id": network_id,
            "config": config,
            "run_seed": 7,
            "episode_count": 3,
            "mode": "baseline",
        },
    )
    assert response.status_code == 202, response.text
    body = response.json()
    run_id = body["run_id"]
    assert run_id
    assert body["seeds"][0] == 7
    assert len(body["seeds"]) == 1 + 3
    assert body["input_hashes"]["network"]
    assert body["result_url"] == f"{API_PREFIX}/runs/{run_id}"

    # The result URL resolves and metrics conserve episode membership.
    run_detail = client.get(body["result_url"])
    assert run_detail.status_code == 200
    assert run_detail.json()["episode_count"] == 3

    metrics = client.get(f"{API_PREFIX}/runs/{run_id}/metrics")
    assert metrics.status_code == 200
    metrics_body = metrics.json()
    assert sum(metrics_body["outcomes"].values()) == 3
    assert metrics_body["claim_boundary"]["verified"] is False


def test_run_id_is_idempotent_for_same_request(client):
    network_id = _prepare_network(client)
    config = {
        "dt_s": 1.0,
        "police_speed_mps": 15.0,
        "fugitive_speed_mps": 13.0,
        "capture_radius_m": 25.0,
        "max_steps": 20,
    }
    body = {
        "network_id": network_id,
        "config": config,
        "run_seed": 11,
        "episode_count": 2,
    }
    first = client.post(f"{API_PREFIX}/runs", json=body).json()["run_id"]
    second = client.post(f"{API_PREFIX}/runs", json=body).json()["run_id"]
    assert first == second


def test_missing_network_maps_to_404(client):
    response = client.get(f"{API_PREFIX}/networks/does-not-exist")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "NETWORK_NOT_FOUND"


def test_unknown_schema_version_is_rejected_422(client):
    body = _osm_request_body()
    body["schema_version"] = "9.9"
    response = client.post(f"{API_PREFIX}/networks/osm", json=body)
    assert response.status_code == 422


def test_wrong_police_count_is_rejected_422(client):
    network_id = _prepare_network(client)
    response = client.post(
        f"{API_PREFIX}/actions/recommend",
        json={
            "network_id": network_id,
            "police": [{"intersection_id": 0}],  # only one, needs six
            "fugitive": {"intersection_id": 6},
            "step": 0,
            "max_steps": 20,
        },
    )
    assert response.status_code == 422


def test_invalid_bbox_ordering_is_rejected(client):
    body = _osm_request_body()
    body["north"], body["south"] = body["south"], body["north"]  # north <= south
    response = client.post(f"{API_PREFIX}/networks/osm", json=body)
    # The domain rejects the inverted bbox as an unprocessable entity.
    assert response.status_code == 422


def test_export_create_and_fetch_roundtrip(client):
    network_id = _prepare_network(client)
    config = {
        "dt_s": 1.0,
        "police_speed_mps": 15.0,
        "fugitive_speed_mps": 13.0,
        "capture_radius_m": 25.0,
        "max_steps": 6,
    }
    run_id = client.post(
        f"{API_PREFIX}/runs",
        json={"network_id": network_id, "config": config, "run_seed": 3, "episode_count": 1},
    ).json()["run_id"]

    created = client.post(
        f"{API_PREFIX}/runs/{run_id}/exports",
        json={
            "implemented_features": ["offline pipeline"],
            "limitations": ["no verified OSM performance"],
            "max_frames": 2,
        },
    )
    assert created.status_code == 201, created.text
    export_id = created.json()["export_id"]

    # Re-creating with the same report reuses the immutable export.
    again = client.post(
        f"{API_PREFIX}/runs/{run_id}/exports",
        json={
            "implemented_features": ["offline pipeline"],
            "limitations": ["no verified OSM performance"],
            "max_frames": 2,
        },
    )
    assert again.status_code == 201
    assert again.json()["export_id"] == export_id

    fetched = client.get(f"{API_PREFIX}/exports/{export_id}")
    assert fetched.status_code == 200
    body = fetched.json()
    assert body["run_id"] == run_id
    assert body["files"]


# ---------------------------------------------------------------------------
# Error taxonomy, sanitization and OpenAPI examples (Requirements 12.4-12.8)
# ---------------------------------------------------------------------------

from pursuit_evasion_rl.osm_demo.api_v1 import (  # noqa: E402
    OSM_TRAINING_PATH_GUIDANCE,
    _classify_exception,
    _sanitize,
    _sanitize_text,
)
from pursuit_evasion_rl.osm_demo.models import DomainValidationError  # noqa: E402


def test_sanitize_text_redacts_absolute_paths_but_keeps_api_urls():
    windows = _sanitize_text(r"failed loading C:\Users\me\secret\model.pt now")
    assert "C:\\Users" not in windows and "<redacted-path>" in windows

    posix = _sanitize_text("checkpoint at /home/me/private/model.pt failed")
    assert "/home/me/private" not in posix and "<redacted-path>" in posix

    unc = _sanitize_text(r"share \\server\share\model.pt unreachable")
    assert "\\\\server" not in unc and "<redacted-path>" in unc

    # This API's own relative URLs and content hashes must be preserved.
    preserved = _sanitize_text(
        f"{API_PREFIX}/runs/abc hash d24b5791524514f8321ccf2b23a1cc28"
    )
    assert f"{API_PREFIX}/runs/abc" in preserved
    assert "d24b5791524514f8321ccf2b23a1cc28" in preserved


def test_sanitize_walks_nested_structures():
    payload = {
        "message": r"open C:\Temp\x.pt",
        "items": ["/var/data/secret/a.json", 42, {"p": "/etc/passwd/extra"}],
    }
    cleaned = _sanitize(payload)
    assert "C:\\Temp" not in str(cleaned)
    assert "/var/data/secret" not in str(cleaned)
    assert "/etc/passwd/extra" not in str(cleaned)
    assert cleaned["items"][1] == 42  # non-strings pass through unchanged


def test_classify_not_found_conflict_dependency_and_validation():
    status, _ = _classify_exception(
        DomainValidationError("RUN_NOT_FOUND", "missing", actual="r1")
    )
    assert status == 404

    status, detail = _classify_exception(
        DomainValidationError("SEMANTIC_PROFILE_REQUIRED", "profile mismatch")
    )
    assert status == 409
    assert detail["osm_training_path"] == OSM_TRAINING_PATH_GUIDANCE

    status, _ = _classify_exception(
        DomainValidationError("OSMNX_UNAVAILABLE", "no osmnx")
    )
    assert status == 503

    status, _ = _classify_exception(
        DomainValidationError("INVALID_BOUNDED_AREA", "north <= south")
    )
    assert status == 422


def test_classify_importerror_maps_to_503_and_sanitizes():
    status, detail = _classify_exception(
        ImportError(r"cannot import torch from C:\py\torch\__init__.py")
    )
    assert status == 503
    assert detail["code"] == "DEPENDENCY_UNAVAILABLE"
    assert "C:\\py" not in detail["message"]


def test_classify_unexpected_error_is_correlation_id_only():
    status, detail = _classify_exception(
        RuntimeError(r"boom with secret /home/me/token and C:\keys\priv.pem")
    )
    assert status == 500
    # A 500 body carries only a stable code and a correlation id: no message,
    # no path, no secret (Requirement 12.7).
    assert set(detail) == {"code", "correlation_id"}
    assert detail["code"] == "INTERNAL_ERROR"
    assert detail["correlation_id"]
    assert "message" not in detail


def test_conflict_error_returns_409_with_training_guidance_via_route(client, monkeypatch):
    network_id = _prepare_network(client)

    def _raise_conflict(*args, **kwargs):
        raise DomainValidationError(
            "INFERENCE_BLOCKED",
            "structural compatibility failed",
            actual=[{"section": "structure", "code": "OBS_DIM"}],
        )

    # Force the learned-policy build (invoked when a checkpoint is supplied) to
    # report an incompatibility so the route surfaces the 409 mapping.
    from pursuit_evasion_rl.osm_demo.service import DemoService

    monkeypatch.setattr(DemoService, "build_learned_policy", _raise_conflict)

    police, fugitive = _placements(client, network_id)
    response = client.post(
        f"{API_PREFIX}/actions/recommend",
        json={
            "network_id": network_id,
            "police": police,
            "fugitive": fugitive,
            "step": 0,
            "max_steps": 20,
            "checkpoint_path": "checkpoints/legacy_ep10.pt",
        },
    )
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "INFERENCE_BLOCKED"
    assert detail["osm_training_path"] == OSM_TRAINING_PATH_GUIDANCE


def test_unavailable_osm_source_maps_to_503_without_paths(tmp_path):
    service = DemoService(cache_store=CacheStore(tmp_path / "cache"))
    app = create_app(service, osm_source=None, export_root=tmp_path / "exports")
    offline_client = TestClient(app)
    response = offline_client.post(f"{API_PREFIX}/networks/osm", json=_osm_request_body())
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["code"] == "OSM_SOURCE_UNAVAILABLE"
    assert "<redacted-path>" not in str(detail)
    assert ":\\" not in str(detail)


def test_unexpected_internal_error_maps_to_500_correlation_id(client, monkeypatch):
    network_id = _prepare_network(client)
    from pursuit_evasion_rl.osm_demo.service import DemoService

    def _boom(self, run_id):
        raise RuntimeError(r"unexpected failure touching C:\secrets\key.pem")

    monkeypatch.setattr(DemoService, "get_run_metrics", _boom)
    response = client.get(f"{API_PREFIX}/runs/{network_id}/metrics")
    assert response.status_code == 500
    detail = response.json()["detail"]
    assert set(detail) == {"code", "correlation_id"}
    assert detail["code"] == "INTERNAL_ERROR"
    assert "C:\\secrets" not in str(detail)


def test_openapi_documents_request_and_error_examples(client):
    schema = client.get("/openapi.json").json()

    # Request example for the OSM prepare operation.
    osm_body = schema["paths"][f"{API_PREFIX}/networks/osm"]["post"]["requestBody"]
    osm_schema = osm_body["content"]["application/json"]["schema"]
    resolved = _resolve_schema(schema, osm_schema)
    assert "example" in resolved

    # Documented error-response classes with examples.
    recommend = schema["paths"][f"{API_PREFIX}/actions/recommend"]["post"]["responses"]
    for status in ("404", "409", "422", "503", "500"):
        assert status in recommend
        example = recommend[status]["content"]["application/json"]["example"]
        assert "detail" in example
    # The 409 example advertises the OSM training guidance.
    assert (
        recommend["409"]["content"]["application/json"]["example"]["detail"][
            "osm_training_path"
        ]
        == OSM_TRAINING_PATH_GUIDANCE
    )
    # The 500 example carries only a correlation id, no path or secret.
    five_hundred = recommend["500"]["content"]["application/json"]["example"]["detail"]
    assert set(five_hundred) == {"code", "correlation_id"}


def _resolve_schema(document: dict, schema: dict) -> dict:
    """Resolve a possibly ``$ref`` OpenAPI schema node to its definition."""
    ref = schema.get("$ref")
    if not ref:
        return schema
    node: dict = document
    for part in ref.lstrip("#/").split("/"):
        node = node[part]
    return node

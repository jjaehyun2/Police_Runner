"""Focused tests for the SUMO web demo server stream contract.

These tests avoid public networking and lock the dashboard-facing NDJSON fields
that are easy to regress: open-exit coordinates (``ox``) and background traffic
(``bg``) propagation from the SUMO frame callback.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from demo.server import app as demo_app

pytestmark = pytest.mark.offline


def test_health_endpoint_reports_server_ready() -> None:
    client = TestClient(demo_app.app)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert response.headers["cache-control"] == "no-store"


def test_point_stream_includes_open_exit_coordinates() -> None:
    client = TestClient(demo_app.app)
    example = client.post("/api/example", json={"map": "daejeon_demo", "n": 2, "seed": 0}).json()

    response = client.post(
        "/api/run",
        json={
            "map": "daejeon_demo",
            "police": example["police"],
            "fugitive": example["fugitive"],
            "policy": "greedy",
            "engine": "point",
            "profile": "ideal",
            "traffic": None,
            "ignore_red": False,
            "max_seconds": 30,
        },
    )

    assert response.status_code == 200
    events = [demo_app.json.loads(line) for line in response.text.splitlines() if line]
    frames = [event for event in events if event["type"] == "frame"]
    assert frames
    assert any("ox" in frame and isinstance(frame["ox"], list) for frame in frames)
    assert all(frame["open"] == len(frame["ox"]) for frame in frames)
    assert events[-1]["type"] == "end"


def test_sumo_runner_emits_open_exit_coordinates_and_background_traffic(monkeypatch) -> None:
    captured = {}

    class FakeSumoPursuit:
        def __init__(self, map_name, case, spec, **kwargs):
            captured.update(map_name=map_name, case=case, spec=spec, kwargs=kwargs)
            kwargs["on_frame"](
                {
                    "t": 1.0,
                    "p": [(0.0, 0.0)],
                    "f": (10.0, 0.0),
                    "goals": [("hold", 2)],
                    "reach": [(1, 0.0), (2, 0.0)],
                    "bg": [(5.0, 0.0)],
                    "known": [0],
                    "bars": [(0.0, 0.0)],
                    "kit": [False],
                }
            )

        def run(self):
            return SimpleNamespace(outcome="success", seconds=1.25)

    fake_module = types.SimpleNamespace(SumoPursuit=FakeSumoPursuit)
    monkeypatch.setitem(sys.modules, "sumo_sim", fake_module)
    fake_map = SimpleNamespace(
        name="fake_map",
        pos={1: (0.0, 0.0), 2: (10.0, 0.0)},
        region_bits=lambda reach, g=None: ("AA==", [[36.0, 127.0], [36.1, 127.1]]),
    )
    req = demo_app.RunRequest(
        map="daejeon_demo",
        police=[(36.0, 127.0)],
        fugitive=(36.1, 127.1),
        engine="sumo",
        profile="real",
        traffic=10.0,
        ignore_red=False,
        max_seconds=30,
    )
    emitted = []

    result = demo_app.run_sumo(fake_map, SimpleNamespace(police=(1,)), "greedy", req, emitted.append)

    assert result == {"outcome": "success", "seconds": 1.25}
    assert captured["kwargs"]["profile"] == "real"
    assert captured["kwargs"]["traffic"] == 10.0
    assert captured["kwargs"]["ignore_red"] is False
    assert captured["kwargs"]["background"] is True
    assert len(emitted) == 1
    frame = emitted[0]
    assert frame["type"] == "frame"
    assert frame["t"] == 1.0
    assert frame["role"] == [1]
    assert frame["reg"] == "AA=="
    assert frame["open"] == 2
    assert frame["ox"] == [[36.0, 127.0], [36.1, 127.1]]
    assert len(frame["p"]) == 1
    assert len(frame["f"]) == 2
    assert len(frame["tgt"]) == 1
    assert len(frame["bg"]) == 1
    assert frame["bk"] == [0]
    assert frame["kit"] == [False]
    assert len(frame["bars"]) == 1 and len(frame["bars"][0]) == 2


def test_barricades_are_rejected_outside_sumo() -> None:
    client = TestClient(demo_app.app)
    response = client.post("/api/run", json={
        "map": "daejeon_demo", "police": [[36.35, 127.38]], "fugitive": [36.352, 127.385],
        "barricades": [[36.351, 127.383]], "engine": "point",
    })
    assert response.status_code == 422
    response = client.post("/api/run", json={
        "map": "daejeon_demo", "police": [[36.35, 127.38]], "fugitive": [36.352, 127.385],
        "carry_barricades": True, "engine": "point",
    })
    assert response.status_code == 422


def test_barricade_closes_both_directions_of_the_nearest_road() -> None:
    import common
    from sumo_sim import barricade_at

    net = common.load_network("daejeon_demo")
    seg = next(s for s in net.segments if s.length_m > 150 and not s.crosses_boundary
               and any((o.start_id, o.end_id) == (s.end_id, s.start_id) for o in net.segments))
    a, b = seg.geometry_xy[0], seg.geometry_xy[-1]
    bar = barricade_at(net, ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
    pairs = {(s.start_id, s.end_id) for s in net.segments if s.id in {sid for sid, _ in bar.at}}
    assert pairs == {(seg.start_id, seg.end_id), (seg.end_id, seg.start_id)}
    assert all(0.0 < frac < 1.0 for _, frac in bar.at)


def test_api_token_is_optional_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("PURSUIT_DEMO_TOKEN", raising=False)
    client = TestClient(demo_app.app)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert response.headers["cache-control"] == "no-store"


def test_api_token_rejects_missing_or_wrong_header_without_leaking_secret(monkeypatch) -> None:
    secret = "temporary-demo-secret"
    monkeypatch.setenv("PURSUIT_DEMO_TOKEN", secret)
    client = TestClient(demo_app.app)

    missing = client.get("/api/health")
    wrong = client.get("/api/health", headers={"X-Demo-Token": "wrong", "Origin": "https://example.test"})
    maps = client.get("/api/maps")

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert maps.status_code == 401
    assert missing.headers["cache-control"] == "no-store"
    assert wrong.headers["access-control-allow-origin"] == "*"
    assert secret not in missing.text
    assert secret not in wrong.text
    assert secret not in maps.text


def test_api_token_accepts_matching_header(monkeypatch) -> None:
    secret = "temporary-demo-secret"
    monkeypatch.setenv("PURSUIT_DEMO_TOKEN", secret)
    client = TestClient(demo_app.app)

    response = client.get("/api/health", headers={"X-Demo-Token": secret})

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert response.headers["cache-control"] == "no-store"


def test_api_token_allows_cors_preflight_without_token(monkeypatch) -> None:
    monkeypatch.setenv("PURSUIT_DEMO_TOKEN", "temporary-demo-secret")
    client = TestClient(demo_app.app)

    response = client.options(
        "/api/health",
        headers={"Origin": "https://example.test", "Access-Control-Request-Method": "GET"},
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"

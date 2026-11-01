"""Focused tests for the SUMO web demo server stream contract.

These tests avoid public networking and lock the dashboard-facing NDJSON fields
that are easy to regress: open-exit coordinates (``ox``) and background traffic
(``bg``) propagation from the SUMO frame callback.
"""

from __future__ import annotations

import math
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

        def signal_stops(self):
            return [{"tls": 0, "idx": [0, 1], "line": [(0.0, 0.0), (0.0, 3.0)]}]

        def run(self):
            # like the real simulator, frames come from run(), after the server has sent "net"
            captured["kwargs"]["on_frame"](
                {
                    "t": 1.0,
                    "p": [(0.0, 0.0)],
                    "f": (10.0, 0.0),
                    "goals": [("hold", 2)],
                    "reach": [(1, 0.0), (2, 0.0)],
                    "bg": [(5.0, 0.0)],
                    "known": [0],
                    "bars": [[(0.0, -4.0), (0.0, 4.0)]],
                    "kit": [False],
                    "sig": ["GGrr"],
                }
            )
            return SimpleNamespace(outcome="success", seconds=1.25)

    fake_module = types.SimpleNamespace(SumoPursuit=FakeSumoPursuit)
    monkeypatch.setitem(sys.modules, "sumo_sim", fake_module)
    fake_build = types.SimpleNamespace(crossings=lambda name, profile: [{"w": 4.0, "pts": [(1.0, 0.0), (1.0, 8.0)]}])
    monkeypatch.setitem(sys.modules, "build_net", fake_build)
    fake_map = SimpleNamespace(
        name="fake_map",
        pos={1: (0.0, 0.0), 2: (10.0, 0.0)},
        region_bits=lambda reach, g=None, police_g=None, closed=(): ("AA==", [[36.0, 127.0], [36.1, 127.1]]),
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
    assert [item["type"] for item in emitted] == ["net", "frame"]   # net first, then the run's frames
    frame = emitted[1]
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
    assert frame["kit"] == []                      # nobody set out carrying a barricade
    # barricades travel as a line across the road (two lat/lon ends), signals as per-frame states
    assert len(frame["bars"]) == 1 and len(frame["bars"][0]) == 2 and len(frame["bars"][0][0]) == 2
    assert frame["sig"] == ["GGrr"]
    net = emitted[0]
    assert net["stops"] == [{"tls": 0, "idx": [0, 1], "line": [demo_app.ll((0.0, 0.0)), demo_app.ll((0.0, 3.0))]}]
    assert net["cross"] == [{"w": 4.0, "pts": [demo_app.ll((1.0, 0.0)), demo_app.ll((1.0, 8.0))]}]


@pytest.mark.parametrize("policy", sorted(demo_app.POLICIES))
def test_every_site_policy_streams_a_role_per_officer_on_the_point_engine(policy) -> None:
    """Each policy the site offers must name every officer's role (the officer table reads role[k])."""
    client = TestClient(demo_app.app)
    example = client.post("/api/example", json={"map": "daejeon_demo", "n": 3, "seed": 0}).json()
    response = client.post("/api/run", json={
        "map": "daejeon_demo", "police": example["police"], "fugitive": example["fugitive"],
        "policy": policy, "engine": "point", "profile": "ideal", "traffic": None, "ignore_red": False,
        "max_seconds": 30,
    })
    assert response.status_code == 200
    events = [demo_app.json.loads(line) for line in response.text.splitlines() if line]
    frames = [event for event in events if event.get("type") == "frame"]
    assert frames, events[:2]
    assert all(len(f["role"]) == len(f["p"]) == 3 for f in frames)


def test_escape_region_uses_the_officers_map_and_never_colours_a_barricaded_road(monkeypatch) -> None:
    """Officers drive through barricades, the fugitive does not: the region is computed with both
    maps, and a barricaded road is not drawn as escapable even when both its ends are."""
    m = demo_app.get_map("daejeon_demo")
    k = next(i for i, sids in enumerate(m.road_sids) if len(sids) == 2)
    a, b = m.road_ends[k]
    seen = {}

    def fake_region(g, reach, vp, vf, exits, margin_s=25.0 / 16.0, police_g=None):
        seen.update(g=g, police_g=police_g)
        return {a, b}, []

    monkeypatch.setattr(demo_app.multi_sim, "escape_region", fake_region)
    fugitive_map, officers_map = object(), object()
    bits_open, _ = m.region_bits([], fugitive_map, police_g=officers_map)
    bits_closed, _ = m.region_bits([], fugitive_map, police_g=officers_map, closed=m.road_sids[k])
    assert seen == {"g": fugitive_map, "police_g": officers_map}
    is_set = lambda bits: bool(demo_app.base64.b64decode(bits)[k >> 3] & (1 << (k & 7)))
    assert is_set(bits_open) and not is_set(bits_closed)


def test_goal_payload_treats_a_chasers_next_road_as_a_chase() -> None:
    """SUMO policies send ("edge", segment) for a chaser's next road; it is not an intersection to hold."""
    fake_map = SimpleNamespace(pos={7: (0.0, 0.0)})
    role, tgt = demo_app._goal_payload(fake_map, [("pursue", None), ("edge", 2801), ("hold", 7)])
    assert role == [0, 0, 1]
    assert tgt == [None, None, demo_app.ll((0.0, 0.0))]


def _toy_network():
    """Two intersections joined by a straight two-way road (segments 1, 2) and, separately, by a long
    detour that comes back the other way (segment 3)."""
    seg = lambda sid, a, b, pts: SimpleNamespace(id=sid, start_id=a, end_id=b, geometry_xy=pts)
    return SimpleNamespace(segments=[
        seg(1, 10, 20, [(0.0, 0.0), (100.0, 0.0)]),
        seg(2, 20, 10, [(100.0, 0.0), (0.0, 0.0)]),
        seg(3, 20, 10, [(100.0, 0.0), (100.0, 300.0), (-50.0, 300.0), (-50.0, 0.0), (0.0, 0.0)]),
    ])


def _sumo_sim():
    sim_dir = demo_app.BG / "sumo"
    if str(sim_dir) not in sys.path:
        sys.path.insert(0, str(sim_dir))
    import sumo_sim
    return sumo_sim


def test_barricade_stands_where_it_blocks() -> None:
    """A click next to an intersection is clamped 5 % into the road; the barricade's point is that
    clamped spot (where the blockers stand), not the raw projection of the click."""
    sumo_sim = _sumo_sim()
    bar = sumo_sim.barricade_at(_toy_network(), (1.0, 6.0))
    assert dict(bar.at)[1] == 0.05
    assert bar.xy == (5.0, 0.0)
    assert bar.heading == 0.0


def test_barricade_closes_the_opposite_carriageway_but_not_a_parallel_road() -> None:
    sumo_sim = _sumo_sim()
    bar = sumo_sim.barricade_at(_toy_network(), (40.0, 1.0))
    assert {sid for sid, _ in bar.at} == {1, 2}
    fracs = dict(bar.at)
    assert abs(fracs[1] - 0.4) < 1e-9 and abs(fracs[2] - 0.6) < 1e-9


def test_opposite_carriageway_of_another_length_is_closed_where_the_barricade_stands() -> None:
    """The other carriageway may be longer or shaped differently: its blocker goes where the
    barricade point projects onto it, not at the mirrored fraction."""
    sumo_sim = _sumo_sim()
    seg = lambda sid, a, b, pts: SimpleNamespace(id=sid, start_id=a, end_id=b, geometry_xy=pts)
    other = [(100.0, 3.0), (98.0, 4.8), (0.0, 3.0)]                     # kinks near one end (within 5 m)
    net = SimpleNamespace(segments=[seg(1, 10, 20, [(0.0, 0.0), (100.0, 0.0)]), seg(2, 20, 10, other)])
    bar = sumo_sim.barricade_at(net, (40.0, 1.0))
    length = sumo_sim._polyline_length(other)
    placed, _h = sumo_sim._point_along(other, dict(bar.at)[2] * length)
    mirrored, _h = sumo_sim._point_along(other, (1.0 - dict(bar.at)[1]) * length)
    nearest = sumo_sim._distance_to_polyline(bar.xy, other)
    assert abs(math.dist(placed, bar.xy) - nearest) < 1e-6           # straight across from the barricade
    assert math.dist(mirrored, placed) > 0.2                         # the mirrored fraction would miss it


def test_a_bent_road_and_a_straight_road_between_the_same_intersections_are_different_roads() -> None:
    sumo_sim = _sumo_sim()
    seg = lambda sid, a, b, pts: SimpleNamespace(id=sid, start_id=a, end_id=b, geometry_xy=pts)
    bent = seg(1, 10, 20, [(0.0, 0.0), (50.0, 30.0), (100.0, 0.0)])
    straight = seg(2, 20, 10, [(100.0, 0.0), (0.0, 0.0)])
    assert not sumo_sim.opposite_carriageway(bent, straight)
    assert sumo_sim.opposite_carriageway(straight, seg(3, 10, 20, [(0.0, 2.0), (100.0, 2.0)]))


def test_barricade_line_crosses_the_road_at_right_angles() -> None:
    sumo_sim = _sumo_sim()
    bar = sumo_sim.barricade_at(_toy_network(), (40.0, 1.0))
    (x0, y0), (x1, y1) = sumo_sim.barricade_line(bar)
    assert (x0, x1) == (40.0, 40.0)                      # road runs along x: the line runs along y
    assert abs(abs(y1 - y0) - 2 * sumo_sim.BAR_HALF_M) < 1e-9


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

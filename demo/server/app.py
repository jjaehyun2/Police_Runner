"""Pursuit demo server: place officers and a fugitive on a real map, run the chase in SUMO
(TraCI) or in the point-mass simulator, and stream it second by second.

  GET  /api/health
  GET  /api/maps                      -> available maps
  GET  /api/map/{name}                -> roads (lat/lon), exits, bounds, road count
  POST /api/example                   -> an example start {police:[[lat,lon]], fugitive:[lat,lon]}
  POST /api/run                       -> NDJSON stream: setup, frame (one per simulated second), end
                                         (optional barricades [[lat,lon]]: closed roads, and
                                          carry_barricades: one per patrol car, placed by the team; SUMO only)

Run:  uvicorn app:app --host 0.0.0.0 --port 8600   (from demo/server, with the project venv)
Env:  PURSUIT_MAX_RUNS (concurrent simulations, default 6), PURSUIT_CORS (comma list, default *),
      PURSUIT_DEMO_TOKEN (optional X-Demo-Token required for /api when set)
"""
from __future__ import annotations

import asyncio
import base64
import hmac
import json
import math
import os
from pathlib import Path
import sys
import threading
from typing import Literal

ROOT = Path(__file__).resolve().parents[2]
BG = ROOT / "scripts/research/real_scale/beat_greedy"
sys.path[:0] = [str(BG / "sumo"), str(BG), str(ROOT)]
os.environ.setdefault("ME_MAX_STEPS", "600")
os.environ.setdefault("BG_FUGITIVE_SPEED", "16")

import numpy as np  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from pyproj import Transformer  # noqa: E402

import common  # noqa: E402
import multi_eval  # noqa: E402
import multi_sim  # noqa: E402
from obs_ext import boundary_distance, exit_nodes  # noqa: E402

MAPS = {"daejeon_demo": "대전 둔산", "seoul_zero_shot": "서울 (중구·용산)", "busan_zero_shot": "부산"}
POLICIES = {
    "ga": ("rmacro:" + str(BG / "runs/ms_ga/best.pt"), "GA 역할 정책"),
    "encircle": ("encircle", "포위"),
    "greedy": ("greedy", "탐욕 추격"),
    "assign": ("rmacro:assign", "배정 규칙"),
}
SPEED = 16.0
TO_LL = Transformer.from_crs("EPSG:32652", "EPSG:4326", always_xy=True)
TO_XY = Transformer.from_crs("EPSG:4326", "EPSG:32652", always_xy=True)
_SEM = threading.BoundedSemaphore(int(os.environ.get("PURSUIT_MAX_RUNS", "6")))
_START_LOCK = threading.Lock()                       # traci.start is not thread-safe


def ll(xy) -> list[float]:
    lon, lat = TO_LL.transform(xy[0], xy[1])
    return [round(lat, 6), round(lon, 6)]


class MapData:
    """Per-map geometry for the web client plus snapping and containment helpers."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.net = common.load_network(name)
        self.g = multi_sim.graph_of(self.net)
        self.exits = exit_nodes(self.net)
        bd = boundary_distance(self.net)
        self.pos = {i.id: i.position_xy for i in self.net.intersections}
        # one entry per road: the two carriageways of a road are drawn once (the same test barricades use
        # to close both, sumo_sim.opposite_carriageway), but a different road between the same two
        # intersections keeps its own entry, so a barricade placed on it snaps to it, not its neighbour
        from sumo_sim import opposite_carriageway
        self.road_ends: list[tuple[int, int]] = []
        self.road_sids: list[set[int]] = []                  # the directed segments each road entry draws
        by_ends: dict[tuple[int, int], list[tuple[int, object]]] = {}
        roads = []
        for s in self.net.segments:
            ends = (min(s.start_id, s.end_id), max(s.start_id, s.end_id))
            match = next((i for i, rep in by_ends.get(ends, []) if opposite_carriageway(rep, s)), None)
            if match is not None:
                self.road_sids[match].add(int(s.id))
                continue
            by_ends.setdefault(ends, []).append((len(roads), s))
            self.road_ends.append((s.start_id, s.end_id))
            self.road_sids.append({int(s.id)})
            roads.append([ll(p) for p in s.geometry_xy])
        self.roads = roads
        drivable = [i.id for i in self.net.intersections if not i.virtual and i.outgoing_segment_ids]
        self.police_nodes = np.array(drivable)
        inter = {i.id: i for i in self.net.intersections}
        self.fugitive_nodes = np.array([n for n in drivable if inter[n].boundary_kind is None
                                        and bd.get(n, 0.0) >= 60.0] or drivable)
        xs = np.array(list(self.pos.values()))
        lo, hi = xs.min(0), xs.max(0)
        self.bounds = [ll(lo), ll(hi)]

    def snap(self, latlon, pool: np.ndarray) -> int:
        x, y = TO_XY.transform(latlon[1], latlon[0])
        pts = np.array([self.pos[int(n)] for n in pool])
        return int(pool[int(np.argmin(((pts - (x, y)) ** 2).sum(1)))])

    def region_bits(self, reach, g=None, police_g=None, closed=()) -> tuple[str, list]:
        """Escape region as a road bitset (base64) and the open exits' coordinates. With barricades,
        ``g`` is the fugitive's road graph (barricaded roads removed), ``police_g`` the officers'
        (they drive through barricades) and ``closed`` the barricaded segment ids: a closed road is
        not part of the region even when both its ends are."""
        region, open_exits = multi_sim.escape_region(g or self.g, reach, SPEED, SPEED, self.exits,
                                                     police_g=police_g)
        closed = set(closed)
        bits = bytearray((len(self.roads) + 7) // 8)
        for i, (a, b) in enumerate(self.road_ends):
            if a in region and b in region and not self.road_sids[i] <= closed:
                bits[i >> 3] |= 1 << (i & 7)
        return base64.b64encode(bytes(bits)).decode(), [ll(self.pos[e]) for e in open_exits]

    def client_json(self) -> dict:
        return dict(name=self.name, label=MAPS[self.name], roads=self.roads,
                    exits=[ll(self.pos[e]) for e in self.exits], bounds=self.bounds)


_MAPS: dict[str, MapData] = {}


def get_map(name: str) -> MapData:
    if name not in MAPS:
        raise HTTPException(404, f"unknown map {name}")
    if name not in _MAPS:
        _MAPS[name] = MapData(name)
    return _MAPS[name]


# -- simulation runners (worker threads) --------------------------------------------------------
def _goal_payload(m: MapData, goals) -> tuple[list[int], list]:
    """Role (0 chase, 1 hold) and held intersection per officer. ``("edge", segment)`` is a chaser's
    next road toward the fugitive (sumo_sim.TargetPolicy.goals), so it is a chase, not a held point."""
    role = [1 if g[0] == "hold" else 0 for g in goals]
    tgt = [ll(m.pos[g[1]]) if g[0] == "hold" else None for g in goals]
    return role, tgt


def run_sumo(m: MapData, case, spec: str, req: "RunRequest", emit, barricades=()) -> dict:
    from sumo_sim import SumoPursuit
    holder: dict = {}

    def on_frame(fr):
        role, tgt = _goal_payload(m, fr["goals"])
        sim = holder.get("sim")
        reg, ox = m.region_bits([tuple(r) for r in fr["reach"]], getattr(sim, "pg", None),
                                police_g=getattr(sim, "og", None), closed=getattr(sim, "closed", ()))
        # "kit" (who still carries a barricade) only means something when officers set out carrying one;
        # otherwise the panel would read every all-False kit as "already placed"
        carrying = holder.setdefault("carrying", any(fr.get("kit", [])))
        out = dict(type="frame", t=fr["t"], p=[ll(q) for q in fr["p"]], f=ll(fr["f"]), role=role, tgt=tgt,
                   reg=reg, open=len(ox), ox=ox, bk=fr.get("known", []),
                   kit=fr.get("kit", []) if carrying else [],
                   bars=[[ll(q) for q in line] for line in fr.get("bars", [])])
        if "bg" in fr:
            out["bg"] = [ll(q) for q in fr["bg"][:600]]
        if "sig" in fr:
            out["sig"] = fr["sig"]
        emit(out)

    with _START_LOCK:
        sim = SumoPursuit(m.name, case, spec, profile=req.profile, traffic=req.traffic,
                          ignore_red=req.ignore_red, on_frame=on_frame, background=req.traffic is not None,
                          barricades=barricades, carry_barricades=req.carry_barricades)
    holder["sim"] = sim
    emit(_net_payload(m, sim, req.profile))
    r = sim.run()
    return dict(outcome=r.outcome, seconds=round(r.seconds, 2))


def _net_payload(m: MapData, sim, profile: str) -> dict:
    """Static street furniture of the simulated network, sent once before the frames: signal stop
    lines (coloured per frame from ``sig``) and SUMO-generated crosswalks ("real" profile only)."""
    stops = [dict(tls=s["tls"], idx=s["idx"], line=[ll(q) for q in s["line"]]) for s in sim.signal_stops()]
    cross = []
    if profile == "real":
        from build_net import crossings
        cross = [dict(w=c["w"], pts=[ll(q) for q in c["pts"]]) for c in crossings(m.name, profile)]
    return dict(type="net", stops=stops, cross=cross)


def run_point(m: MapData, case, spec: str, req: "RunRequest", emit, barricades=()) -> dict:
    from sumo_sim import TargetPolicy
    tp = TargetPolicy(spec, m.net, len(case.police))
    bridge = multi_sim.RecommendBridge(tp.team)
    sim = multi_sim.MultiPursuitSim(m.net, case)

    def frame():
        # policies stepped through the bridge without a recommendation of their own (greedy) report no
        # goals; every officer is then chasing, which is what the panel must show
        role, tgt = _goal_payload(m, tp.goals() or [("pursue", None)] * len(case.police))
        reach = [sim.reach(v) for v in sim.police] + [sim.reach(sim.fugs[0])]
        reg, ox = m.region_bits(reach)
        emit(dict(type="frame", t=sim.step_count, p=[ll(q) for q in sim.police_xy()], f=ll(sim.position(sim.fugs[0])),
                  role=role, tgt=tgt, reg=reg, open=len(ox), ox=ox))

    while sim.outcome is None:
        providers = bridge.act(sim)
        frame()
        sim.step(providers)
    frame()
    return dict(outcome=sim.outcome, seconds=sim.step_count)


# -- API ---------------------------------------------------------------------------------------
app = FastAPI(title="Pursuit demo")


def _configured_demo_token() -> str | None:
    token = os.environ.get("PURSUIT_DEMO_TOKEN")
    return token if token else None


def _authorized_demo_token(request: Request, expected: str) -> bool:
    provided = request.headers.get("X-Demo-Token", "")
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


@app.middleware("http")
async def require_demo_token(request: Request, call_next):
    is_api = request.url.path.startswith("/api")
    token = _configured_demo_token()
    if token and is_api and request.method != "OPTIONS":
        if not _authorized_demo_token(request, token):
            return JSONResponse(
                {"detail": "Unauthorized"},
                status_code=401,
                headers={"Cache-Control": "no-store"},
            )
    response = await call_next(request)
    if is_api:
        response.headers.setdefault("Cache-Control", "no-store")
    return response


# Keep CORS outermost so even a rejected token yields a readable 401 in browsers.
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("PURSUIT_CORS", "*").split(","),
                   allow_methods=["GET", "POST"], allow_headers=["*"])


class RunRequest(BaseModel):
    map: str = "daejeon_demo"
    police: list[tuple[float, float]] = Field(..., min_length=1, max_length=12)
    fugitive: tuple[float, float]
    barricades: list[tuple[float, float]] = Field(default_factory=list, max_length=20)
    carry_barricades: bool = False                       # each patrol car carries one and places it itself
    policy: Literal["ga", "encircle", "greedy", "assign"] = "ga"
    engine: Literal["sumo", "point"] = "sumo"
    profile: Literal["ideal", "real"] = "ideal"
    traffic: float | None = Field(None, ge=0.5, le=10)
    ignore_red: bool = False
    max_seconds: int = Field(600, ge=30, le=900)


class ExampleRequest(BaseModel):
    map: str = "daejeon_demo"
    n: int = Field(6, ge=1, le=12)
    seed: int = 0


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/maps")
def maps():
    return [{"name": k, "label": v} for k, v in MAPS.items()]


@app.get("/api/map/{name}")
def map_data(name: str):
    return get_map(name).client_json()


@app.post("/api/example")
def example(req: ExampleRequest):
    m = get_map(req.map)
    case = multi_sim.make_ring_case(m.net, f"web-{req.map}", req.seed, req.n)
    return dict(police=[ll(m.pos[n]) for n in case.police], fugitive=ll(m.pos[case.fugitives[0]]))


@app.post("/api/run")
async def run(req: RunRequest):
    m = get_map(req.map)
    police = []
    for p in req.police:                                   # distinct start intersections
        pool = np.array([n for n in m.police_nodes if n not in police])
        police.append(m.snap(p, pool))
    fug = m.snap(req.fugitive, np.array([n for n in m.fugitive_nodes if n not in police]))
    seed = abs(hash((tuple(police), fug))) % 2**31
    case = multi_sim.MultiCase(tuple(police), (fug,), (seed,), SPEED, SPEED, max_steps=req.max_seconds)
    if (req.barricades or req.carry_barricades) and req.engine != "sumo":
        raise HTTPException(422, "바리케이드는 SUMO 시뮬레이터에서만 쓸 수 있습니다.")
    from sumo_sim import barricade_at, barricade_line
    barricades = []
    for b in req.barricades:
        x, y = TO_XY.transform(b[1], b[0])
        bar = barricade_at(m.net, (x, y))
        if all(bar.at != o.at for o in barricades):           # one barricade per road
            barricades.append(bar)
    spec, label = POLICIES[req.policy]
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def emit(obj):
        loop.call_soon_threadsafe(queue.put_nowait, obj)

    def work():
        if not _SEM.acquire(timeout=1.0):
            emit(dict(type="error", message="서버가 다른 시뮬레이션을 처리 중입니다. 잠시 후 다시 실행해 주세요."))
            emit(None)
            return
        try:
            emit(dict(type="setup", police=[ll(m.pos[n]) for n in police], fugitive=ll(m.pos[fug]),
                      policy=label, engine=req.engine, profile=req.profile, traffic=req.traffic,
                      barricades=[[ll(q) for q in barricade_line(b)] for b in barricades],
                      carry_barricades=req.carry_barricades))
            runner = run_sumo if req.engine == "sumo" else run_point
            end = runner(m, case, spec, req, emit, tuple(barricades))
            emit(dict(type="end", **end))
        except Exception as e:                              # noqa: BLE001
            emit(dict(type="error", message=f"{type(e).__name__}: {e}"[:300]))
        finally:
            _SEM.release()
            emit(None)

    threading.Thread(target=work, daemon=True).start()

    async def stream():
        while True:
            item = await queue.get()
            if item is None:
                return
            yield json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

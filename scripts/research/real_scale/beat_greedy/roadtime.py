"""Road travel-time metric (the "travel" distance of the role policy, spec prefix tmacro:).

Distance from x to node n is the fastest legal driving time along the directed road graph,

    T(x -> n) = min over routes [ sum_e  l_e / v_e  +  sum_turns  c(theta) ],
    v_e       = min(v_max, kappa * v_lim(e)),
    c(theta)  = (v_max - v_turn(theta))^2 / (2 v_max) * (1/a + 1/d),

  l_e        segment length; v_lim(e) its OSM maxspeed, or the road-class default (as sumo/build_net.py)
  kappa      drivers' multiple of the posted limit (2.0: the speed factor of the SUMO "real" profile)
  theta      heading change between consecutive segments; v_turn = v_max up to 30 deg, 8 m/s up to
             120 deg, 4 m/s beyond (sharp turns and u-turns)
  c(theta)   time lost braking at d to v_turn and accelerating back at a (SUMO "real": a 3.5, d 7.5)

Times are returned as equivalent metres (T * v_max), so every caller that divides road distances by
the vehicle speed keeps working unchanged. The graph exposes the _Graph interface (pos, seg, fwd,
rev, dist_from, dist_to); turn costs are exact in dist_from / dist_to (Dijkstra over segments),
fwd / rev carry the per-segment costs only.

Interception margin at node n, as the role policy uses it:
    Delta(n) = T_fugitive(n) - T_officer(n)     (an officer can hold n if Delta(n) > capture margin)
"""
from __future__ import annotations

import heapq
import math
import os

from pursuit_evasion_rl.osm_demo.coarsening import _heading

_INF = float("inf")
# km/h by road class; mirrors sumo/build_net.py DEFAULT_KPH (not importable: the pip package "sumo"
# shadows that directory)
DEFAULT_KPH = {"motorway": 100, "trunk": 80, "primary": 60, "secondary": 50, "tertiary": 50,
               "unclassified": 40, "residential": 30, "living_street": 20, "service": 20}
KAPPA, ACCEL, DECEL = 2.0, 3.5, 7.5
# RT_CLASS_FIX=1: look the full road class up first (living_street -> 20 km/h); off keeps the original lookup,
# under which every run so far was made. Must match the SUMO net (sumo/build_net.py reads the same switch).
CLASS_FIX = os.environ.get("RT_CLASS_FIX", "0") == "1"


def limit_mps(seg) -> float:
    attrs = dict(seg.attributes)
    for t in attrs.get("transition_attributes", ()):
        raw = dict(t).get("maxspeed")
        if raw:
            raw = raw[0] if isinstance(raw, (list, tuple)) else raw
            try:
                return float(str(raw).split()[0]) / 3.6
            except ValueError:
                pass
    full = str(attrs.get("road_class", "residential")).split("|")[0]
    cls = full.split("_")[0]
    if CLASS_FIX and full in DEFAULT_KPH:            # e.g. living_street (the prefix "living" fell back to 30)
        cls = full
    return DEFAULT_KPH.get(cls, 30) / 3.6


def turn_seconds(theta_deg: float, v: float) -> float:
    t = abs(theta_deg)
    vt = v if t <= 30 else (8.0 if t <= 120 else 4.0)
    vt = min(vt, v)
    return (v - vt) ** 2 / (2 * v) * (1 / ACCEL + 1 / DECEL)


class TravelGraph:
    def __init__(self, network, v: float) -> None:
        self.network, self.v = network, v
        self.pos = {i.id: i.position_xy for i in network.intersections}
        self.seg = {s.id: s for s in network.segments}
        self.cost: dict[int, float] = {}                  # equivalent metres per segment
        h_in, h_out = {}, {}
        for s in network.segments:
            self.cost[s.id] = float(s.length_m) * v / min(v, KAPPA * limit_mps(s))
            try:
                h_out[s.id], h_in[s.id] = _heading(s.geometry_xy, True), _heading(s.geometry_xy, False)
            except Exception:                               # noqa: BLE001  zero-extent geometry
                h_out[s.id] = h_in[s.id] = None
        self.out_segs = {i.id: [] for i in network.intersections}
        self.in_segs = {i.id: [] for i in network.intersections}
        for s in network.segments:
            self.out_segs[s.start_id].append(s.id)
            self.in_segs[s.end_id].append(s.id)
        self.fwd = {n: [(self.seg[s].end_id, self.cost[s]) for s in ss] for n, ss in self.out_segs.items()}
        self.rev = {n: [(self.seg[s].start_id, self.cost[s]) for s in ss] for n, ss in self.in_segs.items()}
        self._h_in, self._h_out = h_in, h_out
        self._turn: dict[tuple[int, int], float] = {}
        self._to: dict[int, dict[int, float]] = {}
        self._from: dict[int, dict[int, float]] = {}

    def turn(self, a: int, b: int) -> float:
        """Equivalent metres lost turning from segment a into segment b."""
        key = (a, b)
        c = self._turn.get(key)
        if c is None:
            ha, hb = self._h_in[a], self._h_out[b]
            th = 0.0 if ha is None or hb is None else math.degrees(math.atan2(math.sin(hb - ha), math.cos(hb - ha)))
            c = self._turn[key] = turn_seconds(th, self.v) * self.v
        return c

    def dist_from(self, src: int) -> dict[int, float]:
        c = self._from.get(src)
        if c is not None:
            return c
        node = {src: 0.0}
        best: dict[int, float] = {}
        heap = []
        for s in self.out_segs[src]:
            best[s] = self.cost[s]
            heapq.heappush(heap, (self.cost[s], s))
        while heap:
            d, s = heapq.heappop(heap)
            if d > best.get(s, _INF):
                continue
            e = self.seg[s].end_id
            if d < node.get(e, _INF):
                node[e] = d
            for s2 in self.out_segs[e]:
                nd = d + self.turn(s, s2) + self.cost[s2]
                if nd < best.get(s2, _INF):
                    best[s2] = nd
                    heapq.heappush(heap, (nd, s2))
        self._from[src] = node
        return node

    def dist_to(self, dst: int) -> dict[int, float]:
        c = self._to.get(dst)
        if c is not None:
            return c
        node = {dst: 0.0}
        best: dict[int, float] = {}
        heap = []
        for s in self.in_segs[dst]:
            best[s] = self.cost[s]
            heapq.heappush(heap, (self.cost[s], s))
        while heap:
            d, s = heapq.heappop(heap)
            if d > best.get(s, _INF):
                continue
            b = self.seg[s].start_id
            if d < node.get(b, _INF):
                node[b] = d
            for s0 in self.in_segs[b]:
                nd = d + self.turn(s0, s) + self.cost[s0]
                if nd < best.get(s0, _INF):
                    best[s0] = nd
                    heapq.heappush(heap, (nd, s0))
        self._to[dst] = node
        return node


_CACHE: dict[tuple[int, float], TravelGraph] = {}


def travel_graph(network, v: float) -> TravelGraph:
    key = (id(network), v)
    if key not in _CACHE:
        _CACHE[key] = TravelGraph(network, v)
    return _CACHE[key]


def boundary_time(network, v: float, exits) -> dict[int, float]:
    g = travel_graph(network, v)
    if not hasattr(g, "_boundary"):
        best: dict[int, float] = {}
        for e in set(exits):
            for n, d in g.dist_to(e).items():
                if d < best.get(n, _INF):
                    best[n] = d
        g._boundary = best
    return g._boundary


# -- v2: traffic-aware travel time -----------------------------------------------------------------------
# Pursuit vehicles are not bound by limits or signals (emergency driving), but civilian traffic is, and it
# slows them down. For segment e with n_e lanes, length l_e and background traffic (vehicles N_e, halting
# H_e, mean speed u_e) from a traffic-information feed:
#
#   k_e      = N_e / (l_e n_e)                         density per lane (veh/m)
#   p_e      = (1 - min(k_e / k_jam, 1)) ** (g / n_e)  chance of passing / being let through
#   v_e      = u_e + (v_free(e) - u_e) p_e             speed through the traffic (v_free(e) as v1; u_e = v_free if empty)
#   t_e      = l_e / v_e + (H_e / n_e) h (1 - p_e)     + w_sig at a signal when the vehicle obeys red
#
# g, h and w_sig are fit to logged SUMO traversals (traffic_calib.py; runs/traffic_metric/calib.json).
K_JAM = 1.0 / 7.5                                    # SUMO default vehicle length 5 m + min gap 2.5 m
CALIB = {"g": 1.0, "h": 2.0, "w_sig": 10.0}


def load_calib(path=None) -> dict:
    import json
    import os
    from pathlib import Path
    p = Path(path or os.environ.get("RT_CALIB", Path(__file__).resolve().parent / "runs/traffic_metric/calib.json"))
    return {**CALIB, **json.loads(p.read_text())["params"]} if p.exists() else dict(CALIB)


def traffic_seconds(seg, length: float, v_free: float, lanes: int, snap, calib: dict, signal_end: bool,
                    obeys_red: bool) -> float:
    n_veh, halting, u = snap if snap is not None else (0, 0, None)
    lanes = max(int(lanes), 1)
    k = n_veh / (max(length, 1.0) * lanes)
    p = (1.0 - min(k / K_JAM, 1.0)) ** (calib["g"] / lanes)
    u = v_free if (u is None or n_veh == 0) else min(u, v_free)
    v = max(u + (v_free - u) * p, 0.5)
    t = length / v + halting / lanes * calib["h"] * (1.0 - p)
    if obeys_red and signal_end:
        t += calib["w_sig"]
    return t


class TrafficGraph(TravelGraph):
    """TravelGraph whose segment costs follow the live background traffic (set_traffic)."""

    def __init__(self, network, v: float, *, lanes: dict | None = None, signals: set | None = None,
                 obeys_red: bool = False, calib: dict | None = None) -> None:
        super().__init__(network, v)
        self.lanes = lanes or {}
        self.signals = signals or set()
        self.obeys_red = obeys_red
        self.calib = calib or load_calib()
        self.v_free = {s.id: min(v, KAPPA * limit_mps(s)) for s in network.segments}
        self.set_traffic({})

    def set_traffic(self, snap: dict, obeys_red: bool | None = None) -> None:
        if obeys_red is not None:
            self.obeys_red = obeys_red
        for s in self.network.segments:
            t = traffic_seconds(s, float(s.length_m), self.v_free[s.id], self.lanes.get(s.id, 1), snap.get(s.id),
                                self.calib, s.end_id in self.signals, self.obeys_red)
            self.cost[s.id] = t * self.v
        self.fwd = {n: [(self.seg[s].end_id, self.cost[s]) for s in ss] for n, ss in self.out_segs.items()}
        self.rev = {n: [(self.seg[s].start_id, self.cost[s]) for s in ss] for n, ss in self.in_segs.items()}
        self._to, self._from = {}, {}

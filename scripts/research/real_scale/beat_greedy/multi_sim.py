"""N police vs M fugitives on a real OSM ModelNetwork (PEPCRL-MVP style P6-E2, P7-E3, P8-E4 ...).

Same movement semantics as osm_demo.environment.OSMRoadPursuitEnv (constant speed
along segment geometry, zero-time virtual-hop chains at decisions, action slots
from ordered_outgoing_segment_ids) and the same evader (research GoalEvader,
one instance and one RNG stream per fugitive), generalised to M fugitives:

  * a fugitive is captured when any officer is within capture_radius_m of it;
    it then leaves the map (officers keep going);
  * the episode FAILS (escape) as soon as an uncaptured fugitive completes a
    boundary-crossing segment (capture in the same step wins), SUCCEEDS when
    every fugitive is captured, and times out at max_steps.

Team policies implement ``act(sim) -> list[provider]`` where a provider is the
env-style callable ``(agent_id, intersection_id, ordered_segment_ids, hop) -> slot``
(STAY_ACTION = stay); it is only consulted for officers that are parked.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Sequence

import numpy as np

from pursuit_evasion_rl.osm_demo.coarsening import ordered_outgoing_segment_ids
from pursuit_evasion_rl.osm_demo.environment import _Vehicle, _geometry_length, _heading_at_end, _point_at_fraction
from pursuit_evasion_rl.osm_demo.models import ModelNetwork
from pursuit_evasion_rl.research.policies.baselines import STAY_ACTION, GoalEvader, _Graph

VISION_M = 150.0                      # same as the sealed replay (REPLAY_VISION_RANGE_M)
Provider = Callable[[str, int, Sequence[int], int], int]


@dataclass(frozen=True)
class MultiCase:
    police: tuple[int, ...]           # start intersections
    fugitives: tuple[int, ...]
    seeds: tuple[int, ...]            # one evader RNG seed per fugitive
    vp: float = 16.0
    vf: float = 16.0
    capture_radius_m: float = 25.0
    max_steps: int = 600


@dataclass
class MultiResult:
    outcome: str                      # success | escape | timeout
    steps: int
    captured: int
    capture_steps: list = field(default_factory=list)
    trajectory: list | None = None


_GRAPHS: dict[int, _Graph] = {}


def graph_of(network: ModelNetwork) -> _Graph:
    g = _GRAPHS.get(id(network))
    if g is None or g.network is not network:
        g = _GRAPHS[id(network)] = _Graph(network)
    return g


class MultiPursuitSim:
    def __init__(self, network: ModelNetwork, case: MultiCase, record: bool = False) -> None:
        self.network, self.case = network, case
        self.g = graph_of(network)
        self.seg = {s.id: s for s in network.segments}
        self.inter = {i.id: i for i in network.intersections}
        self.seg_len = {s.id: _geometry_length(s.geometry_xy) for s in network.segments}
        self.hop_cap = sum(1 for i in network.intersections if i.virtual) + 1
        self.police = [_Vehicle(f"police_{k}", True, case.vp, "parked", intersection_id=n, physical_anchor=n)
                       for k, n in enumerate(case.police)]
        self.fugs = [_Vehicle(f"fugitive_{j}", False, case.vf, "parked", intersection_id=n, physical_anchor=n)
                     for j, n in enumerate(case.fugitives)]
        self.evaders = [GoalEvader(network, rng=np.random.default_rng(s), graph=self.g, vision_range_m=VISION_M)
                        for s in case.seeds]
        self.alive = [True] * len(self.fugs)
        self.capture_steps: list[int] = []
        self.step_count = 0
        self.outcome: str | None = None
        self.trajectory = [] if record else None
        self._check_capture()

    # -- geometry ---------------------------------------------------------------
    def position(self, v: _Vehicle) -> tuple[float, float]:
        if v.mode == "segment" and v.segment_id is not None:
            return _point_at_fraction(self.seg[v.segment_id].geometry_xy, v.progress)
        return self.inter[v.intersection_id].position_xy

    def reach(self, v: _Vehicle) -> tuple[int, float]:
        """(next decision node, remaining metres to it)."""
        if v.mode == "segment" and v.segment_id is not None:
            s = self.seg[v.segment_id]
            return int(s.end_id), (1.0 - v.progress) * float(s.length_m)
        return int(v.intersection_id), 0.0

    def police_xy(self):
        return [self.position(v) for v in self.police]

    def alive_fugitives(self) -> list[int]:
        return [j for j, a in enumerate(self.alive) if a]

    # -- dynamics (mirrors OSMRoadPursuitEnv) -------------------------------------
    def _advance(self, v: _Vehicle) -> bool:
        if v.mode != "segment" or v.segment_id is None:
            return False
        s = self.seg[v.segment_id]
        length = self.seg_len[v.segment_id]
        p = 1.0 if length <= 0 else v.progress + v.speed / length
        if p < 1.0:
            v.progress = p
            return False
        v.mode, v.progress, v.intersection_id, v.physical_anchor = "parked", 1.0, s.end_id, s.end_id
        h = _heading_at_end(s.geometry_xy)
        if h is not None:
            v.heading = h
        return bool(s.crosses_boundary)

    def _resolve(self, v: _Vehicle, provider: Provider) -> None:
        hops = 0
        for hop in range(self.hop_cap + 1):
            node = v.intersection_id
            ordered = ordered_outgoing_segment_ids(self.network, node, v.heading)
            a = int(provider(v.agent_id, node, ordered, hop))
            if a == STAY_ACTION or a < 0 or a >= len(ordered):
                return
            s = self.seg[ordered[a]]
            if not self.inter[node].virtual:
                v.physical_anchor = node
            if s.virtual:
                if hops >= self.hop_cap:
                    return
                hops += 1
                v.intersection_id = s.end_id
                continue
            v.mode, v.segment_id, v.progress = "segment", s.id, 0.0
            return

    def _check_capture(self) -> None:
        pxy = self.police_xy()
        r = self.case.capture_radius_m
        for j in self.alive_fugitives():
            fxy = self.position(self.fugs[j])
            if any(math.dist(q, fxy) <= r for q in pxy):
                self.alive[j] = False
                self.capture_steps.append(self.step_count)

    def step(self, providers: Sequence[Provider]) -> None:
        pxy = self.police_xy()
        fug_prov = {j: self.evaders[j].env_provider(self.network, pxy) for j in self.alive_fugitives()}
        for v in self.police:
            self._advance(v)
        crossed = [j for j in self.alive_fugitives() if self._advance(self.fugs[j])]
        for k, v in enumerate(self.police):
            if v.mode == "parked":
                self._resolve(v, providers[k])
        for j, prov in fug_prov.items():
            if self.fugs[j].mode == "parked":
                self._resolve(self.fugs[j], prov)
        self.step_count += 1
        self._check_capture()                       # capture in the same step beats escape
        if any(self.alive[j] for j in crossed):
            self.outcome = "escape"
        elif not any(self.alive):
            self.outcome = "success"
        elif self.step_count >= self.case.max_steps:
            self.outcome = "timeout"
        if self.trajectory is not None:
            self.trajectory.append(dict(step=self.step_count, police=self.police_xy(),
                                        fugitives=[self.position(f) if self.alive[j] else None
                                                   for j, f in enumerate(self.fugs)],
                                        alive=list(self.alive), info=getattr(self, "_info", None)))

    def run(self, team_policy) -> MultiResult:
        while self.outcome is None:
            self.step(team_policy.act(self))
        return MultiResult(self.outcome, self.step_count, len(self.capture_steps), list(self.capture_steps),
                           self.trajectory)


# -- placement ----------------------------------------------------------------------
def make_multi_case(network: ModelNetwork, tag: str, idx: int, n_police: int, n_fug: int, *,
                    vp: float = 16.0, vf: float = 16.0, max_steps: int = 600,
                    min_exit_m: float = 200.0, min_sep_m: float = 300.0) -> MultiCase:
    """Distinct random starts; fugitives interior (>= min_exit_m road metres from the boundary,
    >= min_sep_m apart, reachable by some officer) -- the multi-fugitive analogue of BG_SOLVABLE."""
    from common import stable_seed
    from obs_ext import boundary_distance
    g = graph_of(network)
    bd = boundary_distance(network)
    drivable = [i.id for i in network.intersections if not i.virtual and i.outgoing_segment_ids]
    boundary_free = {i.id for i in network.intersections if i.boundary_kind is None}
    interior = [n for n in drivable if n in boundary_free and bd.get(n, 0.0) >= min_exit_m]
    attempt = 0
    while True:
        rng = np.random.default_rng(stable_seed("multi", tag, idx, n_police, n_fug, attempt))
        attempt += 1
        fugs: list[int] = []
        for n in rng.permutation(interior):
            n = int(n)
            if all(math.dist(g.pos[n], g.pos[m]) >= min_sep_m for m in fugs):
                fugs.append(n)
            if len(fugs) == n_fug:
                break
        if len(fugs) < n_fug:
            continue
        pool = [n for n in drivable if n not in fugs]
        police = [int(x) for x in rng.choice(pool, size=n_police, replace=False)]
        if all(any(p in g.dist_to(f) for p in police) for f in fugs):
            seeds = tuple(stable_seed("multi-evader", tag, idx, j) for j in range(n_fug))
            return MultiCase(tuple(police), tuple(fugs), seeds, vp, vf, max_steps=max_steps)


# -- bridge: single-fugitive ``recommend`` team policies (encircle, greedy, macro ...) -----------
class RecommendBridge:
    """Runs a ``recommend(police=, fugitive=, step=, max_steps=, incoming_headings=)`` team policy
    (written for OSMRoadPursuitEnv / replay_episode) in a one-fugitive MultiPursuitSim, with any
    number of officers. Mirrors replay_episode: placements are projected like
    OSMRoadPursuitEnv._placement, and an officer's integer action applies only when it was parked
    with a live choice when the recommendation was made (hop 0; later virtual hops stay)."""

    def __init__(self, team_policy) -> None:
        self.policy = team_policy

    @staticmethod
    def placement(sim: "MultiPursuitSim", v: _Vehicle):
        from pursuit_evasion_rl.osm_demo.models import VehiclePlacement
        if v.mode == "segment" and v.segment_id is not None:
            return VehiclePlacement(segment_id=v.segment_id, progress=v.progress)
        anchor = v.physical_anchor
        if anchor is None or sim.inter[anchor].virtual:
            anchor = v.intersection_id
            if sim.inter[anchor].virtual:
                xy = sim.inter[anchor].position_xy
                anchor = next((i.id for i in sim.network.intersections if not i.virtual and i.position_xy == xy), anchor)
        return VehiclePlacement(intersection_id=anchor)

    def act(self, sim: "MultiPursuitSim") -> list[Provider]:
        recs = self.policy.recommend(
            police=tuple(self.placement(sim, v) for v in sim.police),
            fugitive=self.placement(sim, sim.fugs[0]), step=sim.step_count, max_steps=sim.case.max_steps,
            incoming_headings=tuple(v.heading for v in sim.police),
        )
        providers = []
        for v, rec in zip(sim.police, recs):
            live = v.mode == "parked" and len(ordered_outgoing_segment_ids(sim.network, v.intersection_id, v.heading)) > 0
            a = int(rec.action_index) if live else STAY_ACTION
            providers.append(lambda agent_id, node, ordered, hop, a=a: a if hop == 0 else STAY_ACTION)
        return providers


# -- encirclement-start scenario --------------------------------------------------------------
def make_ring_case(network: ModelNetwork, tag: str, idx: int, n_police: int, *, vp: float = 16.0,
                   vf: float = 16.0, max_steps: int = 600, ring_min_m: float = 400.0,
                   ring_max_m: float = 700.0, centre_frac: float = 0.6) -> MultiCase:
    """One fugitive near the middle of the map (>= centre_frac x the map's largest road distance to
    the boundary), officers already surrounding it: one per equal bearing sector, each
    ring_min_m..ring_max_m of road from the fugitive."""
    from common import stable_seed
    from obs_ext import boundary_distance
    g = graph_of(network)
    bd = boundary_distance(network)
    drivable = [i.id for i in network.intersections if not i.virtual and i.outgoing_segment_ids]
    free = {i.id for i in network.intersections if i.boundary_kind is None}
    deep = max(bd.get(n, 0.0) for n in drivable)
    centre = [n for n in drivable if n in free and bd.get(n, 0.0) >= centre_frac * deep]
    attempt = 0
    while True:
        rng = np.random.default_rng(stable_seed("ring", tag, idx, n_police, attempt))
        attempt += 1
        f = int(rng.choice(centre))
        to_f = g.dist_to(f)
        fx, fy = g.pos[f]
        band = [n for n in drivable if n != f and n in to_f
                and ring_min_m <= to_f[n] <= ring_max_m]
        if len(band) < n_police:
            continue
        bearing = {n: math.atan2(g.pos[n][1] - fy, g.pos[n][0] - fx) % (2 * math.pi) for n in band}
        off = float(rng.uniform(0, 2 * math.pi / n_police))
        police: list[int] = []
        for k in range(n_police):
            lo = (off + 2 * math.pi * k / n_police) % (2 * math.pi)
            mid = (lo + math.pi / n_police) % (2 * math.pi)
            sector = [n for n in band if n not in police and (bearing[n] - lo) % (2 * math.pi) < 2 * math.pi / n_police]
            pool = sector or [n for n in band if n not in police]
            if not sector:   # empty sector: the free ring node closest in bearing to its middle
                pool = sorted(pool, key=lambda n: abs(math.remainder(bearing[n] - mid, 2 * math.pi)))[:1]
            police.append(int(rng.choice(pool)))
        seeds = (stable_seed("ring-evader", tag, idx),)
        return MultiCase(tuple(police), (f,), seeds, vp, vf, max_steps=max_steps)




# -- containment analysis ---------------------------------------------------------------------
def escape_region(g: _Graph, reach: Sequence[tuple[int, float]], vp: float, vf: float,
                  exits: Sequence[int], margin_s: float = 25.0 / 16.0) -> tuple[set, list]:
    """Intersections the fugitive can reach strictly before every officer, and the boundary exits
    among them. ``reach`` = [(next node, metres to it)] for each officer, then the fugitive.
    Dijkstra from the fugitive that may not pass a node an officer reaches first (within
    margin_s, i.e. the capture radius at officer speed)."""
    import heapq
    tp: dict[int, float] = {}
    for n0, off in reach[:-1]:
        for node, d in g.dist_from(n0).items():
            t = (off + d) / vp
            if t < tp.get(node, math.inf):
                tp[node] = t
    f0, foff = reach[-1]
    best, heap, region = {f0: foff / vf}, [(foff / vf, f0)], set()
    while heap:
        t, u = heapq.heappop(heap)
        if t > best.get(u, math.inf) or t >= tp.get(u, math.inf) - margin_s:
            continue
        region.add(u)
        for nb, w in g.fwd[u]:
            nt = t + w / vf
            if nt < best.get(nb, math.inf):
                best[nb] = nt
                heapq.heappush(heap, (nt, nb))
    return region, [e for e in exits if e in region]

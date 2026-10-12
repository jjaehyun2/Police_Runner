"""Police-vs-fugitive pursuit inside SUMO (TraCI), on the one-to-one SUMO copy of a ModelNetwork.

Control split (the PEPCRL-MVP two-level structure, with SUMO as the lower level):
  upper level   the same team policies as the point-mass simulator (role GA policy, encirclement,
                greedy ...) run once per simulated second on placements read back from SUMO and
                output one *target node* per officer (hold an interception node, or pursue = the
                fugitive's next intersection);
  lower level   each officer's route is the shortest road path to its target, handed to SUMO with
                ``setRoute``; SUMO drives it (car following, junctions, lanes, signals in the
                "real" profile). Every route ends with a stop at its last edge, so an officer holds
                its target node instead of leaving the simulation.
  fugitive      the same GoalEvader, asked for its next edge shortly before it reaches each
                intersection, with the officers' current SUMO positions.

Barricades (optional): a closed road, both directions, at a point along it. In SUMO each lane gets a
stopped obstacle there, so traffic queues behind it. The police placed them, so officers route
around them from the start; the fugitive only learns of one once it is within BARRICADE_SIGHT_M,
then drops that road from its own map, re-plans, and turns back if it is already driving towards it.

Carried barricades (carry_barricades=True, prototype): every patrol car carries one; placing it
closes one road at an intersection the officer reached before the fugitive could (all judged on the
fugitive's escape region: the intersections it reaches before every officer, and the open exits in it).
  planned    (off by default) while exits are open, every PLAN_EVERY_S an officer still carrying one (never the one
             closest to the fugitive) is sent ahead to seal one: either the exit road itself or the
             earliest road on the fugitive's shortest route to it that the officer reaches first.
             The candidate that shuts the most open exits wins (ties: the quickest officer). On
             arrival it places the barricade if that still shrinks the escape region (judged as if
             the officer drove on), then rejoins its role.
  at a node  an officer within DEPLOY_REACH_M of an intersection it controls places it there when a
             road there shuts an open exit; "stand-in": when it can replace the officer holding that
             intersection (no larger region without that officer), freeing it for the chase;
             "leave": when the officer is about to leave the intersection it was holding, on the
             road that keeps the region smallest after it has gone; "pass" (off by default): any
             intersection it passes, when a road there cuts the region by PASS_MIN_CUT.
At most one placement per second for the whole team.

Outcomes as in multi_sim: success when an officer is within capture_radius_m of the fugitive
(checked every SUMO step), escape when the fugitive completes a boundary-crossing edge,
timeout at case.max_steps seconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import math
import os
import subprocess
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402

import common  # noqa: E402
import macro  # noqa: E402
from multi_sim import MultiCase, graph_of  # noqa: E402
from pursuit_evasion_rl.osm_demo.coarsening import ordered_outgoing_segment_ids  # noqa: E402
from pursuit_evasion_rl.osm_demo.environment import _heading_at_end  # noqa: E402
from pursuit_evasion_rl.osm_demo.models import VehiclePlacement  # noqa: E402
from pursuit_evasion_rl.research.policies.baselines import STAY_ACTION, GoalEvader, _Graph  # noqa: E402

VISION_M = 150.0
BARRICADE_SIGHT_M = 80.0
DEPLOY_REACH_M = 30.0            # an officer this close to an intersection may barricade a road there
DEPLOY_OFFSET_M = 20.0           # barricade this far into the road from that intersection
# which placement rules are on: leave, stand-in (default); planned, pass (tried, see HANDOFF: they cost more
# than they gain on daejeon_demo, as each closed road also slows the officers)
BARRICADE_RULES = set(os.environ.get("SUMO_BARRICADE_RULES", "leave,stand-in").split(","))
PASS_MIN_CUT = float(os.environ.get("SUMO_BARRICADE_PASS_CUT", "0.05"))  # "pass": region share a road must cut
PLAN_EVERY_S = 4                 # how often a barricade task is planned
PLAN_MAX_TRIP_S = 40.0           # carriers are only sent this far
PLAN_MIN_GAIN = 0.1              # ... to a road that shuts at least this share of the open exits (and 2+)
PLAN_CANDIDATES = 15
PLAN_SLACK = 1.5                 # a carrier may be sent if it arrives within this multiple of the
                                 # fugitive's fastest time there (it rarely drives straight to an exit)
CAPTURE_MARGIN_S = 25.0 / 16.0
HOLD_S = 1e6
PROFILES = {
    # vehicle dynamics per network profile: (accel, decel, emergencyDecel) m/s^2
    "ideal": (50.0, 50.0, 60.0),
    "real": (3.5, 7.5, 9.0),
}
_LABELS = itertools.count()


@dataclass
class SumoResult:
    outcome: str
    seconds: float
    trajectory: list = field(default_factory=list)


def _edge_sid(edge_id: str) -> int | None:
    return int(edge_id[1:]) if edge_id.startswith("e") else None


@dataclass(frozen=True)
class Barricade:
    xy: tuple[float, float]
    at: tuple[tuple[int, float], ...]                           # (directed segment id, fraction along it)


def _polyline_length(pts) -> float:
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))


def barricade_at(network, xy) -> Barricade:
    """Close the road nearest to ``xy`` (every directed segment between its two intersections)."""
    best = None
    for s in network.segments:
        pts, run = s.geometry_xy, 0.0
        for a, b in zip(pts, pts[1:]):
            dx, dy = b[0] - a[0], b[1] - a[1]
            ln2 = dx * dx + dy * dy
            u = 0.0 if ln2 == 0 else min(1.0, max(0.0, ((xy[0] - a[0]) * dx + (xy[1] - a[1]) * dy) / ln2))
            q = (a[0] + u * dx, a[1] + u * dy)
            d = math.dist(q, xy)
            if best is None or d < best[0]:
                best = (d, s, run + u * math.sqrt(ln2), q)
            run += math.sqrt(ln2)
    _, s0, along, q = best
    frac = min(0.95, max(0.05, along / max(_polyline_length(s0.geometry_xy), 1e-6)))
    at = []
    for s in network.segments:
        if (s.start_id, s.end_id) == (s0.start_id, s0.end_id):
            at.append((int(s.id), frac))
        elif (s.start_id, s.end_id) == (s0.end_id, s0.start_id):
            at.append((int(s.id), 1.0 - frac))
    return Barricade((float(q[0]), float(q[1])), tuple(at))


def barricade_on(network, u: int, v: int, offset_m: float = DEPLOY_OFFSET_M) -> Barricade:
    """Close the road u-v (both directions) offset_m in from u."""
    segs = [s for s in network.segments if {s.start_id, s.end_id} == {u, v} and s.start_id != s.end_id]
    s0 = next(s for s in segs if s.start_id == u) if any(s.start_id == u for s in segs) else segs[0]
    length = _polyline_length(s0.geometry_xy)
    d = min(offset_m, 0.4 * length)
    frac_from_u = d / max(length, 1e-6)
    pts, run, xy = s0.geometry_xy if s0.start_id == u else s0.geometry_xy[::-1], 0.0, None
    for a, b in zip(pts, pts[1:]):
        step = math.dist(a, b)
        if run + step >= d:
            w = (d - run) / max(step, 1e-9)
            xy = (a[0] + w * (b[0] - a[0]), a[1] + w * (b[1] - a[1]))
            break
        run += step
    xy = xy or pts[-1]
    at = tuple((int(s.id), frac_from_u if s.start_id == u else 1.0 - frac_from_u) for s in segs)
    return Barricade((float(xy[0]), float(xy[1])), at)


def _close(g: _Graph, sids) -> None:
    """Remove directed segments from a private graph copy (and drop its distance caches)."""
    for sid in sids:
        s = g.seg[sid]
        g.fwd[s.start_id].remove((s.end_id, float(s.length_m)))
        g.rev[s.end_id].remove((s.start_id, float(s.length_m)))
    g._to.clear()
    g._from.clear()


class TargetPolicy:
    """Wraps a ``recommend`` team policy (macro / encirclement / greedy) to expose per-officer
    targets: ("hold", node) or ("pursue", None)."""

    def __init__(self, spec: str, network, n_police: int) -> None:
        import multi_eval
        self.team = multi_eval.team_for(spec, network, n_police).policy
        self.n = n_police
        self.stash: dict = {}
        if hasattr(self.team, "_assign"):                       # encirclement: capture its targets
            orig = self.team._assign

            def assign(decisions, fnode, *a):
                t = orig(decisions, fnode, *a)
                self.stash["targets"], self.stash["fnode"] = t, fnode
                return t
            self.team._assign = assign

    def targets(self, police, fugitive, step, max_steps, headings):
        self.team.recommend(police=police, fugitive=fugitive, step=step, max_steps=max_steps,
                            incoming_headings=headings)
        return self.goals()

    def goals(self):
        """Targets behind the policy's latest recommendation."""
        if isinstance(self.team, macro.MacroTeamPolicy):
            return [("pursue", None) if (r == 0 or t is None) else ("hold", int(t))
                    for r, t in zip(self.team.roles, self.team.targets)]
        if "targets" in self.stash:
            fn = self.stash["fnode"]
            return [("pursue", None) if t == fn else ("hold", int(t)) for t in self.stash["targets"]]
        return [("pursue", None)] * self.n                     # greedy & other pure pursuers


class SumoPursuit:
    def __init__(self, map_name: str, case: MultiCase, spec: str, *, profile: str = "ideal",
                 step_length: float = 0.25, record: bool = False, gui: bool = False,
                 traffic: float | None = None, warmup_s: float = 300.0, ignore_red: bool = False,
                 speed_factor: float | None = None, on_frame=None, background: bool = False,
                 barricades: tuple[Barricade, ...] = (), carry_barricades: bool = False) -> None:
        """traffic      background demand period (s per random trip; None = empty roads)
        warmup_s     background traffic runs this long before the pursuit starts
        ignore_red   officers and fugitive run red lights / ignore right of way (emergency driving)
        speed_factor multiple of each road's speed limit the pursuit vehicles drive at, capped at
                     vp / vf (default: unlimited in "ideal", 2.0 in "real")
        barricades   closed roads (see barricade_at)
        carry_barricades  each officer carries one barricade and places it during the chase"""
        import sumo
        import traci
        self.map_name, self.case, self.profile, self.dt = map_name, case, profile, step_length
        self.network = common.load_network(map_name)
        self.g = graph_of(self.network)
        self.seg = {s.id: s for s in self.network.segments}
        self.heading_end = {s.id: _heading_at_end(s.geometry_xy) for s in self.network.segments}
        self._out: dict[int, list[int]] = {}
        for s in self.network.segments:
            self._out.setdefault(s.start_id, []).append(s.id)
        self.policy = TargetPolicy(spec, self.network, len(case.police))
        self.barricades = list(barricades)
        self.closed = {sid for b in self.barricades for sid, _ in b.at}
        self.kit = [bool(carry_barricades)] * len(case.police)     # officer still carries its barricade
        self.released: set[int] = set()                          # officers a barricade stood in for
        self.deployments: list[dict] = []
        self.tasks: dict[int, tuple[int, int]] = {}             # officer -> (node, neighbour) road to close
        self._held: dict[int, int] = {}                          # officer -> intersection it was holding
        if self.barricades or carry_barricades:
            self.pg = _Graph(self.network)                       # officers' (true) road map
            _close(self.pg, self.closed)
            from obs_ext import exit_nodes
            self.exits = exit_nodes(self.network)
            self.fg = _Graph(self.network)                       # the fugitive's map: closes roads as it sees them
        else:
            self.pg = self.fg = self.g
        self.known: set[int] = set()                             # barricades the fugitive has seen
        self.known_sids: set[int] = set()
        self._uturn: str | None = None
        self._uturn_failed: dict[int, int] = {}
        self.trapped: int | None = None                          # segment the fugitive is stuck on
        self.evader = GoalEvader(self.network, rng=np.random.default_rng(case.seeds[0]), graph=self.fg,
                                 vision_range_m=VISION_M)
        self.record = record
        self.trajectory: list = []
        self.on_frame = on_frame                                 # callback(frame dict), once per simulated second
        self.background = background                             # include other traffic in frames
        net_file = HERE / "nets" / f"{map_name}_{profile}.net.xml"
        if not net_file.exists():
            from build_net import build
            build(map_name, profile)
        self.ignore_red = ignore_red
        self.speed_factor = speed_factor if speed_factor is not None else (1.0 if profile == "ideal" else 2.0)
        self.label = f"pursuit-{os.getpid()}-{next(_LABELS)}"
        binary = str(Path(sumo.SUMO_HOME) / "bin" / ("sumo-gui" if gui else "sumo"))
        cmd = [binary, "-n", str(net_file), "--step-length", str(step_length), "--no-step-log", "true",
               "--no-warnings", "true", "--collision.action", "none",
               "--seed", str(case.seeds[0] % 2**31)]
        if traffic is None:
            cmd += ["--time-to-teleport", "-1"]
            self.warmup_s = 0.0
        else:
            routes = HERE / "nets" / f"{map_name}_{profile}_traffic{traffic:g}.rou.xml"
            if not routes.exists():
                from build_net import traffic as make_traffic
                make_traffic(map_name, profile, traffic)
            # background vehicles stuck in gridlock are teleported; pursuit vehicles are checked below
            cmd += ["-r", str(routes), "--time-to-teleport", "120", "--ignore-route-errors", "true"]
            self.warmup_s = warmup_s
        if gui:
            cmd += ["--start", "--delay", str(int(os.environ.get("SUMO_GUI_DELAY_MS", "100"))), "--quit-on-end"]
        traci.start(cmd, label=self.label, stdout=subprocess.DEVNULL)
        self.conn = traci.getConnection(self.label)
        self.police_ids = [f"police_{k}" for k in range(len(case.police))]
        self.fid = "fugitive"
        self.blocker_ids: list[str] = []
        self._target: dict[str, tuple] = {}
        self._decided: dict[str, int] = {}                     # fugitive: edge -> next edge decided
        self._last_edge: str | None = None
        self.teleports = 0
        self._last_goals: list = []
        self._spawn()

    # -- geometry / state ----------------------------------------------------------------------
    def _path_edges(self, src_node: int, dst_node: int) -> list[str]:
        """Shortest directed path as edge ids for an officer (around every barricade)."""
        to = self.pg.dist_to(dst_node)
        if src_node not in to:
            return []
        out, cur, guard = [], src_node, 0
        while cur != dst_node and guard < 1000:
            guard += 1
            best = None
            for sid in self.network_out(cur):
                if sid in self.closed:
                    continue
                s = self.seg[sid]
                d = to.get(s.end_id)
                if d is not None and (best is None or s.length_m + d < best[0]):
                    best = (s.length_m + d, sid)
            if best is None:
                break
            out.append(f"e{best[1]}")
            cur = self.seg[best[1]].end_id
        return out

    def network_out(self, node: int) -> list[int]:
        return self._out.get(node, [])

    def _current_edge(self, vid: str) -> str | None:
        road = self.conn.vehicle.getRoadID(vid)
        if road.startswith(":") or not road:
            return None
        return road

    def _placement(self, vid: str):
        road = self.conn.vehicle.getRoadID(vid)
        if road.startswith(":"):                                # on a junction: just entering the next edge
            route, idx = self.conn.vehicle.getRoute(vid), self.conn.vehicle.getRouteIndex(vid)
            nxt = route[idx + 1] if idx + 1 < len(route) else route[idx]
            return VehiclePlacement(segment_id=_edge_sid(nxt), progress=0.0), _edge_sid(nxt)
        sid = _edge_sid(road)
        length = self.conn.lane.getLength(self.conn.vehicle.getLaneID(vid))
        p = min(max(self.conn.vehicle.getLanePosition(vid) / max(length, 1e-6), 0.0), 1.0)
        return VehiclePlacement(segment_id=sid, progress=p), sid

    def _reach(self, vid: str) -> tuple[int, float]:
        """(next intersection, road metres to it)."""
        pl, sid = self._placement(vid)
        return int(self.seg[sid].end_id), (1.0 - pl.progress) * float(self.seg[sid].length_m)

    # -- control -------------------------------------------------------------------------------
    def _set_route(self, vid: str, edges: list[str]) -> None:
        """Replace the route (first edge = the current edge) and park at the end of its last edge."""
        c = self.conn.vehicle
        if c.isStopped(vid):
            c.resume(vid)
        for _ in range(len(c.getStops(vid, 8))):
            c.replaceStop(vid, 0, "")
        import traci
        edges = list(edges)
        for _ in range(4):
            c.setRoute(vid, edges)
            last = edges[-1]
            try:
                c.setStop(vid, last, pos=max(self.conn.lane.getLength(f"{last}_0") - 0.5, 0.1), laneIndex=0,
                          duration=HOLD_S)
                return
            except traci.TraCIException:
                # too close to brake for the stop (real decelerations): hold one edge further on,
                # preferring the longest non-U-turn continuation
                sid = _edge_sid(last)
                back = self.seg[sid].start_id
                outs = sorted(self.network_out(self.seg[sid].end_id),
                              key=lambda o: (self.seg[o].end_id == back, -self.seg[o].length_m))
                if not outs:
                    return
                edges.append(f"e{outs[0]}")

    def _steer_police(self, k: int, goal: tuple, fnode: int) -> None:
        vid = self.police_ids[k]
        cur = self._current_edge(vid)
        if cur is None:
            return                                              # crossing a junction; next second
        node = fnode if goal[0] == "pursue" else goal[1]
        trapped = self.trapped if goal[0] == "pursue" else None
        key = (goal[0], node, cur, trapped)
        if self._target.get(vid) == key:
            return
        start = int(self.seg[_edge_sid(cur)].end_id)
        if trapped is not None:                                 # come up behind it on its own road
            edges = [cur] if _edge_sid(cur) == trapped else \
                [cur] + self._path_edges(start, int(self.seg[trapped].start_id)) + [f"e{trapped}"]
        else:
            edges = [cur] + self._path_edges(start, node)
        lane_pos = self.conn.vehicle.getLanePosition(vid)
        if len(edges) == 1 and lane_pos > self.conn.lane.getLength(f"{cur}_0") - 0.6:
            self._target[vid] = key                             # already holding at the node
            return
        try:
            self._set_route(vid, edges)
            self._target[vid] = key
        except Exception:                                       # noqa: BLE001 (TraCI rejects e.g. a stop behind us)
            self._target.pop(vid, None)

    def _fugitive_next(self, cur: str, pxy) -> str | None:
        sid = _edge_sid(cur)
        node = int(self.seg[sid].end_id)
        ordered = [o for o in ordered_outgoing_segment_ids(self.network, node, self.heading_end[sid])
                   if o not in self.known_sids]
        provider = self.evader.env_provider(self.network, pxy)
        slot = int(provider(self.fid, node, ordered, 0))
        if slot == STAY_ACTION or slot >= len(ordered):
            return None
        self.evader.prev_node = node
        return f"e{ordered[slot]}"

    def _steer_fugitive(self, pxy) -> None:
        cur = self._current_edge(self.fid)
        if cur is None or cur in self._decided:
            return
        length = self.conn.lane.getLength(self.conn.vehicle.getLaneID(self.fid))
        remaining = length - self.conn.vehicle.getLanePosition(self.fid)
        if remaining > 2.0 * self.case.vf * self.dt + 1.0:
            return
        nxt = self._fugitive_next(cur, pxy)
        self._decided = {cur: nxt}
        self._set_route(self.fid, [cur] + ([nxt] if nxt else []))

    def _see_barricades(self, fxy) -> None:
        """The fugitive notices barricades within sight: it closes those roads on its own map,
        drops its goal and pending turn, and turns back if one lies ahead on its current road."""
        new = [k for k, b in enumerate(self.barricades) if k not in self.known and math.dist(b.xy, fxy) <= BARRICADE_SIGHT_M]
        if not new:
            return
        for k in new:
            self.known.add(k)
            sids = [sid for sid, _ in self.barricades[k].at]
            self.known_sids.update(sids)
            _close(self.fg, sids)
        self.evader.goal = None
        self._decided = {}

    def _blocked_ahead(self) -> int | None:
        """The fugitive's current segment when a barricade it knows of lies ahead on it."""
        cur = self._current_edge(self.fid)
        if cur is None or _edge_sid(cur) not in self.known_sids:
            return None
        sid = _edge_sid(cur)
        lane_len = self.conn.lane.getLength(self.conn.vehicle.getLaneID(self.fid))
        pos = self.conn.vehicle.getLanePosition(self.fid)
        ahead = any(s == sid and frac * lane_len > pos for k in self.known for s, frac in self.barricades[k].at)
        return sid if ahead else None

    def _turn_back(self) -> None:
        """Turn the fugitive around when it faces a barricade it knows of (if the road is two-way);
        otherwise it is trapped and waits behind it."""
        sid = self._blocked_ahead()
        self.trapped = None
        if sid is None:
            return
        seg = self.seg[sid]
        rev = [o for o in self.network_out(seg.end_id) if self.seg[o].end_id == seg.start_id]
        if not rev or self._uturn_failed.get(sid, 0) >= 3:
            self.trapped = sid                                  # one-way road: stuck behind the barricade
            return
        c = self.conn.vehicle
        if c.isStopped(self.fid):
            c.resume(self.fid)
        for _ in range(len(c.getStops(self.fid, 8))):
            c.replaceStop(self.fid, 0, "")
        x, y = c.getPosition(self.fid)
        try:
            c.moveToXY(self.fid, f"e{rev[0]}", 0, x, y, angle=(c.getAngle(self.fid) + 180.0) % 360.0, keepRoute=0)
        except Exception:                                       # noqa: BLE001 (retried; then it is trapped)
            self._uturn_failed[sid] = self._uturn_failed.get(sid, 0) + 1
            return
        self._uturn, self._uturn_steps, self._uturn_from = f"e{rev[0]}", 0, sid
        self.evader.prev_node = seg.end_id

    # -- carried barricades ----------------------------------------------------------------------
    def _escape(self, tp: dict, freach, skip: set) -> tuple[int, int]:
        """(escape-region size, number of open exits) on the officers' map with the node pairs in
        ``skip`` also closed; ``tp`` = officers' earliest arrival times. The open exits themselves
        are left in ``self._open``."""
        import heapq
        f0, foff = freach
        g, vf = self.pg, self.case.vf
        best, heap, region = {f0: foff / vf}, [(foff / vf, f0)], set()
        while heap:
            t, u = heapq.heappop(heap)
            if t > best.get(u, math.inf) or t >= tp.get(u, math.inf) - CAPTURE_MARGIN_S:
                continue
            region.add(u)
            for nb, w in g.fwd[u]:
                if (u, nb) in skip:
                    continue
                nt = t + w / vf
                if nt < best.get(nb, math.inf):
                    best[nb] = nt
                    heapq.heappush(heap, (nt, nb))
        self._open = [e for e in self.exits
                      if e in region and any(self.seg[o].crosses_boundary and o not in self.closed
                                             and (e, self.seg[o].end_id) not in skip for o in self.network_out(e))]
        return len(region), len(self._open)

    def _deploy(self, goals, t: float) -> list:
        """Place or plan carried barricades (see the module docstring); returns the goals with
        sent officers heading to their road and freed officers chasing."""
        if any(self.kit):
            vp = self.case.vp
            reaches = [self._reach(v) for v in self.police_ids]
            freach = self._reach(self.fid)
            arrive = [{node: (off + d) / vp for node, d in self.pg.dist_from(n).items()} for n, off in reaches]

            def tp_without(skip_k):
                out: dict = {}
                for k, a in enumerate(arrive):
                    if k == skip_k:
                        continue
                    for node, tt in a.items():
                        if tt < out.get(node, math.inf):
                            out[node] = tt
                return out

            tp_all = tp_without(None)
            base_n, base_open = self._escape(tp_all, freach, set())
            open_exits = list(self._open)
            f_from = self.pg.dist_from(freach[0])
            t_f = lambda node: (freach[1] + f_from[node]) / self.case.vf if node in f_from else math.inf  # noqa: E731
            busy: dict = {}                                      # road -> officer on it (fugitive: -1)
            for j, v in enumerate(self.police_ids + [self.fid]):
                cur = self._current_edge(v)
                if cur is not None:
                    sg = self.seg[_edge_sid(cur)]
                    busy[(sg.start_id, sg.end_id)] = busy[(sg.end_id, sg.start_id)] = j if v != self.fid else -1

            def free_road(k, u, v):
                who = busy.get((u, v))
                return who is None or (who == k and reaches[k][1] <= 5.0)

            placed = None
            # 1) officers at an intersection: carry out a task, or place opportunistically
            for k, (u, off) in enumerate(reaches):
                if placed or not self.kit[k] or off > DEPLOY_REACH_M or tp_all.get(u, math.inf) >= t_f(u) - CAPTURE_MARGIN_S:
                    continue
                if k in self.tasks and self.tasks[k][0] == u:
                    tu, tv = self.tasks.pop(k)               # judged as if the officer drove on again
                    tp_k = tp_without(k)
                    if free_road(k, tu, tv) and \
                            self._escape(tp_k, freach, {(tu, tv), (tv, tu)}) < self._escape(tp_k, freach, set()):
                        placed = (k, tu, tv, False, "planned")
                    continue
                holding = goals[k][0] == "hold" and goals[k][1] == u and "stand-in" in BARRICADE_RULES
                leaving = "leave" in BARRICADE_RULES and self._held.get(k) == u and goals[k] != ("hold", u) and off <= 1.0
                passing = "pass" in BARRICADE_RULES and not leaving
                tp_free = tp_without(k) if holding or leaving or passing else None
                if leaving or passing:
                    n0, o0 = self._escape(tp_free, freach, set())
                best = None
                for v in {nb for nb, _ in self.pg.fwd[u]} | {nb for nb, _ in self.pg.rev[u]}:
                    if v == u or not free_road(k, u, v):
                        continue
                    skip = {(u, v), (v, u)}
                    n1, o1 = self._escape(tp_all, freach, skip)
                    cand = None
                    if o1 < base_open:
                        cand = (3, base_open - o1, base_n - n1, v, False, "exit")
                    elif leaving or passing:
                        n2, o2 = self._escape(tp_free, freach, skip)
                        if leaving and (o2, n2) < (o0, n0):
                            cand = (2, o0 - o2, n0 - n2, v, False, "leave")
                        elif passing and n0 and (o2 < o0 or (n0 - n2) / n0 >= PASS_MIN_CUT):
                            cand = (2, o0 - o2, n0 - n2, v, False, "pass")
                    elif holding:
                        n2, o2 = self._escape(tp_free, freach, skip)
                        if o2 <= base_open and n2 <= base_n:
                            cand = (1, 0, base_n - n2, v, True, "stand-in")
                    if cand and (best is None or cand[:3] > best[:3]):
                        best = cand
                if best is not None:
                    placed = (k, u, best[3], best[4], best[5])
            if placed is not None:
                k, u, v, free, kind = placed
                bar = barricade_on(self.network, u, v)
                idx = len(self.barricades)
                self.barricades.append(bar)
                sids = [sid for sid, _ in bar.at]
                self.closed.update(sids)
                _close(self.pg, sids)
                self._place_barricade(idx, bar)
                self.kit[k] = False
                self.tasks.pop(k, None)
                if free:
                    self.released.add(k)
                self._target.clear()                             # every officer re-routes around it
                self.deployments.append(dict(t=t, officer=k, node=u, to=v, freed=free, kind=kind))
            # 2) plan: send a carrier ahead to seal an open exit
            elif "planned" in BARRICADE_RULES and base_open and int(t) % PLAN_EVERY_S == 0:
                nearest = min(range(len(reaches)), key=lambda k: arrive[k].get(freach[0], math.inf))
                carriers = [k for k in range(len(reaches)) if self.kit[k] and k not in self.tasks and k != nearest] \
                    if len(self.tasks) < max(1, len(reaches) // 2) else []
                taken = set(self.tasks.values())
                if carriers:
                    best = None
                    need = max(2, PLAN_MIN_GAIN * base_open)
                    for road in self._choke_candidates(freach, open_exits):
                        if road in taken or (road[1], road[0]) in taken:
                            continue
                        # the carrier closes it from whichever end it reaches first
                        tk, k, w, x = min((arrive[k].get(w, math.inf), k, w, x) for k in carriers
                                          for w, x in (road, road[::-1]))
                        if tk > PLAN_MAX_TRIP_S or tk >= PLAN_SLACK * t_f(w) - CAPTURE_MARGIN_S:
                            continue
                        n1, o1 = self._escape(tp_all, freach, {(w, x), (x, w)})
                        if base_open - o1 < need:
                            continue
                        cand = (base_open - o1, -tk, k, w, x)
                        if best is None or cand[:2] > best[:2]:
                            best = cand
                    if best is not None:
                        self.tasks[best[2]] = (best[3], best[4])
            # drop tasks that can no longer be done first
            for k, (u, v) in list(self.tasks.items()):
                if arrive[k].get(u, math.inf) >= PLAN_SLACK * t_f(u) or (u, v) in busy and busy[(u, v)] == -1:
                    del self.tasks[k]
        for k, g in enumerate(goals):                            # remember who is holding where
            u, off = self._reach(self.police_ids[k])
            if g == ("hold", u) and off <= 1.0:
                self._held[k] = u
            elif g[0] != "hold" or g[1] != self._held.get(k):
                if not (self._held.get(k) == u and off <= 1.0):
                    self._held.pop(k, None)
        out = []
        for k, g in enumerate(goals):
            if k in self.tasks:
                out.append(("hold", self.tasks[k][0]))
            elif k in self.released:
                out.append(("pursue", None))
            else:
                out.append(g)
        return out

    def _choke_candidates(self, freach, open_exits) -> list:
        """Bottleneck roads of the fugitive's escape: in its shortest-path tree (officers' map), the
        roads with the most open exits below them (at least 2), best first."""
        import heapq
        f0, g = freach[0], self.pg
        dist, parent, heap = {f0: 0.0}, {}, [(0.0, f0)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, math.inf):
                continue
            for nb, w in g.fwd[u]:
                if d + w < dist.get(nb, math.inf):
                    dist[nb], parent[nb] = d + w, u
                    heapq.heappush(heap, (d + w, nb))
        below: dict = {}
        for e in open_exits:
            n = e
            while n in parent:
                road = (parent[n], n)
                below[road] = below.get(road, 0) + 1
                n = parent[n]
        ranked = sorted((c, r) for r, c in below.items() if c >= 2)[::-1]
        return [r for _, r in ranked[:PLAN_CANDIDATES]]

    def _fugitive_time(self, node: int, freach) -> float:
        f0, foff = freach
        d = self.pg.dist_from(f0).get(node)
        return math.inf if d is None else (foff + d) / self.case.vf

    def _spawn_barricades(self) -> None:
        c = self.conn
        c.vehicletype.copy("DEFAULT_VEHTYPE", "barricade")
        c.vehicletype.setLength("barricade", 2.0)
        c.vehicletype.setMinGap("barricade", 0.5)
        c.vehicletype.setColor("barricade", (245, 150, 20, 255))
        for k, b in enumerate(self.barricades):
            self._place_barricade(k, b)

    def _place_barricade(self, k: int, b: Barricade) -> None:
        c = self.conn
        for sid, frac in b.at:
            edge = f"e{sid}"
            c.route.add(f"r_bar_{k}_{sid}", [edge])
            for lane in range(c.edge.getLaneNumber(edge)):
                length = c.lane.getLength(f"{edge}_{lane}")
                pos = min(max(frac * length, 2.5), max(length - 0.5, 2.5))
                vid = f"barricade_{k}_{sid}_{lane}"
                try:
                    c.vehicle.add(vid, f"r_bar_{k}_{sid}", typeID="barricade", depart="now", departPos=str(pos),
                                  departSpeed="0", departLane=str(lane))
                    c.vehicle.setStop(vid, edge, pos=pos, laneIndex=lane, duration=HOLD_S)
                except Exception:                           # noqa: BLE001 (lane closed to cars, too short)
                    continue
                self.blocker_ids.append(vid)

    # -- episode -------------------------------------------------------------------------------
    def _spawn(self) -> None:
        c = self.conn
        warm = 0
        while warm * self.dt < self.warmup_s:                    # let background traffic fill the roads
            c.simulationStep(); warm += 1
        self.t0 = c.simulation.getTime()
        self._spawn_barricades()
        police_pl = tuple(VehiclePlacement(intersection_id=n) for n in self.case.police)
        fug_pl = VehiclePlacement(intersection_id=self.case.fugitives[0])
        goals = self.policy.targets(police_pl, fug_pl, 0, self.case.max_steps, (None,) * len(police_pl))
        accel, decel, emerg = PROFILES[self.profile]
        for vt, vmax in (("police", self.case.vp), ("fugitive", self.case.vf)):
            c.vehicletype.copy("DEFAULT_VEHTYPE", vt)
            c.vehicletype.setMaxSpeed(vt, vmax)
            c.vehicletype.setAccel(vt, accel)
            c.vehicletype.setDecel(vt, decel)
            c.vehicletype.setEmergencyDecel(vt, emerg)
            c.vehicletype.setImperfection(vt, 0.0)
            c.vehicletype.setTau(vt, 0.5)
            c.vehicletype.setSpeedFactor(vt, self.speed_factor)
            c.vehicletype.setColor(vt, (40, 90, 220, 255) if vt == "police" else (220, 60, 40, 255))
        fnode = self.case.fugitives[0]
        for k, (vid, n) in enumerate(zip(self.police_ids, self.case.police)):
            node = fnode if goals[k][0] == "pursue" else goals[k][1]
            first_out = [o for o in self.network_out(n) if o not in self.closed] or self.network_out(n)
            edges = self._path_edges(n, node) or [f"e{first_out[0]}"]
            c.route.add(f"r_{vid}", edges)
            c.vehicle.add(vid, f"r_{vid}", typeID="police", depart="now", departPos="0", departSpeed="0",
                          departLane="best")
            if self.ignore_red:
                c.vehicle.setSpeedMode(vid, 7)                   # no red-light / right-of-way braking
            if edges == [f"e{first_out[0]}"] and node == n:
                c.vehicle.setStop(vid, edges[0], pos=1.0, laneIndex=0, duration=HOLD_S)
            else:
                c.vehicle.setStop(vid, edges[-1], pos=max(c.lane.getLength(f"{edges[-1]}_0") - 0.5, 0.1),
                                  laneIndex=0, duration=HOLD_S)
            self._target[vid] = None
        # fugitive: first edge from its start intersection
        pxy = [self.g.pos[n] for n in self.case.police]
        fxy0 = self.g.pos[fnode]
        for k, b in enumerate(self.barricades):                  # barricades already in sight at the start
            if math.dist(b.xy, fxy0) <= BARRICADE_SIGHT_M:
                self.known.add(k)
                self.known_sids.update(sid for sid, _ in b.at)
        if self.known_sids:
            _close(self.fg, self.known_sids)
        ordered = [o for o in ordered_outgoing_segment_ids(self.network, fnode, None) if o not in self.known_sids] \
            or ordered_outgoing_segment_ids(self.network, fnode, None)
        slot = int(self.evader.env_provider(self.network, pxy)(self.fid, fnode, ordered, 0))
        first = f"e{ordered[slot if slot < len(ordered) else 0]}"
        self.evader.prev_node = fnode
        c.route.add("r_fugitive", [first])
        c.vehicle.add(self.fid, "r_fugitive", typeID="fugitive", depart="now", departPos="0", departSpeed="0",
                      departLane="best")
        if self.ignore_red:
            c.vehicle.setSpeedMode(self.fid, 7)
        c.vehicle.setStop(self.fid, first, pos=max(c.lane.getLength(f"{first}_0") - 0.5, 0.1), laneIndex=0,
                          duration=HOLD_S)

    def _emit(self, t, pxy, fxy, goals) -> None:
        self._last_goals = goals
        frame = dict(t=round(t, 2), p=[list(map(float, q)) for q in pxy], f=list(map(float, fxy)), goals=goals,
                     reach=[self._reach(v) for v in self.police_ids + [self.fid]], known=sorted(self.known),
                     bars=[b.xy for b in self.barricades], kit=list(self.kit))
        if self.background:
            mine = set(self.police_ids) | {self.fid} | set(self.blocker_ids)
            frame["bg"] = [list(map(float, self.conn.vehicle.getPosition(v)))
                           for v in self.conn.vehicle.getIDList() if v not in mine]
        if self.record:
            self.trajectory.append(frame)
        if self.on_frame:
            self.on_frame(frame)

    def _positions(self):
        return [self.conn.vehicle.getPosition(v) for v in self.police_ids], self.conn.vehicle.getPosition(self.fid)

    def run(self) -> SumoResult:
        c = self.conn
        per_second = max(1, round(1.0 / self.dt))
        outcome, t, i = None, 0.0, 0
        try:
            want = set(self.police_ids) | {self.fid}
            for _ in range(400):                                 # insert vehicles (a blocked start waits)
                c.simulationStep()
                live = set(c.vehicle.getIDList())
                if want | set(self.blocker_ids) <= live:
                    break
            else:
                if not want <= live:
                    raise RuntimeError("vehicles could not be inserted")
            while outcome is None:
                live = set(c.vehicle.getIDList())
                if self.fid not in live or any(v not in live for v in self.police_ids):
                    raise RuntimeError(f"pursuit vehicle left the simulation at t={t}")
                if set(c.simulation.getStartingTeleportIDList()) & (set(self.police_ids) | {self.fid}):
                    self.teleports += 1
                pxy, fxy = self._positions()
                if i % per_second == 0:                          # upper level, once per second
                    police_pl, headings = [], []
                    for v in self.police_ids:
                        pl, sid = self._placement(v)
                        police_pl.append(pl); headings.append(self.heading_end[sid])
                    fug_pl, fsid = self._placement(self.fid)
                    fnode = int(self.seg[fsid].end_id)
                    goals = self.policy.targets(tuple(police_pl), fug_pl, int(t), self.case.max_steps, tuple(headings))
                    if any(self.kit) or self.released:
                        goals = self._deploy(goals, t)
                    if self.trapped is not None:                # trapped behind a barricade: close in for the arrest
                        near = sorted(range(len(pxy)), key=lambda k: math.dist(pxy[k], fxy))[:2]
                        goals = [("pursue", None) if k in near else g for k, g in enumerate(goals)]
                    for k, goal in enumerate(goals):
                        self._steer_police(k, goal, fnode)
                    if self.record or self.on_frame:
                        self._emit(t, pxy, fxy, goals)
                if self.barricades:
                    self._see_barricades(fxy)
                    if self._uturn is None:
                        self._turn_back()
                turning = self._uturn is not None
                if not turning:
                    self._steer_fugitive(pxy)
                prev = self._current_edge(self.fid)
                c.simulationStep()
                i += 1
                t = i * self.dt
                if turning:
                    self._uturn_steps += 1
                    if self._current_edge(self.fid) == self._uturn:
                        self._set_route(self.fid, [self._uturn])  # heading back: hold at its end until it decides
                        self._uturn = None
                        self._decided = {}
                    elif self._uturn_steps > 8:                 # the move did not take: retry, then give up
                        self._uturn_failed[self._uturn_from] = self._uturn_failed.get(self._uturn_from, 0) + 1
                        self._uturn = None
                pxy, fxy = self._positions()
                if any(math.dist(q, fxy) <= self.case.capture_radius_m for q in pxy):
                    outcome = "success"
                    break
                now = self._current_edge(self.fid)
                if not turning and prev is not None and self.seg[_edge_sid(prev)].crosses_boundary:
                    lane_len = c.lane.getLength(f"{prev}_0")
                    if now != prev or c.vehicle.getLanePosition(self.fid) >= lane_len - 1.0:
                        outcome = "escape"
                        break
                if t >= self.case.max_steps:
                    outcome = "timeout"
            if self.record or self.on_frame:                     # the terminal moment
                pxy, fxy = self._positions()
                self._emit(t, pxy, fxy, self._last_goals)
        finally:
            c.close()
        return SumoResult(outcome, t, self.trajectory)

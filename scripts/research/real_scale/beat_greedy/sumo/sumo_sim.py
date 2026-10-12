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
from pursuit_evasion_rl.research.policies.baselines import STAY_ACTION, GoalEvader  # noqa: E402

VISION_M = 150.0
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
                 speed_factor: float | None = None, on_frame=None, background: bool = False) -> None:
        """traffic      background demand period (s per random trip; None = empty roads)
        warmup_s     background traffic runs this long before the pursuit starts
        ignore_red   officers and fugitive run red lights / ignore right of way (emergency driving)
        speed_factor multiple of each road's speed limit the pursuit vehicles drive at, capped at
                     vp / vf (default: unlimited in "ideal", 2.0 in "real")"""
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
        self.evader = GoalEvader(self.network, rng=np.random.default_rng(case.seeds[0]), graph=self.g,
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
        self._target: dict[str, tuple] = {}
        self._decided: dict[str, int] = {}                     # fugitive: edge -> next edge decided
        self._last_edge: str | None = None
        self.teleports = 0
        self._last_goals: list = []
        self._spawn()

    # -- geometry / state ----------------------------------------------------------------------
    def _path_edges(self, src_node: int, dst_node: int) -> list[str]:
        """Shortest directed path as edge ids (via the shared graph)."""
        to = self.g.dist_to(dst_node)
        if src_node not in to:
            return []
        out, cur, guard = [], src_node, 0
        while cur != dst_node and guard < 1000:
            guard += 1
            best = None
            for sid in self.network_out(cur):
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
        key = (goal[0], node, cur)
        if self._target.get(vid) == key:
            return
        start = int(self.seg[_edge_sid(cur)].end_id)
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
        ordered = ordered_outgoing_segment_ids(self.network, node, self.heading_end[sid])
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

    # -- episode -------------------------------------------------------------------------------
    def _spawn(self) -> None:
        c = self.conn
        warm = 0
        while warm * self.dt < self.warmup_s:                    # let background traffic fill the roads
            c.simulationStep(); warm += 1
        self.t0 = c.simulation.getTime()
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
            edges = self._path_edges(n, node) or [f"e{self.network_out(n)[0]}"]
            c.route.add(f"r_{vid}", edges)
            c.vehicle.add(vid, f"r_{vid}", typeID="police", depart="now", departPos="0", departSpeed="0",
                          departLane="best")
            if self.ignore_red:
                c.vehicle.setSpeedMode(vid, 7)                   # no red-light / right-of-way braking
            if edges == [f"e{self.network_out(n)[0]}"] and node == n:
                c.vehicle.setStop(vid, edges[0], pos=1.0, laneIndex=0, duration=HOLD_S)
            else:
                c.vehicle.setStop(vid, edges[-1], pos=max(c.lane.getLength(f"{edges[-1]}_0") - 0.5, 0.1),
                                  laneIndex=0, duration=HOLD_S)
            self._target[vid] = None
        # fugitive: first edge from its start intersection
        pxy = [self.g.pos[n] for n in self.case.police]
        ordered = ordered_outgoing_segment_ids(self.network, fnode, None)
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
                     reach=[self._reach(v) for v in self.police_ids + [self.fid]])
        if self.background:
            mine = set(self.police_ids) | {self.fid}
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
                if want <= set(c.vehicle.getIDList()):
                    break
            else:
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
                    for k, goal in enumerate(goals):
                        self._steer_police(k, goal, fnode)
                    if self.record or self.on_frame:
                        self._emit(t, pxy, fxy, goals)
                self._steer_fugitive(pxy)
                prev = self._current_edge(self.fid)
                c.simulationStep()
                i += 1
                t = i * self.dt
                pxy, fxy = self._positions()
                if any(math.dist(q, fxy) <= self.case.capture_radius_m for q in pxy):
                    outcome = "success"
                    break
                now = self._current_edge(self.fid)
                if prev is not None and self.seg[_edge_sid(prev)].crosses_boundary:
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

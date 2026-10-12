"""Team policies for multi_sim (N police vs M fugitives).

Two-level structure (PEPCRL-MVP): an upper level assigns every officer a
*target* (which fugitive, and pursue it vs guard one of its escape routes);
the lower level drives there along shortest paths. The output of the upper
level is a target node per officer, so the same policies can drive SUMO
vehicles later (TraCI changeTarget / setRoute).

  nearest   every officer pursues its nearest (road time) live fugitive
  balanced  officers split evenly over live fugitives (Hungarian on road time), pursue
  roles     balanced split, then inside each group: the nearest pursues, the
            others hold the interception node of the fugitive's most urgent
            escape route they can still reach first (assign rule per group)
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from obs_ext import boundary_distance, exit_nodes
from pursuit_evasion_rl.research.policies.baselines import STAY_ACTION

DECISION_PERIOD = 6
REVERSE_PENALTY_M = 300.0
EXIT_HORIZON_FACTOR = 4.5
N_ROUTES = 7
_INF = float("inf")


def shortest_path(g, src: int, dst: int) -> list[int]:
    to = g.dist_to(dst)
    if src not in to:
        return []
    path, cur = [src], src
    while cur != dst and len(path) < 500:
        nxt = min(((nb, w + to[nb]) for nb, w in g.fwd[cur] if nb in to), key=lambda x: x[1], default=None)
        if nxt is None:
            break
        path.append(nxt[0]); cur = nxt[0]
    return path


class _Base:
    profile = "multi_base"

    def __init__(self, network) -> None:
        self.network = network
        self.targets: list[int | None] = []
        self.pursue_of: list[int | None] = []     # fugitive index when pursuing, else None
        self.prev: dict[int, int] = {}

    # -- lower level: drive to a node / chase a moving fugitive --------------------
    def _provider(self, sim, k):
        g = sim.g
        target, chase = self.targets[k], self.pursue_of[k]

        def provider(agent_id, node, ordered, hop):
            if not ordered:
                return STAY_ACTION
            if chase is not None and sim.alive[chase]:
                fxy = sim.position(sim.fugs[chase])
                fnode, _ = sim.reach(sim.fugs[chase])
                to = g.dist_to(fnode)
                best, cost = 0, _INF
                for slot, sid in enumerate(ordered):
                    end = g.seg[sid].end_id
                    d = to.get(end)
                    c = (float(g.seg[sid].length_m) + d) if d is not None else 1e6 + math.dist(g.pos[end], fxy)
                    if c < cost:
                        best, cost = slot, c
                return best
            if target is None or node == target:
                return STAY_ACTION
            to = g.dist_to(target)
            best, cost = None, _INF
            for slot, sid in enumerate(ordered):
                end = g.seg[sid].end_id
                d = to.get(end)
                if d is None:
                    continue
                c = float(g.seg[sid].length_m) + d + (REVERSE_PENALTY_M if self.prev.get(k) == end else 0.0)
                if c < cost:
                    best, cost = slot, c
            if best is None:
                txy = g.pos[target]
                best = min(range(len(ordered)), key=lambda s: math.dist(g.pos[g.seg[ordered[s]].end_id], txy))
            if not g.seg[ordered[best]].virtual:
                self.prev[k] = node
            return best

        return provider

    def act(self, sim):
        n = len(sim.police)
        if not self.targets:
            self.targets, self.pursue_of = [None] * n, [None] * n
        if sim.step_count % DECISION_PERIOD == 0 or any(f is not None and not sim.alive[f] for f in self.pursue_of):
            self.decide(sim)
        return [self._provider(sim, k) for k in range(n)]

    # -- helpers ---------------------------------------------------------------------
    @staticmethod
    def road_time(sim):
        """[N, M] officer -> live fugitive road time (s); inf if unreachable."""
        live = sim.alive_fugitives()
        T = np.full((len(sim.police), len(live)), _INF)
        for c, j in enumerate(live):
            fnode, _ = sim.reach(sim.fugs[j])
            to = sim.g.dist_to(fnode)
            for k, v in enumerate(sim.police):
                n, off = sim.reach(v)
                if n in to:
                    T[k, c] = (off + to[n]) / sim.case.vp
        return live, T

    @staticmethod
    def balanced_groups(live, T):
        """Hungarian assignment with ceil(N/M) slots per live fugitive -> fugitive index per officer."""
        n, m = T.shape
        if m == 0:
            return [None] * n
        slots = math.ceil(n / m)
        cost = np.repeat(np.where(np.isfinite(T), T, 1e6), slots, axis=1)
        rows, cols = linear_sum_assignment(cost)
        out = [None] * n
        for r, c in zip(rows, cols):
            out[r] = live[c // slots]
        return out


class NearestPursuit(_Base):
    profile = "multi_nearest"

    def decide(self, sim):
        live, T = self.road_time(sim)
        for k in range(len(sim.police)):
            self.pursue_of[k] = live[int(np.argmin(T[k]))] if live else None
            self.targets[k] = None


class BalancedPursuit(_Base):
    profile = "multi_balanced"

    def decide(self, sim):
        live, T = self.road_time(sim)
        groups = self.balanced_groups(live, T)
        for k in range(len(sim.police)):
            self.pursue_of[k], self.targets[k] = groups[k], None


class GroupRoles(_Base):
    """Balanced groups + per-group pursue / guard-route roles."""
    profile = "multi_roles"

    def __init__(self, network) -> None:
        super().__init__(network)
        self.exits = exit_nodes(network)
        self.bd = boundary_distance(network)

    def routes(self, sim, j):
        """Escape routes of fugitive j: (fugitive arrival time, path), most urgent first, de-duplicated."""
        g = sim.g
        fnode, foff = sim.reach(sim.fugs[j])
        f_from = g.dist_from(fnode)
        S = max(foff + self.bd.get(fnode, 300.0), 150.0)
        cands = sorted(((foff + f_from[e]) / sim.case.vf, e) for e in self.exits
                       if e in f_from and foff + f_from[e] <= EXIT_HORIZON_FACTOR * S)
        out, seen = [], set()
        for tf, e in cands:
            path = shortest_path(g, fnode, e)
            key = tuple(path[1:3])
            if key in seen and len(path) > 2:
                continue
            seen.add(key)
            out.append((tf, path, f_from, foff))
            if len(out) >= N_ROUTES:
                break
        return out

    @staticmethod
    def intercept(sim, k, path, f_from, foff):
        """First node on the route (from the fugitive) officer k reaches >= 1 s earlier; (node, margin)."""
        g = sim.g
        n0, off = sim.reach(sim.police[k])
        fr = g.dist_from(n0)
        for node in path:
            if node in fr:
                tf = (foff + f_from.get(node, _INF)) / sim.case.vf
                tme = (off + fr[node]) / sim.case.vp
                if tme <= tf - 1.0:
                    return node, tf - tme
        return None, -_INF

    def decide(self, sim):
        live, T = self.road_time(sim)
        groups = self.balanced_groups(live, T)
        for j in live:
            members = [k for k in range(len(sim.police)) if groups[k] == j]
            if not members:
                continue
            col = live.index(j)
            members.sort(key=lambda k: T[k, col])
            chaser = members[0]
            self.pursue_of[chaser], self.targets[chaser] = j, None
            held = set()
            routes = self.routes(sim, j)
            for k in members[1:]:
                choice = None
                for r, (tf, path, f_from, foff) in enumerate(routes):
                    if r in held:
                        continue
                    node, margin = self.intercept(sim, k, path, f_from, foff)
                    if node is not None:
                        choice = (r, node)
                        break
                if choice is None:
                    self.pursue_of[k], self.targets[k] = j, None
                else:
                    held.add(choice[0])
                    self.pursue_of[k], self.targets[k] = None, choice[1]


POLICIES = {"nearest": NearestPursuit, "balanced": BalancedPursuit, "roles": GroupRoles}

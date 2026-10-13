"""Learned barricade placement: each officer carries one barricade; a small MLP decides where to put it.

Shared by the point-mass simulator (multi_sim, used for GA training) and SUMO (sumo_sim), so both
make the same decision from the same features.

When  once per second, for every officer still carrying its barricade that is within REACH_M of
      its next intersection u, and reaches u before the fugitive could (u is outside the
      fugitive's escape region). Officers are tried nearest-first; at most one placement per second.
What  every road at u that no chase vehicle is on is a candidate (closing it closes both
      directions, OFFSET_M in from u). Each candidate is scored by the head; the best one is
      placed when its score is above 0 ("do nothing" scores 0).
Features (one row per candidate road, all roughly in [0, 1]) -- see ``FEATURES``.

The escape region is the set of intersections the fugitive reaches strictly before every officer
(multi_sim.escape_region) on the officers' map with closed roads removed; an exit is open when it
is in the region and still has an open boundary-crossing road.
"""
from __future__ import annotations

import heapq
import math
import os

import numpy as np
import torch
from torch import nn

REACH_M = 30.0
REEVAL_S = 5                         # an officer staying at the same intersection is re-judged this often
OFFSET_M = 20.0
SIGHT_M = 80.0                       # the fugitive notices a barricade this close
MARGIN_S = 25.0 / 16.0               # capture radius at officer speed, as escape_region
# only roads that shrink the escape region or shut an exit (now or once the officer leaves) are
# candidates; v1 (off) placed many no-effect barricades on unseen maps (seoul 2.7/ep, SR -8 pt)
IMPACT_ONLY = os.environ.get("BG_IMPACT_ONLY", "1") == "1"
FEATURES = (
    "open_cut_stay",      # share of open exits the road shuts (officer stays)
    "region_cut_stay",    # share of the escape region it removes (officer stays)
    "open_cut_leave",     # same, judged as if the officer drove on (0 when not holding u)
    "region_cut_leave",
    "open_exits",         # open exits / 20
    "region_size",        # escape region / 500 intersections
    "pursuing",           # officer's role is pursuit
    "holding_here",       # officer holds u
    "dist_to_fugitive",   # officer's road distance to the fugitive / 1500 m
    "fugitive_time_u",    # fugitive's fastest time to u / 60 s
    "kits_left",          # share of the team's barricades not yet placed
    "time",               # step / max_steps
    "road_length",        # length of the road / 300 m
    "towards_fugitive",   # the road's far end is nearer the fugitive than u
    "team_size",          # 6 / N
)
N_FEAT = len(FEATURES)


class BarricadeHead(nn.Module):
    def __init__(self, hidden=(32, 32)) -> None:
        super().__init__()
        layers, prev = [], N_FEAT
        for w in hidden:
            layers += [nn.Linear(prev, w), nn.Tanh()]
            prev = w
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return self.net(rows).squeeze(-1)


def polyline_length(pts) -> float:
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))


def road_segments(network, u: int, v: int) -> list:
    return [s for s in network.segments if {s.start_id, s.end_id} == {u, v} and s.start_id != s.end_id]


def placement(network, u: int, v: int, offset_m: float = OFFSET_M):
    """(xy, {segment id: fraction along it}) of a barricade on road u-v, offset_m in from u."""
    segs = road_segments(network, u, v)
    s0 = next((s for s in segs if s.start_id == u), segs[0])
    pts = s0.geometry_xy if s0.start_id == u else s0.geometry_xy[::-1]
    length = polyline_length(pts)
    d = min(offset_m, 0.4 * length)
    xy, run = pts[-1], 0.0
    for a, b in zip(pts, pts[1:]):
        step = math.dist(a, b)
        if run + step >= d:
            w = (d - run) / max(step, 1e-9)
            xy = (a[0] + w * (b[0] - a[0]), a[1] + w * (b[1] - a[1]))
            break
        run += step
    f = d / max(length, 1e-6)
    return (float(xy[0]), float(xy[1])), {int(s.id): (f if s.start_id == u else 1.0 - f) for s in segs}


class Planner:
    """Decides barricade placements for one episode."""

    def __init__(self, network, head: BarricadeHead | None, *, vp: float, vf: float, exits) -> None:
        self.network, self.head, self.vp, self.vf = network, head, vp, vf
        self.seg = {s.id: s for s in network.segments}
        self.out: dict[int, list[int]] = {}
        for s in network.segments:
            self.out.setdefault(s.start_id, []).append(s.id)
        self.exit_roads = {e: [o for o in self.out.get(e, []) if self.seg[o].crosses_boundary] for e in exits}
        self.decisions = 0
        self._judged: dict[tuple[int, int], int] = {}             # (officer, node) -> step last judged

    # -- escape region ---------------------------------------------------------------------
    def _escape(self, g, tp, freach, closed, skip) -> tuple[int, int]:
        f0, foff = freach
        vf = self.vf
        best, heap, region = {f0: foff / vf}, [(foff / vf, f0)], set()
        while heap:
            t, u = heapq.heappop(heap)
            if t > best.get(u, math.inf) or t >= tp.get(u, math.inf) - MARGIN_S:
                continue
            region.add(u)
            for nb, w in g.fwd[u]:
                if (u, nb) in skip:
                    continue
                nt = t + w / vf
                if nt < best.get(nb, math.inf):
                    best[nb] = nt
                    heapq.heappush(heap, (nt, nb))
        n_open = sum(1 for e, roads in self.exit_roads.items() if e in region and any(
            o not in closed and (e, self.seg[o].end_id) not in skip for o in roads))
        return len(region), n_open

    # -- decision ----------------------------------------------------------------------------
    def decide(self, g, closed: set, reaches, freach, busy: set, *, kit, pursuing, holding,
               step: int, max_steps: int):
        """-> (officer, u, v) to barricade road u-v, or None.
        g        officers' graph (closed roads removed);  reaches  [(next node, metres)] per officer
        freach   the fugitive's (next node, metres);       busy     node pairs a chase vehicle is on
        kit / pursuing / holding  per officer (holding = node it holds, or None)"""
        if self.head is None or not any(kit):
            return None
        n = len(reaches)
        eligible = [k for k in range(n) if kit[k] and reaches[k][1] <= REACH_M
                    and step - self._judged.get((k, reaches[k][0]), -REEVAL_S) >= REEVAL_S]
        if not eligible:
            return None
        f_from = g.dist_from(freach[0])

        def t_f(node):
            d = f_from.get(node)
            return math.inf if d is None else (freach[1] + d) / self.vf

        arrive = [{node: (off + d) / self.vp for node, d in g.dist_from(r).items()} for r, off in reaches]

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
        eligible = [k for k in eligible if tp_all.get(reaches[k][0], math.inf) < t_f(reaches[k][0]) - MARGIN_S]
        if not eligible:
            return None
        base_n, base_open = self._escape(g, tp_all, freach, closed, set())
        kits_left = sum(kit) / n
        best = None
        for k in sorted(eligible, key=lambda k: arrive[k].get(freach[0], math.inf)):
            u = reaches[k][0]
            self._judged[(k, u)] = step
            hold = holding[k] == u
            if hold:
                tp_k = tp_without(k)
                n0, o0 = self._escape(g, tp_k, freach, closed, set())
            nbrs = sorted({nb for nb, _ in g.fwd[u]} | {nb for nb, _ in g.rev[u]})
            rows, roads = [], []
            for v in nbrs:
                if v == u or (u, v) in busy:
                    continue
                skip = {(u, v), (v, u)}
                n1, o1 = self._escape(g, tp_all, freach, closed, skip)
                if hold:
                    n2, o2 = self._escape(g, tp_k, freach, closed, skip)
                    leave = ((o0 - o2) / max(o0, 1), (n0 - n2) / max(n0, 1))
                else:
                    leave = (0.0, 0.0)
                if IMPACT_ONLY and o1 >= base_open and n1 >= base_n and leave[0] <= 0 and leave[1] <= 0:
                    continue                                   # shuts nothing: would only block the police
                segs = [s for s in self.out.get(u, []) if self.seg[s].end_id == v] or \
                       [s for s in self.out.get(v, []) if self.seg[s].end_id == u]
                length = float(self.seg[segs[0]].length_m) if segs else 100.0
                rows.append([
                    (base_open - o1) / max(base_open, 1), (base_n - n1) / max(base_n, 1), leave[0], leave[1],
                    min(base_open / 20.0, 1.0), min(base_n / 500.0, 1.0), float(pursuing[k]), float(hold),
                    min(arrive[k].get(freach[0], math.inf) * self.vp / 1500.0, 1.0), min(t_f(u) / 60.0, 1.0),
                    kits_left, min(step / max(max_steps, 1), 1.0), min(length / 300.0, 1.0),
                    float(t_f(v) < t_f(u)), 6.0 / n,
                ])
                roads.append(v)
            if not rows:
                continue
            self.decisions += 1
            with torch.no_grad():
                scores = self.head(torch.tensor(np.nan_to_num(np.array(rows, dtype=np.float32), posinf=1.0)))
            j = int(torch.argmax(scores))
            s = float(scores[j])
            if s > 0 and (best is None or s > best[0]):
                best = (s, k, u, roads[j])
        return None if best is None else best[1:]


def load_head(payload: dict) -> BarricadeHead | None:
    if "barricade" not in payload:
        return None
    head = BarricadeHead(tuple(payload.get("barricade_hidden", (32, 32))))
    head.load_state_dict(payload["barricade"])
    return head


def close(g, sids) -> None:
    """Remove directed segments from a private graph (and drop its distance caches)."""
    for sid in sids:
        s = g.seg[sid]
        g.fwd[s.start_id].remove((s.end_id, float(s.length_m)))
        g.rev[s.end_id].remove((s.start_id, float(s.length_m)))
    g._to.clear()
    g._from.clear()


def private_graph(network, closed) -> "object":
    from pursuit_evasion_rl.research.policies.baselines import _Graph
    g = _Graph(network)
    close(g, closed)
    return g

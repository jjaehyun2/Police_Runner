"""Target-selection (macro-action) team policy.

Instead of choosing the next road segment at every intersection, each
officer chooses a *role* every DECISION_PERIOD seconds:

  candidate 0      pursue the fugitive directly
  candidates 1..7  block escape route j: the fugitive's shortest road route to
                   boundary exit j (open exits first, by fugitive arrival
                   time). The officer heads for its own interception node on
                   that route -- the first node, counted from the fugitive,
                   that it reaches before the fugitive does -- and holds it.

Movement between decisions is plain shortest-path routing, so a chosen role
is carried out consistently instead of being re-sampled at every
intersection. Officers decide in order of distance to the fugitive and see
the roles teammates already hold (radio coordination).

The policy network is a parameter-shared scorer: one MLP scores every
(officer, candidate) feature row, plus the officer's one-hot id; a masked
softmax over the scores is the role distribution. It implements the
``recommend`` team interface, so it is evaluated by the official
``replay_episode`` loop like every baseline.
"""
from __future__ import annotations

import math
import os
from typing import Sequence

import numpy as np
import torch
from torch import nn

from pursuit_evasion_rl.osm_demo.metrics import placement_position
from pursuit_evasion_rl.osm_demo.policies import build_action_mask, decode_recommendation
from pursuit_evasion_rl.research.policies.baselines import ACTION_DIM, STAY_ACTION

from obs_ext import EXIT_HORIZON_M, boundary_distance, exit_nodes, graph_for

POLICE_COUNT = 6
N_CANDIDATES = 8
ROW_DIM = 16
DECISION_PERIOD = 6
CLIP_M = 2000.0
REVERSE_PENALTY_M = 300.0
_INF = float("inf")
# stall breaker: a fugitive that has stayed within STALL_M of one spot for stall_s seconds while every
# officer holds an interception node is waiting out the cordon; the road-nearest officer closes in
STALL_M = 5.0
# MACRO_MIN_PURSUE=1: always keep one direct pursuer (encirclement-style pressure), whatever the scorer picks
MIN_PURSUE = os.environ.get("MACRO_MIN_PURSUE", "0") == "1"


class CandidateScorer(nn.Module):
    def __init__(self, hidden=(128, 128), kind: str = "onehot") -> None:
        super().__init__()
        # kind "onehot": + officer one-hot (6-officer teams); "size": + team-size feature 6/N
        # (officer-agnostic, any team size)
        self.kind = kind
        layers, prev = [], ROW_DIM + (POLICE_COUNT if kind == "onehot" else 1)
        for w in hidden:
            layers += [nn.Linear(prev, w), nn.Tanh()]
            prev = w
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, rows: torch.Tensor, officer_ids: torch.Tensor) -> torch.Tensor:
        """rows [B, N, ROW_DIM], officer_ids [B] (onehot) or team-size features [B, 1] (size) -> logits [B, N]."""
        if self.kind == "onehot":
            extra = nn.functional.one_hot(officer_ids % POLICE_COUNT, POLICE_COUNT).float()
        else:
            extra = officer_ids.float().reshape(-1, 1)
        extra = extra[:, None, :].expand(-1, rows.shape[1], -1)
        return self.net(torch.cat([rows, extra], dim=-1)).squeeze(-1)


def _reach(g, pl) -> tuple[int, float]:
    if pl.segment_id is not None:
        seg = g.seg[pl.segment_id]
        return int(seg.end_id), (1.0 - pl.progress) * float(seg.length_m)
    return int(pl.intersection_id), 0.0


def _path(g, src: int, dst: int) -> list[int]:
    """Node sequence of a shortest directed path src -> dst (via dist_to(dst))."""
    to_dst = g.dist_to(dst)
    if src not in to_dst:
        return []
    path, cur, guard = [src], src, 0
    while cur != dst and guard < 500:
        guard += 1
        best, best_d = None, _INF
        for nb, w in g.fwd[cur]:
            d = to_dst.get(nb)
            if d is not None and w + d < best_d - 1e-9:
                best_d, best = w + d, nb
        if best is None:
            break
        path.append(best)
        cur = best
    return path


class _Euclid:
    """Straight-line distance from xy to every node the road graph reaches (keys of ``road``); the
    distance metric of PEPCRL-MVP (env/utils.calculate_dis), for the metric ablation."""

    def __init__(self, g, xy, road) -> None:
        self.g, self.xy, self.road = g, xy, road

    def __contains__(self, n) -> bool:
        return n in self.road

    def __getitem__(self, n) -> float:
        if n not in self.road:
            raise KeyError(n)
        return math.dist(self.xy, self.g.pos[n])

    def get(self, n, default=None):
        return self[n] if n in self.road else default


class MacroTeamPolicy:
    profile = "macro_role_policy"
    experimental = False

    def __init__(self, network, scorer: CandidateScorer | None, *, vp: float, vf: float,
                 mode: str = "argmax", rng: np.random.Generator | None = None,
                 fixed_rule: str | None = None, relative: bool = False, metric: str = "road",
                 stall_s: float | None = None) -> None:
        self.network = network
        self.g = graph_for(network)
        self.boundary = boundary_distance(network)
        self.exits = exit_nodes(network)
        self.scorer = scorer
        self.vp, self.vf = vp, vf
        self.mode = mode
        self.rng = rng or np.random.default_rng(0)
        self.fixed_rule = fixed_rule
        # feature scales. absolute (default, original): metres / seconds fixed for Daejeon-sized maps.
        # relative: everything is measured against S = the fugitive's road distance to its nearest
        # exit, calibrated so a typical Daejeon case (S ~ 335 m, vf 16) reproduces the absolute scales.
        self.relative = relative
        # distance metric for every timing estimate (arrival times, margins, ranks); movement always
        # follows the roads. "euclid": straight-line distance, as PEPCRL-MVP (metric ablation)
        # "travel": fastest driving time with speed limits and turn losses (roadtime.py), as equivalent metres
        # "traffic": v2, travel time through the live background traffic (roadtime.TrafficGraph; officers and
        # fugitive each with their own signal compliance), fed by the simulator via set_traffic
        assert metric in ("road", "euclid", "travel", "traffic"), metric
        self.metric = metric
        self.gf = self.g                         # graph for the fugitive's routes and times (its own top speed)
        if metric == "travel":
            import roadtime
            exits = [s.start_id for s in network.segments if s.crosses_boundary]
            self.g, self.gf = roadtime.travel_graph(network, vp), roadtime.travel_graph(network, vf)
            self.boundary = roadtime.boundary_time(network, vf, exits)
        self.wants_traffic = metric == "traffic"
        if metric == "traffic":
            import roadtime
            self.g = roadtime.TrafficGraph(network, vp)
            self.gf = roadtime.TrafficGraph(network, vf)
            self._road_info = None
        self._D, self._T, self._M, self._H = CLIP_M, 60.0, 30.0, EXIT_HORIZON_M
        self.n = 0                               # team size, set from the first observation
        self._size(POLICE_COUNT)
        self._prev: dict[int, int] = {}
        self.closed: set[int] = set()            # barricaded segments (set by the simulator)
        self._fug_hist: list[tuple[float, float]] = []
        self.stall_s = float(os.environ.get("MACRO_STALL_S", "0")) if stall_s is None else stall_s  # 0 = off
        self._still: tuple | None = None         # (anchor xy, step it settled there)
        self.log: list[dict] = []                # decisions (for training)

    def _size(self, n: int) -> None:
        if n != self.n:
            self.n = n
            self.roles = [0] * n                 # candidate index currently held
            self.targets: list[int | None] = [None] * n
            self.role_exit: list[int | None] = [None] * n

    # -- candidate construction ------------------------------------------------
    def _candidates(self, police, fugitive):
        g = self.g
        freach, foff = self._reach(fugitive, self.gf)
        f_from = self.gf.dist_from(freach)
        if self.metric == "euclid":
            f_from, foff = _Euclid(g, placement_position(self.network, fugitive), f_from), 0.0
        if self.relative:
            if self.metric in ("euclid", "traffic"):
                S = max(foff + min((f_from[e] for e in self.exits if e in f_from), default=300.0), 150.0)
            else:
                S = max(foff + self.boundary.get(freach, 300.0), 150.0)
            self._D, self._T, self._M, self._H = 6.0 * S, 3.0 * S / self.vf, 1.5 * S / self.vf, 4.5 * S
        officer = [self._reach(p) for p in police]
        officer_from = [g.dist_from(n) for n, _ in officer]
        if self.metric == "euclid":
            officer_from = [_Euclid(g, placement_position(self.network, p), d) for p, d in zip(police, officer_from)]
            officer = [(n, 0.0) for n, _ in officer]
        rows = []
        for e in self.exits:
            if self.closed and all(o in self.closed for o in self.network_out_crossing(e)):
                continue                                  # sealed by a barricade
            df = f_from.get(e)
            if df is None or foff + df > self._H:
                continue
            tf_exit = (foff + df) / self.vf
            tp = min(((off + officer_from[k][e]) / self.vp for k, (n, off) in enumerate(officer) if e in officer_from[k]),
                     default=_INF)
            rows.append((tf_exit >= tp, tf_exit, e))   # (closed?, fugitive time, exit)
        rows.sort()
        # dedupe exits whose routes share the same first 2 nodes (same corridor)
        chosen, seen = [], set()
        for closed, tf_exit, e in rows:
            path = _path(self.gf, freach, e)
            key = tuple(path[1:3])
            if key in seen and len(path) > 2:
                continue
            seen.add(key)
            chosen.append((not closed, tf_exit, e, path))
            if len(chosen) >= N_CANDIDATES - 1:
                break
        return freach, foff, f_from, officer, officer_from, chosen

    def _reach(self, pl, g=None) -> tuple[int, float]:
        g = g or self.g
        n, off = _reach(g, pl)
        if self.metric in ("travel", "traffic") and pl.segment_id is not None:
            seg = g.seg[pl.segment_id]
            off *= g.cost[pl.segment_id] / max(float(seg.length_m), 1e-6)
        return n, off

    def set_traffic(self, snap: dict, *, ignore_red: tuple[bool, bool], road_info) -> None:
        """Live background traffic per segment (vehicles, halting, mean speed) and the static lanes /
        signals of the map; called by the simulator before each recommend()."""
        if self._road_info is None:
            self._road_info = road_info
            self.g.lanes, self.g.signals = road_info
            self.gf.lanes, self.gf.signals = road_info
        self.g.set_traffic(snap, obeys_red=not ignore_red[0])
        self.gf.set_traffic(snap, obeys_red=not ignore_red[1])

    def network_out_crossing(self, e: int) -> list[int]:
        if not hasattr(self, "_crossing"):
            self._crossing: dict[int, list[int]] = {}
            for s in self.network.segments:
                if s.crosses_boundary:
                    self._crossing.setdefault(s.start_id, []).append(s.id)
        return self._crossing.get(e, [])

    def _intercept(self, k, path, f_from, foff, officer, officer_from) -> tuple[int, float, float]:
        """(node, my margin s, my travel time s) on a route for officer k."""
        n0, off = officer[k]
        best = None
        for node in path:
            d = officer_from[k].get(node)
            if d is None:
                continue
            tf = (foff + f_from.get(node, _INF)) / self.vf
            tme = (off + d) / self.vp
            if tme <= tf - 1.0:
                return node, tf - tme, tme
            if best is None or tf - tme > best[1]:
                best = (node, tf - tme, tme)
        if best is None:
            return path[-1], -60.0, 60.0
        return best

    def _heading(self, fxy):
        if not self._fug_hist or self._fug_hist[-1] != fxy:
            self._fug_hist.append(fxy)
            self._fug_hist = self._fug_hist[-6:]
        if len(self._fug_hist) < 2:
            return None
        (ax, ay), (bx, by) = self._fug_hist[0], self._fug_hist[-1]
        n = math.hypot(bx - ax, by - ay)
        return None if n < 1e-6 else ((bx - ax) / n, (by - ay) / n)

    def _rows(self, k, cands, heading, fxy, road, rank, step, max_steps, n_open):
        freach, foff, f_from, officer, officer_from, chosen = cands
        g = self.g
        rows = np.zeros((N_CANDIDATES, ROW_DIM), dtype=np.float32)
        mask = np.zeros(N_CANDIDATES, dtype=bool)
        intercepts: list[int | None] = [None] * N_CANDIDATES
        holders = [sum(1 for j in range(self.n) if j != k and self.roles[j] == c and self.targets[j] is not None)
                   for c in range(N_CANDIDATES)]
        glob = [
            min(road[k] / self._D, 1.0), rank[k] / max(self.n - 1, 1), min(n_open / 10.0, 1.0),
            min(self.vf / self.vp, 1.5) / 1.5, min(step / max_steps, 1.0),
            min((foff + self.boundary.get(freach, _INF)) / self._D, 1.0) if not self.relative
            else min(min(road) / self._D, 1.0),
        ]
        # pursuit row
        nearest_other = min((road[j] for j in range(self.n) if j != k), default=_INF)
        rows[0, :10] = [1.0, min(road[k] / self._D, 1.0), 0.0,
                        min(max((nearest_other - road[k]) / self._D + 0.5, 0.0), 1.0), min(road[k] / self.vp / self._T, 1.0),
                        0.5, min(holders[0] / 3.0, 1.0), float(self.roles[k] == 0), 0.5, 0.0]
        rows[0, 10:] = glob
        mask[0] = True
        intercepts[0] = freach
        for c, (is_open, tf_exit, e, path) in enumerate(chosen, start=1):
            if not path:
                continue
            node, margin, tme = self._intercept(k, path, f_from, foff, officer, officer_from)
            mate = max((self._intercept(j, path, f_from, foff, officer, officer_from)[1]
                        for j in range(self.n) if j != k), default=-60.0)
            ex, ey = g.pos[e]
            if heading is None:
                align = 0.5
            else:
                vx, vy = ex - fxy[0], ey - fxy[1]
                nrm = math.hypot(vx, vy) or 1.0
                align = ((vx * heading[0] + vy * heading[1]) / nrm + 1) / 2
            rows[c, :10] = [0.0, min(tf_exit / self._T, 1.0), float(is_open),
                            min(max(margin / self._M + 0.5, 0.0), 1.0), min(tme / self._T, 1.0),
                            min(max(mate / self._M + 0.5, 0.0), 1.0), min(holders[c] / 3.0, 1.0),
                            float(self.roles[k] == c and self.role_exit[k] == e), align,
                            min((foff + f_from.get(node, _INF)) / self._D, 1.0)]
            rows[c, 10:] = glob
            mask[c] = True
            intercepts[c] = node
        # unreachable officers produce inf - inf = nan in a few differences; treat as "far"
        rows = np.nan_to_num(rows, nan=1.0, posinf=1.0, neginf=0.0)
        return rows, mask, intercepts

    # -- choosing ----------------------------------------------------------------
    def _choose(self, k, rows, mask) -> tuple[int, float]:
        if self.fixed_rule == "pursue":
            return 0, 0.0
        if self.fixed_rule == "assign":
            # nearest officer pursues; every other officer guards the unheld route it can still
            # reach ahead of the fugitive, most urgent (soonest fugitive arrival) first
            if rows[0, 1] * self._D <= min(self._round_road) + 1e-6:
                return 0, 0.0
            best, best_t = 0, 2.0
            for c in range(1, N_CANDIDATES):
                if mask[c] and rows[c, 3] > 0.5 and rows[c, 6] == 0.0 and rows[c, 1] < best_t:
                    best, best_t = c, rows[c, 1]
            return best, 0.0
        with torch.no_grad():
            ids = torch.tensor([k]) if self.scorer.kind == "onehot" else torch.tensor([[POLICE_COUNT / self.n]])
            logits = self.scorer(torch.from_numpy(rows)[None], ids)[0].numpy()
        z = np.where(mask, logits, -np.inf)
        z = z - z.max()
        p = np.exp(z)
        p /= p.sum()
        a = int(np.argmax(p)) if self.mode == "argmax" else int(self.rng.choice(N_CANDIDATES, p=p))
        return a, float(np.log(max(p[a], 1e-12)))

    def _route(self, ordered, target: int, k: int) -> int:
        dist_to = self.g.dist_to(target)
        prev = self._prev.get(k)
        best, best_cost = None, _INF
        for slot, seg_id in enumerate(ordered):
            if seg_id in self.closed:
                continue
            end = self.g.seg[seg_id].end_id
            dn = dist_to.get(end)
            if dn is None:
                continue
            cost = float(self.g.seg[seg_id].length_m) + dn + (REVERSE_PENALTY_M if prev is not None and end == prev else 0.0)
            if cost < best_cost:
                best_cost, best = cost, slot
        if best is None:
            txy = self.g.pos[target]
            best = min(range(len(ordered)), key=lambda s: (ordered[s] in self.closed,
                                                           math.dist(self.g.pos[self.g.seg[ordered[s]].end_id], txy)))
        return best

    def _pursue_slot(self, ordered, fxy) -> int:
        best, best_cost = 0, _INF
        for slot, seg_id in enumerate(ordered):
            if seg_id in self.closed:
                continue
            cost = math.dist(self.g.pos[self.g.seg[seg_id].end_id], fxy)
            if cost < best_cost:
                best_cost, best = cost, slot
        return best

    def _round_setup(self, police, fugitive):
        cands = self._candidates(police, fugitive)
        to_f = self.g.dist_to(cands[0])
        road = []
        for k, (n, off) in enumerate(cands[3]):
            d = to_f.get(n)
            if self.metric == "euclid":
                d = None if d is None else math.dist(placement_position(self.network, police[k]),
                                                     placement_position(self.network, fugitive))
                off = 0.0
            road.append(_INF if d is None else off + d)
        order = sorted(range(self.n), key=lambda k: road[k])
        rank = {k: r for r, k in enumerate(order)}
        n_open = sum(1 for c in cands[5] if c[0])
        self._round_road = road
        return cands, road, order, rank, n_open

    def _decision_round(self, police, fugitive, step, max_steps, heading, fxy):
        cands, road, order, rank, n_open = self._round_setup(police, fugitive)
        for k in order:
            rows, mask, intercepts = self._rows(k, cands, heading, fxy, road, rank, step, max_steps, n_open)
            a, logp = self._choose(k, rows, mask)
            self.log.append(dict(officer=k, rows=rows, mask=mask, action=a, logp=logp, step=step))
            self.roles[k] = a
            self.targets[k] = intercepts[a]
            self.role_exit[k] = cands[5][a - 1][2] if a > 0 else None
        if (self._stalled(fxy, step) or MIN_PURSUE) and all(r != 0 for r in self.roles):
            k = order[0]
            self.roles[k], self.targets[k], self.role_exit[k] = 0, intercepts[0], None

    def _stalled(self, fxy, step) -> bool:
        if not self.stall_s:
            return False
        if self._still is None or step < self._still[1] or math.dist(fxy, self._still[0]) > STALL_M:
            self._still = (fxy, step)
        return step - self._still[1] >= self.stall_s

    def recommend(self, *, police, fugitive, step, max_steps, incoming_headings=None):
        g = self.g
        self._size(len(police))
        fxy = placement_position(self.network, fugitive)
        heading = self._heading(fxy)
        if step % DECISION_PERIOD == 0:
            self._decision_round(police, fugitive, step, max_steps, heading, fxy)
        recs = []
        for k in range(self.n):
            heading_k = None if incoming_headings is None else incoming_headings[k]
            did, _ = _reach(g, police[k])
            mask, ordered = build_action_mask(self.network, did, heading_k)
            if not ordered:
                chosen = STAY_ACTION
            elif self.roles[k] == 0 or self.targets[k] is None:
                chosen = self._pursue_slot(ordered, fxy)
            elif did == self.targets[k]:
                chosen = STAY_ACTION                      # hold the interception node
            else:
                chosen = self._route(ordered, self.targets[k], k)
            if chosen != STAY_ACTION and ordered and not g.seg[ordered[chosen]].virtual:
                self._prev[k] = did
            probs = np.zeros(ACTION_DIM)
            probs[chosen] = 1.0
            recs.append(decode_recommendation(
                agent_id=f"police_{k}", network=self.network, decision_intersection_id=did, mask=mask,
                ordered_segment_ids=ordered, probabilities=probs, profile=self.profile,
                compatibility_report_id=self.profile, action_index=chosen,
            ))
        return tuple(recs)

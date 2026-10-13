"""Rule-based team policies used as teachers / comparison points.

  enc:key=val,...   EncirclementPolice with overridden EncirclementParameters
                    (min, max = cordon ring band [m]; dp = direct pursuers;
                     cp = close-pursuit radius [m]; deg = cordon degrees a/b/c)
  block:...         boundary-aware interception (see BoundaryBlockPolice)
"""
from __future__ import annotations

from dataclasses import replace
import math
import os

import numpy as np

from pursuit_evasion_rl.osm_demo.metrics import placement_position
from pursuit_evasion_rl.osm_demo.policies import build_action_mask, decode_recommendation
from pursuit_evasion_rl.research.policies.baselines import ACTION_DIM, EncirclementPolice, STAY_ACTION, _Graph

_INF = float("inf")


def _speeds():
    return float(os.environ.get("BG_POLICE_SPEED", "16")), float(os.environ.get("BG_FUGITIVE_SPEED", "9"))


def _parse(spec: str) -> dict[str, str]:
    out = {}
    for part in spec.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


def build(spec: str, network, vp: float | None = None, vf: float | None = None):
    if vp is None or vf is None:
        vp, vf = _speeds()
    kind, _, rest = spec.partition(":")
    args = _parse(rest)
    if kind == "enc":
        police = EncirclementPolice(network, graph=_Graph(network), fugitive_speed_mps=vf, police_speed_mps=vp)
        overrides = {}
        if "min" in args: overrides["cordon_ring_min_m"] = float(args["min"])
        if "max" in args: overrides["cordon_ring_max_m"] = float(args["max"])
        if "dp" in args: overrides["direct_pursuit_count"] = int(args["dp"])
        if "cp" in args: overrides["close_pursuit_m"] = float(args["cp"])
        if "deg" in args: overrides["cordon_degrees"] = tuple(float(x) for x in args["deg"].split("/"))
        police.parameters = replace(police.parameters, **overrides)
        return police
    if kind == "block":
        return BoundaryBlockPolice(network, vp=vp, vf=vf, **{k: float(v) for k, v in args.items()})
    raise ValueError(spec)


class BoundaryBlockPolice:
    """Arrival-time interception of the fugitive's escape routes.

    Every step: the closest ``chasers`` officers pursue the fugitive directly.
    Each remaining officer is assigned (greedily, nearest-first, no sharing) to
    the intersection on the fugitive's reachable region where it arrives
    first with the biggest margin, preferring intersections that lie on the
    fugitive's short routes to boundary exits ("escape funnels"). If no
    intersection can be reached first, it pursues directly.
    """

    profile = "rule_boundary_block"
    experimental = False

    def __init__(self, network, *, vp: float, vf: float, chasers: float = 2, horizon: float = 900.0,
                 funnel_weight: float = 1.0, reverse_penalty: float = 300.0) -> None:
        self.network = network
        self.g = _Graph(network)
        self.vp, self.vf = vp, vf
        self.chasers = int(chasers)
        self.horizon = horizon
        self.funnel_weight = funnel_weight
        self.reverse_penalty = reverse_penalty
        self._prev: dict[int, int | None] = {}
        exits = {s.start_id for s in network.segments if s.crosses_boundary}
        self.to_exit: dict[int, float] = {}
        for e in exits:
            for n, d in self.g.dist_to(e).items():
                self.to_exit[n] = min(self.to_exit.get(n, _INF), d)

    def _decision(self, pl) -> tuple[int, float]:
        if pl.segment_id is not None:
            seg = self.g.seg[pl.segment_id]
            return int(seg.end_id), (1 - pl.progress) * float(seg.length_m)
        return int(pl.intersection_id), 0.0

    def _route(self, ordered, target, k) -> int:
        dist_to = self.g.dist_to(target)
        prev = self._prev.get(k)
        best, best_cost = None, _INF
        for slot, seg_id in enumerate(ordered):
            seg = self.g.seg[seg_id]
            dn = dist_to.get(seg.end_id)
            if dn is None:
                continue
            cost = float(seg.length_m) + dn + (self.reverse_penalty if prev is not None and seg.end_id == prev else 0.0)
            if cost < best_cost:
                best_cost, best = cost, slot
        if best is None:
            txy = self.g.pos[target]
            best = min(range(len(ordered)), key=lambda s: math.dist(self.g.pos[self.g.seg[ordered[s]].end_id], txy))
        return best

    def recommend(self, *, police, fugitive, step, max_steps, incoming_headings=None):
        fnode, foff = self._decision(fugitive)
        f_from = self.g.dist_from(fnode)
        fxy = placement_position(self.network, fugitive)
        dec = [self._decision(p) for p in police]
        road = []
        for (n, off) in dec:
            d = self.g.dist_from(n).get(fnode)
            road.append(_INF if d is None else off + d)
        order = sorted(range(len(police)), key=lambda k: road[k])
        targets = {k: fnode for k in order[: self.chasers]}
        # candidate block nodes: fugitive-reachable within horizon, weighted by funnel value
        cands = []
        for node, df in f_from.items():
            if df > self.horizon or df < 1e-6:
                continue
            te = self.to_exit.get(node, _INF)
            funnel = 1.0 / (1.0 + te / 200.0)
            cands.append((node, foff + df, funnel))
        taken = set()
        for k in order[self.chasers:]:
            n, off = dec[k]
            my_from = self.g.dist_from(n)
            best, best_score = None, -_INF
            for node, fdist, funnel in cands:
                if node in taken:
                    continue
                md = my_from.get(node)
                if md is None:
                    continue
                margin = fdist / self.vf - (off + md) / self.vp
                if margin <= 0:
                    continue
                score = self.funnel_weight * funnel - fdist / self.horizon * 0.5 + min(margin, 30.0) / 60.0
                if score > best_score:
                    best_score, best = score, node
            if best is None:
                targets[k] = fnode
            else:
                taken.add(best)
                targets[k] = best
        recs = []
        for k in range(len(police)):
            heading = None if incoming_headings is None else incoming_headings[k]
            did = dec[k][0]
            mask, ordered = build_action_mask(self.network, did, heading)
            if not ordered:
                chosen = STAY_ACTION
            elif targets[k] == did and k not in order[: self.chasers]:
                chosen = STAY_ACTION if mask[STAY_ACTION] else self._route(ordered, fnode, k)
            else:
                chosen = self._route(ordered, targets[k], k)
            if chosen != STAY_ACTION and ordered and not self.g.seg[ordered[chosen]].virtual:
                self._prev[k] = did
            probs = np.zeros(ACTION_DIM)
            probs[chosen] = 1.0
            recs.append(decode_recommendation(
                agent_id=f"police_{k}", network=self.network, decision_intersection_id=did, mask=mask,
                ordered_segment_ids=ordered, probabilities=probs, profile=self.profile,
                compatibility_report_id=self.profile, action_index=chosen,
            ))
        return tuple(recs)

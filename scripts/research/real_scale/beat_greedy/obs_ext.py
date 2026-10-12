"""Extended per-officer observation: the audited 28D vector plus coordination
features computed per action slot.

Everything is derived from the same inputs every baseline already uses
(officer and fugitive placements on the road graph, plus the fugitive's
recent motion), so no information is given to the learned policy that the
rule-based baselines don't also have.

Layout (64 values, all in [0, 1]):
  [0:28]   audited 28D vector (Observation28DAdapter), unchanged
  [28:34]  globals: fugitive heading cos/sin, fugitive road distance to the
           nearest boundary exit, my rank by road distance to the fugitive,
           fugitive/police speed ratio, my legal exit count
  [34:64]  5 slots x 6 features, in the env's action-slot order:
           euclidean distance slot-end -> fugitive,
           road distance slot-end -> fugitive's next intersection,
           arrival advantage at slot-end (fugitive time - my time),
           teammate coverage at slot-end (closest teammate's road distance
             minus mine through this slot),
           slot-end road distance to the nearest boundary exit,
           slot exists flag

The adapter is stateful only for the fugitive's motion history; it must be
created fresh per episode and fed every step in order, which is exactly how
``replay_episode`` and the training rollouts use observation adapters.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from pursuit_evasion_rl.osm_demo.coarsening import ordered_outgoing_segment_ids
from pursuit_evasion_rl.osm_demo.metrics import placement_position
from pursuit_evasion_rl.osm_demo.models import ModelNetwork, VehiclePlacement
from pursuit_evasion_rl.research.policies.baselines import _Graph
from pursuit_evasion_rl.research.variants.observations import Observation28DAdapter

EXT_PROFILE_ID = "beat_greedy_ext64_v1"
EXT_OBS_DIM = 64
SLOTS = 5
POLICE_COUNT = 6
_INF = float("inf")

# Topology caches shared across episodes (they depend only on the network).
_GRAPHS: dict[int, _Graph] = {}
_BOUNDARY_DIST: dict[int, dict[int, float]] = {}


def graph_for(network: ModelNetwork) -> _Graph:
    key = id(network)
    if key not in _GRAPHS:
        _GRAPHS[key] = _Graph(network)
    return _GRAPHS[key]


def boundary_distance(network: ModelNetwork) -> dict[int, float]:
    """Road distance from every intersection to its nearest boundary exit."""
    key = id(network)
    if key not in _BOUNDARY_DIST:
        graph = graph_for(network)
        exits = [s.start_id for s in network.segments if s.crosses_boundary]
        best: dict[int, float] = {}
        for exit_node in set(exits):
            for node, d in graph.dist_to(exit_node).items():
                if d < best.get(node, _INF):
                    best[node] = d
        _BOUNDARY_DIST[key] = best
    return _BOUNDARY_DIST[key]


class ExtendedObservationAdapter:
    def __init__(
        self, network: ModelNetwork, *, clip_distance_m: float = 2000.0, near_radius_m: float = 200.0,
        police_speed_mps: float = 16.0, fugitive_speed_mps: float = 9.0,
    ) -> None:
        self.net = network
        self.clip = float(clip_distance_m)
        self.base = Observation28DAdapter(network, clip_distance_m=clip_distance_m, near_radius_m=near_radius_m)
        self.g = graph_for(network)
        self.boundary = boundary_distance(network)
        self.vp = float(police_speed_mps)
        self.vf = float(fugitive_speed_mps)
        self._fug_history: list[tuple[float, float]] = []
        self._last_step: int | None = None
        self._step_cache: dict | None = None

    # -- helpers -------------------------------------------------------------
    def _reach(self, placement: VehiclePlacement) -> tuple[int, float]:
        if placement.segment_id is not None:
            seg = self.g.seg[placement.segment_id]
            return int(seg.end_id), (1.0 - placement.progress) * float(seg.length_m)
        return int(placement.intersection_id), 0.0

    def _road_to_fugitive(self, placement: VehiclePlacement, fugitive: VehiclePlacement) -> float:
        reach, offset = self._reach(placement)
        if fugitive.segment_id is not None:
            fseg = self.g.seg[fugitive.segment_id]
            entry, toff = int(fseg.start_id), fugitive.progress * float(fseg.length_m)
            if placement.segment_id == fugitive.segment_id and placement.progress <= fugitive.progress:
                return (fugitive.progress - placement.progress) * float(fseg.length_m)
        else:
            entry, toff = int(fugitive.intersection_id), 0.0
        d = self.g.dist_to(entry).get(reach)
        return _INF if d is None else offset + d + toff

    def _norm(self, d: float) -> float:
        return 1.0 if not math.isfinite(d) else min(max(d, 0.0) / self.clip, 1.0)

    def _per_step(self, police: Sequence[VehiclePlacement], fugitive: VehiclePlacement, step: int) -> dict:
        """Work shared by all six officers' observations of the same state."""
        if self._last_step == step and self._step_cache is not None:
            return self._step_cache
        fxy = placement_position(self.net, fugitive)
        if not self._fug_history or self._fug_history[-1] != fxy:
            self._fug_history.append(fxy)
            self._fug_history = self._fug_history[-6:]
        heading = None
        if len(self._fug_history) >= 2:
            (ax, ay), (bx, by) = self._fug_history[0], self._fug_history[-1]
            n = math.hypot(bx - ax, by - ay)
            if n > 1e-6:
                heading = ((bx - ax) / n, (by - ay) / n)
        freach, foffset = self._reach(fugitive)
        road = [self._road_to_fugitive(p, fugitive) for p in police]
        order = sorted(range(len(police)), key=lambda k: road[k])
        rank = {k: r for r, k in enumerate(order)}
        reaches = [self._reach(p) for p in police]
        self._step_cache = {
            "fxy": fxy, "heading": heading, "freach": freach, "foffset": foffset,
            "f_from": self.g.dist_from(freach), "rank": rank, "reaches": reaches,
        }
        self._last_step = step
        return self._step_cache

    # -- observation -----------------------------------------------------------
    def observe(
        self, *, police_index: int, police: Sequence[VehiclePlacement], fugitive: VehiclePlacement,
        step: int, max_steps: int, incoming_heading: float | None = None,
    ) -> np.ndarray:
        base = self.base.observe(
            police_index=police_index, police=police, fugitive=fugitive, step=step,
            max_steps=max_steps, incoming_heading=incoming_heading,
        )
        c = self._per_step(police, fugitive, step)
        my_reach, my_offset = c["reaches"][police_index]
        ordered = ordered_outgoing_segment_ids(self.net, my_reach, incoming_heading)

        heading = c["heading"]
        globals_ = [
            0.5 if heading is None else (heading[0] + 1) / 2,
            0.5 if heading is None else (heading[1] + 1) / 2,
            self._norm(c["foffset"] + self.boundary.get(c["freach"], _INF)),
            c["rank"][police_index] / (POLICE_COUNT - 1),
            min(self.vf / self.vp, 1.5) / 1.5,
            min(len(ordered), SLOTS) / SLOTS,
        ]

        slots: list[float] = []
        f_from = c["f_from"]
        for slot in range(SLOTS):
            if slot >= len(ordered):
                slots.extend([1.0, 1.0, 0.0, 1.0, 1.0, 0.0])
                continue
            seg = self.g.seg[ordered[slot]]
            end = int(seg.end_id)
            my_dist_to_end = my_offset + float(seg.length_m)
            euclid = math.dist(self.g.pos[end], c["fxy"])
            to_freach = self.g.dist_to(c["freach"]).get(end, _INF)
            f_dist_to_end = c["foffset"] + f_from.get(end, _INF)
            advantage_s = f_dist_to_end / self.vf - my_dist_to_end / self.vp
            teammate_best = _INF
            for k, (r, off) in enumerate(c["reaches"]):
                if k == police_index:
                    continue
                d = self.g.dist_to(end).get(r)
                if d is not None:
                    teammate_best = min(teammate_best, off + d)
            coverage = teammate_best - my_dist_to_end
            slots.extend([
                self._norm(euclid),
                self._norm(to_freach),
                1.0 if not math.isfinite(advantage_s) else min(max(advantage_s / 120.0 + 0.5, 0.0), 1.0),
                1.0 if not math.isfinite(coverage) else min(max(coverage / (2 * self.clip) + 0.5, 0.0), 1.0),
                self._norm(self.boundary.get(end, _INF)),
                1.0,
            ])
        vector = np.concatenate([base, np.asarray(globals_ + slots, dtype=np.float32)]).astype(np.float32)
        assert vector.shape[0] == EXT_OBS_DIM
        return np.clip(vector, 0.0, 1.0)


# =============================================================================
# v2: escape-route containment features
# =============================================================================
EXT2_PROFILE_ID = "beat_greedy_ext71_v2"
EXT2_OBS_DIM = 71
EXIT_HORIZON_M = 1500.0
_EXITS: dict[int, tuple[int, ...]] = {}


def exit_nodes(network: ModelNetwork) -> tuple[int, ...]:
    key = id(network)
    if key not in _EXITS:
        _EXITS[key] = tuple(sorted({int(s.start_id) for s in network.segments if s.crosses_boundary}))
    return _EXITS[key]


def _reach_of(graph: _Graph, placement: VehiclePlacement) -> tuple[int, float]:
    if placement.segment_id is not None:
        seg = graph.seg[placement.segment_id]
        return int(seg.end_id), (1.0 - placement.progress) * float(seg.length_m)
    return int(placement.intersection_id), 0.0


def open_exits(network: ModelNetwork, police, fugitive, vp: float, vf: float) -> tuple[list[tuple[int, float]], int]:
    """Boundary exits the fugitive can reach before every officer.

    Returns ([(exit_node, fugitive_time_s)], number of exits within the
    fugitive's horizon). An exit the fugitive gets to first is an open escape
    route; encirclement is exactly the process of closing all of them.
    """
    g = graph_for(network)
    freach, foff = _reach_of(g, fugitive)
    f_from = g.dist_from(freach)
    officers = [_reach_of(g, p) for p in police]
    officer_from = [(g.dist_from(n), off) for n, off in officers]
    out: list[tuple[int, float]] = []
    relevant = 0
    for e in exit_nodes(network):
        df = f_from.get(e)
        if df is None or foff + df > EXIT_HORIZON_M:
            continue
        relevant += 1
        tf = (foff + df) / vf
        tp = min(((off + d[e]) / vp for d, off in officer_from if e in d), default=_INF)
        if tf < tp:
            out.append((e, tf))
    return out, relevant


class ExtendedObservationAdapterV2(ExtendedObservationAdapter):
    """v1 (64) + open-exit globals (2) + per-slot exits-closed counts (5) = 71."""

    def _per_step(self, police, fugitive, step):
        cache = super()._per_step(police, fugitive, step)
        if "open" not in cache:
            cache["open"], cache["relevant"] = open_exits(self.net, police, fugitive, self.vp, self.vf)
        return cache

    def observe(self, *, police_index, police, fugitive, step, max_steps, incoming_heading=None):
        v1 = super().observe(
            police_index=police_index, police=police, fugitive=fugitive, step=step,
            max_steps=max_steps, incoming_heading=incoming_heading,
        )
        c = self._per_step(police, fugitive, step)
        opened = c["open"]
        globals2 = [
            len(opened) / max(c["relevant"], 1),
            min(min((t for _, t in opened), default=120.0) / 120.0, 1.0),
        ]
        my_reach, my_offset = c["reaches"][police_index]
        ordered = ordered_outgoing_segment_ids(self.net, my_reach, incoming_heading)
        closes = []
        for slot in range(SLOTS):
            if slot >= len(ordered) or not opened:
                closes.append(0.0)
                continue
            seg = self.g.seg[ordered[slot]]
            from_end = self.g.dist_from(int(seg.end_id))
            base = my_offset + float(seg.length_m)
            n = sum(1 for e, tf in opened if e in from_end and (base + from_end[e]) / self.vp < tf)
            closes.append(min(n / 5.0, 1.0))
        vector = np.concatenate([v1, np.asarray(globals2 + closes, dtype=np.float32)])
        assert vector.shape[0] == EXT2_OBS_DIM
        return np.clip(vector, 0.0, 1.0).astype(np.float32)

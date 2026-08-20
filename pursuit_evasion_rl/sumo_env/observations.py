"""Traffic-aware observations for the SUMO pursuit environment.

The abstract environment's 28D vector told an officer where the fugitive was
but nothing about the road it would have to drive.  The reference study's
answer was to feed each lane's vehicle counts by class -- background, pursuing,
evading -- into the policy.  This module reproduces that information locally:
for each legal exit an officer could take, it reports how busy that exit is
and who is on it.

Layout (per officer):
    [ 0: 3]  self: normalized speed, junction flag, remaining route length
    [ 3: 7]  fugitive: road distance, bearing sin/cos, speed ratio
    [ 7:32]  per-exit block, 5 exits x 5 features:
                background count, police count, fugitive present,
                occupancy, mean-speed ratio
    [32:37]  per-exit progress: does this exit reduce road distance to the fugitive
Total: 37 dimensions.

Counts are normalized by ``COUNT_SCALE`` rather than by an observed maximum so
the same number means the same thing across maps and episodes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from pursuit_evasion_rl.sumo_env.environment import (
    ACTION_SIZE,
    FUGITIVE_ID,
    POLICE_IDS,
    SumoPursuitEnv,
)

#: Vehicles per lane that maps to 1.0.  Twelve is roughly a full urban lane
#: between signals, so values above 1.0 genuinely mean "jammed".
COUNT_SCALE = 12.0
MAX_EXITS = ACTION_SIZE - 1
OBSERVATION_DIM = 3 + 4 + MAX_EXITS * 5 + MAX_EXITS


@dataclass(frozen=True, slots=True)
class ExitTraffic:
    """Traffic on one candidate exit edge."""

    edge_id: str
    background: int
    police: int
    fugitive_present: bool
    occupancy: float
    mean_speed_ratio: float
    reduces_distance: bool


class SumoObservationAdapter:
    """Builds per-officer observation vectors from a live TraCI connection.

    Road distances come from a per-episode Dijkstra cache over the SUMO
    network, so "closer" means closer *to drive*, not closer in a straight
    line -- the same correction the reward side needed.
    """

    def __init__(self, env: SumoPursuitEnv, *, clip_distance_m: float = 3000.0) -> None:
        self.env = env
        self.clip_distance_m = float(clip_distance_m)
        self._net = env.sumolib_net
        self._distance_cache: dict[tuple[str, str], float] = {}

    # -- helpers ------------------------------------------------------
    def _connection(self):
        return self.env._connection  # single owner; the env manages the socket

    def _lane_ids(self, edge_id: str) -> list[str]:
        try:
            edge = self._net.getEdge(edge_id)
        except Exception:
            return []
        return [lane.getID() for lane in edge.getLanes()]

    def road_distance_m(self, from_edge: str, to_edge: str) -> float:
        """Shortest driving distance between two edges, cached per pair."""
        if from_edge == to_edge:
            return 0.0
        key = (from_edge, to_edge)
        cached = self._distance_cache.get(key)
        if cached is not None:
            return cached
        try:
            source = self._net.getEdge(from_edge)
            target = self._net.getEdge(to_edge)
            path, cost = self._net.getShortestPath(source, target)
            value = float(cost) if path else self.clip_distance_m
        except Exception:
            value = self.clip_distance_m
        value = min(value, self.clip_distance_m)
        self._distance_cache[key] = value
        return value

    def _vehicle_class_counts(self, edge_id: str) -> tuple[int, int, bool]:
        """(background, police, fugitive_present) currently on an edge."""
        connection = self._connection()
        try:
            vehicles = connection.edge.getLastStepVehicleIDs(edge_id)
        except Exception:
            return 0, 0, False
        police = sum(1 for item in vehicles if item in POLICE_IDS)
        fugitive = any(item == FUGITIVE_ID for item in vehicles)
        background = max(0, len(vehicles) - police - (1 if fugitive else 0))
        return background, police, fugitive

    def _edge_traffic(self, edge_id: str) -> tuple[float, float]:
        """(occupancy 0-1, mean speed / allowed speed)."""
        connection = self._connection()
        occupancies: list[float] = []
        ratios: list[float] = []
        for lane_id in self._lane_ids(edge_id):
            try:
                occupancies.append(float(connection.lane.getLastStepOccupancy(lane_id)))
                mean_speed = float(connection.lane.getLastStepMeanSpeed(lane_id))
                allowed = float(connection.lane.getMaxSpeed(lane_id)) or 1.0
                ratios.append(max(0.0, min(1.0, mean_speed / allowed)))
            except Exception:
                continue
        occupancy = sum(occupancies) / len(occupancies) if occupancies else 0.0
        ratio = sum(ratios) / len(ratios) if ratios else 1.0
        return occupancy, ratio

    # -- public -------------------------------------------------------
    def exit_traffic(self, officer_id: str) -> tuple[ExitTraffic, ...]:
        """Traffic on each legal exit, in the same slot order as the action mask."""
        state = self.env.episode_state()
        fugitive_edge = state.fugitive.edge_id if state.fugitive else None
        officer = next((item for item in state.police if item.vehicle_id == officer_id), None)
        if officer is None:
            return ()
        here_distance = (
            self.road_distance_m(officer.edge_id, fugitive_edge)
            if fugitive_edge and not fugitive_edge.startswith(":")
            else self.clip_distance_m
        )
        results: list[ExitTraffic] = []
        for edge_id in self.env.legal_targets(officer_id):
            background, police, fugitive_here = self._vehicle_class_counts(edge_id)
            occupancy, ratio = self._edge_traffic(edge_id)
            distance = (
                self.road_distance_m(edge_id, fugitive_edge)
                if fugitive_edge and not fugitive_edge.startswith(":")
                else self.clip_distance_m
            )
            results.append(
                ExitTraffic(
                    edge_id=edge_id,
                    background=background,
                    police=police,
                    fugitive_present=fugitive_here,
                    occupancy=occupancy,
                    mean_speed_ratio=ratio,
                    reduces_distance=distance < here_distance,
                )
            )
        return tuple(results)

    def observe(self, officer_id: str) -> np.ndarray:
        """Fixed-width observation vector for one officer."""
        vector = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        state = self.env.episode_state()
        officer = next((item for item in state.police if item.vehicle_id == officer_id), None)
        if officer is None:
            return vector

        max_speed = max(1e-6, self.env.config.police_speed_mps)
        vector[0] = min(1.0, officer.speed_mps / max_speed)
        vector[1] = 1.0 if officer.at_junction else 0.0
        vector[2] = min(1.0, len(officer.route_edges) / 20.0)

        fugitive = state.fugitive
        if fugitive is not None:
            if not officer.edge_id.startswith(":") and not fugitive.edge_id.startswith(":"):
                distance = self.road_distance_m(officer.edge_id, fugitive.edge_id)
            else:
                distance = math.dist(officer.position_xy, fugitive.position_xy)
            vector[3] = min(1.0, distance / self.clip_distance_m)
            dx = fugitive.position_xy[0] - officer.position_xy[0]
            dy = fugitive.position_xy[1] - officer.position_xy[1]
            bearing = math.atan2(dy, dx)
            vector[4] = math.sin(bearing)
            vector[5] = math.cos(bearing)
            vector[6] = min(1.0, fugitive.speed_mps / max(1e-6, self.env.config.fugitive_speed_mps))
        else:
            vector[3] = 1.0

        base = 7
        exits = self.exit_traffic(officer_id)
        for slot, exit_info in enumerate(exits[:MAX_EXITS]):
            offset = base + slot * 5
            vector[offset + 0] = min(2.0, exit_info.background / COUNT_SCALE)
            vector[offset + 1] = min(1.0, exit_info.police / max(1.0, len(POLICE_IDS)))
            vector[offset + 2] = 1.0 if exit_info.fugitive_present else 0.0
            vector[offset + 3] = min(1.0, exit_info.occupancy)
            vector[offset + 4] = exit_info.mean_speed_ratio
            vector[base + MAX_EXITS * 5 + slot] = 1.0 if exit_info.reduces_distance else 0.0
        return vector

    def observe_all(self) -> dict[str, np.ndarray]:
        return {officer: self.observe(officer) for officer in POLICE_IDS}


__all__ = (
    "COUNT_SCALE",
    "MAX_EXITS",
    "OBSERVATION_DIM",
    "ExitTraffic",
    "SumoObservationAdapter",
)
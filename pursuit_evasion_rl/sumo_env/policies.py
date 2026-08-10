"""Rule-based cooperative encirclement for the SUMO environment.

This is the demo policy: no checkpoint, deterministic given a seed, and its
abstract-environment twin (``greedy_intercept``) is the strongest baseline this
project has measured.  Using it keeps every on-screen claim true while the
learned policy is retrained.

Dispatch, not steering.  An operations room tells a car *where to go*, not
which way to turn at the next light, so each officer is assigned a blocking
edge and SUMO routes it there -- rerouting itself around whatever congestion
appears.  That division of labour is also why this policy is cheap: one
assignment every few seconds instead of a graph search at every junction.

Cooperation comes from the assignment, not from the driving: the fugitive's
surroundings are cut into six bearing sectors, each officer owns one, and
officers are matched to sectors by who can reach them soonest.  Six cars
therefore close a ring instead of queueing along one road.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from pursuit_evasion_rl.sumo_env.environment import (
    FUGITIVE_ID,
    POLICE_IDS,
    SumoPursuitEnv,
)


@dataclass(frozen=True, slots=True)
class EncirclementConfig:
    """Blocking-ring geometry and how strongly traffic shapes the choice.

    ``ring_radius_m`` is how far ahead of the fugitive the ring is drawn: too
    tight and officers arrive behind it, too wide and they never close.
    ``congestion_weight`` is what separates this from a distance-greedy
    chaser -- a jammed approach costs time, so a longer clear road can win.
    """

    ring_radius_m: float = 260.0
    spread_start_m: float = 700.0
    max_lead_s: float = 90.0
    replan_every_steps: int = 5
    congestion_weight: float = 900.0
    close_in_distance_m: float = 260.0
    flee_candidates: int = 12
    min_edge_length_m: float = 25.0


class EncirclementPolicy:
    """Assigns each officer a blocking edge in its own bearing sector."""

    def __init__(
        self,
        env: SumoPursuitEnv,
        config: EncirclementConfig | None = None,
    ) -> None:
        self.env = env
        self.config = config or EncirclementConfig()
        self._net = env.sumolib_net
        self._assignments: dict[str, str] = {}
        self._last_plan_step = -10**9
        self._last_fugitive_xy: tuple[float, float] | None = None
        self._velocity_estimate: tuple[float, float] = (0.0, 0.0)
        self._edges = [
            edge for edge in self._net.getEdges()
            if not edge.getID().startswith(":")
            and edge.getLength() >= self.config.min_edge_length_m
        ]
        self._midpoints = {
            edge.getID(): self._midpoint(edge) for edge in self._edges
        }

    @staticmethod
    def _midpoint(edge) -> tuple[float, float]:
        shape = edge.getShape()
        return (
            sum(point[0] for point in shape) / len(shape),
            sum(point[1] for point in shape) / len(shape),
        )

    def _traffic_cost(self, edge_id: str) -> float:
        """Seconds-ish penalty for an edge whose traffic has slowed it down."""
        connection = self.env._connection
        try:
            occupancy = float(connection.edge.getLastStepOccupancy(edge_id))
            mean_speed = float(connection.edge.getLastStepMeanSpeed(edge_id))
        except Exception:
            return 0.0
        try:
            allowed = max(1.0, self._net.getEdge(edge_id).getSpeed())
        except Exception:
            allowed = 1.0
        slowdown = max(0.0, 1.0 - mean_speed / allowed)
        return self.config.congestion_weight * (0.6 * slowdown + 0.4 * occupancy)

    def _nearest_edge(self, point, exclude=()):
        """Edge whose midpoint is closest to a map point."""
        best_id, best_gap = "", math.inf
        for edge_id, midpoint in self._midpoints.items():
            if edge_id in exclude:
                continue
            gap = math.dist(midpoint, point)
            if gap < best_gap:
                best_id, best_gap = edge_id, gap
        return best_id

    def _fugitive_velocity(self, state) -> tuple[float, float]:
        """Estimated fugitive velocity (m/s) from its last two observed positions."""
        fugitive = state.fugitive
        if fugitive is None:
            return 0.0, 0.0
        current = fugitive.position_xy
        previous = self._last_fugitive_xy
        self._last_fugitive_xy = current
        if previous is None:
            return 0.0, 0.0
        dt = max(1e-6, self.env.config.step_length_s)
        vx = (current[0] - previous[0]) / dt
        vy = (current[1] - previous[1]) / dt
        # Smooth: a single step of a car turning a corner is a poor heading.
        sx, sy = self._velocity_estimate
        smoothed = (0.6 * sx + 0.4 * vx, 0.6 * sy + 0.4 * vy)
        self._velocity_estimate = smoothed
        return smoothed

    def _intercept_point(self, officer_xy, fugitive_xy, velocity) -> tuple[float, float]:
        """Where the fugitive will be when this officer could get there.

        Two fixed-point iterations are enough: the estimate converges quickly
        because officer speed exceeds fugitive speed, which is exactly the
        condition that makes interception possible at all.
        """
        speed = max(1e-6, self.env.config.police_speed_mps)
        point = fugitive_xy
        for _ in range(2):
            eta = math.dist(officer_xy, point) / speed
            eta = min(eta, self.config.max_lead_s)
            point = (fugitive_xy[0] + velocity[0] * eta, fugitive_xy[1] + velocity[1] * eta)
        return point

    def _plan(self, state) -> dict[str, str]:
        """Assign every officer an intercept edge ahead of the fugitive.

        Far officers aim straight at the predicted meeting point -- spending
        their speed advantage on closing distance.  Only once the gap is small
        do they fan out into bearing sectors, which is when a ring can actually
        be closed rather than chased.
        """
        fugitive = state.fugitive
        if fugitive is None:
            return {}
        velocity = self._fugitive_velocity(state)
        assignments: dict[str, str] = {}
        claimed: set[str] = set()
        officers = sorted(
            state.police,
            key=lambda item: math.dist(item.position_xy, fugitive.position_xy),
        )
        for officer in officers:
            gap = math.dist(officer.position_xy, fugitive.position_xy)
            aim = self._intercept_point(officer.position_xy, fugitive.position_xy, velocity)
            # Spread grows as the gap closes: 0 while chasing, full ring when near.
            spread = self.config.ring_radius_m * max(
                0.0, min(1.0, (self.config.spread_start_m - gap) / self.config.spread_start_m)
            )
            if spread > 1.0:
                index = POLICE_IDS.index(officer.vehicle_id)
                bearing = (2.0 * math.pi * index) / len(POLICE_IDS)
                aim = (aim[0] + spread * math.cos(bearing), aim[1] + spread * math.sin(bearing))
            edge_id = self._nearest_edge(aim, exclude=claimed)
            if not edge_id:
                continue
            # Reject a target the officer cannot legally drive to.
            if not self.env.reachable(officer.edge_id, edge_id):
                edge_id = self._nearest_edge(fugitive.position_xy, exclude=claimed)
                if not edge_id or not self.env.reachable(officer.edge_id, edge_id):
                    continue
            assignments[officer.vehicle_id] = edge_id
            claimed.add(edge_id)
        return assignments

    def dispatch(self, state) -> dict[str, str]:
        """Target edge per vehicle for this step (empty string = keep current)."""
        fugitive = state.fugitive
        if fugitive is None:
            return {}
        stale = state.step - self._last_plan_step >= self.config.replan_every_steps
        if stale or not self._assignments:
            planned = self._plan(state)
            if planned:
                self._assignments = planned
                self._last_plan_step = state.step
        targets: dict[str, str] = {}
        for officer in state.police:
            officer_id = officer.vehicle_id
            gap = math.dist(officer.position_xy, fugitive.position_xy)
            # Close enough that chasing the actual car beats holding a line.
            if gap <= self.config.close_in_distance_m and not fugitive.edge_id.startswith(":"):
                if self.env.current_target(officer_id) != fugitive.edge_id:
                    targets[officer_id] = fugitive.edge_id
                continue
            assigned = self._assignments.get(officer_id)
            if assigned and self.env.current_target(officer_id) != assigned:
                targets[officer_id] = assigned
        flee = self._flee_target(state)
        if flee:
            targets[FUGITIVE_ID] = flee
        return targets

    def _flee_target(self, state) -> str:
        """Fugitive heads for whichever far edge is least covered by police."""
        fugitive = state.fugitive
        if fugitive is None or not state.police:
            return ""
        if self.env.current_target(FUGITIVE_ID) and state.step % 25 != 0:
            return ""
        if fugitive.edge_id.startswith(":"):
            return ""
        police_xy = [officer.position_xy for officer in state.police]
        ranked = []
        for edge_id, midpoint in self._midpoints.items():
            distance = math.dist(midpoint, fugitive.position_xy)
            if not 500.0 <= distance <= 2000.0:
                continue
            ranked.append((min(math.dist(midpoint, item) for item in police_xy), edge_id))
        ranked.sort(reverse=True)
        # Reachability is checked here rather than left to the environment:
        # a directed road network routinely makes the geometrically best
        # escape edge undrivable, and asking for it produces a rejected
        # dispatch instead of a route.
        for _score, edge_id in ranked[: self.config.flee_candidates]:
            if self.env.reachable(fugitive.edge_id, edge_id):
                return edge_id
        return ""


__all__ = ("EncirclementConfig", "EncirclementPolicy")

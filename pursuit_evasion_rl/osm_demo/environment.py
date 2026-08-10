"""Injected-network Gymnasium episode environment for the OSM road-pursuit demo.

Unlike :class:`pursuit_evasion_rl.road_pursuit.road_pursuit_env.RoadPursuitEnv`,
which generates its own synthetic grid, :class:`OSMRoadPursuitEnv` is constructed
from an immutable :class:`~pursuit_evasion_rl.osm_demo.models.ModelNetwork`.  It
moves vehicles along directed polyline arc length, resolves bounded zero-time
virtual routing microsteps introduced by degree splitting, and enforces the
episode contract from Requirements 3.3-3.5 and 9.2-9.7/9.9:

* six distinct police and one fugitive placed on distinct drivable positions,
  the fugitive in a non-boundary reachable component (no duplicate fallback),
* metric capture when any police is within the capture radius of the fugitive,
* escape only when the fugitive traverses a segment marked ``crosses_boundary``,
* capture wins ties with escape and records the resolution priority,
* timeout when neither capture nor escape occurs by ``max_steps``,
* a logged stay when a vehicle has no movement action (e.g. a dead end).

The environment is deliberately decoupled from the observation adapter
(``osm_topology_v1``) and the learned/baseline policies, which are implemented in
later tasks.  It emits a minimal structural observation (action mask + metric
position) so it stays runnable and testable entirely offline.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import gymnasium
import numpy as np
from gymnasium import spaces

from .coarsening import MAX_OUT_DEGREE, ordered_outgoing_segment_ids
from .models import (
    POLICE_COUNT,
    DomainValidationError,
    EpisodeConfig,
    EpisodeOutcome,
    EpisodeState,
    Intersection,
    ModelNetwork,
    Segment,
    VehiclePlacement,
    validate_vehicle_placements,
)

logger = logging.getLogger(__name__)

STAY_ACTION = MAX_OUT_DEGREE
ACTION_SIZE = MAX_OUT_DEGREE + 1
FUGITIVE_ID = "fugitive"

# An action can be a single index, a per-hop sequence consumed during virtual
# routing, or a callable ``(agent_id, intersection_id, ordered_ids, hop)->int``.
ActionInput = int | Sequence[int] | Callable[[str, int, Sequence[int], int], int]

_NORTH_HEADING = math.pi / 2


def _geometry_length(geometry: Sequence[tuple[float, float]]) -> float:
    return sum(math.dist(geometry[i], geometry[i + 1]) for i in range(len(geometry) - 1))


def _point_at_fraction(
    geometry: Sequence[tuple[float, float]], fraction: float
) -> tuple[float, float]:
    """Interpolate a point at ``fraction`` of the polyline's arc length."""
    fraction = min(1.0, max(0.0, fraction))
    total = _geometry_length(geometry)
    if total <= 0.0:
        return (float(geometry[0][0]), float(geometry[0][1]))
    target = fraction * total
    walked = 0.0
    for first, second in zip(geometry, geometry[1:]):
        segment_len = math.dist(first, second)
        if segment_len <= 0.0:
            continue
        if walked + segment_len >= target:
            local = (target - walked) / segment_len
            return (
                first[0] + local * (second[0] - first[0]),
                first[1] + local * (second[1] - first[1]),
            )
        walked += segment_len
    return (float(geometry[-1][0]), float(geometry[-1][1]))


def _heading_at_end(geometry: Sequence[tuple[float, float]]) -> float | None:
    for first, second in zip(reversed(geometry[:-1]), reversed(geometry[1:])):
        dx, dy = second[0] - first[0], second[1] - first[1]
        if dx or dy:
            return math.atan2(dy, dx)
    return None


@dataclass(slots=True)
class _Vehicle:
    """Mutable internal position of one agent.

    A vehicle is either moving along a physical ``segment`` (``progress`` in
    ``[0, 1]`` along polyline arc length) or ``parked`` at an intersection while
    it awaits/executes a routing decision.  ``physical_anchor`` is the last
    physical intersection visited; it is used to export a drivable placement when
    the vehicle rests on a colocated virtual intersection mid-chain.
    """

    agent_id: str
    is_police: bool
    speed: float
    mode: str  # "segment" | "parked"
    segment_id: int | None = None
    intersection_id: int | None = None
    physical_anchor: int | None = None
    progress: float = 0.0
    heading: float | None = None


class OSMRoadPursuitEnv(gymnasium.Env):
    """Gymnasium episode environment built from an immutable ``ModelNetwork``.

    ``step`` accepts a mapping from agent id to an :data:`ActionInput`.  During a
    step every moving vehicle finishes its physical movement first; then every
    parked vehicle resolves a bounded chain of zero-time virtual routing
    microsteps until it departs on a physical segment or stays.  Capture, escape
    and timeout are evaluated in that priority order on the resulting positions.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        network: ModelNetwork,
        config: EpisodeConfig,
        *,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(network, ModelNetwork):
            raise DomainValidationError("INVALID_NETWORK", "A ModelNetwork instance is required")
        self.network = network
        self.config = config
        self.render_mode = render_mode

        self._segments: dict[int, Segment] = {item.id: item for item in network.segments}
        self._intersections: dict[int, Intersection] = {item.id: item for item in network.intersections}
        self._geometry_length: dict[int, float] = {
            item.id: _geometry_length(item.geometry_xy) for item in network.segments
        }
        # Per-episode effective speed cap per segment (m/s), supplied via
        # ``reset(options={"segment_speeds": {...}})``.  Empty (the default)
        # reproduces prior behavior exactly: every vehicle moves at its own
        # constant speed.  Kept OUTSIDE the network model on purpose -- the
        # sealed ModelNetwork and its content hash must never vary per episode.
        self._segment_speed: dict[int, float] = {}
        virtual_intersections = sum(1 for item in network.intersections if item.virtual)
        # A vehicle can never legally revisit an intersection within a single
        # zero-time routing chain, so bounding hops by the virtual node count
        # guarantees termination even if a malformed network slipped through.
        self._hop_cap = virtual_intersections + 1

        self._police_ids = [f"police_{index}" for index in range(POLICE_COUNT)]
        self._agent_ids = self._police_ids + [FUGITIVE_ID]

        mask_space = spaces.MultiBinary(ACTION_SIZE)
        position_space = spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float64)
        agent_obs = spaces.Dict({"action_mask": mask_space, "position_xy": position_space})
        self.observation_space = spaces.Dict({agent: agent_obs for agent in self._agent_ids})
        self.action_space = spaces.Dict(
            {agent: spaces.Discrete(ACTION_SIZE) for agent in self._agent_ids}
        )

        self._vehicles: dict[str, _Vehicle] | None = None
        self._step: int = 0
        self._outcome: EpisodeOutcome | None = None
        self._terminal_priority: str | None = None
        self._rng: np.random.Generator | None = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self._step = 0
        self._outcome = None
        self._terminal_priority = None

        options = dict(options or {})
        raw_speeds = options.get("segment_speeds")
        self._segment_speed = {}
        if raw_speeds is not None:
            for key, value in dict(raw_speeds).items():
                speed = float(value)
                if int(key) not in self._segments or not math.isfinite(speed) or speed <= 0.0:
                    raise DomainValidationError(
                        "INVALID_SEGMENT_SPEED",
                        "segment_speeds must map known segment ids to positive finite m/s",
                    )
                self._segment_speed[int(key)] = speed
        police_placements = options.get("police")
        fugitive_placement = options.get("fugitive")
        if (police_placements is None) != (fugitive_placement is None):
            raise DomainValidationError(
                "INCOMPLETE_PLACEMENT", "Explicit placement requires both police and fugitive positions"
            )
        if police_placements is None:
            police_placements, fugitive_placement = self._default_placements(self._rng)
        validate_vehicle_placements(self.network, tuple(police_placements), fugitive_placement)

        self._vehicles = {}
        for agent_id, placement in zip(self._police_ids, police_placements):
            self._vehicles[agent_id] = self._vehicle_from_placement(
                agent_id, placement, is_police=True
            )
        self._vehicles[FUGITIVE_ID] = self._vehicle_from_placement(
            FUGITIVE_ID, fugitive_placement, is_police=False
        )

        return self._observations(), self._info(events=())

    def step(
        self, actions: Mapping[str, ActionInput]
    ) -> tuple[dict[str, Any], dict[str, float], dict[str, bool], dict[str, bool], dict[str, Any]]:
        if self._vehicles is None:
            raise RuntimeError("Call reset() before step().")
        if self._outcome is not None:
            return (
                self._observations(),
                {agent: 0.0 for agent in self._agent_ids},
                {agent: True for agent in self._agent_ids},
                {agent: False for agent in self._agent_ids},
                self._info(events=()),
            )

        events: list[str] = []

        # 1. Finish current physical movement for every moving vehicle.
        escape_pending = False
        for agent_id in self._agent_ids:
            escaped = self._advance(self._vehicles[agent_id], events)
            if escaped and agent_id == FUGITIVE_ID:
                escape_pending = True

        # 2. Resolve bounded zero-time virtual routing microsteps for parked vehicles.
        for agent_id in self._agent_ids:
            vehicle = self._vehicles[agent_id]
            if vehicle.mode == "parked":
                provider = self._make_provider(actions.get(agent_id, STAY_ACTION))
                self._resolve_decision(vehicle, provider, events)

        # 3. Advance the clock (virtual microsteps above added no simulated time).
        self._step += 1

        # 4. Evaluate terminal conditions with capture winning ties over escape.
        outcome, priority = self._evaluate_terminal(escape_pending)
        self._outcome = outcome
        self._terminal_priority = priority

        terminated = outcome in (EpisodeOutcome.CAPTURE, EpisodeOutcome.ESCAPE)
        truncated = outcome is EpisodeOutcome.TIMEOUT
        rewards = {agent: 0.0 for agent in self._agent_ids}
        return (
            self._observations(),
            rewards,
            {agent: terminated for agent in self._agent_ids},
            {agent: truncated for agent in self._agent_ids},
            self._info(events=tuple(events)),
        )

    def render(self) -> None:  # pragma: no cover - trivial text dump
        if self._vehicles is None:
            print("environment not reset")
            return
        print(f"=== step {self._step} outcome={self._outcome} ===")
        for agent_id in self._agent_ids:
            position = self._position(self._vehicles[agent_id])
            print(f"  {agent_id}: {self._vehicles[agent_id].mode} @ ({position[0]:.1f}, {position[1]:.1f})")

    # ------------------------------------------------------------------
    # Placement
    # ------------------------------------------------------------------
    def _default_placements(
        self, rng: np.random.Generator
    ) -> tuple[list[VehiclePlacement], VehiclePlacement]:
        """Deterministically choose a fugitive in a non-boundary reachable
        component and six distinct police, with no duplicate fallback."""
        drivable = sorted(
            (
                item
                for item in self.network.intersections
                if not item.virtual and (item.outgoing_segment_ids or item.incoming_segment_ids)
            ),
            key=lambda item: item.id,
        )
        interior_movable = [
            item.id
            for item in drivable
            if item.boundary_kind is None and item.outgoing_segment_ids
        ]
        if not interior_movable:
            raise DomainValidationError(
                "PLACEMENT_FAILED",
                "No non-boundary drivable intersection is available for the fugitive",
            )
        fugitive_id = int(rng.choice(interior_movable))
        police_pool = [item.id for item in drivable if item.id != fugitive_id]
        if len(police_pool) < POLICE_COUNT:
            raise DomainValidationError(
                "PLACEMENT_FAILED",
                "Model network cannot place six distinct police without duplicates",
                expected=POLICE_COUNT,
                actual=len(police_pool),
            )
        order = rng.permutation(len(police_pool))
        chosen = [police_pool[int(index)] for index in order[:POLICE_COUNT]]
        police = [VehiclePlacement(intersection_id=identity) for identity in chosen]
        fugitive = VehiclePlacement(intersection_id=fugitive_id)
        return police, fugitive

    def _vehicle_from_placement(
        self, agent_id: str, placement: VehiclePlacement, *, is_police: bool
    ) -> _Vehicle:
        speed = self.config.police_speed_mps if is_police else self.config.fugitive_speed_mps
        if placement.segment_id is not None:
            return _Vehicle(
                agent_id=agent_id,
                is_police=is_police,
                speed=speed,
                mode="segment",
                segment_id=placement.segment_id,
                progress=placement.progress,
                physical_anchor=self._segments[placement.segment_id].start_id,
                heading=None,
            )
        return _Vehicle(
            agent_id=agent_id,
            is_police=is_police,
            speed=speed,
            mode="parked",
            intersection_id=placement.intersection_id,
            physical_anchor=placement.intersection_id,
            heading=None,
        )

    # ------------------------------------------------------------------
    # Movement and routing
    # ------------------------------------------------------------------
    def _advance(self, vehicle: _Vehicle, events: list[str]) -> bool:
        """Advance a moving vehicle by one dt of arc length.

        Returns ``True`` when the vehicle completed a boundary-crossing segment
        (a candidate escape event).
        """
        if vehicle.mode != "segment" or vehicle.segment_id is None:
            return False
        segment = self._segments[vehicle.segment_id]
        geometry_length = self._geometry_length[vehicle.segment_id]
        speed_cap = self._segment_speed.get(vehicle.segment_id)
        effective_speed = vehicle.speed if speed_cap is None else min(vehicle.speed, speed_cap)
        distance = effective_speed * self.config.dt_s
        if geometry_length <= 0.0:
            new_progress = 1.0
        else:
            new_progress = vehicle.progress + distance / geometry_length
        if new_progress < 1.0:
            vehicle.progress = new_progress
            return False
        # Arrived at the physical end intersection of the segment.
        vehicle.mode = "parked"
        vehicle.progress = 1.0
        vehicle.intersection_id = segment.end_id
        vehicle.physical_anchor = segment.end_id
        heading = _heading_at_end(segment.geometry_xy)
        if heading is not None:
            vehicle.heading = heading
        if segment.crosses_boundary:
            events.append(f"{vehicle.agent_id}:boundary_crossed:{segment.id}")
            return True
        return False

    def _resolve_decision(
        self,
        vehicle: _Vehicle,
        provider: Callable[[str, int, Sequence[int], int], int],
        events: list[str],
    ) -> None:
        """Resolve a bounded chain of zero-time routing microsteps.

        The bound counts virtual segments actually traversed.  One additional
        routing decision is allowed after the final virtual hop so the vehicle
        may depart physically, but another virtual segment is never crossed.
        """
        virtual_hops = 0
        for hop in range(self._hop_cap + 1):
            intersection_id = vehicle.intersection_id
            ordered = ordered_outgoing_segment_ids(self.network, intersection_id, vehicle.heading)
            action = int(provider(vehicle.agent_id, intersection_id, ordered, hop))
            if action == STAY_ACTION or action < 0 or action >= len(ordered):
                if not ordered:
                    events.append(f"{vehicle.agent_id}:stay_no_action")
                else:
                    events.append(f"{vehicle.agent_id}:stay")
                return
            segment = self._segments[ordered[action]]
            if not self._intersections[intersection_id].virtual:
                vehicle.physical_anchor = intersection_id
            if segment.virtual:
                if virtual_hops >= self._hop_cap:
                    events.append(f"{vehicle.agent_id}:hop_cap")
                    return
                # Zero-time continuation to the next colocated virtual node.
                virtual_hops += 1
                vehicle.intersection_id = segment.end_id
                events.append(f"{vehicle.agent_id}:virtual_hop:{segment.id}")
                continue
            vehicle.mode = "segment"
            vehicle.segment_id = segment.id
            vehicle.progress = 0.0
            events.append(f"{vehicle.agent_id}:depart:{segment.id}")
            return
        # The provider could not resolve a physical decision within the bound.
        events.append(f"{vehicle.agent_id}:hop_cap")

    @staticmethod
    def _make_provider(
        value: ActionInput,
    ) -> Callable[[str, int, Sequence[int], int], int]:
        if callable(value):
            return value
        if isinstance(value, (list, tuple)):
            sequence = [int(item) for item in value]

            def sequence_provider(agent_id, intersection_id, ordered, hop):
                return sequence[hop] if hop < len(sequence) else STAY_ACTION

            return sequence_provider

        index = int(value)

        def single_provider(agent_id, intersection_id, ordered, hop):
            return index if hop == 0 else STAY_ACTION

        return single_provider

    # ------------------------------------------------------------------
    # Terminal evaluation
    # ------------------------------------------------------------------
    def _evaluate_terminal(self, escape_pending: bool) -> tuple[EpisodeOutcome | None, str | None]:
        captured = self._capture_detected()
        if captured and escape_pending:
            return EpisodeOutcome.CAPTURE, "capture_over_escape"
        if captured:
            return EpisodeOutcome.CAPTURE, "capture"
        if escape_pending:
            return EpisodeOutcome.ESCAPE, "escape"
        if self._step >= self.config.max_steps:
            return EpisodeOutcome.TIMEOUT, "timeout"
        return None, None

    def _capture_detected(self) -> bool:
        fugitive_position = self._position(self._vehicles[FUGITIVE_ID])
        radius = self.config.capture_radius_m
        for agent_id in self._police_ids:
            if math.dist(self._position(self._vehicles[agent_id]), fugitive_position) <= radius:
                return True
        return False

    # ------------------------------------------------------------------
    # State projection
    # ------------------------------------------------------------------
    def _position(self, vehicle: _Vehicle) -> tuple[float, float]:
        if vehicle.mode == "segment" and vehicle.segment_id is not None:
            return _point_at_fraction(self._segments[vehicle.segment_id].geometry_xy, vehicle.progress)
        return self._intersections[vehicle.intersection_id].position_xy

    def _placement(self, vehicle: _Vehicle) -> VehiclePlacement:
        if vehicle.mode == "segment" and vehicle.segment_id is not None:
            return VehiclePlacement(segment_id=vehicle.segment_id, progress=vehicle.progress)
        anchor = vehicle.physical_anchor
        if anchor is None or self._intersections[anchor].virtual:
            anchor = self._first_physical_anchor(vehicle.intersection_id)
        return VehiclePlacement(intersection_id=anchor)

    def _first_physical_anchor(self, intersection_id: int) -> int:
        if not self._intersections[intersection_id].virtual:
            return intersection_id
        position = self._intersections[intersection_id].position_xy
        for item in self.network.intersections:
            if not item.virtual and item.position_xy == position:
                return item.id
        return intersection_id

    def episode_state(self) -> EpisodeState:
        assert self._vehicles is not None
        police = tuple(self._placement(self._vehicles[agent_id]) for agent_id in self._police_ids)
        fugitive = self._placement(self._vehicles[FUGITIVE_ID])
        headings = {
            agent_id: vehicle.heading
            for agent_id, vehicle in self._vehicles.items()
            if vehicle.heading is not None
        }
        return EpisodeState(
            step=self._step,
            simulated_s=self._step * self.config.dt_s,
            police=police,
            fugitive=fugitive,
            incoming_headings=headings,
        )

    def action_masks(self) -> dict[str, np.ndarray]:
        assert self._vehicles is not None
        masks: dict[str, np.ndarray] = {}
        for agent_id in self._agent_ids:
            masks[agent_id] = self._action_mask(self._vehicles[agent_id])
        return masks

    def _action_mask(self, vehicle: _Vehicle) -> np.ndarray:
        mask = np.zeros(ACTION_SIZE, dtype=bool)
        mask[STAY_ACTION] = True  # stay is always valid
        if vehicle.mode == "parked":
            ordered = ordered_outgoing_segment_ids(
                self.network, vehicle.intersection_id, vehicle.heading
            )
            for index in range(min(len(ordered), MAX_OUT_DEGREE)):
                mask[index] = True
        return mask

    def _observations(self) -> dict[str, Any]:
        assert self._vehicles is not None
        observations: dict[str, Any] = {}
        for agent_id in self._agent_ids:
            vehicle = self._vehicles[agent_id]
            observations[agent_id] = {
                "action_mask": self._action_mask(vehicle),
                "position_xy": np.asarray(self._position(vehicle), dtype=np.float64),
            }
        return observations

    def _info(self, *, events: tuple[str, ...]) -> dict[str, Any]:
        return {
            "step": self._step,
            "simulated_s": self._step * self.config.dt_s,
            "events": events,
            "outcome": self._outcome.value if self._outcome is not None else None,
            "terminal_priority": self._terminal_priority,
        }

    @property
    def outcome(self) -> EpisodeOutcome | None:
        return self._outcome

    @property
    def terminal_priority(self) -> str | None:
        return self._terminal_priority

    @property
    def agent_ids(self) -> list[str]:
        return list(self._agent_ids)

    @property
    def police_ids(self) -> list[str]:
        return list(self._police_ids)

    @property
    def fugitive_id(self) -> str:
        return FUGITIVE_ID


__all__ = (
    "ACTION_SIZE",
    "FUGITIVE_ID",
    "STAY_ACTION",
    "OSMRoadPursuitEnv",
)

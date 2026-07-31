"""Legacy and OSM topology observation profiles for the road-pursuit demo.

Two intentionally separate adapters are provided:

``legacy_grid_v0``
    Reproduces the historical synthetic-grid 21-vector exactly by flattening the
    ``RoadObservationBuilder`` dictionary in ascending key order. It intentionally
    keeps the unnormalized consecutive node-number semantics (``my_position``,
    ``neighbors`` and ``other_positions``) because it exists solely for checkpoint
    archaeology and explicitly experimental direct inference. It provides no
    semantic compatibility with arbitrary OSM networks.

``osm_topology_v1``
    Builds 21 finite ``float32`` topology/distance features in ``[0, 1]`` with a
    stable action-slot ordering and *no* raw or canonical numeric identifier used
    as a feature value. Because every feature is a normalized directed distance,
    progress ratio or per-agent-index one-hot, the vector and action mask are
    invariant under any bijective relabeling of raw/canonical storage identifiers
    that preserves graph topology, geometry and agent correspondence.

Design references: design sections 6 (observation/action adapters) and
Requirements 7.2, 7.6, 7.8, 8.4, 8.11.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .coarsening import ACTION_ORDERING
from .models import (
    DomainValidationError,
    ModelNetwork,
    ObservationContract,
    POLICE_COUNT,
    VehiclePlacement,
)

OBSERVATION_DIM = 21
ACTION_SLOTS = 5
ACTION_DIM = 6  # five ordered outgoing slots + stay

LEGACY_PROFILE_ID = "legacy_grid_v0"
OSM_PROFILE_ID = "osm_topology_v1"

# Ascending key order of the historical RoadObservationBuilder dictionary. The
# concatenation of these fields in this order is the canonical legacy 21-vector.
LEGACY_FIELD_ORDER: tuple[str, ...] = (
    "agent_id_onehot",
    "current_step",
    "my_position",
    "my_progress",
    "nearest_boundary_dist",
    "neighbors",
    "other_positions",
)
LEGACY_FIELD_SIZES: tuple[int, ...] = (POLICE_COUNT, 1, 1, 1, 1, 5, POLICE_COUNT)

OSM_FIELD_ORDER: tuple[str, ...] = (
    "police_onehot",
    "step_ratio",
    "segment_progress",
    "distance_self_to_fugitive",
    "distance_self_to_boundary",
    "action_slot_fugitive_scores",
    "other_agent_distances",
)
OSM_FIELD_SIZES: tuple[int, ...] = (POLICE_COUNT, 1, 1, 1, 1, ACTION_SLOTS, POLICE_COUNT)

# Unreachable and padded slots always saturate to 1.0 (maximum normalized
# distance); the action mask disambiguates padding from real maximal distance.
UNREACHABLE_VALUE = 1.0
DEFAULT_CLIP_DISTANCE_M = 2000.0


def legacy_grid_v0_contract() -> ObservationContract:
    """Observation contract describing the archaeology-only legacy profile."""
    return ObservationContract(
        profile_id=LEGACY_PROFILE_ID,
        fields=LEGACY_FIELD_ORDER,
        dtype="float32",
        padding_value=-1.0,
        normalization={"method": "none", "note": "raw consecutive node numbers"},
        action_ordering="legacy_sorted_dict_keys",
        field_sizes=LEGACY_FIELD_SIZES,
    )


def osm_topology_v1_contract(clip_distance_m: float = DEFAULT_CLIP_DISTANCE_M) -> ObservationContract:
    """Observation contract for the normalized, ID-free OSM topology profile."""
    if not math.isfinite(clip_distance_m) or clip_distance_m <= 0.0:
        raise DomainValidationError(
            "INVALID_CLIP_DISTANCE", "clip_distance_m must be a positive finite number", actual=clip_distance_m
        )
    return ObservationContract(
        profile_id=OSM_PROFILE_ID,
        fields=OSM_FIELD_ORDER,
        dtype="float32",
        padding_value=UNREACHABLE_VALUE,
        normalization={
            "method": "directed_shortest_path_over_segment_lengths",
            "clip_distance_m": float(clip_distance_m),
            "unreachable_value": UNREACHABLE_VALUE,
        },
        action_ordering=ACTION_ORDERING,
        field_sizes=OSM_FIELD_SIZES,
    )


def flatten_legacy_grid_v0(observation: Mapping[str, "np.ndarray | Sequence[float]"]) -> np.ndarray:
    """Reproduce the historical 21-vector by ascending-key flattening.

    The legacy training code fed a ``spaces.Dict`` observation to the network by
    flattening it in ascending key order. This helper reproduces that exact
    vector, preserving the raw (unnormalized) node-number semantics.
    """
    missing = [name for name in LEGACY_FIELD_ORDER if name not in observation]
    if missing:
        raise DomainValidationError(
            "INCOMPLETE_LEGACY_OBSERVATION",
            "Legacy observation is missing required fields",
            expected=list(LEGACY_FIELD_ORDER),
            actual=missing,
        )
    parts: list[np.ndarray] = []
    for name, size in zip(LEGACY_FIELD_ORDER, LEGACY_FIELD_SIZES):
        component = np.asarray(observation[name], dtype=np.float32).reshape(-1)
        if component.shape[0] != size:
            raise DomainValidationError(
                "INVALID_LEGACY_FIELD_SIZE",
                f"Legacy field {name!r} must have exactly {size} values",
                expected=size,
                actual=int(component.shape[0]),
            )
        parts.append(component)
    vector = np.concatenate(parts).astype(np.float32)
    if vector.shape[0] != OBSERVATION_DIM:
        raise DomainValidationError(
            "INVALID_LEGACY_OBSERVATION_LENGTH",
            f"Legacy observation must flatten to exactly {OBSERVATION_DIM} values",
            expected=OBSERVATION_DIM,
            actual=int(vector.shape[0]),
        )
    return vector


@dataclass(frozen=True, slots=True)
class _Placement:
    """Internal, validated view of a vehicle placement for distance math."""

    on_segment: bool
    segment_id: int
    progress: float
    intersection_id: int


class OSMTopologyV1Adapter:
    """Builds ``osm_topology_v1`` observations for police agents.

    The adapter never uses a raw OSM identifier or a canonical numeric ID as a
    feature value. All features derive from normalized directed distances over
    segment lengths, the deterministic action ordering and the per-agent index
    one-hot, which makes the produced vector relabel-invariant.
    """

    def __init__(
        self,
        network: ModelNetwork,
        *,
        clip_distance_m: float = DEFAULT_CLIP_DISTANCE_M,
        police_count: int = POLICE_COUNT,
    ) -> None:
        if not math.isfinite(clip_distance_m) or clip_distance_m <= 0.0:
            raise DomainValidationError(
                "INVALID_CLIP_DISTANCE",
                "clip_distance_m must be a positive finite number",
                actual=clip_distance_m,
            )
        if police_count <= 0:
            raise DomainValidationError("INVALID_POLICE_COUNT", "police_count must be positive")
        self.network = network
        self.clip_distance_m = float(clip_distance_m)
        self.police_count = int(police_count)
        self._segments = {segment.id: segment for segment in network.segments}
        self._boundary_ids: tuple[int, ...] = tuple(
            sorted(item.id for item in network.intersections if item.boundary_kind is not None)
        )
        self._adjacency: dict[int, list[tuple[int, float]]] = {
            item.id: [] for item in network.intersections
        }
        for segment in network.segments:
            self._adjacency[segment.start_id].append((segment.end_id, float(segment.length_m)))
        self._dijkstra_cache: dict[int, dict[int, float]] = {}

    @property
    def contract(self) -> ObservationContract:
        return osm_topology_v1_contract(self.clip_distance_m)

    # -- distance helpers ---------------------------------------------------

    def _dijkstra(self, source: int) -> dict[int, float]:
        cached = self._dijkstra_cache.get(source)
        if cached is not None:
            return cached
        distances: dict[int, float] = {source: 0.0}
        queue: list[tuple[float, int]] = [(0.0, source)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance > distances.get(node, math.inf):
                continue
            for neighbor, weight in self._adjacency.get(node, ()):  # directed edges only
                candidate = distance + weight
                if candidate < distances.get(neighbor, math.inf):
                    distances[neighbor] = candidate
                    heapq.heappush(queue, (candidate, neighbor))
        self._dijkstra_cache[source] = distances
        return distances

    def _view(self, placement: VehiclePlacement) -> _Placement:
        if placement.segment_id is not None:
            if placement.segment_id not in self._segments:
                raise DomainValidationError(
                    "INVALID_POSITION", "Placement references an unknown segment", actual=placement.segment_id
                )
            return _Placement(True, placement.segment_id, float(placement.progress), -1)
        if placement.intersection_id is None or placement.intersection_id not in self._adjacency:
            raise DomainValidationError(
                "INVALID_POSITION",
                "Placement references an unknown intersection",
                actual=placement.intersection_id,
            )
        return _Placement(False, -1, 0.0, int(placement.intersection_id))

    def _reach_intersection(self, view: _Placement) -> tuple[int, float]:
        """Return the first intersection reachable moving forward and its offset."""
        if view.on_segment:
            segment = self._segments[view.segment_id]
            return segment.end_id, (1.0 - view.progress) * float(segment.length_m)
        return view.intersection_id, 0.0

    def _entry_intersection(self, view: _Placement) -> tuple[int, float]:
        """Return the intersection through which a target position is entered."""
        if view.on_segment:
            segment = self._segments[view.segment_id]
            return segment.start_id, view.progress * float(segment.length_m)
        return view.intersection_id, 0.0

    def _directed_distance(self, source: VehiclePlacement, target: VehiclePlacement) -> float:
        source_view = self._view(source)
        target_view = self._view(target)
        candidates: list[float] = []

        # Same segment, self behind or level with the target: travel straight ahead.
        if (
            source_view.on_segment
            and target_view.on_segment
            and source_view.segment_id == target_view.segment_id
            and source_view.progress <= target_view.progress
        ):
            length = float(self._segments[source_view.segment_id].length_m)
            candidates.append((target_view.progress - source_view.progress) * length)

        reach_id, source_offset = self._reach_intersection(source_view)
        entry_id, target_offset = self._entry_intersection(target_view)
        distances = self._dijkstra(reach_id)
        if entry_id in distances:
            candidates.append(source_offset + distances[entry_id] + target_offset)

        return min(candidates) if candidates else math.inf

    def _normalize(self, distance: float) -> float:
        if not math.isfinite(distance):
            return UNREACHABLE_VALUE
        return float(min(max(distance, 0.0) / self.clip_distance_m, 1.0))

    def _distance_to_boundary(self, source: VehiclePlacement) -> float:
        if not self._boundary_ids:
            return UNREACHABLE_VALUE
        source_view = self._view(source)
        reach_id, source_offset = self._reach_intersection(source_view)
        distances = self._dijkstra(reach_id)
        best = math.inf
        for boundary_id in self._boundary_ids:
            if boundary_id in distances:
                best = min(best, source_offset + distances[boundary_id])
        return self._normalize(best)

    # -- observation --------------------------------------------------------

    def observe(
        self,
        *,
        police_index: int,
        police: Sequence[VehiclePlacement],
        fugitive: VehiclePlacement,
        step: int,
        max_steps: int,
        incoming_heading: float | None = None,
    ) -> np.ndarray:
        """Return the 21-value ``osm_topology_v1`` observation for one officer."""
        from .coarsening import ordered_outgoing_segment_ids

        if not 0 <= police_index < self.police_count:
            raise DomainValidationError(
                "INVALID_AGENT_INDEX",
                "police_index must reference a valid officer",
                expected=[0, self.police_count - 1],
                actual=police_index,
            )
        if len(police) != self.police_count:
            raise DomainValidationError(
                "INVALID_POLICE_COUNT",
                f"Exactly {self.police_count} police placements are required",
                expected=self.police_count,
                actual=len(police),
            )
        if max_steps <= 0:
            raise DomainValidationError("INVALID_EPISODE_CONFIG", "max_steps must be positive", actual=max_steps)
        if step < 0:
            raise DomainValidationError("INVALID_EPISODE_STATE", "step must be nonnegative", actual=step)

        self_placement = police[police_index]
        self_view = self._view(self_placement)

        values: list[float] = []

        # police one-hot(6)
        one_hot = [0.0] * self.police_count
        one_hot[police_index] = 1.0
        values.extend(one_hot)

        # step ratio and current segment progress
        values.append(float(min(max(step / max_steps, 0.0), 1.0)))
        values.append(float(min(max(self_view.progress, 0.0), 1.0)))

        # directed distance self -> fugitive and self -> nearest escape boundary
        values.append(self._normalize(self._directed_distance(self_placement, fugitive)))
        values.append(self._distance_to_boundary(self_placement))

        # five destination-to-fugitive scores in deterministic action-slot order
        decision_id, _ = self._reach_intersection(self_view)
        ordered = ordered_outgoing_segment_ids(self.network, decision_id, incoming_heading)
        for slot in range(ACTION_SLOTS):
            if slot < len(ordered):
                destination = self._segments[ordered[slot]].end_id
                distances = self._dijkstra(destination)
                entry_id, target_offset = self._entry_intersection(self._view(fugitive))
                distance = distances[entry_id] + target_offset if entry_id in distances else math.inf
                values.append(self._normalize(distance))
            else:
                values.append(UNREACHABLE_VALUE)

        # directed self-to-other distances: police 0..N excluding self, then fugitive
        others: list[float] = []
        for other_index in range(self.police_count):
            if other_index == police_index:
                continue
            others.append(self._normalize(self._directed_distance(self_placement, police[other_index])))
        others.append(self._normalize(self._directed_distance(self_placement, fugitive)))
        # Pad to exactly six slots (five other police + fugitive) for stable layout.
        while len(others) < POLICE_COUNT:
            others.append(UNREACHABLE_VALUE)
        values.extend(others[:POLICE_COUNT])

        vector = np.asarray(values, dtype=np.float32)
        if vector.shape[0] != OBSERVATION_DIM:
            raise DomainValidationError(
                "INVALID_OBSERVATION_LENGTH",
                f"osm_topology_v1 must produce exactly {OBSERVATION_DIM} values",
                expected=OBSERVATION_DIM,
                actual=int(vector.shape[0]),
            )
        if not np.all(np.isfinite(vector)):
            raise DomainValidationError("NON_FINITE_OBSERVATION", "osm_topology_v1 produced a non-finite value")
        # Clamp defensively so float32 rounding cannot escape [0, 1].
        return np.clip(vector, 0.0, 1.0).astype(np.float32)


__all__ = (
    "ACTION_DIM",
    "ACTION_SLOTS",
    "DEFAULT_CLIP_DISTANCE_M",
    "LEGACY_FIELD_ORDER",
    "LEGACY_FIELD_SIZES",
    "LEGACY_PROFILE_ID",
    "OBSERVATION_DIM",
    "OSM_FIELD_ORDER",
    "OSM_FIELD_SIZES",
    "OSM_PROFILE_ID",
    "UNREACHABLE_VALUE",
    "OSMTopologyV1Adapter",
    "flatten_legacy_grid_v0",
    "legacy_grid_v0_contract",
    "osm_topology_v1_contract",
)

"""Heuristic fugitive policy for directed OSM Model_Network episodes.

This module adapts the legacy grid :class:`HeuristicFugitive`
(:mod:`pursuit_evasion_rl.road_pursuit.heuristic_fugitive`) to the directed,
metric OSM networks produced by :mod:`pursuit_evasion_rl.osm_demo.coarsening`.

The policy is intentionally decoupled from the learned/baseline police policies
(created by task 3.3/4.3 in ``policies.py``): it lives in its own module so the
two work streams never overwrite each other.  It is built directly on the
finished :mod:`environment`, :mod:`coarsening` and :mod:`models` modules and is
fully runnable and testable offline.

Decision rule (Requirements 9.1-9.3, 14.1):

* Within a metric police *visibility* range, the fugitive maximizes separation
  from the visible police by moving toward the outgoing segment whose
  destination keeps the largest minimum distance from any visible officer.
* Otherwise it minimizes the directed distance to a reachable *escape*
  (``crosses_boundary``) segment along directed segment arc length.
* Action slots are considered in the environment's deterministic
  :func:`ordered_outgoing_segment_ids` order, and ties are broken by the lowest
  action index so two runs with the same seed are byte-for-byte reproducible.
* Random exploration / tie perturbation uses an *injected* seeded
  :class:`numpy.random.Generator`; no module-global randomness is used.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from .coarsening import ordered_outgoing_segment_ids
from .environment import STAY_ACTION
from .models import DomainValidationError, ModelNetwork, Segment

# An environment-compatible action provider signature:
# ``(agent_id, intersection_id, ordered_segment_ids, hop) -> action_index``.
ActionProvider = Callable[[str, int, Sequence[int], int], int]

_INFINITE = float("inf")


@dataclass(frozen=True, slots=True)
class FugitiveObservation:
    """The information the fugitive needs to choose one routing action.

    ``intersection_id`` is the decision node the fugitive currently rests on,
    ``ordered_segment_ids`` is the exact deterministic action ordering supplied
    by the environment (physical exits first, virtual continuations last), and
    ``police_xy`` are the metric positions of the six police captured at the
    start of the current step (constant across zero-time virtual microsteps).
    """

    intersection_id: int
    ordered_segment_ids: tuple[int, ...]
    police_xy: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "ordered_segment_ids", tuple(int(x) for x in self.ordered_segment_ids))
        object.__setattr__(
            self,
            "police_xy",
            tuple((float(px), float(py)) for px, py in self.police_xy),
        )


@dataclass(slots=True)
class _NetworkIndex:
    """Cached per-network lookups (immutable network => safe to memoize)."""

    segments: dict[int, Segment]
    positions: dict[int, tuple[float, float]]
    distance_to_escape: dict[int, float]


class OSMHeuristicFugitive:
    """Directed-network heuristic fugitive.

    Parameters
    ----------
    vision_range_m:
        Metric radius within which police are considered *visible*.  When any
        officer is inside this radius the fugitive switches from escape-seeking
        to separation-maximizing behavior.
    random_rate:
        Probability in ``[0, 1]`` of taking a uniformly random legal movement
        instead of the greedy choice, sampled from the injected generator.
    rng:
        Injected :class:`numpy.random.Generator` used for every random / random
        tie-breaking decision.  Supplying the same generator state yields
        identical action sequences (Requirements 9.8, 14.1).
    seed:
        Convenience seed used only when ``rng`` is not supplied.
    """

    def __init__(
        self,
        *,
        vision_range_m: float = 30.0,
        random_rate: float = 0.0,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
    ) -> None:
        vision = float(vision_range_m)
        if not math.isfinite(vision) or vision < 0.0:
            raise DomainValidationError(
                "INVALID_VISION_RANGE", "vision_range_m must be a nonnegative finite number"
            )
        rate = float(random_rate)
        if not 0.0 <= rate <= 1.0:
            raise DomainValidationError(
                "INVALID_RANDOM_RATE", "random_rate must be in [0, 1]", actual=random_rate
            )
        self.vision_range_m = vision
        self.random_rate = rate
        self._rng = rng if rng is not None else np.random.default_rng(seed)
        self._cache: dict[int, _NetworkIndex] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def act(self, network: ModelNetwork, observation: FugitiveObservation) -> int:
        """Return the action index for a single routing decision.

        The returned index selects into ``observation.ordered_segment_ids``; a
        return of :data:`STAY_ACTION` (or any out-of-range index) is interpreted
        by the environment as *stay*.
        """
        ordered = observation.ordered_segment_ids
        if not ordered:
            # No legal movement action exists: the environment logs a stay.
            return STAY_ACTION

        index = self._network_index(network)

        # Random exploration uses the injected generator only.
        if self.random_rate > 0.0 and float(self._rng.random()) < self.random_rate:
            return int(self._rng.integers(0, len(ordered)))

        current_xy = index.positions[observation.intersection_id]
        visible = [
            police
            for police in observation.police_xy
            if math.dist(police, current_xy) <= self.vision_range_m
        ]
        if visible:
            return self._flee(index, ordered, visible)
        return self._seek_escape(index, ordered)

    def env_provider(
        self, network: ModelNetwork, police_xy: Sequence[tuple[float, float]]
    ) -> ActionProvider:
        """Build an environment-compatible action provider.

        ``police_xy`` are captured once for the whole step; the environment
        replays them across every zero-time virtual routing microstep.
        """
        captured = tuple((float(px), float(py)) for px, py in police_xy)

        def provider(agent_id: str, intersection_id: int, ordered: Sequence[int], hop: int) -> int:
            observation = FugitiveObservation(
                intersection_id=intersection_id,
                ordered_segment_ids=tuple(ordered),
                police_xy=captured,
            )
            return self.act(network, observation)

        return provider

    # ------------------------------------------------------------------
    # Decision helpers
    # ------------------------------------------------------------------
    def _flee(
        self,
        index: _NetworkIndex,
        ordered: Sequence[int],
        visible: Sequence[tuple[float, float]],
    ) -> int:
        """Maximize the minimum distance from the destination to visible police."""
        best_index = 0
        best_score = -_INFINITE
        for slot, segment_id in enumerate(ordered):
            destination = index.positions[index.segments[segment_id].end_id]
            score = min(math.dist(destination, police) for police in visible)
            if score > best_score:  # strict keeps the lowest action index on ties
                best_score = score
                best_index = slot
        return best_index

    def _seek_escape(self, index: _NetworkIndex, ordered: Sequence[int]) -> int:
        """Minimize directed arc-length distance to a reachable escape segment."""
        best_index = 0
        best_cost = _INFINITE
        for slot, segment_id in enumerate(ordered):
            segment = index.segments[segment_id]
            if segment.crosses_boundary:
                cost = 0.0  # taking an escape segment ends the pursuit immediately
            else:
                downstream = index.distance_to_escape.get(segment.end_id, _INFINITE)
                cost = segment.length_m + downstream
            if cost < best_cost:  # strict keeps the lowest action index on ties
                best_cost = cost
                best_index = slot
        return best_index

    # ------------------------------------------------------------------
    # Network indexing
    # ------------------------------------------------------------------
    def _network_index(self, network: ModelNetwork) -> _NetworkIndex:
        cached = self._cache.get(id(network))
        if cached is not None:
            return cached
        segments = {segment.id: segment for segment in network.segments}
        positions = {item.id: item.position_xy for item in network.intersections}
        distance = _directed_distance_to_escape(network, segments)
        index = _NetworkIndex(segments=segments, positions=positions, distance_to_escape=distance)
        self._cache[id(network)] = index
        return index


def _directed_distance_to_escape(
    network: ModelNetwork, segments: Mapping[int, Segment]
) -> dict[int, float]:
    """Shortest directed arc-length distance from each intersection to an escape.

    An *escape entry* is any intersection with an outgoing ``crosses_boundary``
    segment (distance ``0`` there).  Distances propagate backward along directed
    segments (reverse-graph Dijkstra) so ``result[v]`` is the least directed
    distance from ``v`` to any escape entry.  Unreachable intersections are
    absent from the result and treated as ``+inf`` by callers.
    """
    reverse: dict[int, list[tuple[int, float]]] = {item.id: [] for item in network.intersections}
    escape_entries: set[int] = set()
    for segment in network.segments:
        reverse[segment.end_id].append((segment.start_id, segment.length_m))
        if segment.crosses_boundary:
            escape_entries.add(segment.start_id)

    distance: dict[int, float] = {node: 0.0 for node in escape_entries}
    heap: list[tuple[float, int]] = [(0.0, node) for node in escape_entries]
    heapq.heapify(heap)
    while heap:
        current_distance, node = heapq.heappop(heap)
        if current_distance > distance.get(node, _INFINITE):
            continue
        for predecessor, weight in reverse[node]:
            candidate = current_distance + weight
            if candidate < distance.get(predecessor, _INFINITE):
                distance[predecessor] = candidate
                heapq.heappush(heap, (candidate, predecessor))
    return distance


__all__ = (
    "ActionProvider",
    "FugitiveObservation",
    "OSMHeuristicFugitive",
)

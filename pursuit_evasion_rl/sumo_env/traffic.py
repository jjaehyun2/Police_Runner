"""Background traffic demand for the SUMO pursuit environment.

The abstract environment only ever had seven vehicles on the map, so
"congestion" had to be injected as an abstract per-segment speed.  Here the
congestion is caused: a few hundred ordinary vehicles drive their own trips,
and the pursuit happens in whatever traffic that produces.  This is the piece
the reference study got for free from its SUMO setup (200 background vehicles
in a 3x3 grid) and the reason its agents could perceive traffic at all.

Everything is derived deterministically from a seed so an episode replays
identically.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random

#: Per-road-class weights for choosing trip endpoints.  Trips concentrate on
#: the arterial classes, which is what makes congestion form where a driver
#: would expect it instead of uniformly across residential lanes.
ROAD_CLASS_WEIGHT: dict[str, float] = {
    "motorway": 6.0, "trunk": 5.0, "primary": 4.0, "secondary": 3.0,
    "tertiary": 2.0, "unclassified": 1.0, "residential": 0.6,
    "living_street": 0.2, "service": 0.2,
}
DEFAULT_WEIGHT = 1.0


@dataclass(frozen=True, slots=True)
class TrafficConfig:
    """Background demand for one episode.

    ``vehicle_count`` is the number of background trips inserted over
    ``insertion_window_s``; with the default 300 vehicles over 150 s the
    network reaches a steady population comparable to the reference study's
    200-vehicle 3x3 grid, scaled for a far larger map.
    ``congested_fraction`` picks that share of arterial edges and slows them
    with extra demand, so every episode has somewhere genuinely jammed.
    """

    vehicle_count: int = 300
    insertion_window_s: float = 150.0
    warmup_steps: int = 30
    congested_fraction: float = 0.05
    congestion_multiplier: float = 4.0
    min_trip_edges: int = 2
    seed: int = 0

    def __post_init__(self) -> None:
        if self.vehicle_count < 0:
            raise ValueError("vehicle_count must be non-negative")
        if self.insertion_window_s <= 0:
            raise ValueError("insertion_window_s must be positive")
        if not 0.0 <= self.congested_fraction <= 1.0:
            raise ValueError("congested_fraction must be within [0, 1]")
        if self.congestion_multiplier < 1.0:
            raise ValueError("congestion_multiplier must be at least 1.0")


def _edge_weight(edge) -> float:
    road_class = ""
    try:
        road_class = edge.getType() or ""
    except AttributeError:
        pass
    # netconvert stores the OSM class as e.g. "highway.primary".
    tail = road_class.split(".")[-1]
    return ROAD_CLASS_WEIGHT.get(tail, DEFAULT_WEIGHT)


def usable_edges(net) -> list:
    """Normal (non-internal) edges a passenger car may drive on."""
    edges = []
    for edge in net.getEdges():
        if edge.getID().startswith(":"):
            continue
        try:
            if not edge.allows("passenger"):
                continue
        except Exception:
            pass
        edges.append(edge)
    return edges


def write_background_routes(
    net, destination: str | Path, config: TrafficConfig
) -> tuple[Path, tuple[str, ...]]:
    """Write a SUMO route file of background trips; return it and the jammed edges.

    Trips are emitted as ``<trip>`` elements so SUMO routes them itself and
    reroutes around whatever congestion forms -- background drivers behave
    like drivers, not like scripted obstacles.
    """
    rng = random.Random(config.seed)
    edges = usable_edges(net)
    if len(edges) < config.min_trip_edges:
        raise ValueError("network has too few drivable edges for background traffic")
    weights = [_edge_weight(edge) for edge in edges]
    ids = [edge.getID() for edge in edges]

    arterials = [
        edge.getID() for edge, weight in zip(edges, weights) if weight >= ROAD_CLASS_WEIGHT["tertiary"]
    ] or ids
    jam_count = max(1, int(len(arterials) * config.congested_fraction)) if config.congested_fraction else 0
    congested = tuple(rng.sample(arterials, min(jam_count, len(arterials)))) if jam_count else ()
    congested_set = set(congested)

    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<routes>",
             '  <vType id="bg" vClass="passenger" speedFactor="normc(1.0,0.15,0.6,1.4)"'
             ' carFollowModel="Krauss" sigma="0.5"/>']
    emitted = 0
    for index in range(config.vehicle_count):
        depart = config.insertion_window_s * index / max(1, config.vehicle_count)
        origin = rng.choices(ids, weights=weights, k=1)[0]
        # Congested edges attract extra through-traffic, which is what jams them.
        if congested_set and rng.random() < config.congested_fraction * config.congestion_multiplier:
            target = rng.choice(tuple(congested_set))
        else:
            target = rng.choices(ids, weights=weights, k=1)[0]
        if target == origin:
            continue
        lines.append(
            '  <trip id="bg%d" type="bg" depart="%.2f" from="%s" to="%s"/>'
            % (index, depart, origin, target)
        )
        emitted += 1
    lines.append("</routes>")
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path, congested


def spawn_edges(net, *, count: int, seed: int, min_length_m: float = 40.0) -> list[str]:
    """Distinct, reasonably long edges to place pursuit vehicles on.

    Short edges are excluded because a vehicle inserted on a 10 m stub can be
    past its end before its first decision, which produced spurious
    zero-displacement steps in the abstract environment.
    """
    rng = random.Random(seed)
    candidates = [
        edge.getID() for edge in usable_edges(net) if edge.getLength() >= min_length_m
    ]
    if len(candidates) < count:
        candidates = [edge.getID() for edge in usable_edges(net)]
    if len(candidates) < count:
        raise ValueError(f"network cannot host {count} distinct spawn edges")
    return rng.sample(sorted(candidates), count)


def dispersed_spawn_edges(
    net, *, count: int, seed: int, min_separation_m: float = 300.0, min_length_m: float = 40.0
) -> list[str]:
    """Spawn edges spread across the map, so officers do not all start together.

    Falls back to fewer separation constraints rather than failing: on a dense
    network the greedy pass can exhaust candidates, and a slightly clustered
    start is better than refusing to build an episode.
    """
    rng = random.Random(seed)
    pool = [edge for edge in usable_edges(net) if edge.getLength() >= min_length_m]
    if len(pool) < count:
        pool = usable_edges(net)
    rng.shuffle(pool)
    chosen: list = []
    for edge in pool:
        x, y = edge.getShape()[0]
        if all(
            math.dist((x, y), (other.getShape()[0][0], other.getShape()[0][1])) >= min_separation_m
            for other in chosen
        ):
            chosen.append(edge)
        if len(chosen) == count:
            break
    while len(chosen) < count and pool:
        candidate = pool[len(chosen) % len(pool)]
        if candidate not in chosen:
            chosen.append(candidate)
        else:
            break
    if len(chosen) < count:
        raise ValueError(f"network cannot host {count} dispersed spawn edges")
    return [edge.getID() for edge in chosen[:count]]


__all__ = (
    "DEFAULT_WEIGHT",
    "ROAD_CLASS_WEIGHT",
    "TrafficConfig",
    "dispersed_spawn_edges",
    "spawn_edges",
    "usable_edges",
    "write_background_routes",
)
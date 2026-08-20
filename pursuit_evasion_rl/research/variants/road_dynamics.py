"""Dynamic road attributes for the pursuit environment (mentoring theme T1).

Per-episode, per-segment effective speeds derived from sealed OSM attributes
(``road_class`` 100% coverage, ``maxspeed`` tags where present), optionally
overlaid with live ITS speeds (``research.traffic.its_client``) and randomized
per episode so the policy cannot overfit one traffic pattern -- the extension
of the domain-randomization idea the mentoring review endorsed.

Design constraints (verified against the fail-closed registry):
- The sealed ``ModelNetwork`` and its ``network_content_hash`` are never
  touched: speeds live in a SIDE-CAR ``{segment_id: m/s}`` map passed through
  ``env.reset(options={"segment_speeds": ...})``.
- Everything is deterministic from ``(config, seed)`` so a Condition replays
  identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Mapping

import numpy as np

from pursuit_evasion_rl.osm_demo.models import DomainValidationError, ModelNetwork, Segment

ROAD_DYNAMICS_SCHEMA_VERSION = "1.0"

#: Statutory-style default speeds by OSM highway class (kph).  ``road_class``
#: is the only attribute with 100% coverage on the sealed Daejeon maps, so
#: this table is the PRIMARY source; explicit ``maxspeed`` tags (~25%
#: coverage) refine it per segment.
ROAD_CLASS_SPEED_KPH: Mapping[str, float] = {
    "motorway": 100.0, "motorway_link": 60.0,
    "trunk": 80.0, "trunk_link": 50.0,
    "primary": 60.0, "primary_link": 40.0,
    "secondary": 50.0, "secondary_link": 40.0,
    "tertiary": 50.0, "tertiary_link": 40.0,
    "unclassified": 40.0,
    "residential": 30.0,
    "living_street": 20.0,
    "service": 20.0,
    "busway": 40.0,
}
DEFAULT_SPEED_KPH = 40.0
KPH_TO_MPS = 1.0 / 3.6


def _maxspeed_kph(segment: Segment) -> float | None:
    """Best explicit maxspeed tag on any raw edge merged into this segment."""
    transitions = segment.attributes.get("transition_attributes") or ()
    best: float | None = None
    for item in transitions:
        raw = item.get("maxspeed") if isinstance(item, Mapping) else None
        if raw is None:
            continue
        try:
            value = float(str(raw).split()[0])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0 and (best is None or value < best):
            best = value
    return best


def base_segment_speed_mps(segment: Segment) -> float:
    """Static per-segment speed from sealed attributes (m/s)."""
    tagged = _maxspeed_kph(segment)
    if tagged is not None:
        return tagged * KPH_TO_MPS
    road_class = str(segment.attributes.get("road_class", ""))
    speeds = [
        ROAD_CLASS_SPEED_KPH[part]
        for part in road_class.split("|")
        if part in ROAD_CLASS_SPEED_KPH
    ]
    kph = min(speeds) if speeds else DEFAULT_SPEED_KPH
    return kph * KPH_TO_MPS


@dataclass(frozen=True, slots=True)
class RoadDynamicsConfig:
    """One dynamics arm: per-episode multiplicative speed randomization.

    ``min_speed_factor``/``max_speed_factor`` bound the uniform per-segment
    factor applied to the static base speed (1.0/1.0 = static speeds, no
    randomization).  ``speed_floor_mps`` prevents degenerate near-zero
    segments.  ``its_overlay`` (segment_id -> m/s) takes precedence over the
    sampled value where present -- the live-data hook; absent = pure
    simulation, honestly labeled.
    """

    min_speed_factor: float = 0.5
    max_speed_factor: float = 1.0
    speed_floor_mps: float = 1.5
    police_speed_cap_mps: float = 16.0
    its_overlay: Mapping[int, float] | None = None
    schema_version: str = ROAD_DYNAMICS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("min_speed_factor", "max_speed_factor", "speed_floor_mps", "police_speed_cap_mps"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise DomainValidationError(
                    "INVALID_ROAD_DYNAMICS", f"{name} must be positive and finite", path=name, actual=value,
                )
            object.__setattr__(self, name, value)
        if self.min_speed_factor > self.max_speed_factor:
            raise DomainValidationError(
                "INVALID_ROAD_DYNAMICS", "min_speed_factor must not exceed max_speed_factor",
            )
        if self.its_overlay is not None:
            cleaned = {}
            for key, value in dict(self.its_overlay).items():
                speed = float(value)
                if not math.isfinite(speed) or speed <= 0.0:
                    raise DomainValidationError(
                        "INVALID_ROAD_DYNAMICS", "its_overlay speeds must be positive finite m/s",
                        path=f"its_overlay[{key}]", actual=value,
                    )
                cleaned[int(key)] = speed
            object.__setattr__(self, "its_overlay", cleaned)


def sample_segment_speeds(
    network: ModelNetwork, config: RoadDynamicsConfig, *, seed: int
) -> dict[int, float]:
    """Deterministic per-episode side-car speed map (m/s), never mutating the network."""
    rng = np.random.default_rng(seed)
    speeds: dict[int, float] = {}
    for segment in sorted(network.segments, key=lambda s: s.id):
        if segment.virtual:
            continue
        base = base_segment_speed_mps(segment)
        factor = float(rng.uniform(config.min_speed_factor, config.max_speed_factor))
        speed = max(config.speed_floor_mps, base * factor)
        speeds[segment.id] = speed
    if config.its_overlay:
        speeds.update(config.its_overlay)
    return speeds


def effective_speed_mps(
    segment: Segment, speeds: Mapping[int, float] | None, *, cap_mps: float
) -> float:
    """The speed a police vehicle actually achieves on this segment."""
    if speeds is not None and segment.id in speeds:
        return min(cap_mps, float(speeds[segment.id]))
    return min(cap_mps, base_segment_speed_mps(segment))


def travel_time_weight(
    speeds: Mapping[int, float] | None, *, cap_mps: float
) -> Callable[[Segment], float]:
    """A ``_Graph``-compatible edge weight in SECONDS instead of meters."""

    def weight(segment: Segment) -> float:
        return float(segment.length_m) / effective_speed_mps(segment, speeds, cap_mps=cap_mps)

    return weight


__all__ = (
    "DEFAULT_SPEED_KPH",
    "KPH_TO_MPS",
    "ROAD_CLASS_SPEED_KPH",
    "ROAD_DYNAMICS_SCHEMA_VERSION",
    "RoadDynamicsConfig",
    "base_segment_speed_mps",
    "effective_speed_mps",
    "sample_segment_speeds",
    "travel_time_weight",
)
"""Live traffic speeds from the Korean national ITS open API (best effort).

Data source: ``openapi.its.go.kr`` real-time link speeds (국가교통정보센터
소통정보).  Requires a free service key from data.go.kr, supplied via the
``ITS_API_KEY`` environment variable or the ``api_key`` argument.

This module is deliberately fail-soft: no key, no network, malformed
response, or zero matched links all return ``None`` so the caller falls back
to the sealed-attribute + randomization path in ``variants.road_dynamics``.
Live overlay is an 실증-단계 integration -- results using it must be labeled
as such (repo scope rules forbid claiming live-traffic evaluation otherwise).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
import os
import urllib.parse
import urllib.request
from typing import Mapping, Sequence

from pursuit_evasion_rl.osm_demo.models import ModelNetwork

logger = logging.getLogger(__name__)

ITS_ENDPOINT = "https://openapi.its.go.kr:9443/trafficInfo"
API_KEY_ENV = "ITS_API_KEY"
KPH_TO_MPS = 1.0 / 3.6


@dataclass(frozen=True, slots=True)
class LinkSpeed:
    """One ITS link observation: WGS84 midpoint (if provided) + speed."""

    link_id: str
    speed_kph: float
    mid_x: float | None = None
    mid_y: float | None = None


def fetch_bbox_speeds(
    *,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    api_key: str | None = None,
    timeout_s: float = 10.0,
) -> tuple[LinkSpeed, ...] | None:
    """Fetch real-time link speeds inside a WGS84 bbox; ``None`` = unavailable."""
    key = api_key or os.environ.get(API_KEY_ENV)
    if not key:
        logger.info("ITS overlay skipped: no %s set", API_KEY_ENV)
        return None
    query = urllib.parse.urlencode(
        {
            "apiKey": key, "type": "all", "drcType": "all", "getType": "json",
            "minX": f"{min_x:.6f}", "maxX": f"{max_x:.6f}",
            "minY": f"{min_y:.6f}", "maxY": f"{max_y:.6f}",
        }
    )
    try:
        with urllib.request.urlopen(f"{ITS_ENDPOINT}?{query}", timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as error:  # fail-soft by design
        logger.warning("ITS overlay skipped: request failed (%s)", error)
        return None
    items = (((payload or {}).get("body") or {}).get("items")) or []
    speeds: list[LinkSpeed] = []
    for item in items:
        try:
            kph = float(item.get("speed"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(kph) or kph <= 0:
            continue
        def _coord(*names: str) -> float | None:
            for name in names:
                value = item.get(name)
                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return None
            return None
        sx, sy = _coord("startX"), _coord("startY")
        ex, ey = _coord("endX"), _coord("endY")
        mid_x = (sx + ex) / 2.0 if sx is not None and ex is not None else None
        mid_y = (sy + ey) / 2.0 if sy is not None and ey is not None else None
        speeds.append(LinkSpeed(str(item.get("linkId", "")), kph, mid_x, mid_y))
    if not speeds:
        logger.warning("ITS overlay skipped: response contained no usable links")
        return None
    return tuple(speeds)


def map_speeds_to_segments(
    network: ModelNetwork,
    link_speeds: Sequence[LinkSpeed],
    *,
    to_xy,
    max_match_m: float = 60.0,
) -> dict[int, float]:
    """Nearest-midpoint matching of ITS links onto network segments (m/s).

    ``to_xy`` projects a WGS84 (lon, lat) into the network's metric plane --
    supply the same projection the snapshot pipeline used.  Links without
    coordinates are skipped (matching by linkId alone would need the national
    link shapefile; that join is 실증-단계 work).
    """
    midpoints: list[tuple[int, float, float]] = []
    for segment in network.segments:
        if segment.virtual or not segment.geometry_xy:
            continue
        xs = [point[0] for point in segment.geometry_xy]
        ys = [point[1] for point in segment.geometry_xy]
        midpoints.append((segment.id, sum(xs) / len(xs), sum(ys) / len(ys)))
    matched: dict[int, float] = {}
    best: dict[int, float] = {}
    for link in link_speeds:
        if link.mid_x is None or link.mid_y is None:
            continue
        x, y = to_xy(link.mid_x, link.mid_y)
        for seg_id, sx, sy in midpoints:
            distance = math.hypot(sx - x, sy - y)
            if distance <= max_match_m and distance < best.get(seg_id, math.inf):
                best[seg_id] = distance
                matched[seg_id] = link.speed_kph * KPH_TO_MPS
    return matched


__all__ = ("API_KEY_ENV", "ITS_ENDPOINT", "LinkSpeed", "fetch_bbox_speeds", "map_speeds_to_segments")
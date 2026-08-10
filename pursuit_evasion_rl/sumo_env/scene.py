"""Scene export for the visual demo renderer.

SUMO-GUI draws an accurate traffic engineer's view: true-to-scale cars on a
whole-city extract, which at demo zoom is an empty-looking road map.  A
presentation needs the opposite -- exaggerated, legible symbols showing who is
where and what the team is doing.  So SUMO stays the physics engine and this
module exports what a custom renderer needs to draw that picture:

* ``export_network`` runs once and dumps the road polylines and junction
  positions in a local metric frame (origin at the network's lower-left).
* ``scene_frame`` runs every step and dumps vehicles, signal states, and the
  current dispatch assignments, so the renderer can show the encirclement
  forming rather than just dots moving.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from pursuit_evasion_rl.sumo_env.environment import (
    FUGITIVE_ID,
    POLICE_IDS,
    SumoPursuitEnv,
)

#: Road classes drawn wider, so arterials read as arterials at a glance.
CLASS_WIDTH: Mapping[str, float] = {
    "motorway": 9.0, "trunk": 8.0, "primary": 7.0, "secondary": 6.0,
    "tertiary": 5.0, "unclassified": 4.0, "residential": 3.5,
    "living_street": 3.0, "service": 3.0,
}
DEFAULT_WIDTH = 4.0


def _road_width(edge) -> float:
    road_class = ""
    try:
        road_class = (edge.getType() or "").split(".")[-1]
    except AttributeError:
        pass
    lanes = 1
    try:
        lanes = max(1, edge.getLaneNumber())
    except Exception:
        pass
    return CLASS_WIDTH.get(road_class, DEFAULT_WIDTH) * (1.0 + 0.25 * (lanes - 1))


def export_network(env: SumoPursuitEnv, destination: str | Path) -> dict[str, Any]:
    """Dump road geometry once; the renderer treats this as its static map."""
    net = env.sumolib_net
    xs: list[float] = []
    ys: list[float] = []
    raw: list[tuple[str, list[tuple[float, float]], float]] = []
    for edge in net.getEdges():
        edge_id = edge.getID()
        if edge_id.startswith(":"):
            continue
        shape = [(float(x), float(y)) for x, y in edge.getShape()]
        if len(shape) < 2:
            continue
        raw.append((edge_id, shape, _road_width(edge)))
        for x, y in shape:
            xs.append(x)
            ys.append(y)
    if not raw:
        raise RuntimeError("network has no drawable edges")
    min_x, min_y = min(xs), min(ys)
    max_x, max_y = max(xs), max(ys)

    roads = [
        {
            "id": edge_id,
            "w": round(width, 2),
            "pts": [[round(x - min_x, 1), round(y - min_y, 1)] for x, y in shape],
        }
        for edge_id, shape, width in raw
    ]
    signals = []
    for tls in net.getTrafficLights():
        try:
            node = net.getNode(tls.getID())
            x, y = node.getCoord()
        except Exception:
            continue
        signals.append({"id": tls.getID(), "x": round(x - min_x, 1), "y": round(y - min_y, 1)})

    payload = {
        "width": round(max_x - min_x, 1),
        "height": round(max_y - min_y, 1),
        "origin": [round(min_x, 1), round(min_y, 1)],
        "roads": roads,
        "signals": signals,
    }
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return payload


def _signal_colour(state: str) -> str:
    """Reduce a SUMO signal string to the one colour a viewer should read.

    A junction's state has one character per controlled connection, and a
    busy intersection almost always has *some* movement on green -- so
    "green if any g" paints every light in the city green and makes the
    signals look decorative.  Reporting the dominant phase instead gives the
    indicator the meaning a viewer expects: a junction shows red while most
    of its movements are stopped.
    """
    lowered = (state or "").lower()
    if not lowered:
        return "off"
    counts = {
        "green": lowered.count("g"),
        "yellow": lowered.count("y"),
        "red": lowered.count("r"),
    }
    if not any(counts.values()):
        return "off"
    # Yellow is a transient the eye should catch, so it wins any tie.
    if counts["yellow"] and counts["yellow"] >= max(counts["green"], counts["red"]):
        return "yellow"
    return "green" if counts["green"] > counts["red"] else "red"


def scene_frame(
    env: SumoPursuitEnv,
    origin: Sequence[float],
    *,
    assignments: Mapping[str, str] | None = None,
    max_background: int = 400,
) -> dict[str, Any]:
    """Per-step drawable scene: vehicles, signals, and dispatch arrows."""
    connection = env._connection
    ox, oy = float(origin[0]), float(origin[1])
    tracked = {FUGITIVE_ID, *POLICE_IDS}

    vehicles: list[dict[str, Any]] = []
    try:
        ids = connection.vehicle.getIDList()
    except Exception:
        ids = ()
    background_drawn = 0
    for vehicle_id in ids:
        try:
            x, y = connection.vehicle.getPosition(vehicle_id)
            angle = float(connection.vehicle.getAngle(vehicle_id))
            speed = float(connection.vehicle.getSpeed(vehicle_id))
        except Exception:
            continue
        if vehicle_id in tracked:
            kind = "fugitive" if vehicle_id == FUGITIVE_ID else "police"
        else:
            if background_drawn >= max_background:
                continue
            background_drawn += 1
            kind = "background"
        vehicles.append({
            "id": vehicle_id if vehicle_id in tracked else "",
            "k": kind,
            "x": round(x - ox, 1),
            "y": round(y - oy, 1),
            # SUMO reports a compass angle (0 = north, clockwise); the renderer
            # wants a maths angle, so convert once here rather than in JS.
            "a": round((90.0 - angle) % 360.0, 1),
            "v": round(speed * 3.6, 1),
        })

    signals = []
    try:
        for tls_id in connection.trafficlight.getIDList():
            signals.append({
                "id": tls_id,
                "c": _signal_colour(connection.trafficlight.getRedYellowGreenState(tls_id)),
            })
    except Exception:
        pass

    arrows = []
    if assignments:
        positions = {item["id"]: (item["x"], item["y"]) for item in vehicles if item["id"]}
        for officer_id, edge_id in assignments.items():
            if officer_id not in positions or not edge_id:
                continue
            try:
                shape = env.sumolib_net.getEdge(edge_id).getShape()
            except Exception:
                continue
            tx = sum(point[0] for point in shape) / len(shape) - ox
            ty = sum(point[1] for point in shape) / len(shape) - oy
            sx, sy = positions[officer_id]
            arrows.append({
                "o": officer_id,
                "x1": sx, "y1": sy,
                "x2": round(tx, 1), "y2": round(ty, 1),
            })

    fugitive = next((item for item in vehicles if item["k"] == "fugitive"), None)
    return {
        "vehicles": vehicles,
        "signals": signals,
        "arrows": arrows,
        "focus": [fugitive["x"], fugitive["y"]] if fugitive else None,
        "background_shown": background_drawn,
    }


__all__ = ("CLASS_WIDTH", "export_network", "scene_frame")
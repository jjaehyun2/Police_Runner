"""Small, committed offline OSM fixtures for reproducible default tests.

These reusable factories build tiny immutable :class:`RawOSMGraph` inputs that
exercise the directed coarsening and degree-splitting paths without any
external OSM service (Requirements 13.10-13.12).  Each fixture has a committed
content hash in :data:`FIXTURE_RAW_HASHES` so that a change in fixture
geometry, topology, or attributes is caught as a regression rather than
silently altering downstream canonical identities.

Fixtures provided:

``one_way`` -- a one-way non-branching chain that collapses to one segment.
``bidirectional`` -- a two-way road kept as two distinct directed segments.
``parallel_edge`` -- exact parallel edges that deduplicate to one segment
    while the mapping still records every source edge.
``degree_six`` -- a hub whose out-degree exceeds five and therefore requires
    deterministic degree splitting during :func:`prepare_model_network`.
``boundary_crossing`` -- a chain between two clipped boundary nodes.
``daejeon`` -- a small Daejeon-shaped downtown grid bound to the Daejeon
    drive preset area so it can back a :class:`FixtureOSMSource`.
"""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .canonical import content_hash
from .models import RawEdge, RawNode, RawOSMGraph
from .osm_source import FixtureOSMSource
from .presets import DAEJEON_DRIVE_PRESET

# Representative geographic anchor for fixture nodes.  The exact latitude and
# longitude are provenance only; coarsening uses the metric x/y coordinates.
_FIXTURE_LAT = 36.35
_FIXTURE_LON = 127.40


def _node(source_id: str, x: float, y: float = 0.0, **tags: Any) -> RawNode:
    return RawNode(source_id, _FIXTURE_LAT, _FIXTURE_LON, x, y, dict(tags))


def _edge(
    source_u: str,
    source_v: str,
    key: str,
    positions: Mapping[str, tuple[float, float]],
    *,
    oneway: bool = True,
    road_class: str = "residential",
    way_ref: str | None = None,
    **tags: Any,
) -> RawEdge:
    start, end = positions[source_u], positions[source_v]
    return RawEdge(
        source_u,
        source_v,
        key,
        length_m=math.dist(start, end),
        geometry_xy=(start, end),
        way_refs=(way_ref or key,),
        oneway=oneway,
        road_class=road_class,
        tags=dict(tags),
    )


# ---------------------------------------------------------------------------
# Individual fixture factories
# ---------------------------------------------------------------------------


def one_way() -> RawOSMGraph:
    """A one-way A -> middle -> Z chain collapsing to a single directed segment."""
    positions = {"a": (0.0, 0.0), "middle": (10.0, 0.0), "z": (25.0, 0.0)}
    nodes = tuple(_node(source_id, *position) for source_id, position in positions.items())
    edges = (
        _edge("a", "middle", "ow-1", positions, way_ref="way-10"),
        _edge("middle", "z", "ow-2", positions, way_ref="way-11"),
    )
    return RawOSMGraph(nodes=nodes, edges=edges)


def bidirectional() -> RawOSMGraph:
    """A two-way road preserved as two opposite directed segments."""
    positions = {"left": (0.0, 0.0), "middle": (10.0, 0.0), "right": (20.0, 0.0)}
    nodes = tuple(_node(source_id, *position) for source_id, position in positions.items())
    edges = (
        _edge("left", "middle", "bd-lm", positions, oneway=False),
        _edge("middle", "right", "bd-mr", positions, oneway=False),
        _edge("right", "middle", "bd-rm", positions, oneway=False),
        _edge("middle", "left", "bd-ml", positions, oneway=False),
    )
    return RawOSMGraph(nodes=nodes, edges=edges)


def parallel_edge() -> RawOSMGraph:
    """Two exact parallel edges that deduplicate to one canonical segment."""
    positions = {"start": (0.0, 0.0), "end": (10.0, 0.0)}
    nodes = tuple(_node(source_id, *position) for source_id, position in positions.items())
    edges = (
        _edge("start", "end", "parallel-a", positions, way_ref="way-a"),
        _edge("start", "end", "parallel-b", positions, way_ref="way-b"),
    )
    return RawOSMGraph(nodes=nodes, edges=edges)


def degree_six() -> RawOSMGraph:
    """A hub with six one-way exits, forcing deterministic degree splitting."""
    spokes = 6
    radius = 10.0
    positions: dict[str, tuple[float, float]] = {"hub": (0.0, 0.0)}
    for index in range(spokes):
        angle = math.radians(60.0 * index)
        positions[f"spoke-{index}"] = (
            round(radius * math.cos(angle), 6),
            round(radius * math.sin(angle), 6),
        )
    nodes = tuple(_node(source_id, *position) for source_id, position in positions.items())
    edges = tuple(
        _edge("hub", f"spoke-{index}", f"d6-{index}", positions)
        for index in range(spokes)
    )
    return RawOSMGraph(nodes=nodes, edges=edges)


def boundary_crossing() -> RawOSMGraph:
    """A directed chain between two clipped bbox boundary nodes."""
    positions = {"edge-in": (0.0, 0.0), "middle": (10.0, 0.0), "edge-out": (25.0, 0.0)}
    nodes = (
        _node("edge-in", *positions["edge-in"], boundary_clipped=True),
        _node("middle", *positions["middle"]),
        _node("edge-out", *positions["edge-out"], boundary_clipped=True),
    )
    edges = (
        _edge("edge-in", "middle", "bc-1", positions),
        _edge("middle", "edge-out", "bc-2", positions),
    )
    return RawOSMGraph(nodes=nodes, edges=edges)


def daejeon() -> RawOSMGraph:
    """A small Daejeon-shaped downtown grid of two-way streets (3x3 blocks)."""
    spacing = 100.0
    positions: dict[str, tuple[float, float]] = {}
    for row in range(3):
        for col in range(3):
            positions[f"n{row}{col}"] = (col * spacing, row * spacing)
    nodes = tuple(_node(source_id, *position) for source_id, position in positions.items())

    edges: list[RawEdge] = []
    for row in range(3):
        for col in range(3):
            here = f"n{row}{col}"
            if col + 1 < 3:
                right = f"n{row}{col + 1}"
                edges.append(_edge(here, right, f"h-{here}-{right}", positions, oneway=False, road_class="secondary"))
                edges.append(_edge(right, here, f"h-{right}-{here}", positions, oneway=False, road_class="secondary"))
            if row + 1 < 3:
                down = f"n{row + 1}{col}"
                edges.append(_edge(here, down, f"v-{here}-{down}", positions, oneway=False, road_class="secondary"))
                edges.append(_edge(down, here, f"v-{down}-{here}", positions, oneway=False, road_class="secondary"))
    return RawOSMGraph(nodes=nodes, edges=tuple(edges))


# ---------------------------------------------------------------------------
# Registry, committed identity hashes and reusable sources
# ---------------------------------------------------------------------------

OFFLINE_FIXTURES: Mapping[str, Callable[[], RawOSMGraph]] = MappingProxyType(
    {
        "one_way": one_way,
        "bidirectional": bidirectional,
        "parallel_edge": parallel_edge,
        "degree_six": degree_six,
        "boundary_crossing": boundary_crossing,
        "daejeon": daejeon,
    }
)

# Committed content hashes; a mismatch means a fixture changed and every
# dependent canonical identity/test baseline must be reviewed intentionally.
FIXTURE_RAW_HASHES: Mapping[str, str] = MappingProxyType(
    {
        "one_way": "806bb380efc63c3f673e69adae073dda0088394878a565175b6a313329fec01b",
        "bidirectional": "218ba578d014692bb18c940d11b7064a465ecfba39ff28c41b1f6d8b50ebc14a",
        "parallel_edge": "69b9529125b0ffa89b38906ec15cd775749d21d56aeb86ad66a823f22896c779",
        "degree_six": "94d560b18cf4a91bd75af3f4c7a75e487c8fc1ee2b9b7e53b849af543703c2df",
        "boundary_crossing": "ddc243105e2557dbff2e6a353969b01026187fc3656098851b7eed616fd58d8e",
        "daejeon": "457901044ab8c4d9c405097c2e8301c758c8610b17db92553a07c055470dabb4",
    }
)


def build_fixture(name: str) -> RawOSMGraph:
    """Return a fresh immutable raw graph for the named offline fixture."""
    try:
        factory = OFFLINE_FIXTURES[name]
    except KeyError as exc:
        raise KeyError(f"Unknown offline OSM fixture {name!r}") from exc
    return factory()


def fixture_raw_hash(name: str) -> str:
    """Return the committed content hash for the named offline fixture."""
    return FIXTURE_RAW_HASHES[name]


def daejeon_fixture_source() -> FixtureOSMSource:
    """A :class:`FixtureOSMSource` backing the Daejeon preset area offline."""
    return FixtureOSMSource({DAEJEON_DRIVE_PRESET.area.name: daejeon()})


__all__ = (
    "OFFLINE_FIXTURES",
    "FIXTURE_RAW_HASHES",
    "one_way",
    "bidirectional",
    "parallel_edge",
    "degree_six",
    "boundary_crossing",
    "daejeon",
    "build_fixture",
    "fixture_raw_hash",
    "daejeon_fixture_source",
)

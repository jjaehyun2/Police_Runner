"""OSM cache -> SUMO road network (mentoring T1 / SUMO transition).

Converts the Overpass JSON snapshots this repo already ships in ``cache/``
into an OSM XML file and runs ``netconvert`` on it, producing a SUMO network
that covers the same Daejeon roads the abstract environment used.  The result
is cached on disk and keyed by a content hash of the source file plus the
netconvert options, so repeated runs are free and reproducible.

Why a converter instead of a checked-in ``.net.xml``: the sealed map registry
identifies networks by content hash, and a generated artifact that nobody can
re-derive would break that discipline.  Anyone can re-run this and get the
same bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import xml.sax.saxutils as saxutils

#: netconvert options that materially shape the produced network.  They enter
#: the cache key, so changing one produces a different network directory
#: rather than silently overwriting the previous one.
DEFAULT_NETCONVERT_OPTIONS: tuple[str, ...] = (
    "--geometry.remove",
    "--roundabouts.guess",
    "--ramps.guess",
    "--junctions.join",
    "--tls.guess-signals",
    "--tls.discard-simple",
    "--tls.join",
    "--no-turnarounds", "true",
)

#: The cached Overpass responses carry ways but not the node tags that mark
#: signalised junctions, so netconvert finds zero traffic lights.  Callers that
#: want signals add this and SUMO synthesises them at qualifying junctions.
GUESS_TRAFFIC_LIGHTS: tuple[str, ...] = ("--tls.guess", "true")


class SumoToolchainError(RuntimeError):
    """Raised when the SUMO binaries are unavailable or netconvert fails."""


def sumo_home() -> Path:
    """Locate the SUMO installation (``SUMO_HOME`` or the pip ``sumo`` package)."""
    env_home = os.environ.get("SUMO_HOME")
    if env_home and (Path(env_home) / "bin").is_dir():
        return Path(env_home)
    try:
        import sumo as sumo_pkg  # provided by the ``eclipse-sumo`` wheel
    except ImportError as error:
        raise SumoToolchainError(
            "SUMO not found: set SUMO_HOME or `pip install eclipse-sumo traci sumolib`"
        ) from error
    home = Path(sumo_pkg.__file__).parent
    if not (home / "bin").is_dir():
        raise SumoToolchainError(f"SUMO package at {home} has no bin/ directory")
    os.environ.setdefault("SUMO_HOME", str(home))
    return home


def sumo_binary(name: str) -> str:
    """Absolute path to a SUMO executable (``sumo``, ``sumo-gui``, ``netconvert``)."""
    home = sumo_home()
    for candidate in (home / "bin" / f"{name}.exe", home / "bin" / name):
        if candidate.exists():
            return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    raise SumoToolchainError(f"SUMO binary {name!r} not found under {home}")


def _digest(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:12]


def overpass_to_osm_xml(source: Path, destination: Path) -> tuple[int, int]:
    """Rewrite an Overpass JSON response as the OSM XML netconvert expects.

    Returns ``(node_count, way_count)``.  Only nodes and ways are emitted --
    relations carry turn restrictions we do not model, and including them
    without their members would make netconvert reject the file.
    """
    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    elements = payload.get("elements", payload if isinstance(payload, list) else [])
    nodes = [item for item in elements if item.get("type") == "node"]
    ways = [item for item in elements if item.get("type") == "way"]
    if not nodes or not ways:
        raise SumoToolchainError(f"{source} holds no OSM nodes/ways to convert")
    lines = [
        "<?xml version='1.0' encoding='UTF-8'?>",
        "<osm version='0.6' generator='police-runner-net-builder'>",
    ]
    for node in nodes:
        header = "  <node id='%s' lat='%s' lon='%s' version='1'" % (
            node["id"], node["lat"], node["lon"]
        )
        tags = node.get("tags") or {}
        if not tags:
            lines.append(header + "/>")
            continue
        lines.append(header + ">")
        for key, value in tags.items():
            lines.append(
                "    <tag k=%s v=%s/>"
                % (saxutils.quoteattr(str(key)), saxutils.quoteattr(str(value)))
            )
        lines.append("  </node>")
    for way in ways:
        lines.append("  <way id='%s' version='1'>" % way["id"])
        for reference in way.get("nodes", []):
            lines.append("    <nd ref='%s'/>" % reference)
        for key, value in (way.get("tags") or {}).items():
            lines.append(
                "    <tag k=%s v=%s/>"
                % (saxutils.quoteattr(str(key)), saxutils.quoteattr(str(value)))
            )
        lines.append("  </way>")
    lines.append("</osm>")
    Path(destination).write_text("\n".join(lines), encoding="utf-8")
    return len(nodes), len(ways)


@dataclass(frozen=True, slots=True)
class SumoNetwork:
    """A built SUMO network plus the provenance needed to re-derive it."""

    net_path: Path
    osm_path: Path
    source_path: Path
    source_sha256: str
    options: tuple[str, ...]
    edge_count: int
    node_count: int
    traffic_light_count: int

    @property
    def cache_key(self) -> str:
        return _digest(self.source_sha256, " ".join(self.options))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_network(
    source: str | Path,
    output_root: str | Path = "cache/sumo",
    *,
    options: tuple[str, ...] = DEFAULT_NETCONVERT_OPTIONS,
    guess_traffic_lights: bool = True,
    force: bool = False,
) -> SumoNetwork:
    """Build (or reuse) the SUMO network for one Overpass JSON snapshot."""
    source = Path(source)
    if not source.is_file():
        raise SumoToolchainError(f"OSM source not found: {source}")
    effective = tuple(options) + (GUESS_TRAFFIC_LIGHTS if guess_traffic_lights else ())
    source_hash = _sha256_file(source)
    directory = Path(output_root) / _digest(source_hash, " ".join(effective))
    net_path = directory / "network.net.xml"
    osm_path = directory / "source.osm.xml"
    if net_path.is_file() and not force:
        return _describe(net_path, osm_path, source, source_hash, effective)
    directory.mkdir(parents=True, exist_ok=True)
    overpass_to_osm_xml(source, osm_path)
    command = [sumo_binary("netconvert"), "--osm-files", str(osm_path),
               "--output-file", str(net_path), *effective]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0 or not net_path.is_file():
        raise SumoToolchainError(
            f"netconvert failed (rc={completed.returncode}): {completed.stderr[-800:]}"
        )
    return _describe(net_path, osm_path, source, source_hash, effective)


def _describe(
    net_path: Path, osm_path: Path, source: Path, source_hash: str, options: tuple[str, ...]
) -> SumoNetwork:
    import sumolib  # imported lazily so the module is importable without SUMO

    net = sumolib.net.readNet(str(net_path))
    return SumoNetwork(
        net_path=net_path,
        osm_path=osm_path,
        source_path=source,
        source_sha256=source_hash,
        options=options,
        edge_count=len(net.getEdges()),
        node_count=len(net.getNodes()),
        traffic_light_count=len(net.getTrafficLights()),
    )


def largest_cached_snapshot(cache_root: str | Path = "cache") -> Path:
    """The biggest Overpass response in ``cache/`` -- the widest Daejeon extract."""
    candidates = [item for item in Path(cache_root).glob("*.json") if item.stat().st_size > 100_000]
    if not candidates:
        raise SumoToolchainError(f"no Overpass snapshots found under {cache_root}")
    return max(candidates, key=lambda item: item.stat().st_size)


__all__ = (
    "DEFAULT_NETCONVERT_OPTIONS",
    "GUESS_TRAFFIC_LIGHTS",
    "SumoNetwork",
    "SumoToolchainError",
    "build_network",
    "largest_cached_snapshot",
    "overpass_to_osm_xml",
    "sumo_binary",
    "sumo_home",
)
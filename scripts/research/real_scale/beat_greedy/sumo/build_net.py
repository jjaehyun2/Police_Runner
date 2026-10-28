"""ModelNetwork -> SUMO network, one-to-one.

Every physical intersection becomes SUMO node ``n<id>`` and every segment becomes SUMO edge
``e<id>`` with the segment's own geometry, in the network's metric CRS (UTM) with offset
normalisation disabled, so SUMO positions equal ModelNetwork positions and the two road graphs
are the same graph. (The sealed maps used here have no virtual degree-split elements; the
builder refuses networks that do.)

profile "ideal"   all edges 1 lane, speed limit above vehicle speeds, no signals, no turn-speed
                  limits: closest to the point-mass simulator (stage 1 calibration)
profile "real"    OSM lanes and maxspeed (default by road class), signals guessed at major
                  junctions, SUMO's default turn-speed limits (stage 2)

Usage: python sumo/build_net.py <map> [ideal|real]   -> sumo/nets/<map>_<profile>.net.xml
       python sumo/build_net.py <map> real --traffic <period_s>   -> + background demand
                (randomTrips + duarouter, one trip every period_s seconds, 1200 s)
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from xml.sax.saxutils import quoteattr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import common  # noqa: E402

NETS = HERE / "nets"
DEFAULT_KPH = {"motorway": 100, "trunk": 80, "primary": 60, "secondary": 50, "tertiary": 50,
               "unclassified": 40, "residential": 30, "living_street": 20, "service": 20}


def sumo_bin(name: str) -> str:
    import sumo
    return str(Path(sumo.SUMO_HOME) / "bin" / name)


def _kph(seg) -> float:
    attrs = dict(seg.attributes)
    for t in attrs.get("transition_attributes", ()):
        raw = dict(t).get("maxspeed")
        if raw:
            raw = raw[0] if isinstance(raw, (list, tuple)) else raw
            try:
                return float(str(raw).split()[0])
            except ValueError:
                pass
    full = str(attrs.get("road_class", "residential"))
    cls = full.split("_")[0]
    if os.environ.get("RT_CLASS_FIX", "0") == "1" and full in DEFAULT_KPH:   # see roadtime.CLASS_FIX
        cls = full
    return float(DEFAULT_KPH.get(cls, 30))


def _lanes(seg) -> int:
    for t in dict(seg.attributes).get("transition_attributes", ()):
        raw = dict(t).get("lanes")
        if raw:
            raw = raw[0] if isinstance(raw, (list, tuple)) else raw
            try:
                n = int(float(str(raw).split(";")[0]))
                return max(1, min(n // (1 if dict(seg.attributes).get("oneway") else 2) or 1, 4))
            except ValueError:
                pass
    return 1


def build(map_name: str, profile: str = "ideal") -> Path:
    net = common.load_network(map_name)
    if any(i.virtual for i in net.intersections) or any(s.virtual for s in net.segments):
        raise SystemExit(f"{map_name}: virtual elements present; one-to-one export not supported")
    NETS.mkdir(parents=True, exist_ok=True)
    stem = NETS / f"{map_name}_{profile}"
    with open(f"{stem}.nod.xml", "w") as f:
        f.write("<nodes>\n")
        for i in net.intersections:
            x, y = i.position_xy
            f.write(f'  <node id="n{i.id}" x="{x:.3f}" y="{y:.3f}" type="{"priority" if profile == "ideal" else "priority"}"/>\n')
        f.write("</nodes>\n")
    with open(f"{stem}.edg.xml", "w") as f:
        f.write("<edges>\n")
        for s in net.segments:
            if s.start_id == s.end_id:                       # netconvert drops self-loop edges; skip them here too
                continue
            shape = " ".join(f"{x:.3f},{y:.3f}" for x, y in s.geometry_xy)
            if profile == "ideal":
                speed, lanes = 50.0, 1
            else:
                speed, lanes = _kph(s) / 3.6, _lanes(s)
            f.write(f'  <edge id="e{s.id}" from="n{s.start_id}" to="n{s.end_id}" numLanes="{lanes}" '
                    f'speed="{speed:.2f}" shape={quoteattr(shape)}/>\n')
        f.write("</edges>\n")
    if True:
        # every turn the point-mass simulator allows (incl. U-turns) is a SUMO connection; in the
        # "real" profile only the edge pair is fixed and netconvert assigns the lanes
        incoming: dict[int, list[int]] = {}
        loops = {s.id for s in net.segments if s.start_id == s.end_id}
        for s in net.segments:
            if s.id not in loops:
                incoming.setdefault(s.end_id, []).append(s.id)
        with open(f"{stem}.con.xml", "w") as f:
            f.write("<connections>\n")
            for s in net.segments:
                if s.id in loops:
                    continue
                for sid in incoming.get(s.start_id, []):
                    lanes = ' fromLane="0" toLane="0"' if profile == "ideal" else ""
                    f.write(f'  <connection from="e{sid}" to="e{s.id}"{lanes}/>\n')
            f.write("</connections>\n")
    # SE_OPPOSITES=1 (real profile): opposite-direction lanes are linked so vehicles may overtake slow traffic on
    # two-way single-lane roads; written to <map>_<profile>_ovt.net.xml, leaving the default net untouched
    ovt = os.environ.get("SE_OPPOSITES") == "1" and profile != "ideal"
    out_net = f"{stem}_ovt.net.xml" if ovt else f"{stem}.net.xml"
    args = [sumo_bin("netconvert"), "--node-files", f"{stem}.nod.xml", "--edge-files", f"{stem}.edg.xml",
            "--output-file", out_net, "--offset.disable-normalization", "true",
            "--geometry.remove", "false", "--no-warnings", "true"]
    if profile == "ideal":
        args += ["--connection-files", f"{stem}.con.xml", "--junctions.limit-turn-speed", "-1", "--tls.discard-loaded", "true",
                 "--default.junctions.radius", "1", "--junctions.corner-detail", "0"]
    else:
        args += ["--connection-files", f"{stem}.con.xml", "--tls.guess", "true", "--tls.guess.threshold", "40",
                 "--tls.join", "true", "--junctions.join", "false"]
        if ovt:
            args += ["--opposites.guess", "true", "--opposites.guess.fix-lengths", "true"]
    subprocess.run(args, check=True)
    return Path(out_net)


def traffic(map_name: str, profile: str, period_s: float, end_s: float = 1200.0, seed: int = 0) -> Path:
    """Background traffic: random trips (fringe-weighted, so through traffic crosses the map) routed by
    duarouter. -> sumo/nets/<map>_<profile>_traffic<period>.rou.xml"""
    import sumo
    net_file = NETS / f"{map_name}_{profile}.net.xml"
    out = NETS / f"{map_name}_{profile}_traffic{period_s:g}.rou.xml"
    tools = Path(sumo.SUMO_HOME) / "tools" / "randomTrips.py"
    env = dict(os.environ, SUMO_HOME=sumo.SUMO_HOME, PATH=f"{Path(sumo.SUMO_HOME) / 'bin'}:{os.environ['PATH']}")
    subprocess.run([sys.executable, str(tools), "-n", str(net_file), "-o", str(out.with_suffix(".trips.xml")),
                    "-r", str(out), "-b", "0", "-e", str(end_s), "-p", str(period_s), "--fringe-factor", "5",
                    "--min-distance", "600", "--seed", str(seed), "--validate", "--prefix", "bg",
                    "--trip-attributes", 'departLane="best" departSpeed="max"'],
                   check=True, env=env, stdout=subprocess.DEVNULL)
    return out


if __name__ == "__main__":
    if "--traffic" in sys.argv:
        print(traffic(sys.argv[1], sys.argv[2], float(sys.argv[sys.argv.index("--traffic") + 1])))
    else:
        print(build(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "ideal"))

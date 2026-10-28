"""Seal a new map snapshot from the OSM editing API (api.openstreetmap.org /map) instead of Overpass.

Same pipeline as ``pursuit_evasion_rl.research.cli.fetch_osm_snapshot`` (OSMnxSource -> convert_osm_graph
-> coarsen -> prepare_model_network -> immutable snapshot + sealed config + provenance); only the download
differs: the bbox is fetched as OSM XML and turned into the graph OSMnx's graph_from_bbox would build
(osmnx's own "drive" way filter, truncate_by_edge, simplify, retain_all). Used when Overpass is unreachable.

Usage: python fetch_osm_api.py <map_name> <city> <south> <west> [size_deg=0.03] [role=validation]
Env OSM_TILES=k fetches the bbox as k x k tiles (needed above ~0.04 deg in dense cities).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import os
import re
import subprocess
import sys
import tempfile

import yaml

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

from pursuit_evasion_rl.osm_demo import BoundedArea, OSMnxSource, coarsen_raw_graph, prepare_model_network  # noqa: E402
from pursuit_evasion_rl.research.canonical import canonical_data, canonical_json, content_hash  # noqa: E402
from pursuit_evasion_rl.research.cli.fetch_osm_snapshot import _coordinate_pairs, run_integration  # noqa: E402

API = "https://api.openstreetmap.org/api/0.6/map?bbox={w},{s},{e},{n}"
# osmnx 2.1.1 _overpass._get_network_filter("drive"), as tag -> excluded-value regex
EXCLUDE = {"area": "yes", "access": "private",
           "highway": "abandoned|bridleway|bus_guideway|construction|corridor|cycleway|elevator|escalator|footway|no|"
                      "path|pedestrian|planned|platform|proposed|raceway|razed|rest_area|service|services|steps|track",
           "motor_vehicle": "no", "motorcar": "no",
           "service": "alley|driveway|emergency_access|parking|parking_aisle|private"}


class _ApiOsmnx:
    """The two osmnx calls OSMnxSource makes, backed by an OSM API download."""

    def __init__(self, cache: Path) -> None:
        import osmnx as ox
        self.ox, self.cache = ox, cache
        self.project_graph = ox.project_graph

    def graph_from_bbox(self, bbox, *, network_type, simplify, retain_all, truncate_by_edge):
        assert network_type == "drive"
        ox = self.ox
        import networkx as nx
        w, s, e, n = bbox
        # the API caps one /map call at 50,000 nodes: large bboxes are fetched as k x k tiles (OSM_TILES) and
        # composed. A tile returns every way touching it with all its nodes, so roads across tile edges join up.
        k = int(os.environ.get("OSM_TILES", "1"))
        g = None
        for i in range(k):
            for j in range(k):
                tw, te = w + (e - w) * i / k, w + (e - w) * (i + 1) / k
                ts, tn = s + (n - s) * j / k, s + (n - s) * (j + 1) / k
                f = self.cache / f"map_{i}_{j}.osm"
                subprocess.run(["curl", "-sf", "--retry", "3", "--retry-delay", "10", "--retry-all-errors", "-A", "pursuit-research/1.0", "-o", str(f),
                                API.format(w=round(tw, 6), s=round(ts, 6), e=round(te, 6), n=round(tn, 6))],
                               check=True, timeout=600)
                try:
                    t = ox.graph_from_xml(f, simplify=False, retain_all=True)
                except ox._errors.InsufficientResponseError:     # an all-sea / empty tile
                    continue
                g = t if g is None else nx.compose(g, t)

        def keep(d):
            if "highway" not in d:
                return False
            for tag, rx in EXCLUDE.items():
                v = d.get(tag)
                vals = v if isinstance(v, list) else ([] if v is None else [v])
                if any(re.search(rx, str(x)) for x in vals):
                    return False
            return True
        g.remove_edges_from([(u, v, k) for u, v, k, d in g.edges(keys=True, data=True) if not keep(d)])
        g.remove_nodes_from([x for x in list(g) if g.degree(x) == 0])
        g = ox.truncate.truncate_graph_bbox(g, bbox, truncate_by_edge=truncate_by_edge)
        if simplify:
            g = ox.simplify_graph(g)
        if not retain_all:
            g = ox.truncate.largest_component(g, strongly=False)
        return g


def main() -> int:
    a = sys.argv[1:]
    name, city, south, west = a[0], a[1], float(a[2]), float(a[3])
    size = float(a[4]) if len(a) > 4 else 0.03
    role = a[5] if len(a) > 5 else "validation"
    n, e = round(south + size, 6), round(west + size, 6)
    geometry = canonical_data({"type": "Polygon", "coordinates": [[[west, south], [e, south], [e, n], [west, n], [west, south]]]})
    poly = content_hash(geometry)
    tmp = Path(tempfile.mkdtemp())
    spec = {"schema_version": "1.0", "map_id": f"{name.replace('_', '-')}-v1", "city": city, "role": role,
            "query": "bounded OSM API 0.6 /map download over the recorded bbox, OSMnx drive filter (Overpass unreachable)",
            "query_geometry": geometry, "source": "OpenStreetMap contributors via OSM API 0.6 + OSMnx",
            "service": "OSM API 0.6", "service_version": "pending", "acquired_at_utc": "2026-01-01T00:00:00+00:00",
            "preprocessing_version": "pending", "metric_crs": "EPSG:32652",
            "raw_path": f"cache/research/maps/{poly}/raw_osm.json", "network_path": f"cache/research/maps/{poly}/model_network.json",
            "raw_file_hash": None, "network_file_hash": None, "network_content_hash": None, "polygon_content_hash": poly}
    cfg = tmp / f"{name}.unsealed.yaml"
    cfg.write_text(yaml.safe_dump(spec, sort_keys=False))
    shim = _ApiOsmnx(tmp)

    def transport(request):
        lons, lats = zip(*_coordinate_pairs(request.query_geometry.get("coordinates")))
        area = BoundedArea(name=request.integration_run_id, north=max(lats), south=min(lats), east=max(lons),
                           west=min(lons), max_area_km2=1000.0)
        acq = OSMnxSource(osmnx_module=shim).fetch_with_metadata(area, "drive")
        return canonical_json(acq.graph), canonical_json(prepare_model_network(coarsen_raw_graph(acq.graph)).network)

    args = argparse.Namespace(config=cfg, root=REPO, run_id=f"fetch-{name}-osmapi", service_version="cgimap (OSM API 0.6)",
                              sealed_config_output=Path(f"configs/research/maps/{name}.yaml"), provenance_output=None,
                              opt_in=True)
    rec = run_integration(args, transport=transport)
    print(rec["outputs"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

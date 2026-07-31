"""Explicit opt-in command for publishing immutable external OSM snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Callable

import yaml

from pursuit_evasion_rl.osm_demo import (
    BoundedArea,
    COARSENER_VERSION,
    DEGREE_SPLITTER_VERSION,
    OSMnxSource,
    coarsen_raw_graph,
    prepare_model_network,
)

from ..canonical import canonical_json, content_hash
from ..errors import ResearchValidationError
from ..maps.snapshots import (
    ExternalFetchRequest,
    ExternalOSMFetchAdapter,
    OfflineSnapshotStore,
    load_snapshot_spec,
    write_immutable_snapshot,
)


def _coordinate_pairs(value: Any) -> list[tuple[float, float]]:
    if isinstance(value, (list, tuple)) and len(value) >= 2 and all(
        isinstance(item, (int, float)) for item in value[:2]
    ):
        return [(float(value[0]), float(value[1]))]
    if isinstance(value, (list, tuple)):
        return [point for item in value for point in _coordinate_pairs(item)]
    return []


def _live_transport(request: ExternalFetchRequest) -> tuple[bytes, bytes]:
    points = _coordinate_pairs(request.query_geometry.get("coordinates"))
    if not points:
        raise ResearchValidationError(
            "INVALID_QUERY_GEOMETRY", "query geometry contains no coordinate pairs"
        )
    longitudes, latitudes = zip(*points)
    area = BoundedArea(
        name=request.integration_run_id, north=max(latitudes), south=min(latitudes),
        east=max(longitudes), west=min(longitudes), max_area_km2=1000.0,
    )
    acquisition = OSMnxSource().fetch_with_metadata(area, "drive")
    prepared = prepare_model_network(coarsen_raw_graph(acquisition.graph))
    return canonical_json(acquisition.graph), canonical_json(prepared.network)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="unsealed map YAML describing query geometry")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--run-id", required=True, help="separate integration execution identifier")
    parser.add_argument("--service-version", required=True, help="observed external service version")
    parser.add_argument(
        "--sealed-config-output", type=Path, required=True,
        help="new configs/research/maps YAML path; existing files are never replaced",
    )
    parser.add_argument(
        "--provenance-output", type=Path,
        help="immutable provenance JSON path (default: artifacts/research/maps/integration/<run-id>.json)",
    )
    parser.add_argument(
        "--opt-in", action="store_true",
        help="required acknowledgement that this command may access external OSM",
    )
    return parser


def run_integration(
    args: argparse.Namespace, *,
    transport: Callable[[ExternalFetchRequest], tuple[bytes, bytes]] | None = None,
) -> dict[str, Any]:
    root = args.root.resolve()
    spec = load_snapshot_spec(args.config if args.config.is_absolute() else root / args.config)
    store = OfflineSnapshotStore(root)
    raw_path, network_path = store._resolve(spec.raw_path), store._resolve(spec.network_path)
    config_output = (root / args.sealed_config_output).resolve()
    config_root = (root / "configs/research/maps").resolve()
    if config_output.parent != config_root:
        raise ResearchValidationError(
            "SNAPSHOT_CONFIG_PATH_ESCAPE", "sealed config must be a direct map-config child",
            path=str(args.sealed_config_output),
        )
    provenance_output = (
        (root / args.provenance_output).resolve() if args.provenance_output
        else root / "artifacts/research/maps/integration" / f"{args.run_id}.json"
    )
    for output in (raw_path, network_path, config_output, provenance_output):
        if output.exists():
            raise ResearchValidationError(
                "IMMUTABLE_SNAPSHOT_EXISTS", "integration output already exists", path=str(output)
            )
    preprocessing_version = f"{COARSENER_VERSION}+{DEGREE_SPLITTER_VERSION}"
    request = ExternalFetchRequest(
        integration_run_id=args.run_id, query=spec.query,
        query_geometry=spec.query_geometry, service=spec.service,
        service_version=args.service_version,
        preprocessing_version=preprocessing_version,
    )
    raw_bytes, network_bytes, provenance = ExternalOSMFetchAdapter(
        transport or _live_transport
    ).fetch(request, opt_in=args.opt_in)
    write_immutable_snapshot(raw_path, raw_bytes)
    write_immutable_snapshot(network_path, network_bytes)
    sealed = spec.as_dict()
    sealed.pop("content_hash", None)
    sealed.update({
        "schema_version": "1.0",
        "service_version": args.service_version,
        "acquired_at_utc": provenance.executed_at_utc,
        "preprocessing_version": preprocessing_version,
        "raw_file_hash": provenance.raw_output_hash,
        "network_file_hash": provenance.network_output_hash,
        "network_content_hash": provenance.network_content_hash,
    })
    config_bytes = yaml.safe_dump(
        sealed, allow_unicode=True, sort_keys=False
    ).encode("utf-8")
    write_immutable_snapshot(config_output, config_bytes)
    record = {
        "schema_version": "1.0",
        "integration_run_id": args.run_id,
        "input_config_hash": spec.content_hash,
        "sealed_config_hash": content_hash(sealed),
        "outputs": {
            "raw_path": spec.raw_path,
            "network_path": spec.network_path,
            "sealed_config_path": str(config_output.relative_to(root)).replace("\\", "/"),
        },
        "provenance": provenance.as_dict(),
    }
    write_immutable_snapshot(provenance_output, canonical_json(record) + b"\n")
    return record


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        record = run_integration(args)
    except (ResearchValidationError, ValueError) as exc:
        payload = exc.as_dict() if isinstance(exc, ResearchValidationError) else {
            "code": "EXTERNAL_FETCH_FAILED", "message": str(exc)
        }
        sys.stderr.buffer.write(canonical_json(payload) + b"\n")
        return 2
    sys.stdout.buffer.write(canonical_json(record) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

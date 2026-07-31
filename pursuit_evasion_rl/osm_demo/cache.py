"""Immutable, atomically published offline cache bundles with verified loading.

A cache entry is an immutable directory named by its timestamp-free cache key::

    <cache_root>/<key>/
      raw_osm.json          # canonical RawOSMGraph bytes
      model_network.json    # canonical ModelNetwork bytes
      mapping_manifest.json # canonical MappingManifest bytes
      network_metadata.json # canonical NetworkMetadata bytes (identity + provenance)
      bundle_manifest.json  # commit marker written last; artifact hashes + sizes

Writers stage every artifact in a sibling temporary directory, flush it to disk,
write the ``bundle_manifest.json`` commit marker last and atomically rename the
directory into place.  Because directory rename fails when the destination
already exists, concurrent creators of the same key never overwrite one another;
the loser accepts the winner only when the winner's identity content is
byte-equivalent, otherwise a classified ``CACHE_KEY_COLLISION`` is raised.

Readers require the commit marker, supported schemas and a matching SHA-256 for
every artifact before any deserialization.  Any missing marker, schema mismatch
or hash mismatch means the entry is logically quarantined: it is never returned
as data and instead surfaces a classified corruption error.  OSM attribution and
the acquisition/commit timestamps travel with the bundle so exports can preserve
provenance (Requirements 4.1-4.7).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping

from .canonical import (
    artifact_identity_hash,
    build_cache_key,
    canonical_json,
    content_hash,
    sha256_bytes,
)
from .models import (
    CONFIG_VERSION,
    DOMAIN_SCHEMA_VERSION,
    ArtifactDescriptor,
    BoundedArea,
    BundleManifest,
    DomainValidationError,
    Intersection,
    MappingManifest,
    ModelNetwork,
    NetworkMetadata,
    RawEdge,
    RawNode,
    RawOSMGraph,
    Segment,
)

CACHE_SCHEMA_VERSION = DOMAIN_SCHEMA_VERSION
DEFAULT_ATTRIBUTION = "OpenStreetMap contributors via OSMnx"

# Immutable bundles are always directories; a bare ``*.json`` file sitting at
# the cache root is the single-file layout used by the legacy cache and must be
# migrated explicitly rather than silently reused (Requirements 4.3-4.7).
LEGACY_CACHE_SUFFIX = ".json"

RAW_GRAPH_ARTIFACT = "raw_osm.json"
MODEL_NETWORK_ARTIFACT = "model_network.json"
MAPPING_MANIFEST_ARTIFACT = "mapping_manifest.json"
NETWORK_METADATA_ARTIFACT = "network_metadata.json"
BUNDLE_MANIFEST_ARTIFACT = "bundle_manifest.json"

# Artifacts the cache verifies against the recorded Network_Metadata hashes.
_METADATA_TRACKED_ARTIFACTS = {
    RAW_GRAPH_ARTIFACT: "raw_osm",
    MODEL_NETWORK_ARTIFACT: "model_network",
    MAPPING_MANIFEST_ARTIFACT: "mapping_manifest",
}
_CONTENT_ARTIFACTS = (
    RAW_GRAPH_ARTIFACT,
    MODEL_NETWORK_ARTIFACT,
    MAPPING_MANIFEST_ARTIFACT,
    NETWORK_METADATA_ARTIFACT,
)


class CacheError(DomainValidationError):
    """A classified cache failure that never returns quarantined data."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        cache_key: str | None = None,
        path: str | None = None,
        expected: Any = None,
        actual: Any = None,
    ) -> None:
        super().__init__(code, message, path=path, expected=expected, actual=actual)
        self.cache_key = cache_key

    def as_dict(self) -> dict[str, Any]:
        payload = super().as_dict()
        payload["cache_key"] = self.cache_key
        return payload


@dataclass(frozen=True, slots=True)
class CacheBundle:
    """An immutable, verified cache entry with its provenance."""

    cache_key: str
    raw_graph: RawOSMGraph
    network: ModelNetwork
    mapping: MappingManifest
    metadata: NetworkMetadata
    manifest: BundleManifest

    @property
    def attribution(self) -> str:
        return self.manifest.attribution

    @property
    def acquired_at(self) -> str:
        return self.metadata.acquired_at

    @property
    def committed_at(self) -> str:
        return self.manifest.committed_at

    def provenance(self) -> dict[str, Any]:
        """Attribution and timestamps to preserve when exporting (Req 4.7)."""
        return {
            "attribution": self.manifest.attribution,
            "source": self.metadata.source,
            "acquired_at": self.metadata.acquired_at,
            "committed_at": self.manifest.committed_at,
            "cache_key": self.cache_key,
        }


# ---------------------------------------------------------------------------
# Deserialization of canonical artifact bytes back into immutable domain models
# ---------------------------------------------------------------------------


def _points(values: Any) -> tuple[tuple[float, float], ...]:
    return tuple((float(point[0]), float(point[1])) for point in values)


def _raw_node(data: Mapping[str, Any]) -> RawNode:
    return RawNode(
        source_id=data["source_id"],
        lat=data["lat"],
        lon=data["lon"],
        x_m=data["x_m"],
        y_m=data["y_m"],
        tags=dict(data.get("tags", {})),
    )


def _raw_edge(data: Mapping[str, Any]) -> RawEdge:
    return RawEdge(
        source_u=data["source_u"],
        source_v=data["source_v"],
        source_key=data["source_key"],
        length_m=data["length_m"],
        geometry_xy=_points(data["geometry_xy"]),
        way_refs=tuple(data.get("way_refs", ())),
        oneway=bool(data.get("oneway", False)),
        road_class=data.get("road_class", ""),
        tags=dict(data.get("tags", {})),
    )


def _raw_graph(data: Mapping[str, Any]) -> RawOSMGraph:
    return RawOSMGraph(
        nodes=tuple(_raw_node(node) for node in data["nodes"]),
        edges=tuple(_raw_edge(edge) for edge in data["edges"]),
        schema_version=data.get("schema_version", DOMAIN_SCHEMA_VERSION),
    )


def _intersection(data: Mapping[str, Any]) -> Intersection:
    return Intersection(
        id=data["id"],
        position_xy=(float(data["position_xy"][0]), float(data["position_xy"][1])),
        source_signature=data["source_signature"],
        boundary_kind=data.get("boundary_kind"),
        virtual=bool(data.get("virtual", False)),
        outgoing_segment_ids=tuple(data.get("outgoing_segment_ids", ())),
        incoming_segment_ids=tuple(data.get("incoming_segment_ids", ())),
    )


def _segment(data: Mapping[str, Any]) -> Segment:
    return Segment(
        id=data["id"],
        start_id=data["start_id"],
        end_id=data["end_id"],
        length_m=data["length_m"],
        geometry_xy=_points(data["geometry_xy"]),
        source_edge_refs=tuple(data.get("source_edge_refs", ())),
        attributes=dict(data.get("attributes", {})),
        virtual=bool(data.get("virtual", False)),
        crosses_boundary=bool(data.get("crosses_boundary", False)),
    )


def _model_network(data: Mapping[str, Any]) -> ModelNetwork:
    return ModelNetwork(
        intersections=tuple(_intersection(item) for item in data["intersections"]),
        segments=tuple(_segment(item) for item in data["segments"]),
        schema_version=data.get("schema_version", DOMAIN_SCHEMA_VERSION),
    )


def _mapping_manifest(data: Mapping[str, Any]) -> MappingManifest:
    return MappingManifest(
        coarsener_version=data["coarsener_version"],
        raw_node_to_intersection=dict(data["raw_node_to_intersection"]),
        segment_sources={key: tuple(value) for key, value in data["segment_sources"].items()},
        virtual_mappings=dict(data["virtual_mappings"]),
        canonical_rules=data["canonical_rules"],
        action_ordering=data["action_ordering"],
        schema_version=data.get("schema_version", DOMAIN_SCHEMA_VERSION),
    )


def _bounded_area(data: Mapping[str, Any]) -> BoundedArea:
    return BoundedArea(
        name=data["name"],
        north=data["north"],
        south=data["south"],
        east=data["east"],
        west=data["west"],
        max_area_km2=data["max_area_km2"],
        config_version=data.get("config_version", CONFIG_VERSION),
        schema_version=data.get("schema_version", DOMAIN_SCHEMA_VERSION),
    )


def _network_metadata(data: Mapping[str, Any]) -> NetworkMetadata:
    return NetworkMetadata(
        area=_bounded_area(data["area"]),
        source=data["source"],
        acquired_at=data["acquired_at"],
        network_type=data["network_type"],
        metric_crs=data["metric_crs"],
        schema_versions=dict(data["schema_versions"]),
        settings_hash=data["settings_hash"],
        artifact_hashes=dict(data.get("artifact_hashes", {})),
        statistics=dict(data.get("statistics", {})),
        reachability_digest=data.get("reachability_digest"),
        validation=dict(data.get("validation", {})),
        operation_status=data.get("operation_status", "READY"),
        schema_version=data.get("schema_version", DOMAIN_SCHEMA_VERSION),
    )


def _bundle_manifest(data: Mapping[str, Any]) -> BundleManifest:
    return BundleManifest(
        cache_key=data["cache_key"],
        artifacts=tuple(
            ArtifactDescriptor(name=item["name"], sha256=item["sha256"], size=item["size"])
            for item in data["artifacts"]
        ),
        attribution=data["attribution"],
        committed_at=data["committed_at"],
        schema_version=data.get("schema_version", DOMAIN_SCHEMA_VERSION),
    )


def _with_artifact_hashes(
    metadata: NetworkMetadata, hashes: Mapping[str, str]
) -> NetworkMetadata:
    merged = dict(metadata.artifact_hashes)
    merged.update(hashes)
    return NetworkMetadata(
        area=metadata.area,
        source=metadata.source,
        acquired_at=metadata.acquired_at,
        network_type=metadata.network_type,
        metric_crs=metadata.metric_crs,
        schema_versions=dict(metadata.schema_versions),
        settings_hash=metadata.settings_hash,
        artifact_hashes=merged,
        statistics=dict(metadata.statistics),
        reachability_digest=metadata.reachability_digest,
        validation=dict(metadata.validation),
        operation_status=metadata.operation_status,
        schema_version=metadata.schema_version,
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _write_file(path: Path, data: bytes) -> None:
    with open(path, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    # Directory fsync is a durability best-effort; it is unsupported on Windows.
    if os.name == "nt":
        return
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class CacheStore:
    """Immutable atomic cache of prepared OSM bundles with verified loading."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.root = Path(root)
        self._clock = clock
        self._quarantine: set[str] = set()

    # -- keys ---------------------------------------------------------------

    def cache_key(
        self,
        area: BoundedArea,
        *,
        coarsener_version: str,
        preprocessing_settings_hash: str,
        network_type: str = "drive",
        raw_schema_version: str = DOMAIN_SCHEMA_VERSION,
    ) -> str:
        return build_cache_key(
            area,
            network_type=network_type,
            raw_schema_version=raw_schema_version,
            coarsener_version=coarsener_version,
            preprocessing_settings_hash=preprocessing_settings_hash,
        )

    def _entry_path(self, key: str) -> Path:
        return self.root / key

    def _legacy_path(self, key: str) -> Path:
        return self.root / f"{key}{LEGACY_CACHE_SUFFIX}"

    # -- legacy migration boundary -----------------------------------------

    def legacy_entries(self) -> tuple[str, ...]:
        """Return the names of any single-file legacy cache entries at the root.

        Immutable bundles are only ever published as directories, so a bare
        ``*.json`` file directly under the cache root is legacy single-file
        data.  Returning it lets callers require an explicit migration instead
        of silently reusing content the current loader cannot verify.
        """
        if not self.root.is_dir():
            return ()
        return tuple(
            sorted(
                path.name
                for path in self.root.iterdir()
                if path.is_file() and path.suffix == LEGACY_CACHE_SUFFIX
            )
        )

    def require_migration(self) -> None:
        """Raise if legacy single-file cache data exists (Requirements 4.3-4.7).

        The current cache reads only verified immutable bundles.  Rather than
        silently reusing legacy single-file data whose hashes and schema it
        cannot validate, it surfaces a classified ``CACHE_MIGRATION_REQUIRED``
        error listing the legacy files that must be migrated first.
        """
        legacy = self.legacy_entries()
        if legacy:
            raise self._migration_required(legacy)

    # -- saving -------------------------------------------------------------

    def save(
        self,
        *,
        raw_graph: RawOSMGraph,
        network: ModelNetwork,
        mapping: MappingManifest,
        metadata: NetworkMetadata,
        coarsener_version: str,
        preprocessing_settings_hash: str,
        attribution: str | None = None,
        network_type: str = "drive",
        raw_schema_version: str = DOMAIN_SCHEMA_VERSION,
    ) -> CacheBundle:
        """Atomically publish an immutable bundle and return the verified entry.

        Raw graph, model graph, mapping and metadata are staged first; the
        ``bundle_manifest.json`` commit marker with every artifact hash is
        written last, then the whole directory is renamed into place.  When the
        destination already exists, the existing committed entry is reused only
        if its identity content is byte-equivalent (Requirements 4.1, 4.2).
        """
        key = self.cache_key(
            metadata.area,
            coarsener_version=coarsener_version,
            preprocessing_settings_hash=preprocessing_settings_hash,
            network_type=network_type,
            raw_schema_version=raw_schema_version,
        )

        raw_bytes = canonical_json(raw_graph)
        model_bytes = canonical_json(network)
        mapping_bytes = canonical_json(mapping)
        artifact_hashes = {
            "raw_osm": sha256_bytes(raw_bytes),
            "model_network": sha256_bytes(model_bytes),
            "mapping_manifest": sha256_bytes(mapping_bytes),
        }
        stamped_metadata = _with_artifact_hashes(metadata, artifact_hashes)
        metadata_bytes = canonical_json(stamped_metadata)

        files: dict[str, bytes] = {
            RAW_GRAPH_ARTIFACT: raw_bytes,
            MODEL_NETWORK_ARTIFACT: model_bytes,
            MAPPING_MANIFEST_ARTIFACT: mapping_bytes,
            NETWORK_METADATA_ARTIFACT: metadata_bytes,
        }
        descriptors = tuple(
            ArtifactDescriptor(name=name, sha256=sha256_bytes(payload), size=len(payload))
            for name, payload in files.items()
        )
        manifest = BundleManifest(
            cache_key=key,
            artifacts=descriptors,
            attribution=attribution or metadata.source or DEFAULT_ATTRIBUTION,
            committed_at=self._clock().isoformat(),
        )
        files[BUNDLE_MANIFEST_ARTIFACT] = canonical_json(manifest)

        dest = self._entry_path(key)
        if dest.exists():
            return self._accept_existing(key, dest, files)

        self.root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=self.root))
        try:
            # Write the content artifacts first, then the commit marker last.
            for name in _CONTENT_ARTIFACTS:
                _write_file(staging / name, files[name])
            _write_file(staging / BUNDLE_MANIFEST_ARTIFACT, files[BUNDLE_MANIFEST_ARTIFACT])
            _fsync_directory(staging)
            try:
                os.rename(staging, dest)
            except OSError:
                # Another creator won the race for this key; accept only a
                # byte-equivalent winner, otherwise report the collision.
                shutil.rmtree(staging, ignore_errors=True)
                return self._accept_existing(key, dest, files)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        _fsync_directory(self.root)
        return self.load(key)

    def _accept_existing(
        self, key: str, dest: Path, files: Mapping[str, bytes]
    ) -> CacheBundle:
        existing = self.load(key)
        for name in _CONTENT_ARTIFACTS:
            existing_bytes = (dest / name).read_bytes()
            if not self._identity_equivalent(name, existing_bytes, files[name]):
                raise CacheError(
                    "CACHE_KEY_COLLISION",
                    f"Cache key {key} already stores byte-different content for {name}",
                    cache_key=key,
                    actual=name,
                )
        return existing

    @staticmethod
    def _identity_equivalent(name: str, existing: bytes, candidate: bytes) -> bool:
        if existing == candidate:
            return True
        # Metadata legitimately differs only by provenance timestamps between
        # two acquisitions of the same identity; compare timestamp-free identity.
        if name == NETWORK_METADATA_ARTIFACT:
            return _metadata_identity(existing) == _metadata_identity(candidate)
        return False

    # -- loading ------------------------------------------------------------

    def has_valid_entry(self, key: str) -> bool:
        try:
            self.load(key)
        except CacheError:
            return False
        return True

    def load(self, key: str) -> CacheBundle:
        """Load and fully verify a bundle, or raise a classified cache error.

        The commit marker, supported schemas and every artifact hash are checked
        before the data is returned.  A quarantined or corrupt entry is never
        returned as data (Requirements 4.3, 4.6).
        """
        if key in self._quarantine:
            raise CacheError(
                "CACHE_QUARANTINED",
                f"Cache entry {key} was quarantined after a prior integrity failure",
                cache_key=key,
            )

        dest = self._entry_path(key)
        marker = dest / BUNDLE_MANIFEST_ARTIFACT
        if not dest.is_dir() or not marker.is_file():
            # A legacy single-file entry for this key must never be silently
            # reused; require an explicit migration first (Requirements 4.3-4.7).
            if self._legacy_path(key).is_file():
                raise self._migration_required((self._legacy_path(key).name,), cache_key=key)
            raise self._miss(key)

        try:
            manifest = _bundle_manifest(json.loads(marker.read_bytes().decode("utf-8")))
        except (json.JSONDecodeError, KeyError, TypeError, DomainValidationError) as exc:
            raise self._corrupt(key, "bundle manifest could not be parsed", exc) from exc

        if manifest.cache_key != key:
            raise self._corrupt(
                key,
                "bundle manifest cache key does not match its location",
                None,
                expected=key,
                actual=manifest.cache_key,
            )

        recorded = {descriptor.name: descriptor for descriptor in manifest.artifacts}
        missing = [name for name in _CONTENT_ARTIFACTS if name not in recorded]
        if missing:
            raise self._corrupt(key, f"bundle manifest omits artifacts: {missing}", None)

        payloads: dict[str, bytes] = {}
        for name in _CONTENT_ARTIFACTS:
            path = dest / name
            if not path.is_file():
                raise self._corrupt(key, f"artifact {name} is missing", None)
            data = path.read_bytes()
            descriptor = recorded[name]
            digest = sha256_bytes(data)
            if digest != descriptor.sha256 or len(data) != descriptor.size:
                raise self._corrupt(
                    key,
                    f"artifact {name} failed hash verification",
                    None,
                    expected={"sha256": descriptor.sha256, "size": descriptor.size},
                    actual={"sha256": digest, "size": len(data)},
                )
            payloads[name] = data

        try:
            raw_graph = _raw_graph(json.loads(payloads[RAW_GRAPH_ARTIFACT].decode("utf-8")))
            network = _model_network(json.loads(payloads[MODEL_NETWORK_ARTIFACT].decode("utf-8")))
            mapping = _mapping_manifest(json.loads(payloads[MAPPING_MANIFEST_ARTIFACT].decode("utf-8")))
            metadata = _network_metadata(json.loads(payloads[NETWORK_METADATA_ARTIFACT].decode("utf-8")))
        except (json.JSONDecodeError, KeyError, TypeError, IndexError, DomainValidationError) as exc:
            raise self._corrupt(key, "artifact failed schema validation", exc) from exc

        deserialized = {
            RAW_GRAPH_ARTIFACT: raw_graph,
            MODEL_NETWORK_ARTIFACT: network,
            MAPPING_MANIFEST_ARTIFACT: mapping,
        }
        for artifact_name, metadata_key in _METADATA_TRACKED_ARTIFACTS.items():
            declared = metadata.artifact_hashes.get(metadata_key)
            computed = content_hash(deserialized[artifact_name])
            if declared != computed:
                raise self._corrupt(
                    key,
                    f"artifact {artifact_name} hash disagrees with Network_Metadata",
                    None,
                    expected=declared,
                    actual=computed,
                )

        return CacheBundle(
            cache_key=key,
            raw_graph=raw_graph,
            network=network,
            mapping=mapping,
            metadata=metadata,
            manifest=manifest,
        )

    def load_offline(self, key: str) -> CacheBundle:
        """Return a verified entry without any external lookup (Req 4.4/4.5)."""
        return self.load(key)

    # -- error construction -------------------------------------------------

    def _miss(self, key: str) -> CacheError:
        guidance = (
            "Prepare this bounded area online once (fetch -> coarsen -> validate -> "
            "save) to populate the cache, or provide a committed bundle for this key "
            "before running offline."
        )
        return CacheError(
            "CACHE_MISS",
            f"No valid offline cache entry for cache key {key}. {guidance}",
            cache_key=key,
            expected=key,
            actual={"required_cache_key": key, "preparation": guidance},
        )

    def _migration_required(
        self, legacy_files: tuple[str, ...], *, cache_key: str | None = None
    ) -> CacheError:
        guidance = (
            "Legacy single-file cache data was detected. Migrate it into a "
            "verified immutable bundle (fetch -> coarsen -> validate -> save) "
            "before offline reuse; the current cache never silently reuses "
            "unverified single-file entries."
        )
        return CacheError(
            "CACHE_MIGRATION_REQUIRED",
            f"Explicit migration required for legacy cache data {list(legacy_files)}. {guidance}",
            cache_key=cache_key,
            actual={"legacy_files": list(legacy_files), "migration": guidance},
        )

    def _corrupt(
        self,
        key: str,
        reason: str,
        cause: Exception | None,
        *,
        expected: Any = None,
        actual: Any = None,
    ) -> CacheError:
        # Logical quarantine: never return this entry as data again this session.
        self._quarantine.add(key)
        detail = f"{reason}" if cause is None else f"{reason}: {cause}"
        return CacheError(
            "CACHE_CORRUPT",
            f"Cache entry {key} is corrupt and was quarantined ({detail})",
            cache_key=key,
            expected=expected,
            actual=actual,
        )


def _metadata_identity(payload: bytes) -> str:
    """Identity hash of metadata content excluding provenance timestamps."""
    return artifact_identity_hash(_network_metadata(json.loads(payload.decode("utf-8"))))


__all__ = (
    "BUNDLE_MANIFEST_ARTIFACT",
    "CACHE_SCHEMA_VERSION",
    "DEFAULT_ATTRIBUTION",
    "LEGACY_CACHE_SUFFIX",
    "MAPPING_MANIFEST_ARTIFACT",
    "MODEL_NETWORK_ARTIFACT",
    "NETWORK_METADATA_ARTIFACT",
    "RAW_GRAPH_ARTIFACT",
    "CacheBundle",
    "CacheError",
    "CacheStore",
)

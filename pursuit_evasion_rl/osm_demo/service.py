"""Dependency-injected application services for the OSM road-pursuit demo.

This module is the application (use-case) layer that sits between the pure
domain modules and any delivery mechanism (FastAPI v1, a CLI or a reproducibility
script).  It replaces direct module-global orchestration with explicit
*repository interfaces* and two interchangeable implementations per repository:

* an in-process, dictionary-backed implementation for the fast default demo, and
* a file-backed artifact implementation that persists every prepared network,
  run and export so results survive a process restart and remain reproducible.

Design boundaries (design section 11, "FastAPI v1"; Requirements 12.1-12.3):

* The domain layer stays free of FastAPI/HTTP/filesystem-path knowledge.  The
  service exposes plain Python use cases returning immutable domain values; only
  the *file-backed repository implementations* touch the filesystem, and even
  they only ever read/write artifacts under an injected root directory.
* Use cases are wired here once and injected everywhere: online/offline network
  preparation, checkpoint/network compatibility, six-police action
  recommendation with measured latency, deterministic episode runs returning a
  run identifier, input hashes and deterministic seeds, run metrics and staged
  competition export creation.
* Endpoints never mutate module globals; they hold a :class:`DemoService`
  constructed with the repositories they want (in-process for the demo,
  file-backed for reproducibility).

All not-found conditions raise a classified :class:`DomainValidationError`
(``*_NOT_FOUND``) that the API layer maps to HTTP 404, and compatibility failures
surface the sectioned :class:`CompatibilityReport` for HTTP 409 mapping.
"""

from __future__ import annotations

import copyreg
import os
import pickle
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .cache import DEFAULT_ATTRIBUTION, CacheBundle, CacheStore
from .canonical import canonical_json, content_hash
from .checkpoint import CheckpointInspection, CheckpointInspector
from .coarsening import (
    COARSENER_VERSION,
    DEGREE_SPLITTER_VERSION,
    CoarseningSettings,
    NetworkStatistics,
    coarsen_raw_graph,
    compute_statistics,
    prepare_model_network,
)
from .exports import CompetitionExporter, ExportReport, ExportResult
from .metrics import not_measured_latency, summarize_metrics
from .models import (
    DOMAIN_SCHEMA_VERSION,
    ClaimRecord,
    ClaimStatus,
    CompatibilityReport,
    DomainValidationError,
    EpisodeConfig,
    EpisodeRecord,
    MappingManifest,
    MetricsSummary,
    ModelNetwork,
    NetworkMetadata,
    RunLineage,
    RunManifest,
    RunMode,
    VehiclePlacement,
    validate_vehicle_placements,
)
from .observations import DEFAULT_CLIP_DISTANCE_M, osm_topology_v1_contract
from .osm_source import OSMSource, RawGraphAcquisition
from .policies import (
    ActionRecommendation,
    BaselinePolicePolicy,
    LearnedPolicePolicy,
    PolicePolicy,
    osm_execution_config,
)
from .rendering import OSMRenderer
from .runner import FugitiveFactory, derive_stream_seed, run_batch

CODE_VERSION = "osm-demo-service-v1"


# ---------------------------------------------------------------------------
# Immutable application-layer value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PreparedNetworkRecord:
    """A prepared, validated Model_Network with its provenance and identity.

    ``network_id`` is the timestamp-free cache key, so online and offline
    preparation of the same bounded area and settings yield the same identifier.
    ``network_hash`` is the content hash of the Model_Network itself, used for
    checkpoint/network compatibility and run provenance.
    """

    network_id: str
    network_hash: str
    network: ModelNetwork
    mapping: MappingManifest
    metadata: NetworkMetadata
    statistics: NetworkStatistics

    def summary(self) -> dict[str, Any]:
        """Metadata and validation summary safe to return over an API."""
        return {
            "network_id": self.network_id,
            "network_hash": self.network_hash,
            "source": self.metadata.source,
            "acquired_at": self.metadata.acquired_at,
            "network_type": self.metadata.network_type,
            "metric_crs": self.metadata.metric_crs,
            "operation_status": self.metadata.operation_status,
            "attribution": self.metadata.source,
            "statistics": self.statistics.as_dict(),
            "validation": dict(self.metadata.validation),
        }


@dataclass(frozen=True, slots=True)
class RecommendationResult:
    """Six atomic recommendations plus the measured (or not-measured) latency."""

    network_id: str
    profile: str
    experimental: bool
    compatibility_report_id: str
    recommendations: tuple[ActionRecommendation, ...]
    latency: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "network_id": self.network_id,
            "profile": self.profile,
            "experimental": self.experimental,
            "compatibility_report_id": self.compatibility_report_id,
            "recommendations": [rec.as_dict() for rec in self.recommendations],
            "latency": dict(self.latency),
        }


@dataclass(frozen=True, slots=True)
class RunEntry:
    """An immutable, persisted episode-run result."""

    manifest: RunManifest
    records: tuple[EpisodeRecord, ...]
    metrics: MetricsSummary

    @property
    def run_id(self) -> str:
        return self.manifest.run_id


@dataclass(frozen=True, slots=True)
class RunResult:
    """The accepted run identifier, provenance and result-location for an API."""

    run_id: str
    manifest: RunManifest
    metrics: MetricsSummary
    result_location: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mode": self.manifest.mode.value,
            "network_hash": self.manifest.network_hash,
            "checkpoint_hash": self.manifest.checkpoint_hash,
            "observation_profile": self.manifest.observation_profile,
            "policy_kind": self.manifest.policy_kind,
            "claim_status": self.manifest.claim_status.value,
            "seeds": list(self.manifest.seeds),
            "input_hashes": dict(self.manifest.artifacts),
            "result_location": self.result_location,
        }


@dataclass(frozen=True, slots=True)
class ExportEntry:
    """A persisted competition export result."""

    run_id: str
    result: ExportResult

    @property
    def export_id(self) -> str:
        return self.result.manifest.export_id


# ---------------------------------------------------------------------------
# Repository interfaces (domain-facing; no HTTP/filesystem knowledge leaks out)
# ---------------------------------------------------------------------------


@runtime_checkable
class NetworkRepository(Protocol):
    """Persistence boundary for prepared, validated Model_Networks."""

    def put(self, record: PreparedNetworkRecord) -> None: ...

    def get(self, network_id: str) -> PreparedNetworkRecord: ...

    def exists(self, network_id: str) -> bool: ...

    def list_ids(self) -> tuple[str, ...]: ...

    def location_for(self, network_id: str) -> str: ...


@runtime_checkable
class RunRepository(Protocol):
    """Persistence boundary for deterministic episode-run results."""

    def put(self, entry: RunEntry) -> None: ...

    def get(self, run_id: str) -> RunEntry: ...

    def exists(self, run_id: str) -> bool: ...

    def list_ids(self) -> tuple[str, ...]: ...

    def location_for(self, run_id: str) -> str: ...


@runtime_checkable
class ExportRepository(Protocol):
    """Persistence boundary for published competition exports."""

    def put(self, entry: ExportEntry) -> None: ...

    def get(self, export_id: str) -> ExportEntry: ...

    def exists(self, export_id: str) -> bool: ...

    def list_ids(self) -> tuple[str, ...]: ...

    def location_for(self, export_id: str) -> str: ...


def _not_found(kind: str, identifier: str) -> DomainValidationError:
    return DomainValidationError(
        f"{kind.upper()}_NOT_FOUND",
        f"No {kind.replace('_', ' ')} exists for identifier {identifier!r}",
        actual=identifier,
    )


# ---------------------------------------------------------------------------
# In-process (dictionary-backed) repository implementations
# ---------------------------------------------------------------------------


class InMemoryNetworkRepository:
    """Process-local prepared-network store for the fast default demo."""

    def __init__(self) -> None:
        self._entries: dict[str, PreparedNetworkRecord] = {}

    def put(self, record: PreparedNetworkRecord) -> None:
        self._entries[record.network_id] = record

    def get(self, network_id: str) -> PreparedNetworkRecord:
        try:
            return self._entries[network_id]
        except KeyError as exc:
            raise _not_found("network", network_id) from exc

    def exists(self, network_id: str) -> bool:
        return network_id in self._entries

    def list_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def location_for(self, network_id: str) -> str:
        return f"memory://networks/{network_id}"


class InMemoryRunRepository:
    """Process-local run store for the fast default demo."""

    def __init__(self) -> None:
        self._entries: dict[str, RunEntry] = {}

    def put(self, entry: RunEntry) -> None:
        self._entries[entry.run_id] = entry

    def get(self, run_id: str) -> RunEntry:
        try:
            return self._entries[run_id]
        except KeyError as exc:
            raise _not_found("run", run_id) from exc

    def exists(self, run_id: str) -> bool:
        return run_id in self._entries

    def list_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def location_for(self, run_id: str) -> str:
        return f"memory://runs/{run_id}"


class InMemoryExportRepository:
    """Process-local export store for the fast default demo."""

    def __init__(self) -> None:
        self._entries: dict[str, ExportEntry] = {}

    def put(self, entry: ExportEntry) -> None:
        self._entries[entry.export_id] = entry

    def get(self, export_id: str) -> ExportEntry:
        try:
            return self._entries[export_id]
        except KeyError as exc:
            raise _not_found("export", export_id) from exc

    def exists(self, export_id: str) -> bool:
        return export_id in self._entries

    def list_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def location_for(self, export_id: str) -> str:
        return f"memory://exports/{export_id}"


# ---------------------------------------------------------------------------
# File-backed artifact repository implementations
# ---------------------------------------------------------------------------


def _rebuild_mappingproxy(data: dict[Any, Any]) -> MappingProxyType:
    """Reconstruct a read-only mapping proxy from a plain dict on unpickling."""
    return MappingProxyType(data)


def _reduce_mappingproxy(proxy: MappingProxyType) -> tuple[Any, tuple[dict[Any, Any]]]:
    """Make read-only mapping proxies picklable for the file-backed store.

    The immutable domain models wrap their mapping fields in
    :class:`types.MappingProxyType`, which the default pickler cannot serialize
    (the type is not importable by name).  Registering this reducer lets the
    file-backed repositories persist and reload those frozen values; unpickling
    restores an equivalent read-only proxy over a plain dict.
    """
    return (_rebuild_mappingproxy, (dict(proxy),))


copyreg.pickle(MappingProxyType, _reduce_mappingproxy)


def _atomic_write(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically via a sibling temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp-{os.getpid()}"
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


class _FileRepositoryBase:
    """Shared file-backed persistence keyed by identifier under a root.

    Each entry is stored as a pickled snapshot of its immutable domain value
    (for faithful reload across process restarts) alongside a canonical-JSON
    ``*.summary.json`` artifact recording its provenance and content hash.  A
    small in-memory cache keeps hot reads cheap while the on-disk artifacts make
    every result reproducible.  Only this implementation layer knows about
    filesystem paths; the service and domain never do.
    """

    kind = "entry"

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._root = Path(root) / f"{self.kind}s"
        self._cache: dict[str, Any] = {}

    def _entry_dir(self, identifier: str) -> Path:
        return self._root / identifier

    def _state_path(self, identifier: str) -> Path:
        return self._entry_dir(identifier) / "state.pkl"

    def _summary_path(self, identifier: str) -> Path:
        return self._entry_dir(identifier) / "summary.json"

    def _store(self, identifier: str, value: Any, summary: Mapping[str, Any]) -> None:
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        _atomic_write(self._state_path(identifier), payload)
        summary_bytes = canonical_json(dict(summary))
        _atomic_write(self._summary_path(identifier), summary_bytes)
        self._cache[identifier] = value

    def _load(self, identifier: str) -> Any:
        cached = self._cache.get(identifier)
        if cached is not None:
            return cached
        state_path = self._state_path(identifier)
        if not state_path.is_file():
            raise _not_found(self.kind, identifier)
        # The pickled artifacts are produced only by this service; they are not
        # untrusted external input.
        value = pickle.loads(state_path.read_bytes())
        self._cache[identifier] = value
        return value

    def exists(self, identifier: str) -> bool:
        return identifier in self._cache or self._state_path(identifier).is_file()

    def list_ids(self) -> tuple[str, ...]:
        disk = (
            {path.name for path in self._root.iterdir() if (path / "state.pkl").is_file()}
            if self._root.is_dir()
            else set()
        )
        return tuple(sorted(disk | set(self._cache)))

    def location_for(self, identifier: str) -> str:
        return str(self._entry_dir(identifier))


class FileNetworkRepository(_FileRepositoryBase):
    """File-backed prepared-network store for reproducibility."""

    kind = "network"

    def put(self, record: PreparedNetworkRecord) -> None:
        self._store(record.network_id, record, record.summary())

    def get(self, network_id: str) -> PreparedNetworkRecord:
        return self._load(network_id)


class FileRunRepository(_FileRepositoryBase):
    """File-backed run store for reproducibility."""

    kind = "run"

    def put(self, entry: RunEntry) -> None:
        summary = {
            "run_id": entry.run_id,
            "mode": entry.manifest.mode.value,
            "network_hash": entry.manifest.network_hash,
            "checkpoint_hash": entry.manifest.checkpoint_hash,
            "observation_profile": entry.manifest.observation_profile,
            "policy_kind": entry.manifest.policy_kind,
            "claim_status": entry.manifest.claim_status.value,
            "seeds": list(entry.manifest.seeds),
            "input_hashes": dict(entry.manifest.artifacts),
            "episode_count": len(entry.records),
        }
        self._store(entry.run_id, entry, summary)

    def get(self, run_id: str) -> RunEntry:
        return self._load(run_id)


class FileExportRepository(_FileRepositoryBase):
    """File-backed export store for reproducibility."""

    kind = "export"

    def put(self, entry: ExportEntry) -> None:
        summary = {
            "export_id": entry.export_id,
            "run_id": entry.run_id,
            "export_dir": entry.result.export_dir,
            "manifest_path": entry.result.manifest_path,
            "dimensions": list(entry.result.manifest.dimensions),
            "warnings": list(entry.result.warnings),
        }
        self._store(entry.export_id, entry, summary)

    def get(self, export_id: str) -> ExportEntry:
        return self._load(export_id)


# ---------------------------------------------------------------------------
# Service configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DemoServiceSettings:
    """Injected configuration shared by every service use case."""

    coarsening_settings: CoarseningSettings = field(default_factory=CoarseningSettings)
    clip_distance_m: float = DEFAULT_CLIP_DISTANCE_M
    minimum_evaluation_episodes: int = 30
    confidence_interval_method: str = "wilson"
    code_version: str = CODE_VERSION
    attribution: str = DEFAULT_ATTRIBUTION
    latency_sample_count: int = 5
    latency_warmup: int = 1

    def preprocessing_settings_hash(self) -> str:
        """Timestamp-free hash of the deterministic preprocessing configuration."""
        return content_hash(
            {
                "coarsener_version": COARSENER_VERSION,
                "degree_splitter_version": DEGREE_SPLITTER_VERSION,
                "coordinate_decimals": self.coarsening_settings.coordinate_decimals,
                "geometry_tolerance_m": self.coarsening_settings.geometry_tolerance_m,
                "maximum_turn_degrees": self.coarsening_settings.maximum_turn_degrees,
                "transition_tags": list(self.coarsening_settings.transition_tags),
                "clip_distance_m": self.clip_distance_m,
            }
        )


# ---------------------------------------------------------------------------
# Application service
# ---------------------------------------------------------------------------


class DemoService:
    """Dependency-injected use cases wiring the OSM demo end to end.

    The service owns no module globals; it receives its repositories, cache
    store and settings at construction so an API layer can wire the in-process
    implementations for the demo or the file-backed implementations for
    reproducibility without any code change to the use cases themselves.
    """

    def __init__(
        self,
        *,
        cache_store: CacheStore,
        networks: NetworkRepository | None = None,
        runs: RunRepository | None = None,
        exports: ExportRepository | None = None,
        settings: DemoServiceSettings | None = None,
    ) -> None:
        self.cache_store = cache_store
        self.networks: NetworkRepository = networks or InMemoryNetworkRepository()
        self.runs: RunRepository = runs or InMemoryRunRepository()
        self.exports: ExportRepository = exports or InMemoryExportRepository()
        self.settings = settings or DemoServiceSettings()

    # ------------------------------------------------------------------
    # Network preparation (online and offline)
    # ------------------------------------------------------------------
    def prepare_online(
        self,
        area: Any,
        *,
        source: OSMSource,
        network_type: str = "drive",
    ) -> PreparedNetworkRecord:
        """Fetch, coarsen, validate and cache a bounded OSM network.

        The bounded area is fetched through the injected ``source`` (a live
        OSMnx source or an offline fixture source), coarsened deterministically,
        degree-bounded and reachability-validated, then published as an immutable
        cache bundle and registered for later compatibility, recommendation, run
        and export use cases (Requirement 12.1).
        """
        acquisition = _acquire(source, area, network_type)
        prepared = _coarsen_and_validate(acquisition, self.settings.coarsening_settings)
        metadata = _build_metadata(acquisition, prepared.statistics, self.settings)
        bundle = self.cache_store.save(
            raw_graph=acquisition.graph,
            network=prepared.network,
            mapping=prepared.mapping,
            metadata=metadata,
            coarsener_version=COARSENER_VERSION,
            preprocessing_settings_hash=self.settings.preprocessing_settings_hash(),
            attribution=self.settings.attribution,
            network_type=network_type,
        )
        return self._register_bundle(bundle, prepared.statistics)

    def prepare_offline(self, cache_key: str) -> PreparedNetworkRecord:
        """Load and register a verified offline cache bundle (Requirement 12.1).

        No external OSM access occurs; the bundle's commit marker, schemas and
        every artifact hash are verified by the cache store before use.
        """
        bundle = self.cache_store.load_offline(cache_key)
        statistics = compute_statistics(bundle.network)
        return self._register_bundle(bundle, statistics)

    def _register_bundle(
        self, bundle: CacheBundle, statistics: NetworkStatistics
    ) -> PreparedNetworkRecord:
        record = PreparedNetworkRecord(
            network_id=bundle.cache_key,
            network_hash=content_hash(bundle.network),
            network=bundle.network,
            mapping=bundle.mapping,
            metadata=bundle.metadata,
            statistics=statistics,
        )
        self.networks.put(record)
        return record

    def get_network(self, network_id: str) -> PreparedNetworkRecord:
        return self.networks.get(network_id)

    # ------------------------------------------------------------------
    # Checkpoint / network compatibility
    # ------------------------------------------------------------------
    def check_compatibility(
        self,
        network_id: str,
        checkpoint_path: str | os.PathLike[str],
        *,
        manifest: Any | None = None,
    ) -> CompatibilityReport:
        """Return the sectioned checkpoint/network compatibility report.

        The report separates structural, semantic, network and execution checks
        so the API can block inference on structural/network/execution failure
        and return HTTP 409 with the OSM training guidance for a semantic
        mismatch (Requirements 12.1, 12.6).
        """
        record = self.networks.get(network_id)
        inspection = self._inspect(record, checkpoint_path, manifest)
        return inspection.report

    def _inspect(
        self,
        record: PreparedNetworkRecord,
        checkpoint_path: str | os.PathLike[str],
        manifest: Any | None,
    ) -> CheckpointInspection:
        contract = osm_topology_v1_contract(self.settings.clip_distance_m)
        return CheckpointInspector().inspect(
            checkpoint_path,
            manifest,
            selected_observation_contract=contract,
            execution_config=osm_execution_config(),
            network_hash=record.network_hash,
        )

    # ------------------------------------------------------------------
    # Policy construction
    # ------------------------------------------------------------------
    def build_baseline_policy(self, network_id: str) -> BaselinePolicePolicy:
        """Build the deterministic non-learned shortest-path baseline policy."""
        record = self.networks.get(network_id)
        return BaselinePolicePolicy(record.network)

    def build_learned_policy(
        self,
        network_id: str,
        checkpoint_path: str | os.PathLike[str],
        *,
        manifest: Any | None = None,
    ) -> LearnedPolicePolicy:
        """Build a gated learned OSM policy from a compatible checkpoint.

        Compatibility is checked first; a structurally/semantically incompatible
        checkpoint raises before any actor weights are loaded so the API can map
        the failure to HTTP 409 with the sectioned report (Requirement 12.6).
        """
        record = self.networks.get(network_id)
        return LearnedPolicePolicy.from_checkpoint(
            record.network,
            checkpoint_path,
            manifest=manifest,
            network_hash=record.network_hash,
            clip_distance_m=self.settings.clip_distance_m,
        )

    # ------------------------------------------------------------------
    # Action recommendation with measured latency
    # ------------------------------------------------------------------
    def recommend_actions(
        self,
        network_id: str,
        policy: PolicePolicy,
        *,
        police: Sequence[VehiclePlacement],
        fugitive: VehiclePlacement,
        step: int,
        max_steps: int,
        incoming_headings: Sequence[float | None] | None = None,
        measure_latency: bool | None = None,
    ) -> RecommendationResult:
        """Produce six atomic recommendations and their inference latency.

        Placements are validated against the registered network first; the
        latency of generating all six police actions is measured with
        ``perf_counter_ns`` for learned policies and reported as ``not_measured``
        for the non-inference baseline (Requirements 12.2, 10.5-10.6, 10.9).
        """
        record = self.networks.get(network_id)
        validate_vehicle_placements(record.network, police, fugitive)

        def _recommend() -> tuple[ActionRecommendation, ...]:
            return policy.recommend(
                police=police,
                fugitive=fugitive,
                step=step,
                max_steps=max_steps,
                incoming_headings=incoming_headings,
            )

        recommendations = _recommend()
        experimental = bool(getattr(policy, "experimental", False))
        is_learned = isinstance(policy, LearnedPolicePolicy)
        should_measure = is_learned if measure_latency is None else measure_latency
        if should_measure:
            from .metrics import measure_inference_latency

            latency = measure_inference_latency(
                _recommend,
                sample_count=self.settings.latency_sample_count,
                warmup=self.settings.latency_warmup,
            )
        else:
            latency = not_measured_latency()

        report_id = recommendations[0].compatibility_report_id if recommendations else ""
        return RecommendationResult(
            network_id=network_id,
            profile=str(getattr(policy, "profile", "unknown")),
            experimental=experimental,
            compatibility_report_id=report_id,
            recommendations=recommendations,
            latency=latency,
        )

    # ------------------------------------------------------------------
    # Deterministic episode runs
    # ------------------------------------------------------------------
    def run_episodes(
        self,
        network_id: str,
        config: EpisodeConfig,
        policy: PolicePolicy,
        *,
        run_seed: int,
        episode_count: int,
        mode: RunMode = RunMode.BASELINE,
        policy_kind: str | None = None,
        observation_profile: str | None = None,
        checkpoint_hash: str | None = None,
        plan_id: str | None = None,
        lineage: RunLineage | None = None,
        claim_status: ClaimStatus = ClaimStatus.EXPERIMENTAL,
        fugitive_factory: FugitiveFactory | None = None,
        latency: Mapping[str, Any] | None = None,
    ) -> RunResult:
        """Run a deterministic batch and persist an addressable run result.

        Returns the run identifier, immutable input hashes, the deterministic
        seed list and the result-location for later metric/export retrieval
        (Requirement 12.3).  The run identifier is a content hash over the run's
        identity inputs, so re-running the same request is idempotent.
        """
        record = self.networks.get(network_id)
        profile = observation_profile or str(getattr(policy, "profile", "unknown"))
        kind = policy_kind or str(getattr(policy, "profile", "unknown"))

        run_seed = _require_int(run_seed, "run_seed")
        episode_count = _require_positive_int(episode_count, "episode_count")

        seeds = (run_seed,) + tuple(
            derive_stream_seed(run_seed, index, "episode") for index in range(episode_count)
        )
        run_id = content_hash(
            {
                "network_hash": record.network_hash,
                "config": config,
                "mode": mode.value,
                "policy_kind": kind,
                "observation_profile": profile,
                "run_seed": run_seed,
                "episode_count": episode_count,
                "checkpoint_hash": checkpoint_hash or "",
                "plan_id": plan_id or "",
            }
        )

        records = run_batch(
            record.network,
            config,
            policy,
            run_id=run_id,
            run_seed=run_seed,
            episode_count=episode_count,
            fugitive_factory=fugitive_factory,
        )
        metrics = summarize_metrics(
            records,
            record.network,
            policy_kind=kind,
            minimum_evaluation_episodes=self.settings.minimum_evaluation_episodes,
            confidence_interval_method=self.settings.confidence_interval_method,
            latency=latency,
        )
        input_hashes = {
            "network": record.network_hash,
            "config": content_hash(config),
            "metrics": content_hash(metrics),
        }
        manifest = RunManifest(
            run_id=run_id,
            mode=mode,
            network_hash=record.network_hash,
            observation_profile=profile,
            seeds=seeds,
            policy_kind=kind,
            claim_status=claim_status,
            checkpoint_hash=checkpoint_hash,
            plan_id=plan_id,
            artifacts=input_hashes,
            run_config={"episode_count": episode_count, "network_id": network_id},
            lineage=lineage,
        )
        self.runs.put(RunEntry(manifest=manifest, records=records, metrics=metrics))
        return RunResult(
            run_id=run_id,
            manifest=manifest,
            metrics=metrics,
            result_location=self.runs.location_for(run_id),
        )

    def get_run(self, run_id: str) -> RunEntry:
        return self.runs.get(run_id)

    def get_run_metrics(self, run_id: str) -> MetricsSummary:
        return self.runs.get(run_id).metrics

    # ------------------------------------------------------------------
    # Competition export
    # ------------------------------------------------------------------
    def create_export(
        self,
        run_id: str,
        *,
        export_id: str,
        destination: str | os.PathLike[str],
        report: ExportReport,
        acquired_at: str,
        claims: Sequence[ClaimRecord] = (),
        representative_episode_index: int = 0,
        generated_at: str | None = None,
        max_frames: int | None = None,
        renderer: OSMRenderer | None = None,
    ) -> ExportEntry:
        """Render and publish an immutable competition export for a run.

        The run's episode records and aggregate metrics are rendered into ordered
        frames and summary figures and published as a content-addressed, immutable
        export directory whose manifest preserves OSM attribution, input hashes,
        the code version and claim statuses (Requirements 12.1, 11.5-11.8).
        """
        if self.exports.exists(export_id):
            return self.exports.get(export_id)

        entry = self.runs.get(run_id)
        if not entry.records:
            raise DomainValidationError(
                "EMPTY_RUN", "Run produced no episodes to export", actual=run_id
            )
        if not 0 <= representative_episode_index < len(entry.records):
            raise DomainValidationError(
                "INVALID_EPISODE_INDEX",
                "representative_episode_index is out of range",
                expected=[0, len(entry.records) - 1],
                actual=representative_episode_index,
            )
        network_id = str(entry.manifest.run_config.get("network_id", ""))
        record = self.networks.get(network_id)
        config = _config_from_records(entry.records)

        active_renderer = renderer or OSMRenderer(record.network, config)
        exporter = CompetitionExporter(
            active_renderer,
            code_version=self.settings.code_version,
            attribution=self.settings.attribution,
        )
        representative = entry.records[representative_episode_index]
        input_hashes = {
            "network": record.network_hash,
            "run": entry.run_id,
            "metrics": content_hash(entry.metrics),
            **{str(key): str(value) for key, value in entry.manifest.artifacts.items()},
        }
        result = exporter.export(
            export_id=export_id,
            record=representative,
            summary=entry.metrics,
            report=report,
            claims=tuple(claims),
            input_hashes=input_hashes,
            destination=destination,
            acquired_at=acquired_at,
            generated_at=generated_at,
            max_frames=max_frames,
        )
        export_entry = ExportEntry(run_id=run_id, result=result)
        self.exports.put(export_entry)
        return export_entry

    def get_export(self, export_id: str) -> ExportEntry:
        return self.exports.get(export_id)


# ---------------------------------------------------------------------------
# Internal helpers (no HTTP/filesystem knowledge)
# ---------------------------------------------------------------------------


def _acquire(source: OSMSource, area: Any, network_type: str) -> RawGraphAcquisition:
    """Fetch a bounded raw graph plus provenance metadata from any source."""
    fetch_with_metadata = getattr(source, "fetch_with_metadata", None)
    if callable(fetch_with_metadata):
        return fetch_with_metadata(area, network_type)
    # A minimal source only exposing ``fetch`` still works: derive metadata by
    # converting through the raw graph's own provenance is not possible, so we
    # require ``fetch_with_metadata`` for online preparation.
    raise DomainValidationError(
        "SOURCE_METADATA_UNAVAILABLE",
        "Online preparation requires an OSM source exposing fetch_with_metadata",
    )


def _coarsen_and_validate(
    acquisition: RawGraphAcquisition, settings: CoarseningSettings
):
    """Coarsen the raw graph and bound/validate it into a PreparedNetwork."""
    coarsened = coarsen_raw_graph(acquisition.graph, settings=settings)
    return prepare_model_network(coarsened)


def _build_metadata(
    acquisition: RawGraphAcquisition,
    statistics: NetworkStatistics,
    settings: DemoServiceSettings,
) -> NetworkMetadata:
    """Enrich acquisition metadata with model statistics and validation reasons."""
    base = acquisition.metadata
    validation = dict(base.validation)
    validation.update(
        {
            "boundary_reachability_preserved": True,
            "validation_reasons": list(statistics.validation_reasons),
        }
    )
    schema_versions = dict(base.schema_versions)
    schema_versions.update(
        {
            "model_network": DOMAIN_SCHEMA_VERSION,
            "mapping_manifest": DOMAIN_SCHEMA_VERSION,
        }
    )
    return replace(
        base,
        settings_hash=settings.preprocessing_settings_hash(),
        schema_versions=schema_versions,
        statistics=dict(statistics.as_dict()),
        reachability_digest=statistics.reachability_digest,
        validation=validation,
    )


def _config_from_records(records: Sequence[EpisodeRecord]) -> EpisodeConfig:
    """Reconstruct a rendering EpisodeConfig from run provenance when possible.

    The episode records do not persist the full EpisodeConfig, so a rendering
    default is used.  Only the metric capture radius affects the drawn capture
    circle; a conservative default keeps offline rendering deterministic.
    """
    return EpisodeConfig(
        dt_s=1.0,
        police_speed_mps=15.0,
        fugitive_speed_mps=13.0,
        capture_radius_m=20.0,
        max_steps=max((len(record.transitions) for record in records), default=1) or 1,
    )


def _require_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DomainValidationError("INVALID_SEED", f"{name} must be an integer", actual=value)
    return value


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DomainValidationError(
            "INVALID_EPISODE_COUNT", f"{name} must be a positive integer", actual=value
        )
    return value


__all__ = (
    "CODE_VERSION",
    "DemoService",
    "DemoServiceSettings",
    "ExportEntry",
    "ExportRepository",
    "FileExportRepository",
    "FileNetworkRepository",
    "FileRunRepository",
    "InMemoryExportRepository",
    "InMemoryNetworkRepository",
    "InMemoryRunRepository",
    "NetworkRepository",
    "PreparedNetworkRecord",
    "RecommendationResult",
    "RunEntry",
    "RunRepository",
    "RunResult",
)

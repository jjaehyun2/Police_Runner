"""FastAPI v1 transport contracts and routes for the OSM road-pursuit demo.

This module is the HTTP delivery mechanism described in design section 11.  It
maps the versioned ``/api/v1`` operations onto the dependency-injected
:class:`~pursuit_evasion_rl.osm_demo.service.DemoService` use cases without
leaking FastAPI, filesystem or OSMnx knowledge into the domain layer.

Design boundaries (Requirements 12.1-12.3):

* Pydantic request/response models validate the transport surface only: they
  reject unknown schema versions, malformed bounding boxes and wrong agent
  counts with HTTP 422 field paths before any domain call runs.
* Every route resolves a domain use case on :class:`DemoService`; the service
  owns orchestration and persistence.  Endpoints never mutate module globals.
* Responses return accepted run identifiers, immutable input hashes, the
  deterministic seed list and result URLs for runs, and six complete atomic
  recommendations plus the measured (or ``not_measured``) inference latency for
  action recommendations.

Error taxonomy (Requirements 12.4-12.8).  Every failure is classified once, in
:func:`_classify_exception`, and rendered as a sanitized JSON body:

* transport/field validation -> HTTP 422 with the offending field path,
* a missing network/run/export/cache identifier -> HTTP 404 with the id,
* a checkpoint/network incompatibility -> HTTP 409 with OSM training guidance,
* an unavailable OSM or model dependency -> HTTP 503,
* any unexpected internal error -> HTTP 500 carrying only a correlation id.

Responses never expose absolute local filesystem paths or secrets: every error
body and every location-bearing success body is passed through :func:`_sanitize`,
which redacts Windows drive paths, UNC paths and POSIX absolute paths.  500
bodies deliberately carry a correlation id and nothing else so an operator can
find the logged stack trace without any sensitive detail crossing the wire.
"""

from __future__ import annotations

import logging
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .canonical import content_hash
from .exports import ExportReport
from .models import (
    DOMAIN_SCHEMA_VERSION,
    POLICE_COUNT,
    SUPPORTED_SCHEMA_VERSIONS,
    BoundedArea,
    ClaimStatus,
    DomainValidationError,
    EpisodeConfig,
    MetricsSummary,
    RunMode,
    VehiclePlacement,
)
from .osm_source import OSMSource
from .service import DemoService, PreparedNetworkRecord, RecommendationResult, RunResult

API_PREFIX = "/api/v1"
API_SCHEMA_VERSION = DOMAIN_SCHEMA_VERSION

logger = logging.getLogger("osm_demo.api_v1")

# Guidance returned with every HTTP 409 so an operator knows the checkpoint must
# be fine-tuned or retrained on OSM data before its results can be claimed
# (Requirement 12.6, design section 11).
OSM_TRAINING_PATH_GUIDANCE = (
    "Checkpoint/network incompatibility blocks verified OSM inference. Follow the "
    "OSM training path: prepare region-disjoint train/validation/test networks, "
    "fine-tune a structurally compatible actor or train from scratch with the "
    "osm_topology_v1 observation profile, then evaluate unseen regions before "
    "promoting any OSM performance claim. Structurally compatible legacy "
    "checkpoints may only run through the explicit experimental route."
)


# ---------------------------------------------------------------------------
# Shared transport base and small value models
# ---------------------------------------------------------------------------


class _StrictModel(BaseModel):
    """Reject unknown request fields so typos surface as 422, not silent drops."""

    model_config = ConfigDict(extra="forbid")


class _VersionedModel(_StrictModel):
    """A request carrying an explicit, validated domain schema version."""

    schema_version: str = Field(default=DOMAIN_SCHEMA_VERSION)

    @field_validator("schema_version")
    @classmethod
    def _supported_schema(cls, value: str) -> str:
        if value not in SUPPORTED_SCHEMA_VERSIONS:
            supported = sorted(SUPPORTED_SCHEMA_VERSIONS)
            raise ValueError(
                f"Unsupported schema version {value!r}; supported versions: {supported}"
            )
        return value


class PlacementModel(_StrictModel):
    """A single vehicle placement on a segment or an intersection."""

    segment_id: int | None = Field(default=None, ge=0)
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    intersection_id: int | None = Field(default=None, ge=0)

    def to_domain(self) -> VehiclePlacement:
        return VehiclePlacement(
            segment_id=self.segment_id,
            progress=self.progress,
            intersection_id=self.intersection_id,
        )


class EpisodeConfigModel(_VersionedModel):
    """Transport form of the deterministic :class:`EpisodeConfig`."""

    dt_s: float = Field(gt=0.0)
    police_speed_mps: float = Field(gt=0.0)
    fugitive_speed_mps: float = Field(gt=0.0)
    capture_radius_m: float = Field(gt=0.0)
    max_steps: int = Field(gt=0)
    deterministic: bool = True

    def to_domain(self) -> EpisodeConfig:
        return EpisodeConfig(
            dt_s=self.dt_s,
            police_speed_mps=self.police_speed_mps,
            fugitive_speed_mps=self.fugitive_speed_mps,
            capture_radius_m=self.capture_radius_m,
            max_steps=self.max_steps,
            deterministic=self.deterministic,
            schema_version=self.schema_version,
        )


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class NetworkOSMRequest(_VersionedModel):
    """Prepare a bounded OSM drive network from explicit coordinates."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "schema_version": DOMAIN_SCHEMA_VERSION,
                "name": "daejeon-demo",
                "north": 36.360,
                "south": 36.340,
                "east": 127.400,
                "west": 127.370,
                "max_area_km2": 12.0,
                "config_version": "1.0",
                "network_type": "drive",
            }
        },
    )

    name: str = Field(min_length=1, description="Human-readable bounded-area name.")
    north: float = Field(description="Northern latitude bound; must exceed 'south'.")
    south: float = Field(description="Southern latitude bound.")
    east: float = Field(description="Eastern longitude bound; must exceed 'west'.")
    west: float = Field(description="Western longitude bound.")
    max_area_km2: float = Field(gt=0.0, description="Maximum permitted query area in km^2.")
    config_version: str = Field(default="1.0", description="Bounded-area preset config version.")
    network_type: str = Field(default="drive", description="OSM network type; drivable roads.")

    @field_validator("north", "south", "east", "west", "max_area_km2")
    @classmethod
    def _finite(cls, value: float) -> float:
        # Pydantic already rejects NaN/inf for these when strict; guard anyway so
        # an invalid bbox is a 422 with a field path rather than a 500 later.
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("coordinate must be a finite number")
        return value

    def to_area(self) -> BoundedArea:
        return BoundedArea(
            name=self.name,
            north=self.north,
            south=self.south,
            east=self.east,
            west=self.west,
            max_area_km2=self.max_area_km2,
            config_version=self.config_version,
            schema_version=self.schema_version,
        )


class NetworkCacheRequest(_StrictModel):
    """Load a verified offline cache bundle by its content-addressed key."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {"cache_key": "d24b5791524514f8321ccf2b23a1cc28f8a60d69"}
        },
    )

    cache_key: str = Field(
        min_length=1, description="Content-addressed offline cache bundle key."
    )


class CompatibilityRequest(_StrictModel):
    """Check a checkpoint against a prepared network."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "network_id": "daejeon-demo",
                "checkpoint_path": "checkpoints/osm_finetune_ep1000.pt",
            }
        },
    )

    network_id: str = Field(min_length=1, description="Prepared network identifier.")
    checkpoint_path: str = Field(
        min_length=1,
        description="Path to the police-actor checkpoint (relative to the service root).",
    )


class RecommendRequest(_VersionedModel):
    """Request six atomic police recommendations for one decision state."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "schema_version": DOMAIN_SCHEMA_VERSION,
                "network_id": "daejeon-demo",
                "police": [
                    {"intersection_id": 0},
                    {"intersection_id": 3},
                    {"intersection_id": 7},
                    {"intersection_id": 11},
                    {"intersection_id": 15},
                    {"intersection_id": 19},
                ],
                "fugitive": {"segment_id": 4, "progress": 0.25},
                "step": 0,
                "max_steps": 200,
            }
        },
    )

    network_id: str = Field(min_length=1, description="Prepared network identifier.")
    police: list[PlacementModel] = Field(
        min_length=POLICE_COUNT,
        max_length=POLICE_COUNT,
        description="Exactly six police placements.",
    )
    fugitive: PlacementModel = Field(description="The single fugitive placement.")
    step: int = Field(ge=0, description="Current episode step index.")
    max_steps: int = Field(gt=0, description="Configured maximum episode steps.")
    incoming_headings: list[float | None] | None = Field(
        default=None,
        description="Optional incoming headings, one per officer plus the fugitive.",
    )
    checkpoint_path: str | None = Field(
        default=None,
        description="Optional learned-policy checkpoint; omit for the baseline policy.",
    )

    @field_validator("incoming_headings")
    @classmethod
    def _headings_length(cls, value: list[float | None] | None) -> list[float | None] | None:
        if value is not None and len(value) != POLICE_COUNT + 1:
            raise ValueError(
                f"incoming_headings must supply {POLICE_COUNT + 1} entries "
                "(one per police officer plus the fugitive)"
            )
        return value


class RunRequest(_VersionedModel):
    """Launch a deterministic episode batch and persist an addressable result."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "schema_version": DOMAIN_SCHEMA_VERSION,
                "network_id": "daejeon-demo",
                "config": {
                    "schema_version": DOMAIN_SCHEMA_VERSION,
                    "dt_s": 1.0,
                    "police_speed_mps": 15.0,
                    "fugitive_speed_mps": 13.0,
                    "capture_radius_m": 20.0,
                    "max_steps": 200,
                    "deterministic": True,
                },
                "run_seed": 20240601,
                "episode_count": 30,
                "mode": "baseline",
            }
        },
    )

    network_id: str = Field(min_length=1, description="Prepared network identifier.")
    config: EpisodeConfigModel = Field(description="Deterministic episode configuration.")
    run_seed: int = Field(description="Root deterministic seed for the batch.")
    episode_count: int = Field(gt=0, description="Number of episodes to run.")
    mode: RunMode = Field(default=RunMode.BASELINE, description="Run mode classification.")


class ExportRequest(_StrictModel):
    """Create (or reuse) an immutable competition export for a run."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "implemented_features": ["bounded OSM prepare", "deterministic runs"],
                "verified_results": [],
                "limitations": ["OSM capture rate not yet verified"],
                "hypotheses": ["six-officer coordination improves capture rate"],
                "training_path": ["osm-finetune", "osm-from-scratch"],
                "representative_episode_index": 0,
                "max_frames": 120,
            }
        },
    )

    implemented_features: list[str] = Field(default_factory=list)
    verified_results: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    training_path: list[str] = Field(default_factory=list)
    representative_episode_index: int = Field(default=0, ge=0)
    max_frames: int | None = Field(default=None, ge=1)

    def to_report(self) -> ExportReport:
        return ExportReport(
            implemented_features=tuple(self.implemented_features),
            verified_results=tuple(self.verified_results),
            limitations=tuple(self.limitations),
            hypotheses=tuple(self.hypotheses),
            training_path=tuple(self.training_path),
        )


# ---------------------------------------------------------------------------
# Response serialization helpers
# ---------------------------------------------------------------------------


def _metrics_to_dict(metrics: MetricsSummary) -> dict[str, Any]:
    """Serialize a :class:`MetricsSummary` into JSON-safe primitives."""
    return {
        "schema_version": metrics.schema_version,
        "outcomes": dict(metrics.outcomes),
        "rates": dict(metrics.rates),
        "confidence_intervals": {
            name: list(bounds) for name, bounds in metrics.confidence_intervals.items()
        },
        "episode_lengths": list(metrics.episode_lengths),
        "latency": dict(metrics.latency),
        "preliminary": metrics.preliminary,
    }


def _network_response(record: PreparedNetworkRecord) -> dict[str, Any]:
    """API-safe network metadata and validation summary (no local paths)."""
    return record.summary()


def _recommendation_response(result: RecommendationResult) -> dict[str, Any]:
    return result.as_dict()


def _run_response(result: RunResult) -> dict[str, Any]:
    payload = result.as_dict()
    payload["result_url"] = f"{API_PREFIX}/runs/{result.run_id}"
    payload["metrics_url"] = f"{API_PREFIX}/runs/{result.run_id}/metrics"
    # ``result_location`` may be a filesystem path for the file-backed repository;
    # redact any absolute local path before returning (Requirement 12.7).
    return _sanitize(payload)


# ---------------------------------------------------------------------------
# Error classification and sanitization (Requirements 12.4-12.8)
# ---------------------------------------------------------------------------

# A missing identifier: any ``*_NOT_FOUND`` domain code plus the classified cache
# absence codes map to HTTP 404 (Requirement 12.5).
NOT_FOUND_SUFFIX = "_NOT_FOUND"
NOT_FOUND_CODES = frozenset({"CACHE_MISS", "CACHE_QUARANTINED"})

# A checkpoint/network incompatibility maps to HTTP 409 with training guidance
# (Requirement 12.6).  These are raised by the compatibility gate and the OSM
# training path when a checkpoint cannot be used as requested.
CONFLICT_CODES = frozenset(
    {
        "INFERENCE_BLOCKED",
        "SEMANTIC_PROFILE_REQUIRED",
        "ACTOR_WEIGHT_LOAD_FAILED",
        "INCOMPATIBLE_PARENT_CHECKPOINT",
        "OUT_DEGREE_EXCEEDS_ACTION_SPACE",
        "BOUNDARY_REACHABILITY_CHANGED",
    }
)

# An unavailable external OSM/model dependency maps to HTTP 503 (Requirement
# 12.6 dependency clause / design section 11).
UNAVAILABLE_CODES = frozenset(
    {
        "OSM_SOURCE_UNAVAILABLE",
        "OSMNX_UNAVAILABLE",
        "SOURCE_METADATA_UNAVAILABLE",
        "TORCH_UNAVAILABLE",
        "DEPENDENCY_UNAVAILABLE",
    }
)

# Absolute-path shapes that must never cross the wire (Requirement 12.7): a
# Windows drive path (``C:\...`` or ``c:/...``), a UNC share (``\\host\share``)
# and a POSIX absolute path with at least one nested segment (``/home/a/b``),
# excluding our own API URLs which are matched and preserved separately.
_WINDOWS_PATH = re.compile(r"[A-Za-z]:[\\/][^\s\"']*")
_UNC_PATH = re.compile(r"\\\\[^\s\"']+")
_POSIX_PATH = re.compile(r"(?<![\w./])/(?:[\w.\-]+/)+[\w.\-]+")
_REDACTED = "<redacted-path>"


def _sanitize_text(value: str) -> str:
    """Redact any absolute local path in a single string (Requirement 12.7)."""
    text = _WINDOWS_PATH.sub(_REDACTED, value)
    text = _UNC_PATH.sub(_REDACTED, text)

    def _posix(match: "re.Match[str]") -> str:
        token = match.group(0)
        # Preserve this API's own relative URLs (e.g. ``/api/v1/runs/<id>``).
        return token if token.startswith(f"{API_PREFIX}/") or token.startswith("/api/") else _REDACTED

    return _POSIX_PATH.sub(_posix, text)


def _sanitize(value: Any) -> Any:
    """Recursively redact absolute local paths from any JSON-safe structure."""
    if isinstance(value, str):
        return _sanitize_text(value)
    if isinstance(value, Mapping):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    return value


def _status_for(code: str) -> int:
    """Map a classified domain error code to its HTTP status."""
    if code.endswith(NOT_FOUND_SUFFIX) or code in NOT_FOUND_CODES:
        return 404
    if code in CONFLICT_CODES:
        return 409
    if code in UNAVAILABLE_CODES:
        return 503
    return 422


def _classify_exception(exc: Exception) -> tuple[int, dict[str, Any]]:
    """Classify any raised exception into ``(status, sanitized_detail)``.

    Domain validation failures carry a stable code that selects 404/409/503/422;
    a missing Python dependency (``ImportError``) is a 503; anything else is an
    unexpected internal error rendered as a 500 whose body is only a correlation
    id, with the full context written to the server log (Requirements 12.4-12.8).
    """
    if isinstance(exc, DomainValidationError):
        code = str(exc.code)
        status = _status_for(code)
        detail = _sanitize(exc.as_dict())
        if status == 409:
            detail["osm_training_path"] = OSM_TRAINING_PATH_GUIDANCE
        return status, detail
    if isinstance(exc, ImportError):
        return 503, {
            "code": "DEPENDENCY_UNAVAILABLE",
            "message": _sanitize_text(str(exc) or "A required dependency is unavailable"),
        }
    correlation_id = uuid.uuid4().hex
    # The stack trace and message stay in the server log only; nothing sensitive
    # (message, arguments, local paths) is echoed to the client (Requirement 12.7).
    logger.error("Unexpected API error [correlation_id=%s]", correlation_id, exc_info=exc)
    return 500, {"code": "INTERNAL_ERROR", "correlation_id": correlation_id}


def _http_error(exc: Exception) -> HTTPException:
    """Convert any exception into a sanitized :class:`HTTPException`."""
    if isinstance(exc, HTTPException):
        # Already-classified transport error (e.g. dependency 503 raised inline).
        detail = _sanitize(exc.detail) if exc.detail is not None else None
        return HTTPException(status_code=exc.status_code, detail=detail)
    status, detail = _classify_exception(exc)
    return HTTPException(status_code=status, detail=detail)


# ---------------------------------------------------------------------------
# OpenAPI error-response examples (Requirement 12.8)
# ---------------------------------------------------------------------------
#
# Each documented error class carries a representative, path-free example body so
# the generated OpenAPI describes the request, the success response and every
# error response class without any absolute local path.

_ERROR_EXAMPLES: dict[int, dict[str, Any]] = {
    404: {
        "description": "A requested network, run, export or cache identifier does not exist.",
        "content": {
            "application/json": {
                "example": {
                    "detail": {
                        "code": "NETWORK_NOT_FOUND",
                        "message": "No network exists for identifier 'daejeon-demo'",
                        "path": None,
                        "expected": None,
                        "actual": "daejeon-demo",
                    }
                }
            }
        },
    },
    409: {
        "description": "The checkpoint is incompatible with the network or observation profile.",
        "content": {
            "application/json": {
                "example": {
                    "detail": {
                        "code": "SEMANTIC_PROFILE_REQUIRED",
                        "message": (
                            "OSM inference requires a matching osm_topology_v1 observation "
                            "profile; use the OSM training path before claiming OSM performance"
                        ),
                        "path": None,
                        "expected": "osm_topology_v1",
                        "actual": "legacy_grid_v0",
                        "osm_training_path": OSM_TRAINING_PATH_GUIDANCE,
                    }
                }
            }
        },
    },
    422: {
        "description": "The request body violates the required schema or a domain constraint.",
        "content": {
            "application/json": {
                "example": {
                    "detail": {
                        "code": "INVALID_BOUNDED_AREA",
                        "message": "north must be greater than south",
                        "path": "body.north",
                        "expected": "north > south",
                        "actual": 36.0,
                    }
                }
            }
        },
    },
    503: {
        "description": "A required OSM or model dependency is unavailable.",
        "content": {
            "application/json": {
                "example": {
                    "detail": {
                        "code": "OSM_SOURCE_UNAVAILABLE",
                        "message": "Bounded OSM acquisition is not configured for this service",
                    }
                }
            }
        },
    },
    500: {
        "description": "An unexpected internal error. The body carries only a correlation id.",
        "content": {
            "application/json": {
                "example": {
                    "detail": {
                        "code": "INTERNAL_ERROR",
                        "correlation_id": "9f2c1d7e4b8a4c1e9a0f3b6d5e7c2a11",
                    }
                }
            }
        },
    },
}


def _responses(*statuses: int) -> dict[int | str, dict[str, Any]]:
    """Assemble the OpenAPI ``responses`` map for the given error classes."""
    return {status: _ERROR_EXAMPLES[status] for status in statuses}


# ---------------------------------------------------------------------------
# Router construction
# ---------------------------------------------------------------------------


def create_router(
    service: DemoService,
    *,
    osm_source: OSMSource | None = None,
    export_root: str | Path | None = None,
) -> APIRouter:
    """Build the ``/api/v1`` router backed by an injected :class:`DemoService`.

    ``osm_source`` supplies bounded OSM acquisition for ``POST /networks/osm``
    (an offline fixture source in tests, a live source in deployment).  When it
    is absent the online-preparation route reports the dependency as
    unavailable.  ``export_root`` is the directory under which immutable
    competition exports are published; a per-router temporary directory is used
    when none is supplied.
    """
    router = APIRouter(prefix=API_PREFIX, tags=["osm-demo-v1"])
    exports_root = Path(export_root) if export_root is not None else Path(
        tempfile.mkdtemp(prefix="osm-demo-exports-")
    )

    @router.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "schema_version": API_SCHEMA_VERSION,
            "osm_source_available": osm_source is not None,
            "registries": {
                "networks": len(service.networks.list_ids()),
                "runs": len(service.runs.list_ids()),
                "exports": len(service.exports.list_ids()),
            },
        }

    @router.post("/networks/osm", status_code=201, responses=_responses(422, 503, 500))
    def prepare_network_osm(request: NetworkOSMRequest) -> dict[str, Any]:
        if osm_source is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "OSM_SOURCE_UNAVAILABLE",
                    "message": "Bounded OSM acquisition is not configured for this service",
                },
            )
        try:
            record = service.prepare_online(
                request.to_area(),
                source=osm_source,
                network_type=request.network_type,
            )
        except Exception as exc:
            raise _http_error(exc) from exc
        return _network_response(record)

    @router.post("/networks/cache", status_code=201, responses=_responses(404, 422, 500))
    def load_network_cache(request: NetworkCacheRequest) -> dict[str, Any]:
        try:
            record = service.prepare_offline(request.cache_key)
        except Exception as exc:  # CacheError is a classified DomainValidationError
            raise _http_error(exc) from exc
        return _network_response(record)

    @router.get("/networks/{network_id}", responses=_responses(404, 500))
    def get_network(network_id: str) -> dict[str, Any]:
        try:
            record = service.get_network(network_id)
        except Exception as exc:
            raise _http_error(exc) from exc
        return _network_response(record)

    @router.post("/compatibility/check", responses=_responses(404, 422, 503, 500))
    def check_compatibility(request: CompatibilityRequest) -> dict[str, Any]:
        try:
            report = service.check_compatibility(
                request.network_id, request.checkpoint_path
            )
        except Exception as exc:
            raise _http_error(exc) from exc
        return _sanitize(report.as_dict())

    @router.post("/actions/recommend", responses=_responses(404, 409, 422, 503, 500))
    def recommend_actions(request: RecommendRequest) -> dict[str, Any]:
        police = [placement.to_domain() for placement in request.police]
        fugitive = request.fugitive.to_domain()
        try:
            if request.checkpoint_path:
                policy = service.build_learned_policy(
                    request.network_id, request.checkpoint_path
                )
            else:
                policy = service.build_baseline_policy(request.network_id)
            result = service.recommend_actions(
                request.network_id,
                policy,
                police=police,
                fugitive=fugitive,
                step=request.step,
                max_steps=request.max_steps,
                incoming_headings=request.incoming_headings,
            )
        except Exception as exc:
            raise _http_error(exc) from exc
        return _recommendation_response(result)

    @router.post("/runs", status_code=202, responses=_responses(404, 409, 422, 503, 500))
    def create_run(request: RunRequest) -> dict[str, Any]:
        try:
            policy = service.build_baseline_policy(request.network_id)
            result = service.run_episodes(
                request.network_id,
                request.config.to_domain(),
                policy,
                run_seed=request.run_seed,
                episode_count=request.episode_count,
                mode=request.mode,
            )
        except Exception as exc:
            raise _http_error(exc) from exc
        return _run_response(result)

    @router.get("/runs/{run_id}", responses=_responses(404, 500))
    def get_run(run_id: str) -> dict[str, Any]:
        try:
            entry = service.get_run(run_id)
        except Exception as exc:
            raise _http_error(exc) from exc
        return {
            "run_id": entry.run_id,
            "mode": entry.manifest.mode.value,
            "network_hash": entry.manifest.network_hash,
            "observation_profile": entry.manifest.observation_profile,
            "policy_kind": entry.manifest.policy_kind,
            "claim_status": entry.manifest.claim_status.value,
            "seeds": list(entry.manifest.seeds),
            "input_hashes": dict(entry.manifest.artifacts),
            "episode_count": len(entry.records),
            "episodes": [
                {"episode_id": record.episode_id, "seed": record.seed, "outcome": record.outcome.value}
                for record in entry.records
            ],
            "metrics_url": f"{API_PREFIX}/runs/{run_id}/metrics",
        }

    @router.get("/runs/{run_id}/metrics", responses=_responses(404, 500))
    def get_run_metrics(run_id: str) -> dict[str, Any]:
        try:
            metrics = service.get_run_metrics(run_id)
        except Exception as exc:
            raise _http_error(exc) from exc
        payload = _metrics_to_dict(metrics)
        # The claim boundary: preliminary summaries are never verified evidence.
        payload["claim_boundary"] = {
            "preliminary": metrics.preliminary,
            "verified": False,
            "note": (
                "Metrics summarize this run only and are not verified OSM "
                "performance evidence without a pre-registered evaluation plan."
            ),
        }
        return payload

    @router.post("/runs/{run_id}/exports", status_code=201, responses=_responses(404, 422, 500))
    def create_export(run_id: str, request: ExportRequest) -> dict[str, Any]:
        report = request.to_report()
        export_id = content_hash(
            {
                "run_id": run_id,
                "report": report.as_dict(),
                "representative_episode_index": request.representative_episode_index,
            }
        )
        acquired_at = datetime.now(timezone.utc).isoformat()
        try:
            entry = service.create_export(
                run_id,
                export_id=export_id,
                destination=exports_root,
                report=report,
                acquired_at=acquired_at,
                representative_episode_index=request.representative_episode_index,
                max_frames=request.max_frames,
            )
        except Exception as exc:
            raise _http_error(exc) from exc
        return _sanitize(
            {
                "export_id": entry.export_id,
                "run_id": entry.run_id,
                "location": service.exports.location_for(entry.export_id),
                "dimensions": list(entry.result.manifest.dimensions),
                "warnings": list(entry.result.warnings),
                "export_url": f"{API_PREFIX}/exports/{entry.export_id}",
            }
        )

    @router.get("/exports/{export_id}", responses=_responses(404, 500))
    def get_export(export_id: str) -> dict[str, Any]:
        try:
            entry = service.get_export(export_id)
        except Exception as exc:
            raise _http_error(exc) from exc
        manifest = entry.result.manifest
        return _sanitize(
            {
                "export_id": entry.export_id,
                "run_id": entry.run_id,
                "location": service.exports.location_for(entry.export_id),
                "dimensions": list(manifest.dimensions),
                "code_version": manifest.code_version,
                "attribution": manifest.attribution,
                "input_hashes": dict(manifest.input_hashes),
                "files": [
                    {
                        "relative_path": file.relative_path,
                        "sha256": file.sha256,
                        "size": file.size,
                        "media_type": file.media_type,
                    }
                    for file in manifest.files
                ],
                "warnings": list(manifest.warnings),
            }
        )

    return router


def create_app(
    service: DemoService,
    *,
    osm_source: OSMSource | None = None,
    export_root: str | Path | None = None,
) -> FastAPI:
    """Build a FastAPI application mounting the ``/api/v1`` router.

    Provided as a convenience for offline tests and local runs; deployment code
    can instead mount :func:`create_router` on an existing application.
    """
    app = FastAPI(title="OSM Road Pursuit Demo", version=API_SCHEMA_VERSION)
    app.include_router(
        create_router(service, osm_source=osm_source, export_root=export_root)
    )

    @app.exception_handler(DomainValidationError)
    async def _domain_error_handler(_: Request, exc: DomainValidationError) -> JSONResponse:
        """Safety net: classify any domain error that escapes a route directly."""
        http = _http_error(exc)
        return JSONResponse(status_code=http.status_code, content={"detail": http.detail})

    @app.exception_handler(Exception)
    async def _unexpected_error_handler(_: Request, exc: Exception) -> JSONResponse:
        """Sanitize any unexpected error to a 500 carrying only a correlation id."""
        http = _http_error(exc)
        return JSONResponse(status_code=http.status_code, content={"detail": http.detail})

    return app


__all__ = (
    "API_PREFIX",
    "API_SCHEMA_VERSION",
    "CompatibilityRequest",
    "EpisodeConfigModel",
    "ExportRequest",
    "NetworkCacheRequest",
    "NetworkOSMRequest",
    "PlacementModel",
    "RecommendRequest",
    "RunRequest",
    "create_app",
    "create_router",
)

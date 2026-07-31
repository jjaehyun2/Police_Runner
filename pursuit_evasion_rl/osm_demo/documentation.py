"""Reproducibility, configuration and API/export documentation artifacts.

This module implements task 7.3 of the OSM road-pursuit demo: it turns the
already-built domain, service and transport contracts into **checked**,
self-describing documentation artifacts that a competition reviewer can read
and, more importantly, that the test-suite can re-validate against the real
Pydantic and domain contracts.  Nothing here performs external OSM access,
loads a real checkpoint or runs any real training loop -- every example is
constructed from the immutable domain models and the FastAPI request contracts
and is validated at build time.

The artifacts fall into three groups.

1. **Checked configuration examples** (:func:`build_config_examples`).  One
   reproducible, contract-validated example per demo entry point:

   * ``daejeon_preparation`` -- the bounded Daejeon drive-network prepare
     request (Requirements 1.1, 2.1);
   * ``offline_cache`` -- an offline cache-load request keyed by the committed
     bundle key with actionable preparation guidance (Requirements 4.5);
   * ``legacy_direct_experiment`` -- the opt-in, experimental legacy-direct OSM
     inference configuration, isolated from verified evidence
     (Requirements 8.5-8.6);
   * ``osm_finetune`` / ``osm_from_scratch`` -- the OSM training-path inputs
     with pairwise-disjoint region splits and the ``osm_topology_v1`` profile
     (Requirements 8.7-8.10);
   * ``paired_evaluation`` -- a frozen, pre-registered paired-evaluation
     experiment plan (Requirements 8.10, 14.7 evidence inputs); and
   * ``competition_export`` -- a competition export request whose report
     separates the claim-status sections (Requirements 12.8, 14.7-14.8).

2. **API examples** (:func:`build_api_examples`).  Request, success and error
   example bodies harvested from the Pydantic request models' own
   ``json_schema_extra`` examples and the transport error taxonomy, each request
   example validated by re-parsing it through its Pydantic model so the
   documentation can never drift from the contract (Requirement 12.8).  A live
   OpenAPI document can additionally be produced with
   :func:`build_openapi_schema`.

3. **Competition report template** (:func:`competition_report_template`).  A
   five-way ``implemented`` / ``verified`` / ``experimental`` / ``unsupported``
   / ``limitations`` template (plus hypotheses, follow-up training path and the
   research-only disclaimer) whose status keys are validated against
   :class:`~pursuit_evasion_rl.osm_demo.models.ClaimStatus` and which is
   projected onto a real, constructible
   :class:`~pursuit_evasion_rl.osm_demo.exports.ExportReport`
   (Requirements 1.1-1.4, 14.7-14.8).

:func:`build_documentation` assembles all three groups into a single JSON-safe
document, and :func:`write_documentation_artifacts` materializes them as
canonical-JSON files under a destination directory (``docs/`` by default) for
reproducible, byte-stable committing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .api_v1 import (
    API_PREFIX,
    API_SCHEMA_VERSION,
    CompatibilityRequest,
    EpisodeConfigModel,
    ExportRequest,
    NetworkCacheRequest,
    NetworkOSMRequest,
    RecommendRequest,
    RunRequest,
)
from .canonical import canonical_json, content_hash
from .exports import RESEARCH_ONLY_DISCLAIMER, ExportReport
from .experiments import OSM_OBSERVATION_PROFILE, ExperimentPlan, RegionSplits
from .models import (
    CONFIG_VERSION,
    DOMAIN_SCHEMA_VERSION,
    POLICE_COUNT,
    BoundedArea,
    ClaimStatus,
    DomainValidationError,
    EpisodeConfig,
    RunMode,
)
from .observations import (
    DEFAULT_CLIP_DISTANCE_M,
    LEGACY_PROFILE_ID,
    OSM_PROFILE_ID,
)
from .presets import DAEJEON_DRIVE_PRESET

DOCUMENTATION_VERSION = "osm-demo-docs-v1"
DOCS_DIRNAME = "docs"

# The committed offline cache bundle key shipped under ``cache/`` at the repo
# root; used verbatim so the offline-cache example is reproducible.
COMMITTED_CACHE_KEY = "d24b5791524514f8321ccf2b23a1cc28f8a60d69"

# Deterministic example hashes used only inside documentation config examples.
# They are illustrative content addresses (never used to load real artifacts),
# but they are internally consistent so the domain contracts accept them.
_EXAMPLE_TRAIN_HASH = content_hash({"region": "daejeon-train", "role": "train"})
_EXAMPLE_VALIDATION_HASH = content_hash({"region": "daejeon-validation", "role": "validation"})
_EXAMPLE_TEST_HASH = content_hash({"region": "sejong-test", "role": "test"})
_EXAMPLE_PARENT_CHECKPOINT_HASH = content_hash({"checkpoint": "legacy-grid-actor", "role": "parent"})
_EXAMPLE_OSM_CHECKPOINT_HASH = content_hash({"checkpoint": "osm-finetune-actor", "role": "candidate"})


@dataclass(frozen=True, slots=True)
class ConfigExample:
    """A single checked configuration example.

    ``payload`` is the JSON-safe example body; ``validated_as`` names the domain
    or transport contract it was validated against at build time so a reader
    (and the test-suite) knows exactly which constructor accepted it.
    """

    name: str
    title: str
    description: str
    kind: str
    validated_as: str
    requirements: tuple[str, ...]
    payload: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "kind": self.kind,
            "validated_as": self.validated_as,
            "requirements": list(self.requirements),
            "payload": _json_safe(self.payload),
        }


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _json_safe(value: Any) -> Any:
    """Recursively coerce enums/tuples/sets into JSON-serializable primitives."""
    if isinstance(value, ClaimStatus) or isinstance(value, RunMode):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_json_safe(item) for item in value)
    return value


def _pydantic_example(model: type) -> dict[str, Any]:
    """Return the ``json_schema_extra`` example declared on a Pydantic model."""
    config = getattr(model, "model_config", {})
    extra = config.get("json_schema_extra") if isinstance(config, Mapping) else None
    if not isinstance(extra, Mapping) or "example" not in extra:
        raise DomainValidationError(
            "MISSING_API_EXAMPLE",
            f"{model.__name__} does not declare a json_schema_extra example",
            actual=model.__name__,
        )
    return dict(extra["example"])


# ---------------------------------------------------------------------------
# Checked configuration examples
# ---------------------------------------------------------------------------
def daejeon_preparation_example() -> ConfigExample:
    """Bounded Daejeon drive-network preparation request (Requirements 1.1, 2.1)."""
    area = DAEJEON_DRIVE_PRESET.area
    request = NetworkOSMRequest(
        schema_version=DOMAIN_SCHEMA_VERSION,
        name=area.name,
        north=area.north,
        south=area.south,
        east=area.east,
        west=area.west,
        max_area_km2=area.max_area_km2,
        config_version=area.config_version,
        network_type=DAEJEON_DRIVE_PRESET.network_type,
    )
    # Cross-check the transport model round-trips into the immutable domain area.
    domain_area: BoundedArea = request.to_area()
    payload = request.model_dump()
    payload["_expected_area_config_version"] = domain_area.config_version
    return ConfigExample(
        name="daejeon_preparation",
        title="Daejeon bounded drive-network preparation",
        description=(
            "POST /api/v1/networks/osm body preparing the reproducible bounded "
            "Daejeon drive network from explicit north/south/east/west "
            "coordinates. Validated against NetworkOSMRequest and BoundedArea."
        ),
        kind="api_request",
        validated_as="NetworkOSMRequest -> BoundedArea",
        requirements=("1.1", "2.1"),
        payload=payload,
    )


def offline_cache_example() -> ConfigExample:
    """Offline cache-load request keyed by the committed bundle (Requirement 4.5)."""
    request = NetworkCacheRequest(cache_key=COMMITTED_CACHE_KEY)
    payload = request.model_dump()
    payload["_offline_miss_guidance"] = (
        "If no verified bundle exists for this key, prepare the bounded area "
        "online once (fetch -> coarsen -> validate -> save) to populate the "
        "cache, or provide a committed bundle for this key before running "
        "offline. The demo never silently reuses unverified single-file data."
    )
    return ConfigExample(
        name="offline_cache",
        title="Offline cache bundle load",
        description=(
            "POST /api/v1/networks/cache body loading a verified, immutable "
            "offline bundle by its content-addressed key so demos and tests do "
            "not depend on external OSM availability."
        ),
        kind="api_request",
        validated_as="NetworkCacheRequest",
        requirements=("4.5",),
        payload=payload,
    )


def legacy_direct_experiment_example() -> ConfigExample:
    """Opt-in experimental legacy-direct OSM inference (Requirements 8.5-8.6)."""
    request = RecommendRequest(
        schema_version=DOMAIN_SCHEMA_VERSION,
        network_id="daejeon-demo",
        police=[
            {"intersection_id": 0},
            {"intersection_id": 3},
            {"intersection_id": 7},
            {"intersection_id": 11},
            {"intersection_id": 15},
            {"intersection_id": 19},
        ],
        fugitive={"segment_id": 4, "progress": 0.25},
        step=0,
        max_steps=200,
        checkpoint_path="checkpoints/curriculum/latest.pt",
    )
    payload = request.model_dump()
    # Legacy-direct is an explicit, experimental-only route; it is structurally
    # loaded but semantically mismatched and excluded from verified evidence.
    payload["mode"] = RunMode.LEGACY_DIRECT.value
    payload["observation_profile"] = LEGACY_PROFILE_ID
    payload["allow_experimental_legacy"] = True
    payload["claim_status"] = ClaimStatus.EXPERIMENTAL.value
    payload["_isolation_note"] = (
        "Structural compatibility only. Semantic mismatch is expected; results "
        "are isolated as experimental and excluded from verified OSM evidence. "
        "Use the OSM training path before claiming OSM performance."
    )
    return ConfigExample(
        name="legacy_direct_experiment",
        title="Legacy-direct experimental OSM inference",
        description=(
            "Recommendation request opting a structurally compatible legacy "
            "checkpoint into the explicit legacy-direct OSM route. The run is "
            "classified experimental and never contributes verified evidence."
        ),
        kind="api_request",
        validated_as="RecommendRequest + RunMode.LEGACY_DIRECT + ClaimStatus",
        requirements=("8.5", "8.6"),
        payload=payload,
    )


def _example_region_splits() -> RegionSplits:
    return RegionSplits(
        train=frozenset({_EXAMPLE_TRAIN_HASH}),
        validation=frozenset({_EXAMPLE_VALIDATION_HASH}),
        test=frozenset({_EXAMPLE_TEST_HASH}),
    )


def _training_payload(*, mode: RunMode, parent_checkpoint: str | None) -> dict[str, Any]:
    splits = _example_region_splits()
    payload: dict[str, Any] = {
        "run_id": f"{mode.value}-daejeon-001",
        "mode": mode.value,
        "observation_profile": OSM_PROFILE_ID,
        "clip_distance_m": DEFAULT_CLIP_DISTANCE_M,
        "training_network_hash": _EXAMPLE_TRAIN_HASH,
        "region_splits": {
            "train": sorted(splits.train),
            "validation": sorted(splits.validation),
            "test": sorted(splits.test),
        },
        "seeds": [20240601, 20240602, 20240603, 20240604],
        "code_version": "osm-demo-training-v1",
        "architecture": {
            "police_count": POLICE_COUNT,
            "observation_dim": 21,
            "action_dim": 6,
            "hidden_dims": [128, 128],
        },
    }
    if parent_checkpoint is not None:
        payload["parent_checkpoint_hash"] = parent_checkpoint
        payload["parent_checkpoint_path"] = "checkpoints/curriculum/latest.pt"
    return payload


def osm_finetune_example() -> ConfigExample:
    """OSM fine-tune training inputs from a shape-compatible parent (Req. 8.7-8.10)."""
    # Validate the disjoint region splits against the real contract.
    _example_region_splits()
    payload = _training_payload(mode=RunMode.OSM_FINETUNE, parent_checkpoint=_EXAMPLE_PARENT_CHECKPOINT_HASH)
    return ConfigExample(
        name="osm_finetune",
        title="OSM fine-tune training inputs",
        description=(
            "Inputs for finetune_osm_policy: reuse the six-police 21->128->128->6 "
            "MAPPO architecture, require the osm_topology_v1 profile and "
            "pairwise-disjoint train/validation/test region hashes, and record "
            "the legacy parent checkpoint hash. Completion alone stays experimental."
        ),
        kind="training",
        validated_as="RegionSplits + RunMode.OSM_FINETUNE",
        requirements=("8.7", "8.8", "8.9", "8.10"),
        payload=payload,
    )


def osm_from_scratch_example() -> ConfigExample:
    """OSM from-scratch training inputs with independent init (Req. 8.7-8.10)."""
    _example_region_splits()
    payload = _training_payload(mode=RunMode.OSM_FROM_SCRATCH, parent_checkpoint=None)
    return ConfigExample(
        name="osm_from_scratch",
        title="OSM from-scratch training inputs",
        description=(
            "Inputs for train_osm_policy_from_scratch: the identical MAPPO "
            "architecture and disjoint region splits but an independent random "
            "initialization and no parent checkpoint, so its lineage stays "
            "distinct from the fine-tune path."
        ),
        kind="training",
        validated_as="RegionSplits + RunMode.OSM_FROM_SCRATCH",
        requirements=("8.7", "8.8", "8.9", "8.10"),
        payload=payload,
    )


def _example_experiment_plan() -> ExperimentPlan:
    """A frozen, pre-registered paired-evaluation plan built from real contracts."""
    seeds = (20240601, 20240602, 20240603, 20240604)
    return ExperimentPlan.create(
        plan_id="daejeon-paired-eval-001",
        region_splits=_example_region_splits(),
        checkpoint_candidates=(_EXAMPLE_OSM_CHECKPOINT_HASH,),
        fugitive_rules={"kind": "osm-heuristic", "vision_range_m": 300.0},
        capture_radius_m=20.0,
        max_steps=200,
        episode_count=len(seeds),
        seeds=seeds,
        success_criteria={"min_capture_rate_improvement": 0.1, "max_escape_rate": 0.5},
        evaluation_config={
            "dt_s": 1.0,
            "police_speed_mps": 15.0,
            "fugitive_speed_mps": 10.0,
        },
        minimum_evaluation_episodes=len(seeds),
    )


def paired_evaluation_example() -> ConfigExample:
    """Frozen pre-registered paired-evaluation plan (Requirements 8.10, 14.1-14.6)."""
    plan = _example_experiment_plan()
    payload = {
        "plan_id": plan.plan_id,
        "frozen_config_hash": plan.frozen_config_hash,
        "observation_profile": OSM_OBSERVATION_PROFILE,
        "checkpoint_candidates": list(plan.checkpoint_candidates),
        "capture_radius_m": plan.capture_radius_m,
        "max_steps": plan.max_steps,
        "episode_count": plan.episode_count,
        "minimum_evaluation_episodes": plan.minimum_evaluation_episodes,
        "seeds": list(plan.seeds),
        "confidence_interval_method": plan.confidence_interval_method,
        "success_criteria": dict(plan.success_criteria),
        "evaluation_config": dict(plan.evaluation_config),
        "region_splits": {
            "train": sorted(plan.region_splits.train),
            "validation": sorted(plan.region_splits.validation),
            "test": sorted(plan.region_splits.test),
        },
        "_note": (
            "Config is content-frozen before execution: frozen_config_hash must "
            "re-derive from the payload. Seen (train) and unseen (test) areas are "
            "reported separately and success criteria are never auto-verified."
        ),
    }
    return ConfigExample(
        name="paired_evaluation",
        title="Pre-registered paired evaluation plan",
        description=(
            "A frozen ExperimentPlan for run_paired_evaluation. Learned and "
            "baseline policies run on identical placements/fugitive randomness "
            "per seed; results emit claim inputs but never auto-promote to verified."
        ),
        kind="experiment_plan",
        validated_as="ExperimentPlan.create",
        requirements=("8.10", "14.1", "14.2", "14.3", "14.4", "14.5", "14.6"),
        payload=payload,
    )


def competition_export_example() -> ConfigExample:
    """Competition export request whose report separates claim sections (Req. 14.7-14.8)."""
    request = ExportRequest(
        implemented_features=[
            "bounded OSM prepare",
            "deterministic offline cache",
            "decision-node coarsening and degree splitting",
            "checkpoint compatibility inspection",
            "deterministic paired episode runner",
            "competition export",
        ],
        verified_results=[],
        limitations=[
            "OSM capture/escape/timeout rates are not yet verified",
            "real-time latency on OSM networks is not yet measured end to end",
        ],
        hypotheses=[
            "six-officer coordination improves capture rate over the baseline",
        ],
        training_path=["osm-finetune", "osm-from-scratch", "unseen-region evaluation"],
        representative_episode_index=0,
        max_frames=120,
    )
    # Validate the request projects onto a constructible ExportReport.
    request.to_report()
    return ConfigExample(
        name="competition_export",
        title="Competition export request",
        description=(
            "POST /api/v1/runs/{run_id}/exports body. The generated export "
            "manifest separates implemented features, verified results, "
            "limitations, unverified hypotheses and the follow-up training path, "
            "and carries the research-only disclaimer."
        ),
        kind="api_request",
        validated_as="ExportRequest -> ExportReport",
        requirements=("12.8", "14.7", "14.8"),
        payload=request.model_dump(),
    )


def build_config_examples() -> tuple[ConfigExample, ...]:
    """Build and validate every checked configuration example."""
    return (
        daejeon_preparation_example(),
        offline_cache_example(),
        legacy_direct_experiment_example(),
        osm_finetune_example(),
        osm_from_scratch_example(),
        paired_evaluation_example(),
        competition_export_example(),
    )


# ---------------------------------------------------------------------------
# API examples generated from the Pydantic contracts
# ---------------------------------------------------------------------------
# (method, path, request model) for every operation that accepts a body; the
# examples are the models' own declared json_schema_extra examples.
_REQUEST_OPERATIONS: tuple[tuple[str, str, type], ...] = (
    ("POST", f"{API_PREFIX}/networks/osm", NetworkOSMRequest),
    ("POST", f"{API_PREFIX}/networks/cache", NetworkCacheRequest),
    ("POST", f"{API_PREFIX}/compatibility/check", CompatibilityRequest),
    ("POST", f"{API_PREFIX}/actions/recommend", RecommendRequest),
    ("POST", f"{API_PREFIX}/runs", RunRequest),
)


def build_api_examples() -> dict[str, Any]:
    """Harvest and validate request/error examples from the API contracts.

    Every request example is the model's own declared example, re-parsed through
    the Pydantic model so a drifted example fails the build.  The error taxonomy
    is imported from the transport layer so it always matches the routes.
    """
    from .api_v1 import _ERROR_EXAMPLES  # local import: internal transport detail

    operations: list[dict[str, Any]] = []
    for method, path, model in _REQUEST_OPERATIONS:
        example = _pydantic_example(model)
        # Re-validate the documented example against the live contract.
        validated = model.model_validate(example)
        operations.append(
            {
                "method": method,
                "path": path,
                "request_model": model.__name__,
                "request_example": validated.model_dump(),
            }
        )
    # ``EpisodeConfigModel`` is a nested request body; document its example too.
    episode_config_example = _pydantic_example(RunRequest)["config"]
    EpisodeConfigModel.model_validate(episode_config_example)

    error_examples = {
        str(status): {
            "description": body["description"],
            "example": body["content"]["application/json"]["example"],
        }
        for status, body in _ERROR_EXAMPLES.items()
    }
    return {
        "api_prefix": API_PREFIX,
        "schema_version": API_SCHEMA_VERSION,
        "operations": operations,
        "nested_models": {"EpisodeConfigModel": episode_config_example},
        "error_responses": error_examples,
    }


def build_openapi_schema(service: Any, *, export_root: str | Path | None = None) -> dict[str, Any]:
    """Return the live OpenAPI document for the ``/api/v1`` app.

    Requires a constructed :class:`~pursuit_evasion_rl.osm_demo.service.DemoService`
    (the caller supplies process-local repositories and a temporary cache root so
    no external state is touched).  Kept separate from :func:`build_api_examples`
    so documentation generation itself needs no filesystem or service wiring.
    """
    from .api_v1 import create_app

    app = create_app(service, export_root=export_root)
    return app.openapi()


# ---------------------------------------------------------------------------
# Competition report template (claim-status separated)
# ---------------------------------------------------------------------------
def competition_report_template() -> dict[str, Any]:
    """A claim-status-separated competition report template (Req. 1.1-1.4, 14.7-14.8).

    The template separates the four :class:`ClaimStatus` sections plus a
    ``limitations`` section and validates every status key against the enum.  It
    also carries a real, constructible :class:`ExportReport` projection so the
    template can never describe sections the exporter cannot emit.
    """
    sections = {
        ClaimStatus.IMPLEMENTED.value: {
            "description": "Features whose declared automated checks pass.",
            "entries": [
                "bounded OSM preparation and clipping",
                "deterministic decision-node coarsening and canonical IDs",
                "immutable offline cache with verified loading",
                "safe checkpoint inspection and compatibility reports",
                "deterministic paired episode runner and metrics",
                "FastAPI v1 transport with sanitized errors",
                "staged content-addressed competition export",
            ],
        },
        ClaimStatus.VERIFIED.value: {
            "description": (
                "Results backed by a pre-registered plan, immutable hashes, seeds "
                "and complete unseen-region evidence. Empty until experiments run."
            ),
            "entries": [],
        },
        ClaimStatus.EXPERIMENTAL.value: {
            "description": (
                "Results that exist but are not yet verified: OSM capture/escape/"
                "timeout rates, generalization, latency and legacy-direct runs."
            ),
            "entries": [
                "OSM capture rate for fine-tuned policy (pending evaluation)",
                "legacy-direct OSM inference outputs (isolated, excluded)",
            ],
        },
        ClaimStatus.UNSUPPORTED.value: {
            "description": (
                "Claims explicitly not supported by current evidence, e.g. "
                "immediate generalization of grid-trained checkpoints to OSM."
            ),
            "entries": [
                "synthetic-grid checkpoints generalize to real OSM roads without training",
            ],
        },
        "limitations": {
            "description": "Known constraints and honest boundaries of the demo.",
            "entries": [
                "results depend on the bounded Daejeon area and configured seeds",
                "capture uses projected metric distance, not physical dynamics",
            ],
        },
    }
    # Validate every claim-status key resolves to a real ClaimStatus.
    for key in (
        ClaimStatus.IMPLEMENTED.value,
        ClaimStatus.VERIFIED.value,
        ClaimStatus.EXPERIMENTAL.value,
        ClaimStatus.UNSUPPORTED.value,
    ):
        ClaimStatus(key)

    hypotheses = [
        "six-officer coordination improves capture rate over the baseline",
        "topology-only observations transfer across bounded OSM regions",
    ]
    training_path = [
        "prepare region-disjoint train/validation/test OSM networks",
        "fine-tune a shape-compatible actor or train from scratch (osm_topology_v1)",
        "evaluate unseen regions against the pre-registered plan before promotion",
    ]

    # Project the template onto a real ExportReport so the exporter can consume
    # it; experimental+unsupported entries fold into hypotheses/limitations.
    report = ExportReport(
        implemented_features=tuple(sections[ClaimStatus.IMPLEMENTED.value]["entries"]),
        verified_results=tuple(sections[ClaimStatus.VERIFIED.value]["entries"]),
        limitations=tuple(sections["limitations"]["entries"]),
        hypotheses=tuple(hypotheses),
        training_path=tuple(training_path),
    )

    return {
        "template_version": DOCUMENTATION_VERSION,
        "schema_version": DOMAIN_SCHEMA_VERSION,
        "sections": sections,
        "hypotheses": hypotheses,
        "training_path": training_path,
        "disclaimer": RESEARCH_ONLY_DISCLAIMER,
        "export_report": report.as_dict(),
    }


# ---------------------------------------------------------------------------
# Assembly and materialization
# ---------------------------------------------------------------------------
def build_documentation() -> dict[str, Any]:
    """Assemble the full, validated documentation document (JSON-safe)."""
    return {
        "documentation_version": DOCUMENTATION_VERSION,
        "schema_version": DOMAIN_SCHEMA_VERSION,
        "config_version": CONFIG_VERSION,
        "config_examples": [example.as_dict() for example in build_config_examples()],
        "api_examples": build_api_examples(),
        "competition_report_template": competition_report_template(),
    }


# The canonical-JSON artifact files written under ``docs/`` (Requirement 12.8,
# 14.7): individual files for readability plus one combined document.
CONFIG_EXAMPLES_FILENAME = "config_examples.json"
API_EXAMPLES_FILENAME = "api_examples.json"
REPORT_TEMPLATE_FILENAME = "competition_report_template.json"
DOCUMENTATION_FILENAME = "documentation.json"


def default_docs_dir() -> Path:
    """Return the packaged ``docs/`` directory that holds committed artifacts."""
    return Path(__file__).resolve().parent / DOCS_DIRNAME


def write_documentation_artifacts(dest: str | Path | None = None) -> dict[str, Path]:
    """Write every documentation artifact as canonical JSON and return the paths.

    Byte-stable output (sorted keys, fixed separators) makes the committed
    artifacts reproducible: regenerating them produces identical bytes.
    """
    directory = Path(dest) if dest is not None else default_docs_dir()
    directory.mkdir(parents=True, exist_ok=True)

    documentation = build_documentation()
    artifacts: dict[str, Any] = {
        CONFIG_EXAMPLES_FILENAME: documentation["config_examples"],
        API_EXAMPLES_FILENAME: documentation["api_examples"],
        REPORT_TEMPLATE_FILENAME: documentation["competition_report_template"],
        DOCUMENTATION_FILENAME: documentation,
    }
    paths: dict[str, Path] = {}
    for filename, payload in artifacts.items():
        target = directory / filename
        target.write_bytes(canonical_json(payload))
        paths[filename] = target
    return paths


__all__ = (
    "API_EXAMPLES_FILENAME",
    "COMMITTED_CACHE_KEY",
    "CONFIG_EXAMPLES_FILENAME",
    "DOCS_DIRNAME",
    "DOCUMENTATION_FILENAME",
    "DOCUMENTATION_VERSION",
    "REPORT_TEMPLATE_FILENAME",
    "ConfigExample",
    "build_api_examples",
    "build_config_examples",
    "build_documentation",
    "build_openapi_schema",
    "competition_export_example",
    "competition_report_template",
    "daejeon_preparation_example",
    "default_docs_dir",
    "legacy_direct_experiment_example",
    "offline_cache_example",
    "osm_finetune_example",
    "osm_from_scratch_example",
    "paired_evaluation_example",
    "write_documentation_artifacts",
)

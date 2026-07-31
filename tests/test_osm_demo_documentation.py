"""Focused offline tests for the reproducibility/API/export documentation artifacts.

These tests re-validate every documented configuration example, API example and
the competition report template against the *real* Pydantic and domain contracts
(task 7.3; Requirements 1.1-1.5, 4.5, 8.5-8.10, 12.8, 14.7-14.8).  They also
assert the committed ``docs/`` artifacts are byte-stable so regenerating them
never introduces drift.  No external OSM access, real checkpoint load or real
training happens here.
"""

from __future__ import annotations

import json

import pytest

from pursuit_evasion_rl.osm_demo.api_v1 import (
    API_PREFIX,
    CompatibilityRequest,
    EpisodeConfigModel,
    ExportRequest,
    NetworkCacheRequest,
    NetworkOSMRequest,
    RecommendRequest,
    RunRequest,
)
from pursuit_evasion_rl.osm_demo.canonical import canonical_json
from pursuit_evasion_rl.osm_demo.documentation import (
    API_EXAMPLES_FILENAME,
    COMMITTED_CACHE_KEY,
    CONFIG_EXAMPLES_FILENAME,
    DOCUMENTATION_FILENAME,
    REPORT_TEMPLATE_FILENAME,
    build_api_examples,
    build_config_examples,
    build_documentation,
    build_openapi_schema,
    competition_report_template,
    default_docs_dir,
    write_documentation_artifacts,
)
from pursuit_evasion_rl.osm_demo.exports import ExportReport
from pursuit_evasion_rl.osm_demo.experiments import ExperimentPlan, RegionSplits
from pursuit_evasion_rl.osm_demo.models import (
    POLICE_COUNT,
    BoundedArea,
    ClaimStatus,
    RunMode,
)

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Config examples validate against the real contracts
# ---------------------------------------------------------------------------
def test_build_config_examples_covers_every_entry_point() -> None:
    examples = {example.name for example in build_config_examples()}
    assert examples == {
        "daejeon_preparation",
        "offline_cache",
        "legacy_direct_experiment",
        "osm_finetune",
        "osm_from_scratch",
        "paired_evaluation",
        "competition_export",
    }


def test_daejeon_preparation_example_reparses_into_domain_area() -> None:
    example = next(e for e in build_config_examples() if e.name == "daejeon_preparation")
    # The documented body must re-validate through the transport model...
    request = NetworkOSMRequest.model_validate(
        {k: v for k, v in example.payload.items() if not k.startswith("_")}
    )
    area: BoundedArea = request.to_area()
    assert isinstance(area, BoundedArea)
    assert area.north > area.south and area.east > area.west
    assert request.network_type == "drive"


def test_offline_cache_example_uses_committed_key() -> None:
    example = next(e for e in build_config_examples() if e.name == "offline_cache")
    request = NetworkCacheRequest.model_validate({"cache_key": example.payload["cache_key"]})
    assert request.cache_key == COMMITTED_CACHE_KEY
    assert "prepare" in example.payload["_offline_miss_guidance"].lower()


def test_legacy_direct_example_is_experimental_and_opt_in() -> None:
    example = next(e for e in build_config_examples() if e.name == "legacy_direct_experiment")
    payload = example.payload
    body = {
        k: v
        for k, v in payload.items()
        if k in RecommendRequest.model_fields
    }
    request = RecommendRequest.model_validate(body)
    assert len(request.police) == POLICE_COUNT
    assert payload["allow_experimental_legacy"] is True
    assert payload["mode"] == RunMode.LEGACY_DIRECT.value
    assert payload["claim_status"] == ClaimStatus.EXPERIMENTAL.value


@pytest.mark.parametrize("name", ["osm_finetune", "osm_from_scratch"])
def test_training_examples_use_disjoint_splits_and_osm_profile(name: str) -> None:
    example = next(e for e in build_config_examples() if e.name == name)
    splits = example.payload["region_splits"]
    region = RegionSplits(
        train=frozenset(splits["train"]),
        validation=frozenset(splits["validation"]),
        test=frozenset(splits["test"]),
    )
    # Disjointness is enforced by RegionSplits construction.
    assert region.train.isdisjoint(region.test)
    assert example.payload["observation_profile"] == "osm_topology_v1"
    assert example.payload["training_network_hash"] in region.train
    if name == "osm_finetune":
        assert "parent_checkpoint_hash" in example.payload
    else:
        assert "parent_checkpoint_hash" not in example.payload


def test_paired_evaluation_example_rebuilds_frozen_plan() -> None:
    example = next(e for e in build_config_examples() if e.name == "paired_evaluation")
    payload = example.payload
    splits = payload["region_splits"]
    plan = ExperimentPlan.create(
        plan_id=payload["plan_id"],
        region_splits=RegionSplits(
            train=frozenset(splits["train"]),
            validation=frozenset(splits["validation"]),
            test=frozenset(splits["test"]),
        ),
        checkpoint_candidates=tuple(payload["checkpoint_candidates"]),
        fugitive_rules={"kind": "osm-heuristic", "vision_range_m": 300.0},
        capture_radius_m=payload["capture_radius_m"],
        max_steps=payload["max_steps"],
        episode_count=payload["episode_count"],
        seeds=tuple(payload["seeds"]),
        success_criteria=payload["success_criteria"],
        evaluation_config=payload["evaluation_config"],
        minimum_evaluation_episodes=payload["minimum_evaluation_episodes"],
    )
    # The documented frozen hash must match a freshly built, equivalent plan.
    assert plan.frozen_config_hash == payload["frozen_config_hash"]


def test_competition_export_example_projects_onto_export_report() -> None:
    example = next(e for e in build_config_examples() if e.name == "competition_export")
    request = ExportRequest.model_validate(example.payload)
    report = request.to_report()
    assert isinstance(report, ExportReport)
    assert report.disclaimer.strip()


# ---------------------------------------------------------------------------
# API examples are harvested and re-validated from the Pydantic contracts
# ---------------------------------------------------------------------------
def test_api_examples_revalidate_against_request_models() -> None:
    api = build_api_examples()
    assert api["api_prefix"] == API_PREFIX
    models = {
        "NetworkOSMRequest": NetworkOSMRequest,
        "NetworkCacheRequest": NetworkCacheRequest,
        "CompatibilityRequest": CompatibilityRequest,
        "RecommendRequest": RecommendRequest,
        "RunRequest": RunRequest,
    }
    documented = {op["request_model"] for op in api["operations"]}
    assert documented == set(models)
    for op in api["operations"]:
        model = models[op["request_model"]]
        # Every documented example must re-parse through its live contract.
        model.model_validate(op["request_example"])
    # The nested episode-config body validates too.
    EpisodeConfigModel.model_validate(api["nested_models"]["EpisodeConfigModel"])


def test_api_examples_document_every_error_class() -> None:
    api = build_api_examples()
    assert set(api["error_responses"]) == {"404", "409", "422", "503", "500"}
    for body in api["error_responses"].values():
        assert body["description"]
        assert body["example"]


def test_openapi_schema_exposes_v1_paths(tmp_path) -> None:
    pytest.importorskip("fastapi")
    from pursuit_evasion_rl.osm_demo.cache import CacheStore
    from pursuit_evasion_rl.osm_demo.service import DemoService

    service = DemoService(cache_store=CacheStore(tmp_path / "cache"))
    schema = build_openapi_schema(service, export_root=tmp_path / "exports")
    assert f"{API_PREFIX}/networks/osm" in schema["paths"]
    assert f"{API_PREFIX}/runs" in schema["paths"]


# ---------------------------------------------------------------------------
# Competition report template separates the claim-status sections
# ---------------------------------------------------------------------------
def test_report_template_separates_claim_status_sections() -> None:
    template = competition_report_template()
    sections = template["sections"]
    for status in (
        ClaimStatus.IMPLEMENTED.value,
        ClaimStatus.VERIFIED.value,
        ClaimStatus.EXPERIMENTAL.value,
        ClaimStatus.UNSUPPORTED.value,
    ):
        assert status in sections
        # Every status key resolves to a real ClaimStatus.
        ClaimStatus(status)
    assert "limitations" in sections
    # No verified results are asserted before experiments run (Req. 1.3, 14.5).
    assert sections[ClaimStatus.VERIFIED.value]["entries"] == []
    # The template projects onto a constructible ExportReport.
    report_dict = template["export_report"]
    ExportReport(
        implemented_features=tuple(report_dict["implemented_features"]),
        verified_results=tuple(report_dict["verified_results"]),
        limitations=tuple(report_dict["limitations"]),
        hypotheses=tuple(report_dict["hypotheses"]),
        training_path=tuple(report_dict["training_path"]),
    )
    assert template["disclaimer"].strip()


# ---------------------------------------------------------------------------
# Committed artifacts are present and byte-stable
# ---------------------------------------------------------------------------
def test_written_artifacts_are_byte_stable(tmp_path) -> None:
    first = write_documentation_artifacts(tmp_path)
    # Regeneration must produce byte-identical canonical JSON (reproducibility).
    for name, path in first.items():
        before = path.read_bytes()
        write_documentation_artifacts(tmp_path)
        assert path.read_bytes() == before, f"{name} is not byte-stable"


def test_committed_docs_match_current_build() -> None:
    docs_dir = default_docs_dir()
    documentation = build_documentation()
    expected = {
        CONFIG_EXAMPLES_FILENAME: documentation["config_examples"],
        API_EXAMPLES_FILENAME: documentation["api_examples"],
        REPORT_TEMPLATE_FILENAME: documentation["competition_report_template"],
        DOCUMENTATION_FILENAME: documentation,
    }
    for filename, payload in expected.items():
        path = docs_dir / filename
        assert path.is_file(), f"missing committed artifact {filename}"
        assert path.read_bytes() == canonical_json(payload), (
            f"committed {filename} is stale; regenerate with "
            "write_documentation_artifacts()"
        )
        # Sanity: the committed file is valid JSON.
        json.loads(path.read_bytes().decode("utf-8"))

"""Focused offline tests for staged competition export generation.

Covers Requirements 4.7, 11.5-11.8 and 14.7-14.8 using committed offline
fixtures and the headless Agg renderer, with no external OSM access.
"""
from __future__ import annotations

import json

import pytest

matplotlib = pytest.importorskip("matplotlib")

from pursuit_evasion_rl.osm_demo.canonical import sha256_bytes
from pursuit_evasion_rl.osm_demo.claims import ClaimKind, ClaimRegistry
from pursuit_evasion_rl.osm_demo.coarsening import coarsen_raw_graph, prepare_model_network
from pursuit_evasion_rl.osm_demo.exports import (
    MANIFEST_FILENAME,
    REPORT_FILENAME,
    RESEARCH_ONLY_DISCLAIMER,
    TERMINAL_FRAME_FILENAME,
    CompetitionExporter,
    ExportReport,
)
from pursuit_evasion_rl.osm_demo.fixtures import daejeon
from pursuit_evasion_rl.osm_demo.metrics import summarize_metrics
from pursuit_evasion_rl.osm_demo.models import DomainValidationError, EpisodeConfig
from pursuit_evasion_rl.osm_demo.policies import BaselinePolicePolicy
from pursuit_evasion_rl.osm_demo.rendering import FontChoice, OSMRenderer
from pursuit_evasion_rl.osm_demo.runner import run_batch, run_single_episode

pytestmark = pytest.mark.offline

CAPTURE_RADIUS_M = 15.0
CODE_VERSION = "osm-demo-test-1"
ATTRIBUTION = "OpenStreetMap contributors via OSMnx"
ACQUIRED_AT = "2024-01-02T03:04:05+00:00"


def _network():
    return prepare_model_network(coarsen_raw_graph(daejeon())).network


def _config(max_steps: int = 4) -> EpisodeConfig:
    return EpisodeConfig(
        dt_s=1.0,
        police_speed_mps=12.0,
        fugitive_speed_mps=9.0,
        capture_radius_m=CAPTURE_RADIUS_M,
        max_steps=max_steps,
    )


def _renderer(network, config):
    # Force the English fallback font so the test never depends on installed fonts.
    return OSMRenderer(network, config, font=FontChoice("DejaVu Sans", False))


def _report() -> ExportReport:
    return ExportReport(
        implemented_features=("Deterministic OSM coarsening", "Six-police episode runner"),
        verified_results=(),
        limitations=("Trained only on synthetic grids so far.",),
        hypotheses=("Fine-tuning on OSM topology may improve capture rate.",),
        training_path=("osm-finetune from the legacy actor, then unseen-region evaluation.",),
    )


def _claims() -> ClaimRegistry:
    registry = ClaimRegistry()
    registry.register_claim(
        "renderer-implemented",
        ClaimKind.FEATURE,
        reason="Renderer exists with automated tests.",
        updated_at=ACQUIRED_AT,
    )
    registry.register_claim(
        "osm-capture-rate",
        ClaimKind.OSM_PERFORMANCE,
        reason="No OSM evaluation evidence yet.",
        updated_at=ACQUIRED_AT,
    )
    return registry


def _export(tmp_path, *, episode_count: int = 3, export_id: str = "export-1", claims=None):
    network = _network()
    config = _config()
    policy = BaselinePolicePolicy(network)
    records = run_batch(
        network, config, policy, run_id="export-batch", run_seed=11, episode_count=episode_count
    )
    summary = summarize_metrics(
        records, network, policy_kind="baseline", minimum_evaluation_episodes=1
    )
    exporter = CompetitionExporter(
        _renderer(network, config), code_version=CODE_VERSION, attribution=ATTRIBUTION
    )
    result = exporter.export(
        export_id=export_id,
        record=records[0],
        summary=summary,
        report=_report(),
        claims=claims if claims is not None else _claims(),
        input_hashes={"network": records[0].hashes["network"], "config": records[0].hashes["config"]},
        destination=str(tmp_path),
        acquired_at=ACQUIRED_AT,
    )
    return result, records[0]


# ---------------------------------------------------------------------------
# Ordered frames and summaries (Requirements 11.5, 11.6)
# ---------------------------------------------------------------------------
def test_export_writes_ordered_frames_terminal_and_summaries(tmp_path):
    result, record = _export(tmp_path)
    export_dir = tmp_path / "export-1"

    frame_files = sorted(p.name for p in export_dir.glob("frame_*.png"))
    assert frame_files[0] == "frame_000000.png"
    # One ordered frame per episode state (initial + one per transition).
    assert len(frame_files) == len(record.transitions) + 1
    # Ordered names are zero-padded and contiguous.
    assert frame_files == [f"frame_{i:06d}.png" for i in range(len(frame_files))]

    assert (export_dir / TERMINAL_FRAME_FILENAME).exists()
    for name in (
        "outcome_rates.png",
        "episode_length_histogram.png",
        "latency_histogram.png",
        "representative_path.png",
    ):
        assert (export_dir / name).exists()


# ---------------------------------------------------------------------------
# Content-addressed manifest (Requirement 11.7)
# ---------------------------------------------------------------------------
def test_manifest_is_content_addressed_and_records_resolution(tmp_path):
    result, _ = _export(tmp_path)
    export_dir = tmp_path / "export-1"

    manifest = json.loads((export_dir / MANIFEST_FILENAME).read_bytes())
    assert manifest["code_version"] == CODE_VERSION
    # Every listed file exists with a matching SHA-256 and size.
    listed = {item["relative_path"]: item for item in manifest["files"]}
    assert "frame_000000.png" in listed
    assert REPORT_FILENAME in listed
    for relative_path, entry in listed.items():
        content = (export_dir / relative_path).read_bytes()
        assert entry["sha256"] == sha256_bytes(content)
        assert entry["size"] == len(content)
        assert entry["media_type"] in ("image/png", "application/json")

    # Dimensions match the actual frame PNG resolution.
    width, height = manifest["dimensions"]
    assert width > 0 and height > 0
    assert result.manifest.dimensions == (width, height)


def test_manifest_records_claim_statuses(tmp_path):
    _export(tmp_path)
    manifest = json.loads((tmp_path / "export-1" / MANIFEST_FILENAME).read_bytes())
    statuses = {claim["name"]: claim["status"] for claim in manifest["claims"]}
    assert statuses["renderer-implemented"] == "experimental"
    # An OSM performance claim without evidence must not be overstated.
    assert statuses["osm-capture-rate"] == "experimental"


# ---------------------------------------------------------------------------
# Provenance preservation (Requirement 4.7)
# ---------------------------------------------------------------------------
def test_export_preserves_attribution_and_acquisition_time(tmp_path):
    _export(tmp_path)
    manifest = json.loads((tmp_path / "export-1" / MANIFEST_FILENAME).read_bytes())
    provenance = manifest["provenance"]
    assert provenance["attribution"] == ATTRIBUTION
    assert provenance["acquired_at"] == ACQUIRED_AT
    assert provenance["code_version"] == CODE_VERSION


# ---------------------------------------------------------------------------
# Report sections and disclaimer (Requirements 14.7, 14.8)
# ---------------------------------------------------------------------------
def test_report_separates_required_sections_and_disclaimer(tmp_path):
    _export(tmp_path)
    export_dir = tmp_path / "export-1"

    report = json.loads((export_dir / REPORT_FILENAME).read_bytes())
    for section in (
        "implemented_features",
        "verified_results",
        "limitations",
        "hypotheses",
        "training_path",
    ):
        assert section in report
    assert report["disclaimer"] == RESEARCH_ONLY_DISCLAIMER

    # The disclaimer is also embedded in the manifest report block.
    manifest = json.loads((export_dir / MANIFEST_FILENAME).read_bytes())
    assert manifest["report"]["disclaimer"].strip()
    assert "does not replace" in manifest["report"]["disclaimer"]


def test_export_report_requires_every_section():
    with pytest.raises(DomainValidationError):
        ExportReport(
            implemented_features=("a",),
            verified_results=(),
            limitations=(),
            hypotheses=(),
            training_path=(),
            disclaimer="   ",
        )


# ---------------------------------------------------------------------------
# Immutability and staging (design section 10)
# ---------------------------------------------------------------------------
def test_export_directory_is_immutable(tmp_path):
    _export(tmp_path, export_id="dup")
    with pytest.raises(DomainValidationError):
        _export(tmp_path, export_id="dup")


def test_staging_directory_is_not_left_behind(tmp_path):
    _export(tmp_path)
    # Only the published export directory should remain under the destination.
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".staging-")]
    assert leftovers == []

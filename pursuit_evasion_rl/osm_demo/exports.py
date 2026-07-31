"""Staged, content-addressed competition export generation.

This module turns a rendered OSM pursuit episode plus its aggregate metrics into
an immutable competition export directory (design section 10, "Matplotlib
rendering and competition export"; Requirements 4.7, 11.5-11.8, 14.7-14.8).

The exporter:

* renders ordered start-to-terminal PNG frames (``frame_000000.png`` onward) and
  an explicit ``terminal_frame.png`` (Requirement 11.5),
* renders the outcome-rate, episode-length, measured-latency and representative
  path summary figures (Requirement 11.6),
* writes a content-addressed ``export_manifest.json`` recording every file's
  SHA-256, size and media type, the frame image resolution, the generating code
  version, the input hashes and each recorded :class:`ClaimRecord` status
  (Requirement 11.7),
* preserves the OSM attribution and the data acquisition/generation time as
  provenance (Requirement 4.7),
* embeds a competition report separating implemented features, verified results,
  known limitations, unverified hypotheses and the follow-up OSM training path
  (Requirement 14.7) together with a research-only disclaimer (Requirement 14.8).

Finalization is *staged*: everything is written into an isolated temporary
directory and only atomically renamed to the published, immutable export
directory once the manifest has been written last.  A destination that already
exists is never overwritten.  The renderer forces the headless Agg backend so
every default test runs offline (Requirement 13.12).
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .canonical import canonical_json, sha256_bytes
from .claims import ClaimRegistry
from .models import (
    ClaimRecord,
    DomainValidationError,
    EpisodeRecord,
    ExportFile,
    ExportManifest,
    MetricsSummary,
)
from .rendering import OSMRenderer

EXPORTER_VERSION = "osm-export-v1"

MANIFEST_FILENAME = "export_manifest.json"
REPORT_FILENAME = "report.json"
TERMINAL_FRAME_FILENAME = "terminal_frame.png"

PNG_MEDIA_TYPE = "image/png"
JSON_MEDIA_TYPE = "application/json"

# Requirement 14.8: the export must state this is a research aid only.
RESEARCH_ONLY_DISCLAIMER = (
    "이 자료는 경찰 지휘 보조 연구 데모이며 실제 현장 지휘 결정을 대체하지 않는다. "
    "This is a police-command support research demo and does not replace real "
    "field command decisions."
)

# Ordered summary figure file names (Requirement 11.6).
OUTCOME_RATES_FILENAME = "outcome_rates.png"
EPISODE_LENGTH_FILENAME = "episode_length_histogram.png"
LATENCY_FILENAME = "latency_histogram.png"
REPRESENTATIVE_PATH_FILENAME = "representative_path.png"


@dataclass(frozen=True, slots=True)
class ExportReport:
    """The non-overstated competition report sections (Requirements 14.7, 14.8).

    Every section must be supplied so that finalization fails when a required
    section is missing.  Section entries may be empty (for example, no verified
    results yet) but the section itself must be present, and the research-only
    disclaimer must be non-empty.
    """

    implemented_features: tuple[str, ...]
    verified_results: tuple[str, ...]
    limitations: tuple[str, ...]
    hypotheses: tuple[str, ...]
    training_path: tuple[str, ...]
    disclaimer: str = RESEARCH_ONLY_DISCLAIMER

    def __post_init__(self) -> None:
        for name in (
            "implemented_features",
            "verified_results",
            "limitations",
            "hypotheses",
            "training_path",
        ):
            value = getattr(self, name)
            if value is None:
                raise DomainValidationError(
                    "MISSING_REPORT_SECTION",
                    f"Competition report section {name!r} is required",
                    path=name,
                )
            object.__setattr__(self, name, tuple(str(item) for item in value))
        if not isinstance(self.disclaimer, str) or not self.disclaimer.strip():
            raise DomainValidationError(
                "MISSING_RESEARCH_DISCLAIMER",
                "Competition report requires a non-empty research-only disclaimer",
                path="disclaimer",
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "implemented_features": list(self.implemented_features),
            "verified_results": list(self.verified_results),
            "limitations": list(self.limitations),
            "hypotheses": list(self.hypotheses),
            "training_path": list(self.training_path),
            "disclaimer": self.disclaimer,
        }


@dataclass(frozen=True, slots=True)
class ExportResult:
    """The published, immutable export directory and its structured manifest."""

    export_dir: str
    manifest_path: str
    manifest: ExportManifest
    report: ExportReport
    warnings: tuple[str, ...]


def _png_dimensions(path: Path) -> tuple[int, int]:
    """Return ``(width, height)`` in pixels parsed from a PNG IHDR chunk."""
    with open(path, "rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise DomainValidationError(
            "INVALID_PNG", f"{path.name} is not a valid PNG image", actual=path.name
        )
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    return (width, height)


def _claim_as_dict(claim: ClaimRecord) -> dict[str, object]:
    return {
        "name": claim.name,
        "status": claim.status.value,
        "evidence_ids": list(claim.evidence_ids),
        "reason": claim.reason,
        "updated_at": claim.updated_at,
    }


def _resolve_claims(claims: Sequence[ClaimRecord] | ClaimRegistry) -> tuple[ClaimRecord, ...]:
    if isinstance(claims, ClaimRegistry):
        return tuple(claims.claims)
    resolved = tuple(claims)
    if any(not isinstance(item, ClaimRecord) for item in resolved):
        raise DomainValidationError(
            "INVALID_CLAIM_RECORD", "Export claims must be ClaimRecord instances"
        )
    return resolved


def _describe_file(base: Path, path: Path) -> ExportFile:
    content = path.read_bytes()
    media_type = PNG_MEDIA_TYPE if path.suffix == ".png" else JSON_MEDIA_TYPE
    return ExportFile(
        relative_path=path.relative_to(base).as_posix(),
        sha256=sha256_bytes(content),
        size=len(content),
        media_type=media_type,
    )


class CompetitionExporter:
    """Render and publish an immutable, content-addressed competition export."""

    def __init__(
        self,
        renderer: OSMRenderer,
        *,
        code_version: str,
        attribution: str,
        exporter_version: str = EXPORTER_VERSION,
    ) -> None:
        if not isinstance(renderer, OSMRenderer):
            raise DomainValidationError("INVALID_RENDERER", "An OSMRenderer instance is required")
        if not str(code_version).strip():
            raise DomainValidationError("MISSING_CODE_VERSION", "code_version must be nonempty")
        if not str(attribution).strip():
            raise DomainValidationError("MISSING_ATTRIBUTION", "OSM attribution must be preserved")
        self.renderer = renderer
        self.code_version = str(code_version)
        self.attribution = str(attribution)
        self.exporter_version = str(exporter_version)

    # ------------------------------------------------------------------
    # Figure rendering into the staging directory
    # ------------------------------------------------------------------
    def _write_frames(self, staging: Path, record: EpisodeRecord, *, max_frames: int | None) -> int:
        """Render ordered start-to-terminal frames plus an explicit terminal frame."""
        figures = self.renderer.render_episode_frames(record, max_frames=max_frames)
        if not figures:
            raise DomainValidationError(
                "EMPTY_EPISODE", "Episode produced no frames to export", actual=record.episode_id
            )
        last_index = len(figures) - 1
        for index, figure in enumerate(figures):
            frame_path = staging / f"frame_{index:06d}.png"
            if index == last_index:
                # The final ordered frame is also published as the terminal frame;
                # save it under both names before the renderer closes it.
                figure.savefig(str(staging / TERMINAL_FRAME_FILENAME), dpi=self.renderer.style.dpi)
            self.renderer.save_figure(figure, str(frame_path))
        return len(figures)

    def _write_summaries(
        self, staging: Path, summary: MetricsSummary, record: EpisodeRecord
    ) -> None:
        """Render the four competition summary figures (Requirement 11.6)."""
        self.renderer.save_figure(
            self.renderer.render_outcome_rates(summary), str(staging / OUTCOME_RATES_FILENAME)
        )
        self.renderer.save_figure(
            self.renderer.render_episode_length_histogram(summary.episode_lengths),
            str(staging / EPISODE_LENGTH_FILENAME),
        )
        self.renderer.save_figure(
            self.renderer.render_latency_histogram(summary.latency.get("samples_ms", ())),
            str(staging / LATENCY_FILENAME),
        )
        self.renderer.save_figure(
            self.renderer.render_representative_path(record),
            str(staging / REPRESENTATIVE_PATH_FILENAME),
        )

    # ------------------------------------------------------------------
    # Manifest assembly
    # ------------------------------------------------------------------
    def _collect_warnings(self, summary: MetricsSummary) -> tuple[str, ...]:
        warnings: list[str] = []
        if self.renderer.font.warning:
            warnings.append(self.renderer.font.warning)
        if summary.preliminary:
            warnings.append(
                "Metrics are preliminary: fewer episodes than the configured minimum; "
                "OSM performance claims remain experimental."
            )
        return tuple(warnings)

    def _build_manifest(
        self,
        *,
        export_id: str,
        staging: Path,
        input_hashes: Mapping[str, str],
        claims: tuple[ClaimRecord, ...],
        warnings: tuple[str, ...],
    ) -> ExportManifest:
        files = tuple(
            sorted(
                (
                    _describe_file(staging, path)
                    for path in staging.rglob("*")
                    if path.is_file()
                ),
                key=lambda item: item.relative_path,
            )
        )
        dimensions = _png_dimensions(staging / "frame_000000.png")
        return ExportManifest(
            export_id=export_id,
            input_hashes=dict(input_hashes),
            files=files,
            dimensions=dimensions,
            code_version=self.code_version,
            claims=claims,
            attribution=self.attribution,
            warnings=warnings,
        )

    def _manifest_document(
        self, manifest: ExportManifest, report: ExportReport, *, acquired_at: str, generated_at: str
    ) -> dict[str, object]:
        """The on-disk manifest superset: identity content plus provenance/report."""
        return {
            "schema_version": manifest.schema_version,
            "export_id": manifest.export_id,
            "exporter_version": self.exporter_version,
            "code_version": manifest.code_version,
            "dimensions": list(manifest.dimensions),
            "input_hashes": dict(manifest.input_hashes),
            "claims": [_claim_as_dict(claim) for claim in manifest.claims],
            "warnings": list(manifest.warnings),
            "files": [
                {
                    "relative_path": item.relative_path,
                    "sha256": item.sha256,
                    "size": item.size,
                    "media_type": item.media_type,
                }
                for item in manifest.files
            ],
            # Provenance preserved in the export (Requirement 4.7).
            "provenance": {
                "attribution": manifest.attribution,
                "acquired_at": acquired_at,
                "generated_at": generated_at,
                "exporter_version": self.exporter_version,
                "code_version": manifest.code_version,
            },
            # Competition report sections (Requirements 14.7, 14.8).
            "report": report.as_dict(),
        }

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def export(
        self,
        *,
        export_id: str,
        record: EpisodeRecord,
        summary: MetricsSummary,
        report: ExportReport,
        claims: Sequence[ClaimRecord] | ClaimRegistry,
        input_hashes: Mapping[str, str],
        destination: str | os.PathLike[str],
        acquired_at: str,
        generated_at: str | None = None,
        max_frames: int | None = None,
    ) -> ExportResult:
        """Render and atomically publish an immutable competition export.

        Everything is written into an isolated temporary directory first; the
        content-addressed ``export_manifest.json`` is written last, then the
        staging directory is atomically renamed to ``destination/export_id``.  A
        destination that already exists is never overwritten (immutability).
        """
        if not str(export_id).strip():
            raise DomainValidationError("MISSING_EXPORT_ID", "export_id must be nonempty")
        if not isinstance(report, ExportReport):
            raise DomainValidationError("INVALID_REPORT", "An ExportReport instance is required")
        if not isinstance(summary, MetricsSummary):
            raise DomainValidationError("INVALID_SUMMARY", "A MetricsSummary instance is required")
        if not isinstance(record, EpisodeRecord):
            raise DomainValidationError("INVALID_RECORD", "An EpisodeRecord instance is required")
        if not str(acquired_at).strip():
            raise DomainValidationError(
                "MISSING_ACQUISITION_TIME", "OSM data acquisition/generation time must be preserved"
            )

        resolved_claims = _resolve_claims(claims)
        parent = Path(destination)
        final_dir = parent / str(export_id)
        if final_dir.exists():
            raise DomainValidationError(
                "EXPORT_ALREADY_EXISTS",
                f"Export directory {final_dir} already exists and is immutable",
                actual=str(final_dir),
            )
        parent.mkdir(parents=True, exist_ok=True)

        # Isolated staging directory on the same filesystem as the destination so
        # the final publish is a cheap atomic rename.
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=parent))
        try:
            self._write_frames(staging, record, max_frames=max_frames)
            self._write_summaries(staging, summary, record)

            # Report sections file (content-addressed via the manifest).
            (staging / REPORT_FILENAME).write_bytes(canonical_json(report.as_dict()))

            warnings = self._collect_warnings(summary)
            manifest = self._build_manifest(
                export_id=str(export_id),
                staging=staging,
                input_hashes=input_hashes,
                claims=resolved_claims,
                warnings=warnings,
            )
            document = self._manifest_document(
                manifest,
                report,
                acquired_at=str(acquired_at),
                generated_at=str(generated_at) if generated_at is not None else str(acquired_at),
            )
            # Written last so its presence marks a complete, publishable export.
            (staging / MANIFEST_FILENAME).write_bytes(canonical_json(document))

            os.rename(staging, final_dir)
        except BaseException:
            _remove_tree(staging)
            raise

        return ExportResult(
            export_dir=str(final_dir),
            manifest_path=str(final_dir / MANIFEST_FILENAME),
            manifest=manifest,
            report=report,
            warnings=manifest.warnings,
        )


def _remove_tree(path: Path) -> None:
    """Best-effort recursive cleanup of a staging directory."""
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), reverse=True):
        try:
            if child.is_dir():
                child.rmdir()
            else:
                child.unlink()
        except OSError:  # pragma: no cover - defensive cleanup
            pass
    try:
        path.rmdir()
    except OSError:  # pragma: no cover - defensive cleanup
        pass


__all__ = (
    "EXPORTER_VERSION",
    "JSON_MEDIA_TYPE",
    "MANIFEST_FILENAME",
    "PNG_MEDIA_TYPE",
    "REPORT_FILENAME",
    "RESEARCH_ONLY_DISCLAIMER",
    "TERMINAL_FRAME_FILENAME",
    "CompetitionExporter",
    "ExportReport",
    "ExportResult",
)

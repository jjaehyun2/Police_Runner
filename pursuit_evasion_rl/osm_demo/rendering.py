"""Matplotlib rendering for OSM pursuit episode frames and summary figures.

This module draws projected (metric) OSM road-pursuit episodes for the
competition demo (design section 10, "Matplotlib rendering and competition
export"; Requirements 11.1-11.4, 11.6, 11.8).  It renders, in projected metric
coordinates:

* directed road polylines with direction arrows (Requirement 11.1),
* six distinctly styled police, the fugitive, each vehicle's current segment and
  its accumulated trail (Requirement 11.2),
* a scale-correct capture-radius circle drawn on an equal-aspect axis so the
  circle matches the map scale (Requirement 11.3),
* terminal annotations carrying the Episode_Outcome, episode identifier,
  Deterministic_Seed and current step (Requirement 11.4).

It also produces the summary figures (Requirement 11.6): outcome-rate bars, an
episode-length histogram, a measured-latency histogram and a representative
path map.

Fonts are discovered deterministically (Requirement 11.8): a fixed ordered list
of Korean font families is probed against the available fonts and the first
match is used; when none is available the renderer falls back to an English
label set and records a single warning instead of dropping glyphs.

The module forces the headless Agg backend so every default test renders
offline without a display (Requirement 13.12).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import matplotlib

# Force a non-interactive, headless backend before importing pyplot so the
# renderer works in offline test environments with no display server.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow backend selection)
from matplotlib import font_manager  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from .metrics import placement_position  # noqa: E402
from .models import (  # noqa: E402
    POLICE_COUNT,
    DomainValidationError,
    EpisodeConfig,
    EpisodeOutcome,
    EpisodeRecord,
    EpisodeState,
    MetricsSummary,
    ModelNetwork,
)

logger = logging.getLogger(__name__)

RENDERER_VERSION = "osm-renderer-v1"

# Six visually distinct police colours/markers plus a contrasting fugitive.
POLICE_COLORS: tuple[str, ...] = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#17becf",
)
POLICE_MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P")
FUGITIVE_COLOR = "#d62728"
FUGITIVE_MARKER = "*"

# Ordered Korean font candidates probed against the installed font families.
KOREAN_FONT_CANDIDATES: tuple[str, ...] = (
    "Malgun Gothic",
    "NanumGothic",
    "NanumBarunGothic",
    "NanumSquare",
    "AppleGothic",
    "Apple SD Gothic Neo",
    "Noto Sans CJK KR",
    "Noto Sans KR",
    "Source Han Sans KR",
    "Gulim",
    "Dotum",
    "Batang",
    "UnDotum",
)
# DejaVu Sans ships with matplotlib, so the English fallback always resolves.
ENGLISH_FALLBACK_FONT = "DejaVu Sans"

# (korean, english) label pairs selected by :class:`_Localizer`.
_LABELS: Mapping[str, tuple[str, str]] = {
    "police": ("경찰", "Police"),
    "fugitive": ("도주자", "Fugitive"),
    "capture_radius": ("체포 반경", "Capture radius"),
    "outcome": ("결과", "Outcome"),
    "episode": ("에피소드", "Episode"),
    "seed": ("시드", "Seed"),
    "step": ("스텝", "Step"),
    "in_progress": ("진행 중", "in progress"),
    "easting_m": ("동쪽 거리 (m)", "Easting (m)"),
    "northing_m": ("북쪽 거리 (m)", "Northing (m)"),
    "outcome_rates": ("결과 비율", "Outcome rates"),
    "rate": ("비율", "Rate"),
    "episode_length": ("에피소드 길이", "Episode length"),
    "steps": ("스텝 수", "Steps"),
    "count": ("빈도", "Count"),
    "latency": ("추론 지연시간 (ms)", "Inference latency (ms)"),
    "representative_path": ("대표 경로", "Representative path"),
    "no_data": ("데이터 없음", "No data"),
    "capture": ("검거", "Capture"),
    "escape": ("탈출", "Escape"),
    "timeout": ("시간 초과", "Timeout"),
    "start": ("시작", "start"),
    "end": ("끝", "end"),
}


@dataclass(frozen=True, slots=True)
class FontChoice:
    """The resolved label font and whether Korean glyphs are available."""

    family: str
    supports_korean: bool
    warning: str | None = None


class _Localizer:
    """Select Korean labels when available, otherwise the English fallback."""

    def __init__(self, supports_korean: bool) -> None:
        self._index = 0 if supports_korean else 1

    def get(self, key: str) -> str:
        try:
            return _LABELS[key][self._index]
        except KeyError as exc:  # pragma: no cover - defensive
            raise DomainValidationError("UNKNOWN_LABEL", f"No label for {key!r}") from exc


@dataclass(frozen=True, slots=True)
class RenderStyle:
    """Colours, markers and figure geometry for episode rendering."""

    police_colors: tuple[str, ...] = POLICE_COLORS
    police_markers: tuple[str, ...] = POLICE_MARKERS
    fugitive_color: str = FUGITIVE_COLOR
    fugitive_marker: str = FUGITIVE_MARKER
    road_color: str = "#9aa0a6"
    active_segment_color: str = "#d62728"
    capture_circle_color: str = "#2ca02c"
    trail_alpha: float = 0.55
    dpi: int = 150
    figure_size: tuple[float, float] = (8.0, 8.0)

    def __post_init__(self) -> None:
        if len(self.police_colors) < POLICE_COUNT or len(self.police_markers) < POLICE_COUNT:
            raise DomainValidationError(
                "INVALID_RENDER_STYLE",
                f"At least {POLICE_COUNT} police colours and markers are required",
            )
        if self.dpi <= 0 or min(self.figure_size) <= 0:
            raise DomainValidationError("INVALID_RENDER_STYLE", "dpi and figure size must be positive")


def available_font_families() -> frozenset[str]:
    """Return the set of font family names known to matplotlib's font manager."""
    return frozenset(entry.name for entry in font_manager.fontManager.ttflist)


def discover_korean_font(
    *,
    available_families: Iterable[str] | None = None,
    candidates: Sequence[str] = KOREAN_FONT_CANDIDATES,
    fallback: str = ENGLISH_FALLBACK_FONT,
) -> FontChoice:
    """Deterministically resolve a Korean font, else an English fallback.

    The fixed ordered ``candidates`` list is probed against the available font
    families and the first match wins, so the result depends only on which fonts
    are installed (Requirement 11.8).  When no candidate is available the English
    fallback is returned together with a single recorded warning; labels then use
    the English label set so no glyph is dropped.
    """
    available = set(available_families) if available_families is not None else set(available_font_families())
    for candidate in candidates:
        if candidate in available:
            return FontChoice(family=candidate, supports_korean=True, warning=None)
    warning = (
        "No Korean font found among "
        f"{list(candidates)}; falling back to {fallback!r} with English labels"
    )
    logger.warning(warning)
    return FontChoice(family=fallback, supports_korean=False, warning=warning)


def _point_at_fraction(
    geometry: Sequence[tuple[float, float]], fraction: float
) -> tuple[float, float]:
    """Interpolate a point at ``fraction`` of a polyline's arc length."""
    fraction = min(1.0, max(0.0, fraction))
    total = sum(math.dist(geometry[i], geometry[i + 1]) for i in range(len(geometry) - 1))
    if total <= 0.0:
        return (float(geometry[0][0]), float(geometry[0][1]))
    target = fraction * total
    walked = 0.0
    for first, second in zip(geometry, geometry[1:]):
        segment_len = math.dist(first, second)
        if segment_len <= 0.0:
            continue
        if walked + segment_len >= target:
            local = (target - walked) / segment_len
            return (
                first[0] + local * (second[0] - first[0]),
                first[1] + local * (second[1] - first[1]),
            )
        walked += segment_len
    return (float(geometry[-1][0]), float(geometry[-1][1]))


class OSMRenderer:
    """Render OSM pursuit episode frames and summary figures with matplotlib."""

    def __init__(
        self,
        network: ModelNetwork,
        config: EpisodeConfig,
        *,
        style: RenderStyle | None = None,
        font: FontChoice | None = None,
    ) -> None:
        if not isinstance(network, ModelNetwork):
            raise DomainValidationError("INVALID_NETWORK", "A ModelNetwork instance is required")
        if not isinstance(config, EpisodeConfig):
            raise DomainValidationError("INVALID_EPISODE_CONFIG", "An EpisodeConfig instance is required")
        self.network = network
        self.config = config
        self.style = style or RenderStyle()
        self.font = font or discover_korean_font()
        self._loc = _Localizer(self.font.supports_korean)

    # ------------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------------
    def agent_positions(
        self, state: EpisodeState
    ) -> tuple[list[tuple[float, float]], tuple[float, float]]:
        """Return the six police positions and the fugitive position for a state."""
        police = [placement_position(self.network, placement) for placement in state.police]
        fugitive = placement_position(self.network, state.fugitive)
        return police, fugitive

    def _agent_tracks(
        self, states: Sequence[EpisodeState]
    ) -> tuple[list[list[tuple[float, float]]], list[tuple[float, float]]]:
        """Accumulate per-agent position trails across an episode's states."""
        police_tracks: list[list[tuple[float, float]]] = [[] for _ in range(POLICE_COUNT)]
        fugitive_track: list[tuple[float, float]] = []
        for state in states:
            police, fugitive = self.agent_positions(state)
            for index in range(POLICE_COUNT):
                police_tracks[index].append(police[index])
            fugitive_track.append(fugitive)
        return police_tracks, fugitive_track

    # ------------------------------------------------------------------
    # Road / geometry drawing
    # ------------------------------------------------------------------
    def _draw_roads(self, ax) -> None:
        """Draw every physical directed segment as a polyline with a heading arrow."""
        for segment in self.network.segments:
            if segment.virtual or segment.length_m <= 0.0:
                continue
            xs = [point[0] for point in segment.geometry_xy]
            ys = [point[1] for point in segment.geometry_xy]
            ax.plot(xs, ys, color=self.style.road_color, linewidth=1.0, zorder=1)
            tail = _point_at_fraction(segment.geometry_xy, 0.45)
            head = _point_at_fraction(segment.geometry_xy, 0.55)
            if tail != head:
                ax.annotate(
                    "",
                    xy=head,
                    xytext=tail,
                    arrowprops=dict(arrowstyle="->", color=self.style.road_color, lw=1.0),
                    zorder=1,
                )

    def _highlight_active_segment(self, ax, placement, color: str) -> None:
        """Emphasise the segment a vehicle is currently traversing."""
        if placement.segment_id is None:
            return
        segment = next((item for item in self.network.segments if item.id == placement.segment_id), None)
        if segment is None:
            return
        xs = [point[0] for point in segment.geometry_xy]
        ys = [point[1] for point in segment.geometry_xy]
        ax.plot(xs, ys, color=color, linewidth=2.8, alpha=0.9, zorder=2)

    # ------------------------------------------------------------------
    # Frames
    # ------------------------------------------------------------------
    def frame_metadata(
        self,
        *,
        run_id: str,
        episode_id: str,
        seed: int,
        step: int,
        outcome: EpisodeOutcome | None,
    ) -> dict[str, object]:
        """Structured frame metadata (design section 10; Requirement 11.4)."""
        return {
            "renderer_version": RENDERER_VERSION,
            "run_id": run_id,
            "episode_id": episode_id,
            "seed": int(seed),
            "step": int(step),
            "outcome": outcome.value if isinstance(outcome, EpisodeOutcome) else None,
        }

    def render_frame(
        self,
        state: EpisodeState,
        *,
        episode_id: str = "",
        seed: int = 0,
        outcome: EpisodeOutcome | None = None,
        trails: tuple[Sequence[Sequence[tuple[float, float]]], Sequence[tuple[float, float]]]
        | None = None,
        step: int | None = None,
    ) -> Figure:
        """Render a single episode frame and return its matplotlib figure."""
        fig = Figure(figsize=self.style.figure_size, dpi=self.style.dpi)
        ax = fig.subplots()
        # A scale-correct capture circle requires an equal-aspect metric axis
        # (Requirement 11.3): one metre is the same length on both axes.
        ax.set_aspect("equal", adjustable="datalim")

        self._draw_roads(ax)
        police, fugitive = self.agent_positions(state)

        # Accumulated trails (Requirement 11.2).
        if trails is not None:
            police_tracks, fugitive_track = trails
            for index in range(POLICE_COUNT):
                track = police_tracks[index]
                if len(track) > 1:
                    ax.plot(
                        [p[0] for p in track],
                        [p[1] for p in track],
                        color=self.style.police_colors[index],
                        linewidth=1.4,
                        alpha=self.style.trail_alpha,
                        zorder=2,
                    )
            if len(fugitive_track) > 1:
                ax.plot(
                    [p[0] for p in fugitive_track],
                    [p[1] for p in fugitive_track],
                    color=self.style.fugitive_color,
                    linewidth=1.4,
                    alpha=self.style.trail_alpha,
                    linestyle="--",
                    zorder=2,
                )

        # Current segments (Requirement 11.2).
        for index, placement in enumerate(state.police):
            self._highlight_active_segment(ax, placement, self.style.police_colors[index])
        self._highlight_active_segment(ax, state.fugitive, self.style.active_segment_color)

        # Scale-correct capture radius circles around each officer (Requirement 11.3).
        for index, (x, y) in enumerate(police):
            ax.add_patch(
                Circle(
                    (x, y),
                    radius=self.config.capture_radius_m,
                    fill=False,
                    edgecolor=self.style.capture_circle_color,
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.7,
                    zorder=3,
                )
            )

        # Distinct vehicle markers (Requirement 11.2).
        for index, (x, y) in enumerate(police):
            ax.scatter(
                [x],
                [y],
                c=self.style.police_colors[index],
                marker=self.style.police_markers[index],
                s=90,
                edgecolors="black",
                linewidths=0.6,
                zorder=5,
                label=f"{self._loc.get('police')} {index}",
            )
        ax.scatter(
            [fugitive[0]],
            [fugitive[1]],
            c=self.style.fugitive_color,
            marker=self.style.fugitive_marker,
            s=220,
            edgecolors="black",
            linewidths=0.6,
            zorder=6,
            label=self._loc.get("fugitive"),
        )

        ax.set_xlabel(self._loc.get("easting_m"))
        ax.set_ylabel(self._loc.get("northing_m"))
        ax.legend(loc="upper right", fontsize="small", framealpha=0.85)

        # Terminal annotations (Requirement 11.4): outcome, episode id, seed, step.
        current_step = state.step if step is None else int(step)
        outcome_label = (
            self._loc.get(outcome.value) if isinstance(outcome, EpisodeOutcome)
            else self._loc.get("in_progress")
        )
        title = (
            f"{self._loc.get('episode')}: {episode_id}    "
            f"{self._loc.get('seed')}: {seed}    "
            f"{self._loc.get('step')}: {current_step}"
        )
        ax.set_title(title, fontsize="medium")
        ax.text(
            0.01,
            0.99,
            f"{self._loc.get('outcome')}: {outcome_label}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize="medium",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        fig.tight_layout()
        return fig

    def render_episode_frames(
        self, record: EpisodeRecord, *, max_frames: int | None = None
    ) -> list[Figure]:
        """Render one frame per episode state with accumulating trails.

        The final frame is annotated with the terminal Episode_Outcome; earlier
        frames are annotated as in-progress (Requirements 11.2, 11.4).
        """
        states = [record.initial_state] + [transition.state for transition in record.transitions]
        if max_frames is not None and max_frames > 0:
            states = states[:max_frames]
        police_tracks, fugitive_track = self._agent_tracks(states)
        figures: list[Figure] = []
        for frame_index, state in enumerate(states):
            trails = (
                [track[: frame_index + 1] for track in police_tracks],
                fugitive_track[: frame_index + 1],
            )
            is_last = frame_index == len(states) - 1
            figures.append(
                self.render_frame(
                    state,
                    episode_id=record.episode_id,
                    seed=record.seed,
                    outcome=record.outcome if is_last else None,
                    trails=trails,
                )
            )
        return figures

    # ------------------------------------------------------------------
    # Summary figures (Requirement 11.6)
    # ------------------------------------------------------------------
    def _outcome_axes(self, ax, summary: MetricsSummary) -> None:
        keys = [item.value for item in EpisodeOutcome]
        rates = [float(summary.rates.get(key, 0.0)) for key in keys]
        colors = ["#2ca02c", "#d62728", "#7f7f7f"]
        ax.bar([self._loc.get(key) for key in keys], rates, color=colors)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(self._loc.get("rate"))
        ax.set_title(self._loc.get("outcome_rates"))

    def _episode_length_axes(self, ax, lengths: Sequence[int]) -> None:
        ax.set_title(self._loc.get("episode_length"))
        ax.set_xlabel(self._loc.get("steps"))
        ax.set_ylabel(self._loc.get("count"))
        if lengths:
            ax.hist(list(lengths), bins=max(1, min(20, len(set(lengths)))), color="#1f77b4")
        else:
            ax.text(0.5, 0.5, self._loc.get("no_data"), ha="center", va="center", transform=ax.transAxes)

    def _latency_axes(self, ax, samples_ms: Sequence[float]) -> None:
        ax.set_title(self._loc.get("latency"))
        ax.set_xlabel(self._loc.get("latency"))
        ax.set_ylabel(self._loc.get("count"))
        measured = [float(value) for value in samples_ms if float(value) >= 0.0]
        if measured:
            ax.hist(measured, bins=max(1, min(20, len(measured))), color="#ff7f0e")
        else:
            ax.text(0.5, 0.5, self._loc.get("no_data"), ha="center", va="center", transform=ax.transAxes)

    def _representative_path_axes(self, ax, record: EpisodeRecord | None) -> None:
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(self._loc.get("representative_path"))
        ax.set_xlabel(self._loc.get("easting_m"))
        ax.set_ylabel(self._loc.get("northing_m"))
        self._draw_roads(ax)
        if record is None:
            ax.text(0.5, 0.5, self._loc.get("no_data"), ha="center", va="center", transform=ax.transAxes)
            return
        states = [record.initial_state] + [transition.state for transition in record.transitions]
        police_tracks, fugitive_track = self._agent_tracks(states)
        for index in range(POLICE_COUNT):
            track = police_tracks[index]
            ax.plot(
                [p[0] for p in track],
                [p[1] for p in track],
                color=self.style.police_colors[index],
                linewidth=1.3,
                alpha=self.style.trail_alpha,
                zorder=2,
            )
        ax.plot(
            [p[0] for p in fugitive_track],
            [p[1] for p in fugitive_track],
            color=self.style.fugitive_color,
            linewidth=1.6,
            linestyle="--",
            zorder=3,
        )

    def render_outcome_rates(self, summary: MetricsSummary) -> Figure:
        """Bar chart of capture/escape/timeout rates (Requirement 11.6)."""
        fig = Figure(figsize=(6.0, 4.0), dpi=self.style.dpi)
        self._outcome_axes(fig.subplots(), summary)
        fig.tight_layout()
        return fig

    def render_episode_length_histogram(self, lengths: Sequence[int]) -> Figure:
        """Histogram of episode lengths in steps (Requirement 11.6)."""
        fig = Figure(figsize=(6.0, 4.0), dpi=self.style.dpi)
        self._episode_length_axes(fig.subplots(), lengths)
        fig.tight_layout()
        return fig

    def render_latency_histogram(self, samples_ms: Sequence[float]) -> Figure:
        """Histogram of measured inference latency (Requirement 11.6)."""
        fig = Figure(figsize=(6.0, 4.0), dpi=self.style.dpi)
        self._latency_axes(fig.subplots(), samples_ms)
        fig.tight_layout()
        return fig

    def render_representative_path(self, record: EpisodeRecord) -> Figure:
        """Map of a representative episode's police and fugitive paths."""
        fig = Figure(figsize=self.style.figure_size, dpi=self.style.dpi)
        self._representative_path_axes(fig.subplots(), record)
        fig.tight_layout()
        return fig

    def render_summary(
        self, summary: MetricsSummary, *, representative_record: EpisodeRecord | None = None
    ) -> Figure:
        """Combined 2x2 summary figure (Requirement 11.6).

        Panels: outcome rates, episode-length distribution, measured-latency
        distribution and a representative path map.
        """
        if not isinstance(summary, MetricsSummary):
            raise DomainValidationError("INVALID_SUMMARY", "A MetricsSummary instance is required")
        fig = Figure(figsize=(12.0, 10.0), dpi=self.style.dpi)
        (top_left, top_right), (bottom_left, bottom_right) = fig.subplots(2, 2)
        self._outcome_axes(top_left, summary)
        self._episode_length_axes(top_right, summary.episode_lengths)
        self._latency_axes(bottom_left, summary.latency.get("samples_ms", ()))
        self._representative_path_axes(bottom_right, representative_record)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def save_figure(self, figure: Figure, path: str, *, close: bool = True) -> str:
        """Save ``figure`` as a raster image and optionally release it."""
        figure.savefig(path, dpi=self.style.dpi)
        if close:
            plt.close(figure)
        return path


__all__ = (
    "ENGLISH_FALLBACK_FONT",
    "FUGITIVE_COLOR",
    "FUGITIVE_MARKER",
    "KOREAN_FONT_CANDIDATES",
    "POLICE_COLORS",
    "POLICE_MARKERS",
    "RENDERER_VERSION",
    "FontChoice",
    "OSMRenderer",
    "RenderStyle",
    "available_font_families",
    "discover_korean_font",
)

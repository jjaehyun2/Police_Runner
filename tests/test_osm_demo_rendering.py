"""Focused offline tests for OSM pursuit matplotlib rendering (Req 11.1-11.4, 11.6, 11.8)."""
from __future__ import annotations

import pytest

matplotlib = pytest.importorskip("matplotlib")
from matplotlib.patches import Circle

from hypothesis import given, settings
from hypothesis import strategies as st

from pursuit_evasion_rl.osm_demo.coarsening import coarsen_raw_graph, prepare_model_network
from pursuit_evasion_rl.osm_demo.fixtures import daejeon
from pursuit_evasion_rl.osm_demo.metrics import summarize_metrics
from pursuit_evasion_rl.osm_demo.models import EpisodeConfig, EpisodeOutcome
from pursuit_evasion_rl.osm_demo.policies import BaselinePolicePolicy
from pursuit_evasion_rl.osm_demo.rendering import (
    ENGLISH_FALLBACK_FONT,
    KOREAN_FONT_CANDIDATES,
    FontChoice,
    OSMRenderer,
    RenderStyle,
    discover_korean_font,
)
from pursuit_evasion_rl.osm_demo.runner import run_batch, run_single_episode

pytestmark = pytest.mark.offline

CAPTURE_RADIUS_M = 15.0


def _network():
    prepared = prepare_model_network(coarsen_raw_graph(daejeon()))
    return prepared.network


def _config(max_steps: int = 4) -> EpisodeConfig:
    return EpisodeConfig(
        dt_s=1.0,
        police_speed_mps=12.0,
        fugitive_speed_mps=9.0,
        capture_radius_m=CAPTURE_RADIUS_M,
        max_steps=max_steps,
    )


def _record(network, config, *, run_seed: int = 7):
    policy = BaselinePolicePolicy(network)
    return run_single_episode(network, config, policy, run_id="render-test", run_seed=run_seed)


# ---------------------------------------------------------------------------
# Font discovery (Requirement 11.8)
# ---------------------------------------------------------------------------
def test_discover_korean_font_prefers_first_available_candidate():
    choice = discover_korean_font(available_families={"Noto Sans", KOREAN_FONT_CANDIDATES[1]})
    assert choice == FontChoice(family=KOREAN_FONT_CANDIDATES[1], supports_korean=True, warning=None)


def test_discover_korean_font_falls_back_to_english_with_a_single_warning():
    choice = discover_korean_font(available_families={"DejaVu Sans", "Arial"})
    assert choice.family == ENGLISH_FALLBACK_FONT
    assert choice.supports_korean is False
    assert choice.warning is not None


@settings(max_examples=40, deadline=None)
@given(
    available=st.sets(st.sampled_from(list(KOREAN_FONT_CANDIDATES) + ["Arial", "DejaVu Sans"]))
)
def test_font_discovery_selects_korean_iff_a_candidate_is_available(available):
    """Validates: Requirements 11.8 - Korean chosen exactly when a candidate exists."""
    choice = discover_korean_font(available_families=available)
    korean_available = any(candidate in available for candidate in KOREAN_FONT_CANDIDATES)
    assert choice.supports_korean == korean_available
    if korean_available:
        # The first candidate in canonical order must be selected.
        expected = next(c for c in KOREAN_FONT_CANDIDATES if c in available)
        assert choice.family == expected
    else:
        assert choice.family == ENGLISH_FALLBACK_FONT
        assert choice.warning is not None


# ---------------------------------------------------------------------------
# Frame rendering (Requirements 11.1-11.3)
# ---------------------------------------------------------------------------
def test_render_frame_draws_roads_and_scale_correct_capture_circles():
    network = _network()
    config = _config()
    record = _record(network, config)
    renderer = OSMRenderer(network, config, font=FontChoice("DejaVu Sans", False))

    fig = renderer.render_frame(record.initial_state, episode_id=record.episode_id, seed=record.seed)
    ax = fig.axes[0]

    # Equal aspect makes the metric capture circle scale-correct (Requirement 11.3).
    assert ax.get_aspect() in ("equal", 1.0)
    circles = [patch for patch in ax.patches if isinstance(patch, Circle)]
    assert len(circles) == 6  # one capture circle per officer
    assert all(circle.get_radius() == pytest.approx(CAPTURE_RADIUS_M) for circle in circles)
    # Directed roads are drawn as line artists (Requirement 11.1).
    assert len(ax.lines) >= 1


def test_render_frame_marks_six_police_and_one_fugitive_distinctly():
    network = _network()
    config = _config()
    record = _record(network, config)
    renderer = OSMRenderer(network, config, font=FontChoice("DejaVu Sans", False))

    fig = renderer.render_frame(record.initial_state, episode_id=record.episode_id, seed=record.seed)
    ax = fig.axes[0]
    # Six police scatters plus one fugitive scatter (Requirement 11.2).
    assert len(ax.collections) >= 7


# ---------------------------------------------------------------------------
# Terminal annotations (Requirement 11.4)
# ---------------------------------------------------------------------------
def test_terminal_frame_annotates_outcome_seed_episode_and_step():
    network = _network()
    config = _config()
    record = _record(network, config)
    renderer = OSMRenderer(network, config, font=FontChoice("DejaVu Sans", False))

    frames = renderer.render_episode_frames(record)
    assert len(frames) == len(record.transitions) + 1

    last = frames[-1]
    ax = last.axes[0]
    text_blob = " ".join([ax.get_title()] + [artist.get_text() for artist in ax.texts])
    assert str(record.seed) in text_blob
    assert record.episode_id in text_blob
    # The terminal outcome label (English fallback) appears on the final frame.
    outcome_label = {
        EpisodeOutcome.CAPTURE: "Capture",
        EpisodeOutcome.ESCAPE: "Escape",
        EpisodeOutcome.TIMEOUT: "Timeout",
    }[record.outcome]
    assert outcome_label in text_blob


def test_frame_metadata_reports_outcome_and_identity():
    network = _network()
    config = _config()
    renderer = OSMRenderer(network, config, font=FontChoice("DejaVu Sans", False))
    meta = renderer.frame_metadata(
        run_id="r", episode_id="r:ep000000", seed=42, step=3, outcome=EpisodeOutcome.CAPTURE
    )
    assert meta["seed"] == 42 and meta["step"] == 3
    assert meta["outcome"] == "capture"
    assert meta["episode_id"] == "r:ep000000"


# ---------------------------------------------------------------------------
# Summary figures (Requirement 11.6)
# ---------------------------------------------------------------------------
def test_render_summary_produces_four_panel_figure_and_saves(tmp_path):
    network = _network()
    config = _config()
    policy = BaselinePolicePolicy(network)
    records = run_batch(network, config, policy, run_id="render-batch", run_seed=3, episode_count=3)
    summary = summarize_metrics(
        records, network, policy_kind="baseline", minimum_evaluation_episodes=1
    )

    renderer = OSMRenderer(network, config, font=FontChoice("DejaVu Sans", False))
    fig = renderer.render_summary(summary, representative_record=records[0])
    assert len(fig.axes) == 4

    out = tmp_path / "summary.png"
    renderer.save_figure(fig, str(out))
    assert out.exists() and out.stat().st_size > 0


def test_individual_summary_charts_render():
    network = _network()
    config = _config()
    policy = BaselinePolicePolicy(network)
    records = run_batch(network, config, policy, run_id="render-batch", run_seed=5, episode_count=2)
    summary = summarize_metrics(
        records, network, policy_kind="baseline", minimum_evaluation_episodes=1
    )
    renderer = OSMRenderer(network, config, font=FontChoice("DejaVu Sans", False))

    assert renderer.render_outcome_rates(summary).axes
    assert renderer.render_episode_length_histogram(summary.episode_lengths).axes
    assert renderer.render_latency_histogram([1.2, 3.4, 5.6]).axes
    assert renderer.render_representative_path(records[0]).axes


def test_render_style_rejects_insufficient_police_styling():
    with pytest.raises(Exception):
        RenderStyle(police_colors=("#000000",))


def test_render_episode_frame_saves_png(tmp_path):
    network = _network()
    config = _config()
    record = _record(network, config)
    renderer = OSMRenderer(network, config, font=FontChoice("DejaVu Sans", False))
    fig = renderer.render_frame(record.initial_state, episode_id=record.episode_id, seed=record.seed)
    out = tmp_path / "frame_000000.png"
    renderer.save_figure(fig, str(out))
    assert out.exists() and out.stat().st_size > 0

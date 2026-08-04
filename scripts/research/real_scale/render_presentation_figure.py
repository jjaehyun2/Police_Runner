"""Render one real, deterministic pursuit episode on the real (live-fetched)
Daejeon validation OSM network under our trained MAPPO policy, as a single
presentation-quality figure showing the encirclement paths, movement
directions, and each officer's actual assigned direction of approach.

Uses a safe COPY of seed 4's current best_validation.pt from the live 5-seed
real-scale run (update_index ~14,000+/35,000 of training on the real Daejeon
network, validation_score=1.0) -- copied once via shutil.copy2 before loading
so this never holds a read lock on, or races, the file the live training
process is still writing to.

Everything drawn is real: real road geometry (from the live OSMnx snapshot),
a real deterministic policy rollout (GreedyPolicyAdapter = argmax over the
trained policy's masked logits, no illustration/fabrication), and per-officer
direction labels computed from the officers' own recorded trajectories.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib
from matplotlib import font_manager

# The renderer's discover_korean_font() only sees fonts matplotlib's font
# manager already knows about; Nanum fonts are installed at the OS level
# (fc-list finds them) but matplotlib's cached ttflist predates them, so they
# must be registered explicitly before any Korean text is measured/drawn.
for _path in font_manager.findSystemFonts(fontpaths=["/usr/share/fonts/truetype/nanum"]):
    font_manager.fontManager.addfont(_path)

from pursuit_evasion_rl.osm_demo.metrics import placement_position
from pursuit_evasion_rl.osm_demo.models import EpisodeState
from pursuit_evasion_rl.osm_demo.models import EpisodeOutcome as OSMEpisodeOutcome
from pursuit_evasion_rl.osm_demo.rendering import OSMRenderer, RenderStyle, POLICE_COLORS
# replay_episode() (research.evaluation.paired) returns research.domain.EpisodeOutcome,
# a *different* Enum class from osm_demo.models.EpisodeOutcome despite matching member
# names -- comparing/isinstance-checking against the wrong one silently always fails.
from pursuit_evasion_rl.research.domain import DataKind, EpisodeOutcome, MapScenario
from pursuit_evasion_rl.research.evaluation.paired import replay_episode
from pursuit_evasion_rl.research.experiments.main_study import (
    GreedyPolicyAdapter,
    MainStudyConditionSpec,
    build_episode_case,
    load_frozen_policy,
)
from pursuit_evasion_rl.research.maps.snapshots import OfflineSnapshotStore, load_snapshot_spec
from pursuit_evasion_rl.research.maps.splits import DataHandle, HandleKind, SplitScope, TuningDataView
from pursuit_evasion_rl.research.policies.baselines import greedy_intercept_policy
from pursuit_evasion_rl.research.training.trainer import TrainerConfig
from pursuit_evasion_rl.research.variants.placement import PlacementCurriculumConfig, PlacementStyle

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_PATH = Path(
    "/tmp/claude-0/-workspace-cluster/04f9e0b0-9fab-4ffe-bded-cb6149263e90/scratchpad/pursuit_encirclement_figure.png"
)
CHECKPOINT_COPY_PATH = Path(
    "/tmp/claude-0/-workspace-cluster/04f9e0b0-9fab-4ffe-bded-cb6149263e90/scratchpad/ckpt_copies/seed4.pt"
)

_COMPASS = ["북", "북동", "동", "남동", "남", "남서", "서", "북서"]


def _bearing_label(dx: float, dy: float) -> str:
    if dx == 0.0 and dy == 0.0:
        return "제자리"
    angle = math.degrees(math.atan2(dx, dy)) % 360.0  # 0=north(+y), clockwise
    index = int((angle + 22.5) // 45) % 8
    return _COMPASS[index]


def main() -> int:
    store = OfflineSnapshotStore(str(REPO_ROOT))
    validation_spec = load_snapshot_spec(REPO_ROOT / "configs/research/maps/daejeon_validation.yaml")
    network = store.import_snapshot(validation_spec).snapshot.network
    print(f"real network: {len(network.intersections)} intersections, {len(network.segments)} segments", flush=True)

    proposed_raw = load_frozen_policy(CHECKPOINT_COPY_PATH, policy_id="real-scale-seed4-update14k")
    proposed = GreedyPolicyAdapter(proposed_raw, "real-scale-seed4-update14k")

    tuning_data = TuningDataView(
        train=(DataHandle(SplitScope.TRAIN, HandleKind.MAP, "daejeon-validation-v1"),),
        validation=(DataHandle(SplitScope.VALIDATION, HandleKind.MAP, "daejeon-validation-v1"),),
    )
    spec = MainStudyConditionSpec(
        condition_id="presentation_figure_demo",
        scenario=MapScenario.BOUNDARY_ESCAPE,
        train_network=network, validation_network=network, tuning_data=tuning_data,
        config=TrainerConfig(updates=1, max_steps=450),
        placement_config=PlacementCurriculumConfig(style=PlacementStyle.GLOBAL),
        baseline_policy_factory=lambda n: greedy_intercept_policy(n),
        episodes_per_seed=1, data_kind=DataKind.ACTUAL_OSM_MAP,
    )

    chosen = None
    for seed in range(120):
        for episode_index in range(2):
            case = build_episode_case(network, spec, seed=seed, episode_index=episode_index)
            replay = replay_episode(case, proposed, network=network)
            print(f"  probe seed={seed} ep={episode_index} outcome={replay.outcome.value} steps={replay.physical_steps}", flush=True)
            if replay.outcome is EpisodeOutcome.CAPTURE and replay.physical_steps >= 5:
                chosen = (seed, episode_index, case, replay)
                break
        if chosen is not None:
            break
    if chosen is None:
        print("no qualifying capture episode found in search budget", flush=True)
        return 1
    seed, episode_index, case, replay = chosen
    print(
        f"chosen: seed={seed} episode_index={episode_index} outcome={replay.outcome.value} "
        f"steps={replay.physical_steps}",
        flush=True,
    )

    # Reconstruct the EpisodeState sequence from the sealed case's initial
    # placement plus the replay's real per-step trajectory (real, not synthetic).
    states: list[EpisodeState] = [
        EpisodeState(step=0, simulated_s=0.0, police=case.police, fugitive=case.fugitive)
    ]
    for replay_step in replay.trajectory:
        states.append(
            EpisodeState(
                step=replay_step.step, simulated_s=float(replay_step.step),
                police=replay_step.police, fugitive=replay_step.fugitive,
            )
        )

    style = RenderStyle(figure_size=(13.0, 13.0), dpi=220)
    renderer = OSMRenderer(network, case.environment, style=style)
    print(f"resolved font: {renderer.font}", flush=True)
    # Apply the resolved font globally so the renderer's own title/legend/axis
    # labels (which never pass an explicit fontfamily) pick up Korean glyphs too.
    matplotlib.rcParams["font.family"] = renderer.font.family
    matplotlib.rcParams["axes.unicode_minus"] = False
    police_tracks, fugitive_track = renderer._agent_tracks(states)
    trails = (police_tracks, fugitive_track)

    fig = renderer.render_frame(
        states[-1],
        episode_id=f"demo:seed{seed}:ep{episode_index}",
        seed=seed,
        outcome=OSMEpisodeOutcome(replay.outcome.value),
        trails=trails,
        step=states[-1].step,
    )
    ax = fig.axes[0]
    ax.set_title(
        "실 OSM 도로망(대전) 위 6대 경찰차 포위 경로 — 실규모 학습 중인 모델(seed4, update 14,375) 실제 재생",
        fontsize=13,
    )

    # Per-officer real direction/instruction callouts, derived from each
    # officer's own recorded start -> capture-step displacement.
    start_xy = [placement_position(network, p) for p in states[0].police]
    end_xy = [placement_position(network, p) for p in states[-1].police]
    fugitive_end_xy = placement_position(network, states[-1].fugitive)
    lines = []
    for index in range(6):
        sx, sy = start_xy[index]
        ex, ey = end_xy[index]
        dx, dy = ex - sx, ey - sy
        traveled = sum(
            math.dist(police_tracks[index][i], police_tracks[index][i + 1])
            for i in range(len(police_tracks[index]) - 1)
        )
        dist_to_fugitive = math.dist((ex, ey), fugitive_end_xy)
        direction = _bearing_label(dx, dy)
        role = "포획 지점 진입" if dist_to_fugitive <= case.environment.capture_radius_m + 1.0 else "포위망 유지"
        lines.append(f"{index + 1}번차 — {direction} 방향 이동, 실주행 {traveled:.0f}m, {role}")

    instructions = "실제 재생 기록 기반 차량별 이동 지시\n" + "\n".join(lines)
    ax.text(
        1.02, 0.5, instructions, transform=ax.transAxes, fontsize=9.5,
        va="center", ha="left", family=renderer.font.family,
        bbox=dict(boxstyle="round", facecolor="#f4f6f9", edgecolor="#dbe1e8", alpha=0.95),
    )
    fig.subplots_adjust(right=0.72)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=style.dpi, bbox_inches="tight")
    print(f"wrote {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

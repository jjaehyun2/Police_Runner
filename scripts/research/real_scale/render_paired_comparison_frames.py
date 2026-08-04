"""Find ONE real sealed case (identical starting placements for all 6 officers
and the fugitive, identical fugitive RNG stream) where our trained model FAILS
to capture and the greedy_intercept baseline SUCCEEDS, then render both
replays of that exact same case frame-by-frame -- a true apples-to-apples
comparison of the two policies' behavior from the same starting position.

Uses seed 4's final checkpoint (update 35000/35000).
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib
from matplotlib import font_manager

for _path in font_manager.findSystemFonts(fontpaths=["/usr/share/fonts/truetype/nanum"]):
    font_manager.fontManager.addfont(_path)

from PIL import Image, ImageDraw, ImageFont

from pursuit_evasion_rl.osm_demo.models import EpisodeState
from pursuit_evasion_rl.osm_demo.models import EpisodeOutcome as OSMEpisodeOutcome
from pursuit_evasion_rl.osm_demo.rendering import OSMRenderer, RenderStyle
from pursuit_evasion_rl.research.domain import DataKind, MapScenario
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
SCRATCH = Path(__file__).resolve().parent / "output"
CHECKPOINT = REPO_ROOT / "artifacts/research/checkpoints/real_scale_daejeon_main_study/4/20260802T195032-9ae88320f213/best_validation.pt"
NANUM_BOLD = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"
NANUM_REGULAR = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"


def _states_from(case, replay):
    states = [EpisodeState(step=0, simulated_s=0.0, police=case.police, fugitive=case.fugitive)]
    for rs in replay.trajectory:
        states.append(EpisodeState(step=rs.step, simulated_s=float(rs.step), police=rs.police, fugitive=rs.fugitive))
    return states


def _render_gif_and_sheet(renderer, states, *, seed, episode_index, outcome_value, label, frames_dir, gif_path, sheet_path):
    n = len(states) - 1
    frames_dir.mkdir(parents=True, exist_ok=True)
    police_tracks, fugitive_track = renderer._agent_tracks(states)
    frame_paths = []
    for idx, state in enumerate(states):
        trails = ([track[: idx + 1] for track in police_tracks], fugitive_track[: idx + 1])
        is_final = idx == n
        fig = renderer.render_frame(
            state, episode_id=f"{label}:seed{seed}:ep{episode_index}", seed=seed,
            outcome=OSMEpisodeOutcome(outcome_value) if is_final else None,
            trails=trails, step=state.step,
        )
        p = frames_dir / f"frame_{idx:04d}.png"
        fig.savefig(p, dpi=renderer.style.dpi)
        import matplotlib.pyplot as plt
        plt.close(fig)
        frame_paths.append(p)
    print(f"  [{label}] rendered {len(frame_paths)} frames", flush=True)

    frames = [Image.open(p).convert("RGB") for p in frame_paths]
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=180, loop=0, optimize=True)
    print(f"  [{label}] wrote {gif_path} ({len(frames)} frames)", flush=True)

    n_tiles = min(12, len(frames))
    tile_indices = sorted({round(i * (len(frames) - 1) / (n_tiles - 1)) for i in range(n_tiles)})
    cols = 4
    rows = (len(tile_indices) + cols - 1) // cols
    tile_w, tile_h = frames[0].size
    scale = 380 / tile_w
    tw, th = round(tile_w * scale), round(tile_h * scale)
    label_h = 28
    sheet = Image.new("RGB", (cols * tw, rows * (th + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(NANUM_REGULAR, 16)
    for tile_pos, frame_idx in enumerate(tile_indices):
        row, col = divmod(tile_pos, cols)
        thumb = frames[frame_idx].resize((tw, th))
        x, y = col * tw, row * (th + label_h)
        sheet.paste(thumb, (x, y + label_h))
        draw.text((x + 6, y + 4), f"step {states[frame_idx].step}/{n}", fill="black", font=font)
    header_font = ImageFont.truetype(NANUM_BOLD, 22)
    header = Image.new("RGB", (sheet.width, 40), "white")
    hd = ImageDraw.Draw(header)
    hd.text((10, 8), f"{label} (seed{seed} ep{episode_index}) — 결과: {outcome_value}, {n}스텝", fill="black", font=header_font)
    final_sheet = Image.new("RGB", (sheet.width, sheet.height + header.height), "white")
    final_sheet.paste(header, (0, 0))
    final_sheet.paste(sheet, (0, header.height))
    final_sheet.save(sheet_path)
    print(f"  [{label}] wrote {sheet_path}", flush=True)


def main() -> int:
    store = OfflineSnapshotStore(str(REPO_ROOT))
    validation_spec = load_snapshot_spec(REPO_ROOT / "configs/research/maps/daejeon_validation.yaml")
    network = store.import_snapshot(validation_spec).snapshot.network

    proposed_raw = load_frozen_policy(CHECKPOINT, policy_id="real-scale-seed4-final")
    proposed = GreedyPolicyAdapter(proposed_raw, "real-scale-seed4-final")
    baseline = greedy_intercept_policy(network)

    tuning_data = TuningDataView(
        train=(DataHandle(SplitScope.TRAIN, HandleKind.MAP, "daejeon-validation-v1"),),
        validation=(DataHandle(SplitScope.VALIDATION, HandleKind.MAP, "daejeon-validation-v1"),),
    )
    spec = MainStudyConditionSpec(
        condition_id="paired_comparison_demo",
        scenario=MapScenario.BOUNDARY_ESCAPE,
        train_network=network, validation_network=network, tuning_data=tuning_data,
        config=TrainerConfig(updates=1, max_steps=450),
        placement_config=PlacementCurriculumConfig(style=PlacementStyle.GLOBAL),
        baseline_policy_factory=lambda n: greedy_intercept_policy(n),
        episodes_per_seed=1, data_kind=DataKind.ACTUAL_OSM_MAP,
    )

    chosen = None
    for seed in range(150):
        for episode_index in range(2):
            case = build_episode_case(network, spec, seed=seed, episode_index=episode_index)
            proposed_replay = replay_episode(case, proposed, network=network)
            if proposed_replay.outcome.value == "capture":
                continue
            baseline_replay = replay_episode(case, network=network, team_policy=baseline)
            if baseline_replay.outcome.value == "capture":
                chosen = (seed, episode_index, case, proposed_replay, baseline_replay)
                break
        if chosen is not None:
            break
    if chosen is None:
        print("no qualifying case found (proposed fails, baseline succeeds)", flush=True)
        return 1
    seed, episode_index, case, proposed_replay, baseline_replay = chosen
    print(
        f"chosen SAME case: seed={seed} ep={episode_index} | "
        f"our_model={proposed_replay.outcome.value} ({proposed_replay.physical_steps} steps) | "
        f"greedy_intercept={baseline_replay.outcome.value} ({baseline_replay.physical_steps} steps)",
        flush=True,
    )
    print("initial placement (shared by both replays):", flush=True)
    print(f"  fugitive: {case.fugitive.identity}", flush=True)
    for i, p in enumerate(case.police):
        print(f"  police {i}: {p.identity}", flush=True)

    style = RenderStyle(figure_size=(7.5, 7.5), dpi=130)
    renderer = OSMRenderer(network, case.environment, style=style)
    matplotlib.rcParams["font.family"] = renderer.font.family
    matplotlib.rcParams["axes.unicode_minus"] = False

    _render_gif_and_sheet(
        renderer, _states_from(case, proposed_replay),
        seed=seed, episode_index=episode_index, outcome_value=proposed_replay.outcome.value,
        label="우리모델(실패)",
        frames_dir=SCRATCH / "paired_ours_frames",
        gif_path=SCRATCH / "paired_ours.gif",
        sheet_path=SCRATCH / "paired_ours_contact_sheet.png",
    )
    _render_gif_and_sheet(
        renderer, _states_from(case, baseline_replay),
        seed=seed, episode_index=episode_index, outcome_value=baseline_replay.outcome.value,
        label="greedy_intercept(성공)",
        frames_dir=SCRATCH / "paired_greedy_frames",
        gif_path=SCRATCH / "paired_greedy.gif",
        sheet_path=SCRATCH / "paired_greedy_contact_sheet.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

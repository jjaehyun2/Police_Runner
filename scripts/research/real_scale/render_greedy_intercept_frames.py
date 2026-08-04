"""Frame-by-frame render of a real episode under the greedy_intercept baseline
(GreedyInterceptPolice: every officer independently picks the legal exit that
minimizes straight-line distance to the fugitive's current position -- no
learning, no coordination, no trajectory prediction) on the real Daejeon
validation network, so its actual moment-to-moment behavior can be inspected
directly -- this is the pre-existing rule-based baseline our trained model is
currently losing to (74-81% vs ~85-94% capture rate across seeds).

Produces the same three artifacts as render_failure_episode_frames.py, in a
separate directory so neither run overwrites the other.
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
    MainStudyConditionSpec,
    build_episode_case,
)
from pursuit_evasion_rl.research.maps.snapshots import OfflineSnapshotStore, load_snapshot_spec
from pursuit_evasion_rl.research.maps.splits import DataHandle, HandleKind, SplitScope, TuningDataView
from pursuit_evasion_rl.research.policies.baselines import greedy_intercept_policy
from pursuit_evasion_rl.research.training.trainer import TrainerConfig
from pursuit_evasion_rl.research.variants.placement import PlacementCurriculumConfig, PlacementStyle

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRATCH = Path(__file__).resolve().parent / "output"
FRAMES_DIR = SCRATCH / "greedy_intercept_frames"
GIF_PATH = SCRATCH / "greedy_intercept_episode.gif"
CONTACT_SHEET_PATH = SCRATCH / "greedy_intercept_contact_sheet.png"
NANUM_BOLD = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"
NANUM_REGULAR = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"


def main() -> int:
    store = OfflineSnapshotStore(str(REPO_ROOT))
    validation_spec = load_snapshot_spec(REPO_ROOT / "configs/research/maps/daejeon_validation.yaml")
    network = store.import_snapshot(validation_spec).snapshot.network

    baseline = greedy_intercept_policy(network)

    tuning_data = TuningDataView(
        train=(DataHandle(SplitScope.TRAIN, HandleKind.MAP, "daejeon-validation-v1"),),
        validation=(DataHandle(SplitScope.VALIDATION, HandleKind.MAP, "daejeon-validation-v1"),),
    )
    spec = MainStudyConditionSpec(
        condition_id="greedy_intercept_visual_demo",
        scenario=MapScenario.BOUNDARY_ESCAPE,
        train_network=network, validation_network=network, tuning_data=tuning_data,
        config=TrainerConfig(updates=1, max_steps=450),
        placement_config=PlacementCurriculumConfig(style=PlacementStyle.GLOBAL),
        baseline_policy_factory=lambda n: greedy_intercept_policy(n),
        episodes_per_seed=1, data_kind=DataKind.ACTUAL_OSM_MAP,
    )

    chosen = None
    for seed in range(40):
        for episode_index in range(2):
            case = build_episode_case(network, spec, seed=seed, episode_index=episode_index)
            replay = replay_episode(case, network=network, team_policy=baseline)
            if replay.outcome.value == "capture" and 25 <= replay.physical_steps <= 180:
                chosen = (seed, episode_index, case, replay)
                break
        if chosen is not None:
            break
    if chosen is None:
        print("no qualifying capture episode found", flush=True)
        return 1
    seed, episode_index, case, replay = chosen
    print(f"chosen greedy_intercept CAPTURE episode: seed={seed} ep={episode_index} steps={replay.physical_steps}", flush=True)

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
    n = len(states) - 1

    style = RenderStyle(figure_size=(7.5, 7.5), dpi=130)
    renderer = OSMRenderer(network, case.environment, style=style)
    matplotlib.rcParams["font.family"] = renderer.font.family
    matplotlib.rcParams["axes.unicode_minus"] = False

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    police_tracks, fugitive_track = renderer._agent_tracks(states)
    frame_paths: list[Path] = []
    for idx, state in enumerate(states):
        trails = (
            [track[: idx + 1] for track in police_tracks],
            fugitive_track[: idx + 1],
        )
        is_final = idx == n
        fig = renderer.render_frame(
            state, episode_id=f"greedy-intercept:seed{seed}:ep{episode_index}", seed=seed,
            outcome=OSMEpisodeOutcome(replay.outcome.value) if is_final else None,
            trails=trails, step=state.step,
        )
        frame_path = FRAMES_DIR / f"frame_{idx:04d}.png"
        fig.savefig(frame_path, dpi=style.dpi)
        import matplotlib.pyplot as plt
        plt.close(fig)
        frame_paths.append(frame_path)
        if idx % 20 == 0:
            print(f"  rendered frame {idx}/{n}", flush=True)
    print(f"rendered {len(frame_paths)} frames to {FRAMES_DIR}", flush=True)

    frames = [Image.open(p).convert("RGB") for p in frame_paths]
    frames[0].save(
        GIF_PATH, save_all=True, append_images=frames[1:], duration=180, loop=0, optimize=True,
    )
    print(f"wrote {GIF_PATH} ({len(frames)} frames)", flush=True)

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
        label = f"step {states[frame_idx].step}/{n}"
        draw.text((x + 6, y + 4), label, fill="black", font=font)
    header_font = ImageFont.truetype(NANUM_BOLD, 22)
    header = Image.new("RGB", (sheet.width, 40), "white")
    hd = ImageDraw.Draw(header)
    hd.text(
        (10, 8),
        f"greedy_intercept baseline (seed{seed} ep{episode_index}) — 결과: capture, {n}스텝",
        fill="black", font=header_font,
    )
    final_sheet = Image.new("RGB", (sheet.width, sheet.height + header.height), "white")
    final_sheet.paste(header, (0, 0))
    final_sheet.paste(sheet, (0, header.height))
    final_sheet.save(CONTACT_SHEET_PATH)
    print(f"wrote {CONTACT_SHEET_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

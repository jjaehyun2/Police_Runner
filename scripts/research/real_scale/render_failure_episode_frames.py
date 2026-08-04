"""Frame-by-frame render of a real FAILURE episode (our trained model does NOT
capture the fugitive) on the real Daejeon validation network, so the actual
step-by-step behavior can be inspected for problems -- this is diagnostic
material for the finding that our model loses to greedy_intercept baseline.

Uses seed 4's final checkpoint (update 35000/35000, the largest performance
gap vs greedy_intercept: -18.6pp) via a safe read of the already-completed,
static checkpoint file (training for this seed is done; nothing is writing to
it anymore).

Produces:
  - failure_episode_frames/frame_XXX.png -- every physical step, individually
  - failure_episode.gif -- the same frames as a real animated GIF
  - failure_episode_contact_sheet.png -- a labeled grid of evenly-spaced
    frames for quick in-chat inspection without opening the GIF
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
FRAMES_DIR = SCRATCH / "failure_episode_frames"
CHECKPOINT = REPO_ROOT / "artifacts/research/checkpoints/real_scale_daejeon_main_study/4/20260802T195032-9ae88320f213/best_validation.pt"
GIF_PATH = SCRATCH / "failure_episode.gif"
CONTACT_SHEET_PATH = SCRATCH / "failure_episode_contact_sheet.png"
NANUM_BOLD = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"
NANUM_REGULAR = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"


def main() -> int:
    import torch
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    assert payload["update_index"] == 35000
    print(f"checkpoint verified: update_index={payload['update_index']}", flush=True)

    store = OfflineSnapshotStore(str(REPO_ROOT))
    validation_spec = load_snapshot_spec(REPO_ROOT / "configs/research/maps/daejeon_validation.yaml")
    network = store.import_snapshot(validation_spec).snapshot.network

    proposed_raw = load_frozen_policy(CHECKPOINT, policy_id="real-scale-seed4-final")
    proposed = GreedyPolicyAdapter(proposed_raw, "real-scale-seed4-final")

    tuning_data = TuningDataView(
        train=(DataHandle(SplitScope.TRAIN, HandleKind.MAP, "daejeon-validation-v1"),),
        validation=(DataHandle(SplitScope.VALIDATION, HandleKind.MAP, "daejeon-validation-v1"),),
    )
    spec = MainStudyConditionSpec(
        condition_id="failure_episode_diagnostic",
        scenario=MapScenario.BOUNDARY_ESCAPE,
        train_network=network, validation_network=network, tuning_data=tuning_data,
        config=TrainerConfig(updates=1, max_steps=450),
        placement_config=PlacementCurriculumConfig(style=PlacementStyle.GLOBAL),
        baseline_policy_factory=lambda n: greedy_intercept_policy(n),
        episodes_per_seed=1, data_kind=DataKind.ACTUAL_OSM_MAP,
    )

    chosen = None
    for seed in range(60):
        for episode_index in range(2):
            case = build_episode_case(network, spec, seed=seed, episode_index=episode_index)
            replay = replay_episode(case, proposed, network=network)
            if replay.outcome.value != "capture" and 25 <= replay.physical_steps <= 180:
                chosen = (seed, episode_index, case, replay)
                break
        if chosen is not None:
            break
    if chosen is None:
        print("no qualifying failure episode found", flush=True)
        return 1
    seed, episode_index, case, replay = chosen
    print(f"chosen FAILURE episode: seed={seed} ep={episode_index} outcome={replay.outcome.value} steps={replay.physical_steps}", flush=True)

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
            state, episode_id=f"failure:seed{seed}:ep{episode_index}", seed=seed,
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

    # ---- animated GIF: every real frame, in order ----
    frames = [Image.open(p).convert("RGB") for p in frame_paths]
    frames[0].save(
        GIF_PATH, save_all=True, append_images=frames[1:], duration=180, loop=0, optimize=True,
    )
    print(f"wrote {GIF_PATH} ({len(frames)} frames)", flush=True)

    # ---- contact sheet: 12 evenly-spaced frames in a grid, labeled ----
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
        f"실패 에피소드 (seed{seed} ep{episode_index}) — 결과: {replay.outcome.value}, {n}스텝",
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

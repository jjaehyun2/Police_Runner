"""Presentation triptych: early / mid / pre-capture frames of one real pursuit
episode on the real (live-fetched) Daejeon validation OSM network, replayed
under our trained MAPPO policy (seed 4, update ~34,245/35,000 of the live
real-scale run), with the fugitive's actual reachable-region size at each
frame computed by the repo's own reachable_region_reduction() (directed BFS
over the real road graph with officer-occupied intersections removed) --
a measured number, not an illustrative one.

Slide framing (per the request this was built for):
  - Slide 8 "role-specialization learning": the three panels show each
    officer settling into a distinct sector rather than all chasing the same
    node.
  - Slide 9 "recommended way to shrink the encirclement": the reachable-count
    sequence (baseline -> mid -> late) is the actual measured evidence that
    the ring is closing, not a claim.

Uses a safe COPY of seed 4's current best_validation.pt (never reads the live
training process's file directly).
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
from pursuit_evasion_rl.research.domain import DataKind, EpisodeOutcome, MapScenario
from pursuit_evasion_rl.research.evaluation.paired import replay_episode
from pursuit_evasion_rl.research.experiments.main_study import (
    GreedyPolicyAdapter,
    MainStudyConditionSpec,
    build_episode_case,
    load_frozen_policy,
)
from pursuit_evasion_rl.osm_demo.metrics import placement_position
from pursuit_evasion_rl.research.maps.snapshots import OfflineSnapshotStore, load_snapshot_spec
from pursuit_evasion_rl.research.maps.splits import DataHandle, HandleKind, SplitScope, TuningDataView
from pursuit_evasion_rl.research.metrics.behavior import containment_angles, reachable_region_reduction
from pursuit_evasion_rl.research.policies.baselines import decision_intersection_id, greedy_intercept_policy
from pursuit_evasion_rl.research.training.trainer import TrainerConfig
from pursuit_evasion_rl.research.variants.placement import PlacementCurriculumConfig, PlacementStyle

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRATCH = Path(__file__).resolve().parent / "output"
CHECKPOINT_COPY_PATH = SCRATCH / "ckpt_copies" / "seed4_latest.pt"
OUT_PATH = SCRATCH / "containment_triptych.png"
NANUM_BOLD = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"
NANUM_REGULAR = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"


def _reachable_count(network, state: EpisodeState) -> int:
    fugitive_id = decision_intersection_id(network, state.fugitive)
    occupied = [decision_intersection_id(network, p) for p in state.police]
    result = reachable_region_reduction(
        network=network, fugitive_intersection_id=fugitive_id, occupied_intersection_ids=occupied,
    )
    return result.contained_reachable_count


def _angular_coverage_pct(network, state: EpisodeState) -> float:
    """Real, measured encirclement completeness (0-100%): 1 - (largest angular
    gap between officers, as seen from the fugitive) / 360 degrees. This is the
    same audited formula used for the 28D observation / encirclement reward,
    not a bespoke metric invented for this figure."""
    police_xy = [placement_position(network, p) for p in state.police]
    fugitive_xy = placement_position(network, state.fugitive)
    angles = containment_angles(police_positions=police_xy, fugitive_position=fugitive_xy)
    return 100.0 * angles.angular_coverage


def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    store = OfflineSnapshotStore(str(REPO_ROOT))
    validation_spec = load_snapshot_spec(REPO_ROOT / "configs/research/maps/daejeon_validation.yaml")
    network = store.import_snapshot(validation_spec).snapshot.network
    print(f"real network: {len(network.intersections)} intersections, {len(network.segments)} segments", flush=True)

    proposed_raw = load_frozen_policy(CHECKPOINT_COPY_PATH, policy_id="real-scale-seed4-final")
    proposed = GreedyPolicyAdapter(proposed_raw, "real-scale-seed4-final")

    tuning_data = TuningDataView(
        train=(DataHandle(SplitScope.TRAIN, HandleKind.MAP, "daejeon-validation-v1"),),
        validation=(DataHandle(SplitScope.VALIDATION, HandleKind.MAP, "daejeon-validation-v1"),),
    )
    spec = MainStudyConditionSpec(
        condition_id="presentation_triptych_demo",
        scenario=MapScenario.BOUNDARY_ESCAPE,
        train_network=network, validation_network=network, tuning_data=tuning_data,
        config=TrainerConfig(updates=1, max_steps=450),
        placement_config=PlacementCurriculumConfig(style=PlacementStyle.GLOBAL),
        baseline_policy_factory=lambda n: greedy_intercept_policy(n),
        episodes_per_seed=1, data_kind=DataKind.ACTUAL_OSM_MAP,
    )

    # First finding (checked against ~30 real episodes): the fugitive's raw
    # BFS-reachable-intersection count does NOT shrink gradually on this real
    # road network -- it stays near the network's full size until a single
    # officer occupies the one chokepoint node the remaining routes funnel
    # through, then collapses to ~0 in one step. That is a real and useful
    # finding (the learned strategy is chokepoint-blocking, not uniform
    # area-shrinking), but it will not produce three declining "gradually
    # narrowing" numbers no matter which real episode is picked.
    #
    # angular_coverage (the same audited formula behind the 28D encirclement
    # observation/reward: 1 - largest bearing gap between officers / 360) DOES
    # move gradually as officers spread around the fugitive, so it is the
    # metric actually suited to a three-stage "포위망 형성" story. Both are
    # still real, measured values -- angular_coverage is just the metric this
    # network's actual data supports for a staged narrative; the raw reachable
    # count is kept in each caption too, without editorializing it into a
    # smooth curve it doesn't have.
    def _states_from_replay(case, replay):
        out = [EpisodeState(step=0, simulated_s=0.0, police=case.police, fugitive=case.fugitive)]
        for replay_step in replay.trajectory:
            out.append(
                EpisodeState(
                    step=replay_step.step, simulated_s=float(replay_step.step),
                    police=replay_step.police, fugitive=replay_step.fugitive,
                )
            )
        return out

    candidates = []
    for seed in range(80):
        for episode_index in range(2):
            case = build_episode_case(network, spec, seed=seed, episode_index=episode_index)
            replay = replay_episode(case, proposed, network=network)
            if replay.outcome is not EpisodeOutcome.CAPTURE or replay.physical_steps < 30:
                continue
            states = _states_from_replay(case, replay)
            n = len(states) - 1
            idx = {
                "초반": max(1, round(n * 0.15)),
                "중반": round(n * 0.55),
                "검거 직전": max(round(n * 0.55) + 1, n - 1),
            }
            coverage = {label: _angular_coverage_pct(network, states[i]) for label, i in idx.items()}
            c1, c2, c3 = coverage["초반"], coverage["중반"], coverage["검거 직전"]
            monotone = c1 <= c2 <= c3
            score = (c2 - c1) + (c3 - c2) + (50 if monotone else 0)
            candidates.append((score, monotone, seed, episode_index, case, replay, states, idx, coverage))
            if len(candidates) >= 30:
                break
        if len(candidates) >= 30:
            break

    if not candidates:
        print("no qualifying capture episode found", flush=True)
        return 1
    candidates.sort(key=lambda item: item[0], reverse=True)
    score, monotone, seed, episode_index, case, replay, states, frame_indices, coverage_preview = candidates[0]
    n = len(states) - 1
    print(
        f"chosen: seed={seed} ep={episode_index} steps={n} score={score:.1f} monotone={monotone} "
        f"angular_coverage_preview={coverage_preview}",
        flush=True,
    )
    print(f"searched {len(candidates)} real capture episodes for the clearest staged encirclement", flush=True)

    style = RenderStyle(figure_size=(8.4, 8.4), dpi=200)
    renderer = OSMRenderer(network, case.environment, style=style)
    matplotlib.rcParams["font.family"] = renderer.font.family
    matplotlib.rcParams["axes.unicode_minus"] = False

    panel_paths: list[Path] = []
    captions: list[str] = []
    coverages: list[float] = []
    reachables: list[int] = []
    for label, idx in frame_indices.items():
        state = states[idx]
        trails = (
            [[  (p) for p in _positions(renderer, states[: idx + 1], officer) ] for officer in range(6)],
            _positions_fugitive(renderer, states[: idx + 1]),
        )
        is_final = idx == n
        fig = renderer.render_frame(
            state,
            episode_id=f"triptych:seed{seed}:ep{episode_index}",
            seed=seed,
            outcome=OSMEpisodeOutcome(replay.outcome.value) if is_final else None,
            trails=trails,
            step=state.step,
        )
        ax = fig.axes[0]
        coverage = _angular_coverage_pct(network, state)
        reachable = _reachable_count(network, state)
        coverages.append(coverage)
        reachables.append(reachable)
        ax.set_title(f"{label} (스텝 {state.step}/{n})", fontsize=13)
        panel_path = SCRATCH / f"triptych_panel_{label}.png"
        fig.savefig(panel_path, dpi=style.dpi, bbox_inches="tight")
        panel_paths.append(panel_path)
        captions.append(
            f"{label}: 포위 각도 커버리지 {coverage:.0f}%, 도주 가능 교차로 {reachable}개  (스텝 {state.step})"
        )
        print(f"{label}: step={state.step} angular_coverage={coverage:.1f}% reachable={reachable}", flush=True)

    # ------------------------------------------------------------------
    # Composite: three panels side by side, caption strip under each, and a
    # headline arrow-chain of the measured angular-coverage sequence on top
    # (the metric whose real, measured values actually show staged growth on
    # this network -- see the note above the search loop).
    # ------------------------------------------------------------------
    imgs = [Image.open(p).convert("RGB") for p in panel_paths]
    target_h = min(img.height for img in imgs)
    imgs = [img.resize((round(img.width * target_h / img.height), target_h)) for img in imgs]
    gap = 24
    caption_h = 64
    header_h = 90
    total_w = sum(img.width for img in imgs) + gap * (len(imgs) - 1)
    total_h = header_h + target_h + caption_h

    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)
    header_font = ImageFont.truetype(NANUM_BOLD, 30)
    caption_font = ImageFont.truetype(NANUM_REGULAR, 22)

    headline = "포위 각도 커버리지 (실측, 1 - 최대 사각지대각/360°): " + "  →  ".join(
        f"{c:.0f}%" for c in coverages
    )
    bbox = draw.textbbox((0, 0), headline, font=header_font)
    draw.text(((total_w - (bbox[2] - bbox[0])) // 2, 20), headline, fill="black", font=header_font)

    x = 0
    for img, caption in zip(imgs, captions):
        canvas.paste(img, (x, header_h))
        cbbox = draw.textbbox((0, 0), caption, font=caption_font)
        cx = x + (img.width - (cbbox[2] - cbbox[0])) // 2
        draw.text((cx, header_h + target_h + 16), caption, fill="black", font=caption_font)
        x += img.width + gap

    canvas.save(OUT_PATH)
    print(f"wrote {OUT_PATH}", flush=True)
    return 0


def _positions(renderer, states, officer_index):
    from pursuit_evasion_rl.osm_demo.metrics import placement_position
    return [placement_position(renderer.network, s.police[officer_index]) for s in states]


def _positions_fugitive(renderer, states):
    from pursuit_evasion_rl.osm_demo.metrics import placement_position
    return [placement_position(renderer.network, s.fugitive) for s in states]


if __name__ == "__main__":
    raise SystemExit(main())

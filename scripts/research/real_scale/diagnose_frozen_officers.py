"""Diagnose WHY some officers never move in a real failure episode: replay one
episode manually (same logic as replay_episode(), reimplemented here only to
add per-officer, per-step instrumentation) and log, for every officer at every
step: its exact position identity (intersection vs mid-segment), how many
actions are legal, the policy's raw actor logits over legal actions, and the
action actually chosen -- so "never moves" can be told apart into two very
different explanations:
  (a) physically stuck (only STAY is legal, e.g. always mid-segment), or
  (b) sitting at a real decision point with real alternatives, and the
      trained policy's own argmax genuinely keeps choosing STAY.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from pursuit_evasion_rl.osm_demo.environment import FUGITIVE_ID, OSMRoadPursuitEnv, POLICE_COUNT
from pursuit_evasion_rl.osm_demo.metrics import placement_position
from pursuit_evasion_rl.osm_demo.policies import STAY_ACTION
from pursuit_evasion_rl.research.domain import DataKind, MapScenario
from pursuit_evasion_rl.research.evaluation.paired import (
    GoalEvader,
    OfficerObservation,
    REPLAY_CLIP_DISTANCE_M,
    REPLAY_NEAR_RADIUS_M,
    REPLAY_VISION_RANGE_M,
    _Graph,
)
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
from pursuit_evasion_rl.research.variants.observations import Observation28DAdapter
from pursuit_evasion_rl.research.variants.placement import PlacementCurriculumConfig, PlacementStyle

REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = REPO_ROOT / "artifacts/research/checkpoints/real_scale_daejeon_main_study/4/20260802T195032-9ae88320f213/best_validation.pt"


def main() -> int:
    store = OfflineSnapshotStore(str(REPO_ROOT))
    validation_spec = load_snapshot_spec(REPO_ROOT / "configs/research/maps/daejeon_validation.yaml")
    network = store.import_snapshot(validation_spec).snapshot.network

    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    assert payload["update_index"] == 35000
    raw_policy = load_frozen_policy(CHECKPOINT, policy_id="diag")
    proposed = GreedyPolicyAdapter(raw_policy, "diag")

    tuning_data = TuningDataView(
        train=(DataHandle(SplitScope.TRAIN, HandleKind.MAP, "daejeon-validation-v1"),),
        validation=(DataHandle(SplitScope.VALIDATION, HandleKind.MAP, "daejeon-validation-v1"),),
    )
    spec = MainStudyConditionSpec(
        condition_id="frozen_officer_diagnostic",
        scenario=MapScenario.BOUNDARY_ESCAPE,
        train_network=network, validation_network=network, tuning_data=tuning_data,
        config=TrainerConfig(updates=1, max_steps=450),
        placement_config=PlacementCurriculumConfig(style=PlacementStyle.GLOBAL),
        baseline_policy_factory=lambda n: greedy_intercept_policy(n),
        episodes_per_seed=1, data_kind=DataKind.ACTUAL_OSM_MAP,
    )

    # Find a real failure episode within this process (hash() is randomized
    # per-process, so we re-search here rather than trying to re-target the
    # earlier seed=6 episode from a different process).
    from pursuit_evasion_rl.research.evaluation.paired import replay_episode
    chosen_case = None
    for seed in range(60):
        for episode_index in range(2):
            case = build_episode_case(network, spec, seed=seed, episode_index=episode_index)
            replay = replay_episode(case, proposed, network=network)
            if replay.outcome.value != "capture" and 40 <= replay.physical_steps <= 160:
                chosen_case = case
                print(f"chosen failure episode: seed={seed} ep={episode_index} outcome={replay.outcome.value} steps={replay.physical_steps}", flush=True)
                break
        if chosen_case is not None:
            break
    if chosen_case is None:
        print("no qualifying failure episode found", flush=True)
        return 1
    case = chosen_case

    env = OSMRoadPursuitEnv(network, case.environment)
    env.reset(seed=case.fugitive_rng.seed, options={"police": list(case.police), "fugitive": case.fugitive})
    adapter = Observation28DAdapter(network, clip_distance_m=REPLAY_CLIP_DISTANCE_M, near_radius_m=REPLAY_NEAR_RADIUS_M)
    evader = GoalEvader(network, rng=case.fugitive_rng.generator(), graph=_Graph(network), vision_range_m=REPLAY_VISION_RANGE_M)

    last_identity = [None] * POLICE_COUNT
    stay_streak = [0] * POLICE_COUNT
    step = 0
    while env.outcome is None and step < case.termination.timeout_step_limit:
        state = env.episode_state()
        masks = env.action_masks()
        actions = {}
        for officer_id in range(POLICE_COUNT):
            mask = tuple(bool(v) for v in masks[f"police_{officer_id}"])
            legal_count = sum(mask)
            placement = state.police[officer_id]
            identity = placement.identity  # ("segment", id, progress) or ("intersection", id, 0.0)

            features = tuple(float(v) for v in adapter.observe(
                police_index=officer_id, police=state.police, fugitive=state.fugitive,
                step=state.step, max_steps=case.environment.max_steps,
                incoming_heading=state.incoming_headings.get(f"police_{officer_id}"),
            ))
            obs_t = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            officer_ids_t = torch.tensor([officer_id], dtype=torch.long)
            with torch.no_grad():
                logits = raw_policy.actor_logits(obs_t, officer_ids_t)[0]
            masked_logits = [
                (idx, float(logits[idx])) for idx, legal in enumerate(mask) if legal
            ]
            masked_logits.sort(key=lambda x: -x[1])
            action = masked_logits[0][0]
            actions[f"police_{officer_id}"] = action

            moved = identity != last_identity[officer_id]
            if action == STAY_ACTION:
                stay_streak[officer_id] += 1
            else:
                stay_streak[officer_id] = 0

            if officer_id in (0, 1, 2, 3, 4, 5) and step in (0, 1, 2, 5, 10, 20, 40, 60, 80):
                dist = ((placement_position(network, placement)[0] - placement_position(network, state.fugitive)[0]) ** 2 +
                        (placement_position(network, placement)[1] - placement_position(network, state.fugitive)[1]) ** 2) ** 0.5
                print(
                    f"step={step:3d} officer={officer_id} identity={identity} legal={legal_count} "
                    f"top3_legal_logits={masked_logits[:3]} chosen={action} "
                    f"{'STAY' if action == STAY_ACTION else 'MOVE'} moved_since_last={moved} "
                    f"dist_to_fugitive_m={dist:.1f} stay_streak={stay_streak[officer_id]}",
                    flush=True,
                )
            last_identity[officer_id] = identity
        actions[FUGITIVE_ID] = evader.env_provider(network, tuple(placement_position(network, p) for p in state.police))
        env.step(actions)
        step += 1

    print(f"episode ended: outcome={env.outcome} after {step} steps", flush=True)
    print(f"final stay streaks: {dict(enumerate(stay_streak))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

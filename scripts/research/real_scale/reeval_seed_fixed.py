"""Re-run one real-scale seed's final 500-episode paired evaluation against
the lowest-legal-action baseline, using the EpisodeOutcome enum-identity bug
fix in pursuit_evasion_rl/research/experiments/main_study.py.

Background: replay_episode() (research/evaluation/paired.py) returns
research.domain.EpisodeOutcome. main_study.py used to import a *different*
class with the same member names from osm_demo.models, so its
`outcome is EpisodeOutcome.CAPTURE` check was silently always False --
every seed's final paired evaluation recorded a fabricated 0% capture rate
for both the proposed policy and the baseline, regardless of what actually
happened during replay. The import is fixed now; this script redoes only the
evaluation step (training itself was unaffected) against a seed's already-
completed, static final checkpoint.

Usage: python reeval_seed_fixed.py <seed> <checkpoint_path> [output_path]
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from pursuit_evasion_rl.research.canonical import canonical_json
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
from pursuit_evasion_rl.research.statistics.paired import PairedCase
from pursuit_evasion_rl.research.training.trainer import TrainerConfig
from pursuit_evasion_rl.research.variants.placement import PlacementCurriculumConfig, PlacementStyle

CONDITION_ID = "real_scale_daejeon_main_study"
EPISODES = 500


class _LowestLegalActionPolicy:
    policy_id = "lowest-legal-action-v1"

    def act(self, observation) -> int:
        return min(index for index, legal in enumerate(observation.action_mask) if legal)


def main() -> int:
    import torch

    seed = int(sys.argv[1])
    checkpoint_path = Path(sys.argv[2])
    out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(f"real_scale_seed{seed}_outcome_FIXED.json")
    log = lambda msg: print(f"[seed {seed}] {msg}", flush=True)

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["update_index"] == 35000, f"expected the full 35000-update checkpoint, got {payload['update_index']}"
    log(f"checkpoint verified: update_index={payload['update_index']} validation_score={payload['validation_score']}")

    store = OfflineSnapshotStore(str(REPO_ROOT))
    validation_spec = load_snapshot_spec(REPO_ROOT / "configs/research/maps/daejeon_validation.yaml")
    network = store.import_snapshot(validation_spec).snapshot.network

    selected = load_frozen_policy(checkpoint_path, policy_id=f"{CONDITION_ID}-seed{seed}")
    proposed = GreedyPolicyAdapter(selected, f"{CONDITION_ID}-proposed")
    baseline = _LowestLegalActionPolicy()

    tuning_data = TuningDataView(
        train=(DataHandle(SplitScope.TRAIN, HandleKind.MAP, "daejeon-train-v1"),),
        validation=(DataHandle(SplitScope.VALIDATION, HandleKind.MAP, "daejeon-validation-v1"),),
    )
    spec = MainStudyConditionSpec(
        condition_id=CONDITION_ID,
        scenario=MapScenario.BOUNDARY_ESCAPE,
        train_network=network, validation_network=network, tuning_data=tuning_data,
        config=TrainerConfig(updates=1, max_steps=450),
        placement_config=PlacementCurriculumConfig(style=PlacementStyle.GLOBAL),
        baseline_policy_factory=lambda n: _LowestLegalActionPolicy(),
        episodes_per_seed=EPISODES, data_kind=DataKind.ACTUAL_OSM_MAP,
    )

    t0 = time.time()
    paired_cases: list[PairedCase] = []
    proposed_captures = 0
    baseline_captures = 0
    for episode_index in range(EPISODES):
        case = build_episode_case(network, spec, seed=seed, episode_index=episode_index)
        proposed_replay = replay_episode(case, proposed, network=network)
        baseline_replay = replay_episode(case, baseline, network=network)
        outcome_a = proposed_replay.outcome is EpisodeOutcome.CAPTURE
        outcome_b = baseline_replay.outcome is EpisodeOutcome.CAPTURE
        proposed_captures += int(outcome_a)
        baseline_captures += int(outcome_b)
        paired_cases.append(
            PairedCase(case_id=case.case_hash, training_seed=seed, outcome_a=outcome_a, outcome_b=outcome_b)
        )
        if (episode_index + 1) % 100 == 0:
            log(f"  {episode_index + 1}/{EPISODES} proposed_captures={proposed_captures} baseline_captures={baseline_captures}")
    dt = time.time() - t0
    log(
        f"done in {dt:.1f}s: proposed capture_rate={proposed_captures}/{EPISODES}={proposed_captures/EPISODES:.3f} "
        f"baseline capture_rate={baseline_captures}/{EPISODES}={baseline_captures/EPISODES:.3f}"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(canonical_json({
        "condition_id": CONDITION_ID, "training_seed": seed,
        "checkpoint": str(checkpoint_path), "checkpoint_update_index": payload["update_index"],
        "proposed_capture_rate": proposed_captures / EPISODES,
        "baseline_capture_rate": baseline_captures / EPISODES,
        "n_episodes": EPISODES,
        "paired_cases": [
            {"case_id": c.case_id, "outcome_a": c.outcome_a, "outcome_b": c.outcome_b} for c in paired_cases
        ],
    }) + b"\n")
    log(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Evaluate ONE real-scale seed's final trained checkpoint against the two
audited, smarter numeric baselines (greedy-intercept, directed-shortest-path)
on the real (live-fetched) Daejeon validation network -- 500 paired episodes
per baseline, full pre-registered-protocol scale. No training; pure replay,
so it doesn't meaningfully compete with any still-running training seed.

Uses `.outcome.value == "capture"` string comparison throughout (not `is
EpisodeOutcome.CAPTURE`), sidestepping the enum-identity bug fixed earlier in
main_study.py -- this script never imports EpisodeOutcome at all.

Usage: python smart_baseline_eval.py <seed> <checkpoint_path>
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pursuit_evasion_rl.research.canonical import canonical_json
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
from pursuit_evasion_rl.research.policies.baselines import directed_shortest_path_policy, greedy_intercept_policy
from pursuit_evasion_rl.research.statistics.paired import (
    BootstrapPlan,
    EffectDirection,
    PairedCase,
    PracticalThreshold,
    compare_paired_binary,
)
from pursuit_evasion_rl.research.training.trainer import TrainerConfig
from pursuit_evasion_rl.research.variants.placement import PlacementCurriculumConfig, PlacementStyle

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRATCH = Path(__file__).resolve().parent / "output"
EPISODES = 500


def main() -> int:
    seed = int(sys.argv[1])
    checkpoint_path = Path(sys.argv[2])
    log = lambda msg: print(f"[seed {seed}] {msg}", flush=True)

    import torch
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["update_index"] == 35000, f"expected the full 35000-update checkpoint, got {payload['update_index']}"
    log(f"checkpoint verified: update_index={payload['update_index']} validation_score={payload['validation_score']}")

    store = OfflineSnapshotStore(str(REPO_ROOT))
    validation_spec = load_snapshot_spec(REPO_ROOT / "configs/research/maps/daejeon_validation.yaml")
    network = store.import_snapshot(validation_spec).snapshot.network

    proposed_raw = load_frozen_policy(checkpoint_path, policy_id=f"real-scale-seed{seed}-final")
    proposed = GreedyPolicyAdapter(proposed_raw, f"real-scale-seed{seed}-proposed")

    tuning_data = TuningDataView(
        train=(DataHandle(SplitScope.TRAIN, HandleKind.MAP, "daejeon-validation-v1"),),
        validation=(DataHandle(SplitScope.VALIDATION, HandleKind.MAP, "daejeon-validation-v1"),),
    )
    spec_for_case_building = MainStudyConditionSpec(
        condition_id=f"smart_baseline_eval_seed{seed}",
        scenario=MapScenario.BOUNDARY_ESCAPE,
        train_network=network, validation_network=network, tuning_data=tuning_data,
        config=TrainerConfig(updates=1, max_steps=450),
        placement_config=PlacementCurriculumConfig(style=PlacementStyle.GLOBAL),
        baseline_policy_factory=lambda n: greedy_intercept_policy(n),
        episodes_per_seed=EPISODES, data_kind=DataKind.ACTUAL_OSM_MAP,
    )

    results = {}
    for baseline_name, baseline_factory in (
        ("greedy_intercept", greedy_intercept_policy),
        ("directed_shortest_path", directed_shortest_path_policy),
    ):
        baseline = baseline_factory(network)
        log(f"evaluating vs {baseline_name} ({EPISODES} episodes)...")
        t0 = time.time()
        paired_cases: list[PairedCase] = []
        for episode_index in range(EPISODES):
            case = build_episode_case(network, spec_for_case_building, seed=seed, episode_index=episode_index)
            proposed_replay = replay_episode(case, proposed, network=network)
            baseline_replay = replay_episode(case, network=network, team_policy=baseline)
            paired_cases.append(PairedCase(
                case_id=case.case_hash, training_seed=seed,
                outcome_a=proposed_replay.outcome.value == "capture",
                outcome_b=baseline_replay.outcome.value == "capture",
            ))
            if (episode_index + 1) % 100 == 0:
                log(f"  {baseline_name}: {episode_index + 1}/{EPISODES}")
        dt = time.time() - t0
        statistics = compare_paired_binary(
            tuple(paired_cases), comparison_id=f"seed{seed}-vs-{baseline_name}",
            metric="capture_rate", unit="probability",
            plan=BootstrapPlan(n_bootstrap=2000, rng_seed=11, rationale=f"real smart-baseline evaluation of seed {seed}'s final checkpoint"),
            threshold=PracticalThreshold(metric="capture_rate", unit="probability", minimum_effect=0.05, direction=EffectDirection.GREATER_IS_BETTER),
        )
        table = statistics.primary.table
        proposed_capture_rate = (table.n11 + table.n10) / statistics.primary.n_pairs
        baseline_capture_rate = (table.n11 + table.n01) / statistics.primary.n_pairs
        ci = statistics.primary.risk_difference_interval
        log(
            f"  dt={dt:.1f}s proposed_capture_rate={proposed_capture_rate:.3f} "
            f"{baseline_name}_capture_rate={baseline_capture_rate:.3f} "
            f"risk_difference={statistics.primary.risk_difference:.3f} "
            f"ci95=({ci.lower:.3f},{ci.upper:.3f}) "
            f"table(n11={table.n11},n10={table.n10},n01={table.n01},n00={table.n00})"
        )
        results[baseline_name] = {
            "wall_seconds": dt,
            "n_pairs": statistics.primary.n_pairs,
            "proposed_capture_rate": proposed_capture_rate,
            "baseline_capture_rate": baseline_capture_rate,
            "risk_difference": statistics.primary.risk_difference,
            "risk_difference_ci95": [ci.lower, ci.upper],
            "table": {"n11": table.n11, "n10": table.n10, "n01": table.n01, "n00": table.n00},
        }

    out_path = SCRATCH / f"smart_baseline_eval_seed{seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(canonical_json({
        "seed": seed, "checkpoint": str(checkpoint_path), "checkpoint_update_index": payload["update_index"],
        "network": "daejeon-validation-v1 (real, live-fetched)",
        "episodes_per_baseline": EPISODES,
        "results": results,
    }) + b"\n")
    log(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

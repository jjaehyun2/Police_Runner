"""Task 12.4: gate-checked launch of cross-city zero-shot evaluation (numerical baselines only)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from pursuit_evasion_rl.osm_demo.coarsening import coarsen_raw_graph, prepare_model_network
from pursuit_evasion_rl.osm_demo.fixtures import busan, daejeon, seoul

from ..budget import ConditionResourceEstimate, ResourceCeiling
from ..canonical import canonical_json, content_hash
from ..domain import MapScenario
from ..errors import ResearchValidationError
from ..execution import ConditionCostEstimate, plan_execution
from ..experiments.cross_city import (
    LEGACY_BEST_V2_PATH,
    CrossCityTargetSpec,
    require_cross_city_admission,
    run_cross_city_zero_shot,
)
from ..protocol import ProtocolStore
from ..quality import build_attestation
from ..statistics.paired import BootstrapPlan, EffectDirection, PracticalThreshold
from ..training.trainer import TrainerConfig
from ..variants.placement import PlacementCurriculumConfig, PlacementStyle

_TARGET_CITIES = {"busan": busan, "seoul": seoul}


class _LowestLegalActionPolicy:
    """Deterministic FrozenPolicy stand-in -- see cli/main_study.py's identical note."""

    policy_id = "lowest-legal-action-v1"

    def act(self, observation) -> int:
        return min(index for index, legal in enumerate(observation.action_mask) if legal)


def _build_network(fixture_name: str):
    from pursuit_evasion_rl.osm_demo.fixtures import build_fixture

    return prepare_model_network(coarsen_raw_graph(build_fixture(fixture_name))).network


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol-id", required=True, help="a protocol_id already sealed in --protocols-dir")
    parser.add_argument("--protocols-dir", type=Path, default=None)
    parser.add_argument(
        "--frozen-checkpoint-path", required=True,
        help="the selected checkpoint of an already-trained, already-selected policy (never trained further here)",
    )
    parser.add_argument("--source-fixture", default="daejeon", help="the fixture the frozen policy was trained/selected on")
    parser.add_argument("--target-cities", nargs="+", default=["busan", "seoul"], choices=list(_TARGET_CITIES))
    parser.add_argument("--scenario", default="interior_contained", choices=[item.value for item in MapScenario])
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--measured-at-utc", required=True)
    parser.add_argument("--junit-output-dir", type=Path, default=None)
    parser.add_argument("--best-v2-path", default=LEGACY_BEST_V2_PATH)
    parser.add_argument("--output", type=Path, help="write the canonical cross-city report JSON to this path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    protocols_dir = args.protocols_dir or (repo_root / "artifacts" / "research" / "protocols")
    junit_output_dir = args.junit_output_dir or (repo_root / ".quality-gate")

    try:
        attestation = build_attestation(repo_root=repo_root, junit_output_dir=junit_output_dir, generated_at_utc=args.measured_at_utc)
        protocol = ProtocolStore(protocols_dir).read(args.protocol_id)

        source_network = _build_network(args.source_fixture)
        source_map_hash = content_hash(source_network)
        scenario = MapScenario(args.scenario)

        targets = tuple(
            CrossCityTargetSpec(
                city_id=city_id, scenario=scenario, network=_build_network(city_id),
                config=TrainerConfig(updates=1, episodes_per_update=1, max_steps=10, hidden_dims=(4,)),
                placement_config=PlacementCurriculumConfig(style=PlacementStyle.GLOBAL),
                baseline_policy_factory=lambda _network: _LowestLegalActionPolicy(), episodes=args.episodes,
            )
            for city_id in args.target_cities
        )

        ceiling = ResourceCeiling(resource_ceiling_id="cross-city-cli-ceiling", measured_at_utc=args.measured_at_utc, accelerator_hours=10.0, wall_clock_hours=10.0)
        cost_estimates = tuple(
            ConditionCostEstimate(resource=ConditionResourceEstimate(condition_id=target.condition_id, accelerator_hours_per_seed=0.01, wall_clock_hours_per_seed=0.01, env_steps_per_seed=30))
            for target in targets
        )
        resource_plan = plan_execution(protocol_hash=protocol.protocol_hash, ceiling=ceiling, cost_estimates=cost_estimates)
        token = require_cross_city_admission(quality_attestation=attestation, protocol=protocol, resource_plan=resource_plan, best_v2_path=args.best_v2_path)

        report = run_cross_city_zero_shot(
            token, targets, frozen_checkpoint_path=args.frozen_checkpoint_path, source_map_hash=source_map_hash,
            protocol_hash=protocol.protocol_hash,
            bootstrap_plan=BootstrapPlan(n_bootstrap=200, rng_seed=11, rationale="sanity-scale CLI run"),
            threshold=PracticalThreshold(metric="capture_rate", unit="probability", minimum_effect=0.05, direction=EffectDirection.GREATER_IS_BETTER),
        )
    except ResearchValidationError as exc:
        sys.stderr.buffer.write(canonical_json(exc.as_dict()) + b"\n")
        return 2

    payload = {
        "admission_hash": report.admission_hash,
        "report_hash": report.report_hash,
        "ledger_conserved": report.ledger.conserved,
        "source_map_hash": source_map_hash,
        "records": [
            {
                "condition_id": record.condition_id, "execution_status": record.execution_status.value,
                "status_reason": record.status_reason, "measured": record.is_measured,
                "result_value": record.result.value if record.result is not None else None,
            }
            for record in report.ledger.records
        ],
    }
    output = canonical_json(payload) + b"\n"
    if args.output is None:
        sys.stdout.buffer.write(output)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(output)
    return 0 if report.ledger.conserved else 1


if __name__ == "__main__":
    raise SystemExit(main())

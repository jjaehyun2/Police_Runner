"""Task 12.3: gate-checked launch of the observation/reward/placement/stabilization ablation matrix."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from pursuit_evasion_rl.osm_demo.coarsening import coarsen_raw_graph, prepare_model_network
from pursuit_evasion_rl.osm_demo.fixtures import daejeon

from ..budget import ConditionResourceEstimate, ResourceCeiling
from ..canonical import canonical_json
from ..domain import MapScenario
from ..errors import ResearchValidationError
from ..execution import ConditionCostEstimate, plan_execution
from ..experiments.ablations import LEGACY_BEST_V2_PATH, build_ablation_matrix, require_ablation_admission, run_ablation_matrix
from ..maps.splits import DataHandle, HandleKind, SplitScope, TuningDataView
from ..protocol import ProtocolStore
from ..quality import build_attestation
from ..runs.manifest import RunManifestStore, RunProvenance
from ..statistics.paired import BootstrapPlan, EffectDirection, PracticalThreshold
from ..training.trainer import TrainerConfig
from ..variants.placement import PlacementCurriculumConfig, PlacementStyle


class _LowestLegalActionPolicy:
    """Deterministic FrozenPolicy stand-in -- see cli/main_study.py's identical note."""

    policy_id = "lowest-legal-action-v1"

    def act(self, observation) -> int:
        return min(index for index, legal in enumerate(observation.action_mask) if legal)


def _build_network():
    return prepare_model_network(coarsen_raw_graph(daejeon())).network


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol-id", required=True, help="a protocol_id already sealed in --protocols-dir")
    parser.add_argument("--protocols-dir", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-output-root", type=Path, default=None)
    parser.add_argument("--scenario", default="interior_contained", choices=[item.value for item in MapScenario])
    parser.add_argument("--seeds", type=int, default=1, help="explicitly reduced below the confirmatory floor for this sanity-scale run")
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--episodes-per-update", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--one-arm-per-axis", action="store_true", help="run only the first trainable arm of each axis, plus every not_run arm")
    parser.add_argument("--measured-at-utc", required=True)
    parser.add_argument("--junit-output-dir", type=Path, default=None)
    parser.add_argument("--best-v2-path", default=LEGACY_BEST_V2_PATH)
    parser.add_argument("--output", type=Path, help="write the canonical ablation report JSON to this path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    protocols_dir = args.protocols_dir or (repo_root / "artifacts" / "research" / "protocols")
    runs_dir = args.runs_dir or (repo_root / "artifacts" / "research" / "runs")
    checkpoint_output_root = args.checkpoint_output_root or (repo_root / "artifacts" / "research" / "checkpoints")
    junit_output_dir = args.junit_output_dir or (repo_root / ".quality-gate")

    try:
        attestation = build_attestation(repo_root=repo_root, junit_output_dir=junit_output_dir, generated_at_utc=args.measured_at_utc)
        protocol = ProtocolStore(protocols_dir).read(args.protocol_id)

        network = _build_network()
        tuning_data = TuningDataView(
            train=(DataHandle(SplitScope.TRAIN, HandleKind.MAP, "ablation-train"),),
            validation=(DataHandle(SplitScope.VALIDATION, HandleKind.MAP, "ablation-validation"),),
        )
        base_config = TrainerConfig(updates=args.updates, episodes_per_update=args.episodes_per_update, max_steps=args.max_steps)
        scenario = MapScenario(args.scenario)
        arms = build_ablation_matrix(
            scenario, network=network, tuning_data=tuning_data, base_config=base_config,
            baseline_policy_factory=lambda _network: _LowestLegalActionPolicy(), episodes_per_seed=args.episodes_per_seed,
            placement_config=PlacementCurriculumConfig(style=PlacementStyle.GLOBAL),
        )

        if args.one_arm_per_axis:
            seen: set[str] = set()
            subset = []
            for arm in arms:
                if arm.condition_spec is None:
                    subset.append(arm)
                    continue
                prefix = arm.condition_id.split("__")[0]
                if prefix in seen:
                    continue
                seen.add(prefix)
                subset.append(arm)
            arms = tuple(subset)

        ceiling = ResourceCeiling(
            resource_ceiling_id="ablation-cli-ceiling", measured_at_utc=args.measured_at_utc,
            accelerator_hours=10.0, wall_clock_hours=10.0,
        )
        cost_estimates = tuple(
            ConditionCostEstimate(resource=ConditionResourceEstimate(
                condition_id=arm.condition_id, accelerator_hours_per_seed=0.01, wall_clock_hours_per_seed=0.01, env_steps_per_seed=30,
            ))
            for arm in arms if arm.condition_spec is not None
        )
        resource_plan = plan_execution(protocol_hash=protocol.protocol_hash, ceiling=ceiling, cost_estimates=cost_estimates)
        token = require_ablation_admission(
            quality_attestation=attestation, protocol=protocol, resource_plan=resource_plan, best_v2_path=args.best_v2_path,
        )

        runs = RunManifestStore(runs_dir)

        def _provenance() -> RunProvenance:
            return RunProvenance(
                code_hash=attestation.code_hash_at_attestation, dirty_tree=False,
                dependency_hash=attestation.code_hash_at_attestation, runtime="python3.11+torch2.8.0+cpu",
                device="cpu", map_hash="ablation-network", split="train", seed=args.seeds,
                input_hashes={"map": "ablation-network"},
            )

        report = run_ablation_matrix(
            token, arms, seeds=args.seeds, runs=runs, output_root=str(checkpoint_output_root), provenance_factory=_provenance,
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

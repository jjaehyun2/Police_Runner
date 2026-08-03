"""Task 12.1: gate-checked launch of the protocol-blind pilot's sanity matrix."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from pursuit_evasion_rl.osm_demo.coarsening import coarsen_raw_graph, prepare_model_network
from pursuit_evasion_rl.osm_demo.fixtures import daejeon

from ..budget import ResourceCeiling
from ..canonical import canonical_json
from ..errors import ResearchValidationError
from ..experiments.pilot import (
    LEGACY_BEST_V2_PATH,
    PilotConditionSpec,
    cost_pilot_matrix,
    require_pilot_admission,
    run_pilot,
)
from ..maps.splits import DataHandle, HandleKind, SplitScope, TuningDataView
from ..protocol import ProtocolStore
from ..quality import build_attestation
from ..runs.manifest import RunManifestStore, RunProvenance
from ..training.trainer import TrainerConfig


def _build_pilot_network():
    """The same small, offline, real fixture network the offline test suite trains against."""
    return prepare_model_network(coarsen_raw_graph(daejeon())).network


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol-id", required=True, help="a protocol_id already sealed in --protocols-dir")
    parser.add_argument("--protocols-dir", type=Path, default=None, help="defaults to <repo-root>/artifacts/research/protocols")
    parser.add_argument("--runs-dir", type=Path, default=None, help="defaults to <repo-root>/artifacts/research/runs")
    parser.add_argument("--checkpoint-output-root", type=Path, default=None, help="defaults to <repo-root>/artifacts/research/checkpoints")
    parser.add_argument("--condition-id", default="pilot_sanity_interior")
    parser.add_argument("--training-seed", type=int, default=11)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--episodes-per-update", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--resource-ceiling-id", default="pilot-ceiling")
    parser.add_argument("--measured-at-utc", required=True)
    parser.add_argument("--accelerator-hours", type=float, default=10.0)
    parser.add_argument("--wall-clock-hours", type=float, default=10.0)
    parser.add_argument("--junit-output-dir", type=Path, default=None)
    parser.add_argument("--best-v2-path", default=LEGACY_BEST_V2_PATH)
    parser.add_argument("--output", type=Path, help="write the canonical pilot report JSON to this path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    protocols_dir = args.protocols_dir or (repo_root / "artifacts" / "research" / "protocols")
    runs_dir = args.runs_dir or (repo_root / "artifacts" / "research" / "runs")
    checkpoint_output_root = args.checkpoint_output_root or (repo_root / "artifacts" / "research" / "checkpoints")
    junit_output_dir = args.junit_output_dir or (repo_root / ".quality-gate")

    try:
        attestation = build_attestation(
            repo_root=repo_root, junit_output_dir=junit_output_dir, generated_at_utc=args.measured_at_utc,
        )
        protocols = ProtocolStore(protocols_dir)
        protocol = protocols.read(args.protocol_id)

        network = _build_pilot_network()
        tuning_data = TuningDataView(
            train=(DataHandle(SplitScope.TRAIN, HandleKind.MAP, "pilot-train"),),
            validation=(DataHandle(SplitScope.VALIDATION, HandleKind.MAP, "pilot-validation"),),
        )
        config = TrainerConfig(updates=args.updates, episodes_per_update=args.episodes_per_update, max_steps=args.max_steps)
        spec = PilotConditionSpec(
            condition_id=args.condition_id, train_network=network, validation_network=network,
            tuning_data=tuning_data, config=config, training_seed=args.training_seed,
        )
        ceiling = ResourceCeiling(
            resource_ceiling_id=args.resource_ceiling_id, measured_at_utc=args.measured_at_utc,
            accelerator_hours=args.accelerator_hours, wall_clock_hours=args.wall_clock_hours,
        )
        plan = cost_pilot_matrix(
            (spec,), protocol_hash=protocol.protocol_hash, ceiling=ceiling, evaluation_episodes=1,
            accelerator_hours_per_million_env_steps=1.0, wall_clock_hours_per_million_env_steps=1.0,
        )
        token = require_pilot_admission(quality_attestation=attestation, protocol=protocol, resource_plan=plan, best_v2_path=args.best_v2_path)

        runs = RunManifestStore(runs_dir)

        def _provenance() -> RunProvenance:
            return RunProvenance(
                code_hash=attestation.code_hash_at_attestation, dirty_tree=False,
                dependency_hash=attestation.code_hash_at_attestation, runtime="python3.11+torch2.8.0+cpu",
                device="cpu", map_hash="pilot-network", split="train", seed=args.training_seed,
                input_hashes={"map": "pilot-network"},
            )

        report = run_pilot(token, (spec,), runs=runs, output_root=str(checkpoint_output_root), provenance_factory=_provenance)
    except ResearchValidationError as exc:
        sys.stderr.buffer.write(canonical_json(exc.as_dict()) + b"\n")
        return 2

    payload = {
        "admission_hash": report.admission_hash,
        "all_completed": report.all_completed,
        "report_hash": report.report_hash,
        "outcomes": [
            {
                "condition_id": item.condition_id, "execution_status": item.execution_status.value,
                "status_reason": item.status_reason, "run_id": item.run_id,
                "resource_actual": item.resource_actual, "error_artifact_hash": item.error_artifact_hash,
            }
            for item in report.outcomes
        ],
    }
    output = canonical_json(payload) + b"\n"
    if args.output is None:
        sys.stdout.buffer.write(output)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(output)
    return 0 if report.all_completed else 1


if __name__ == "__main__":
    raise SystemExit(main())

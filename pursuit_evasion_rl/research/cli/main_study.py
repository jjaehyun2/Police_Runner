"""Task 12.2: gate-checked launch of the multi-seed main-study sanity matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from pursuit_evasion_rl.osm_demo.coarsening import coarsen_raw_graph, prepare_model_network
from pursuit_evasion_rl.osm_demo.fixtures import daejeon

from ..budget import SampleSizePlan, SampleSizeStatus
from ..canonical import canonical_json
from ..domain import ExecutionStatus, MapScenario
from ..errors import ResearchValidationError
from ..experiments.main_study import (
    LEGACY_BEST_V2_PATH,
    MainStudyConditionSpec,
    require_main_study_admission,
    run_main_study,
)
from ..experiments.pilot import PilotConditionOutcome, PilotReport
from ..maps.splits import DataHandle, HandleKind, SplitScope, TuningDataView
from ..protocol import ProtocolStore
from ..quality import build_attestation
from ..runs.manifest import RunManifestStore, RunProvenance
from ..statistics.paired import BootstrapPlan, EffectDirection, PracticalThreshold
from ..training.trainer import TrainerConfig
from ..variants.placement import PlacementCurriculumConfig, PlacementStyle


class _LowestLegalActionPolicy:
    """Deterministic FrozenPolicy stand-in for the sanity matrix's baseline arm.

    A real study run would compare against the audited numeric baselines
    (``greedy_intercept_policy``, ``directed_shortest_path_policy``); those
    are team-coordinated and expose a ``recommend()`` interface rather than
    :class:`~pursuit_evasion_rl.research.evaluation.paired.FrozenPolicy`'s
    per-officer ``act()``, so bridging them is deliberately left to a
    dedicated follow-up rather than risking an incorrect adapter here.
    """

    policy_id = "lowest-legal-action-v1"

    def act(self, observation) -> int:
        return min(index for index, legal in enumerate(observation.action_mask) if legal)


def _build_network():
    return prepare_model_network(coarsen_raw_graph(daejeon())).network


def _load_pilot_report(path: Path) -> PilotReport:
    payload = json.loads(path.read_text(encoding="utf-8"))
    outcomes = tuple(
        PilotConditionOutcome(
            condition_id=item["condition_id"], execution_status=ExecutionStatus(item["execution_status"]),
            status_reason=item["status_reason"], run_id=item["run_id"],
            resource_actual=dict(item.get("resource_actual", {})), error_artifact_hash=item.get("error_artifact_hash"),
        )
        for item in payload["outcomes"]
    )
    return PilotReport(admission_hash=payload["admission_hash"], outcomes=outcomes)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol-id", required=True, help="a protocol_id already sealed in --protocols-dir")
    parser.add_argument("--protocols-dir", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-output-root", type=Path, default=None)
    parser.add_argument("--pilot-report-path", type=Path, required=True, help="the JSON report written by cli.pilot")
    parser.add_argument("--condition-id", default="main_study_sanity_interior")
    parser.add_argument("--seeds", type=int, default=2, help="explicitly reduced below the confirmatory floor for this sanity-scale run")
    parser.add_argument("--episodes-per-seed", type=int, default=2)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--episodes-per-update", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--measured-at-utc", required=True)
    parser.add_argument("--junit-output-dir", type=Path, default=None)
    parser.add_argument("--best-v2-path", default=LEGACY_BEST_V2_PATH)
    parser.add_argument("--output", type=Path, help="write the canonical main-study report JSON to this path")
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
        pilot_report = _load_pilot_report(args.pilot_report_path)
        token = require_main_study_admission(
            quality_attestation=attestation, protocol=protocol, pilot_report=pilot_report, best_v2_path=args.best_v2_path,
        )

        network = _build_network()
        tuning_data = TuningDataView(
            train=(DataHandle(SplitScope.TRAIN, HandleKind.MAP, "main-study-train"),),
            validation=(DataHandle(SplitScope.VALIDATION, HandleKind.MAP, "main-study-validation"),),
        )
        config = TrainerConfig(updates=args.updates, episodes_per_update=args.episodes_per_update, max_steps=args.max_steps)
        spec = MainStudyConditionSpec(
            condition_id=args.condition_id, scenario=MapScenario.INTERIOR_CONTAINED, train_network=network,
            validation_network=network, tuning_data=tuning_data, config=config,
            placement_config=PlacementCurriculumConfig(style=PlacementStyle.GLOBAL),
            baseline_policy_factory=lambda _network: _LowestLegalActionPolicy(), episodes_per_seed=args.episodes_per_seed,
        )
        sample_size_plan = SampleSizePlan(
            status=SampleSizeStatus.EXPLORATORY, seeds_per_condition=args.seeds, episodes_per_condition=args.episodes_per_seed,
            resource_ceiling_id="main-study-sanity-ceiling", reduction_inputs={"reason": "bounded sanity-scale proof run"},
            power_limitation="explicitly reduced below the confirmatory floor for a bounded sanity-scale proof run",
        )
        runs = RunManifestStore(runs_dir)

        def _provenance() -> RunProvenance:
            return RunProvenance(
                code_hash=attestation.code_hash_at_attestation, dirty_tree=False,
                dependency_hash=attestation.code_hash_at_attestation, runtime="python3.11+torch2.8.0+cpu",
                device="cpu", map_hash="main-study-network", split="train", seed=args.seeds,
                input_hashes={"map": "main-study-network"},
            )

        report = run_main_study(
            token, (spec,), sample_size_plan=sample_size_plan, runs=runs, output_root=str(checkpoint_output_root),
            provenance_factory=_provenance,
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
        "condition_results": [
            {
                "condition_id": item.condition_id,
                "execution_status": item.execution_record.execution_status.value,
                "status_reason": item.execution_record.status_reason,
                "measured": item.execution_record.is_measured,
                "result_value": item.execution_record.result.value if item.execution_record.result is not None else None,
                "n_pairs": item.statistics.primary.n_pairs if item.statistics is not None else None,
                "n_seeds": item.statistics.primary.n_seeds if item.statistics is not None else None,
                "risk_difference": item.statistics.primary.risk_difference if item.statistics is not None else None,
                "seed_outcomes": [
                    {"training_seed": seed.training_seed, "execution_status": seed.execution_status.value, "run_id": seed.run_id}
                    for seed in item.seed_outcomes
                ],
            }
            for item in report.condition_results
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

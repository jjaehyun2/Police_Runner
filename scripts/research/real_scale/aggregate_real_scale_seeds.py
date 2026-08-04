"""Pool the 5 independently-run real_scale_seed{N}_outcome.json files into the
final MainStudyReport-equivalent, using main_study.pool_seed_outcomes() so the
result is byte-identical to what a single-process 5-seed run would produce.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pursuit_evasion_rl.research.canonical import canonical_json
from pursuit_evasion_rl.research.domain import ExecutionStatus
from pursuit_evasion_rl.research.experiments.main_study import (
    LEGACY_BEST_V2_PATH,
    SeedOutcome,
    pool_seed_outcomes,
    require_main_study_admission,
)
from pursuit_evasion_rl.research.experiments.pilot import PilotConditionOutcome, PilotReport
from pursuit_evasion_rl.research.protocol import ProtocolStore
from pursuit_evasion_rl.research.quality import build_attestation
from pursuit_evasion_rl.research.statistics.paired import (
    BootstrapPlan,
    CaseStatus,
    EffectDirection,
    PairedCase,
    PracticalThreshold,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRATCH = Path(__file__).resolve().parent / "output"


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


def _paired_case_from_dict(data: dict) -> PairedCase:
    return PairedCase(
        case_id=data["case_id"], training_seed=data["training_seed"],
        status=CaseStatus(data["status"]), outcome_a=data["outcome_a"], outcome_b=data["outcome_b"],
        checkpoint_time=data["checkpoint_time"], reason=data.get("reason"),
        schema_version=data.get("schema_version", "1.0"),
    )


def load_seed_outcome(path: Path) -> SeedOutcome:
    data = json.loads(path.read_text(encoding="utf-8"))
    return SeedOutcome(
        condition_id=data["condition_id"], training_seed=data["training_seed"],
        execution_status=ExecutionStatus(data["execution_status"]), status_reason=data["status_reason"],
        run_id=data["run_id"],
        paired_cases=tuple(_paired_case_from_dict(item) for item in data.get("paired_cases", ())),
        error_artifact_hash=data.get("error_artifact_hash"),
        schema_version=data.get("schema_version", "1.0"),
    )


def main(seed_outcome_paths: list[Path], *, out_path: Path) -> int:
    outcomes = tuple(load_seed_outcome(path) for path in seed_outcome_paths)
    print(f"loaded {len(outcomes)} seed outcomes", flush=True)
    for outcome in outcomes:
        print(
            f"  seed {outcome.training_seed}: {outcome.execution_status.value}, "
            f"{len(outcome.paired_cases)} paired case(s) -- {outcome.status_reason}",
            flush=True,
        )

    attestation = build_attestation(
        repo_root=REPO_ROOT, junit_output_dir=REPO_ROOT / ".quality-gate",
        generated_at_utc="2026-08-01T00:00:00Z",
    )
    protocol = ProtocolStore(REPO_ROOT / "artifacts/research/protocols").read(
        "20260731T193445-df6a1df8902d"
    )
    pilot_report = _load_pilot_report(SCRATCH / "pilot_report.json")
    token = require_main_study_admission(
        quality_attestation=attestation, protocol=protocol, pilot_report=pilot_report,
        best_v2_path=LEGACY_BEST_V2_PATH,
    )

    class _Spec:
        condition_id = outcomes[0].condition_id if outcomes else "real_scale_daejeon_main_study"

    result = pool_seed_outcomes(
        token, _Spec(), outcomes,
        bootstrap_plan=BootstrapPlan(n_bootstrap=2000, rng_seed=11, rationale="real-scale 5-seed confirmatory pooling"),
        threshold=PracticalThreshold(metric="capture_rate", unit="probability", minimum_effect=0.05, direction=EffectDirection.GREATER_IS_BETTER),
    )
    print(f"execution_status={result.execution_record.execution_status.value}", flush=True)
    print(f"status_reason={result.execution_record.status_reason}", flush=True)
    if result.execution_record.result is not None:
        print(f"result_value={result.execution_record.result.value}", flush=True)
    if result.statistics is not None:
        table = result.statistics.primary.table
        print(f"n_pairs={result.statistics.primary.n_pairs} n_seeds={result.statistics.primary.n_seeds}", flush=True)
        print(f"risk_difference={result.statistics.primary.risk_difference}", flush=True)
        print(f"table n11={table.n11} n10={table.n10} n01={table.n01} n00={table.n00}", flush=True)

    payload = {
        "condition_id": result.condition_id,
        "execution_status": result.execution_record.execution_status.value,
        "status_reason": result.execution_record.status_reason,
        "result_value": result.execution_record.result.value if result.execution_record.result is not None else None,
        "n_pairs": result.statistics.primary.n_pairs if result.statistics is not None else None,
        "n_seeds": result.statistics.primary.n_seeds if result.statistics is not None else None,
        "risk_difference": result.statistics.primary.risk_difference if result.statistics is not None else None,
        "seed_outcomes": [
            {"training_seed": o.training_seed, "execution_status": o.execution_status.value, "run_id": o.run_id}
            for o in outcomes
        ],
    }
    out_path.write_bytes(canonical_json(payload) + b"\n")
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    # Usage: aggregate_real_scale_seeds.py [seed_outcome.json ...] [-o output.json]
    args = sys.argv[1:]
    output = SCRATCH / "real_scale_pooled_report.json"
    if "-o" in args:
        index = args.index("-o")
        output = Path(args[index + 1])
        del args[index : index + 2]
    paths = [Path(p) for p in args] if args else [
        SCRATCH / f"real_scale_seed{i}_outcome.json" for i in range(5)
    ]
    raise SystemExit(main(paths, out_path=output))

"""Plan a Condition matrix's execution against a Resource_Ceiling and report its ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from ..budget import DEFAULT_EPISODES, ResourceCeiling, SampleSizeStatus
from ..canonical import canonical_json
from ..domain import ExecutionStatus
from ..errors import ResearchValidationError
from ..execution import (
    ConditionExecutionRecord,
    LlmCostEstimate,
    MeasuredResult,
    build_condition_execution_ledger,
    cost_estimates_for_matrix,
    plan_execution,
    success_only_views,
)
from ..variants.factory import full_condition_matrix, validate_condition_matrix


class _Schedule:
    """The environment-step-determining subset of a ``TrainerConfig``."""

    def __init__(self, updates: int, episodes_per_update: int, max_steps: int) -> None:
        self.updates = updates
        self.episodes_per_update = episodes_per_update
        self.max_steps = max_steps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-hash", required=True, help="content hash of the frozen ResearchProtocol")
    parser.add_argument("--resource-ceiling-id", required=True)
    parser.add_argument("--measured-at-utc", required=True)
    parser.add_argument("--accelerator-hours", type=float, required=True)
    parser.add_argument("--wall-clock-hours", type=float, required=True)
    parser.add_argument("--accelerator-hours-per-million-steps", type=float, default=1.0)
    parser.add_argument("--wall-clock-hours-per-million-steps", type=float, default=1.0)
    parser.add_argument("--updates", type=int, default=750)
    parser.add_argument("--episodes-per-update", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=450)
    parser.add_argument("--evaluation-episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--llm-provider", help="omit entirely when no Condition consults an LLM")
    parser.add_argument("--llm-calls-per-seed", type=int, default=0)
    parser.add_argument("--llm-usd-per-call", type=float, default=0.0)
    parser.add_argument(
        "--records",
        type=Path,
        help="JSON array of execution records; omit to emit the pre-execution plan only",
    )
    parser.add_argument("--output", type=Path, help="write canonical JSON to this path instead of stdout")
    return parser


def _record(payload: Mapping[str, Any], *, protocol_hash: str, default_status: SampleSizeStatus) -> ConditionExecutionRecord:
    result = payload.get("result", {"measured": False})
    if not isinstance(result, Mapping):
        raise ResearchValidationError(
            "INVALID_EXECUTION_RESULT", "result must be an object with a measured flag", path="result", actual=result
        )
    return ConditionExecutionRecord(
        condition_id=payload["condition_id"],
        protocol_hash=protocol_hash,
        execution_status=ExecutionStatus(payload["execution_status"]),
        status_reason=payload["status_reason"],
        sample_size_status=SampleSizeStatus(payload.get("sample_size_status", default_status)),
        artifact_hashes=dict(payload.get("artifact_hashes", {})),
        result=MeasuredResult(value=result.get("value")) if result.get("measured") else None,
        run_ids=tuple(payload.get("run_ids", ())),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        specs = full_condition_matrix()
        validate_condition_matrix(specs)
        plan = plan_execution(
            protocol_hash=args.protocol_hash,
            ceiling=ResourceCeiling(
                resource_ceiling_id=args.resource_ceiling_id,
                measured_at_utc=args.measured_at_utc,
                accelerator_hours=args.accelerator_hours,
                wall_clock_hours=args.wall_clock_hours,
            ),
            cost_estimates=cost_estimates_for_matrix(
                specs,
                schedule=_Schedule(args.updates, args.episodes_per_update, args.max_steps),
                evaluation_episodes=args.evaluation_episodes,
                accelerator_hours_per_million_env_steps=args.accelerator_hours_per_million_steps,
                wall_clock_hours_per_million_env_steps=args.wall_clock_hours_per_million_steps,
                llm_cost=(
                    LlmCostEstimate(
                        provider=args.llm_provider,
                        calls_per_seed=args.llm_calls_per_seed,
                        usd_per_call=args.llm_usd_per_call,
                    )
                    if args.llm_provider
                    else None
                ),
            ),
        )
        report: dict[str, Any] = {
            "protocol_hash": plan.protocol_hash,
            "plan_hash": plan.plan_hash,
            "sample_size_status": plan.sample_size_plan.status.value,
            "seeds_per_condition": plan.sample_size_plan.seeds_per_condition,
            "episodes_per_condition": plan.sample_size_plan.episodes_per_condition,
            "planned_condition_count": len(plan.planned_condition_ids),
            "total_env_steps": plan.total_env_steps,
            "total_llm_usd": plan.total_llm_usd,
            "confirmatory_eligible": plan.is_confirmatory_eligible,
        }
        if args.records is not None:
            payloads = json.loads(args.records.read_text(encoding="utf-8"))
            ledger = build_condition_execution_ledger(
                plan,
                (
                    _record(item, protocol_hash=args.protocol_hash, default_status=plan.sample_size_plan.status)
                    for item in payloads
                ),
            )
            views = success_only_views(ledger)
            report["ledger"] = ledger.claim_input()
            report["success_only_classification"] = views.classification.value
            report["success_only_condition_count"] = views.success_only.count
            report["full_population_condition_count"] = views.full_population.count
    except (ResearchValidationError, OSError, json.JSONDecodeError, KeyError) as exc:
        payload = (
            exc.as_dict()
            if isinstance(exc, ResearchValidationError)
            else {"code": "EXECUTION_CLI_ERROR", "message": str(exc)}
        )
        sys.stderr.buffer.write(canonical_json(payload) + b"\n")
        return 2
    output = canonical_json(report) + b"\n"
    if args.output is None:
        sys.stdout.buffer.write(output)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the offline quality gate and report or attest pilot-admission eligibility."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from ..canonical import canonical_json
from ..errors import ResearchValidationError
from ..quality import GateAttestation, PilotAdmissionError, assert_pilot_admission, build_attestation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--junit-output-dir", type=Path, default=None, help="defaults to <repo-root>/.quality-gate")
    parser.add_argument("--output", type=Path, help="write the canonical attestation JSON to this path")
    parser.add_argument("--python-executable", help="defaults to the interpreter running this CLI")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    junit_output_dir = (args.junit_output_dir or (repo_root / ".quality-gate")).resolve()

    try:
        attestation: GateAttestation = build_attestation(
            repo_root=repo_root,
            junit_output_dir=junit_output_dir,
            generated_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            python_executable=args.python_executable,
        )
    except ResearchValidationError as exc:
        sys.stderr.buffer.write(canonical_json(exc.as_dict()) + b"\n")
        return 2

    payload = {
        "attestation_hash": attestation.attestation_hash,
        "generated_at_utc": attestation.generated_at_utc,
        "pilot_admission_eligible": attestation.pilot_admission_eligible,
        "missing_property_ids": list(attestation.coverage.missing_property_ids),
        "noncompliant_property_ids": list(attestation.coverage.noncompliant_property_ids),
        "deferred_property_ids": list(attestation.coverage.deferred_property_ids),
        "suite_runs": [
            {
                "target": run.target,
                "total": run.total,
                "passed": run.passed,
                "failures": run.failures,
                "errors": run.errors,
                "skipped": run.skipped,
                "clean": run.clean,
            }
            for run in attestation.suite_runs
        ],
    }
    output = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    if args.output is None:
        sys.stdout.buffer.write(output)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_json(payload))

    try:
        assert_pilot_admission(attestation, repo_root=repo_root)
    except PilotAdmissionError as exc:
        sys.stderr.buffer.write(canonical_json(exc.as_dict()) + b"\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

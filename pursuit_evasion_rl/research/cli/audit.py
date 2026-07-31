"""Create a canonical legacy-asset and prior-result audit report."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ..audit import audit_repository
from ..canonical import canonical_json
from ..errors import ResearchValidationError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--output", type=Path, help="write canonical JSON to this path")
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="record eligible assets as not_tested instead of executing targeted pytest files",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = audit_repository(args.root, execute_tests=not args.skip_tests)
    except ResearchValidationError as exc:
        sys.stderr.buffer.write(canonical_json(exc.as_dict()) + b"\n")
        return 2
    payload = canonical_json(report) + b"\n"
    if args.output is None:
        sys.stdout.buffer.write(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Deprecated wrapper for :mod:`pursuit_evasion_rl.research.cli.evaluate`."""
from __future__ import annotations

from pursuit_evasion_rl.research.cli.evaluate import LearnedPolice


def _legacy_main(argv=None):
    from pursuit_evasion_rl.research.cli.evaluate import build_parser, main as package_main
    build_parser().parse_args(argv)
    return package_main(argv)


def main(argv=None):
    from pursuit_evasion_rl.research.migration import dispatch_legacy_script
    return dispatch_legacy_script("eval_trained", argv, legacy_main=_legacy_main)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ("LearnedPolice", "main")

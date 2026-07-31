"""Deprecated wrapper for :mod:`pursuit_evasion_rl.research.cli.train`."""
from __future__ import annotations


def _legacy_main(argv=None):
    from pursuit_evasion_rl.research.cli.train import build_parser, main as package_main
    build_parser().parse_args(argv)
    return package_main(argv)


def main(argv=None):
    from pursuit_evasion_rl.research.migration import dispatch_legacy_script
    return dispatch_legacy_script("train_osm_pursuit", argv, legacy_main=_legacy_main)


if __name__ == "__main__":
    raise SystemExit(main())

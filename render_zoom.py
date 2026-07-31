"""Deprecated zoom-render wrapper for the package evaluation CLI."""
from __future__ import annotations
import sys
from pursuit_evasion_rl.research.cli.evaluate import main

if __name__ == "__main__":
    raise SystemExit(main(["render", *sys.argv[1:]]))

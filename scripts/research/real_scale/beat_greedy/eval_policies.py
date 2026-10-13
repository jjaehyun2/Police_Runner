"""Paired capture-rate comparison of several policies on the same sealed cases.

Usage: python eval_policies.py <network> <episodes> <spec> [<spec> ...]
  spec: greedy | dsp | encircle | rule:<name> | ckpt:<path>
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import capture_rate, evaluate  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"


def main() -> int:
    network_name, episodes, specs = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    tag = f"beatgreedy-{network_name}"
    from common import FUGITIVE_SPEED_MPS, POLICE_SPEED_MPS
    print(f"police={POLICE_SPEED_MPS} fugitive={FUGITIVE_SPEED_MPS}", flush=True)
    all_results = {}
    for spec in specs:
        t0 = time.time()
        res = evaluate(spec, network_name, tag, episodes)
        all_results[spec] = res
        outcomes = [o for o, _ in res.values()]
        cap_steps = [s for o, s in res.values() if o == "capture"]
        print(
            f"{spec:60s} capture={capture_rate(res):.3f} escape={outcomes.count('escape')/len(outcomes):.3f} "
            f"timeout={outcomes.count('timeout')/len(outcomes):.3f} "
            f"mean_capture_steps={sum(cap_steps)/max(len(cap_steps),1):.1f} ({time.time()-t0:.0f}s)",
            flush=True,
        )
    if len(specs) >= 2:
        base = all_results[specs[0]]
        for spec in specs[1:]:
            other = all_results[spec]
            n10 = sum(other[i][0] == "capture" and base[i][0] != "capture" for i in base)
            n01 = sum(other[i][0] != "capture" and base[i][0] == "capture" for i in base)
            diff = (n10 - n01) / len(base)
            se = math.sqrt(max(n10 + n01 - (n10 - n01) ** 2 / len(base), 0)) / len(base)
            print(f"  {spec} vs {specs[0]}: diff={diff:+.3f} (wins {n10}, losses {n01}, ~95%CI ±{1.96*se:.3f})")
    OUT.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    (OUT / f"eval_{network_name}_{stamp}.json").write_text(json.dumps(
        {"network": network_name, "episodes": episodes, "tag": tag,
         "results": {s: {str(k): v for k, v in r.items()} for s, r in all_results.items()}}
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

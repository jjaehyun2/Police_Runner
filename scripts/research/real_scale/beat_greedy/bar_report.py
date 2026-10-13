"""Verification report for a bar_ga run (run after training; writes <run>/report.json and report.md).

  1 validation maps (the selection maps), same cases for every policy:
      never     warm role policy, never places (= no barricades)
      role      the searched role scorer without its barricade head (ablation)
      learned   the searched role scorer + barricade head
  2 held-out maps, one shot (never used in training or selection): never vs learned
  3 SUMO, daejeon_demo, ring starts, same seeds: no barricades / hand rules (leave,stand-in) / learned
Each block reports success rate by team size, mean capture time, barricades per episode,
outcome mix, and paired wins/losses against "never".

Usage: python bar_report.py <run_dir>
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "sumo"), str(HERE)]
os.environ.setdefault("ME_MAX_STEPS", "450")
os.environ["ME_PLACEMENT"] = "ring"

import numpy as np
import torch

from multi_eval import evaluate_mixed

NS = (3, 4, 5, 6, 8)
VAL = {"daejeon_validation": 600, "incheon_validation": 300, "suwon2_validation": 300}
TEST = {"daejeon_test": 600, "seoul_zero_shot": 400, "busan_zero_shot": 400}
SUMO_NS = {3: 60, 4: 100, 6: 60}
if os.environ.get("BG_SMOKE"):
    VAL = {k: v // 20 for k, v in VAL.items()}
    TEST = {k: v // 20 for k, v in TEST.items()}
    SUMO_NS = {4: 4}


def preload(*maps) -> None:
    """Load the networks (and their graph caches) once, so forked workers inherit them."""
    import common
    import multi_sim
    from obs_ext import graph_for
    for group in maps:
        for m in group:
            net = common.load_network(m)
            multi_sim.graph_of(net)
            graph_for(net)


def block(results: dict) -> dict:
    """results: policy -> {i: (n, outcome, steps, barricades)} on the same cases."""
    base = results.get("never")
    out = {}
    for name, res in results.items():
        vals = list(res.values())
        ok = [v for v in vals if v[1] == "success"]
        row = dict(SR=round(float(np.mean([v[1] == "success" for v in vals])), 4),
                   capture_s=round(float(np.mean([v[2] for v in ok])), 1) if ok else None,
                   barricades=round(float(np.mean([v[3] for v in vals])), 2),
                   outcomes={o: sum(v[1] == o for v in vals) for o in ("success", "escape", "timeout")},
                   by_n={str(n): round(float(np.mean([v[1] == "success" for v in vals if v[0] == n])), 3)
                         for n in sorted({v[0] for v in vals})})
        if base is not None and name != "never":
            row["paired_wins"] = sum(base[i][1] != "success" and res[i][1] == "success" for i in res)
            row["paired_losses"] = sum(base[i][1] == "success" and res[i][1] != "success" for i in res)
        out[name] = row
    return out


def maps_block(specs: dict, maps: dict, tag: str) -> dict:
    out = {}
    for m, eps in maps.items():
        out[m] = block({name: evaluate_mixed(spec, m, f"{tag}-{m}", eps, NS, detail=True) for name, spec in specs.items()})
    return out


def _sumo_one(args):
    spec, seed, n, rules = args
    os.environ["SUMO_BARRICADE_RULES"] = rules or "leave,stand-in"
    torch.set_num_threads(1)
    import common, multi_sim, sumo_sim
    sumo_sim.BARRICADE_RULES = set(os.environ["SUMO_BARRICADE_RULES"].split(","))
    net = common.load_network("daejeon_demo")
    case = multi_sim.make_ring_case(net, "carry", seed, n)
    try:
        sim = sumo_sim.SumoPursuit("daejeon_demo", case, spec, carry_barricades=rules is not None)
        r = sim.run()
        return n, r.outcome, r.seconds, len(sim.deployments)
    except Exception as e:                                       # noqa: BLE001
        return n, "error:" + type(e).__name__, 0.0, 0


def sumo_block(specs: dict) -> dict:
    from multiprocessing import get_context
    out = {}
    for n, eps in SUMO_NS.items():
        res = {}
        with get_context("fork").Pool(int(os.environ.get("ME_WORKERS", "32"))) as p:
            for name, (spec, rules) in specs.items():
                rows = p.map(_sumo_one, [(spec, s, n, rules) for s in range(eps)], chunksize=1)
                res[name] = {i: r for i, r in enumerate(rows)}
        out[str(n)] = block(res)
    return out


def markdown(rep: dict) -> str:
    lines = [f"# Barricade GA report ({rep['run']})", "", f"generated {rep['time']}", ""]

    def table(title, blk):
        lines.append(f"## {title}")
        for key, rows in blk.items():
            lines.extend(["", f"**{key}**", "", "| policy | SR | capture s | barricades/ep | W/L vs never | escape/timeout | by N |",
                          "|---|---|---|---|---|---|---|"])
            for name, r in rows.items():
                wl = f"{r.get('paired_wins', '–')}/{r.get('paired_losses', '–')}"
                lines.append(f"| {name} | {100 * r['SR']:.1f} | {r['capture_s']} | {r['barricades']} | {wl} | "
                             f"{r['outcomes']['escape']}/{r['outcomes']['timeout']} | {r['by_n']} |")
        lines.append("")

    table("Validation maps (selection)", rep["validation"])
    table("Held-out maps (one shot)", rep["test"])
    table("SUMO daejeon_demo (team size)", rep["sumo"])
    return "\n".join(lines)


def main() -> int:
    run = Path(sys.argv[1]).resolve()
    best = run / "best.pt"
    never = run / "pop" / "never.pt"
    role_only = run / "role_only.pt"
    p = torch.load(best, weights_only=False)
    torch.save({k: v for k, v in p.items() if not k.startswith("barricade")}, role_only)
    preload(VAL, TEST, ["daejeon_demo"])
    if len(sys.argv) > 2 and sys.argv[2] == "--md":             # re-render report.md from report.json
        (run / "report.md").write_text(markdown(json.loads((run / "report.json").read_text())))
        return 0
    rep = dict(run=str(run), time=time.strftime("%Y-%m-%d %H:%M"), meta=p.get("meta"))
    rep["validation"] = maps_block({"never": f"bmacro:{never}", "role": f"bmacro:{role_only}",
                                    "learned": f"bmacro:{best}"}, VAL, "rep-val")
    rep["test"] = maps_block({"never": f"bmacro:{never}", "learned": f"bmacro:{best}"}, TEST, "rep-test")
    rep["sumo"] = sumo_block({"never": (f"bmacro:{never}", None), "role": (f"bmacro:{role_only}", None),
                              "rules": (f"bmacro:{role_only}", "leave,stand-in"), "learned": (f"bmacro:{best}", None)})
    (run / "report.json").write_text(json.dumps(rep, indent=1))
    (run / "report.md").write_text(markdown(rep))
    print(markdown(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

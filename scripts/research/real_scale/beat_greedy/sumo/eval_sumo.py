"""Capture rates of team policies in SUMO vs the point-mass simulator on the same cases.

Usage: python sumo/eval_sumo.py <map> <episodes> <n_police> <profile> <spec...>
Env: SE_WORKERS (16), ME_PLACEMENT (ring | random), ME_MAX_STEPS (600),
     SE_TRAFFIC (background trip period s; unset = empty roads), SE_IGNORE_RED (1 = emergency driving for both, p = police only, f = fugitive only)
     SE_SPEED_FACTOR: pursuit vehicles' multiple of each road's limit (default 2.0 on "real").
     Values >=3 use direct vp/vf commands with SUMO safety and acceleration/deceleration checks retained;
     this removes the road-limit cap without the pathological factor-3 lane-planning slowdown.
"""
from __future__ import annotations

import json
import os
from multiprocessing import get_context
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("ME_PLACEMENT", "ring")
os.environ.setdefault("ME_MAX_STEPS", "600")
# the policies' fugitive-speed belief (common.FUGITIVE_SPEED_MPS) must equal the case's (multi_eval ME_VF);
# common is imported first below, so set it here (before this, common fell back to its 9.0 default)
os.environ.setdefault("BG_FUGITIVE_SPEED", os.environ.get("ME_VF", "16"))
os.environ.setdefault("BG_POLICE_SPEED", os.environ.get("ME_VP", "16"))   # same for the officers' speed (ME_VP)

import numpy as np  # noqa: E402

import common  # noqa: E402
import multi_eval as me  # noqa: E402
import multi_sim  # noqa: E402

_W: dict = {}


def _init(map_name):
    import torch
    torch.set_num_threads(1)
    _W["map"], _W["net"] = map_name, common.load_network(map_name)


def _one(args):
    from sumo_sim import SumoPursuit
    idx, n, spec, profile = args
    net = _W["net"]
    case = me.make_case(net, f"sumo-{_W['map']}", idx, n, 1)
    pm = multi_sim.MultiPursuitSim(net, case).run(me.team_for(spec, net, n))
    try:
        tr = os.environ.get("SE_TRAFFIC")
        sp = SumoPursuit(_W["map"], case, spec, profile=profile, traffic=float(tr) if tr else None,
                         ignore_red={"1": True, "p": "police", "f": "fugitive"}.get(os.environ.get("SE_IGNORE_RED", "0"), False),
                         speed_factor=float(os.environ["SE_SPEED_FACTOR"]) if os.environ.get("SE_SPEED_FACTOR") else None)
        r = sp.run()
        so, st = r.outcome, r.seconds
        if os.environ.get("SE_EDGELOG"):                       # calibration: edge traversals of this episode
            d = Path(os.environ["SE_EDGELOG"]); d.mkdir(parents=True, exist_ok=True)
            (d / f"{idx}.json").write_text(json.dumps(sp.edge_log))
    except Exception as e:                                    # noqa: BLE001
        so, st = f"error:{type(e).__name__}:{str(e)[:80]}", 0.0
    return idx, so, st, pm.outcome, pm.steps


def main() -> int:
    m, eps, n, profile, specs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5:]
    from build_net import build
    if not (HERE / "nets" / f"{m}_{profile}.net.xml").exists():
        build(m, profile)
    out = {}
    for spec in specs:
        t = time.time()
        with get_context("fork").Pool(int(os.environ.get("SE_WORKERS", "16")), initializer=_init, initargs=(m,)) as p:
            off = int(os.environ.get("SE_OFFSET", "0"))           # case index offset: fresh cases for confirmation runs
            res = p.map(_one, [(i, n, spec, profile) for i in range(off, off + eps)], chunksize=1)
        out[spec] = res
        so = [r[1] for r in res]; pm = [r[3] for r in res]
        err = [s for s in so if s.startswith("error")]
        ok = [r[2] for r in res if r[1] == "success"]
        print(f"{m} {profile} traffic={os.environ.get('SE_TRAFFIC', 'none')} ignore_red={os.environ.get('SE_IGNORE_RED', '0')} P{n} {spec[-28:]:28s} SUMO SR={np.mean([s == 'success' for s in so]):.3f} "
              f"ATS={np.mean(ok) if ok else float('nan'):.1f}s esc={np.mean([s == 'escape' for s in so]):.3f} "
              f"err={len(err)} | point-mass SR={np.mean([s == 'success' for s in pm]):.3f} "
              f"agree={np.mean([(a == 'success') == (b == 'success') for a, b in zip(so, pm)]):.2f} ({time.time()-t:.0f}s)",
              flush=True)
        if err:
            print("   e.g.", err[:2], flush=True)
    res_dir = HERE / "results"; res_dir.mkdir(exist_ok=True)
    (res_dir / f"sumo_{m}_{profile}_tr{os.environ.get('SE_TRAFFIC', 'none')}_red{os.environ.get('SE_IGNORE_RED', '0')}_P{n}_{os.environ['ME_PLACEMENT']}{os.environ.get('ME_RING', '').replace(',', '-')}{'_off' + os.environ['SE_OFFSET'] if os.environ.get('SE_OFFSET') else ''}_{time.strftime('%Y%m%dT%H%M%S')}.json").write_text(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

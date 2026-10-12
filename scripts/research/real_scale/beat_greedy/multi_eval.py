"""Evaluate multi-fugitive team policies on sealed-by-seed multi cases.

Usage: python multi_eval.py <network> <episodes> <n_police> <n_fugitives> <policy...>
Env: ME_WORKERS (default 40), ME_VF (16), ME_MAX_STEPS (600)
Metrics follow PEPCRL-MVP: SR (all fugitives captured), mean captured fraction,
ATS (steps of successful episodes) and their standard deviations.
"""
from __future__ import annotations
import json, os, sys, time
from multiprocessing import get_context
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
VF = float(os.environ.get("ME_VF", "16"))
os.environ.setdefault("BG_FUGITIVE_SPEED", str(VF))     # single-fugitive specs read it via common
import numpy as np
import common, multi_policies, multi_sim
from pursuit_evasion_rl.research.policies import baselines

MAX_STEPS = int(os.environ.get("ME_MAX_STEPS", "600"))
_W = {}


PLACEMENT = os.environ.get("ME_PLACEMENT", "random")     # random | ring (fugitive in the middle, officers surrounding)


def make_case(net, tag, idx, n_p, n_f):
    if PLACEMENT == "ring":
        assert n_f == 1, "ring placement is single-fugitive"
        return multi_sim.make_ring_case(net, tag, idx, n_p, vf=VF, max_steps=MAX_STEPS)
    return multi_sim.make_multi_case(net, tag, idx, n_p, n_f, vf=VF, max_steps=MAX_STEPS)


def _init(net_name):
    import torch
    torch.set_num_threads(1)
    _W["net"] = common.load_network(net_name)


def _run(args):
    tag, idx, n_p, n_f, pol = args
    net = _W["net"]
    case = make_case(net, tag, idx, n_p, n_f)
    r = multi_sim.MultiPursuitSim(net, case).run(team_for(pol, net, n_p))
    return idx, r.outcome, r.steps, r.captured


class AnySizeEncirclement(baselines.EncirclementPolice):
    """EncirclementPolice whose diagnostic goal-assignment record (validated for exactly 6
    officers) is skipped, so the same targeting runs with any team size."""

    def _assign(self, *args):
        saved = baselines.GoalAssignment, baselines.GoalAssignmentPlan
        baselines.GoalAssignment = baselines.GoalAssignmentPlan = lambda *a, **k: None
        try:
            return super()._assign(*args)
        finally:
            baselines.GoalAssignment, baselines.GoalAssignmentPlan = saved


def team_for(spec, net, n_police):
    """Multi-fugitive policy by name, or any common._build single-fugitive spec (one fugitive,
    any team size) run through RecommendBridge."""
    if spec in multi_policies.POLICIES:
        return multi_policies.POLICIES[spec](net)
    pol = common._build(spec, net)["team_policy"]
    if hasattr(pol, "pc"):                                   # EncirclementPolice
        pol.__class__ = AnySizeEncirclement
        pol.pc, pol._prev = n_police, {k: None for k in range(n_police)}
    if hasattr(pol, "police_count"):                         # GreedyInterceptPolice
        pol.police_count = n_police
    return multi_sim.RecommendBridge(pol)


def evaluate(net_name, episodes, n_p, n_f, pol, workers=int(os.environ.get("ME_WORKERS", "40"))):
    with get_context("fork").Pool(workers, initializer=_init, initargs=(net_name,)) as p:
        res = p.map(_run, [(f"multi-{net_name}", i, n_p, n_f, pol) for i in range(episodes)], chunksize=1)
    return {i: (o, s, c) for i, o, s, c in res}


def _run_mixed(args):
    tag, idx, ns, spec = args
    n = ns[idx % len(ns)]
    net = _W["net"]
    case = make_case(net, tag, idx, n, 1)
    r = multi_sim.MultiPursuitSim(net, case).run(team_for(spec, net, n))
    return idx, n, r.outcome, r.steps


def evaluate_mixed(spec, net_name, tag, episodes, ns, workers=int(os.environ.get("ME_WORKERS", "40"))):
    """One fugitive; episode i uses ns[i % len(ns)] officers. -> {i: (n, outcome, steps)}."""
    with get_context("fork").Pool(workers, initializer=_init, initargs=(net_name,)) as p:
        res = p.map(_run_mixed, [(tag, i, tuple(ns), spec) for i in range(episodes)], chunksize=1)
    return {i: (n, o, s) for i, n, o, s in res}


def summary(res, n_f):
    o = [v[0] for v in res.values()]
    succ = [v[1] for v in res.values() if v[0] == "success"]
    frac = [v[2] / n_f for v in res.values()]
    return dict(SR=np.mean([x == "success" for x in o]), escape=np.mean([x == "escape" for x in o]),
                timeout=np.mean([x == "timeout" for x in o]), capt_frac=np.mean(frac), capt_frac_sd=np.std(frac),
                ATS=np.mean(succ) if succ else float("nan"), ATS_sd=np.std(succ) if succ else float("nan"))


if __name__ == "__main__":
    net, eps, n_p, n_f = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    allr = {}
    for pol in sys.argv[5:]:
        t = time.time()
        allr[pol] = res = evaluate(net, eps, n_p, n_f, pol)
        s = summary(res, n_f)
        print(f"{net} P{n_p}-E{n_f} {pol:28s} " + " ".join(f"{k}={v:.3f}" if k[:3] != "ATS" else f"{k}={v:.1f}" for k, v in s.items())
              + f" ({time.time()-t:.0f}s)", flush=True)
    out = Path(__file__).resolve().parent / "results" / f"multi_{net}_{PLACEMENT}_P{n_p}E{n_f}_{time.strftime('%Y%m%dT%H%M%S')}.json"
    out.write_text(json.dumps({p: {str(k): v for k, v in r.items()} for p, r in allr.items()}))

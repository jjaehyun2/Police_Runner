"""Behavior-clone a macro rule (default: ``assign``) into the role scorer.

Rollouts are driven by the rule itself on solvable training cases; every
role decision's candidate rows/mask and the rule's chosen candidate are
recorded and the scorer is trained with masked cross-entropy.

Usage: python bc_macro.py <out.pt> [episodes] [rule]
"""
from __future__ import annotations

import json
import os
from multiprocessing import get_context
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("BG_SOLVABLE", "1")

import numpy as np
import torch

import common
import macro

SPEEDS = (12.0, 14.0, 16.0, 16.0, 16.0)
HIDDEN = (128, 128)


def _job(args):
    idx, vf, rule = args
    torch.set_num_threads(1)
    common.SOLVABLE_ONLY = True
    net = common.load_network("daejeon_train")
    common.POLICE_SPEED_MPS, common.FUGITIVE_SPEED_MPS = 16.0, vf
    case = common.make_case(net, "bcmacro", idx)
    pol = macro.MacroTeamPolicy(net, None, vp=16.0, vf=vf, fixed_rule=rule)
    rec = common.replay_episode(case, network=net, team_policy=pol)
    return rec.outcome.value, [(d["rows"], d["officer"], d["mask"], d["action"]) for d in pol.log]


def _eval(args):
    idx, state = args
    torch.set_num_threads(1)
    common.SOLVABLE_ONLY = True
    net = common.load_network("daejeon_validation")
    common.POLICE_SPEED_MPS, common.FUGITIVE_SPEED_MPS = 16.0, 16.0
    scorer = macro.CandidateScorer(HIDDEN)
    scorer.load_state_dict(state)
    case = common.make_case(net, "beatgreedy-daejeon_validation", idx)
    pol = macro.MacroTeamPolicy(net, scorer, vp=16.0, vf=16.0)
    return common.replay_episode(case, network=net, team_policy=pol).outcome.value


def main() -> int:
    out = Path(sys.argv[1])
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    rule = sys.argv[3] if len(sys.argv) > 3 else "assign"
    rng = np.random.default_rng(0)
    t0 = time.time()
    with get_context("fork").Pool(24) as pool:
        res = pool.map(_job, [(i, float(rng.choice(SPEEDS)), rule) for i in range(episodes)], chunksize=4)
    print("rule capture on train", np.mean([o == "capture" for o, _ in res]), "sec", round(time.time() - t0), flush=True)
    data = [d for _, lg in res for d in lg]
    R = torch.from_numpy(np.stack([d[0] for d in data])); I = torch.tensor([d[1] for d in data])
    M = torch.from_numpy(np.stack([d[2] for d in data])); A = torch.tensor([d[3] for d in data])
    torch.manual_seed(0)
    torch.set_num_threads(16)
    scorer = macro.CandidateScorer(HIDDEN)
    opt = torch.optim.Adam(scorer.parameters(), lr=1e-3)
    for epoch in range(30):
        perm = torch.randperm(len(R))
        correct = 0
        for s in range(0, len(R), 2048):
            b = perm[s:s + 2048]
            logits = scorer(R[b], I[b]).masked_fill(~M[b], -1e9)
            loss = torch.nn.functional.cross_entropy(logits, A[b])
            opt.zero_grad(); loss.backward(); opt.step()
            correct += int((logits.argmax(-1) == A[b]).sum())
    acc = correct / len(R)
    with get_context("fork").Pool(24) as pool:
        outs = pool.map(_eval, [(i, scorer.state_dict()) for i in range(400)], chunksize=4)
    val = float(np.mean([o == "capture" for o in outs]))
    print(json.dumps(dict(samples=len(R), train_acc=acc, val16=val)), flush=True)
    torch.save({"kind": "macro", "hidden": HIDDEN, "scorer": scorer.state_dict(),
                "meta": dict(stage="bc", rule=rule, train_acc=acc, val16=val)}, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

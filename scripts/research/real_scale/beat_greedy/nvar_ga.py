"""Team-size-agnostic role policy search (one fugitive, variable number of officers).

The scorer sees the candidate row plus the team-size feature 6/N instead of the officer
one-hot (CandidateScorer kind="size"), so one network drives any team size. Cases come from
multi_sim (one fugitive, N officers); episode i of a map uses NS[i % len(NS)] officers.

  fitness    mean capture over the TRAIN maps, fresh cases every generation
  selection  mean capture over the VALIDATION maps -> best.pt
             (Daejeon-test / Seoul / Busan are never touched here)

Generation 0: the ms_ga winner converted to kind="size" (officer one-hot columns averaged
into the size column, exact at N=6 up to the id term), mutants of it, and random scorers.
Later generations: elites + Gaussian mutation + unit crossover, as ga.py.

Usage: python nvar_ga.py <out_dir> <generations>
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("ME_MAX_STEPS", "450")

import numpy as np
import torch

import macro
from ga import HIDDEN, crossover, mutate
from multi_eval import evaluate_mixed

NS = (3, 4, 5, 6, 8, 10)
TRAIN = {"daejeon_train": 360, "daegu_train": 240, "ulsan_train": 240}
VAL = {"daejeon_validation": 600, "incheon_validation": 300, "suwon2_validation": 300}
POP = int(os.environ.get("NV_POP", "16"))
ELITE = int(os.environ.get("NV_ELITE", "5"))
N_RANDOM = int(os.environ.get("NV_RANDOM", "24"))
WARM = os.environ.get("NV_WARM", "runs/ms_ga/best.pt")
VAL_TOP = 3


def save(state: dict, path: Path, meta: dict) -> None:
    torch.save({"kind": "macro", "scorer_kind": "size", "hidden": HIDDEN, "scorer": state, "meta": meta}, path)


def load(path: Path) -> dict:
    return torch.load(path, weights_only=False)["scorer"]


def from_onehot(state: dict) -> dict:
    """onehot scorer -> size scorer: the 6 officer columns collapse into their mean (input 6/N = 1 at N=6)."""
    out = dict(state)
    w = state["net.0.weight"]
    out["net.0.weight"] = torch.cat([w[:, :macro.ROW_DIM], w[:, macro.ROW_DIM:].mean(1, keepdim=True)], 1)
    return out


def random_state(seed: int) -> dict:
    torch.manual_seed(seed)
    return macro.CandidateScorer(HIDDEN, kind="size").state_dict()


def per_n(res: dict) -> dict:
    out = {}
    for n, o, _ in res.values():
        a = out.setdefault(n, [0, 0]); a[0] += o == "success"; a[1] += 1
    return {n: round(a[0] / a[1], 3) for n, a in sorted(out.items())}


def score(path: Path, maps: dict, tag: str) -> tuple[float, dict]:
    per, by_n = {}, {}
    for m, n_eps in maps.items():
        res = evaluate_mixed(f"rmacro:{path}", m, f"{tag}-{m}", n_eps, NS)
        per[m] = float(np.mean([o == "success" for _, o, _ in res.values()]))
        for n, v in per_n(res).items():
            by_n.setdefault(n, []).append(v)
    return float(np.mean(list(per.values()))), dict(maps={k: round(v, 3) for k, v in per.items()},
                                                    by_n={n: round(float(np.mean(v)), 3) for n, v in by_n.items()})


def main() -> int:
    out, gens = Path(sys.argv[1]), int(sys.argv[2])
    pop_dir = out / "pop"
    pop_dir.mkdir(parents=True, exist_ok=True)
    log = (out / "nvar_ga.jsonl").open("a")
    rng = np.random.default_rng(0)

    def emit(rec):
        log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)

    # generation 0 candidates
    cands = []
    warm = from_onehot(load(Path(WARM)))
    p = pop_dir / "warm.pt"; save(warm, p, dict(origin=f"from_onehot({WARM})")); cands.append(p)
    for k in range(8):
        p = pop_dir / f"warm_m{k}.pt"; save(mutate(warm, rng, 0.05 if k % 2 == 0 else 0.15), p, dict(origin="m(warm)"))
        cands.append(p)
    for s in range(N_RANDOM):
        p = pop_dir / f"seed{s}.pt"; save(random_state(s), p, dict(origin=f"seed{s}")); cands.append(p)
    screened = []
    for p in cands:
        f, info = score(p, TRAIN, "nv-screen")
        screened.append((f, p))
        emit(dict(stage="screen", name=p.name, fit=round(f, 4), **info))
    screened.sort(key=lambda x: -x[0])
    pop = [p for _, p in screened[:POP]]
    best_val, best_name = -1.0, None
    for g in range(gens + 1):
        t0 = time.time()
        fits = sorted(((score(p, TRAIN, f"nv-gen{g}")[0], p) for p in pop), key=lambda x: -x[0])
        elites = [p for _, p in fits[:ELITE]]
        vals = {}
        for _, p in fits[:VAL_TOP]:
            v, info = score(p, VAL, "nv-val")
            vals[p.name] = dict(mean=round(v, 4), **info)
            if v > best_val:
                best_val, best_name = v, p.name
                torch.save(torch.load(p, weights_only=False), out / "best.pt")
        emit(dict(stage="gen", gen=g, fit_top=[round(f, 4) for f, _ in fits[:ELITE]],
                  fit_mean=round(float(np.mean([f for f, _ in fits])), 4), val=vals,
                  best_val=round(best_val, 4), best=best_name, seconds=round(time.time() - t0)))
        if g == gens:
            break
        children, k = list(elites), 0
        while len(children) < POP:
            if k % 4 == 3:
                a, b = rng.choice(len(elites), 2, replace=False)
                st, origin = crossover(load(elites[a]), load(elites[b]), rng), f"x({elites[a].name},{elites[b].name})"
            else:
                a = int(rng.integers(len(elites)))
                scale = 0.05 if k % 2 == 0 else 0.15
                st, origin = mutate(load(elites[a]), rng, scale), f"m{scale}({elites[a].name})"
            p = pop_dir / f"g{g + 1}_{k}.pt"
            save(st, p, dict(origin=origin))
            children.append(p)
            k += 1
        pop = children
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

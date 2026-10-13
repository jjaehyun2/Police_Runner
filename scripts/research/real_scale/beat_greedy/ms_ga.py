"""Multi-city search for a scale-free role policy (relative features, rmacro:).

  fitness    mean capture over the TRAIN maps (Daejeon-train, Daegu, Ulsan),
             fresh sealed solvable cases every generation
  selection  mean capture over the VALIDATION maps (Daejeon-val, Incheon, Suwon)
             -> best.pt; Daejeon-test / Seoul / Busan are never touched here

Generation 0 screens N_SEEDS untrained scorers and keeps the POP fittest;
later generations: elites + Gaussian mutation (+ unit crossover), as ga.py.

Usage: python ms_ga.py <out_dir> <generations>
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("BG_SOLVABLE", "1")
os.environ.setdefault("BG_FUGITIVE_SPEED", "16")

import numpy as np
import torch

import macro
from common import capture_rate, evaluate
from ga import HIDDEN, crossover, init_scorer, load, mutate, save

TRAIN = {"daejeon_train": 400, "daegu_train": 200, "ulsan_train": 200}
VAL = {"daejeon_validation": 600, "incheon_validation": 300, "suwon2_validation": 300}
N_SEEDS = int(os.environ.get("MS_SEEDS", "60"))
POP = int(os.environ.get("MS_POP", "16"))
ELITE = int(os.environ.get("MS_ELITE", "5"))
VAL_TOP = 3


def multi(path_or_spec: str, maps: dict, tag: str) -> dict:
    spec = path_or_spec if ":" in path_or_spec else f"rmacro:{path_or_spec}"
    return {m: capture_rate(evaluate(spec, m, f"{tag}-{m}", n)) for m, n in maps.items()}


def main() -> int:
    out, gens = Path(sys.argv[1]), int(sys.argv[2])
    pop_dir = out / "pop"
    pop_dir.mkdir(parents=True, exist_ok=True)
    log = (out / "ms_ga.jsonl").open("a")
    rng = np.random.default_rng(0)

    def emit(rec):
        log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)

    # generation 0: seed screen on the train maps (same cases for every seed)
    screened = []
    for s in range(N_SEEDS):
        p = pop_dir / f"seed{s}.pt"
        save(init_scorer(s), p, dict(origin=f"seed{s}"))
        per = multi(str(p), TRAIN, "ms-screen")
        screened.append((float(np.mean(list(per.values()))), p))
        emit(dict(stage="screen", seed=s, fit=screened[-1][0], per=per))
    screened.sort(key=lambda x: -x[0])
    pop = [p for _, p in screened[:POP]]
    best_val, best_name = -1.0, None
    for g in range(gens + 1):
        t0 = time.time()
        fits = [(float(np.mean(list(multi(str(p), TRAIN, f"ms-gen{g}").values()))), p) for p in pop]
        fits.sort(key=lambda x: -x[0])
        elites = [p for _, p in fits[:ELITE]]
        vals = {}
        for _, p in fits[:VAL_TOP]:
            per = multi(str(p), VAL, "ms-val")
            v = float(np.mean(list(per.values())))
            vals[p.name] = dict(mean=round(v, 4), **{k: round(x, 3) for k, x in per.items()})
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
            sc = macro.CandidateScorer(HIDDEN); sc.load_state_dict(st)
            p = pop_dir / f"g{g + 1}_{k}.pt"
            save(sc, p, dict(origin=origin))
            children.append(p)
            k += 1
        pop = children
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

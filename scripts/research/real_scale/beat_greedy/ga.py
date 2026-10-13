"""Genetic search over role-scorer weights (mutation + truncation selection).

Generation 0 = the best seeds of seed_search.py. Every generation:
  fitness   capture rate on FIT_EPISODES fresh sealed solvable *training* cases
            (new cases every generation, all individuals on the same cases);
  select    the ELITE best survive unchanged;
  offspring Gaussian mutation of an elite, noise scaled per tensor by that
            tensor's std (half small, half large steps) and uniform crossover
            between two elites for a quarter of the children;
  monitor   the generation's top individuals on VAL_EPISODES validation cases;
            the best validation individual is kept as best.pt (test is never used here).

Usage: python ga.py <out_dir> <generations>
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

POP = int(os.environ.get("GA_POP", "24"))
ELITE = int(os.environ.get("GA_ELITE", "6"))
FIT_EPISODES = int(os.environ.get("GA_FIT", "1000"))
VAL_EPISODES = 1000
VAL_TOP = 3
HIDDEN = (128, 128)
SEEDS0 = [int(s) for s in os.environ.get("GA_SEEDS", "42,35,65,0,80,76,32,64,5,6,11,23").split(",")]


def init_scorer(seed: int) -> macro.CandidateScorer:
    torch.manual_seed(seed)
    return macro.CandidateScorer(HIDDEN)


def save(scorer, path: Path, meta: dict) -> None:
    torch.save({"kind": "macro", "hidden": HIDDEN, "scorer": scorer.state_dict(), "meta": meta}, path)


def load(path: Path) -> dict:
    return torch.load(path, weights_only=False)["scorer"]


def mutate(state: dict, rng: np.random.Generator, scale: float) -> dict:
    out = {}
    for k, v in state.items():
        std = float(v.std()) if v.numel() > 1 else 0.1
        out[k] = v + torch.from_numpy(rng.standard_normal(tuple(v.shape)).astype(np.float32)) * scale * max(std, 1e-3)
    return out


def crossover(a: dict, b: dict, rng: np.random.Generator) -> dict:
    # unit-level crossover: each hidden unit (row of a weight matrix + its bias) comes from one parent
    out = {}
    for k in a:
        if a[k].dim() == 2:
            pick = torch.from_numpy(rng.random(a[k].shape[0]) < 0.5)
            out[k] = torch.where(pick[:, None], a[k], b[k])
            out[k.replace("weight", "bias")] = torch.where(pick, a[k.replace("weight", "bias")], b[k.replace("weight", "bias")])
    return out


def fitness(path: Path, tag: str, network: str, episodes: int) -> float:
    return capture_rate(evaluate(f"macro:{path}", network, tag, episodes))


def main() -> int:
    out, gens = Path(sys.argv[1]), int(sys.argv[2])
    pop_dir = out / "pop"
    pop_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    log = (out / "ga.jsonl").open("a")
    # generation 0
    pop = []
    for i, s in enumerate(SEEDS0[:POP]):
        p = pop_dir / f"g0_{i}.pt"
        save(init_scorer(s), p, dict(origin=f"seed{s}"))
        pop.append(p)
    best_val, best_path = -1.0, None
    for g in range(gens + 1):
        t0 = time.time()
        fit = [fitness(p, f"ga-gen{g}", "daejeon_train", FIT_EPISODES) for p in pop]
        order = np.argsort(fit)[::-1]
        elites = [pop[i] for i in order[:ELITE]]
        vals = {}
        for i in order[:VAL_TOP]:
            vals[pop[i].name] = fitness(pop[i], "beatgreedy-daejeon_validation", "daejeon_validation", VAL_EPISODES)
            if vals[pop[i].name] > best_val:
                best_val, best_path = vals[pop[i].name], pop[i]
                torch.save(torch.load(pop[i], weights_only=False), out / "best.pt")
        rec = dict(gen=g, fit_max=float(max(fit)), fit_mean=float(np.mean(fit)), fit_elite=[round(fit[i], 3) for i in order[:ELITE]],
                   val_top=vals, best_val=best_val, best=str(best_path.name), seconds=round(time.time() - t0))
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(json.dumps(rec), flush=True)
        if g == gens:
            break
        # next generation
        children = list(elites)
        k = 0
        while len(children) < POP:
            if k % 4 == 3:
                a, b = rng.choice(len(elites), 2, replace=False)
                st = crossover(load(elites[a]), load(elites[b]), rng)
                origin = f"x({elites[a].name},{elites[b].name})"
            else:
                a = int(rng.integers(len(elites)))
                scale = 0.05 if k % 2 == 0 else 0.15
                st = mutate(load(elites[a]), rng, scale)
                origin = f"m{scale}({elites[a].name})"
            sc = macro.CandidateScorer(HIDDEN); sc.load_state_dict(st)
            p = pop_dir / f"g{g + 1}_{k}.pt"
            save(sc, p, dict(origin=origin))
            children.append(p)
            k += 1
        pop = children
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

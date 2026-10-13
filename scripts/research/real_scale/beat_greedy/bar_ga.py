"""Joint search: role policy + barricade placement (every officer carries one barricade).

Genome = the team-size-agnostic role scorer (macro.CandidateScorer, kind="size") + the barricade
head (barricade.BarricadeHead). Both are evolved together: the role policy plans on the map with
placed barricades removed, so it can learn to lean on them, and the head learns where they pay.

  scenario   ring starts (fugitive in the middle, officers surrounding it), one fugitive,
             team sizes NS, every officer carrying one barricade
  fitness    mean capture over the TRAIN maps, fresh cases every generation
  selection  mean capture over the VALIDATION maps -> best.pt
             (daejeon_test / seoul_zero_shot / busan_zero_shot are never touched here)

Generation 0: the warm role scorer with (a) a head that never places (= the warm policy without
barricades), (b) a hand-set head that places when a road shuts an exit or keeps the region small
once the officer leaves, (c) random heads with different eagerness, and mutants of these.
Later generations: elites + mutation of the head, the role scorer or both + unit crossover.

Usage: python bar_ga.py <out_dir> <generations>
Env:   BG_POP (24) BG_ELITE (6) BG_WARM (runs/nvar_ga/best.pt) ME_WORKERS
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("ME_MAX_STEPS", "450")
os.environ["ME_PLACEMENT"] = "ring"

import numpy as np
import torch

import barricade
from ga import HIDDEN, crossover, mutate
from multi_eval import evaluate_mixed

NS = (3, 4, 5, 6, 8)
TRAIN = {"daejeon_train": 300, "daegu_train": 200, "ulsan_train": 200}
VAL = {"daejeon_validation": 400, "incheon_validation": 200, "suwon2_validation": 200}
if os.environ.get("BG_SMOKE"):                             # tiny run to check the plumbing
    TRAIN = {k: v // 20 for k, v in TRAIN.items()}
    VAL = {k: v // 20 for k, v in VAL.items()}
POP = int(os.environ.get("BG_POP", "24"))
ELITE = int(os.environ.get("BG_ELITE", "6"))
WARM = os.environ.get("BG_WARM", "runs/nvar_ga/best.pt")
VAL_TOP = 3
# v1 role steps (0.03, 0.1) overfit the training maps: role-only -3/-6 pt on unseen maps, -15 pt in SUMO N=3
ROLE_STEP = tuple(float(x) for x in os.environ.get("BG_ROLE_STEP", "0.01,0.03").split(","))
FINAL_K = int(os.environ.get("BG_FINAL_K", "6"))
SUMO_FINAL = {3: 60, 4: 60}
if os.environ.get("BG_SMOKE"):
    SUMO_FINAL = {4: 4}
HEAD_HIDDEN = (32, 32)


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


def save(scorer: dict, head: dict, path: Path, meta: dict) -> None:
    torch.save({"kind": "macro", "scorer_kind": "size", "hidden": HIDDEN, "scorer": scorer,
                "barricade": head, "barricade_hidden": HEAD_HIDDEN, "meta": meta}, path)


def load(path: Path) -> tuple[dict, dict]:
    p = torch.load(path, weights_only=False)
    return p["scorer"], p["barricade"]


def head_state(seed: int, bias: float) -> dict:
    torch.manual_seed(seed)
    h = barricade.BarricadeHead(HEAD_HIDDEN)
    with torch.no_grad():
        h.net[-1].bias.fill_(bias)
    return h.state_dict()


def rule_head() -> dict:
    """Hand-set start: place when the road shuts an open exit or (officer holding) keeps the region
    from growing once the officer leaves; otherwise not."""
    h = barricade.BarricadeHead(HEAD_HIDDEN)
    f = {name: i for i, name in enumerate(barricade.FEATURES)}
    with torch.no_grad():
        for layer in h.net:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.zero_(); layer.bias.zero_()
        h.net[0].weight[0, f["open_cut_stay"]] = 6.0
        h.net[0].weight[0, f["open_cut_leave"]] = 4.0
        h.net[0].weight[0, f["region_cut_leave"]] = 4.0
        h.net[2].weight[0, 0] = 2.0
        h.net[4].weight[0, 0] = 2.0
        h.net[4].bias.fill_(-0.6)
    return h.state_dict()


def score(path: Path, maps: dict, tag: str) -> tuple[float, dict]:
    per, by_n, bars = {}, {}, []
    for m, n_eps in maps.items():
        res = evaluate_mixed(f"bmacro:{path}", m, f"{tag}-{m}", n_eps, NS, detail=True)
        per[m] = float(np.mean([v[1] == "success" for v in res.values()]))
        bars += [v[3] for v in res.values()]
        for n in NS:
            vs = [v[1] == "success" for v in res.values() if v[0] == n]
            by_n.setdefault(n, []).append(float(np.mean(vs)))
    return float(np.mean(list(per.values()))), dict(
        maps={k: round(v, 3) for k, v in per.items()}, by_n={n: round(float(np.mean(v)), 3) for n, v in by_n.items()},
        barricades=round(float(np.mean(bars)), 2))


def main() -> int:
    out, gens = Path(sys.argv[1]), int(sys.argv[2])
    pop_dir = out / "pop"
    pop_dir.mkdir(parents=True, exist_ok=True)
    log = (out / "bar_ga.jsonl").open("a")
    rng = np.random.default_rng(0)

    def emit(rec):
        log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)

    preload(TRAIN, VAL)
    warm = torch.load(WARM, weights_only=False)["scorer"]
    cands = []

    def add(name, scorer, head, origin):
        p = pop_dir / f"{name}.pt"; save(scorer, head, p, dict(origin=origin)); cands.append(p)

    add("never", warm, head_state(0, -5.0), "warm + never place")
    add("rule", warm, rule_head(), "warm + hand-set head")
    for s, bias in enumerate((-1.0, -0.5, -0.2, 0.0, 0.3, -1.0, -0.5, -0.2, 0.0, 0.3)):
        add(f"rand{s}", warm, head_state(s + 1, bias), f"warm + random head (bias {bias})")
    for k in range(6):
        base = rule_head() if k % 2 == 0 else head_state(k + 20, -0.3)
        add(f"rule_m{k}", warm, mutate(base, rng, 0.2 if k < 3 else 0.5), "warm + mutated head")
    screened = []
    for p in cands:
        f, info = score(p, TRAIN, "bg-screen")
        screened.append((f, p))
        emit(dict(stage="screen", name=p.name, fit=round(f, 4), **info))
    screened.sort(key=lambda x: -x[0])
    pop = [p for _, p in screened[:POP]]
    base_val, base_info = score(pop_dir / "never.pt", VAL, "bg-val")
    emit(dict(stage="baseline", name="never.pt", val=round(base_val, 4), **base_info))
    best_val, best_name = -1.0, None
    history: dict = {}
    for g in range(gens + 1):
        t0 = time.time()
        fits = sorted(((score(p, TRAIN, f"bg-gen{g}")[0], p) for p in pop), key=lambda x: -x[0])
        elites = [p for _, p in fits[:ELITE]]
        vals = {}
        for _, p in fits[:VAL_TOP]:
            v, info = score(p, VAL, "bg-val")
            vals[p.name] = dict(mean=round(v, 4), **info)
            history[p] = max(history.get(p, -1.0), v)
            if v > best_val:
                best_val, best_name = v, p.name
                torch.save(torch.load(p, weights_only=False), out / "best.pt")
        emit(dict(stage="gen", gen=g, fit_top=[round(f, 4) for f, _ in fits[:ELITE]],
                  fit_mean=round(float(np.mean([f for f, _ in fits])), 4), val=vals,
                  best_val=round(best_val, 4), best=best_name, baseline_val=round(base_val, 4),
                  seconds=round(time.time() - t0)))
        if g == gens:
            break
        children, k = list(elites), 0
        while len(children) < POP:
            if k % 5 == 4:
                a, b = rng.choice(len(elites), 2, replace=False)
                (sa, ha), (sb, hb) = load(elites[a]), load(elites[b])
                sc, hd = crossover(sa, sb, rng), crossover(ha, hb, rng)
                origin = f"x({elites[a].name},{elites[b].name})"
            else:
                a = int(rng.integers(len(elites)))
                sc, hd = load(elites[a])
                what = ("head", "role", "both", "head")[k % 4]
                hs, rs = (0.1, ROLE_STEP[0]) if k % 2 == 0 else (0.3, ROLE_STEP[1])
                if what in ("head", "both"):
                    hd = mutate(hd, rng, hs)
                if what in ("role", "both"):
                    sc = mutate(sc, rng, rs)
                origin = f"m-{what}({elites[a].name})"
            p = pop_dir / f"g{g + 1}_{k}.pt"
            save(sc, hd, p, dict(origin=origin))
            children.append(p)
            k += 1
        pop = children
    final_select(out, history, emit)
    return 0


def sumo_score(path: Path) -> float:
    """Success rate in SUMO (daejeon_demo, ring starts; not a held-out map) over SUMO_FINAL."""
    from multiprocessing import get_context
    import bar_report
    rows = []
    with get_context("fork").Pool(int(os.environ.get("ME_WORKERS", "32"))) as p:
        for n, eps in SUMO_FINAL.items():
            rows += p.map(bar_report._sumo_one, [(f"bmacro:{path}", s, n, None) for s in range(1000, 1000 + eps)],
                          chunksize=1)
    return float(np.mean([r[1] == "success" for r in rows]))


def final_select(out: Path, history: dict, emit) -> None:
    """The per-generation best is the max of many noisy validation scores (winner's curse: v1 picked
    0.935 vs 0.9125 for never, but fresh cases gave never 91.5 / learned 92.5). Re-score the FINAL_K
    best and "never" on fresh validation cases (x2) and in SUMO; the best mean of the two wins,
    "never" included, so "use no barricades" stays a possible answer."""
    pool = [p for p, _ in sorted(history.items(), key=lambda kv: -kv[1])[:FINAL_K]]
    pool += [] if out / "pop" / "never.pt" in pool else [out / "pop" / "never.pt"]
    fresh = {k: 2 * v for k, v in VAL.items()}
    best = None
    for p in pool:
        v, info = score(p, fresh, "bg-final")
        su = sumo_score(p)
        total = 0.5 * v + 0.5 * su
        emit(dict(stage="final", name=p.name, val_fresh=round(v, 4), sumo=round(su, 4), total=round(total, 4), **info))
        if best is None or total > best[0]:
            best = (total, p)
    torch.save(torch.load(best[1], weights_only=False), out / "best.pt")
    emit(dict(stage="chosen", name=best[1].name, total=round(best[0], 4)))


if __name__ == "__main__":
    raise SystemExit(main())

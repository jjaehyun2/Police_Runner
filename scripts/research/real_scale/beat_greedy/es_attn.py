"""Evolution strategies on the attention target-assignment policy.

Optimizes the *deterministic* (argmax) team policy that is actually deployed
and evaluated, instead of a sampled surrogate: every iteration evaluates
POP antithetic parameter perturbations theta +- sigma*eps on the same CASES
sealed solvable training cases (common random numbers), rank-normalizes the
mean outcome scores and steps along the estimated gradient (OpenAI-ES, Adam).

Usage: python es_attn.py <out_dir> <iters>
Env: ES_INIT (warm start .pt), ES_POP (pairs, 24), ES_CASES (64), ES_SIGMA (0.02),
     ES_LR (0.01), ES_WORKERS (46), ES_EVAL (1000)
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
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import attn
import common
from train_attn import SPEEDS, score

POP = int(os.environ.get("ES_POP", "24"))
CASES = int(os.environ.get("ES_CASES", "64"))
SIGMA = float(os.environ.get("ES_SIGMA", "0.02"))
LR = float(os.environ.get("ES_LR", "0.01"))
WD = 0.005
WORKERS = int(os.environ.get("ES_WORKERS", "46"))
EVAL_EPISODES = int(os.environ.get("ES_EVAL", "1000"))
EVAL_EVERY = 5

_W: dict = {}


def _init():
    torch.set_num_threads(1)
    common.SOLVABLE_ONLY = True
    _W["train"] = common.load_network("daejeon_train")
    _W["val"] = common.load_network("daejeon_validation")
    _W["net"] = attn.AttnAssigner()
    _W["net"].eval()


def noise(seed: int, n: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32)


def _job(args):
    theta_path, version, noise_seed, sign, net_name, tag, idx, vf = args
    if _W.get("version") != version:
        _W["theta"], _W["version"] = np.load(theta_path), version
    theta = _W["theta"]
    vec = theta if noise_seed is None else theta + sign * SIGMA * noise(noise_seed, len(theta))
    with torch.no_grad():
        vector_to_parameters(torch.from_numpy(vec), _W["net"].parameters())
    net = _W[net_name]
    common.POLICE_SPEED_MPS, common.FUGITIVE_SPEED_MPS = 16.0, vf
    case = common.make_case(net, tag, idx)
    pol = attn.AttnTeamPolicy(net, _W["net"], vp=16.0, vf=vf)
    rec = common.replay_episode(case, network=net, team_policy=pol)
    return rec.outcome.value, rec.physical_steps


def centered_ranks(x: np.ndarray) -> np.ndarray:
    r = np.empty(len(x), dtype=np.float32)
    r[x.argsort()] = np.arange(len(x), dtype=np.float32)
    return r / (len(x) - 1) - 0.5


def main() -> int:
    out_dir, iters = Path(sys.argv[1]), int(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    net = attn.AttnAssigner()
    if os.environ.get("ES_INIT"):
        net.load_state_dict(torch.load(os.environ["ES_INIT"], weights_only=False)["net"])
    theta = parameters_to_vector(net.parameters()).detach().numpy().astype(np.float32)
    n = len(theta)
    m, v = np.zeros(n, np.float32), np.zeros(n, np.float32)
    pool = get_context("fork").Pool(WORKERS, initializer=_init)
    rng = np.random.default_rng(0)
    log = open(out_dir / "train.jsonl", "a")
    best = -1.0
    theta_file = out_dir / "theta.npy"

    def save(path, meta):
        with torch.no_grad():
            vector_to_parameters(torch.from_numpy(theta.copy()), net.parameters())
        torch.save({"kind": "attn", "net": net.state_dict(), "meta": meta}, path)

    for it in range(iters + 1):
        t0 = time.time(); rec = dict(iter=it)
        if it > 0:
            seeds = rng.integers(1 << 31, size=POP)
            vfs = [float(rng.choice(SPEEDS)) for _ in range(CASES)]
            tag = f"es-it{it}"
            np.save(theta_file, theta)
            jobs = [(str(theta_file), it, int(s), sign, "train", tag, c, vfs[c])
                    for s in seeds for sign in (1.0, -1.0) for c in range(CASES)]
            res = pool.map(_job, jobs, chunksize=4)
            fit = np.array([score(o, st) for o, st in res], dtype=np.float32).reshape(POP, 2, CASES).mean(-1)
            caps = np.array([o == "capture" for o, _ in res]).reshape(POP, 2, CASES).mean(-1)
            ranks = centered_ranks(fit.ravel()).reshape(POP, 2)
            w = ranks[:, 0] - ranks[:, 1]
            grad = np.zeros(n, np.float32)
            for s, wi in zip(seeds, w):
                grad += wi * noise(int(s), n)
            grad = grad / (2 * POP * SIGMA) - WD * theta       # ascent direction
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad * grad
            theta = theta + LR * (m / (1 - 0.9 ** it)) / (np.sqrt(v / (1 - 0.999 ** it)) + 1e-8)
            rec.update(pop_capture=float(caps.mean()), pop_best=float(caps.max()), pop_worst=float(caps.min()),
                       fit_mean=float(fit.mean()), theta_norm=float(np.linalg.norm(theta)))
        if it % EVAL_EVERY == 0 or it == iters:
            np.save(theta_file, theta)
            res = pool.map(_job, [(str(theta_file), -it - 1, None, 0.0, "val", "beatgreedy-daejeon_validation", i, 16.0)
                                  for i in range(EVAL_EPISODES)], chunksize=4)
            val = float(np.mean([o == "capture" for o, _ in res]))
            rec["val16"] = val
            if val >= best:
                best = val
                save(out_dir / "best.pt", dict(iter=it, val16=val, algo="es"))
                rec["selected"] = True
            save(out_dir / "last.pt", dict(iter=it, algo="es"))
        rec["seconds"] = round(time.time() - t0, 1)
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(json.dumps(rec), flush=True)
    pool.close(); pool.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

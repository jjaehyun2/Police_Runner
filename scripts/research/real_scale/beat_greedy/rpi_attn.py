"""Rollout policy iteration for the attention target-assignment policy.

The sealed cases are deterministic (fixed fugitive RNG stream), so the value
of a single role decision can be measured exactly: replay the episode with
only that one decision (officer k at decision round t) forced to each
alternative target and the current policy everywhere else. That gives a
Monte-Carlo Q-value for every candidate at that state -- per-decision credit
assignment instead of one team-level outcome for ~50 decisions.

Each iteration:
  1. run the current argmax policy on fresh training cases;
  2. probe PROBES random decisions per case: Q(s, a) for every legal a;
  3. aggregate (state, Q) samples (DAgger-style buffer) and fit the network
     with a soft target softmax(Q / tau), only on states where the choice matters;
  4. validate (argmax, 1000 sealed validation cases), keep the best.

Usage: python rpi_attn.py <out_dir> <iters>
Env: RPI_INIT, RPI_CASES (256), RPI_PROBES (2), RPI_WORKERS (46), RPI_EVAL (1000), RPI_TAU (0.1)
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

import attn
import common
from train_attn import SPEEDS, score

CASES = int(os.environ.get("RPI_CASES", "256"))
PROBES = int(os.environ.get("RPI_PROBES", "2"))
WORKERS = int(os.environ.get("RPI_WORKERS", "46"))
EVAL_EPISODES = int(os.environ.get("RPI_EVAL", "1000"))
TAU = float(os.environ.get("RPI_TAU", "0.1"))
BUFFER_ITERS = int(os.environ.get("RPI_BUFFER", "1000"))
EPOCHS = 6
COMMIT_S = int(os.environ.get("RPI_COMMIT", "24"))   # probe: hold the target for this many seconds

_W: dict = {}


def _init():
    torch.set_num_threads(1)
    common.SOLVABLE_ONLY = True
    _W["train"] = common.load_network("daejeon_train")
    _W["val"] = common.load_network("daejeon_validation")
    _W["net"] = attn.AttnAssigner()
    _W["net"].eval()


def _load(path, version):
    if _W.get("version") != version:
        _W["net"].load_state_dict(torch.load(path, weights_only=False)["net"])
        _W["version"] = version


def _play(net_name, tag, idx, vf, override=None):
    net = _W[net_name]
    common.POLICE_SPEED_MPS, common.FUGITIVE_SPEED_MPS = 16.0, vf
    case = common.make_case(net, tag, idx)
    pol = attn.AttnTeamPolicy(net, _W["net"], vp=16.0, vf=vf)
    pol.override = override or {}
    rec = common.replay_episode(case, network=net, team_policy=pol)
    return rec, pol.log


def _probe(args):
    path, version, tag, idx, vf, seed = args
    _load(path, version)
    rec, log = _play("train", tag, idx, vf)
    base = score(rec.outcome.value, rec.physical_steps)
    rng = np.random.default_rng(seed)
    out = []
    choices = [d for d in log if d["mask"].sum() > 1]
    # early decisions set the formation and matter most: sample them more often
    w = np.array([1.0 / (1.0 + d["step"] / 12.0) for d in choices])
    picks = rng.choice(len(choices), size=min(PROBES, len(choices)), replace=False, p=w / w.sum()) if choices else []
    for d in picks:
        d = choices[d]
        q = np.full(len(d["mask"]), np.nan, dtype=np.float32)
        own, _ = _play("train", tag, idx, vf, {d["officer"]: (d["step"], d["step"] + COMMIT_S, d["targets"][d["action"]])})
        q[d["action"]] = score(own.outcome.value, own.physical_steps)
        for a in np.flatnonzero(d["mask"]):
            if a == d["action"]:
                continue
            r, _ = _play("train", tag, idx, vf, {d["officer"]: (d["step"], d["step"] + COMMIT_S, d["targets"][a])})
            q[a] = score(r.outcome.value, r.physical_steps)
        out.append((d["O"], d["C"], d["P"], d["officer"], d["mask"], d["action"], q))
    return rec.outcome.value, out


def _eval(args):
    path, version, idx = args
    _load(path, version)
    rec, _ = _play("val", "beatgreedy-daejeon_validation", idx, 16.0)
    return rec.outcome.value


def main() -> int:
    out_dir, iters = Path(sys.argv[1]), int(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0); torch.set_num_threads(8)
    net = attn.AttnAssigner()
    if os.environ.get("RPI_INIT"):
        net.load_state_dict(torch.load(os.environ["RPI_INIT"], weights_only=False)["net"])
    opt = torch.optim.Adam(net.parameters(), lr=float(os.environ.get("RPI_LR", "1e-4")))
    pool = get_context("fork").Pool(WORKERS, initializer=_init)
    rng = np.random.default_rng(0)
    log = open(out_dir / "train.jsonl", "a")
    cur = out_dir / "current.pt"
    buffer: list[list] = []
    best = -1.0
    for it in range(iters + 1):
        t0 = time.time(); rec = dict(iter=it)
        torch.save({"kind": "attn", "net": net.state_dict()}, cur)
        if it > 0:
            jobs = [(str(cur), it, f"rpi-it{it}", c, float(rng.choice(SPEEDS)), int(rng.integers(1 << 31)))
                    for c in range(CASES)]
            res = pool.map(_probe, jobs, chunksize=1)
            samples = [s for _, ss in res for s in ss]
            rec["train_capture"] = float(np.mean([o == "capture" for o, _ in res]))
            gaps = [np.nanmax(s[6]) - s[6][s[5]] for s in samples]
            matters = sum(np.nanmax(s[6]) - np.nanmin(s[6]) > 1e-6 for s in samples)
            rec.update(probes=len(samples), matters=int(matters), improvable=int(sum(g > 1e-6 for g in gaps)),
                       mean_gap=float(np.mean(gaps)) if gaps else 0.0)
            # every probed state is kept: where no alternative is strictly better the target is the
            # policy's own choice (anchors behaviour), otherwise the best measured alternative
            buffer.append(samples)
            buffer = buffer[-BUFFER_ITERS:]
            data = [s for b in buffer for s in b]
            if data:
                O = torch.from_numpy(np.stack([s[0] for s in data])); C = torch.from_numpy(np.stack([s[1] for s in data]))
                P = torch.from_numpy(np.stack([s[2] for s in data])); K = torch.tensor([s[3] for s in data])
                M = torch.from_numpy(np.stack([s[4] for s in data]))
                Q = torch.from_numpy(np.nan_to_num(np.stack([s[6] for s in data]), nan=-9.0))
                own = torch.tensor([s[5] for s in data])
                q_own = Q.gather(1, own[:, None]).squeeze(1)
                best_a = Q.masked_fill(~M, -1e9).argmax(-1)
                better = Q.max(-1).values > q_own + 1e-6
                tgt_a = torch.where(better, best_a, own)
                target = torch.nn.functional.one_hot(tgt_a, M.shape[1]).float()
                net.train()
                for _ in range(EPOCHS):
                    perm = torch.randperm(len(K))
                    for s in range(0, len(K), 512):
                        b = perm[s:s + 512]
                        logits = net(O[b], C[b], P[b], K[b]).masked_fill(~M[b], -1e9)
                        loss = -(target[b] * torch.log_softmax(logits, -1)).sum(-1).mean()
                        opt.zero_grad(); loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                net.eval()
                with torch.no_grad():
                    pick = net(O, C, P, K).masked_fill(~M, -1e9).argmax(-1)
                    rec["fit_best"] = float((Q.gather(1, pick[:, None]).squeeze(1) >= Q.max(-1).values - 1e-6).float().mean())
                    rec["fit_improve"] = float((pick[better] == tgt_a[better]).float().mean()) if better.any() else 0.0
                    rec["fit_keep"] = float((pick[~better] == tgt_a[~better]).float().mean())
                rec["buffer"] = len(data)
                torch.save({"kind": "attn", "net": net.state_dict()}, cur)
        outs = pool.map(_eval, [(str(cur), -it - 1, i) for i in range(EVAL_EPISODES)], chunksize=4)
        val = float(np.mean([o == "capture" for o in outs]))
        rec["val16"] = val
        if val >= best:
            best = val
            torch.save({"kind": "attn", "net": net.state_dict(), "meta": dict(iter=it, val16=val, algo="rpi")},
                       out_dir / "best.pt")
            rec["selected"] = True
        rec["seconds"] = round(time.time() - t0, 1)
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(json.dumps(rec), flush=True)
    pool.close(); pool.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

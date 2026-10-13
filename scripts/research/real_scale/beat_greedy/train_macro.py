"""Train the role-selection policy from scratch (no imitation).

Rollouts run through the official ``replay_episode`` loop on solvable
training cases. Two advantage estimators (same shared-actor policy, same
decentralized execution):

  grpo   G rollouts per case from the identical initial state; each rollout's
         outcome is compared with its own group (no critic).
  mappo  centralized critic over the team's decision rows (PPO-clip, GAE over
         each officer's role decisions).

Usage: python train_macro.py <grpo|mappo> <out_dir> <iterations> [cases] [group]
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

POLICE_SPEED = 16.0
SPEEDS = tuple(float(x) for x in os.environ.get("TM_SPEEDS", "12,14,16,16,16").split(","))
LR = float(os.environ.get("TM_LR", "3e-4"))
EPOCHS = int(os.environ.get("TM_EPOCHS", "4"))
ENTROPY = float(os.environ.get("TM_ENT", "0.01"))
HIDDEN = (128, 128)
CLIP = 0.2
EVAL_EVERY = 5
EVAL_EPISODES = 400

_W: dict = {}


def _init():
    torch.set_num_threads(1)
    common.SOLVABLE_ONLY = True
    _W["train"] = common.load_network("daejeon_train")
    _W["val"] = common.load_network("daejeon_validation")
    _W["scorer"] = macro.CandidateScorer(HIDDEN)


def _job(args):
    version, state, net_name, tag, idx, vf, seed, mode = args
    if _W.get("version") != version:
        _W["scorer"].load_state_dict(state)
        _W["version"] = version
    net = _W[net_name]
    common.POLICE_SPEED_MPS, common.FUGITIVE_SPEED_MPS = POLICE_SPEED, vf
    case = common.make_case(net, tag, idx)
    pol = macro.MacroTeamPolicy(net, _W["scorer"], vp=POLICE_SPEED, vf=vf, mode=mode,
                                rng=np.random.default_rng(seed))
    rec = common.replay_episode(case, network=net, team_policy=pol)
    log = pol.log if mode == "sample" else []
    return idx, rec.outcome.value, rec.physical_steps, log


def score(outcome, steps):
    if outcome == "capture":
        return 1.0 + 0.5 * (1.0 - steps / 450.0)
    return -1.0 if outcome == "escape" else -0.5


def save(path, scorer, meta):
    torch.save({"kind": "macro", "hidden": HIDDEN, "scorer": scorer.state_dict(), "meta": meta}, path)


def main() -> int:
    algo, out_dir, iters = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
    cases = int(sys.argv[4]) if len(sys.argv) > 4 else 128
    group = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(16)
    scorer = macro.CandidateScorer(HIDDEN)
    if os.environ.get("TM_INIT"):
        scorer.load_state_dict(torch.load(os.environ["TM_INIT"], weights_only=False)["scorer"])
    opt = torch.optim.Adam(scorer.parameters(), lr=LR)
    critic = None
    if algo == "mappo":
        critic = torch.nn.Sequential(torch.nn.Linear(macro.N_CANDIDATES * macro.ROW_DIM + macro.POLICE_COUNT, 256),
                                     torch.nn.Tanh(), torch.nn.Linear(256, 256), torch.nn.Tanh(), torch.nn.Linear(256, 1))
        opt_c = torch.optim.Adam(critic.parameters(), lr=1e-3)
    pool = get_context("fork").Pool(44, initializer=_init)
    rng = np.random.default_rng(0)
    log = open(out_dir / "train.jsonl", "a")
    best = -1.0

    def evaluate(version):
        st = scorer.state_dict()
        res = pool.map(_job, [(version, st, "val", "beatgreedy-daejeon_validation", i, 16.0, i, "argmax")
                              for i in range(EVAL_EPISODES)], chunksize=1)
        return float(np.mean([o == "capture" for _, o, _, _ in res]))

    for it in range(0, iters + 1):
        t0 = time.time()
        version = it + 1
        rec = dict(iter=it)
        if it > 0:
            st = scorer.state_dict()
            jobs = []
            for c in range(cases):
                vf = float(rng.choice(SPEEDS))
                for _ in range(group if algo == "grpo" else 1):
                    jobs.append((version, st, "train", f"tm-{algo}-it{it}", c, vf, int(rng.integers(1 << 31)), "sample"))
            res = pool.map(_job, jobs, chunksize=1)
            rows, ids, masks, acts, lps, adv, wts = [], [], [], [], [], [], []
            rec["train_capture"] = float(np.mean([o == "capture" for _, o, _, _ in res]))
            if algo == "grpo":
                by = {}
                for idx, o, s, lg in res:
                    by.setdefault(idx, []).append((score(o, s), lg))
                informative = 0
                for items in by.values():
                    sc = np.array([x[0] for x in items])
                    if sc.std() < 1e-6:
                        continue
                    informative += 1
                    a = (sc - sc.mean()) / (sc.std() + 1e-6)
                    for (_, lg), ai in zip(items, a):
                        for d in lg:
                            rows.append(d["rows"]); ids.append(d["officer"]); masks.append(d["mask"])
                            acts.append(d["action"]); lps.append(d["logp"]); adv.append(ai); wts.append(1.0 / len(lg))
                rec["informative"] = informative
            else:
                # MAPPO: per-officer decision sequence, terminal outcome reward, critic baseline (GAE).
                gamma, lam = 0.98, 0.95
                for idx, o, s, lg in res:
                    final = score(o, s)
                    seqs = {}
                    for d in lg:
                        seqs.setdefault(d["officer"], []).append(d)
                    for k, seq in seqs.items():
                        x = torch.tensor(np.stack([np.concatenate([d["rows"].ravel(), np.eye(6)[k]]) for d in seq]),
                                         dtype=torch.float32)
                        with torch.no_grad():
                            v = critic(x).squeeze(-1).numpy()
                        gae, rets, advs = 0.0, [], []
                        for j in reversed(range(len(seq))):
                            nxt = v[j + 1] if j + 1 < len(seq) else 0.0
                            r = final if j + 1 == len(seq) else 0.0
                            delta = r + gamma * nxt - v[j]
                            gae = delta + gamma * lam * gae
                            advs.append(gae); rets.append(gae + v[j])
                        advs.reverse(); rets.reverse()
                        for d, a_, r_ in zip(seq, advs, rets):
                            rows.append(d["rows"]); ids.append(k); masks.append(d["mask"]); acts.append(d["action"])
                            lps.append(d["logp"]); adv.append(a_); wts.append(r_)
            if rows:
                R = torch.from_numpy(np.stack(rows)); I = torch.tensor(ids, dtype=torch.long)
                M = torch.from_numpy(np.stack(masks)); A = torch.tensor(acts, dtype=torch.long)
                LP = torch.tensor(lps, dtype=torch.float32); ADV = torch.tensor(adv, dtype=torch.float32)
                if algo == "mappo":
                    RET = torch.tensor(wts, dtype=torch.float32)
                    ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
                    W = torch.ones(len(R))
                else:
                    W = torch.tensor(wts, dtype=torch.float32)
                    W = W / W.sum() * len(W)
                n = len(R)
                stats = []
                for _ in range(EPOCHS):
                    perm = torch.randperm(n)
                    for s in range(0, n, 2048):
                        b = perm[s:s + 2048]
                        logits = scorer(R[b], I[b]).masked_fill(~M[b], -1e9)
                        lp_all = torch.log_softmax(logits, -1)
                        lp = lp_all.gather(1, A[b][:, None]).squeeze(1)
                        ratio = torch.exp(lp - LP[b])
                        surr = torch.min(ratio * ADV[b], ratio.clamp(1 - CLIP, 1 + CLIP) * ADV[b])
                        p = lp_all.exp()
                        ent = -(p * lp_all.masked_fill(~M[b], 0.0)).sum(-1)
                        loss = -(W[b] * surr).mean() - ENTROPY * ent.mean()
                        opt.zero_grad(); loss.backward()
                        torch.nn.utils.clip_grad_norm_(scorer.parameters(), 0.5); opt.step()
                        if algo == "mappo":
                            xb = torch.cat([R[b].flatten(1), torch.nn.functional.one_hot(I[b], 6).float()], -1)
                            vl = torch.nn.functional.mse_loss(critic(xb).squeeze(-1), RET[b])
                            opt_c.zero_grad(); vl.backward(); opt_c.step()
                        stats.append(float(ent.mean()))
                rec.update(entropy=float(np.mean(stats)), decisions=n)
        if it % EVAL_EVERY == 0 or it == iters:
            v = evaluate(10_000 + it)
            rec["val16"] = v
            if v >= best:
                best = v
                save(out_dir / "best.pt", scorer, dict(iter=it, val16=v, algo=algo))
                rec["selected"] = True
            save(out_dir / "last.pt", scorer, dict(iter=it, algo=algo))
        rec["seconds"] = round(time.time() - t0, 1)
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(json.dumps(rec), flush=True)
    pool.close(); pool.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

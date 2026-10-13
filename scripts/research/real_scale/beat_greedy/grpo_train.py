"""Multi-agent GRPO fine-tuning (group-relative advantages, no critic).

For each sampled training case, G stochastic rollouts start from the *same*
sealed initial state (same placement, same deterministic evader). Each
rollout's outcome score is compared with its own group's mean/std; every
decision in that rollout gets that advantage. Groups whose rollouts all end
the same way (unwinnable, or trivially won) carry zero advantage and are
dropped, so learning concentrates on exactly the cases where the team's
choices decide the outcome. The actor architecture (parameter-shared, officer
one-hot, decentralized execution) is unchanged; only the critic baseline is
replaced by the group baseline.

Usage: python grpo_train.py <init.pt> <out_dir> <iterations> [cases_per_iter] [group]
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import ckpt
from engine import Collector

POLICE_SPEED = 16.0
FUGITIVE_SPEEDS = tuple(float(x) for x in os.environ.get("GRPO_SPEEDS", "12,14,16,16,16").split(","))
REWARD = dict(time_penalty=0.01, capture=10.0, escape=10.0, shaping=0.0, containment=0.0, gamma=0.995)
LR = float(os.environ.get("GRPO_LR", "1e-4"))
KL_COEF = float(os.environ.get("GRPO_KL", "0.05"))
ENTROPY = float(os.environ.get("GRPO_ENT", "0.003"))
CLIP = 0.2
EPOCHS = int(os.environ.get("GRPO_EPOCHS", "2"))
MINIBATCH = 4096
EVAL_EVERY = 5
EVAL_EPISODES = 400


def outcome_score(ep) -> float:
    if ep.outcome == "capture":
        return 1.0 + 0.5 * (1.0 - ep.steps / 450.0)   # faster capture is better
    if ep.outcome == "escape":
        return -1.0
    return -0.5


def masked_log_softmax(logits, masks):
    return torch.log_softmax(logits.masked_fill(~masks, -1e9), dim=-1)


def evaluate(collector, actor, version, vf):
    jobs = [((version, actor.state_dict()), "beatgreedy-daejeon_validation", i, POLICE_SPEED, vf,
             "argmax", None, 0.0, i, REWARD) for i in range(EVAL_EPISODES)]
    eps = collector.collect(jobs)
    return float(np.mean([e.outcome == "capture" for e in eps]))


def main() -> int:
    init, out_dir, iterations = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    cases = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    group = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(2)
    torch.set_num_threads(16)
    actor, critic, payload = ckpt.load(init)
    hidden = tuple(payload["hidden_dims"])
    anchor = copy.deepcopy(actor).eval()
    for p in anchor.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(actor.parameters(), lr=LR)
    train_c = Collector("daejeon_train", hidden)
    val_c = Collector("daejeon_validation", hidden)
    rng = np.random.default_rng(2)
    log = open(out_dir / "train.jsonl", "a")

    v16, v9 = evaluate(val_c, actor, 1, 16.0), evaluate(val_c, actor, 2, 9.0)
    best = v16 + v9
    print(json.dumps(dict(iter=0, val16=v16, val9=v9, note="init")), flush=True)
    ckpt.save(out_dir / "best.pt", actor, critic, hidden=hidden, meta=dict(stage="grpo", iter=0, val16=v16, val9=v9))

    for it in range(1, iterations + 1):
        t0 = time.time()
        version = 20_000 + it
        state = (version, actor.state_dict())
        jobs, keys = [], []
        for c in range(cases):
            vf = float(rng.choice(FUGITIVE_SPEEDS))
            for g in range(group):
                jobs.append((state, f"grpo-it{it}", c, POLICE_SPEED, vf, "sample", None, 0.0,
                             int(rng.integers(1 << 31)), REWARD))
                keys.append(c)
        episodes = train_c.collect(jobs)

        obs, ids, masks, acts, logps, adv, wts = [], [], [], [], [], [], []
        informative = 0
        by_case: dict[int, list] = {}
        for k, ep in zip(keys, episodes):
            by_case.setdefault(k, []).append(ep)
        for eps in by_case.values():
            scores = np.array([outcome_score(e) for e in eps])
            if scores.std() < 1e-6:
                continue
            informative += 1
            a = (scores - scores.mean()) / (scores.std() + 1e-6)
            for ep, ai in zip(eps, a):
                n = len(ep.decisions)
                for d in ep.decisions:
                    obs.append(d.obs); ids.append(d.officer); masks.append(d.mask); acts.append(d.action)
                    logps.append(d.logp); adv.append(ai); wts.append(1.0 / max(n, 1))
        rec = dict(iter=it, capture=float(np.mean([e.outcome == "capture" for e in episodes])),
                   informative_groups=informative, decisions=len(obs))
        if obs:
            O = torch.from_numpy(np.stack(obs)); I = torch.tensor(ids, dtype=torch.long)
            M = torch.from_numpy(np.stack(masks)); A = torch.tensor(acts, dtype=torch.long)
            LP = torch.tensor(logps, dtype=torch.float32); ADV = torch.tensor(adv, dtype=torch.float32)
            W = torch.tensor(wts, dtype=torch.float32)
            W = W / W.sum() * len(W)
            n = len(O)
            stats = []
            for _ in range(EPOCHS):
                perm = torch.randperm(n)
                for s in range(0, n, MINIBATCH):
                    idx = perm[s:s + MINIBATCH]
                    lp_all = masked_log_softmax(actor(O[idx], I[idx]), M[idx])
                    lp = lp_all.gather(1, A[idx][:, None]).squeeze(1)
                    ratio = torch.exp(lp - LP[idx])
                    adv_i = ADV[idx]
                    surr = torch.min(ratio * adv_i, ratio.clamp(1 - CLIP, 1 + CLIP) * adv_i)
                    p = lp_all.exp()
                    ent = -(p * lp_all.masked_fill(~M[idx], 0.0)).sum(-1)
                    with torch.no_grad():
                        alp = masked_log_softmax(anchor(O[idx], I[idx]), M[idx])
                    kl = (p * (lp_all - alp).masked_fill(~M[idx], 0.0)).sum(-1)
                    w = W[idx]
                    loss = -(w * surr).mean() - ENTROPY * (w * ent).mean() + KL_COEF * (w * kl).mean()
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5); opt.step()
                    stats.append((float(ent.mean()), float(kl.mean()), float(ratio.mean())))
            ent_m, kl_m, ratio_m = np.mean(stats, axis=0)
            rec.update(entropy=ent_m, kl=kl_m, ratio=ratio_m)
        rec["seconds"] = round(time.time() - t0, 1)
        if it % EVAL_EVERY == 0 or it == iterations:
            v16, v9 = evaluate(val_c, actor, version + 1, 16.0), evaluate(val_c, actor, version + 2, 9.0)
            rec.update(val16=v16, val9=v9)
            if v16 + v9 >= best:
                best = v16 + v9
                ckpt.save(out_dir / "best.pt", actor, critic, hidden=hidden,
                          meta=dict(stage="grpo", iter=it, val16=v16, val9=v9))
                rec["selected"] = True
            ckpt.save(out_dir / "last.pt", actor, critic, hidden=hidden, meta=dict(stage="grpo", iter=it))
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(json.dumps(rec), flush=True)
    train_c.close(); val_c.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

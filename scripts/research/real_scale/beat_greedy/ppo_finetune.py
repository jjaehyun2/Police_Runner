"""MAPPO fine-tuning from a behavior-cloned initialization.

Differences from the audited trainer that matter here:
  * the policy starts from a competent BC policy, so exploration never has to
    discover "moving at all" from a near-uniform policy (the STAY collapse);
  * escape is penalized (the audited reward had no escape term at all, so an
    early escape was never worse than a long chase);
  * a KL anchor to the BC policy keeps early updates from undoing it;
  * advantages are computed per officer over its own decision sequence with
    the semi-Markov discount gamma^(steps between decisions);
  * checkpoint selection uses argmax play on a fixed validation case set
    (the exact rule used in evaluation), not stochastic play.

Usage: python ppo_finetune.py <init.pt> <out_dir> <iterations> [episodes_per_iter]
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import ckpt
from engine import Collector

import os
FUGITIVE_SPEEDS = tuple(float(x) for x in os.environ.get("PPO_SPEEDS", "9,12,14,16,16").split(","))
POLICE_SPEED = 16.0
REWARD = dict(time_penalty=0.01, capture=10.0, escape=10.0, shaping=0.0,
              containment=float(os.environ.get("PPO_CONTAIN", "0.0")), gamma=0.995)
GAMMA = 0.995
LAMBDA = 0.95
CLIP = 0.2
ENTROPY = float(os.environ.get("PPO_ENT", "0.003"))
KL_START, KL_END = float(os.environ.get("PPO_KL_START", "0.5")), float(os.environ.get("PPO_KL_END", "0.02"))
PPO_EPOCHS = 4
MINIBATCH = 4096
LR_ACTOR, LR_CRITIC = float(os.environ.get("PPO_LR", "1e-4")), 5e-4
EVAL_EVERY = 5
EVAL_EPISODES = 400


def build_batch(episodes, critic):
    obs, ids, masks, acts, logps, ctx, adv, ret = [], [], [], [], [], [], [], []
    for ep in episodes:
        if not ep.decisions:
            continue
        rewards = np.asarray(ep.rewards, dtype=np.float64)
        T = len(rewards)
        disc_cum = np.concatenate([[0.0], np.cumsum(rewards * GAMMA ** np.arange(T))])
        contexts = torch.from_numpy(np.stack([d.context for d in ep.decisions]))
        with torch.no_grad():
            values = critic(contexts).numpy().astype(np.float64)
        by_officer: dict[int, list[int]] = {}
        for i, d in enumerate(ep.decisions):
            by_officer.setdefault(d.officer, []).append(i)
        advantages = np.zeros(len(ep.decisions))
        returns = np.zeros(len(ep.decisions))
        for seq in by_officer.values():
            gae = 0.0
            for j in reversed(range(len(seq))):
                i = seq[j]
                t0 = ep.decisions[i].t
                t1 = ep.decisions[seq[j + 1]].t if j + 1 < len(seq) else T
                reward_sum = (disc_cum[t1] - disc_cum[t0]) / (GAMMA ** t0)
                discount = GAMMA ** (t1 - t0)
                next_value = values[seq[j + 1]] if j + 1 < len(seq) else 0.0
                delta = reward_sum + discount * next_value - values[i]
                gae = delta + discount * LAMBDA * gae if j + 1 < len(seq) else delta
                advantages[i] = gae
                returns[i] = gae + values[i]
        for i, d in enumerate(ep.decisions):
            obs.append(d.obs); ids.append(d.officer); masks.append(d.mask); acts.append(d.action)
            logps.append(d.logp); ctx.append(d.context)
        adv.extend(advantages); ret.extend(returns)
    adv = np.asarray(adv)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return dict(
        obs=torch.from_numpy(np.stack(obs)), ids=torch.tensor(ids, dtype=torch.long),
        masks=torch.from_numpy(np.stack(masks)), acts=torch.tensor(acts, dtype=torch.long),
        logps=torch.tensor(logps, dtype=torch.float32), ctx=torch.from_numpy(np.stack(ctx)),
        adv=torch.tensor(adv, dtype=torch.float32), ret=torch.tensor(np.asarray(ret), dtype=torch.float32),
    )


def masked_log_softmax(logits, masks):
    return torch.log_softmax(logits.masked_fill(~masks, -1e9), dim=-1)


def ppo_update(actor, critic, anchor, opt_a, opt_c, batch, kl_coef):
    n = batch["obs"].shape[0]
    stats = []
    for _ in range(PPO_EPOCHS):
        perm = torch.randperm(n)
        for start in range(0, n, MINIBATCH):
            idx = perm[start:start + MINIBATCH]
            m = batch["masks"][idx]
            logp_all = masked_log_softmax(actor(batch["obs"][idx], batch["ids"][idx]), m)
            logp = logp_all.gather(1, batch["acts"][idx][:, None]).squeeze(1)
            ratio = torch.exp(logp - batch["logps"][idx])
            a = batch["adv"][idx]
            policy_loss = -torch.min(ratio * a, ratio.clamp(1 - CLIP, 1 + CLIP) * a).mean()
            p = logp_all.exp()
            entropy = -(p * logp_all.masked_fill(~m, 0.0)).sum(-1).mean()
            with torch.no_grad():
                anchor_logp = masked_log_softmax(anchor(batch["obs"][idx], batch["ids"][idx]), m)
            kl = (p * (logp_all - anchor_logp).masked_fill(~m, 0.0)).sum(-1).mean()
            loss_a = policy_loss - ENTROPY * entropy + kl_coef * kl
            opt_a.zero_grad(); loss_a.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5); opt_a.step()
            v = critic(batch["ctx"][idx])
            loss_c = torch.nn.functional.mse_loss(v, batch["ret"][idx])
            opt_c.zero_grad(); loss_c.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5); opt_c.step()
            stats.append((float(policy_loss), float(entropy), float(kl), float(loss_c), float(ratio.mean())))
    return np.mean(stats, axis=0)


def pretrain_critic(critic, opt_c, batch, epochs=20):
    n = batch["obs"].shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for start in range(0, n, MINIBATCH):
            idx = perm[start:start + MINIBATCH]
            loss = torch.nn.functional.mse_loss(critic(batch["ctx"][idx]), batch["ret"][idx])
            opt_c.zero_grad(); loss.backward(); opt_c.step()


def evaluate(collector, actor, version, vf):
    jobs = [((version, actor.state_dict()), "beatgreedy-daejeon_validation", i, POLICE_SPEED, vf,
             "argmax", None, 0.0, i, REWARD) for i in range(EVAL_EPISODES)]
    eps = collector.collect(jobs)
    return float(np.mean([e.outcome == "capture" for e in eps]))


def main() -> int:
    init, out_dir, iterations = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    per_iter = int(sys.argv[4]) if len(sys.argv) > 4 else 400
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1)
    torch.set_num_threads(16)
    actor, critic, payload = ckpt.load(init)
    hidden = tuple(payload["hidden_dims"])
    anchor = copy.deepcopy(actor).eval()
    for prm in anchor.parameters():
        prm.requires_grad_(False)
    opt_a = torch.optim.Adam(actor.parameters(), lr=LR_ACTOR)
    opt_c = torch.optim.Adam(critic.parameters(), lr=LR_CRITIC)
    train_c = Collector("daejeon_train", hidden)
    val_c = Collector("daejeon_validation", hidden)
    rng = np.random.default_rng(1)
    log = open(out_dir / "train.jsonl", "a")

    v16, v9 = evaluate(val_c, actor, 1, 16.0), evaluate(val_c, actor, 2, 9.0)
    best = v16 + v9
    print(json.dumps(dict(iter=0, val16=v16, val9=v9, note="init")), flush=True)
    ckpt.save(out_dir / "best.pt", actor, critic, hidden=hidden, meta=dict(stage="ppo", iter=0, val16=v16, val9=v9))

    for it in range(1, iterations + 1):
        t0 = time.time()
        version = 10_000 + it
        state = (version, actor.state_dict())
        jobs = [(state, f"ppo-it{it}", e, POLICE_SPEED, float(rng.choice(FUGITIVE_SPEEDS)), "sample", None, 0.0,
                 int(rng.integers(1 << 31)), REWARD) for e in range(per_iter)]
        episodes = train_c.collect(jobs)
        batch = build_batch(episodes, critic)
        if it == 1:
            pretrain_critic(critic, opt_c, batch)
            batch = build_batch(episodes, critic)
        frac = min(1.0, (it - 1) / max(1, iterations * 0.6))
        kl_coef = KL_START + (KL_END - KL_START) * frac
        pl, ent, kl, vl, ratio = ppo_update(actor, critic, anchor, opt_a, opt_c, batch, kl_coef)
        rec = dict(
            iter=it, train_capture=float(np.mean([e.outcome == "capture" for e in episodes])),
            train_escape=float(np.mean([e.outcome == "escape" for e in episodes])),
            decisions=int(batch["obs"].shape[0]), policy_loss=pl, entropy=ent, kl=kl, value_loss=vl,
            ratio=ratio, kl_coef=kl_coef, seconds=round(time.time() - t0, 1),
        )
        if it % EVAL_EVERY == 0 or it == iterations:
            v16, v9 = evaluate(val_c, actor, version + 1, 16.0), evaluate(val_c, actor, version + 2, 9.0)
            rec.update(val16=v16, val9=v9)
            if v16 + v9 >= best:
                best = v16 + v9
                ckpt.save(out_dir / "best.pt", actor, critic, hidden=hidden,
                          meta=dict(stage="ppo", iter=it, val16=v16, val9=v9))
                rec["selected"] = True
            ckpt.save(out_dir / "last.pt", actor, critic, hidden=hidden, meta=dict(stage="ppo", iter=it))
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(json.dumps(rec), flush=True)
    train_c.close(); val_c.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

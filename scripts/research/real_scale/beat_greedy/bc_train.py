"""DAgger behavior cloning of a rule teacher into the extended-observation actor.

Iteration 0 lets the teacher drive; later iterations let the student drive
(argmax, the same rule it is evaluated with) with a decaying chance of
executing the teacher's action, and always label with the teacher. Data is
aggregated across iterations. Fugitive speed is randomized per episode so one
policy covers every speed ratio.

Usage: python bc_train.py <teacher> <iterations> <episodes_per_iter> <out.pt>
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import ckpt
from engine import Collector

HIDDEN = (256, 256)
FUGITIVE_SPEEDS = (9.0, 12.0, 14.0, 16.0)
POLICE_SPEED = 16.0
REWARD = dict(time_penalty=0.01, capture=10.0, escape=10.0, shaping=0.0)
EVAL_EPISODES = 400


def train_epochs(actor, data, epochs: int, lr: float) -> float:
    obs = torch.from_numpy(np.stack([d[0] for d in data]))
    ids = torch.tensor([d[1] for d in data], dtype=torch.long)
    masks = torch.from_numpy(np.stack([d[2] for d in data]))
    labels = torch.tensor([d[3] for d in data], dtype=torch.long)
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    n = len(data)
    last = 0.0
    for _ in range(epochs):
        perm = torch.randperm(n)
        total, correct = 0.0, 0
        for start in range(0, n, 4096):
            idx = perm[start:start + 4096]
            logits = actor(obs[idx], ids[idx]).masked_fill(~masks[idx], -1e9)
            loss = torch.nn.functional.cross_entropy(logits, labels[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
            correct += int((logits.argmax(-1) == labels[idx]).sum())
        last = correct / n
    return last


def evaluate(collector, actor, version: int, vf: float) -> float:
    jobs = [
        ((version, actor.state_dict()), "beatgreedy-daejeon_validation", i, POLICE_SPEED, vf,
         "argmax", None, 0.0, i, REWARD)
        for i in range(EVAL_EPISODES)
    ]
    eps = collector.collect(jobs)
    return float(np.mean([e.outcome == "capture" for e in eps]))


def main() -> int:
    teacher, iterations, per_iter, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
    torch.manual_seed(0)
    torch.set_num_threads(16)
    import os
    if os.environ.get("BC_INIT"):
        actor, critic, _ = ckpt.load(Path(os.environ["BC_INIT"]))
    else:
        actor, critic = ckpt.new_actor(HIDDEN), ckpt.new_critic(HIDDEN)
    first_beta = float(os.environ.get("BC_FIRST_BETA", "1.0"))
    train_c = Collector("daejeon_train", HIDDEN)
    val_c = Collector("daejeon_validation", HIDDEN)
    rng = np.random.default_rng(0)
    data: list[tuple] = []
    history = []
    best = -1.0
    for it in range(iterations):
        t0 = time.time()
        beta = first_beta if it == 0 else max(0.0, 0.5 - 0.15 * (it - 1))
        version = 1000 + it
        jobs = []
        for e in range(per_iter):
            vf = float(rng.choice(FUGITIVE_SPEEDS))
            state = None if (it == 0 and first_beta >= 1.0) else (version, actor.state_dict())
            mode = "teacher" if state is None else "argmax"
            jobs.append((state, f"bc-{teacher}-it{it}", e, POLICE_SPEED, vf, mode, teacher, beta,
                         int(rng.integers(1 << 31)), REWARD))
        episodes = train_c.collect(jobs)
        drive_capture = float(np.mean([ep.outcome == "capture" for ep in episodes]))
        for ep in episodes:
            for d in ep.decisions:
                data.append((d.obs, d.officer, d.mask, d.teacher_action))
        acc = train_epochs(actor, data, epochs=8 if it == 0 else 5, lr=1e-3 if it < 3 else 5e-4)
        v16 = evaluate(val_c, actor, version + 500, 16.0)
        v9 = evaluate(val_c, actor, version + 700, 9.0)
        rec = dict(iter=it, beta=beta, samples=len(data), train_acc=acc, drive_capture=drive_capture,
                   val_capture_vf16=v16, val_capture_vf9=v9, seconds=time.time() - t0)
        history.append(rec)
        print(json.dumps(rec), flush=True)
        score = v16 + v9
        if score > best:
            best = score
            ckpt.save(out, actor, critic, hidden=HIDDEN, meta=dict(stage="bc", teacher=teacher, **rec))
    (out.with_suffix(".history.json")).write_text(json.dumps(history, indent=1))
    train_c.close()
    val_c.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

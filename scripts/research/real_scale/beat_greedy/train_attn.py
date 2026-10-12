"""Train the attention target-assignment policy (attn.py).

  bc    imitate a macro rule (default ``assign``) -> warm start
  grpo  group-relative policy optimisation on sealed solvable training cases:
        G rollouts per case from the same initial state, outcome score
        compared within the group (no critic). Optional prioritized replay of
        decisions (|advantage|^alpha, importance-weighted) -- the paper's
        prioritized-experience idea in its simplest form.

Usage:
  python train_attn.py bc   <out.pt> [episodes]
  python train_attn.py grpo <out_dir> <iters> [cases] [group]
Env: TA_INIT (grpo warm start), TA_LR, TA_ENT, TA_TEMP, TA_PRIO, TA_WORKERS,
     TA_SPEEDS, TA_EVAL (episodes per validation eval)
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

POLICE_SPEED = 16.0
SPEEDS = tuple(float(x) for x in os.environ.get("TA_SPEEDS", "12,14,16,16,16").split(","))
LR = float(os.environ.get("TA_LR", "3e-4"))
ENTROPY = float(os.environ.get("TA_ENT", "0.01"))
TEMP = float(os.environ.get("TA_TEMP", "1.0"))
PRIO = float(os.environ.get("TA_PRIO", "0"))          # alpha; 0 = uniform
WORKERS = int(os.environ.get("TA_WORKERS", "44"))
EVAL_EPISODES = int(os.environ.get("TA_EVAL", "1000"))
EPOCHS = 4
CLIP = 0.2
EVAL_EVERY = 5

_W: dict = {}


def _init():
    torch.set_num_threads(1)
    common.SOLVABLE_ONLY = True
    _W["train"] = common.load_network("daejeon_train")
    _W["val"] = common.load_network("daejeon_validation")
    _W["net"] = attn.AttnAssigner()
    _W["net"].eval()


def _job(args):
    version, state, net_name, tag, idx, vf, seed, mode, rule, temp = args
    if state is not None and _W.get("version") != version:
        _W["net"].load_state_dict(state)
        _W["version"] = version
    net = _W[net_name]
    common.POLICE_SPEED_MPS, common.FUGITIVE_SPEED_MPS = POLICE_SPEED, vf
    case = common.make_case(net, tag, idx)
    pol = attn.AttnTeamPolicy(net, _W["net"], vp=POLICE_SPEED, vf=vf, mode=mode,
                              rng=np.random.default_rng(seed), fixed_rule=rule)
    pol.temperature = temp
    if rule and rule.startswith("mlp:"):
        if _W.get("teacher_path") != rule:
            import macro
            pay = torch.load(rule[4:], weights_only=False)
            t = macro.CandidateScorer(tuple(pay["hidden"])); t.load_state_dict(pay["scorer"]); t.eval()
            _W["teacher"], _W["teacher_path"] = t, rule
        pol.fixed_rule, pol.teacher = None, _W["teacher"]
    rec = common.replay_episode(case, network=net, team_policy=pol)
    log = [(d["O"], d["C"], d["P"], d["officer"], d["mask"], d["action"], d["logp"]) for d in pol.log] \
        if mode == "sample" or rule else []
    return idx, rec.outcome.value, rec.physical_steps, log


def score(outcome, steps):
    if outcome == "capture":
        return 1.0 + 0.5 * (1.0 - steps / 450.0)
    return -1.0 if outcome == "escape" else -0.5


def stack(data):
    O = torch.from_numpy(np.stack([d[0] for d in data])); C = torch.from_numpy(np.stack([d[1] for d in data]))
    P = torch.from_numpy(np.stack([d[2] for d in data])); K = torch.tensor([d[3] for d in data])
    M = torch.from_numpy(np.stack([d[4] for d in data])); A = torch.tensor([d[5] for d in data])
    LP = torch.tensor([d[6] for d in data], dtype=torch.float32)
    return O, C, P, K, M, A, LP


def save(path, net, meta):
    torch.save({"kind": "attn", "net": net.state_dict(), "meta": meta}, path)


def evaluate(pool, net, version, episodes=EVAL_EPISODES, network="val", vf=16.0):
    st = net.state_dict()
    tag = "beatgreedy-daejeon_validation" if network == "val" else "beatgreedy-daejeon_test"
    res = pool.map(_job, [(version, st, network, tag, i, vf, i, "argmax", None, 1.0)
                          for i in range(episodes)], chunksize=2)
    return float(np.mean([o == "capture" for _, o, _, _ in res]))


def run_bc(out: Path, episodes: int, rule: str = "assign") -> int:
    rng = np.random.default_rng(0)
    pool = get_context("fork").Pool(WORKERS, initializer=_init)
    t0 = time.time()
    res = pool.map(_job, [(0, None, "train", "ta-bc", i, float(rng.choice(SPEEDS)), i, "argmax", rule, 1.0)
                          for i in range(episodes)], chunksize=4)
    print("rule capture on train", np.mean([o == "capture" for _, o, _, _ in res]), round(time.time() - t0), "s", flush=True)
    O, C, P, K, M, A, _ = stack([d for _, _, _, lg in res for d in lg])
    torch.manual_seed(0); torch.set_num_threads(16)
    net = attn.AttnAssigner()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = len(A)
    for epoch in range(25):
        perm = torch.randperm(n); correct = 0
        for s in range(0, n, 1024):
            b = perm[s:s + 1024]
            logits = net(O[b], C[b], P[b], K[b]).masked_fill(~M[b], -1e9)
            loss = torch.nn.functional.cross_entropy(logits, A[b])
            opt.zero_grad(); loss.backward(); opt.step()
            correct += int((logits.argmax(-1) == A[b]).sum())
        print(f"epoch {epoch} acc {correct / n:.4f}", flush=True)
    net.eval()
    val = evaluate(pool, net, 1)
    pool.close(); pool.join()
    print(json.dumps(dict(samples=n, train_acc=correct / n, val16=val)), flush=True)
    save(out, net, dict(stage="bc", rule=rule, train_acc=correct / n, val16=val))
    return 0


def run_grpo(out_dir: Path, iters: int, cases: int, group: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0); torch.set_num_threads(8)
    net = attn.AttnAssigner()
    if os.environ.get("TA_INIT"):
        net.load_state_dict(torch.load(os.environ["TA_INIT"], weights_only=False)["net"])
    net.eval()                                   # no dropout/bn; eval() only for clarity
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    pool = get_context("fork").Pool(WORKERS, initializer=_init)
    rng = np.random.default_rng(0)
    log = open(out_dir / "train.jsonl", "a")
    best = -1.0
    for it in range(iters + 1):
        t0 = time.time(); rec = dict(iter=it)
        if it > 0:
            st = net.state_dict()
            jobs = []
            for c in range(cases):
                vf = float(rng.choice(SPEEDS))
                for _ in range(group):
                    jobs.append((it, st, "train", f"ta-grpo-it{it}", c, vf, int(rng.integers(1 << 31)), "sample", None, TEMP))
            res = pool.map(_job, jobs, chunksize=1)
            rec["train_capture"] = float(np.mean([o == "capture" for _, o, _, _ in res]))
            by = {}
            for idx, o, s, lg in res:
                by.setdefault(idx, []).append((score(o, s), lg))
            data, adv, wts, informative = [], [], [], 0
            for items in by.values():
                sc = np.array([x[0] for x in items])
                if sc.std() < 1e-6:
                    continue
                informative += 1
                a = (sc - sc.mean()) / (sc.std() + 1e-6)
                for (_, lg), ai in zip(items, a):
                    for d in lg:
                        data.append(d); adv.append(ai); wts.append(1.0 / len(lg))
            rec["informative"] = informative
            if data:
                O, C, P, K, M, A, LP = stack(data)
                ADV = torch.tensor(adv, dtype=torch.float32)
                W = torch.tensor(wts, dtype=torch.float32); W = W / W.sum() * len(W)
                n = len(A)
                if PRIO > 0:
                    pr = (ADV.abs() + 0.05) ** PRIO
                    pr = pr / pr.sum()
                    isw = (1.0 / (n * pr)) ** 0.5
                    W = W * isw / isw.mean()
                ents, kls = [], []
                for _ in range(EPOCHS):
                    order = torch.multinomial(pr, n, replacement=True) if PRIO > 0 else torch.randperm(n)
                    for s in range(0, n, 2048):
                        b = order[s:s + 2048]
                        logits = (net(O[b], C[b], P[b], K[b]) / TEMP).masked_fill(~M[b], -1e9)
                        lp_all = torch.log_softmax(logits, -1)
                        lp = lp_all.gather(1, A[b][:, None]).squeeze(1)
                        ratio = torch.exp(lp - LP[b])
                        surr = torch.min(ratio * ADV[b], ratio.clamp(1 - CLIP, 1 + CLIP) * ADV[b])
                        ent = -(lp_all.exp() * lp_all.masked_fill(~M[b], 0.0)).sum(-1)
                        loss = -(W[b] * surr).mean() - ENTROPY * ent.mean()
                        opt.zero_grad(); loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
                        ents.append(float(ent.mean())); kls.append(float((LP[b] - lp).mean()))
                rec.update(entropy=float(np.mean(ents)), approx_kl=float(np.mean(kls[-5:])), decisions=n)
        if it % EVAL_EVERY == 0 or it == iters:
            v = evaluate(pool, net, 100_000 + it)
            rec["val16"] = v
            if v >= best:
                best = v
                save(out_dir / "best.pt", net, dict(iter=it, val16=v))
                rec["selected"] = True
            save(out_dir / "last.pt", net, dict(iter=it))
        rec["seconds"] = round(time.time() - t0, 1)
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(json.dumps(rec), flush=True)
    pool.close(); pool.join()
    return 0


def main() -> int:
    if sys.argv[1] == "bc":
        return run_bc(Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 3000,
                      sys.argv[4] if len(sys.argv) > 4 else "assign")
    return run_grpo(Path(sys.argv[2]), int(sys.argv[3]),
                    int(sys.argv[4]) if len(sys.argv) > 4 else 128, int(sys.argv[5]) if len(sys.argv) > 5 else 8)


if __name__ == "__main__":
    raise SystemExit(main())

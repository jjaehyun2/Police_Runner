"""Episode engine for data collection (DAgger/BC) and PPO rollouts.

The step loop mirrors ``evaluation.paired.replay_episode`` exactly (same env,
same sealed-case reset, same GoalEvader, same police-then-fugitive ordering),
so what the policy experiences in training is what it is evaluated on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from multiprocessing import get_context
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import common
from common import load_network, make_case
from ckpt import EXT_OBS_DIM, EXT_PROFILE_ID
from obs_ext import EXT2_PROFILE_ID, ExtendedObservationAdapter, ExtendedObservationAdapterV2, open_exits
from pursuit_evasion_rl.osm_demo.environment import FUGITIVE_ID, OSMRoadPursuitEnv, STAY_ACTION
from pursuit_evasion_rl.osm_demo.metrics import placement_position
from pursuit_evasion_rl.osm_demo.models import EpisodeConfig
from pursuit_evasion_rl.research.policies.baselines import EncirclementPolice, GoalEvader, _Graph, greedy_intercept_policy
from pursuit_evasion_rl.research.policies.masked_mappo import SharedOfficerActor

POLICE_COUNT = 6
ACTION_DIM = 6


@dataclass
class Decision:
    officer: int
    t: int                 # physical step index at which the decision was taken
    obs: np.ndarray
    mask: np.ndarray       # bool[6]
    action: int
    logp: float
    teacher_action: int    # -1 when no teacher
    context: np.ndarray    # concatenated obs of all six officers (critic input)


@dataclass
class EpisodeData:
    decisions: list[Decision] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)   # team reward per physical step
    outcome: str = "timeout"
    steps: int = 0
    final_context: np.ndarray | None = None


def build_teacher(name: str, network, vp: float, vf: float):
    if name == "encircle":
        return EncirclementPolice(network, graph=_Graph(network), fugitive_speed_mps=vf, police_speed_mps=vp)
    if name == "greedy":
        return greedy_intercept_policy(network)
    if name.startswith("rule:"):
        import rules
        return rules.build(name[5:], network)
    raise ValueError(name)


def run_episode(
    network, *, tag: str, index: int, vp: float, vf: float, actor: SharedOfficerActor | None,
    mode: str, teacher_name: str | None, beta: float, rng: np.random.Generator,
    reward_cfg: dict,
) -> EpisodeData:
    """mode: 'sample' (stochastic actor), 'argmax', or 'teacher' (teacher drives).

    ``beta`` is the DAgger mixing probability: at each decision the teacher's
    action is executed with probability beta, otherwise the student's.
    """
    common.POLICE_SPEED_MPS, common.FUGITIVE_SPEED_MPS = vp, vf
    case = make_case(network, tag, index)
    env = OSMRoadPursuitEnv(network, case.environment)
    env.reset(seed=case.fugitive_rng.seed, options={"police": list(case.police), "fugitive": case.fugitive})
    adapter_cls = ExtendedObservationAdapterV2 if EXT_PROFILE_ID == EXT2_PROFILE_ID else ExtendedObservationAdapter
    adapter = adapter_cls(network, police_speed_mps=vp, fugitive_speed_mps=vf)
    containment = reward_cfg.get("containment", 0.0)
    gamma = reward_cfg.get("gamma", 0.995)
    potential = (-len(open_exits(network, case.police, case.fugitive, vp, vf)[0]) / 10.0) if containment else 0.0
    evader = GoalEvader(network, rng=case.fugitive_rng.generator(), graph=_Graph(network), vision_range_m=150.0)
    planning = teacher_name is not None and teacher_name.startswith("plan")
    if planning:
        from planner import PlanningTeacher
        opts = dict(kv.split("=") for kv in teacher_name.split(":")[1:]) if ":" in teacher_name else {}
        teacher = PlanningTeacher(network, vp=vp, vf=vf, max_steps=case.environment.max_steps,
                                  every=int(opts.get("every", 8)), horizon=int(opts.get("horizon", 200)))
    else:
        teacher = build_teacher(teacher_name, network, vp, vf) if teacher_name else None
    max_steps = case.environment.max_steps
    data = EpisodeData()

    while env.outcome is None and data.steps < case.termination.timeout_step_limit:
        state = env.episode_state()
        masks = env.action_masks()
        obs = [
            adapter.observe(
                police_index=k, police=state.police, fugitive=state.fugitive, step=state.step,
                max_steps=max_steps, incoming_heading=state.incoming_headings.get(f"police_{k}"),
            )
            for k in range(POLICE_COUNT)
        ]
        context = np.concatenate(obs)
        teacher_actions = None
        if planning:
            teacher_actions = teacher.act(env, evader, state, masks)
        elif teacher is not None:
            recs = teacher.recommend(
                police=state.police, fugitive=state.fugitive, step=state.step, max_steps=max_steps,
                incoming_headings=tuple(state.incoming_headings.get(f"police_{k}") for k in range(POLICE_COUNT)),
            )
            teacher_actions = [r.action_index for r in recs]
        deciding = [k for k in range(POLICE_COUNT) if int(masks[f"police_{k}"].sum()) > 1]
        actions = {f"police_{k}": STAY_ACTION for k in range(POLICE_COUNT)}
        if deciding:
            logits = None
            if actor is not None:
                with torch.no_grad():
                    logits = actor(
                        torch.from_numpy(np.stack([obs[k] for k in deciding])),
                        torch.tensor(deciding, dtype=torch.long),
                    ).numpy()
            for row, k in enumerate(deciding):
                mask = masks[f"police_{k}"].astype(bool)
                t_act = -1
                if teacher_actions is not None:
                    t_act = int(teacher_actions[k])
                    if not mask[t_act]:
                        t_act = int(np.flatnonzero(mask)[0])
                logp = 0.0
                if logits is not None:
                    z = np.where(mask, logits[row], -np.inf)
                    z = z - z.max()
                    p = np.exp(z)
                    p /= p.sum()
                    if mode == "argmax":
                        a = int(np.argmax(z))
                    else:
                        a = int(rng.choice(ACTION_DIM, p=p))
                    if teacher_actions is not None and rng.random() < beta:
                        a = t_act
                    logp = float(np.log(max(p[a], 1e-12)))
                else:
                    a = t_act
                actions[f"police_{k}"] = a
                data.decisions.append(Decision(k, data.steps, obs[k], mask, a, logp, t_act, context))
        actions[FUGITIVE_ID] = evader.env_provider(
            network, tuple(placement_position(network, p) for p in state.police)
        )
        before_min = _min_dist(network, state)
        env.step(actions)
        after = env.episode_state()
        data.steps += 1
        outcome = env.outcome.value if env.outcome is not None else None
        r = -reward_cfg["time_penalty"]
        if reward_cfg.get("shaping", 0.0):
            r += reward_cfg["shaping"] * (before_min - _min_dist(network, after)) / 50.0
        if containment:
            if outcome is None:
                new_potential = -len(open_exits(network, after.police, after.fugitive, vp, vf)[0]) / 10.0
            else:
                new_potential = 0.0
            r += containment * (gamma * new_potential - potential)
            potential = new_potential
        if outcome == "capture":
            r += reward_cfg["capture"]
        elif outcome == "escape":
            r -= reward_cfg["escape"]
        data.rewards.append(r)
    data.outcome = env.outcome.value if env.outcome is not None else "timeout"
    final = env.episode_state()
    data.final_context = np.concatenate([
        adapter.observe(
            police_index=k, police=final.police, fugitive=final.fugitive, step=final.step,
            max_steps=max_steps, incoming_heading=final.incoming_headings.get(f"police_{k}"),
        )
        for k in range(POLICE_COUNT)
    ])
    return data


def _min_dist(network, state) -> float:
    f = placement_position(network, state.fugitive)
    return min(math.dist(placement_position(network, p), f) for p in state.police)


# -- parallel collection -------------------------------------------------------
_W: dict = {}


def _init(network_name: str, hidden: tuple[int, ...]) -> None:
    torch.set_num_threads(1)
    _W["network"] = load_network(network_name)
    _W["actor"] = SharedOfficerActor(EXT_OBS_DIM, ACTION_DIM, POLICE_COUNT, hidden)


def _job(args):
    (actor_state, tag, index, vp, vf, mode, teacher, beta, seed, reward_cfg) = args
    actor = None
    if actor_state is not None:
        if _W.get("actor_version") != actor_state[0]:
            _W["actor"].load_state_dict(actor_state[1])
            _W["actor_version"] = actor_state[0]
        actor = _W["actor"]
    rng = np.random.default_rng(seed)
    return run_episode(
        _W["network"], tag=tag, index=index, vp=vp, vf=vf, actor=actor, mode=mode,
        teacher_name=teacher, beta=beta, rng=rng, reward_cfg=reward_cfg,
    )


class Collector:
    def __init__(self, network_name: str, hidden=(256, 256), workers: int = 40) -> None:
        self.pool = get_context("fork").Pool(workers, initializer=_init, initargs=(network_name, hidden))

    def collect(self, jobs: list[tuple]) -> list[EpisodeData]:
        return self.pool.map(_job, jobs, chunksize=1)

    def close(self) -> None:
        self.pool.close()
        self.pool.join()

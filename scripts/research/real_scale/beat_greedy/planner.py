"""Strategy-selection planning teacher (hindsight expert).

Keeps a live instance of several rule-based team strategies (each fed every
step so its internal history stays warm). Every ``every`` steps it copies the
environment and the evader model, simulates each strategy forward for up to
``horizon`` steps, and commits to the one with the best outcome (capture
soonest > timeout closest > escape latest). Its actions are then whatever the
committed strategy recommends.

This uses the simulator and evader model at decision time, so it is only a
*teacher*: the student distilled from it sees only its own observation and
runs with no simulator at test time.
"""
from __future__ import annotations

import copy
import math

from pursuit_evasion_rl.osm_demo.environment import FUGITIVE_ID, STAY_ACTION
from pursuit_evasion_rl.osm_demo.metrics import placement_position

import rules
from pursuit_evasion_rl.research.policies.baselines import EncirclementPolice, _Graph, greedy_intercept_policy, directed_shortest_path_policy

POLICE_COUNT = 6
DEFAULT_STRATEGIES = ("encircle", "enc:dp=1", "enc:dp=3", "greedy", "dsp", "block:chasers=2")


def make_strategy(name: str, network, vp: float, vf: float):
    if name == "encircle":
        return EncirclementPolice(network, graph=_Graph(network), fugitive_speed_mps=vf, police_speed_mps=vp)
    if name == "greedy":
        return greedy_intercept_policy(network)
    if name == "dsp":
        return directed_shortest_path_policy(network)
    return rules.build(name, network, vp, vf)


def _team_actions(policy, state, masks, max_steps):
    recs = policy.recommend(
        police=state.police, fugitive=state.fugitive, step=state.step, max_steps=max_steps,
        incoming_headings=tuple(state.incoming_headings.get(f"police_{k}") for k in range(POLICE_COUNT)),
    )
    actions = {}
    for k in range(POLICE_COUNT):
        mask = masks[f"police_{k}"]
        a = recs[k].action_index if int(mask.sum()) > 1 else STAY_ACTION
        actions[f"police_{k}"] = a if mask[a] else STAY_ACTION
    return actions, [r.action_index for r in recs]


class PlanningTeacher:
    def __init__(self, network, *, vp: float, vf: float, max_steps: int, strategies=DEFAULT_STRATEGIES,
                 every: int = 8, horizon: int = 200) -> None:
        self.network = network
        self.max_steps = max_steps
        self.names = list(strategies)
        self.live = [make_strategy(n, network, vp, vf) for n in self.names]
        self.every = every
        self.horizon = horizon
        self.current = 0
        self.last_plan_step: int | None = None
        self.choices: list[str] = []

    def _score(self, env, evader, policy, start_step: int) -> float:
        net = self.network
        steps = 0
        while env.outcome is None and steps < self.horizon and env.episode_state().step < self.max_steps:
            state = env.episode_state()
            masks = env.action_masks()
            actions, _ = _team_actions(policy, state, masks, self.max_steps)
            actions[FUGITIVE_ID] = evader.env_provider(net, tuple(placement_position(net, p) for p in state.police))
            env.step(actions)
            steps += 1
        outcome = env.outcome.value if env.outcome is not None else None
        if outcome == "capture":
            return 10_000.0 - steps
        if outcome == "escape":
            return -10_000.0 + steps
        state = env.episode_state()
        f = placement_position(net, state.fugitive)
        return -min(math.dist(placement_position(net, p), f) for p in state.police)

    def _reachable(self, evader, state) -> bool:
        """False when no officer can reach the fugitive at all (planning is moot)."""
        g = evader.g
        fp = state.fugitive
        fnode = int(g.seg[fp.segment_id].end_id) if fp.segment_id is not None else int(fp.intersection_id)
        to_f = g.dist_to(fnode)
        for p in state.police:
            n = int(g.seg[p.segment_id].end_id) if p.segment_id is not None else int(p.intersection_id)
            if n in to_f:
                return True
        return False

    def act(self, env, evader, state, masks) -> list[int]:
        """Teacher action per officer for the current state (call every step, in order)."""
        step = state.step
        if (self.last_plan_step is None or step - self.last_plan_step >= self.every) and self._reachable(evader, state):
            memo_base = {id(self.network): self.network}
            graph = evader.g
            scores = []
            for policy in self.live:
                memo = dict(memo_base)
                memo[id(graph)] = graph
                if hasattr(policy, "g"):
                    memo[id(policy.g)] = policy.g
                env_c = copy.deepcopy(env, memo)
                ev_c = copy.deepcopy(evader, memo)
                pol_c = copy.deepcopy(policy, memo)
                scores.append(self._score(env_c, ev_c, pol_c, step))
            best = max(range(len(scores)), key=lambda i: (scores[i], -i if i != self.current else 1))
            self.current = best
            self.last_plan_step = step
            self.choices.append(self.names[best])
        out = None
        for i, policy in enumerate(self.live):
            _, raw = _team_actions(policy, state, masks, self.max_steps)
            if i == self.current:
                out = raw
        return out

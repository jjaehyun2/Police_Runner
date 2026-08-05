"""도로 추격 데모 레이어 보상/학습 결함 수정에 대한 회귀 테스트.

다루는 결함:
1. 협력 보너스가 매 스텝, 전원에게 지급되어 체포보다 배회가 이득이던 문제
2. shortest_path_hops의 '경로 없음' sentinel이 shaping에 임펄스로 새던 문제
3. 경찰 shaping이 도망자 이동에 의존해 행동과 무관한 노이즈/공짜 왕복을 주던 문제
4. _compute_returns가 에이전트 경계를 넘어 할인하던 문제
5. 마스크 샘플링과 마스크 없는 로그 확률 재계산의 불일치
6. 종료 후 step() 재호출 시 종료 보상이 반복 지급되던 문제
7. 세그먼트 중간 차량의 전 행동 허용 마스크 (무시되는 행동이 학습에 섞임)
"""

import numpy as np
import pytest

from pursuit_evasion_rl.road_pursuit.rewards import RoadRewardCalculator
from pursuit_evasion_rl.road_pursuit.road_network_2d import RoadNetwork2D
from pursuit_evasion_rl.road_pursuit.road_pursuit_env import RoadPursuitEnv
from pursuit_evasion_rl.road_pursuit.vehicle_model import VehicleState
from pursuit_evasion_rl.training.algorithms import MAPPOAlgorithm

GRID_CONFIG = {
    "grid_rows": 3,
    "grid_cols": 3,
    "block_size": 100.0,
    "speed_limit": 50.0,
    "num_dead_ends": 0,
    "boundary_ratio": 1.0,
    "seed": 0,
}


def make_network(**overrides) -> RoadNetwork2D:
    """테스트용 3x3 완전 그리드 네트워크를 만든다."""
    config = dict(GRID_CONFIG)
    config.update(overrides)
    return RoadNetwork2D(config)


def at_intersection(
    network: RoadNetwork2D, agent_id: str, intersection_id: int, agent_type: str
) -> VehicleState:
    """해당 교차로를 nearest_intersection으로 갖는 차량 상태를 만든다."""
    outgoing = network.get_outgoing_segments(intersection_id)
    assert outgoing, f"교차로 {intersection_id}에 outgoing 세그먼트가 없다"
    seg = outgoing[0]
    return VehicleState(
        agent_id=agent_id,
        segment_id=seg.segment_id,
        progress=0.0,  # progress < 0.5 → nearest = start_intersection
        speed=seg.speed_limit,
        agent_type=agent_type,
    )


# ---------------------------------------------------------------------------
# 1. 협력 보너스: 점유자 한정 + 에피소드당 1회 + 시간 페널티
# ---------------------------------------------------------------------------


def make_encirclement(network: RoadNetwork2D) -> dict[str, VehicleState]:
    """도망자를 중앙(4)에 두고 인접 교차로 3곳(1,3,5)을 경찰이 점유한 상태.

    police_far(교차로 0)는 4의 인접 교차로가 아니므로 포위 참여자가 아니다.
    """
    states = {
        "fugitive": at_intersection(network, "fugitive", 4, "fugitive"),
        "police_0": at_intersection(network, "police_0", 1, "police"),
        "police_1": at_intersection(network, "police_1", 3, "police"),
        "police_2": at_intersection(network, "police_2", 5, "police"),
        "police_far": at_intersection(network, "police_far", 0, "police"),
    }
    return states


def test_cooperation_bonus_only_pays_occupying_officers():
    network = make_network()
    calc = RoadRewardCalculator(
        network, {"shaping_scale": 0.0, "time_penalty": 0.0, "cooperation_bonus": 0.05}
    )
    states = make_encirclement(network)

    rewards = calc.compute_step_rewards(states, states, None)

    assert rewards["police_0"] == pytest.approx(0.05)
    assert rewards["police_1"] == pytest.approx(0.05)
    assert rewards["police_2"] == pytest.approx(0.05)
    # 포위에 참여하지 않은 경찰은 받지 않는다 (예전에는 전원 지급).
    assert rewards["police_far"] == pytest.approx(0.0)
    assert rewards["fugitive"] == pytest.approx(0.0)


def test_cooperation_bonus_is_one_shot_per_episode():
    network = make_network()
    calc = RoadRewardCalculator(
        network, {"shaping_scale": 0.0, "time_penalty": 0.0, "cooperation_bonus": 0.05}
    )
    states = make_encirclement(network)

    first = calc.compute_step_rewards(states, states, None)
    second = calc.compute_step_rewards(states, states, None)
    third = calc.compute_step_rewards(states, states, None)

    assert first["police_0"] == pytest.approx(0.05)
    # 포위를 유지해도 다시 지급되지 않는다 → 배회가 연금이 되지 않는다.
    assert second["police_0"] == pytest.approx(0.0)
    assert third["police_0"] == pytest.approx(0.0)

    # reset() 이후에는 새 에피소드이므로 다시 지급된다.
    calc.reset()
    after_reset = calc.compute_step_rewards(states, states, None)
    assert after_reset["police_0"] == pytest.approx(0.05)


def test_hovering_is_strictly_negative_after_bonus_is_paid():
    """포위를 유지한 채 배회하면 누적 보상이 음수가 되어야 한다."""
    network = make_network()
    calc = RoadRewardCalculator(
        network,
        {"shaping_scale": 0.0, "time_penalty": -0.01, "cooperation_bonus": 0.05},
    )
    states = make_encirclement(network)

    total = 0.0
    for _ in range(20):
        total += calc.compute_step_rewards(states, states, None)["police_0"]

    # 1회 +0.05, 이후 매 스텝 -0.01
    assert total == pytest.approx(0.05 - 0.01 * 20)
    assert total < 0.0


def test_time_penalty_applies_to_police_only_on_non_terminal_steps():
    network = make_network()
    calc = RoadRewardCalculator(
        network,
        {"shaping_scale": 0.0, "time_penalty": -0.01, "cooperation_bonus": 0.0},
    )
    states = {
        "fugitive": at_intersection(network, "fugitive", 4, "fugitive"),
        "police_0": at_intersection(network, "police_0", 0, "police"),
    }

    step = calc.compute_step_rewards(states, states, None)
    assert step["police_0"] == pytest.approx(-0.01)
    assert step["fugitive"] == pytest.approx(0.0)

    # 종료 스텝에는 시간 페널티가 붙지 않는다.
    terminal = calc.compute_step_rewards(states, states, "police_win")
    assert terminal["police_0"] == pytest.approx(1.0)
    assert terminal["fugitive"] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# 2. 도달 불가 sentinel 처리
# ---------------------------------------------------------------------------


def test_police_shaping_is_zero_when_target_unreachable():
    network = make_network()
    isolated = 8
    # 그래프에서 8번 교차로의 모든 간선을 끊어 경로 없음을 만든다.
    # (세그먼트 자체는 남아 있어 VehicleState는 그대로 유효하다.)
    network.graph.remove_edges_from(
        list(network.graph.in_edges(isolated)) + list(network.graph.out_edges(isolated))
    )
    assert network.shortest_path_hops(0, isolated) >= network.num_intersections

    calc = RoadRewardCalculator(network, {"shaping_scale": 0.1, "time_penalty": 0.0})

    prev = {
        "fugitive": at_intersection(network, "fugitive", isolated, "fugitive"),
        "police_0": at_intersection(network, "police_0", 0, "police"),
    }
    curr = {
        "fugitive": at_intersection(network, "fugitive", isolated, "fugitive"),
        "police_0": at_intersection(network, "police_0", 1, "police"),
    }

    # sentinel끼리 빼면 임의의 ±0.1 임펄스가 생기므로 0이어야 한다.
    assert calc._compute_shaping_police("police_0", curr, prev) == pytest.approx(0.0)


def test_fugitive_shaping_is_zero_when_all_boundaries_unreachable():
    network = make_network()
    # 내부 교차로(4)만 남기고 4에서 나가는 간선을 모두 끊는다.
    network.graph.remove_edges_from(list(network.graph.out_edges(4)))
    calc = RoadRewardCalculator(network, {"shaping_scale": 0.1, "time_penalty": 0.0})

    states = {"fugitive": at_intersection(network, "fugitive", 4, "fugitive")}
    assert calc._compute_shaping_fugitive(states, states) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3. potential 기반 경찰 shaping
# ---------------------------------------------------------------------------


def test_police_shaping_ignores_fugitive_movement():
    """경찰이 가만히 있으면 도망자가 움직여도 shaping은 0이어야 한다."""
    network = make_network()
    calc = RoadRewardCalculator(network, {"shaping_scale": 0.1, "time_penalty": 0.0})

    prev = {
        "fugitive": at_intersection(network, "fugitive", 8, "fugitive"),
        "police_0": at_intersection(network, "police_0", 0, "police"),
    }
    curr = {
        # 도망자만 이동 (8 → 5)
        "fugitive": at_intersection(network, "fugitive", 5, "fugitive"),
        "police_0": at_intersection(network, "police_0", 0, "police"),
    }

    assert calc._compute_shaping_police("police_0", curr, prev) == pytest.approx(0.0)


def test_round_trip_nets_negative_with_time_penalty():
    """A→B→A 왕복은 shaping이 정확히 상쇄되고, 시간 페널티만 남아야 한다."""
    network = make_network()
    calc = RoadRewardCalculator(
        network,
        {"shaping_scale": 0.1, "time_penalty": -0.01, "cooperation_bonus": 0.0},
    )

    fugitive = at_intersection(network, "fugitive", 8, "fugitive")
    at_a = {"fugitive": fugitive, "police_0": at_intersection(network, "police_0", 0, "police")}
    at_b = {"fugitive": fugitive, "police_0": at_intersection(network, "police_0", 1, "police")}

    r1 = calc.compute_step_rewards(at_b, at_a, None)["police_0"]  # A → B
    r2 = calc.compute_step_rewards(at_a, at_b, None)["police_0"]  # B → A

    # shaping은 상쇄되고 시간 페널티 2회만 남는다 → 왕복은 순손실
    assert r1 + r2 == pytest.approx(-0.02)
    assert r1 + r2 < 0.0


# ---------------------------------------------------------------------------
# 4. 궤적 경계를 인식하는 returns 계산
# ---------------------------------------------------------------------------


def make_algo(**overrides) -> MAPPOAlgorithm:
    params = {
        "obs_dim": 4,
        "action_dim": 3,
        "num_police": 2,
        "discount_factor": 0.5,
    }
    params.update(overrides)
    return MAPPOAlgorithm(**params)


def test_compute_returns_resets_at_trajectory_boundaries():
    algo = make_algo()
    rewards = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]
    dones = [False, False, True, False, False, True]

    returns = algo._compute_returns(rewards, dones)

    assert returns == pytest.approx([0.25, 0.5, 1.0, 0.25, 0.5, 1.0])


def test_compute_returns_without_dones_preserves_old_behavior():
    algo = make_algo()
    rewards = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0]

    returns = algo._compute_returns(rewards)

    # 경계가 없으면 뒤쪽 궤적의 보상이 앞쪽으로 새어 들어온다 (기존 동작).
    assert returns[2] == pytest.approx(1.125)
    assert returns == pytest.approx(algo._compute_returns(rewards, None))


def test_compute_returns_boundary_blocks_cross_agent_leakage():
    algo = make_algo()
    rewards = [0.0, 5.0]

    without_boundary = algo._compute_returns(rewards)
    with_boundary = algo._compute_returns(rewards, [True, False])

    assert without_boundary[0] == pytest.approx(2.5)
    assert with_boundary[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. 마스크 일관성 (샘플링 분포 == 로그 확률 재계산 분포)
# ---------------------------------------------------------------------------


def make_obs(obs_dim: int = 4) -> dict:
    return {"a": np.arange(obs_dim, dtype=np.float32)}


def test_get_log_prob_uses_action_mask():
    algo = make_algo(action_dim=4)
    obs = make_obs()
    mask = np.array([True, False, True, False])

    log_probs = [
        algo.get_log_prob("police_0", obs, a, action_mask=mask) for a in range(4)
    ]

    # 유효 행동의 확률 합 = 1
    valid_prob = np.exp(log_probs[0]) + np.exp(log_probs[2])
    assert valid_prob == pytest.approx(1.0, abs=1e-5)
    # 무효 행동은 사실상 확률 0
    assert np.exp(log_probs[1]) == pytest.approx(0.0, abs=1e-8)
    assert np.exp(log_probs[3]) == pytest.approx(0.0, abs=1e-8)

    # 마스크를 주지 않으면 (기존 동작) 무효 행동에도 확률이 남는다.
    unmasked = algo.get_log_prob("police_0", obs, 1)
    assert np.exp(unmasked) > 1e-3


def test_get_action_never_samples_masked_actions():
    algo = make_algo(action_dim=4)
    obs = make_obs()
    mask = np.array([True, False, True, False])

    sampled = {algo.get_action("police_0", obs, action_mask=mask) for _ in range(200)}

    assert sampled.issubset({0, 2})


def test_ppo_ratio_is_one_when_masks_match_sampling():
    """마스크로 뽑은 old_log_prob과 마스크로 재계산한 log_prob이 일치해야 한다."""
    algo = make_algo(action_dim=4)
    obs = make_obs()
    mask = np.array([True, False, True, False])

    actions = [algo.get_action("police_0", obs, action_mask=mask) for _ in range(8)]
    old_log_probs = [
        algo.get_log_prob("police_0", obs, a, action_mask=mask) for a in actions
    ]
    masks = np.asarray([mask] * len(actions))

    recomputed = [
        algo.get_log_prob("police_0", obs, a, action_mask=masks[i])
        for i, a in enumerate(actions)
    ]
    assert recomputed == pytest.approx(old_log_probs)

    # 마스크를 실어 학습 스텝이 정상 동작하는지 (NaN/에러 없이)
    metrics = algo.train_step(
        {
            "police_obs": [np.asarray(list(obs["a"]), dtype=np.float32)] * len(actions),
            "police_actions": actions,
            "police_rewards": [0.1] * len(actions),
            "police_old_log_probs": old_log_probs,
            "police_dones": [False] * (len(actions) - 1) + [True],
            "police_masks": masks,
        }
    )
    assert np.isfinite(metrics["police_policy_loss"])
    assert np.isfinite(metrics["police_value_loss"])


# ---------------------------------------------------------------------------
# 6~7. 환경: 종료 후 step 가드, 세그먼트 중간 행동 마스크
# ---------------------------------------------------------------------------


ENV_CONFIG = {
    "grid_rows": 4,
    "grid_cols": 4,
    "block_size": 100.0,
    "num_police": 2,
    "speed": 50.0,
    "capture_radius": 5.0,
    "max_steps": 5,
    "dt": 0.1,
    "fixed_max_degree": 5,
    "num_dead_ends": 0,
    "boundary_ratio": 1.0,
    "seed": 0,
}


def run_until_done(env: RoadPursuitEnv):
    """에피소드가 끝날 때까지 stay 행동으로 진행한다."""
    stay = env._fixed_max_degree
    actions = {aid: stay for aid in env.agent_ids}
    while True:
        obs, rewards, terminated, truncated, info = env.step(actions)
        aid = env.agent_ids[0]
        if terminated[aid] or truncated[aid]:
            return obs, rewards, terminated, truncated, info


def test_step_after_done_returns_zero_rewards():
    env = RoadPursuitEnv(dict(ENV_CONFIG))
    env.reset(seed=0)

    _, last_rewards, last_term, last_trunc, _ = run_until_done(env)
    assert any(abs(r) > 0.0 for r in last_rewards.values())

    stay = env._fixed_max_degree
    obs, rewards, terminated, truncated, info = env.step(
        {aid: stay for aid in env.agent_ids}
    )

    # 종료 보상이 재지급되지 않는다.
    assert all(r == 0.0 for r in rewards.values())
    # 종료 플래그는 유지된다.
    assert terminated == last_term
    assert truncated == last_trunc
    assert set(obs.keys()) == set(env.agent_ids)


def test_step_after_done_does_not_advance_state():
    env = RoadPursuitEnv(dict(ENV_CONFIG))
    env.reset(seed=1)
    run_until_done(env)

    step_before = env._current_step
    states_before = {
        aid: (s.segment_id, s.progress) for aid, s in env._vehicle_states.items()
    }

    stay = env._fixed_max_degree
    env.step({aid: stay for aid in env.agent_ids})

    assert env._current_step == step_before
    assert {
        aid: (s.segment_id, s.progress) for aid, s in env._vehicle_states.items()
    } == states_before


def test_mid_segment_mask_allows_only_stay():
    env = RoadPursuitEnv(dict(ENV_CONFIG))
    env.reset(seed=0)

    stay = env._fixed_max_degree
    masks = env.get_action_masks()

    for aid in env.agent_ids:
        state = env._vehicle_states[aid]
        if state.at_intersection or env._arrives_this_step(state):
            continue
        mask = masks[aid]
        # 행동이 무시되는 구간에서는 stay만 유효 → 무의미한 랜덤 행동 차단
        assert mask[stay]
        assert not mask[:stay].any()
        assert len(mask) == env._fixed_max_degree + 1


def test_mask_on_arrival_step_uses_upcoming_intersection():
    env = RoadPursuitEnv(dict(ENV_CONFIG))
    env.reset(seed=0)

    aid = env.police_ids[0]
    state = env._vehicle_states[aid]
    segment = env.network.get_segment(state.segment_id)
    delta = state.speed * env._dt / segment.length
    # 이번 스텝에 도착하도록 progress를 맞춘다.
    env._vehicle_states[aid] = VehicleState(
        agent_id=state.agent_id,
        segment_id=state.segment_id,
        progress=1.0 - delta / 2.0,
        speed=state.speed,
        agent_type=state.agent_type,
    )

    mask = env.get_action_masks()[aid]
    end_inter = segment.end_intersection_id
    outgoing = env.network.get_outgoing_segments(end_inter)
    expected = min(len(outgoing), env._fixed_max_degree)

    assert mask[:expected].all()
    assert not mask[expected:env._fixed_max_degree].any()
    assert mask[env._fixed_max_degree]  # stay는 항상 허용
    assert len(mask) == env._fixed_max_degree + 1


# ---------------------------------------------------------------------------
# 8. generate_grid 중복 막다른 길 제거
# ---------------------------------------------------------------------------


def dead_end_count(network: RoadNetwork2D) -> int:
    boundary = set(network.get_boundary_intersections())
    return sum(
        1
        for inter in network.intersections.values()
        if len(inter.outgoing_segments) == 1 and inter.intersection_id not in boundary
    )


def test_generate_grid_only_applies_configured_dead_ends():
    """막다른 길 로직이 한 번만 돌아야 한다 (예전엔 중복 블록이 더 잘라냈다)."""
    rows = cols = 5
    network = make_network(grid_rows=rows, grid_cols=cols, num_dead_ends=2, seed=7)

    full_grid_segments = 2 * (rows * (cols - 1) + cols * (rows - 1))
    # 내부 교차로(차수 4) 2곳에서 3개씩 제거 → 정확히 6개만 사라진다.
    assert network.num_segments == full_grid_segments - 6

    # 중복 블록은 그래프 간선만 더 지우고 segments 딕셔너리는 놔뒀기 때문에
    # 세 값이 어긋났다. 이제는 일치해야 한다.
    assert network.graph.number_of_edges() == network.num_segments
    assert (
        sum(len(i.outgoing_segments) for i in network.intersections.values())
        == network.num_segments
    )
    assert dead_end_count(network) == 2


def test_generate_grid_dead_ends_follow_config_seed():
    """하드코딩 시드가 사라져, config seed가 실제로 결과를 바꿔야 한다."""
    same_a = make_network(grid_rows=5, grid_cols=5, num_dead_ends=2, seed=7)
    same_b = make_network(grid_rows=5, grid_cols=5, num_dead_ends=2, seed=7)
    other = make_network(grid_rows=5, grid_cols=5, num_dead_ends=2, seed=123)

    assert set(same_a.segments) == set(same_b.segments)
    assert set(same_a.segments) != set(other.segments)

    none_removed = make_network(grid_rows=5, grid_cols=5, num_dead_ends=0, seed=7)
    assert dead_end_count(none_removed) == 0
    assert none_removed.num_segments == 2 * (5 * 4 + 5 * 4)

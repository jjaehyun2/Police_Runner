"""ActionHandler 단위 테스트."""

import numpy as np
import pytest

from pursuit_evasion_rl.env.actions import ActionHandler
from pursuit_evasion_rl.env.road_network import RoadNetwork


@pytest.fixture
def simple_network():
    """간단한 고정 그래프 네트워크 (5 노드, 최대 차수 3).

    구조:
        0 -- 1 -- 2
        |    |
        3 -- 4

    노드별 차수:
        0: 2 (이웃: 1, 3)
        1: 3 (이웃: 0, 2, 4)
        2: 1 (이웃: 1)
        3: 2 (이웃: 0, 4)
        4: 2 (이웃: 1, 3)
    """
    config = {
        "network_mode": "fixed",
        "fixed_edges": [(0, 1), (1, 2), (0, 3), (3, 4), (1, 4)],
    }
    return RoadNetwork(config)


@pytest.fixture
def action_handler(simple_network):
    """ActionHandler 인스턴스 (max_degree=3)."""
    return ActionHandler(simple_network, simple_network.max_degree)


class TestBuildActionSpace:
    """build_action_space 메서드 테스트."""

    def test_action_space_size(self, action_handler):
        """행동 공간 크기는 max_degree + 1이어야 한다."""
        space = action_handler.build_action_space()
        assert space.n == action_handler.max_degree + 1

    def test_action_space_type(self, action_handler):
        """행동 공간은 Discrete 타입이어야 한다."""
        import gymnasium

        space = action_handler.build_action_space()
        assert isinstance(space, gymnasium.spaces.Discrete)

    def test_action_space_max_degree_3(self, action_handler):
        """max_degree=3인 경우 행동 공간 크기는 4."""
        space = action_handler.build_action_space()
        assert space.n == 4  # 3 neighbor slots + 1 stay


class TestExecuteAction:
    """execute_action 메서드 테스트."""

    def test_move_to_valid_neighbor(self, action_handler):
        """유효한 인접 노드 인덱스 선택 시 해당 노드로 이동."""
        # 노드 1의 정렬된 이웃: [0, 2, 4]
        # action=0 → 노드 0으로 이동
        new_pos = action_handler.execute_action("police_0", 0, 1)
        assert new_pos == 0

    def test_move_to_second_neighbor(self, action_handler):
        """두 번째 인접 노드로의 이동."""
        # 노드 1의 정렬된 이웃: [0, 2, 4]
        # action=1 → 노드 2로 이동
        new_pos = action_handler.execute_action("police_0", 1, 1)
        assert new_pos == 2

    def test_move_to_third_neighbor(self, action_handler):
        """세 번째 인접 노드로의 이동."""
        # 노드 1의 정렬된 이웃: [0, 2, 4]
        # action=2 → 노드 4로 이동
        new_pos = action_handler.execute_action("police_0", 2, 1)
        assert new_pos == 4

    def test_stay_action(self, action_handler):
        """stay 행동 선택 시 현재 위치 유지."""
        # stay_index = max_degree = 3
        new_pos = action_handler.execute_action("police_0", 3, 1)
        assert new_pos == 1

    def test_invalid_action_stays(self, action_handler):
        """무효한 행동 선택 시 현재 위치 유지."""
        # 노드 0의 이웃: [1, 3] (2개), action=2는 무효 (< max_degree=3)
        new_pos = action_handler.execute_action("police_0", 2, 0)
        assert new_pos == 0

    def test_stay_from_leaf_node(self, action_handler):
        """말단 노드에서 stay 행동."""
        # 노드 2의 이웃: [1] (1개)
        new_pos = action_handler.execute_action("fugitive", 3, 2)
        assert new_pos == 2

    def test_invalid_action_on_leaf_node(self, action_handler):
        """말단 노드에서 무효한 행동 (인덱스 1, 2)."""
        # 노드 2의 이웃: [1] (1개), action=1은 무효
        new_pos = action_handler.execute_action("fugitive", 1, 2)
        assert new_pos == 2


class TestGetActionMask:
    """get_action_mask 메서드 테스트."""

    def test_mask_size(self, action_handler):
        """마스크 크기는 max_degree + 1이어야 한다."""
        mask = action_handler.get_action_mask(0)
        assert len(mask) == action_handler.max_degree + 1

    def test_mask_dtype(self, action_handler):
        """마스크는 boolean 타입이어야 한다."""
        mask = action_handler.get_action_mask(0)
        assert mask.dtype == np.bool_

    def test_stay_always_true(self, action_handler):
        """stay 인덱스는 항상 True."""
        for node in range(5):
            mask = action_handler.get_action_mask(node)
            assert mask[action_handler.stay_index] is np.bool_(True)

    def test_mask_for_max_degree_node(self, action_handler):
        """최대 차수 노드 (노드 1, 차수 3)의 마스크: 모두 True."""
        # 노드 1 이웃: [0, 2, 4] → 인덱스 0,1,2 True, stay True
        mask = action_handler.get_action_mask(1)
        expected = np.array([True, True, True, True], dtype=np.bool_)
        np.testing.assert_array_equal(mask, expected)

    def test_mask_for_degree_2_node(self, action_handler):
        """차수 2 노드 (노드 0)의 마스크."""
        # 노드 0 이웃: [1, 3] → 인덱스 0,1 True, 인덱스 2 False, stay True
        mask = action_handler.get_action_mask(0)
        expected = np.array([True, True, False, True], dtype=np.bool_)
        np.testing.assert_array_equal(mask, expected)

    def test_mask_for_leaf_node(self, action_handler):
        """말단 노드 (노드 2, 차수 1)의 마스크."""
        # 노드 2 이웃: [1] → 인덱스 0 True, 인덱스 1,2 False, stay True
        mask = action_handler.get_action_mask(2)
        expected = np.array([True, False, False, True], dtype=np.bool_)
        np.testing.assert_array_equal(mask, expected)


class TestIsValidAction:
    """is_valid_action 메서드 테스트."""

    def test_valid_neighbor_action(self, action_handler):
        """유효한 인접 노드 인덱스는 True."""
        # 노드 1 이웃: [0, 2, 4], 인덱스 0, 1, 2 유효
        assert action_handler.is_valid_action(0, 1) is True
        assert action_handler.is_valid_action(1, 1) is True
        assert action_handler.is_valid_action(2, 1) is True

    def test_stay_always_valid(self, action_handler):
        """stay 행동은 항상 유효."""
        for node in range(5):
            assert action_handler.is_valid_action(action_handler.stay_index, node) is True

    def test_invalid_action_beyond_neighbors(self, action_handler):
        """인접 노드 수를 초과하는 인덱스는 False."""
        # 노드 0 이웃: [1, 3] (2개), 인덱스 2는 무효 (< stay_index=3)
        assert action_handler.is_valid_action(2, 0) is False

    def test_invalid_action_on_leaf(self, action_handler):
        """말단 노드에서 무효 행동 확인."""
        # 노드 2 이웃: [1] (1개), 인덱스 1, 2는 무효
        assert action_handler.is_valid_action(1, 2) is False
        assert action_handler.is_valid_action(2, 2) is False

    def test_first_action_always_valid_for_connected_node(self, action_handler):
        """연결된 노드에서 인덱스 0은 항상 유효 (최소 1개 이웃)."""
        for node in range(5):
            assert action_handler.is_valid_action(0, node) is True

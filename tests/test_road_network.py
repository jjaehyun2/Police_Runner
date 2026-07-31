"""RoadNetwork 클래스 단위 테스트 모듈."""

import pytest
import networkx as nx

from pursuit_evasion_rl.env.road_network import (
    NetworkGenerationError,
    RoadNetwork,
)


class TestRoadNetworkRandomMode:
    """랜덤 모드 그래프 생성 테스트."""

    def test_random_graph_is_connected(self):
        """랜덤 모드로 생성된 그래프가 연결 그래프인지 확인."""
        config = {"network_mode": "random", "num_nodes": 10, "density": 0.5}
        network = RoadNetwork(config, seed=42)
        assert nx.is_connected(network.graph)

    def test_random_graph_is_undirected(self):
        """랜덤 모드로 생성된 그래프가 무향 그래프인지 확인."""
        config = {"network_mode": "random", "num_nodes": 10, "density": 0.5}
        network = RoadNetwork(config, seed=42)
        assert not network.graph.is_directed()

    def test_random_graph_node_count_matches(self):
        """생성된 그래프의 노드 수가 설정값과 일치하는지 확인."""
        config = {"network_mode": "random", "num_nodes": 15, "density": 0.4}
        network = RoadNetwork(config, seed=42)
        assert network.num_nodes == 15
        assert network.graph.number_of_nodes() == 15

    def test_random_graph_with_seed_reproducibility(self):
        """동일 seed로 생성 시 동일 그래프가 생성되는지 확인."""
        config = {"network_mode": "random", "num_nodes": 10, "density": 0.5}
        net1 = RoadNetwork(config, seed=123)
        net2 = RoadNetwork(config, seed=123)
        assert set(net1.graph.edges()) == set(net2.graph.edges())
        assert set(net1.graph.nodes()) == set(net2.graph.nodes())

    def test_random_graph_min_nodes(self):
        """최소 노드 수(4)로 그래프 생성 확인."""
        config = {"network_mode": "random", "num_nodes": 4, "density": 0.8}
        network = RoadNetwork(config, seed=42)
        assert network.num_nodes == 4
        assert nx.is_connected(network.graph)

    def test_random_graph_max_nodes(self):
        """최대 노드 수(200)로 그래프 생성 확인."""
        config = {"network_mode": "random", "num_nodes": 200, "density": 0.1}
        network = RoadNetwork(config, seed=42)
        assert network.num_nodes == 200
        assert nx.is_connected(network.graph)

    def test_random_graph_density_one(self):
        """density=1.0 (완전 그래프)으로 생성 확인."""
        config = {"network_mode": "random", "num_nodes": 5, "density": 1.0}
        network = RoadNetwork(config, seed=42)
        assert network.num_nodes == 5
        # 완전 그래프는 n*(n-1)/2 엣지
        assert network.graph.number_of_edges() == 10


class TestRoadNetworkFixedMode:
    """고정 모드 그래프 생성 테스트."""

    def test_fixed_graph_creation(self):
        """고정 엣지 리스트로 그래프 생성 확인."""
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        config = {"network_mode": "fixed", "fixed_edges": edges}
        network = RoadNetwork(config)
        assert network.graph.number_of_nodes() == 4
        assert network.graph.number_of_edges() == 4

    def test_fixed_graph_idempotent(self):
        """동일 설정으로 고정 그래프를 여러 번 생성하면 동일 그래프 확인."""
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        config = {"network_mode": "fixed", "fixed_edges": edges}
        net1 = RoadNetwork(config)
        net2 = RoadNetwork(config)
        assert set(net1.graph.edges()) == set(net2.graph.edges())
        assert set(net1.graph.nodes()) == set(net2.graph.nodes())

    def test_fixed_graph_no_edges_raises_error(self):
        """고정 모드에서 엣지 리스트 미제공 시 ValueError 발생."""
        config = {"network_mode": "fixed", "fixed_edges": None}
        with pytest.raises(ValueError, match="fixed_edges가 제공되어야"):
            RoadNetwork(config)

    def test_fixed_graph_empty_edges_raises_error(self):
        """고정 모드에서 빈 엣지 리스트 시 ValueError 발생."""
        config = {"network_mode": "fixed", "fixed_edges": []}
        with pytest.raises(ValueError, match="fixed_edges가 제공되어야"):
            RoadNetwork(config)


class TestParameterValidation:
    """파라미터 범위 검증 테스트."""

    def test_num_nodes_below_min_raises_error(self):
        """num_nodes < 4일 때 ValueError 발생."""
        config = {"network_mode": "random", "num_nodes": 3, "density": 0.5}
        with pytest.raises(ValueError, match="num_nodes"):
            RoadNetwork(config, seed=42)

    def test_num_nodes_above_max_raises_error(self):
        """num_nodes > 200일 때 ValueError 발생."""
        config = {"network_mode": "random", "num_nodes": 201, "density": 0.5}
        with pytest.raises(ValueError, match="num_nodes"):
            RoadNetwork(config, seed=42)

    def test_density_zero_raises_error(self):
        """density == 0.0일 때 ValueError 발생."""
        config = {"network_mode": "random", "num_nodes": 10, "density": 0.0}
        with pytest.raises(ValueError, match="density"):
            RoadNetwork(config, seed=42)

    def test_density_negative_raises_error(self):
        """density < 0일 때 ValueError 발생."""
        config = {"network_mode": "random", "num_nodes": 10, "density": -0.1}
        with pytest.raises(ValueError, match="density"):
            RoadNetwork(config, seed=42)

    def test_density_above_one_raises_error(self):
        """density > 1.0일 때 ValueError 발생."""
        config = {"network_mode": "random", "num_nodes": 10, "density": 1.1}
        with pytest.raises(ValueError, match="density"):
            RoadNetwork(config, seed=42)


class TestBoundaryNodes:
    """경계 노드 식별 테스트."""

    def test_boundary_nodes_exist(self):
        """생성된 네트워크에 최소 1개의 경계 노드가 존재."""
        config = {"network_mode": "random", "num_nodes": 20, "density": 0.3}
        network = RoadNetwork(config, seed=42)
        assert len(network.boundary_nodes) >= 1

    def test_boundary_nodes_degree_constraint(self):
        """경계 노드의 degree가 2 이하인지 확인."""
        config = {"network_mode": "random", "num_nodes": 20, "density": 0.2}
        network = RoadNetwork(config, seed=42)
        for node in network.boundary_nodes:
            assert network.graph.degree(node) <= 2

    def test_boundary_nodes_centrality_constraint(self):
        """경계 노드의 closeness centrality가 하위 25%인지 확인.

        Note: 조건을 만족하는 노드가 없어 강제 지정된 경우는 이 테스트에서 제외.
        """
        config = {"network_mode": "random", "num_nodes": 30, "density": 0.15}
        network = RoadNetwork(config, seed=42)

        centrality = nx.closeness_centrality(network.graph)
        centrality_values = sorted(centrality.values())
        threshold_idx = max(0, len(centrality_values) // 4 - 1)
        threshold = centrality_values[threshold_idx]

        # 경계 노드가 2개 이상이면 정상 식별된 것 (forced 아님)
        if len(network.boundary_nodes) > 1:
            for node in network.boundary_nodes:
                assert centrality[node] <= threshold

    def test_fixed_linear_graph_boundary_nodes(self):
        """선형 그래프에서 양 끝 노드가 경계 노드로 식별되는지 확인."""
        # 0 - 1 - 2 - 3 - 4: 양 끝(0, 4)은 degree=1, centrality 낮음
        edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        config = {"network_mode": "fixed", "fixed_edges": edges}
        network = RoadNetwork(config)
        # 양 끝 노드는 degree=1이므로 boundary 후보
        assert 0 in network.boundary_nodes or 4 in network.boundary_nodes


class TestUtilityMethods:
    """유틸리티 메서드 테스트."""

    def _create_simple_network(self) -> RoadNetwork:
        """테스트용 간단한 네트워크 생성 (0-1-2-3-4, 추가 0-2)."""
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2)]
        config = {"network_mode": "fixed", "fixed_edges": edges}
        return RoadNetwork(config)

    def test_get_neighbors(self):
        """get_neighbors가 올바른 인접 노드를 반환하는지 확인."""
        network = self._create_simple_network()
        neighbors = network.get_neighbors(2)
        assert set(neighbors) == {1, 3, 0}

    def test_get_neighbors_leaf_node(self):
        """리프 노드의 인접 노드가 1개인지 확인."""
        network = self._create_simple_network()
        neighbors = network.get_neighbors(4)
        assert neighbors == [3]

    def test_shortest_path_length(self):
        """최단 경로 거리가 올바른지 확인."""
        network = self._create_simple_network()
        assert network.shortest_path_length(0, 4) == 3
        assert network.shortest_path_length(0, 2) == 1
        assert network.shortest_path_length(0, 0) == 0

    def test_shortest_path(self):
        """최단 경로 노드 리스트가 올바른지 확인."""
        network = self._create_simple_network()
        path = network.shortest_path(0, 4)
        # 0 -> 2 -> 3 -> 4 또는 0 -> 1 -> 2 -> 3 -> 4 중 최단
        assert path[0] == 0
        assert path[-1] == 4
        assert len(path) == 4  # 0, 2, 3, 4

    def test_shortest_path_same_node(self):
        """동일 노드 간 최단 경로는 자기 자신만 포함."""
        network = self._create_simple_network()
        path = network.shortest_path(2, 2)
        assert path == [2]


class TestMaxDegree:
    """max_degree 속성 테스트."""

    def test_max_degree_complete_graph(self):
        """완전 그래프의 max_degree 확인."""
        config = {"network_mode": "random", "num_nodes": 5, "density": 1.0}
        network = RoadNetwork(config, seed=42)
        # 완전 그래프: 각 노드의 degree = n-1 = 4
        assert network.max_degree == 4

    def test_max_degree_linear_graph(self):
        """선형 그래프의 max_degree 확인."""
        edges = [(0, 1), (1, 2), (2, 3)]
        config = {"network_mode": "fixed", "fixed_edges": edges}
        network = RoadNetwork(config)
        # 선형 그래프: 중간 노드 degree=2
        assert network.max_degree == 2


class TestNetworkGenerationError:
    """NetworkGenerationError 예외 테스트."""

    def test_very_low_density_raises_error(self):
        """극도로 낮은 density로 연결 그래프 생성 실패 시 예외 발생 확인.

        Note: density가 매우 낮으면 100회 안에 연결 그래프가 생성되지 않을 수 있다.
        이 테스트는 약간의 확률적 요소가 있으므로 적절한 파라미터를 선택한다.
        """
        # density=0.001, num_nodes=100이면 거의 연결되지 않음
        config = {"network_mode": "random", "num_nodes": 100, "density": 0.001}
        with pytest.raises(NetworkGenerationError):
            RoadNetwork(config, seed=42)

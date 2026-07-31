"""휴리스틱 에이전트 구현 모듈.

HeuristicPoliceAgent와 HeuristicFugitiveAgent를 제공한다.
각각 최단 경로 기반 추격/탈출 전략을 사용한다.
"""

from __future__ import annotations

import networkx as nx

from pursuit_evasion_rl.env.road_network import RoadNetwork


class HeuristicPoliceAgent:
    """최단 경로 기반 추격 휴리스틱 경찰 에이전트.

    도망자 방향으로 최단 경로를 따라 이동한다.
    경로가 존재하지 않으면 정지(stay) 행동을 선택한다.

    Attributes:
        network: 도로 네트워크 인스턴스
    """

    def __init__(self, network: RoadNetwork) -> None:
        """HeuristicPoliceAgent 초기화.

        Args:
            network: 도로 네트워크 인스턴스
        """
        self.network = network

    def act(self, current_node: int, fugitive_pos: int) -> int:
        """도망자 방향 최단 경로의 다음 노드에 해당하는 행동 인덱스를 반환한다.

        networkx shortest_path를 사용하여 현재 위치에서 도망자까지의
        최단 경로를 계산하고, 경로의 두 번째 노드(다음 이동 노드)에 해당하는
        행동 인덱스를 반환한다.

        Args:
            current_node: 경찰 에이전트의 현재 노드 ID
            fugitive_pos: 도망자의 현재 노드 ID

        Returns:
            행동 인덱스 (sorted neighbors 리스트 내 위치 또는 stay 인덱스)
        """
        # 이미 도망자와 같은 위치에 있으면 stay
        if current_node == fugitive_pos:
            return self.network.max_degree

        try:
            path = nx.shortest_path(
                self.network.graph, source=current_node, target=fugitive_pos
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return self.network.max_degree

        # path[0]은 current_node, path[1]은 다음 이동할 노드
        next_node = path[1]

        # sorted neighbors에서 next_node의 인덱스를 찾아 반환
        neighbors = sorted(self.network.get_neighbors(current_node))
        try:
            action_index = neighbors.index(next_node)
        except ValueError:
            return self.network.max_degree

        return action_index


class HeuristicFugitiveAgent:
    """최단 경로 기반 탈출 휴리스틱 도망자 에이전트.

    가장 가까운 경계 노드 방향으로 최단 경로를 따라 이동한다.
    동일 거리의 경계 노드가 여러 개이면 노드 ID가 가장 작은 것을 선택한다.
    경로가 존재하지 않으면 정지(stay) 행동을 선택한다.

    Attributes:
        network: 도로 네트워크 인스턴스
    """

    def __init__(self, network: RoadNetwork) -> None:
        """HeuristicFugitiveAgent 초기화.

        Args:
            network: 도로 네트워크 인스턴스
        """
        self.network = network

    def act(self, current_node: int) -> int:
        """가장 가까운 경계 노드 방향의 다음 노드에 해당하는 행동 인덱스를 반환한다.

        모든 경계 노드까지의 최단 경로 거리를 계산하고, 가장 가까운 경계 노드를
        선택한다. 동일 거리의 경계 노드가 여러 개이면 노드 ID가 가장 작은 것을
        선택한다. 선택된 경계 노드까지의 최단 경로에서 다음 이동 노드에 해당하는
        행동 인덱스를 반환한다.

        Args:
            current_node: 도망자의 현재 노드 ID

        Returns:
            행동 인덱스 (sorted neighbors 리스트 내 위치 또는 stay 인덱스)
        """
        # 이미 경계 노드에 있으면 stay
        if current_node in self.network.boundary_nodes:
            return self.network.max_degree

        # 모든 경계 노드까지의 거리를 계산
        best_distance = float("inf")
        best_boundary_node = None

        for boundary_node in self.network.boundary_nodes:
            try:
                distance = nx.shortest_path_length(
                    self.network.graph, source=current_node, target=boundary_node
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            # 더 가까운 경계 노드 발견 또는 동일 거리일 때 ID가 더 작은 노드
            if distance < best_distance or (
                distance == best_distance and boundary_node < best_boundary_node
            ):
                best_distance = distance
                best_boundary_node = boundary_node

        # 도달 가능한 경계 노드가 없으면 stay
        if best_boundary_node is None:
            return self.network.max_degree

        try:
            path = nx.shortest_path(
                self.network.graph, source=current_node, target=best_boundary_node
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return self.network.max_degree

        # path[0]은 current_node, path[1]은 다음 이동할 노드
        next_node = path[1]

        # sorted neighbors에서 next_node의 인덱스를 찾아 반환
        neighbors = sorted(self.network.get_neighbors(current_node))
        try:
            action_index = neighbors.index(next_node)
        except ValueError:
            return self.network.max_degree

        return action_index

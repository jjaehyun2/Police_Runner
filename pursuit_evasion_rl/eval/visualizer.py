"""시각화 모듈.

Visualizer 클래스를 제공한다.
matplotlib과 networkx를 사용하여 추격-도주 에피소드를 시각화한다.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx

from pursuit_evasion_rl.env.road_network import RoadNetwork


class Visualizer:
    """추격-도주 에피소드를 시각화하는 클래스.

    matplotlib를 사용하여 그래프 네트워크 위의 에이전트 위치를 렌더링한다.
    자동 재생 및 수동 스텝 모드를 지원한다.

    Attributes:
        network: 도로 네트워크 인스턴스
        render_backend: 렌더링 백엔드 ("matplotlib")
    """

    def __init__(self, network: RoadNetwork, render_backend: str = "matplotlib") -> None:
        """Visualizer 초기화.

        Args:
            network: 도로 네트워크 인스턴스
            render_backend: 렌더링 백엔드 (현재 "matplotlib"만 지원)
        """
        self.network = network
        self.render_backend = render_backend

        # 그래프 레이아웃 계산 (고정된 위치를 사용하여 프레임 간 일관성 유지)
        self._pos = nx.spring_layout(self.network.graph, seed=42)

    def render_frame(self, positions: dict[str, int], step: int) -> None:
        """단일 프레임을 렌더링한다.

        그래프 위에 에이전트 위치를 표시한다.
        노드 색상:
            - 경찰 위치: 파랑 (blue)
            - 도망자 위치: 빨강 (red)
            - 경계 노드: 초록 (green)
            - 기타 노드: 연회색 (lightgray)
        엣지 색상: 회색 (gray)

        Args:
            positions: 에이전트별 현재 노드 위치 딕셔너리
            step: 현재 스텝 번호
        """
        plt.clf()

        # 노드 색상 결정
        node_colors = self._get_node_colors(positions)

        # 그래프 그리기
        nx.draw(
            self.network.graph,
            pos=self._pos,
            node_color=node_colors,
            edge_color="gray",
            with_labels=True,
            node_size=300,
            font_size=8,
            font_color="black",
        )

        # 타이틀에 스텝 번호 표시
        plt.title(f"Step {step}")
        plt.draw()

    def replay_episode(
        self,
        history: list[dict],
        auto_play: bool = True,
        frame_interval_ms: int = 500,
    ) -> None:
        """에피소드를 프레임 단위로 재생한다.

        Args:
            history: 에피소드 기록 리스트. 각 항목은 다음 키를 가진 딕셔너리:
                - positions: dict[str, int] 에이전트별 위치
                - step: int 스텝 번호
            auto_play: True이면 자동 재생 (frame_interval_ms 간격),
                       False이면 수동 모드 (키 입력으로 진행)
            frame_interval_ms: 자동 재생 시 프레임 간격 (밀리초)
        """
        if not history:
            print("재생할 에피소드 기록이 없습니다.")
            return

        plt.figure(figsize=(8, 6))
        plt.ion()  # 인터랙티브 모드 활성화

        for frame in history:
            positions = frame["positions"]
            step = frame["step"]

            self.render_frame(positions, step)

            if auto_play:
                plt.pause(frame_interval_ms / 1000.0)
            else:
                plt.waitforbuttonpress()

        plt.ioff()
        plt.show()

    def _get_node_colors(self, positions: dict[str, int]) -> list[str]:
        """각 노드의 표시 색상을 결정한다.

        우선순위: 도망자(red) > 경찰(blue) > 경계(green) > 기타(lightgray)

        Args:
            positions: 에이전트별 현재 노드 위치

        Returns:
            그래프 노드 순서에 맞는 색상 리스트
        """
        # 에이전트 위치를 역매핑
        police_nodes: set[int] = set()
        fugitive_node: int | None = None

        for agent_id, node in positions.items():
            if agent_id.startswith("police"):
                police_nodes.add(node)
            elif agent_id == "fugitive":
                fugitive_node = node

        boundary_set = set(self.network.boundary_nodes)

        colors: list[str] = []
        for node in self.network.graph.nodes():
            if fugitive_node is not None and node == fugitive_node:
                colors.append("red")
            elif node in police_nodes:
                colors.append("blue")
            elif node in boundary_set:
                colors.append("green")
            else:
                colors.append("lightgray")

        return colors

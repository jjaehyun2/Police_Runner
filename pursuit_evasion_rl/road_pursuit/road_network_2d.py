"""2D 방향 도로 네트워크 모듈.

RoadNetwork2D 클래스를 제공한다.
교차로(Intersection2D)와 도로 세그먼트(RoadSegment2D)로 구성된
방향 그래프 기반 도로 네트워크를 생성하고 관리한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import networkx as nx
import numpy as np


@dataclass
class RoadSegment2D:
    """방향 있는 도로 세그먼트.

    Attributes:
        segment_id: 세그먼트 고유 ID
        start_intersection_id: 출발 교차로 ID
        end_intersection_id: 도착 교차로 ID
        start_pos: 출발점 (x, y) 좌표
        end_pos: 도착점 (x, y) 좌표
        length: 세그먼트 길이 (유클리드 거리)
        width: 도로 폭 (미터)
        speed_limit: 속도 제한 (단위/스텝)
        lanes: 차선 수
    """

    segment_id: int
    start_intersection_id: int
    end_intersection_id: int
    start_pos: tuple[float, float]
    end_pos: tuple[float, float]
    length: float = 0.0
    width: float = 10.0
    speed_limit: float = 50.0
    lanes: int = 2

    def __post_init__(self) -> None:
        """길이를 좌표로부터 자동 계산."""
        if self.length == 0.0:
            dx = self.end_pos[0] - self.start_pos[0]
            dy = self.end_pos[1] - self.start_pos[1]
            self.length = math.sqrt(dx * dx + dy * dy)

    @property
    def direction(self) -> tuple[float, float]:
        """정규화된 방향 벡터 (dx, dy)."""
        dx = self.end_pos[0] - self.start_pos[0]
        dy = self.end_pos[1] - self.start_pos[1]
        mag = math.sqrt(dx * dx + dy * dy)
        if mag < 1e-9:
            return (0.0, 0.0)
        return (dx / mag, dy / mag)


@dataclass
class Intersection2D:
    """교차로 노드.

    Attributes:
        intersection_id: 교차로 고유 ID
        position: (x, y) 좌표
        outgoing_segments: 이 교차로에서 나가는 세그먼트 ID 목록
        incoming_segments: 이 교차로로 들어오는 세그먼트 ID 목록
    """

    intersection_id: int
    position: tuple[float, float]
    outgoing_segments: list[int] = field(default_factory=list)
    incoming_segments: list[int] = field(default_factory=list)


class RoadNetwork2D:
    """2D 방향 도로 네트워크.

    교차로와 도로 세그먼트로 구성된 방향 그래프를 관리한다.
    그리드 형태의 네트워크 생성을 지원한다.

    Attributes:
        intersections: 교차로 딕셔너리 {id: Intersection2D}
        segments: 세그먼트 딕셔너리 {id: RoadSegment2D}
        graph: networkx DiGraph (최단 경로 계산용)
    """

    def __init__(self, config: dict | None = None) -> None:
        """RoadNetwork2D 초기화.

        Args:
            config: 설정 딕셔너리. 다음 키를 지원:
                - grid_rows: 그리드 행 수 (기본 4)
                - grid_cols: 그리드 열 수 (기본 4)
                - block_size: 블록 크기 미터 (기본 100.0)
                - speed_limit: 속도 제한 (기본 50.0)
                - road_width: 도로 폭 (기본 10.0)
        """
        if config is None:
            config = {}

        self._config = config
        self.intersections: dict[int, Intersection2D] = {}
        self.segments: dict[int, RoadSegment2D] = {}
        self.graph: nx.DiGraph = nx.DiGraph()
        self._boundary_intersections: list[int] = []

        # 그리드 생성
        rows = config.get("grid_rows", 4)
        cols = config.get("grid_cols", 4)
        block_size = config.get("block_size", 100.0)
        speed_limit = config.get("speed_limit", 50.0)
        road_width = config.get("road_width", 10.0)

        self.generate_grid(rows, cols, block_size, speed_limit, road_width)

    def generate_grid(
        self,
        rows: int,
        cols: int,
        block_size: float = 100.0,
        speed_limit: float = 50.0,
        road_width: float = 10.0,
    ) -> None:
        """그리드 형태의 도로 네트워크를 생성한다.

        rows x cols 격자 교차로를 생성하고, 인접한 교차로 간에
        양방향 도로 세그먼트(각 방향 별도)를 연결한다.

        Args:
            rows: 행 수
            cols: 열 수
            block_size: 교차로 간 거리 (미터)
            speed_limit: 속도 제한
            road_width: 도로 폭
        """
        self.intersections.clear()
        self.segments.clear()
        self.graph = nx.DiGraph()
        self._boundary_intersections = []

        # 1. 교차로 생성
        for r in range(rows):
            for c in range(cols):
                iid = r * cols + c
                pos = (c * block_size, r * block_size)
                self.intersections[iid] = Intersection2D(
                    intersection_id=iid,
                    position=pos,
                )
                self.graph.add_node(iid, pos=pos)

                # 경계 교차로: 그리드 테두리에 위치
                if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                    self._boundary_intersections.append(iid)

        # 2. 도로 세그먼트 생성 (양방향: 각 방향 별도 세그먼트)
        segment_id = 0
        for r in range(rows):
            for c in range(cols):
                current_id = r * cols + c
                current_pos = self.intersections[current_id].position

                # 오른쪽 연결
                if c < cols - 1:
                    right_id = r * cols + (c + 1)
                    right_pos = self.intersections[right_id].position

                    # current -> right
                    seg = RoadSegment2D(
                        segment_id=segment_id,
                        start_intersection_id=current_id,
                        end_intersection_id=right_id,
                        start_pos=current_pos,
                        end_pos=right_pos,
                        width=road_width,
                        speed_limit=speed_limit,
                    )
                    self.segments[segment_id] = seg
                    self.intersections[current_id].outgoing_segments.append(segment_id)
                    self.intersections[right_id].incoming_segments.append(segment_id)
                    self.graph.add_edge(current_id, right_id, segment_id=segment_id, weight=seg.length)
                    segment_id += 1

                    # right -> current
                    seg = RoadSegment2D(
                        segment_id=segment_id,
                        start_intersection_id=right_id,
                        end_intersection_id=current_id,
                        start_pos=right_pos,
                        end_pos=current_pos,
                        width=road_width,
                        speed_limit=speed_limit,
                    )
                    self.segments[segment_id] = seg
                    self.intersections[right_id].outgoing_segments.append(segment_id)
                    self.intersections[current_id].incoming_segments.append(segment_id)
                    self.graph.add_edge(right_id, current_id, segment_id=segment_id, weight=seg.length)
                    segment_id += 1

                # 위쪽 연결
                if r < rows - 1:
                    up_id = (r + 1) * cols + c
                    up_pos = self.intersections[up_id].position

                    # current -> up
                    seg = RoadSegment2D(
                        segment_id=segment_id,
                        start_intersection_id=current_id,
                        end_intersection_id=up_id,
                        start_pos=current_pos,
                        end_pos=up_pos,
                        width=road_width,
                        speed_limit=speed_limit,
                    )
                    self.segments[segment_id] = seg
                    self.intersections[current_id].outgoing_segments.append(segment_id)
                    self.intersections[up_id].incoming_segments.append(segment_id)
                    self.graph.add_edge(current_id, up_id, segment_id=segment_id, weight=seg.length)
                    segment_id += 1

                    # up -> current
                    seg = RoadSegment2D(
                        segment_id=segment_id,
                        start_intersection_id=up_id,
                        end_intersection_id=current_id,
                        start_pos=up_pos,
                        end_pos=current_pos,
                        width=road_width,
                        speed_limit=speed_limit,
                    )
                    self.segments[segment_id] = seg
                    self.intersections[up_id].outgoing_segments.append(segment_id)
                    self.intersections[current_id].incoming_segments.append(segment_id)
                    self.graph.add_edge(up_id, current_id, segment_id=segment_id, weight=seg.length)
                    segment_id += 1

        # 3. 탈출구 랜덤 절반 축소 (경계 20개 → 약 10개)
        boundary_ratio = self._config.get("boundary_ratio", 1.0)
        boundary_seed = self._config.get("seed", None)
        if boundary_ratio < 1.0 and len(self._boundary_intersections) > 4:
            rng_b = np.random.default_rng(boundary_seed)
            all_boundary = list(self._boundary_intersections)
            num_keep = max(4, int(len(all_boundary) * boundary_ratio))
            kept = list(rng_b.choice(all_boundary, size=num_keep, replace=False))
            self._boundary_intersections = sorted(int(k) for k in kept)

        # 4. 막다른 길 추가 (내부 교차로 일부에서 outgoing 세그먼트 제거)
        self._add_dead_ends(self._config.get("num_dead_ends", 2), self._config.get("seed", None))

        # 4. 막다른 길 추가 (내부 교차로 1~2개에서 outgoing 세그먼트 일부 제거)
        internal = [
            iid for iid in self.intersections
            if iid not in set(self._boundary_intersections)
            and len(self.intersections[iid].outgoing_segments) > 1
        ]
        if internal:
            rng2 = np.random.default_rng(rows * cols + 1)
            num_dead_ends = min(2, len(internal))
            dead_end_candidates = rng2.choice(internal, size=num_dead_ends, replace=False)

            for dead_iid in dead_end_candidates:
                inter = self.intersections[dead_iid]
                # outgoing 세그먼트 중 1~2개 제거 (최소 1개는 남김)
                if len(inter.outgoing_segments) > 1:
                    num_remove = min(2, len(inter.outgoing_segments) - 1)
                    to_remove = rng2.choice(
                        inter.outgoing_segments, size=num_remove, replace=False
                    ).tolist()
                    for sid in to_remove:
                        seg = self.segments[sid]
                        # 그래프에서 엣지 제거
                        if self.graph.has_edge(seg.start_intersection_id, seg.end_intersection_id):
                            self.graph.remove_edge(seg.start_intersection_id, seg.end_intersection_id)
                        # outgoing 목록에서 제거
                        inter.outgoing_segments.remove(sid)
                        # incoming 목록에서도 제거
                        end_inter = self.intersections[seg.end_intersection_id]
                        if sid in end_inter.incoming_segments:
                            end_inter.incoming_segments.remove(sid)
                        # 세그먼트 자체는 보존 (ID 참조 깨짐 방지)

    def _reduce_boundary_exits(self, keep_ratio: float, seed: int | None = None) -> None:
        """경계 교차로 중 일부만 탈출구로 유지한다.

        Args:
            keep_ratio: 유지할 비율 (0.5 = 절반)
            seed: 랜덤 시드
        """
        if not self._boundary_intersections or keep_ratio >= 1.0:
            return

        rng = np.random.default_rng(seed)
        all_boundary = list(self._boundary_intersections)
        num_keep = max(4, int(len(all_boundary) * keep_ratio))  # 최소 4개, 비율 적용

        kept = list(rng.choice(all_boundary, size=num_keep, replace=False))
        self._boundary_intersections = sorted(kept)

    def _add_dead_ends(self, num_dead_ends: int, seed: int | None = None) -> None:
        """내부 교차로 일부를 막다른 길로 만든다.

        선택된 교차로에서 outgoing 세그먼트를 1개만 남기고 나머지를 제거한다.
        (해당 교차로로 들어오는 incoming은 유지 → 들어올 수는 있지만 나갈 길이 1개뿐)

        Args:
            num_dead_ends: 막다른 길로 만들 교차로 수
            seed: 랜덤 시드
        """
        if num_dead_ends <= 0:
            return

        rng = np.random.default_rng(seed if seed is not None else 42)

        # 내부 교차로만 (경계 제외)
        boundary_set = set(self._boundary_intersections)
        internal = [
            iid for iid in self.intersections
            if iid not in boundary_set and len(self.intersections[iid].outgoing_segments) > 1
        ]

        if not internal:
            return

        num_to_block = min(num_dead_ends, len(internal))
        chosen = list(rng.choice(internal, size=num_to_block, replace=False))

        for iid in chosen:
            inter = self.intersections[iid]
            outgoing = list(inter.outgoing_segments)

            if len(outgoing) <= 1:
                continue

            # 1개만 남기고 나머지 제거
            keep_seg_id = outgoing[0]
            remove_seg_ids = outgoing[1:]

            for seg_id in remove_seg_ids:
                seg = self.segments[seg_id]

                # outgoing에서 제거
                inter.outgoing_segments.remove(seg_id)

                # 도착 교차로의 incoming에서 제거
                dest_inter = self.intersections[seg.end_intersection_id]
                if seg_id in dest_inter.incoming_segments:
                    dest_inter.incoming_segments.remove(seg_id)

                # networkx 그래프에서 엣지 제거
                if self.graph.has_edge(seg.start_intersection_id, seg.end_intersection_id):
                    self.graph.remove_edge(seg.start_intersection_id, seg.end_intersection_id)

                # 세그먼트 삭제
                del self.segments[seg_id]

    def get_intersection(self, intersection_id: int) -> Intersection2D:
        """교차로를 반환한다."""
        return self.intersections[intersection_id]

    def get_segment(self, segment_id: int) -> RoadSegment2D:
        """세그먼트를 반환한다."""
        return self.segments[segment_id]

    def get_outgoing_segments(self, intersection_id: int) -> list[RoadSegment2D]:
        """교차로에서 나가는 모든 세그먼트를 반환한다."""
        intersection = self.intersections[intersection_id]
        return [self.segments[sid] for sid in intersection.outgoing_segments if sid in self.segments]

    def get_boundary_intersections(self) -> list[int]:
        """경계 교차로 ID 목록을 반환한다."""
        return list(self._boundary_intersections)

    def shortest_path_hops(self, source: int, target: int) -> int:
        """두 교차로 간 최단 경로 홉 수를 반환한다.

        Args:
            source: 출발 교차로 ID
            target: 도착 교차로 ID

        Returns:
            최단 홉 수. 경로 없으면 num_intersections 반환.
        """
        try:
            return nx.shortest_path_length(self.graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return self.num_intersections

    @property
    def num_intersections(self) -> int:
        """교차로 수."""
        return len(self.intersections)

    @property
    def num_segments(self) -> int:
        """세그먼트 수."""
        return len(self.segments)

    @property
    def max_outgoing_degree(self) -> int:
        """모든 교차로 중 최대 outgoing 차수."""
        if not self.intersections:
            return 0
        return max(
            len(inter.outgoing_segments) for inter in self.intersections.values()
        )

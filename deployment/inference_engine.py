"""경찰 추격 RL 모델 추론 엔진.

학습된 MAPPO 모델을 로드하여 경찰 배치 추천을 제공한다.
training 코드의 MAPPOAlgorithm을 임포트하여 모델을 복원하고,
수동으로 관측값을 구성하여 추론을 수행한다.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# 독립 실행을 위한 경량 MLP + 모델 로더 (training 패키지 미설치 환경 대응)
# ──────────────────────────────────────────────────────────────────────


class _MLPNetwork(nn.Module):
    """추론 전용 MLP 네트워크 (학습 코드와 동일 구조)."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int] | None = None) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def _flatten_observation(observation: dict) -> np.ndarray:
    """관측값 딕셔너리를 1D numpy 배열로 평탄화한다."""
    parts = []
    for key in sorted(observation.keys()):
        arr = np.asarray(observation[key], dtype=np.float32).flatten()
        parts.append(arr)
    return np.concatenate(parts)


def _get_obs_dim(num_police: int, max_degree: int) -> int:
    """관측 공간의 총 차원을 계산한다.

    관측값 구조 (sorted key 순서):
        agent_id_onehot: (num_police,)
        current_step: (1,)
        my_position: (1,)
        my_progress: (1,)
        nearest_boundary_dist: (1,)
        neighbors: (max_degree,)
        other_positions: (num_police,)  # num_agents - 1 = num_police (도주자 관점에서는 num_police, 경찰 관점에서는 num_police)
    """
    # agent_id_onehot(num_police) + current_step(1) + my_position(1) + my_progress(1)
    # + nearest_boundary_dist(1) + neighbors(max_degree) + other_positions(num_police)
    return num_police + 1 + 1 + 1 + 1 + max_degree + num_police


class PursuitInferenceEngine:
    """학습된 모델을 사용하여 경찰 배치 추천을 제공하는 추론 엔진.

    학습된 MAPPO 체크포인트를 로드하고, 현재 상태에서
    각 경찰 에이전트의 최적 이동 방향을 추천한다.
    """

    def __init__(
        self,
        model_path: str,
        num_police: int = 4,
        fixed_max_degree: int = 5,
        hidden_dims: list[int] | None = None,
    ) -> None:
        """추론 엔진 초기화 및 모델 로드.

        Args:
            model_path: 학습된 모델 체크포인트 경로 (.pt 파일)
            num_police: 경찰 에이전트 수
            fixed_max_degree: 고정 최대 차수 (행동 공간 크기 결정)
            hidden_dims: MLP 히든 레이어 차원

        Raises:
            FileNotFoundError: 모델 파일을 찾을 수 없는 경우
        """
        self.num_police = num_police
        self.fixed_max_degree = fixed_max_degree
        self.action_dim = fixed_max_degree + 1  # 0~max_degree-1: 세그먼트 선택, max_degree: 대기

        if hidden_dims is None:
            hidden_dims = [64, 64]

        # 관측 차원 계산
        self.obs_dim = _get_obs_dim(num_police, fixed_max_degree)

        # Actor 네트워크 생성 (경찰 공유 정책만 사용)
        self.police_actor = _MLPNetwork(self.obs_dim, self.action_dim, hidden_dims)

        # 모델 로드
        self._load_model(model_path)
        self.police_actor.eval()

        self._model_path = model_path
        logger.info(
            "추론 엔진 초기화 완료: obs_dim=%d, action_dim=%d, num_police=%d",
            self.obs_dim, self.action_dim, self.num_police,
        )

    def _load_model(self, model_path: str) -> None:
        """모델 체크포인트를 로드한다.

        Args:
            model_path: .pt 체크포인트 경로

        Raises:
            FileNotFoundError: 파일 미존재
            RuntimeError: 체크포인트 형식 불일치
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        if "police_actor" not in checkpoint:
            raise RuntimeError(
                f"체크포인트에 'police_actor' 키가 없습니다. "
                f"사용 가능한 키: {list(checkpoint.keys())}"
            )

        self.police_actor.load_state_dict(checkpoint["police_actor"])
        logger.info("모델 로드 완료: %s", model_path)

    def recommend_actions(
        self,
        police_positions: list[int],
        fugitive_position: int,
        network_data: dict,
        current_step: int = 0,
    ) -> dict:
        """각 경찰에게 다음 이동 방향을 추천한다.

        Args:
            police_positions: 각 경찰의 현재 교차로 ID 리스트
            fugitive_position: 도주자의 현재/마지막 목격 교차로 ID
            network_data: 도로 네트워크 데이터 (osm_loader 형식)
            current_step: 현재 시뮬레이션 스텝 (기본 0)

        Returns:
            {
                "police_0": {
                    "action": int,
                    "next_intersection": int,
                    "direction": str,
                    "confidence": float
                },
                "police_1": {...},
                ...
            }
        """
        if len(police_positions) != self.num_police:
            raise ValueError(
                f"경찰 수 불일치: 예상 {self.num_police}, 입력 {len(police_positions)}"
            )

        # 네트워크 데이터 인덱싱 구조 생성
        intersection_map = self._build_intersection_map(network_data)
        segment_map = self._build_segment_map(network_data)
        boundary_set = set(network_data.get("boundary_intersections", []))

        recommendations = {}

        for police_idx in range(self.num_police):
            agent_id = f"police_{police_idx}"

            # 관측값 구성
            obs = self._build_observation(
                police_id=police_idx,
                police_positions=police_positions,
                fugitive_position=fugitive_position,
                network_data=network_data,
                intersection_map=intersection_map,
                segment_map=segment_map,
                boundary_set=boundary_set,
                current_step=current_step,
            )

            # 행동 마스크 생성
            action_mask = self._build_action_mask(
                police_positions[police_idx], intersection_map
            )

            # 추론
            action, confidence = self._infer_action(obs, action_mask)

            # 다음 교차로 및 방향 계산
            next_intersection, direction = self._resolve_action(
                police_positions[police_idx], action, intersection_map, segment_map
            )

            recommendations[agent_id] = {
                "action": action,
                "next_intersection": next_intersection,
                "direction": direction,
                "confidence": confidence,
            }

        return recommendations

    def _build_observation(
        self,
        police_id: int,
        police_positions: list[int],
        fugitive_position: int,
        network_data: dict,
        intersection_map: dict,
        segment_map: dict,
        boundary_set: set,
        current_step: int,
    ) -> dict:
        """추론용 관측값을 수동 구성한다.

        관측값 딕셔너리 키 (sorted order):
            agent_id_onehot, current_step, my_position, my_progress,
            nearest_boundary_dist, neighbors, other_positions
        """
        my_position = police_positions[police_id]

        # agent_id_onehot
        agent_id_onehot = np.zeros(self.num_police, dtype=np.float32)
        agent_id_onehot[police_id] = 1.0

        # current_step
        step_arr = np.array([current_step], dtype=np.int64)

        # my_position
        my_pos_arr = np.array([my_position], dtype=np.int64)

        # my_progress (교차로에 있으므로 1.0)
        my_progress = np.array([1.0], dtype=np.float32)

        # nearest_boundary_dist
        boundary_dist = self._compute_boundary_distance(
            my_position, network_data, boundary_set
        )
        nearest_boundary_dist = np.array([boundary_dist], dtype=np.int64)

        # neighbors: 현재 교차로의 outgoing 목적지 (패딩)
        neighbors = self._get_padded_neighbors_with_segments(
            my_position, intersection_map, segment_map
        )

        # other_positions: 다른 경찰 + 도주자
        other_positions = []
        for i in range(self.num_police):
            if i != police_id:
                other_positions.append(police_positions[i])
        other_positions.append(fugitive_position)

        other_positions_arr = np.array(other_positions, dtype=np.int64)

        return {
            "agent_id_onehot": agent_id_onehot,
            "current_step": step_arr,
            "my_position": my_pos_arr,
            "my_progress": my_progress,
            "nearest_boundary_dist": nearest_boundary_dist,
            "neighbors": neighbors,
            "other_positions": other_positions_arr,
        }

    def _build_intersection_map(self, network_data: dict) -> dict:
        """교차로 데이터를 ID → 정보 딕셔너리로 변환한다."""
        imap = {}
        for inter in network_data.get("intersections", []):
            iid = inter["intersection_id"]
            imap[iid] = {
                "position": tuple(inter["position"]),
                "outgoing_segments": inter.get("outgoing_segments", []),
                "incoming_segments": inter.get("incoming_segments", []),
            }
        return imap

    def _build_segment_map(self, network_data: dict) -> dict:
        """세그먼트 데이터를 ID → 정보 딕셔너리로 변환한다."""
        smap = {}
        for seg in network_data.get("segments", []):
            sid = seg["segment_id"]
            smap[sid] = {
                "start_intersection_id": seg["start_intersection_id"],
                "end_intersection_id": seg["end_intersection_id"],
                "start_pos": tuple(seg["start_pos"]),
                "end_pos": tuple(seg["end_pos"]),
                "length": seg.get("length", 0.0),
            }
        return smap

    def _get_padded_neighbors_with_segments(
        self, intersection_id: int, intersection_map: dict, segment_map: dict
    ) -> np.ndarray:
        """교차로의 outgoing 목적지 ID를 세그먼트 맵으로부터 추출하여 패딩한다."""
        padded = np.full(self.fixed_max_degree, -1, dtype=np.int64)
        inter_data = intersection_map.get(intersection_id)
        if inter_data is None:
            return padded

        outgoing_seg_ids = inter_data["outgoing_segments"]
        for i, seg_id in enumerate(outgoing_seg_ids[:self.fixed_max_degree]):
            seg_data = segment_map.get(seg_id)
            if seg_data:
                padded[i] = seg_data["end_intersection_id"]

        return padded

    def _build_action_mask(self, intersection_id: int, intersection_map: dict) -> np.ndarray:
        """유효한 행동 마스크를 생성한다.

        유효 행동: outgoing 세그먼트가 있는 인덱스 + 대기(마지막 인덱스)
        """
        mask = np.zeros(self.action_dim, dtype=bool)

        inter_data = intersection_map.get(intersection_id)
        if inter_data is None:
            mask[-1] = True  # 대기만 허용
            return mask

        num_outgoing = len(inter_data["outgoing_segments"])
        for i in range(min(num_outgoing, self.fixed_max_degree)):
            mask[i] = True

        # 대기 행동은 항상 유효
        mask[-1] = True
        return mask

    def _infer_action(self, observation: dict, action_mask: np.ndarray) -> tuple[int, float]:
        """모델에서 행동을 추론한다.

        Args:
            observation: 관측값 딕셔너리
            action_mask: 유효 행동 마스크

        Returns:
            (action_index, confidence) 튜플
        """
        obs_flat = _flatten_observation(observation)
        obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0)

        with torch.no_grad():
            logits = self.police_actor(obs_tensor)

            # 마스크 적용
            mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0)
            logits = logits.masked_fill(~mask_tensor, float("-inf"))

            # Softmax로 확률 계산
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(logits=logits)

            # 가장 높은 확률의 행동 선택 (결정적 추론)
            action = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, action].item()

        return action, confidence

    def _resolve_action(
        self,
        current_intersection: int,
        action: int,
        intersection_map: dict,
        segment_map: dict,
    ) -> tuple[int, str]:
        """행동 인덱스를 실제 교차로 ID와 방향으로 변환한다.

        Args:
            current_intersection: 현재 교차로 ID
            action: 행동 인덱스
            intersection_map: 교차로 맵
            segment_map: 세그먼트 맵

        Returns:
            (next_intersection_id, direction_description) 튜플
        """
        # 대기 행동
        if action >= self.fixed_max_degree:
            return current_intersection, "대기 (stay)"

        inter_data = intersection_map.get(current_intersection)
        if inter_data is None:
            return current_intersection, "알 수 없음 (unknown intersection)"

        outgoing_seg_ids = inter_data["outgoing_segments"]

        if action >= len(outgoing_seg_ids):
            return current_intersection, "대기 (invalid action → stay)"

        seg_id = outgoing_seg_ids[action]
        seg_data = segment_map.get(seg_id)

        if seg_data is None:
            return current_intersection, "알 수 없음 (unknown segment)"

        next_intersection = seg_data["end_intersection_id"]

        # 방향 계산
        dx = seg_data["end_pos"][0] - seg_data["start_pos"][0]
        dy = seg_data["end_pos"][1] - seg_data["start_pos"][1]
        direction = self._compute_direction_label(dx, dy)

        return next_intersection, direction

    def _compute_direction_label(self, dx: float, dy: float) -> str:
        """방향 벡터를 텍스트 레이블로 변환한다."""
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return "정지"

        angle = math.degrees(math.atan2(dy, dx))

        if -22.5 <= angle < 22.5:
            return "동 (east)"
        elif 22.5 <= angle < 67.5:
            return "북동 (northeast)"
        elif 67.5 <= angle < 112.5:
            return "북 (north)"
        elif 112.5 <= angle < 157.5:
            return "북서 (northwest)"
        elif angle >= 157.5 or angle < -157.5:
            return "서 (west)"
        elif -157.5 <= angle < -112.5:
            return "남서 (southwest)"
        elif -112.5 <= angle < -67.5:
            return "남 (south)"
        elif -67.5 <= angle < -22.5:
            return "남동 (southeast)"
        else:
            return "알 수 없음"

    def _compute_boundary_distance(
        self, intersection_id: int, network_data: dict, boundary_set: set
    ) -> int:
        """BFS로 가장 가까운 경계 교차로까지의 홉 수를 계산한다."""
        if intersection_id in boundary_set:
            return 0

        if not boundary_set:
            return 0

        # 인접 리스트 구성
        adjacency: dict[int, list[int]] = {}
        for seg in network_data.get("segments", []):
            start = seg["start_intersection_id"]
            end = seg["end_intersection_id"]
            if start not in adjacency:
                adjacency[start] = []
            adjacency[start].append(end)

        # BFS
        from collections import deque

        visited = {intersection_id}
        queue = deque([(intersection_id, 0)])

        while queue:
            node, dist = queue.popleft()
            if node in boundary_set:
                return dist
            for neighbor in adjacency.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        # 도달 불가 시
        return len(network_data.get("intersections", []))

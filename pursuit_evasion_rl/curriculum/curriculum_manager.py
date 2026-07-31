"""커리큘럼 학습 난이도 전환 관리 모듈.

CurriculumLevel 데이터클래스와 CurriculumManager 클래스를 제공한다.
경찰 팀의 승률이 임계값을 초과하면 자동으로 다음 난이도 단계로 승급한다.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CurriculumLevel:
    """커리큘럼 난이도 단계 정의.

    Attributes:
        level_id: 난이도 단계 고유 ID (정렬 기준)
        name: 난이도 이름 (예: "easy", "medium", "hard")
        num_nodes: 그래프 노드 수
        density: 연결 밀도 (0.0 ~ 1.0)
        boundary_ratio: 경계 노드 비율 (현재 미사용, 예약됨)
        num_police: 경찰 에이전트 수
    """

    level_id: int
    name: str
    num_nodes: int
    density: float
    boundary_ratio: float
    num_police: int


# 기본 커리큘럼 레벨 정의
DEFAULT_LEVELS: list[CurriculumLevel] = [
    CurriculumLevel(
        level_id=1,
        name="easy",
        num_nodes=10,
        density=0.5,
        boundary_ratio=0.3,
        num_police=3,
    ),
    CurriculumLevel(
        level_id=2,
        name="medium",
        num_nodes=20,
        density=0.3,
        boundary_ratio=0.2,
        num_police=3,
    ),
    CurriculumLevel(
        level_id=3,
        name="hard",
        num_nodes=30,
        density=0.25,
        boundary_ratio=0.15,
        num_police=3,
    ),
]


class CurriculumManager:
    """커리큘럼 학습 난이도 전환 관리자.

    경찰 팀의 승률을 추적하고, 임계값 도달 시 다음 난이도로 자동 승급한다.
    슬라이딩 윈도우 방식으로 최근 에피소드의 승률만을 기준으로 판단한다.

    Attributes:
        levels: 난이도 단계 리스트 (level_id 오름차순 정렬됨)
        win_rate_threshold: 승급 판정 임계값
        promotion_window: 승률 계산 윈도우 크기
    """

    def __init__(self, levels: list[CurriculumLevel], config: dict) -> None:
        """CurriculumManager 초기화.

        Args:
            levels: 난이도 단계 리스트 (최소 3개, level_id 기준 정렬됨)
            config: 설정 딕셔너리. 키:
                - win_rate_threshold: 승률 임계값 (기본 0.9 = 90%)
                - promotion_window: 평가 에피소드 수 (기본 500)

        Raises:
            ValueError: levels가 3개 미만인 경우
        """
        if len(levels) < 3:
            raise ValueError(
                f"커리큘럼 레벨은 최소 3개가 필요합니다. "
                f"제공된 레벨 수: {len(levels)}"
            )

        # level_id 기준 오름차순 정렬
        self.levels: list[CurriculumLevel] = sorted(levels, key=lambda l: l.level_id)
        self.win_rate_threshold: float = config.get("win_rate_threshold", 0.9)
        self.promotion_window: int = config.get("promotion_window", 500)

        # 내부 상태
        self._current_level_index: int = 0
        self._episode_results: deque[bool] = deque(maxlen=self.promotion_window)
        self._total_episodes: int = 0

    @property
    def current_level(self) -> CurriculumLevel:
        """현재 커리큘럼 난이도 단계를 반환한다."""
        return self.levels[self._current_level_index]

    def get_env_config(self) -> dict[str, Any]:
        """현재 레벨에 해당하는 환경 설정 딕셔너리를 반환한다.

        Returns:
            환경 구성에 필요한 파라미터 딕셔너리:
                - num_nodes: 그래프 노드 수
                - density: 연결 밀도
                - boundary_ratio: 경계 노드 비율
                - num_police: 경찰 수
        """
        level = self.current_level
        return {
            "num_nodes": level.num_nodes,
            "density": level.density,
            "boundary_ratio": level.boundary_ratio,
            "num_police": level.num_police,
        }

    def report_episode_result(self, is_police_win: bool) -> None:
        """에피소드 결과를 기록한다.

        슬라이딩 윈도우에 결과를 추가하여 승률을 추적한다.

        Args:
            is_police_win: 경찰 팀의 승리 여부
        """
        self._episode_results.append(is_police_win)
        self._total_episodes += 1

    def should_promote(self) -> bool:
        """승급 조건을 충족하는지 검사한다.

        최근 promotion_window 에피소드의 경찰 승률이 win_rate_threshold
        이상이면 True를 반환한다. promotion_window보다 적은 에피소드가
        플레이된 경우에는 항상 False를 반환한다.

        Returns:
            승급 조건 충족 여부
        """
        if len(self._episode_results) < self.promotion_window:
            return False

        win_rate = sum(self._episode_results) / len(self._episode_results)
        return win_rate >= self.win_rate_threshold

    def promote(self) -> bool:
        """다음 난이도 단계로 승급한다.

        모델 가중치는 변경하지 않으며, 레벨만 전환한다.
        승급 후 에피소드 결과 윈도우는 초기화된다.

        Returns:
            승급 성공 여부. 이미 최종 레벨이면 False.
        """
        if self._current_level_index >= len(self.levels) - 1:
            logger.warning(
                "이미 최종 레벨(%s)에 도달했습니다. 승급 불가.",
                self.current_level.name,
            )
            return False

        prev_level = self.current_level
        self._current_level_index += 1
        new_level = self.current_level

        # 에피소드 결과 윈도우 초기화 (새 레벨에서 새로 시작)
        self._episode_results.clear()

        logger.info(
            "커리큘럼 승급: %s (레벨 %d) → %s (레벨 %d)",
            prev_level.name,
            prev_level.level_id,
            new_level.name,
            new_level.level_id,
        )

        return True

    def is_completed(self) -> bool:
        """커리큘럼 완료 여부를 반환한다.

        최종 레벨에 도달하고, 해당 레벨에서 승률 임계값을 충족하면
        커리큘럼이 완료된 것으로 판단한다.

        Returns:
            커리큘럼 완료 여부
        """
        is_final_level = self._current_level_index >= len(self.levels) - 1
        return is_final_level and self.should_promote()

    def get_stats(self) -> dict[str, Any]:
        """현재 커리큘럼 상태 통계를 반환한다.

        Returns:
            통계 딕셔너리:
                - current_level: 현재 레벨 ID
                - win_rate: 현재 윈도우 내 승률
                - episodes_in_level: 현재 레벨에서 플레이한 에피소드 수
                - total_episodes: 전체 에피소드 수
        """
        episodes_in_level = len(self._episode_results)
        if episodes_in_level > 0:
            win_rate = sum(self._episode_results) / episodes_in_level
        else:
            win_rate = 0.0

        return {
            "current_level": self.current_level.level_id,
            "win_rate": win_rate,
            "episodes_in_level": episodes_in_level,
            "total_episodes": self._total_episodes,
        }

    def save_state(self, path: str) -> None:
        """커리큘럼 상태를 JSON 파일로 저장한다.

        Args:
            path: 저장 경로 (.json)
        """
        state = {
            "current_level_index": self._current_level_index,
            "episode_results": list(self._episode_results),
            "total_episodes": self._total_episodes,
            "levels": [asdict(level) for level in self.levels],
            "win_rate_threshold": self.win_rate_threshold,
            "promotion_window": self.promotion_window,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

        logger.info("커리큘럼 상태 저장: %s", path)

    def load_state(self, path: str) -> None:
        """JSON 파일에서 커리큘럼 상태를 로드한다.

        Args:
            path: 로드 경로 (.json)

        Note:
            파일이 손상되거나 로드 실패 시 경고 로그를 출력하고
            첫 번째 레벨부터 재시작한다.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)

            self._current_level_index = state["current_level_index"]
            self._total_episodes = state["total_episodes"]

            # deque 복원 (maxlen 유지)
            self._episode_results = deque(
                state["episode_results"], maxlen=self.promotion_window
            )

            logger.info(
                "커리큘럼 상태 로드: %s (레벨 %d, 에피소드 %d)",
                path,
                self.current_level.level_id,
                self._total_episodes,
            )

        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.warning(
                "커리큘럼 상태 로드 실패 (%s): %s. 첫 번째 레벨부터 재시작합니다.",
                path,
                str(e),
            )
            self._current_level_index = 0
            self._episode_results.clear()
            self._total_episodes = 0

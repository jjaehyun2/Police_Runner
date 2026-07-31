"""YAML 설정 파싱 및 검증 모듈.

SimulationConfig dataclass와 ConfigManager 클래스를 제공한다.
설정 파일 로드, 검증, 직렬화, round-trip 검증을 담당한다.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field, fields, asdict
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """시뮬레이션 설정을 담는 데이터 클래스.

    모든 필드는 기본값을 가지며, YAML 설정 파일에서 누락된 항목은
    기본값이 적용된다.
    """

    # 네트워크 설정
    network_mode: str = "random"  # "random" | "fixed"
    num_nodes: int = 20  # 4~200
    density: float = 0.3  # 0.0 < x <= 1.0
    fixed_edges: list[tuple[int, int]] | None = None

    # 에이전트 설정
    num_police: int = 3  # 3~20
    max_steps: int = 500  # >= 1

    # 학습 하이퍼파라미터
    algorithm: str = "MAPPO"  # "MAPPO" | "MADDPG"
    learning_rate: float = 3e-4  # > 0
    discount_factor: float = 0.99  # 0~1
    epsilon: float = 0.1  # >= 0
    batch_size: int = 64  # >= 1
    replay_buffer_size: int = 100000  # >= 1
    max_episodes: int = 10000  # >= 1
    checkpoint_interval: int = 100  # >= 1

    # 보상 설정
    terminal_reward: float = 1.0
    shaping_scale: float = 0.1
    cooperation_bonus: float = 0.05
    invalid_action_penalty: float = -0.01


# YAML 키 → SimulationConfig 필드 매핑
_YAML_KEY_MAP: dict[str, str] = {
    "network.mode": "network_mode",
    "network.num_nodes": "num_nodes",
    "network.density": "density",
    "network.fixed_edges": "fixed_edges",
    "agents.num_police": "num_police",
    "environment.max_steps": "max_steps",
    "training.algorithm": "algorithm",
    "training.learning_rate": "learning_rate",
    "training.discount_factor": "discount_factor",
    "training.epsilon": "epsilon",
    "training.batch_size": "batch_size",
    "training.replay_buffer_size": "replay_buffer_size",
    "training.max_episodes": "max_episodes",
    "training.checkpoint_interval": "checkpoint_interval",
    "rewards.terminal_reward": "terminal_reward",
    "rewards.shaping_scale": "shaping_scale",
    "rewards.cooperation_bonus": "cooperation_bonus",
    "rewards.invalid_action_penalty": "invalid_action_penalty",
}

# SimulationConfig 필드 → YAML 경로 역매핑
_FIELD_TO_YAML_KEY: dict[str, str] = {v: k for k, v in _YAML_KEY_MAP.items()}


class ConfigManager:
    """YAML 설정 파일을 로드, 검증, 직렬화하는 관리자 클래스."""

    def __init__(self, config_path: str) -> None:
        """ConfigManager 초기화.

        Args:
            config_path: YAML 설정 파일 경로
        """
        self.config_path = config_path

    def load(self) -> SimulationConfig:
        """설정 파일을 로드하고 검증한다.

        - 누락 항목: 기본값 적용 + 경고 메시지
        - 문법 오류: 행 번호 포함 에러 메시지 + sys.exit(1)
        - 범위 초과: 항목명 + 허용 범위 에러 메시지 + sys.exit(1)

        Returns:
            검증된 SimulationConfig 객체
        """
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error("설정 파일을 찾을 수 없습니다: %s", self.config_path)
            sys.exit(1)
        except yaml.YAMLError as e:
            if hasattr(e, "problem_mark") and e.problem_mark is not None:
                mark = e.problem_mark
                logger.error(
                    "YAML 문법 오류 (행 %d): %s",
                    mark.line + 1,
                    getattr(e, "problem", "알 수 없는 오류"),
                )
            else:
                logger.error("YAML 문법 오류: %s", str(e))
            sys.exit(1)

        if raw is None:
            raw = {}

        # 중첩 YAML에서 플랫 딕셔너리로 변환
        flat = self._flatten_yaml(raw)

        # 누락 항목 확인 및 기본값 적용
        defaults = SimulationConfig()
        config_kwargs: dict[str, Any] = {}

        for yaml_key, field_name in _YAML_KEY_MAP.items():
            if yaml_key in flat:
                value = flat[yaml_key]
                # fixed_edges 특별 처리: 리스트의 리스트 → 튜플 리스트
                if field_name == "fixed_edges" and value is not None:
                    value = [tuple(e) for e in value]
                config_kwargs[field_name] = value
            else:
                logger.warning(
                    "설정 항목 '%s' 누락, 기본값 적용: %s",
                    yaml_key,
                    getattr(defaults, field_name),
                )
                config_kwargs[field_name] = getattr(defaults, field_name)

        config = SimulationConfig(**config_kwargs)

        # 범위 검증
        errors = self.validate(config)
        if errors:
            for error in errors:
                logger.error(error)
            sys.exit(1)

        return config

    def validate(self, config: SimulationConfig) -> list[str]:
        """설정값의 범위 및 타입을 검증한다.

        Args:
            config: 검증할 SimulationConfig 객체

        Returns:
            오류 메시지 목록 (빈 리스트이면 유효)
        """
        errors: list[str] = []

        # network_mode 검증
        if config.network_mode not in ("random", "fixed"):
            errors.append(
                f"network_mode: '{config.network_mode}' 은(는) 유효하지 않습니다. "
                f"허용 범위: 'random' | 'fixed'"
            )

        # num_nodes 검증
        if not isinstance(config.num_nodes, int) or config.num_nodes < 4 or config.num_nodes > 200:
            errors.append(
                f"num_nodes: {config.num_nodes} 은(는) 유효하지 않습니다. "
                f"허용 범위: 4~200 (정수)"
            )

        # density 검증
        if not isinstance(config.density, (int, float)) or config.density <= 0.0 or config.density > 1.0:
            errors.append(
                f"density: {config.density} 은(는) 유효하지 않습니다. "
                f"허용 범위: 0.0 < density <= 1.0"
            )

        # num_police 검증
        if not isinstance(config.num_police, int) or config.num_police < 3 or config.num_police > 20:
            errors.append(
                f"num_police: {config.num_police} 은(는) 유효하지 않습니다. "
                f"허용 범위: 3~20 (정수)"
            )

        # max_steps 검증
        if not isinstance(config.max_steps, int) or config.max_steps < 1:
            errors.append(
                f"max_steps: {config.max_steps} 은(는) 유효하지 않습니다. "
                f"허용 범위: >= 1 (정수)"
            )

        # algorithm 검증
        if config.algorithm not in ("MAPPO", "MADDPG"):
            errors.append(
                f"algorithm: '{config.algorithm}' 은(는) 유효하지 않습니다. "
                f"허용 범위: 'MAPPO' | 'MADDPG'"
            )

        # learning_rate 검증
        if not isinstance(config.learning_rate, (int, float)) or config.learning_rate <= 0:
            errors.append(
                f"learning_rate: {config.learning_rate} 은(는) 유효하지 않습니다. "
                f"허용 범위: > 0"
            )

        # discount_factor 검증
        if not isinstance(config.discount_factor, (int, float)) or config.discount_factor < 0 or config.discount_factor > 1:
            errors.append(
                f"discount_factor: {config.discount_factor} 은(는) 유효하지 않습니다. "
                f"허용 범위: 0~1"
            )

        # epsilon 검증
        if not isinstance(config.epsilon, (int, float)) or config.epsilon < 0:
            errors.append(
                f"epsilon: {config.epsilon} 은(는) 유효하지 않습니다. "
                f"허용 범위: >= 0"
            )

        # batch_size 검증
        if not isinstance(config.batch_size, int) or config.batch_size < 1:
            errors.append(
                f"batch_size: {config.batch_size} 은(는) 유효하지 않습니다. "
                f"허용 범위: >= 1 (정수)"
            )

        # replay_buffer_size 검증
        if not isinstance(config.replay_buffer_size, int) or config.replay_buffer_size < 1:
            errors.append(
                f"replay_buffer_size: {config.replay_buffer_size} 은(는) 유효하지 않습니다. "
                f"허용 범위: >= 1 (정수)"
            )

        # max_episodes 검증
        if not isinstance(config.max_episodes, int) or config.max_episodes < 1:
            errors.append(
                f"max_episodes: {config.max_episodes} 은(는) 유효하지 않습니다. "
                f"허용 범위: >= 1 (정수)"
            )

        # checkpoint_interval 검증
        if not isinstance(config.checkpoint_interval, int) or config.checkpoint_interval < 1:
            errors.append(
                f"checkpoint_interval: {config.checkpoint_interval} 은(는) 유효하지 않습니다. "
                f"허용 범위: >= 1 (정수)"
            )

        # terminal_reward - 범위 제한 없음 (모든 float 허용)
        if not isinstance(config.terminal_reward, (int, float)):
            errors.append(
                f"terminal_reward: {config.terminal_reward} 은(는) 유효하지 않습니다. "
                f"허용 범위: 숫자"
            )

        # shaping_scale - 범위 제한 없음 (모든 float 허용)
        if not isinstance(config.shaping_scale, (int, float)):
            errors.append(
                f"shaping_scale: {config.shaping_scale} 은(는) 유효하지 않습니다. "
                f"허용 범위: 숫자"
            )

        # cooperation_bonus - 범위 제한 없음 (모든 float 허용)
        if not isinstance(config.cooperation_bonus, (int, float)):
            errors.append(
                f"cooperation_bonus: {config.cooperation_bonus} 은(는) 유효하지 않습니다. "
                f"허용 범위: 숫자"
            )

        # invalid_action_penalty - 범위 제한 없음 (모든 float 허용)
        if not isinstance(config.invalid_action_penalty, (int, float)):
            errors.append(
                f"invalid_action_penalty: {config.invalid_action_penalty} 은(는) 유효하지 않습니다. "
                f"허용 범위: 숫자"
            )

        return errors

    def serialize(self, config: SimulationConfig) -> str:
        """SimulationConfig를 YAML 문자열로 직렬화한다.

        YAML 구조는 중첩 키를 사용한다:
        network, agents, environment, training, rewards

        Args:
            config: 직렬화할 SimulationConfig 객체

        Returns:
            YAML 형식 문자열
        """
        yaml_dict: dict[str, Any] = {}

        for field_obj in fields(config):
            field_name = field_obj.name
            value = getattr(config, field_name)
            yaml_key = _FIELD_TO_YAML_KEY.get(field_name)
            if yaml_key is None:
                continue

            parts = yaml_key.split(".")
            section = parts[0]
            key = parts[1]

            if section not in yaml_dict:
                yaml_dict[section] = {}

            # fixed_edges 특별 처리: 튜플 리스트 → 리스트의 리스트
            if field_name == "fixed_edges" and value is not None:
                value = [list(e) for e in value]

            yaml_dict[section][key] = value

        return yaml.dump(yaml_dict, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def round_trip(self, config: SimulationConfig) -> SimulationConfig:
        """직렬화 후 재파싱하여 동일 객체를 반환한다 (round-trip 검증).

        Args:
            config: 검증할 SimulationConfig 객체

        Returns:
            재파싱된 SimulationConfig 객체 (원본과 동일해야 함)
        """
        yaml_str = self.serialize(config)
        raw = yaml.safe_load(yaml_str)
        flat = self._flatten_yaml(raw)

        config_kwargs: dict[str, Any] = {}
        for yaml_key, field_name in _YAML_KEY_MAP.items():
            if yaml_key in flat:
                value = flat[yaml_key]
                if field_name == "fixed_edges" and value is not None:
                    value = [tuple(e) for e in value]
                config_kwargs[field_name] = value
            else:
                config_kwargs[field_name] = getattr(SimulationConfig(), field_name)

        return SimulationConfig(**config_kwargs)

    def _flatten_yaml(self, raw: dict[str, Any]) -> dict[str, Any]:
        """중첩 YAML 딕셔너리를 점 표기법 플랫 딕셔너리로 변환한다.

        Args:
            raw: 파싱된 YAML 딕셔너리

        Returns:
            점 표기법 키를 가진 플랫 딕셔너리
        """
        flat: dict[str, Any] = {}
        for section_key, section_val in raw.items():
            if isinstance(section_val, dict):
                for key, val in section_val.items():
                    flat[f"{section_key}.{key}"] = val
            else:
                flat[section_key] = section_val
        return flat

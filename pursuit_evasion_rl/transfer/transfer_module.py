"""이산→연속 환경 전이 학습 모듈.

이산 환경(MAPPOAlgorithm)의 Critic 가중치를 연속 환경(GaussianMAPPOAlgorithm)으로
전이한다. 차원이 일치하는 레이어만 복사하고, 불일치 레이어는 경고 후 건너뛴다.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)


class TransferModule:
    """이산→연속 환경 전이 학습 모듈."""

    def __init__(self, config: dict | None = None) -> None:
        """TransferModule 초기화.

        Args:
            config: 전이 학습 설정. 키:
                - transfer_from: 이산 환경 체크포인트 경로 (빈 문자열이면 전이 없음)
        """
        self._config = config or {}
        self._transfer_from: str = self._config.get("transfer_from", "")

    def is_transfer_enabled(self) -> bool:
        """전이 학습 활성화 여부를 반환한다."""
        return bool(self._transfer_from and self._transfer_from.strip())

    def transfer_critic_weights(
        self,
        source_path: str,
        target_model: object,
    ) -> dict[str, str]:
        """이산 모델의 Critic 가중치를 연속 모델로 전이한다.

        차원이 일치하는 레이어의 가중치만 복사하고,
        차원이 불일치하는 레이어는 경고 후 건너뛴다.

        Args:
            source_path: 이산 환경 체크포인트 파일 경로
            target_model: GaussianMAPPOAlgorithm 인스턴스

        Returns:
            레이어별 전이 결과 딕셔너리 {"layer_name": "transferred"|"skipped (reason)"}
        """
        results: dict[str, str] = {}

        # 소스 파일 확인
        if not os.path.exists(source_path):
            logger.warning("전이 학습 소스 체크포인트 미존재: %s", source_path)
            return {"error": f"source not found: {source_path}"}

        # 소스 체크포인트 로드
        try:
            source_checkpoint = torch.load(source_path, map_location="cpu")
        except Exception as e:
            logger.warning("소스 체크포인트 로드 실패: %s", e)
            return {"error": f"load failed: {e}"}

        # 소스 Critic state_dict 추출
        source_police_critic = source_checkpoint.get("police_critic", {})
        source_fugitive_critic = source_checkpoint.get("fugitive_critic", {})

        # 타겟 모델의 Critic state_dict
        target_police_critic_sd = target_model.police_critic.state_dict()  # type: ignore
        target_fugitive_critic_sd = target_model.fugitive_critic.state_dict()  # type: ignore

        # 경찰 Critic 전이
        results.update(
            self._transfer_state_dict(
                source_sd=source_police_critic,
                target_sd=target_police_critic_sd,
                target_module=target_model.police_critic,  # type: ignore
                prefix="police_critic",
            )
        )

        # 도망자 Critic 전이
        results.update(
            self._transfer_state_dict(
                source_sd=source_fugitive_critic,
                target_sd=target_fugitive_critic_sd,
                target_module=target_model.fugitive_critic,  # type: ignore
                prefix="fugitive_critic",
            )
        )

        return results

    def _transfer_state_dict(
        self,
        source_sd: dict[str, torch.Tensor],
        target_sd: dict[str, torch.Tensor],
        target_module: torch.nn.Module,
        prefix: str,
    ) -> dict[str, str]:
        """개별 모듈의 state_dict를 전이한다."""
        results: dict[str, str] = {}
        new_sd = dict(target_sd)  # 복사본 생성

        for key in target_sd:
            full_key = f"{prefix}.{key}"
            if key not in source_sd:
                results[full_key] = "skipped (not in source)"
                logger.info("전이 건너뜀 (소스에 없음): %s", full_key)
                continue

            source_tensor = source_sd[key]
            target_tensor = target_sd[key]

            if source_tensor.shape == target_tensor.shape:
                new_sd[key] = source_tensor.clone()
                results[full_key] = "transferred"
                logger.info("전이 완료: %s (shape=%s)", full_key, source_tensor.shape)
            else:
                results[full_key] = (
                    f"skipped (shape mismatch: source={source_tensor.shape}, "
                    f"target={target_tensor.shape})"
                )
                logger.warning(
                    "전이 건너뜀 (차원 불일치): %s source=%s target=%s",
                    full_key,
                    source_tensor.shape,
                    target_tensor.shape,
                )

        # 전이된 가중치 적용
        target_module.load_state_dict(new_sd)
        return results

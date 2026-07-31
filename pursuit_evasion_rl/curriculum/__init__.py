"""커리큘럼 학습 모듈.

CurriculumManager와 CurriculumLevel을 제공한다.
점진적 난이도 조절을 통해 경찰 정책을 단계적으로 강화한다.
"""

from pursuit_evasion_rl.curriculum.curriculum_manager import (
    CurriculumLevel,
    CurriculumManager,
)

__all__ = ["CurriculumLevel", "CurriculumManager"]

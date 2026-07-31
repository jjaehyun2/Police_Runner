"""커리큘럼 학습 10,000 에피소드 실행 스크립트."""
import sys
import warnings
import logging

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

# 그래프 생성/경계 노드 로그 억제 (환경 모듈의 INFO/WARNING 숨김)
logging.getLogger("pursuit_evasion_rl.env.road_network").setLevel(logging.ERROR)
logging.getLogger("pursuit_evasion_rl.env.actions").setLevel(logging.ERROR)
logging.getLogger("pursuit_evasion_rl.curriculum.curriculum_manager").setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

from pursuit_evasion_rl.training.long_train_pipeline import LongTrainPipeline

config = {
    "max_episodes": 10000,
    "checkpoint_interval": 500,
    "log_interval": 200,
    "checkpoint_dir": "checkpoints/curriculum_v3",
    "algorithm": "MAPPO",
    "num_police": 4,
    "fixed_max_degree": 10,
    "max_steps": 50,
    "learning_rate": 1e-4,
    "discount_factor": 0.99,
    "epsilon": 0.1,
    "batch_size": 64,
    "hidden_dims": [128, 128],
    "curriculum": {
        "levels": [
            {"level_id": 1, "name": "easy", "num_nodes": 10, "density": 0.5, "num_police": 4},
            {"level_id": 2, "name": "medium", "num_nodes": 20, "density": 0.3, "num_police": 4},
            {"level_id": 3, "name": "hard", "num_nodes": 30, "density": 0.25, "num_police": 4},
        ],
        "win_rate_threshold": 0.9,
        "promotion_window": 500,
    },
}

print("=" * 60)
print("  커리큘럼 학습 v3 시작")
print("  - 10,000 에피소드")
print("  - 경찰 4대, max_steps=50")
print("  - 신경망 128x128, lr=1e-4, clip=0.1")
print("  - 동료 행동 관측 추가")
print("  - 타임아웃 보상: 경찰 -0.3 / 도망자 -0.8")
print("  - 3단계: easy(10노드) → medium(20노드) → hard(30노드)")
print("  - 승급 조건: 500 에피소드 윈도우 90% 경찰 승률")
print("=" * 60)

pipeline = LongTrainPipeline(config)
pipeline.train()

"""배포 패키지 검증 테스트."""
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
import logging; logging.disable(logging.WARNING)

from deployment.inference_engine import PursuitInferenceEngine

# 엔진 로드
engine = PursuitInferenceEngine(
    "checkpoints/road_pursuit/road_pursuit_final.pt",
    num_police=4, fixed_max_degree=5, hidden_dims=[128, 128]
)
print("추론 엔진 로드 OK")

# 테스트 네트워크 (4x4 그리드)
network_data = {
    "intersections": [
        {"intersection_id": i, "position": [(i % 4) * 30, (i // 4) * 30],
         "outgoing_segments": [], "incoming_segments": []}
        for i in range(16)
    ],
    "segments": [],
    "boundary_intersections": [0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15],
}

seg_id = 0
for i in range(16):
    r, c = i // 4, i % 4
    if c < 3:
        network_data["segments"].append({
            "segment_id": seg_id, "start_intersection_id": i, "end_intersection_id": i + 1,
            "start_pos": [c * 30, r * 30], "end_pos": [(c + 1) * 30, r * 30], "length": 30.0
        })
        network_data["intersections"][i]["outgoing_segments"].append(seg_id)
        network_data["intersections"][i + 1]["incoming_segments"].append(seg_id)
        seg_id += 1
    if r < 3:
        network_data["segments"].append({
            "segment_id": seg_id, "start_intersection_id": i, "end_intersection_id": i + 4,
            "start_pos": [c * 30, r * 30], "end_pos": [c * 30, (r + 1) * 30], "length": 30.0
        })
        network_data["intersections"][i]["outgoing_segments"].append(seg_id)
        network_data["intersections"][i + 4]["incoming_segments"].append(seg_id)
        seg_id += 1

print(f"테스트 네트워크: {len(network_data['intersections'])}교차로, {len(network_data['segments'])}세그먼트")

# 추천 요청
result = engine.recommend_actions(
    police_positions=[0, 3, 12, 15],
    fugitive_position=5,
    network_data=network_data,
)

print("\n=== 경찰 배치 추천 결과 ===")
print(f"상황: 경찰=[0,3,12,15], 도주자=5")
print()
for pid, rec in result.items():
    print(f"  {pid}: action={rec['action']}, "
          f"next={rec['next_intersection']}, "
          f"direction={rec['direction']}, "
          f"confidence={rec['confidence']:.2f}")

print("\n검증 완료!")

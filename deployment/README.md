# 경찰 추격 RL 모델 배포 패키지

학습된 MAPPO 강화학습 모델을 REST API로 서비스하는 배포 패키지입니다.
실제 도로 네트워크(OpenStreetMap)를 로드하고, 경찰 배치 추천을 제공합니다.

## 설치

### 1. 의존성 설치

```bash
pip install -r deployment/requirements.txt
```

**참고:** `osmnx`는 OpenStreetMap 도로 로딩에 필요하며, 선택적 의존성입니다.
OSM 기능 없이도 JSON 파일로 네트워크를 로드하여 사용할 수 있습니다.

### 2. 모델 파일 준비

학습된 모델 체크포인트를 다음 경로에 배치하세요:
```
checkpoints/road_pursuit/road_pursuit_final.pt
```

또는 환경 변수로 경로를 지정할 수 있습니다:
```bash
export PURSUIT_MODEL_PATH="path/to/your/model.pt"
export PURSUIT_NUM_POLICE=4
export PURSUIT_MAX_DEGREE=5
```

## 도로 네트워크 로딩

### OSM에서 직접 로드 (Python)

```python
from deployment.osm_loader import OSMRoadLoader

loader = OSMRoadLoader()

# 지역명으로 로드
network = loader.load_from_place("Daejeon, South Korea", network_type="drive")

# bounding box로 로드
network = loader.load_from_bbox(
    north=36.38, south=36.30, east=127.40, west=127.33
)

# JSON으로 저장
loader.save_network(network, "networks/daejeon.json")

# 저장된 네트워크 로드
network = loader.load_network("networks/daejeon.json")
```

### API를 통해 로드

서버 실행 후 API 엔드포인트를 통해 로드할 수도 있습니다 (아래 참조).

## API 서버 실행

```bash
# 프로젝트 루트에서 실행
python -m deployment.run_server

# 또는 직접 uvicorn 실행
uvicorn deployment.api_server:app --host 0.0.0.0 --port 8000 --reload
```

서버가 시작되면 다음 URL에서 API 문서를 확인할 수 있습니다:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API 엔드포인트

### `GET /health` - 서버 상태 확인

```bash
curl http://localhost:8000/health
```

응답:
```json
{
  "status": "ok",
  "model_loaded": true,
  "networks_loaded": ["default"]
}
```

### `POST /load_network` - OSM에서 도로 네트워크 로드

지역명으로 로드:
```bash
curl -X POST http://localhost:8000/load_network \
  -H "Content-Type: application/json" \
  -d '{
    "place_name": "Yuseong-gu, Daejeon, South Korea",
    "network_id": "yuseong"
  }'
```

Bounding box로 로드:
```bash
curl -X POST http://localhost:8000/load_network \
  -H "Content-Type: application/json" \
  -d '{
    "bbox": {"north": 36.38, "south": 36.30, "east": 127.40, "west": 127.33},
    "network_id": "daejeon_center"
  }'
```

### `POST /load_network_file` - JSON 파일에서 네트워크 로드

```bash
curl -X POST http://localhost:8000/load_network_file \
  -H "Content-Type: application/json" \
  -d '{
    "filepath": "networks/daejeon.json",
    "network_id": "default"
  }'
```

### `POST /recommend` - 경찰 배치 추천

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "police_positions": [0, 5, 10, 15],
    "fugitive_position": 8,
    "network_id": "default",
    "current_step": 0
  }'
```

응답:
```json
{
  "recommendations": [
    {
      "police_id": "police_0",
      "action_index": 2,
      "next_intersection_id": 1,
      "direction_description": "동 (east)",
      "confidence": 0.85
    },
    {
      "police_id": "police_1",
      "action_index": 0,
      "next_intersection_id": 6,
      "direction_description": "북 (north)",
      "confidence": 0.72
    }
  ],
  "network_id": "default"
}
```

### `GET /networks` - 로드된 네트워크 목록

```bash
curl http://localhost:8000/networks
```

### `DELETE /networks/{network_id}` - 네트워크 삭제

```bash
curl -X DELETE http://localhost:8000/networks/yuseong
```

## 환경 변수

| 변수명 | 기본값 | 설명 |
|--------|--------|------|
| `PURSUIT_MODEL_PATH` | `checkpoints/road_pursuit/road_pursuit_final.pt` | 모델 체크포인트 경로 |
| `PURSUIT_NUM_POLICE` | `4` | 경찰 에이전트 수 |
| `PURSUIT_MAX_DEGREE` | `5` | 최대 outgoing 차수 (행동 공간 크기) |

## 네트워크 데이터 형식

JSON 네트워크 파일의 구조:

```json
{
  "intersections": [
    {
      "intersection_id": 0,
      "position": [100.0, 200.0],
      "outgoing_segments": [0, 1],
      "incoming_segments": [2, 3]
    }
  ],
  "segments": [
    {
      "segment_id": 0,
      "start_intersection_id": 0,
      "end_intersection_id": 1,
      "start_pos": [100.0, 200.0],
      "end_pos": [200.0, 200.0],
      "length": 100.0
    }
  ],
  "boundary_intersections": [0, 3, 12, 15],
  "metadata": {
    "num_intersections": 16,
    "num_segments": 48,
    "source": "OpenStreetMap via osmnx"
  }
}
```

## 프로젝트 구조

```
deployment/
├── __init__.py          # 패키지 초기화
├── osm_loader.py        # OSM 도로 네트워크 로더
├── inference_engine.py  # RL 모델 추론 엔진
├── api_server.py        # FastAPI REST API 서버
├── run_server.py        # 서버 실행 스크립트
├── requirements.txt     # 의존성 목록
└── README.md            # 이 파일
```

## 주의사항

- 모델은 학습 시 사용된 네트워크 크기(그리드)와 동일한 `num_police`, `fixed_max_degree` 설정으로 추론해야 합니다.
- OSM에서 로드한 네트워크가 매우 클 경우, 추론 속도가 느려질 수 있습니다. 적절한 bounding box로 범위를 제한하는 것을 권장합니다.
- `osmnx`가 설치되지 않아도 JSON 파일 기반 네트워크 로딩 및 추론은 정상 동작합니다.

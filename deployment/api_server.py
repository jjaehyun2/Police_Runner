"""경찰 추격 RL 모델 REST API 서버.

FastAPI 기반으로 학습된 모델의 추론 결과를 HTTP API로 제공한다.
도로 네트워크 로딩, 경찰 배치 추천 등의 엔드포인트를 포함한다.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from deployment.inference_engine import PursuitInferenceEngine
from deployment.osm_loader import OSMRoadLoader

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Police Pursuit RL API",
    description="경찰 추격 강화학습 모델 추론 API. 도로 네트워크 위에서 경찰 배치를 추천합니다.",
    version="1.0.0",
)

# ──────────────────────────────────────────────────────────────────────
# 전역 상태
# ──────────────────────────────────────────────────────────────────────

# 기본 모델 경로
DEFAULT_MODEL_PATH = os.environ.get(
    "PURSUIT_MODEL_PATH",
    "checkpoints/road_pursuit/road_pursuit_final.pt",
)
DEFAULT_NUM_POLICE = int(os.environ.get("PURSUIT_NUM_POLICE", "4"))
DEFAULT_MAX_DEGREE = int(os.environ.get("PURSUIT_MAX_DEGREE", "5"))
_HIDDEN_DIMS_RAW = os.environ.get("PURSUIT_HIDDEN_DIMS")
DEFAULT_HIDDEN_DIMS = (
    [int(part) for part in _HIDDEN_DIMS_RAW.split(",") if part.strip()]
    if _HIDDEN_DIMS_RAW
    else None
)

# 전역 엔진 및 네트워크 저장소
_engine: Optional[PursuitInferenceEngine] = None
_networks: dict[str, dict] = {}  # network_id -> network_data
_osm_loader = OSMRoadLoader()


def _get_engine() -> PursuitInferenceEngine:
    """추론 엔진을 lazy하게 초기화한다."""
    global _engine
    if _engine is None:
        model_path = DEFAULT_MODEL_PATH
        if not Path(model_path).exists():
            raise HTTPException(
                status_code=503,
                detail=f"모델 파일을 찾을 수 없습니다: {model_path}. "
                f"PURSUIT_MODEL_PATH 환경 변수를 설정하거나 "
                f"checkpoints/road_pursuit/road_pursuit_final.pt 에 모델을 배치하세요.",
            )
        engine_kwargs = dict(
            model_path=model_path,
            num_police=DEFAULT_NUM_POLICE,
            fixed_max_degree=DEFAULT_MAX_DEGREE,
        )
        if DEFAULT_HIDDEN_DIMS is not None:
            engine_kwargs["hidden_dims"] = DEFAULT_HIDDEN_DIMS
        _engine = PursuitInferenceEngine(**engine_kwargs)
    return _engine


# ──────────────────────────────────────────────────────────────────────
# 요청/응답 모델
# ──────────────────────────────────────────────────────────────────────


class PursuitRequest(BaseModel):
    """경찰 배치 추천 요청."""

    police_positions: list[int] = Field(
        ..., description="각 경찰의 현재 교차로 ID 리스트"
    )
    fugitive_position: int = Field(
        ..., description="도주자의 현재/마지막 목격 교차로 ID"
    )
    network_id: str = Field(
        default="default", description="사용할 도로 네트워크 ID"
    )
    current_step: int = Field(
        default=0, description="현재 시뮬레이션 스텝"
    )


class ActionRecommendation(BaseModel):
    """단일 경찰에 대한 행동 추천."""

    police_id: str = Field(..., description="경찰 에이전트 ID")
    action_index: int = Field(..., description="선택된 행동 인덱스")
    next_intersection_id: int = Field(..., description="다음 이동할 교차로 ID")
    direction_description: str = Field(..., description="이동 방향 설명")
    confidence: float = Field(..., description="행동 확률 (0~1)")


class PursuitResponse(BaseModel):
    """경찰 배치 추천 응답."""

    recommendations: list[ActionRecommendation] = Field(
        ..., description="각 경찰에 대한 추천 리스트"
    )
    network_id: str = Field(..., description="사용된 네트워크 ID")


class LoadNetworkRequest(BaseModel):
    """OSM 네트워크 로딩 요청."""

    place_name: Optional[str] = Field(
        default=None, description="지역명 (예: 'Daejeon, South Korea')"
    )
    bbox: Optional[dict] = Field(
        default=None,
        description="bounding box: {north, south, east, west}",
    )
    network_id: str = Field(
        default="default", description="저장할 네트워크 ID"
    )
    save_path: Optional[str] = Field(
        default=None, description="네트워크를 JSON으로 저장할 경로 (선택)"
    )


class LoadNetworkFromFileRequest(BaseModel):
    """파일에서 네트워크 로딩 요청."""

    filepath: str = Field(..., description="네트워크 JSON 파일 경로")
    network_id: str = Field(
        default="default", description="저장할 네트워크 ID"
    )


class NetworkInfo(BaseModel):
    """네트워크 정보 응답."""

    network_id: str
    num_intersections: int
    num_segments: int
    num_boundary: int


class HealthResponse(BaseModel):
    """헬스 체크 응답."""

    status: str
    model_loaded: bool
    networks_loaded: list[str]


# ──────────────────────────────────────────────────────────────────────
# API 엔드포인트
# ──────────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """서버 상태 확인."""
    return HealthResponse(
        status="ok",
        model_loaded=_engine is not None,
        networks_loaded=list(_networks.keys()),
    )


@app.post("/recommend", response_model=PursuitResponse)
async def recommend_actions(request: PursuitRequest):
    """경찰 배치 추천 API.

    현재 경찰/도주자 위치를 받아 각 경찰의 최적 이동 방향을 추천한다.
    """
    # 네트워크 확인
    if request.network_id not in _networks:
        raise HTTPException(
            status_code=404,
            detail=f"네트워크 '{request.network_id}'가 로드되지 않았습니다. "
            f"/load_network 또는 /load_network_file 엔드포인트로 먼저 로드하세요. "
            f"현재 로드된 네트워크: {list(_networks.keys())}",
        )

    network_data = _networks[request.network_id]

    # 추론 엔진 가져오기
    try:
        engine = _get_engine()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"추론 엔진 초기화 실패: {e}") from e

    # 입력 검증
    num_intersections = len(network_data.get("intersections", []))
    for i, pos in enumerate(request.police_positions):
        if pos < 0 or pos >= num_intersections:
            raise HTTPException(
                status_code=400,
                detail=f"police_positions[{i}]={pos}가 유효 범위(0~{num_intersections - 1})를 벗어납니다.",
            )

    if request.fugitive_position < 0 or request.fugitive_position >= num_intersections:
        raise HTTPException(
            status_code=400,
            detail=f"fugitive_position={request.fugitive_position}가 유효 범위를 벗어납니다.",
        )

    # 추천 수행
    try:
        results = engine.recommend_actions(
            police_positions=request.police_positions,
            fugitive_position=request.fugitive_position,
            network_data=network_data,
            current_step=request.current_step,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 실패: {e}") from e

    # 응답 구성
    recommendations = []
    for agent_id, rec in results.items():
        recommendations.append(
            ActionRecommendation(
                police_id=agent_id,
                action_index=rec["action"],
                next_intersection_id=rec["next_intersection"],
                direction_description=rec["direction"],
                confidence=rec["confidence"],
            )
        )

    return PursuitResponse(
        recommendations=recommendations,
        network_id=request.network_id,
    )


@app.post("/load_network", response_model=NetworkInfo)
async def load_network(request: LoadNetworkRequest):
    """OSM에서 도로 네트워크를 로드한다.

    place_name 또는 bbox 중 하나를 지정해야 한다.
    """
    if request.place_name is None and request.bbox is None:
        raise HTTPException(
            status_code=400,
            detail="place_name 또는 bbox 중 하나를 지정하세요.",
        )

    try:
        if request.place_name:
            network_data = _osm_loader.load_from_place(request.place_name)
        else:
            bbox = request.bbox
            required_keys = {"north", "south", "east", "west"}
            if not required_keys.issubset(bbox.keys()):
                raise HTTPException(
                    status_code=400,
                    detail=f"bbox에 {required_keys} 키가 모두 필요합니다.",
                )
            network_data = _osm_loader.load_from_bbox(
                north=bbox["north"],
                south=bbox["south"],
                east=bbox["east"],
                west=bbox["west"],
            )
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"osmnx가 설치되지 않았습니다: {e}. "
            f"pip install osmnx 명령으로 설치하세요.",
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"네트워크 로딩 실패: {e}"
        ) from e

    # 저장
    _networks[request.network_id] = network_data

    if request.save_path:
        try:
            _osm_loader.save_network(network_data, request.save_path)
        except Exception as e:
            logger.warning("네트워크 저장 실패: %s", e)

    return NetworkInfo(
        network_id=request.network_id,
        num_intersections=len(network_data.get("intersections", [])),
        num_segments=len(network_data.get("segments", [])),
        num_boundary=len(network_data.get("boundary_intersections", [])),
    )


@app.post("/load_network_file", response_model=NetworkInfo)
async def load_network_from_file(request: LoadNetworkFromFileRequest):
    """JSON 파일에서 도로 네트워크를 로드한다."""
    try:
        network_data = _osm_loader.load_network(request.filepath)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"파일 로딩 실패: {e}"
        ) from e

    _networks[request.network_id] = network_data

    return NetworkInfo(
        network_id=request.network_id,
        num_intersections=len(network_data.get("intersections", [])),
        num_segments=len(network_data.get("segments", [])),
        num_boundary=len(network_data.get("boundary_intersections", [])),
    )


@app.get("/networks", response_model=list[NetworkInfo])
async def list_networks():
    """현재 로드된 모든 네트워크 목록을 반환한다."""
    result = []
    for network_id, data in _networks.items():
        result.append(
            NetworkInfo(
                network_id=network_id,
                num_intersections=len(data.get("intersections", [])),
                num_segments=len(data.get("segments", [])),
                num_boundary=len(data.get("boundary_intersections", [])),
            )
        )
    return result


@app.delete("/networks/{network_id}")
async def delete_network(network_id: str):
    """로드된 네트워크를 메모리에서 제거한다."""
    if network_id not in _networks:
        raise HTTPException(
            status_code=404,
            detail=f"네트워크 '{network_id}'를 찾을 수 없습니다.",
        )
    del _networks[network_id]
    return {"message": f"네트워크 '{network_id}' 삭제 완료."}

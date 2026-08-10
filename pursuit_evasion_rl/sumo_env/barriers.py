"""도로 차단(바리케이드) — 에피소드마다 무작위로 배치되는 통행 제한.

현실의 추격에는 공사·사고·집회로 막힌 길이 늘 있고, 경찰의 차단 작전 자체가
길목을 막는 행위다. 그런데 지금까지 환경은 모든 도로가 항상 열려 있다고
가정했기 때문에, 정책이 "막힌 길을 우회한다"는 판단을 배울 기회가 없었다.

여기서는 에피소드마다 시드에서 결정되는 위치에 차단 구간을 만든다. 도메인
랜덤화의 일부다: 특정 도로 구성에 과적합하지 않도록 매번 지형이 조금씩
달라진다.

안전장치가 하나 필요하다. 아무 도로나 막으면 도주자가 갇혀 시작부터 검거가
확정되거나, 반대로 지도가 두 조각으로 갈려 추격 자체가 성립하지 않는다.
그래서 차단 후보는 도주자 주변 일정 반경 밖에서 고르고, 출구가 하나뿐인
도로(막으면 그 지역이 고립되는 길)는 제외한다.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Sequence

#: 차단을 걸 수 있는 도로 등급. 이면도로를 막으면 눈에 띄지도 않고
#: 통행에 주는 영향도 미미해서, 실제로 체감되는 간선급만 고른다.
BLOCKABLE_CLASSES = frozenset({
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "unclassified", "residential",
})


@dataclass(frozen=True, slots=True)
class BarrierConfig:
    """에피소드당 차단 구간 설정.

    ``count`` 는 차단 구간 수. 도주자 시작점 기준으로 ``min_distance``
    안쪽은 막지 않고(시작하자마자 갇히면 정책을 평가할 수 없다),
    ``max_distance`` 바깥도 막지 않는다(추격과 무관한 외곽 차단은 화면만
    어지럽히고 경로 판단에 영향을 주지 않는다).
    """

    count: int = 8
    min_distance_from_fugitive_m: float = 400.0
    #: 차단은 추격이 벌어지는 권역 안에 있어야 의미가 있다. 도시 전역에
    #: 흩뿌리면 대부분이 추격과 무관한 외곽에 놓여, 화면만 어지럽히고
    #: 경로 판단에는 아무 영향도 주지 않는다.
    max_distance_from_fugitive_m: float = 2200.0
    min_edge_length_m: float = 60.0
    min_separation_m: float = 300.0
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("count must be non-negative")
        for name in ("min_distance_from_fugitive_m", "max_distance_from_fugitive_m",
                     "min_edge_length_m", "min_separation_m"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.max_distance_from_fugitive_m <= self.min_distance_from_fugitive_m:
            raise ValueError("max_distance_from_fugitive_m must exceed the minimum")


@dataclass(frozen=True, slots=True)
class Barrier:
    """설치된 차단 구간 한 곳."""

    edge_id: str
    x: float
    y: float
    angle_deg: float


def _edge_class(edge) -> str:
    try:
        return (edge.getType() or "").split(".")[-1]
    except AttributeError:
        return ""


def _midpoint_and_angle(edge) -> tuple[float, float, float]:
    shape = edge.getShape()
    mid = len(shape) // 2
    x0, y0 = shape[max(0, mid - 1)]
    x1, y1 = shape[min(len(shape) - 1, mid)]
    return (
        sum(p[0] for p in shape) / len(shape),
        sum(p[1] for p in shape) / len(shape),
        math.degrees(math.atan2(y1 - y0, x1 - x0)),
    )


def choose_barriers(
    net, config: BarrierConfig, *, seed: int, fugitive_xy: Sequence[float] | None = None
) -> tuple[Barrier, ...]:
    """차단할 도로를 고른다. 같은 시드면 같은 배치가 나온다."""
    if not config.enabled or config.count == 0:
        return ()
    rng = random.Random(seed)
    candidates = []
    for edge in net.getEdges():
        edge_id = edge.getID()
        if edge_id.startswith(":"):
            continue
        if edge.getLength() < config.min_edge_length_m:
            continue
        if _edge_class(edge) not in BLOCKABLE_CLASSES:
            continue
        # 출구가 하나뿐인 도로를 막으면 그 너머가 통째로 고립된다.
        if len(edge.getOutgoing()) < 2:
            continue
        x, y, angle = _midpoint_and_angle(edge)
        if fugitive_xy is not None:
            gap = math.dist((x, y), fugitive_xy)
            if not (config.min_distance_from_fugitive_m <= gap
                    <= config.max_distance_from_fugitive_m):
                continue
        candidates.append(Barrier(edge_id, x, y, angle))
    if not candidates:
        return ()

    rng.shuffle(candidates)
    chosen: list[Barrier] = []
    for candidate in candidates:
        if all(
            math.dist((candidate.x, candidate.y), (other.x, other.y)) >= config.min_separation_m
            for other in chosen
        ):
            chosen.append(candidate)
        if len(chosen) == config.count:
            break
    return tuple(chosen)


def apply_barriers(connection: Any, barriers: Sequence[Barrier]) -> tuple[str, ...]:
    """차단을 실제 시뮬레이션에 반영하고, 성공한 도로 목록을 돌려준다.

    승용차 통행을 막고 제한속도를 걷는 속도까지 낮춘다. 통행 금지만으로는
    이미 그 도로 위에 있던 차가 빠져나가지 못하는 경우가 있어, 속도까지
    낮춰 "막혀서 기어가는 구간"으로 보이게 한다.
    """
    applied: list[str] = []
    for barrier in barriers:
        try:
            connection.edge.setDisallowed(barrier.edge_id, ["passenger"])
            connection.edge.setMaxSpeed(barrier.edge_id, 0.8)
            applied.append(barrier.edge_id)
        except Exception:
            # 일부 엣지는 통행 클래스 설정을 거부한다. 차단 한 곳이 빠지는
            # 것은 에피소드를 중단할 이유가 되지 않는다.
            continue
    return tuple(applied)


def write_poi_additional(
    barriers: Sequence[Barrier], destination, *, image_file: str | None = None, size: float = 55.0
) -> str:
    """SUMO-GUI 에서 차단 지점을 볼 수 있는 additional 파일을 쓴다.

    ``image_file`` 을 주면 해당 이미지를 아이콘으로 띄운다(SUMO 는 POI 의
    ``imgFile`` 속성으로 PNG 를 렌더링한다). 없으면 색이 있는 사각형으로
    표시한다. 우리 관제 화면은 이 파일 없이도 그리므로, 이 출력은 SUMO 자체
    창을 함께 띄울 때를 위한 것이다.
    """
    from pathlib import Path

    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<additional>"]
    for index, barrier in enumerate(barriers):
        attrs = (
            f'id="barrier{index}" x="{barrier.x:.2f}" y="{barrier.y:.2f}" '
            f'layer="20" type="barrier"'
        )
        if image_file:
            lines.append(
                f'  <poi {attrs} imgFile="{image_file}" width="{size}" height="{size}"'
                f' angle="{-barrier.angle_deg:.1f}"/>'
            )
        else:
            # imgFile 이 없을 때는 SUMO 가 색 사각형으로 그린다. 도시 전체를
            # 볼 때도 눈에 띄도록 기본 크기보다 크게 잡는다.
            lines.append(
                f'  <poi {attrs} color="255,150,20" width="{size}" height="{size}"'
                f' imgFile="" fill="1"/>'
            )
    lines.append("</additional>")
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


__all__ = (
    "BLOCKABLE_CLASSES",
    "Barrier",
    "BarrierConfig",
    "apply_barriers",
    "choose_barriers",
    "write_poi_additional",
)
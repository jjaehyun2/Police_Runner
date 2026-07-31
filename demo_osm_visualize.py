"""OSM 도로 추격 데모 시각화 스크립트 (오프라인).

대전 오프라인 픽스처 네트워크를 준비 → 경찰 6대 baseline 정책으로 에피소드 실행 →
matplotlib 프레임과 요약 그림을 out_demo/ 폴더에 저장한다. 외부 OSM 접근 없음.
"""

from __future__ import annotations

from pathlib import Path

from pursuit_evasion_rl.osm_demo.coarsening import coarsen_raw_graph, prepare_model_network
from pursuit_evasion_rl.osm_demo.fixtures import daejeon
from pursuit_evasion_rl.osm_demo.metrics import summarize_metrics
from pursuit_evasion_rl.osm_demo.models import EpisodeConfig
from pursuit_evasion_rl.osm_demo.policies import BaselinePolicePolicy
from pursuit_evasion_rl.osm_demo.rendering import FontChoice, OSMRenderer, discover_korean_font
from pursuit_evasion_rl.osm_demo.runner import run_batch, run_single_episode

OUT = Path("out_demo")


def main() -> None:
    OUT.mkdir(exist_ok=True)

    # 1) 대전 오프라인 원본 그래프 → 결정노드 코어싱 → 검증된 Model_Network
    prepared = prepare_model_network(coarsen_raw_graph(daejeon()))
    network = prepared.network
    print(f"[prepare] intersections={len(network.intersections)} segments={len(network.segments)}")

    # 2) 에피소드 설정과 경찰 6대 baseline(최단거리) 정책
    config = EpisodeConfig(
        dt_s=1.0,
        police_speed_mps=14.0,
        fugitive_speed_mps=11.0,
        capture_radius_m=25.0,
        max_steps=40,
    )
    policy = BaselinePolicePolicy(network)

    # 한글 폰트가 있으면 사용, 없으면 영어 fallback
    font = discover_korean_font()
    print(f"[font] family={font.family} korean={font.supports_korean}")
    renderer = OSMRenderer(network, config, font=font)

    # 3) 대표 에피소드 1건을 프레임 시퀀스로 렌더링
    record = run_single_episode(network, config, policy, run_id="demo", run_seed=7)
    print(f"[episode] outcome={record.outcome.value} steps={len(record.transitions)}")
    frames = renderer.render_episode_frames(record)
    for i, fig in enumerate(frames):
        renderer.save_figure(fig, str(OUT / f"frame_{i:03d}.png"))
    print(f"[frames] {len(frames)} PNG saved")

    # 4) 여러 에피소드를 배치로 돌려 요약 통계 그림 생성
    batch = run_batch(network, config, policy, run_id="demo-batch", run_seed=7, episode_count=12)
    summary = summarize_metrics(batch, network, policy_kind="baseline", minimum_evaluation_episodes=1)
    print(f"[batch] outcomes={dict(summary.outcomes)} rates={dict(summary.rates)}")

    renderer.save_figure(renderer.render_summary(summary, representative_record=batch[0]),
                         str(OUT / "summary.png"))
    print(f"[summary] saved -> {OUT / 'summary.png'}")

    print("\n생성된 파일:")
    for path in sorted(OUT.iterdir()):
        print(f"  {path}  ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

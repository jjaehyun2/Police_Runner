# Police Runner

> 도주 차량을 **여러 순찰차가 협력해서 포위·검거**하는 다중 에이전트 강화학습 연구 저장소.
> 실제 대전 도로망을 SUMO 교통 시뮬레이터에 올려, 배경 교통 속에서 추격 정책을 검증한다.

---

## 이 저장소가 하는 일

| | |
|---|---|
| **문제** | 도주 차량과 순찰차의 속도가 같으면 단순 추격으로는 절대 못 잡는다 |
| **접근** | MAPPO 기반 다중 에이전트 학습 — 순찰차들이 길목을 나눠 맡아 **사전 차단** |
| **환경** | 대전 실도로 2,630 엣지 / 교차로 1,031 / 신호 84개 (OpenStreetMap → SUMO) |
| **현재 상태** | SUMO 파이프라인·관제 화면·보상 설계 완료. 학습 정책 재학습은 진행 예정 |

설계 배경과 채택 근거는 **[`process.md`](process.md)**, 실행 방법 전체는 **[`RUNNING.md`](RUNNING.md)** 에 있다.

---

## 빠른 시작

### 0. 설치 (한 번만)

```powershell
py -3.12 -m pip install eclipse-sumo traci sumolib torch fastapi uvicorn numpy matplotlib pytest
```

`eclipse-sumo` 휠 안에 SUMO 바이너리(`sumo`, `sumo-gui`, `netconvert`)가 들어 있다.
**SUMO 별도 설치나 `SUMO_HOME` 설정은 필요 없다.**

확인:

```powershell
py -3.12 -c "import sumolib; print(sumolib.checkBinary('sumo'))"
```

### 1. 관제 화면 데모 — 제일 먼저 볼 것

터미널 **두 개**를 쓴다.

```powershell
# 터미널 A — 시뮬레이션 (--gui 는 넣지 않는다)
py -3.12 scripts\sumo_demo\run_demo.py --episodes 1 --max-steps 400 --background 320 --realtime 0.25

# 터미널 B — 화면 서버
py -3.12 scripts\sumo_demo\dashboard\server.py --port 8020
```

브라우저에서 <http://127.0.0.1:8020> 을 연다.

첫 실행은 대전 OSM을 SUMO 도로망으로 변환하느라 수십 초 걸리고, 이후에는 캐시를 쓴다.

화면에서 보이는 것:

```
빨간 차        도주 차량          회색 차        배경 교통
파란 차        순찰차 6대         주황 줄무늬판   도로 차단(바리케이드) — "차단" 라벨
초록/빨강 점    신호등 현재 현시    원             검거 반경
```

### 2. 숫자만 빠르게 (화면 없이)

```powershell
py -3.12 scripts\sumo_demo\run_demo.py --episodes 12 --max-steps 400 --background 280
```

```
network: edges=2630 nodes=1031 traffic_lights=84
seed=7   outcome=capture  steps=94  background=217  min_sep=19.0m p50=0.01ms
...
scored=12 capture=12 (100%) escape=0 timeout=0  [void=0 excluded]
```

주요 옵션(`--barriers`, `--gui`, `--seed` 등)과 SUMO 자체 창으로 보는 법은
[`RUNNING.md`](RUNNING.md) 에 정리되어 있다.

### 3. 테스트

```powershell
py -3.12 -m pytest tests/test_sumo_env.py -q            # SUMO 환경 24개 (실제 SUMO 연결, 약 30초)
py -3.12 -m pytest tests/research -q                     # 연구층 전체 (약 4분)
```

---

## 폴더 구조

```
README.md                 이 문서 — 개요와 빠른 시작
RUNNING.md                실행 가이드 (옵션·SUMO-GUI·문제 해결)
process.md                설계서 (파이프라인·보상·채택 근거)
PROJECT_REPORT.md         프로젝트 경과 보고
제안서_치안아이디어.md      치안 아이디어 제안서

pursuit_evasion_rl/       ── 패키지 본체
  sumo_env/                 SUMO 환경
    net_builder.py            OSM 캐시 → SUMO 도로망 (내용 해시 캐싱)
    traffic.py                배경 교통 수요 생성
    environment.py            TraCI 환경 (dispatch 방식 배차)
    observations.py           교통 인지 관측 37차원
    policies.py               규칙기반 협력 요격 정책
    barriers.py               도로 차단(바리케이드) 배치
    scene.py                  화면용 장면 스냅샷
  research/                 연구층 (보상 변형·지표·학습)
    variants/rewards.py       감사된 기준 보상 (변경 금지)
    variants/remediation.py   S1/S2 보상 수정
    variants/road_dynamics.py 구간별 속도 (동적 도로 변수)
    variants/safety.py        안전 보상 (주거지 회피·위험 유도 방지)
    metrics/kpi.py            3축 KPI
    traffic/its_client.py     ITS 실시간 교통 API 연동
  road_pursuit/             도로 추격 환경과 보상 계산
  continuous_env/           연속 공간 환경
  training/                 학습 알고리즘 (MAPPO·self-play)

scripts/                  ── 실행 스크립트
  sumo_demo/run_demo.py           데모 실행
  sumo_demo/dashboard/server.py   관제 대시보드 서버
  sumo_demo/open_sumo_gui.py      SUMO 자체 창 런처
  deck/                           발표용 수치·차트 생성
  research/real_scale/            재학습 런처

tools/                    ── 일회성 진단·시각화 스크립트 (본 흐름과 무관)
docs/
  pipeline.html             파이프라인 한 장 요약
  images/                   문서용 그림
  screenshots/              관제 화면 캡처
tests/                    테스트
configs/                  실험 설정
```

### 루트에 남은 스크립트에 대하여

`demo_pursuit.py`, `eval_trained.py`, `run_*.py` 등은 서로를
`from demo_pursuit import ...` 형태로 **형제 임포트**한다. 이 구조에서는 저장소 루트가
곧 임포트 경로라서, 일부만 하위 폴더로 옮기면 조용히 `ImportError` 가 난다.
그래서 이번 정리에서는 **임포트 관계가 없는 일회성 스크립트만** `tools/` 로 옮겼고,
서로 얽힌 진입점들은 루트에 그대로 두었다.

`train_osm_pursuit.py` · `eval_trained.py` · `diag_episode.py` · `render_zoom.py` 네 개는
패키지 CLI 를 얇게 감싼 래퍼이고, **루트에 있어야 한다는 것을 테스트가 강제한다**
(`tests/research/test_trainer_cli_integration.py`). 옮기면 임포트가 아니라 그 테스트가 깨진다.

더 깊은 재배치는 임포트를 패키지 경로로 바꾸는 별도 작업으로 다루는 편이 안전하다.

---

## 무엇이 검증되었나

대전 실도로, 배경차량 280대, 12 에피소드 (`--episodes 12 --max-steps 400 --background 280`):

| 항목 | 값 |
|---|---|
| 검거 | **12 / 12** |
| 탈출 / 시간초과 / 무효 | 0 / 0 / 0 |
| 검거 시 최근접 거리 | 15.7 ~ 25.0 m (검거 반경 25 m) |
| 에피소드 중 배경차량 | 83 ~ 256대 |
| 의사결정 지연 | p50 0.01 ms / p99 89.3 ms (n=1,475) |

**이 수치는 학습된 정책이 아니라 규칙기반 포위 정책의 것이다.** 학습 정책이 넘어야 할
기준선이며, 이 데모가 증명하는 것은 성능이 아니라 **메커니즘이 실제 교통 속에서 작동한다**는
사실이다. 학습 정책의 수치는 재학습 후 확정한다.

---

## 자주 겪는 문제

| 증상 | 원인과 해결 |
|---|---|
| 화면이 계속 "대기 중" | 터미널 A(시뮬레이션)가 아직 도로망을 변환 중이다. 첫 실행은 수십 초 걸린다 |
| 바뀐 화면이 반영되지 않음 | 브라우저 캐시. `Ctrl+Shift+R` 로 강제 새로고침 |
| `PermissionError` (파일 교체 실패) | 대시보드가 파일을 읽는 중 발생. 러너가 자동 재시도하므로 무시해도 된다 |
| SUMO 창에 지도만 보이고 차가 없음 | SUMO-GUI 는 배경 지도만 그린다. 추격 재생은 `open_sumo_gui.py --live` 를 쓴다 |

더 자세한 항목은 [`RUNNING.md`](RUNNING.md) §6 에 있다.

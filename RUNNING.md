# 실행 가이드 (RUNNING.md)

> 위치: `C:\Users\dmsak\Police_Runner` · 브랜치 `feat/mentoring-feedback`
> 설계 배경은 `process.md`, 이 문서는 **돌리는 방법**만 다룹니다.

---

## 0. 한 번만 하는 준비

```powershell
cd C:\Users\dmsak\Police_Runner
py -3.12 -m pip install eclipse-sumo traci sumolib torch fastapi uvicorn numpy matplotlib pytest
```

`eclipse-sumo` 휠에 SUMO 바이너리(sumo, sumo-gui, netconvert)가 같이 들어옵니다.
별도 SUMO 설치나 `SUMO_HOME` 설정이 필요 없습니다.

확인:

```powershell
py -3.12 -c "import sumolib; print(sumolib.checkBinary('sumo'))"
```

---

## 1. 제일 먼저 볼 것 — 관제 화면 데모

**터미널 2개**를 씁니다.

### 터미널 A — 시뮬레이션

```powershell
cd C:\Users\dmsak\Police_Runner
py -3.12 scripts\sumo_demo\run_demo.py --episodes 1 --max-steps 400 --background 320 --realtime 0.25
```

`--gui`는 **넣지 마세요.** SUMO 자체 창은 띄우지 않고 데이터만 뽑습니다.
`--realtime` 이 재생 속도입니다 (0.25 = 스텝당 0.25초, 숫자를 키우면 느려짐).

첫 실행은 대전 OSM을 SUMO 도로망으로 변환하느라 수십 초 걸리고, 이후에는 캐시를 씁니다.

### 터미널 B — 화면 서버

```powershell
cd C:\Users\dmsak\Police_Runner
py -3.12 scripts\sumo_demo\dashboard\server.py --port 8020
```

### 브라우저

<http://127.0.0.1:8020>

### 화면 보는 법

| 요소 | 의미 |
|---|---|
| 빨간 차 + 발광 + 점선 원 | 도주차량. 점선 원이 검거 반경 25 m, 바깥 흐린 원이 250 m 포위선 |
| 파란 차 (지붕 경광등) | 경찰차 6대. 차 위 라벨이 P0~P5 |
| 회색 차 | 배경 일반차량 (기본 320대) |
| 파란 곡선 화살표 | 각 경찰차 → 배정된 차단 지점 |
| 3구 신호기 | 실제 SUMO 신호. 점등된 색이 현재 현시 |
| 오른쪽 패널 | 6대 권고 / 추론 지연 / 이벤트 로그 |
| 위쪽 타일 | 경과·배경차량·최근접 경찰·포위 완성률·상태 |

### 지도 조작

| 조작 | 동작 |
|---|---|
| 마우스 휠 | 확대·축소 |
| 드래그 | 화면 이동 (자유 이동 모드로 전환) |
| 더블클릭 또는 **시점 초기화** 버튼 | 도주차량 추적 모드로 복귀 |

왼쪽 아래 HUD에 현재 카메라 모드와 배율이 표시됩니다.

---

## 2. 숫자만 빠르게 (헤드리스)

```powershell
py -3.12 scripts\sumo_demo\run_demo.py --episodes 12 --max-steps 400 --background 280
```

```
network: edges=2630 nodes=1031 traffic_lights=84
seed=7   outcome=capture  steps=94  background=217  min_sep=19.0m p50=0.01ms
...
scored=12 capture=11 (92%) escape=0 timeout=1  [void=0 excluded]
```

### 주요 옵션

| 옵션 | 뜻 | 기본값 |
|---|---|---|
| `--episodes N` | 에피소드 수 | 1 |
| `--max-steps N` | 에피소드당 최대 스텝(=초) | 450 |
| `--background N` | 배경차량 수 | 300 |
| `--realtime S` | 스텝당 대기 초 (화면으로 볼 때 0.2~0.3) | 0 |
| `--seed N` | 시작 시드 | 7 |
| `--gui` | SUMO 자체 창도 함께 띄움 (보통 불필요) | 꺼짐 |
| `--no-live` | 화면용 파일을 쓰지 않음 | 씀 |

> SUMO 자체 GUI(`--gui`)는 교통공학용이라 차량을 실제 축척으로 그립니다.
> 도시 한 구역이 들어오는 배율에서 5 m 승용차는 1픽셀도 안 되므로 빈 도로지도처럼
> 보입니다. 발표용으로는 위의 관제 화면을 쓰세요.

---

## 3. 테스트

```powershell
# SUMO 환경 (실제 SUMO 연결, 약 30초)
py -3.12 -m pytest tests\test_sumo_env.py -q

# 보상 수정 (S1/S2)
py -3.12 -m pytest tests\research\test_remediation.py -q

# 동적 도로 변수 / 안전 보상·KPI
py -3.12 -m pytest tests\research\test_road_dynamics.py tests\research\test_safety_kpi.py -q

# 연구층 전체 (약 4분)
py -3.12 -m pytest tests\research -q
```

> `tests\test_actions.py`, `tests\test_road_network.py`, `tests\test_osm_demo_osm_source.py`는
> 이 체크아웃에 없는 모듈(`pursuit_evasion_rl.env`)과 `osmnx`를 요구해 수집 단계에서 실패합니다.
> 이번 작업 전부터 있던 상태이고 우리가 건드린 코드와 무관합니다.

---

## 4. 발표 자료용 수치·그림

```powershell
# 검거율 재계산 (커밋된 실험 결과에서, 재학습 불필요)
py -3.12 scripts\deck\recompute_capture_stats.py
py -3.12 scripts\deck\make_capture_chart.py

# 추론 지연 벤치마크 (FastAPI 실제 호출 1000회)
py -3.12 scripts\deck\bench_latency.py
py -3.12 scripts\deck\make_latency_chart.py
```

산출물은 `scripts\deck\out\` 에 생깁니다:
`capture_vs_trivial.png`, `capture_vs_smart.png`, `latency_dist.png`,
`capture_stats.json`, `latency_stats.json`.

> **주의**: 검거율 차트는 베이스라인 계열이 2개이고 **절대 섞으면 안 됩니다**.
> 단순 베이스라인 대비 +30.1pp지만, greedy_intercept 대비로는 12~17pp 뒤집니다.
> 두 장을 따로 그리는 이유가 이것입니다.

---

## 5. 재학습 (형 머신에서, 시간 오래 걸림)

이 체크아웃에는 **봉인된 대전 지도 스냅샷과 연구층 체크포인트가 없습니다**
(`cache/research/maps/`, `artifacts/research/checkpoints/` 는 gitignore + 클러스터 보관).
따라서 아래는 그 파일들이 있는 환경에서만 돌아갑니다.

```powershell
# S1/S2 보상 수정본
py -3.12 scripts\research\real_scale\run_seed_remediated.py 0     # 시드 0~4

# + 동적 도로 변수 + 안전 보상
py -3.12 scripts\research\real_scale\run_seed_dynamic.py 0
```

실시간 교통 데이터를 쓰려면(선택):

```powershell
$env:ITS_API_KEY = "data.go.kr 에서 발급받은 키"
$env:PURSUIT_ITS_BBOX = "127.390,127.402,36.350,36.362"
```

키가 없으면 자동으로 도로등급 기반 샘플링으로 폴백합니다.

---

## 6. 자주 겪는 문제

| 증상 | 원인 / 해결 |
|---|---|
| `No module named 'traci'` | 0번 준비 단계의 pip 설치 |
| `netconvert failed` | 첫 실행 시 변환 실패. `cache\sumo` 폴더를 지우고 재실행 |
| 콘솔에 `ChangeTarget failed ... unreachable` | SUMO가 찍는 경고. 일방통행 도로에서 정상적으로 발생하며 코드가 처리합니다 |
| 대시보드가 "대기 중"에서 안 바뀜 | 터미널 A의 시뮬레이션이 안 돌고 있음. `--no-live` 옵션을 뺐는지 확인 |
| SUMO-GUI가 너무 빨라 안 보임 | `--realtime 0.15` 로 올리기 |
| 첫 실행이 느림 | OSM → SUMO 변환 중. 두 번째부터는 캐시 재사용 |

---

## 7. 어디에 뭐가 있나

```
process.md                          설계서 (파이프라인·보상·채택 근거)
RUNNING.md                          이 문서

pursuit_evasion_rl/
  sumo_env/                         SUMO 환경 (신규)
    net_builder.py                    OSM 캐시 → SUMO 도로망
    traffic.py                        배경교통 수요 생성
    environment.py                    TraCI 환경 (dispatch 방식)
    observations.py                   교통 인지 관측 37D
    policies.py                       규칙기반 협력 요격
  research/
    variants/remediation.py           S1/S2 보상 수정
    variants/road_dynamics.py         구간별 속도 (멘토링 T1)
    variants/safety.py                안전 보상 (멘토링 T2)
    metrics/kpi.py                    3축 KPI (멘토링 T6)
    traffic/its_client.py             ITS 실시간 교통 API

scripts/
  sumo_demo/run_demo.py             데모 실행
  sumo_demo/dashboard/server.py     관제 대시보드
  deck/                             발표용 수치·차트
  research/real_scale/              재학습 런처

tests/
  test_sumo_env.py                  SUMO 환경 16개
  research/test_remediation.py      보상 수정 11개
  research/test_road_dynamics.py    동적 도로 11개
  research/test_safety_kpi.py       안전·KPI 33개
```
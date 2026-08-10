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

## 2-1. 도로 차단(바리케이드)

에피소드마다 무작위 위치에 통행 차단 구간이 생깁니다. 공사·사고·집회로 막힌 길이자
경찰 차단선이기도 하고, 매번 지형이 조금씩 달라지므로 정책이 특정 도로 구성에
과적합하는 것을 막습니다(도메인 랜덤화).

```powershell
# 차단 12곳으로 늘리기
py -3.12 scripts\sumo_demo\run_demo.py --episodes 1 --barriers 12 --realtime 0.25

# 차단 없이
py -3.12 scripts\sumo_demo\run_demo.py --episodes 1 --barriers 0
```

화면에는 **주황색 줄무늬 판 + 발광**으로 표시되고 "차단" 라벨이 붙습니다.
왼쪽 아래 HUD와 오른쪽 위 범례에도 개수가 나옵니다.

차단 지점은 다음 두 조건을 지켜서 고릅니다.

| 조건 | 이유 |
|---|---|
| 도주자 시작점에서 500 m 밖 | 시작하자마자 갇히면 그 에피소드는 정책을 평가하지 못함 |
| 출구가 2개 이상인 도로만 | 막다른 길을 막으면 그 너머 지역이 통째로 고립됨 |

---

## 2-2. 차단 아이콘을 그림으로 바꾸기 (SUMO XML)

SUMO 는 `additional` XML 의 POI 에 **`imgFile` 속성으로 PNG 를 띄울 수 있습니다.**
러너는 매 에피소드 `scripts\sumo_demo\out\barriers.add.xml` 을 자동으로 만듭니다.

```powershell
# 아이콘 지정해서 실행 (PNG 경로는 절대경로 또는 XML 기준 상대경로)
py -3.12 scripts\sumo_demo\run_demo.py --episodes 1 --barriers 8 ^
    --barrier-image C:\Users\dmsak\Police_Runner\assets\barrier.png
```

만들어지는 XML 은 이런 모양입니다.

```xml
<additional>
  <poi id="barrier0" x="1334.20" y="2755.20" layer="20" type="barrier"
       imgFile="barrier.png" width="18.0" height="18.0" angle="-88.8"/>
</additional>
```

이 파일을 SUMO 자체 창에서 보려면 `--gui` 와 함께 additional 로 넘기면 됩니다.

```powershell
# SUMO-GUI 로 직접 열기 (네트워크 경로는 cache\sumo\<해시>
etwork.net.xml)
& "$env:SUMO_HOME\bin\sumo-gui.exe" `
    -n cache\sumo\<해시>\network.net.xml `
    -a scripts\sumo_demo\out\barriers.add.xml
```

> `SUMO_HOME` 은 pip 로 설치한 경우
> `C:\Users\dmsak\AppData\Local\Programs\Python\Python312\Lib\site-packages\sumo` 입니다.
> 네트워크 해시 폴더 이름은 `py -3.12 -c "from pursuit_evasion_rl.sumo_env.net_builder import *; print(build_network(largest_cached_snapshot('cache')).net_path)"` 로 확인할 수 있습니다.

**차량 아이콘까지 그림으로 바꾸려면** 같은 방식으로 POI 를 쓰는 대신, SUMO 의
`vehicleQuality="3"` (실사 형상) 을 쓰거나 vType 에 `imgFile` 을 지정합니다.
다만 관제 화면(브라우저)은 SUMO 렌더링을 쓰지 않고 직접 그리므로, 그쪽 아이콘을
바꾸려면 `scripts\sumo_demo\dashboard\map.js` 의 `drawCar` / `drawBarriers` 를
수정하면 됩니다.

---

## 2-3. SUMO 자체 창(SUMO-GUI)으로 보기

관제 화면과 SUMO-GUI는 **목적이 다릅니다.**

| | 관제 화면 (브라우저) | SUMO-GUI |
|---|---|---|
| 목적 | 발표·시연 | 물리 검증 |
| 차량 크기 | 과장 (읽히도록) | 실제 축척 |
| 도시 전체 볼 때 | 차량이 또렷함 | 차량이 점보다 작음 |
| 보이는 것 | 포위 상황·권고·지표 | 차선·신호 현시·차간거리 |

"진짜 교통 시뮬레이터 위에서 돌고 있다"를 보여줄 때 SUMO-GUI를 씁니다.

### 방법 1 — 전용 런처 (권장)

```powershell
cd C:\Users\dmsak\Police_Runner

# 배경교통 + 차단만 있는 지도 (추격 차량 없음)
py -3.12 scripts\sumo_demo\open_sumo_gui.py

# 옵션 지정
py -3.12 scripts\sumo_demo\open_sumo_gui.py --background 400 --barriers 12 --delay 200

# 추격까지 함께 재생
py -3.12 scripts\sumo_demo\open_sumo_gui.py --live --barriers 8
```

해시가 붙은 네트워크 경로를 직접 찾을 필요 없이 알아서 빌드하고 엽니다.

### 방법 2 — run_demo 에서 바로

```powershell
py -3.12 scripts\sumo_demo\run_demo.py --gui --episodes 1 --realtime 0.15 --barriers 8
```

브라우저 관제 화면도 **동시에** 갱신되므로, SUMO-GUI와 관제 화면을 나란히 놓고
같은 에피소드를 두 관점으로 보여줄 수 있습니다.

### SUMO-GUI 조작

| 조작 | 동작 |
|---|---|
| 마우스 휠 | 확대·축소 (**처음엔 반드시 확대하세요** — 도시 전체 배율에서는 차가 안 보입니다) |
| 마우스 드래그 | 이동 |
| 차량 우클릭 → Show Parameter | 속도·경로·차종 확인 |
| 차량 우클릭 → Start Tracking | 그 차를 카메라가 따라감 (도주차 `F0` 추천) |
| 상단 ▶/⏸ | 재생·일시정지 · 옆 슬라이더로 속도 조절 |
| View → Vehicles → Exaggerate | 차량 크게 그리기 (기본 뷰에서 차가 안 보일 때) |

차량 색: **파랑=경찰(P0~P5) · 빨강=도주차(F0) · 노랑=배경차**

### 차단 지점이 안 보일 때

차단은 POI로 표시됩니다. `View settings → POIs` 에서 크기를 키우거나,
`--barrier-image` 로 PNG 아이콘을 지정하면 눈에 잘 띕니다(§2-2).

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
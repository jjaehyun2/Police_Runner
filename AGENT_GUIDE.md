# AGENT_GUIDE: 현재 작업을 이어받는 AI 에이전트용 안내

이 문서는 이 저장소를 처음 읽는 AI 코딩 에이전트(와 사람)가 **지금 시점의 작업**을 빠르게 따라오도록 만든 지도입니다.
저장소에는 예전 MAPPO 실험 코드(`pursuit_evasion_rl/`, 루트의 `run_*.py`, `README.md`의 설명)가 많이 남아 있습니다. **2026년 10월 기준 실제로 쓰는 코드는 아래에 적은 파일들뿐**이니, 나머지는 필요할 때만 보세요.

- 기준 시점: 2026-10-14
- 배포 사이트: https://express-krypton-ciz5965.vercel.app
- 상세 실험 기록과 수치: [`HANDOFF_CODEX.md`](HANDOFF_CODEX.md) (이 문서와 겹치는 내용은 그쪽이 원본입니다)

---

## 0. 읽는 순서

1. 이 문서의 1~2절. 무엇을 만들고 있고 어떤 규칙을 지키는지 봅니다.
2. 3절. 배포된 사이트가 어떤 파일로 돌아가는지 봅니다.
3. 4~5절. 시뮬레이터와 정책(학습 전략)을 봅니다.
4. 실제 수치가 필요하면 `HANDOFF_CODEX.md` 4절(정책 비교), 7.y·7.z절(바리케이드)을 읽습니다.

경로 약어: **`BG/` = `scripts/research/real_scale/beat_greedy/`**. 아래 경로는 별도 표시가 없으면 저장소 루트 기준입니다.

---

## 1. 프로젝트 한눈에

- **문제**: 실제 도로망에서 도주차 1대를 순찰차 N대가 포위해 검거합니다. 도주차와 경찰차의 속도는 같습니다(16 m/s. `multi_eval`이 `ME_VF`=16으로 `BG_FUGITIVE_SPEED`를 설정합니다. `common.py` 단독 기본값은 9 m/s입니다). 그래서 단순 추격으로는 잡기 어렵고, 탈출로를 먼저 끊어야 합니다.
- **현재 최고 정책**: 역할 선택 정책(`BG/macro.py`)입니다. 경찰차마다 "직접 추격"할지, 몇 번 탈출로의 "차단 지점을 지킬지"를 작은 MLP로 점수 매겨 고르고, 이동은 최단경로로 합니다. 이 MLP의 가중치는 **GA(진화 탐색)로 찾았습니다.** 강화학습(GRPO, PPO, ES 등)은 시도했지만 개선이 없었습니다. **"RL로 학습했다"고 쓰면 안 됩니다.**
- **시뮬레이터 두 개**:
  - 빠른 자체 시뮬레이터 `BG/multi_sim.py`: 학습과 대량 평가에 씁니다.
  - SUMO(TraCI) 실도로 시뮬레이터 `BG/sumo/sumo_sim.py`: 최종 검증과 시연에 씁니다.
- **시연**: 대전 둔산(`daejeon_demo` 지도)에서 웹 대시보드로 합니다. 지도에 경찰차, 도주차, 바리케이드를 놓고 SUMO로 추격을 돌려 실시간으로 봅니다.
- **최근 작업**: 순찰차 바리케이드(1대당 1개)를 넣었습니다.
  - 규칙 기반 설치를 만들었고, 역할 정책과 함께 GA로 학습하는 실험도 2차까지 했습니다.
  - 결론은 **"현재 설정(포위 상태에서 시작해, 추격 도중에 설치)에서는 바리케이드가 검거율을 올리지 못한다"**입니다.
  - 사이트는 기존 GA 정책과 규칙 기반 바리케이드 옵션(프로토타입)으로 운영 중입니다.

## 2. 반드시 지킬 규칙

| 규칙 | 내용 |
|---|---|
| 보류 지도 | `daejeon_test`, `seoul_zero_shot`, `busan_zero_shot`은 **학습과 모델 선택에 쓰지 않습니다.** 최종 평가에만 씁니다 |
| 시나리오 | **도주차 1대 고정, 경찰 수 N만 바뀝니다.** 정책은 N에 무관해야 합니다(`CandidateScorer(kind="size")`) |
| 기본 시작 배치 | 포위 시작(ring)입니다. 도주차는 지도 중앙, 경찰은 도로 거리 400–700 m에서 방향별로 둘러쌉니다(`ME_PLACEMENT=ring`, `multi_sim.make_ring_case`) |
| 보고 | 결과는 정직하게, 실패와 한계도 숫자로 씁니다. 사용자 응답은 한국어로 합니다 |
| 비교 방법 | 정책 비교는 **같은 시드의 같은 케이스로 짝지어(paired)** 합니다. 이긴 판과 진 판 수를 같이 봅니다 |
| 프로세스 종료 | `pkill -f`를 쓰지 마세요. 자기 셸까지 죽습니다. PID로 종료하세요 |

### 지도 분할 (`configs/research/maps/*.yaml`)

| 용도 | 지도 |
|---|---|
| 학습(fitness) | `daejeon_train`, `daegu_train`, `ulsan_train` |
| 검증(모델 선택) | `daejeon_validation`, `incheon_validation`, `suwon2_validation` |
| 보류(최종 평가만) | `daejeon_test`, `seoul_zero_shot`, `busan_zero_shot` |
| 시연 | `daejeon_demo` (대전 둔산. 학습에 쓴 대전 지역과 겹치므로 보류 평가에 쓰면 안 됩니다) |

지도 데이터(도로망 스냅샷)는 `cache/research/maps/<hash>/`에 있습니다(`model_network.json`, `raw_osm.json` 등). yaml의 `network_path`가 이 위치를 가리킵니다. 로딩은 `BG/common.py`의 `load_network(name)`이 합니다.

---

## 3. 배포된 사이트 구조

```
[브라우저] ──HTTPS──> Vercel 정적 사이트 (demo/web 빌드 결과)
     │  fetch(VITE_API_BASE + /api/...)
     ▼
Cloudflare quick tunnel (https://*.trycloudflare.com, 주소가 바뀔 수 있음)
     ▼
GPU 서버의 FastAPI (demo/server/app.py, uvicorn :8600)
     ├─ engine="sumo"  → BG/sumo/sumo_sim.py  (SUMO + TraCI)
     └─ engine="point" → BG/multi_sim.py      (빠른 자체 시뮬레이터)
```

### 3.1 프런트엔드: `demo/web/` (Vite + TypeScript + Leaflet)

| 파일 | 역할 |
|---|---|
| `demo/web/index.html` | 화면 구조입니다. 지도 선택, 배치 모드(경찰차·도주차·바리케이드), 정책 선택, 시뮬레이터(SUMO/빠른), 순찰차 바리케이드 토글(`#carry`), 추격 시작, 재생 컨트롤, 경찰별 임무 표, 실행 기록 |
| `demo/web/src/main.ts` | 앱 로직 전부입니다. 서버 호출(`/api/maps`, `/api/map/{name}`, `/api/example`, `/api/run`), NDJSON 스트림 수신, 프레임 재생, 지도 레이어(경찰·도주차·도주 가능 영역·열린 출구·바리케이드), 녹화 시연 재생, 오류 표시 |
| `demo/web/src/style.css` | 토스 스타일 디자인입니다. 색은 `:root` 토큰으로 정의하고 다크 모드를 지원하며, 글꼴은 Pretendard입니다. `main.ts`가 `getComputedStyle`로 읽는 토큰(`--road`, `--region`, `--exit`, `--fugitive`, `--police`, `--target`, `--barricade` 등)의 이름은 바꾸면 안 됩니다 |
| `demo/web/public/fallback.json` | 서버가 꺼져 있을 때 재생하는 **녹화된 SUMO 시연 1판**입니다 |
| `demo/web/scripts/build_fallback.py` | `fallback.json`을 다시 만드는 스크립트입니다 |
| `demo/web/vite.config.ts` | 로컬 개발 시 `/api`를 `localhost:8600`으로 프록시합니다 |
| `demo/web/vercel.json` | Vercel 빌드 설정입니다(`npm run build` → `dist`) |

- 로컬 실행: `cd demo/web && npm install && npm run dev`. 서버(3.2)가 :8600에 떠 있어야 합니다.
- 빌드 확인: `npm run build` (`tsc --noEmit`과 `vite build`).
- **Vercel 설정**:
  - 프로젝트 `express-krypton-ciz5965`, Root Directory `demo/web`
  - 환경변수 `VITE_API_BASE` = 현재 터널 주소
  - GitHub `jjaehyun2/Police_Runner`와 연결되어 있고, **production 브랜치는 `codex/dashboard-deploy-20261012`**입니다. 이 브랜치에 push하면 사이트가 다시 빌드됩니다. `main`에는 PR로 머지합니다.

### 3.2 백엔드: `demo/server/app.py` (FastAPI)

| 엔드포인트 | 설명 |
|---|---|
| `GET /api/health` | 상태 확인 |
| `GET /api/maps` | 지도 목록: `daejeon_demo`(대전 둔산), `seoul_zero_shot`, `busan_zero_shot` |
| `GET /api/map/{name}` | 도로 폴리라인과 노드 좌표(위경도) |
| `POST /api/example` | 예시 배치(포위 시작 케이스) |
| `POST /api/run` | 추격 실행. **NDJSON 스트림**으로 `setup` → `frame`×N → `end`(또는 `error`)를 보냅니다 |

- `RunRequest` 주요 필드:
  - `police`, `fugitive`, `barricades`: [lat, lon] 좌표
  - `carry_barricades`: 순찰차가 규칙에 따라 직접 설치
  - `policy`: `ga`, `encircle`, `greedy`, `assign`
  - `engine`: `sumo`, `point`
  - `profile`: `ideal`(1차로, 신호 없음) 또는 `real`(OSM 차로, 신호)
  - `traffic`: 일반 교통 발생 주기(초)
  - `ignore_red`, `max_seconds`
- 바리케이드는 SUMO 엔진에서만 쓸 수 있습니다. point 엔진에서 요청하면 422를 돌려줍니다.
- `frame` 필드:
  - `t`: 초
  - `p`: 경찰 위치, `f`: 도주차 위치
  - `role`, `tgt`: 경찰별 역할과 목표
  - `reg`: 도주차가 경찰보다 먼저 닿는 영역의 비트열
  - `open`, `ox`: 열린 출구 수와 좌표
  - `bk`: 도주차가 알아챈 바리케이드
  - `kit`: 바리케이드를 아직 들고 있는지
  - `bars`: 바리케이드 위치
  - `bg`: 일반 차량(real + traffic일 때)
- 정책 이름과 체크포인트의 대응은 `POLICIES` 딕셔너리에 있습니다. **사이트의 "GA 역할 정책"은 `rmacro:BG/runs/ms_ga/best.pt`입니다.**
- 실행: `cd demo/server && <venv>/bin/uvicorn app:app --host 127.0.0.1 --port 8600`
- 테스트: `pytest tests/test_demo_server_app.py`. API 계약, 프레임 필드, 바리케이드 422, 양방향 차단을 확인합니다.
- 보호 장치:
  - 동시 실행 수 제한: `PURSUIT_MAX_RUNS`, 기본 6
  - CORS: `PURSUIT_CORS`
  - 토큰(선택): `REQUIRE_TOKEN=1`, `PURSUIT_DEMO_TOKEN`

### 3.3 상시 실행과 공개: `demo/run_public.sh`

- uvicorn과 cloudflared quick tunnel을 띄우고, 둘 중 하나가 죽으면 다시 살립니다.
- `REDEPLOY_ON_URL_CHANGE=1`이면 터널 주소가 바뀔 때 두 가지를 합니다.
  - Vercel 환경변수 `VITE_API_BASE`를 갱신합니다(`vercel env add ... --force`).
  - deploy hook(`demo/logs/deploy_hook`, git에는 없음)을 호출해 재배포합니다.
- 실행: `REDEPLOY_ON_URL_CHANGE=1 setsid nohup demo/run_public.sh > demo/logs/supervisor.log 2>&1 < /dev/null &`
- 현재 터널 주소는 `demo/logs/tunnel_url`, 로그는 `demo/logs/{supervisor,server,tunnel,deploy}.log`에 있습니다. `demo/logs/`는 git에 없습니다.
- 한계:
  - 서버가 꺼지면 SUMO 실행이 멈춥니다. 녹화 시연은 계속 재생됩니다.
  - quick tunnel 주소는 고정이 아닙니다.

---

## 4. 시뮬레이터와 공용 모듈 (`BG/`)

| 파일 | 역할 |
|---|---|
| `BG/common.py` | 지도 로딩 `load_network`. 단일 도주차 봉인 케이스(경찰 6대). 정책 spec 빌더 `_build`(`greedy`, `encircle`, `rmacro:<pt>`, `rmacro:assign`, `macro:<pt>` 등). 속도 상수 `POLICE_SPEED_MPS`, `FUGITIVE_SPEED_MPS` |
| `BG/multi_sim.py` | **빠른 자체 시뮬레이터**(1초 스텝, 검거 반경 25 m). 다루는 것은 다음과 같습니다. 케이스: `make_ring_case`(포위 시작), `make_multi_case`(무작위 시작). `escape_region`: 도주차가 모든 경찰보다 먼저 닿는 교차로 집합. `RecommendBridge`: 팀 정책을 N대로 실행하고, 바리케이드 플래너가 있으면 설치 결정까지 호출. 바리케이드: 경찰용 그래프 `pg`와 도주차용 그래프 `fg`를 따로 두고, 도주차는 80 m 안에서만 바리케이드를 알아채며 U턴 |
| `BG/sumo/sumo_sim.py` | **SUMO 시뮬레이터** `SumoPursuit(map, case, spec, profile, traffic, ignore_red, ..., barricades, carry_barricades).run()`. 경찰은 매초 목표 노드를 정하고, 최단경로를 `setRoute`로 넘겨 목표에서 정지합니다. 검거는 0.25 s마다 25 m 이내로 판정합니다. 바리케이드는 차로마다 정지한 `barricade` 차량을 세워 양방향을 막습니다. 규칙 기반 설치는 `_deploy`(`SUMO_BARRICADE_RULES`, 기본 `leave,stand-in`), 학습 기반 설치는 `_deploy_learned`가 맡습니다 |
| `BG/sumo/build_net.py` | ModelNetwork를 SUMO 도로망으로 1:1 변환합니다. 프로필은 `ideal`/`real`이고, `--traffic`으로 일반 교통을 만듭니다. 결과물 `BG/sumo/nets/`는 git에 없으니 필요하면 이 스크립트로 다시 만드세요 |
| `BG/sumo/eval_sumo.py` | SUMO 대량 평가: `python sumo/eval_sumo.py <map> <eps> <n> <ideal|real> <spec...>` |
| `BG/sumo/demo_gui.py` | 화면이 있는 PC에서 sumo-gui로 한 판을 재생합니다(서버에는 디스플레이가 없음) |
| `BG/obs_ext.py` | 관측 확장, `exit_nodes`(경계 출구), `graph_for`(그래프 캐시) |
| `BG/multi_eval.py` | 평가 진입점입니다. `team_for(spec, net, n)`이 정책을 만들고, `bmacro:<pt>`는 역할 정책에 바리케이드 헤드를 더한 것입니다. `evaluate_mixed`는 판마다 경찰 수를 다르게 해서 돌립니다. 환경변수: `ME_PLACEMENT=ring|random`, `ME_MAX_STEPS`, `ME_WORKERS`, `ME_VF` |
| `BG/viewer/` | 정적 리플레이 뷰어 생성기(`export_ring.py`, `export_sumo.py`, `build_viewer.py`) |

기존 기준 정책(`encircle`, `greedy`)과 그래프 클래스 `_Graph`는 `pursuit_evasion_rl/research/policies/baselines.py`에 있습니다.

---

## 5. 정책과 학습 전략 (시간순)

| # | 전략 | 핵심 파일 | 결과 | 상태 |
|---|---|---|---|---|
| 1 | 규칙 기반: 탐욕 추격, 포위, 배정 규칙 | `pursuit_evasion_rl/research/policies/baselines.py`, `BG/rules.py`, `BG/multi_policies.py` | 기준선. 포위는 대전 test 81.3%(경찰 6대, 무작위 시작) | 비교용 |
| 2 | RL 계열: DAgger/BC → PPO, GRPO, 어텐션 대상 배정 + ES, rollout policy iteration | `BG/bc_train.py`, `BG/ppo_finetune.py`, `BG/grpo_train.py`, `BG/attn.py`, `BG/train_attn.py`, `BG/es_attn.py`, `BG/rpi_attn.py`, `BG/engine.py`, `BG/planner.py` | 규칙을 넘지 못함 | **실패, 재사용 불필요** |
| 3 | 역할 선택 정책 + 무작위 탐색과 GA (대전 데이터만) | `BG/macro.py`, `BG/seed_search.py`, `BG/ga.py` → `BG/runs/ga/best.pt` | 대전 test 91.3%, 하지만 다른 도시로 일반화 약함 | 대체됨 |
| 4 | **여러 도시 GA (지도 규모 무관 특징, `rmacro:`)** | `BG/ms_ga.py` → **`BG/runs/ms_ga/best.pt`** | 대전 test 88.5, 서울 77.9, 부산 76.0(포위 정책 대비 +6~7%p, 경찰 6대) | **현재 최고, 사이트에서 사용** |
| 5 | 경찰 수 무관 GA (`CandidateScorer(kind="size")`, N=3–10) | `BG/nvar_ga.py` → `BG/runs/nvar_ga/best.pt` | 검증 0.834에서 정체. 4번과 비슷한 수준 | 6번의 출발점(warm start) |
| 6a | 바리케이드: 수동 배치 | `sumo_sim.Barricade`, `barricade_at` | 대시보드에서 직접 배치. 도주차는 80 m 안에서만 인지 | 사이트 기능 |
| 6b | 바리케이드: 순찰차가 규칙으로 설치(1대 1개) | `sumo_sim._deploy` | SUMO 경찰 4대 87 → 89%(같은 시드 100판, 2승 0패). 오차 범위 수준 | 사이트의 "순찰차 바리케이드" 옵션 |
| 6c | **바리케이드: 역할 정책과 함께 GA 학습** | `BG/barricade.py`(설치 결정 MLP와 특징 15개), `BG/bar_ga.py`(공동 GA), `BG/bar_report.py`(검증 보고서), `BG/run_bar_pipeline.sh` | 1차: 처음 보는 지도와 SUMO에서 악화(서울 -11%p). 2차: 설치 0개로 수렴, 부산 65.0 → 46.5%. **효과 없음** | 결론 기록됨(`HANDOFF_CODEX.md` 7.z) |

### 5.1 역할 선택 정책 (`BG/macro.py`)

- 6초마다 경찰이 도주차와 가까운 순서대로 역할을 고릅니다. 후보 0은 직접 추격, 1~7은 탈출로 j의 차단 지점입니다.
- `CandidateScorer`(MLP)가 후보마다 점수를 냅니다. GA가 이 가중치를 찾습니다.
- 바리케이드로 막힌 출구는 후보에서 빠지고, 경로도 막힌 도로를 피합니다(`self.closed`).

### 5.2 GA 설정 공통 (`BG/ga.py`, `BG/ms_ga.py`, `BG/nvar_ga.py`, `BG/bar_ga.py`)

- **fitness**: 학습 지도 3개의 평균 검거율입니다. 세대마다 새 케이스를 씁니다.
- **선택**: 상위 몇 개를 검증 지도 3개로 평가해 `best.pt`를 고릅니다.
- 변이는 `ga.mutate`(파라미터 표준편차에 비례한 가우시안), 교차는 `ga.crossover`(은닉 유닛 단위)입니다.
- `bar_ga.py` 2차 버전에서 추가한 것:
  - 바리케이드 후보 제한: 도주 영역을 줄이는 도로만 후보로 둡니다(`barricade.IMPACT_ONLY`).
  - 역할 정책 변이 축소(`BG_ROLE_STEP`).
  - 최종 재선발: 새 검증 케이스와 SUMO로 다시 평가하고, "바리케이드 없음"도 후보에 넣습니다. 매 세대 최고값을 고르면 운 좋게 높게 나온 후보가 뽑히는 문제를 줄이기 위해서입니다.

### 5.3 체크포인트 형식

`torch.save` 딕셔너리 하나에 다음 키가 들어갑니다.
- `kind`, `scorer_kind`, `hidden`, `scorer`(state_dict)
- 바리케이드 정책이면 추가로 `barricade`, `barricade_hidden`
- `meta`

spec 문자열 `rmacro:<path>`(역할 정책)이나 `bmacro:<path>`(역할 + 바리케이드)로 불러옵니다.

---

## 6. 자주 쓰는 명령 (모두 `BG/`에서, 저장소의 venv Python으로)

```bash
# 정책 비교 (자체 시뮬레이터, 포위 시작, 경찰 6대, 300판)
ME_PLACEMENT=ring python multi_eval.py daejeon_validation 300 6 1 rmacro:runs/ms_ga/best.pt encircle greedy

# SUMO 평가 (대전 둔산, ideal 프로필, 경찰 6대, 100판)
python sumo/eval_sumo.py daejeon_demo 100 6 ideal rmacro:runs/ms_ga/best.pt encircle

# 바리케이드 공동 학습 파이프라인 (학습 → 검증 보고서), 작은 규모로 배관 확인
BG_SMOKE=1 BG_POP=6 BG_ELITE=2 BG_FINAL_K=2 ./run_bar_pipeline.sh /tmp/smoke 1
# 실제 실행 (48코어 기준 세대당 약 7.5분)
ME_WORKERS=36 ./run_bar_pipeline.sh $PWD/runs/bar_ga_v3 20
```

---

## 7. 저장소에 있는 것과 없는 것

| 항목 | git에 있음? | 비고 |
|---|---|---|
| 위에 적은 코드 전부 | 있음 | |
| 지도 스냅샷 `cache/research/maps/<hash>/` (10개 지도, 약 27 MB) | 있음 | `cache/`는 .gitignore 대상이지만 이 10개는 강제로 추가했습니다 |
| `BG/runs/ms_ga/best.pt`, `BG/runs/ga/best.pt` | 있음 | 사이트 정책 |
| `BG/runs/nvar_ga/best.pt`, `BG/runs/bar_ga*/best.pt`, `report.md`, `report.json`, `bar_ga.jsonl` | 있음 | 바리케이드 실험 재현과 결과 확인용 |
| SUMO 도로망 `BG/sumo/nets/` | 없음 | `sumo/build_net.py`로 다시 생성 |
| 학습 중간 개체 `BG/runs/*/pop/`, 로그 `BG/logs/`, 결과 JSON `BG/results/` | 없음 | 서버에만 있음 |
| `demo/logs/` (터널 주소, deploy hook) | 없음 | 비밀 정보 포함 |

## 8. 환경

- Python 3.11 venv입니다(서버에서는 `/workspace/cluster/.venv/bin/python`). 시스템 Python에는 shapely가 없어서 실패합니다. 필요한 패키지는 `torch`, `numpy`, `shapely`, `pyproj`, `fastapi`, `uvicorn`, `eclipse-sumo==1.28.0`, `traci==1.28.0`, `sumolib==1.28.0`입니다.
- SUMO는 pip 휠에 바이너리가 들어 있어서 `SUMO_HOME`을 따로 설정할 필요가 없습니다.
- 학습 서버는 48코어이고 디스플레이가 없습니다. 병렬 평가는 `ME_WORKERS`로 조절합니다.
- 프런트엔드는 `demo/web`에서 `npm install`로 설치합니다.

## 9. 남은 일과 아이디어

- **바리케이드 다음 단계(제안)**: 추격 시작 전에 출구나 길목에 미리 설치하는 "검문소" 시나리오, 또는 경찰 수가 적거나 포위가 안 된 시작 배치처럼 바리케이드가 차이를 만들 수 있는 조건에서 다시 평가합니다.
- **SUMO 실도로 조건**(real 프로필, 신호, 교통)에서는 모든 정책이 93~99%로 수렴해 차이가 드러나지 않습니다. 정책 차이가 보이는 설정을 찾는 일이 남아 있습니다(`HANDOFF_CODEX.md` 5절 P2·P3).
- **논문 형식 정리**(PEPCRL-MVP, arXiv 2306.05016 기준 SR/ATS 표): `HANDOFF_CODEX.md` 5절 P5를 보세요.

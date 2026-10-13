# 인계 문서: 경찰 추격 정책 + SUMO 실도로 시뮬레이터

작성: 2026-10-12, Claude Code 세션. Codex가 이어받아 바로 작업할 수 있도록 현재 상태, 실행 방법, 남은 일을 정리했습니다.

---

## 0. 먼저 읽을 것 (사용자 요구사항)

- **응답은 한국어**로 합니다. 결과는 정직하게 보고하고, 실패와 한계도 숫자로 같이 씁니다.
- **현재 최고 정책은 강화학습이 아닙니다.** 역할 선택 MLP(`macro.CandidateScorer`)의 가중치를 **무작위 탐색과 GA(진화 탐색)**로 찾은 것입니다. RL(GRPO, PPO, ES, rollout policy iteration)은 시도했지만 개선이 없었습니다. 보고서나 발표에서 "RL로 학습"이라고 쓰면 안 됩니다.
- **보류 지도(held-out)**: `daejeon_test`, `seoul_zero_shot`, `busan_zero_shot`은 학습과 선발에 쓰지 않습니다. 평가에만 씁니다.
- **가정 시나리오**는 **도주차 1대 고정, 경찰차 수만 변동**입니다. 사용자가 보고 싶은 장면은 이렇습니다.
  - 도주차는 지도 중앙에서 출발합니다.
  - 경찰차는 처음부터 사방을 둘러싼 상태에서 출발합니다.
  - 경찰이 탈출로를 끊어 가며 검거합니다.
  - 이 시나리오의 코드는 `ME_PLACEMENT=ring`, `multi_sim.make_ring_case`입니다.
- **최종 목표**는 **SUMO(TraCI)와 연동한 실도로 시뮬레이터**입니다. **시연은 대전 지역**으로 합니다(`daejeon_demo` 지도).
- 목표 논문은 PEPCRL-MVP(arXiv 2306.05016, `2306.05016v1.pdf`)입니다. 지표는 SR(성공률), ATS(평균 검거시간), 표준편차입니다.

## 1. 환경

- Python venv는 `/workspace/cluster/.venv`입니다. 항상 `/workspace/cluster/.venv/bin/python`을 직접 쓰세요. 시스템 python에는 shapely 같은 패키지가 없습니다.
- SUMO는 pip로 설치돼 있습니다. `eclipse-sumo==1.28.0`, `traci==1.28.0`, `sumolib==1.28.0`이며 바이너리는 `python -c "import sumo; print(sumo.SUMO_HOME)"`의 `bin/`에 있습니다.
- 서버는 48코어이고 **디스플레이가 없습니다.** 그래서 sumo-gui는 서버에서 띄울 수 없고, 화면이 있는 PC에서 `sumo/demo_gui.py`로 실행합니다.
- 작업 디렉터리는 `scripts/research/real_scale/beat_greedy/`입니다. 아래 경로는 모두 여기 기준입니다.
- **주의: `pkill -f <패턴>`을 쓰지 마세요.** 명령줄에 그 패턴이 들어 있는 자기 셸까지 죽습니다(exit 144). 프로세스는 PID로 종료하세요.
  ```bash
  ps -eo pid,args | awk '$2 ~ /python/ && $3=="nvar_ga.py" {print $1}' | xargs -r kill
  ```
- 출력을 `| tail`로 파이프하면 결과가 끝날 때까지 안 보입니다. 긴 작업은 `nohup ... > logs/x.log 2>&1 &`로 돌리고 로그를 확인하세요.
- **아직 git 커밋을 하나도 하지 않았습니다.** `beat_greedy/` 전체, 새 지도 설정 5개, 이 문서가 모두 untracked 상태입니다.

## 2. 코드 지도 (`beat_greedy/`)

| 파일 | 역할 |
|---|---|
| `common.py` | 지도 로딩(`load_network`), 봉인된 단일 도주차 케이스(경찰 6대 고정), 정책 spec 빌더 `_build`, 병렬 평가 `evaluate`. spec 종류: `greedy`, `encircle`, `rmacro:<pt>`, `rmacro:assign`, `rmacro:init<seed>`, `rmacro:sinit<seed>`, `macro:...` |
| `macro.py` | **역할 선택 정책**. 6초마다 경찰이 도주차와 가까운 순서대로 역할을 고릅니다. 후보 0은 직접 추격, 1~7은 탈출로 j의 차단 지점입니다. 이동은 최단경로입니다. `relative=True`이면 지도 규모에 무관한 특징을 씁니다. `CandidateScorer(kind="onehot"|"size")`: size는 경찰 번호 대신 6/N을 입력으로 받아 경찰 수와 무관하게 동작합니다. 팀 크기는 `len(police)`로 자동 설정됩니다 |
| `multi_sim.py` | 경찰 N대 대 도주차 M대 시뮬레이터. 단일 도주차 봉인 환경과 위치가 정확히 일치하는 것을 검증했습니다(30판 3632스텝, 위치 불일치 0). 케이스 생성: `make_multi_case`(무작위 시작), `make_ring_case`(**포위 시작**: 경찰이 도주차에서 도로 거리 400–700 m에 방향별로 1대씩, 도주차는 지도 중앙). `RecommendBridge`는 기존 `recommend`형 정책을 N대로 실행합니다. 봉인 환경과 75판에서 결과·스텝 차이 0이었습니다 |
| `multi_policies.py` | 다중 도주차 기준 정책: nearest, balanced, roles |
| `multi_eval.py` | `python multi_eval.py <map> <eps> <n_police> <n_fug> <spec...>`. 환경변수 `ME_PLACEMENT=ring|random`, `ME_MAX_STEPS`, `ME_WORKERS`, `ME_VF`. 함수: `team_for(spec, net, n)`, `evaluate_mixed`(판마다 다른 N) |
| `ms_ga.py` | 여러 도시 GA. 학습: 대전train·대구·울산. 선발: 대전val·인천·수원. 결과 `runs/ms_ga/best.pt`가 **현재 최고 정책**입니다 |
| `nvar_ga.py` | 경찰 수 무관(kind="size") GA. NS=(3,4,5,6,8,10). `runs/nvar_ga/`에서 실행 중이며 아래 3절을 참고하세요 |
| `sumo/build_net.py` | ModelNetwork를 SUMO 도로망으로 1:1 변환합니다(`n<id>` 노드, `e<id>` 엣지, UTM 좌표 그대로). 모든 회전을 `.con.xml`로 명시합니다. 프로필 `ideal`은 1차로·신호 없음, `real`은 OSM 차로·제한속도·신호 추정(대전 데모 245개)입니다. `--traffic <period>`는 randomTrips와 duarouter로 일반 교통을 만듭니다 |
| `sumo/sumo_sim.py` | `SumoPursuit(map, case, spec, profile, traffic, ignore_red, speed_factor, record, gui).run()`. 정책은 매초 경찰마다 목표 노드(hold 또는 pursue)를 정하고, 우리 그래프의 최단경로를 `setRoute`로 넘깁니다. 경로 끝에는 정지(stop)를 걸어 목표 노드를 지키게 합니다. 도주차(GoalEvader)는 교차로 직전에 다음 엣지를 정합니다. 검거는 0.25 s마다 25 m 이내로 판정합니다. 탈출은 경계 엣지를 완주하면 성립합니다 |
| `sumo/eval_sumo.py` | `python sumo/eval_sumo.py <map> <eps> <n> <ideal|real> <spec...>`. 환경변수 `SE_TRAFFIC`, `SE_IGNORE_RED`, `SE_WORKERS`. SUMO와 자체 시뮬레이터를 같은 케이스로 비교합니다 |
| `sumo/demo_gui.py` | 화면이 있는 PC에서 sumo-gui로 한 판을 재생합니다 |
| `viewer/` | 웹 리플레이 뷰어. `export_ring.py`는 자체 시뮬레이터, `export_sumo.py`는 SUMO 결과를 내보내고, `build_viewer.py`가 페이지를 조립합니다. 붉은 도로는 "도주차가 어느 경찰보다 먼저 닿는 영역"이고, 열린 경계 출구 수를 시간에 따라 보여줍니다 |
| `seed_search.py`, `ga.py`, `train_attn.py`, `es_attn.py`, `rpi_attn.py`, `grpo_train.py`, `ppo_finetune.py` | 과거 실험(RL 실패 기록 포함). 재사용할 필요는 없습니다 |

지도 설정은 `configs/research/maps/*.yaml`에 있습니다. 새로 받은 지도는 다음 5개입니다.
- 학습: `daegu_train`, `ulsan_train`
- 검증: `incheon_validation`, `suwon2_validation`
- 시연: **`daejeon_demo`**(127.370–127.400, 36.336–36.366, 둔산 중심, 노드 1157개, 엣지 3051개). 학습에 쓴 대전 지역과 **겹치므로** 보류 평가용으로 쓰면 안 됩니다

지도를 새로 받는 방법은 이렇습니다.
1. unsealed yaml을 만들고, `polygon_content_hash`를 `pursuit_evasion_rl.research.domain.content_hash(query_geometry)`로 채웁니다.
2. 아래 명령을 실행합니다.
   ```bash
   python -m pursuit_evasion_rl.research.cli.fetch_osm_snapshot <unsealed.yaml> --run-id ... --service-version "osmnx==2.1.1" --sealed-config-output configs/research/maps/<name>.yaml --opt-in
   ```
3. 주의할 점:
   - role 값은 train, validation, test, zero_shot 중 하나만 됩니다.
   - 광주, 청주 영역은 "cyclic virtual routing chain" 오류로 실패했습니다.

## 3. 인계 시점에 돌고 있는 작업

| 작업 | 로그 / 출력 | 확인 방법 |
|---|---|---|
| `nvar_ga.py runs/nvar_ga 20` | `logs/nvar_ga.log`, `runs/nvar_ga/nvar_ga.jsonl`, `best.pt` | 9세대까지 검증 최고값이 0.8344(`warm_m0.pt`)에서 **개선 없음**. 끝나면 결과만 기록하고 우선순위를 낮추세요 |
| SUMO real 평가 4개 조건 | `logs/sumo_real_daejeon.log`, `logs/sumo_real_daejeon2.log` | **완료**. 결과는 4.4절에 있습니다 |

결과 JSON은 `results/`, `sumo/results/`에 쌓입니다.

## 4. 지금까지의 핵심 결과

### 4.1 단일 도주차, 경찰 6대, 봉인 케이스(무작위 시작), 지도당 1000판, 검거율

| 정책 | 대전 test | 서울 | 부산 |
|---|---|---|---|
| 탐욕 추격 `greedy` | 76.3 | 51.2 | 49.7 |
| 포위 `encircle` (기준) | 81.3 | 71.1 | 70.2 |
| 배정 규칙 `rmacro:assign` | 81.7 | 59.0 | 58.3 |
| 대전 데이터로만 찾은 GA `macro:runs/ga/best.pt` | **91.3** | 67.4 | 70.8 |
| **여러 도시 GA `rmacro:runs/ms_ga/best.pt`** | 88.5 | **77.9** | **76.0** |
| 포위 대비 (paired, 95% CI) | +7.2 (±2.4) | +6.8 (±3.0) | +5.8 (±3.3) |

### 4.2 포위 시작(ring), 자체 시뮬레이터, 300판, 검거율

| 지도 · 경찰 수 | greedy | encircle | assign | **ms_ga best** |
|---|---|---|---|---|
| 서울 · 4대 | 36.7 | 34.7 | 41.0 | **56.0** |
| 서울 · 6대 | 63.0 | 49.3 | 61.7 | **77.7** |
| 서울 · 8대 | 67.0 | 53.0 | 66.7 | **84.3** |
| 부산 · 4대 | 34.3 | 32.7 | 43.7 | **44.3** |
| 부산 · 6대 | 39.0 | 31.3 | 48.7 | **64.3** |
| 부산 · 8대 | 46.0 | 44.7 | 59.0 | **71.7** |

### 4.3 경찰 수 가변, 무작위 시작, 대전 validation 500판, 검거율

| 경찰 수 | 3 | 4 | 5 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|
| greedy | 60.8 | 70.6 | 75.4 | 80.8 | 87.2 | 89.4 |
| encircle | 57.2 | 69.6 | 73.6 | 82.4 | 89.6 | 90.0 |
| assign | 64.4 | 74.8 | 80.6 | 85.6 | 92.4 | 93.4 |
| ms_ga best | 64.0 | 76.0 | 80.4 | 85.4 | 92.2 | 94.2 |

이 분포(대전 val, 무작위 시작)에서는 GA와 assign이 거의 같습니다. GA의 이점은 처음 보는 큰 도시와 포위 시작 시나리오에서 크게 드러납니다.

### 4.4 SUMO, `daejeon_demo`, 포위 시작, 경찰 6대, 검거율 (판 수: 1단계 200판, 나머지 150판)

| 조건 | GA (ms_ga) | encircle | greedy | assign |
|---|---|---|---|---|
| 1단계: ideal(1차로, 신호·교통 없음) | **95.5** | 88.0 | 86.0 | 85.0 |
| real, 일반 교통 없음, 신호 준수 | 96.7 | **98.7** | 98.0 | 96.0 |
| real + 혼잡 교통(0.7 s/대), 신호 준수 | 94.0 | 95.3 | 95.3 | 91.3 |
| real + 혼잡 교통 + 긴급주행(신호 무시, speedMode 7) | **94.7** | 92.7 | 92.7 | 94.0 |
| real + 덜 붐빈 교통(2 s/대) | 95.3 | **96.7** | **96.7** | 95.3 |

해석은 이렇습니다.
- 실제 도로 조건(신호, 정체)에서는 도주차도 막히기 때문에 **모든 정책이 93~99%로 수렴**하고, 정책 간 차이가 거의 사라집니다.
- 실제 탈출 판은 2~5%뿐입니다. 실패 중 4~7판은 SUMO 오류이며 검거율에서는 실패로 셌습니다.
- 그래서 지금 설정으로는 "실도로에서도 잡힌다"는 시연은 되지만, "GA가 낫다"는 주장의 근거는 1단계(ideal) 결과뿐입니다.
- 혼잡 교통 조건에서는 추격 시작(300 s) 시점에 약 400대가 평균 17 km/h로 달리고, 이후 정체가 쌓입니다.

### 4.5 웹 리플레이 뷰어

- 주소: https://claude.ai/artifact/2rEYHbcG51hERqj4cCVgea (비공개)
- 대전 데모 SUMO(ideal) 5판과 서울·부산 자체 시뮬레이터 4판이 들어 있습니다.
- 원본 데이터는 `viewer/sumo_dj.json`, `viewer/ring.json`에 있고, `python viewer/build_viewer.py out.html viewer/sumo_dj.json:sumo viewer/ring.json:point`로 같은 페이지를 다시 만들 수 있습니다.
- 아직 real 프로필 리플레이는 없습니다.

## 5. Codex가 할 일 (우선순위 순)

### P1. SUMO 오류 제거 (작고 확실한 일)

오류는 `eval_sumo.py`가 `err=`로 집계합니다. 원인은 두 가지입니다.

1. **`stop ... has an invalid position`**: 경로를 처음 설정할 때(`_spawn`의 `setStop`) 정지 위치가 차로 길이를 넘거나, 차량이 이미 그 위치를 지난 경우입니다.
   - `_spawn`도 `_set_route`와 같은 재시도 방식을 쓰게 하세요. 정지 위치를 확장하고 `try/except traci.TraCIException`로 감쌉니다.
   - 차로가 여러 개인 엣지에서는 `laneIndex=0`(가장 오른쪽 차로) 정지가 다음 연결과 어긋날 수 있으니, 차로 선택도 확인하세요.
2. **`Vehicle 'police_k' is not known`**: 정지가 실패한 경로 끝에 차량이 도착해 시뮬레이션에서 빠진 경우입니다. 1번을 고치면 대부분 사라질 것입니다.
   - 그래도 남으면, 도착 직전에 경로를 연장하는 안전장치를 넣으세요(매 스텝 `getRouteIndex`가 마지막 엣지인지 검사).
- **완료 기준**: real + 교통 0.7 조건 150판에서 `err=0`.

### P2. 정책 차이가 드러나는 실도로 설정 찾기

실도로에서 모든 정책이 수렴하는 것은 시나리오가 너무 쉽기 때문입니다. 다음 조건을 순서대로 평가하세요. 각 조건 150~300판, 정책은 `rmacro:runs/ms_ga/best.pt encircle greedy rmacro:assign`입니다.

- 경찰 3대와 4대(`eval_sumo.py ... 3 real ...`, `... 4 real ...`).
- **비대칭 조건**: 도주차만 신호를 무시하고(`ignore_red`), 경찰은 신호를 지킵니다. 지금은 `ignore_red`가 경찰과 도주차에 함께 적용되므로, `SumoPursuit`에 `fugitive_ignore_red`와 `police_ignore_red`를 따로 두세요.
- 도주차 speed_factor를 경찰보다 높게 둡니다(난폭 운전).
- 포위 거리를 넓힙니다. `make_ring_case(ring_min_m=600, ring_max_m=1000)`.

결과는 4.4절 형식의 표로 정리하고, paired 차이와 95% CI를 붙이세요. 계산 방법은 `eval_policies.py`의 wins/losses 계산을 참고하면 됩니다.

### P3. 실도로 조건에서 정책 재탐색

- P2에서 GA가 기준 정책보다 못한 조건이 나오면 정책을 다시 찾습니다.
  - 방법 1: SUMO 안에서 GA를 돌립니다. `ms_ga.py` 구조를 그대로 쓰고 fitness만 `eval_sumo`로 바꿉니다. 판당 5~20 s라 느리므로 판 수를 줄이세요.
  - 방법 2: 자체 시뮬레이터에 속도 변동, 교차로 지연, 신호 대기 같은 무작위 요소를 넣어 탐색합니다.
- 경찰 수 무관 정책은 `nvar_ga`가 개선되지 않았습니다. 당장은 `ms_ga/best.pt`를 그대로 쓰면 됩니다. 이 정책은 `len(police)`로 팀 크기를 받으며, 경찰 번호 one-hot은 `k % 6`으로 처리됩니다. 4.3절의 표가 이 방식의 근거입니다.

### P4. 시연 준비 (대전)

- real 프로필 리플레이를 만듭니다. `viewer/export_sumo.py`에 `profile`, `traffic`, `ignore_red`를 인자로 넘기도록 확장하세요. 지금은 ideal로 고정돼 있습니다. 그다음 `build_viewer.py`로 페이지를 다시 만듭니다.
  - 일반 교통 차량도 보여주려면 매초 `conn.vehicle.getIDList()` 위치를 기록해 회색 점으로 그리세요. 데이터가 커지므로 시연용 1~2판만 하세요.
- sumo-gui 시연은 화면이 있는 PC에서 합니다.
  ```bash
  pip install eclipse-sumo==1.28.0 traci==1.28.0 sumolib==1.28.0
  python sumo/demo_gui.py --map daejeon_demo --n 6 --idx 0 --profile real --traffic 2
  ```
  - 필요한 파일: 레포 전체, `cache/research/maps/<hash>/`(지도 캐시), `sumo/nets/`(없으면 자동 생성).
  - 경찰은 파란색, 도주차는 빨간색입니다. 재생 속도는 `SUMO_GUI_DELAY_MS`로 조절합니다.
- GA가 실패하는 판도 있으니, 시연할 판은 미리 골라 두세요(`viewer/export_sumo.py`의 kind 인자: win, both, lose).

### P5. 논문 형식 결과 정리 (T-ITS 스타일)

- 지표: SR, ATS, 각각의 표준편차. 축: 경찰 수 × 지도 × 시뮬레이터(자체, SUMO ideal, SUMO real).
- 기준 정책: greedy, encircle, assign.
- ablation: 상대 특징 끔(`macro:` vs `rmacro:`), 대전 전용 GA vs 여러 도시 GA.
- case study: 뷰어 장면.
- **보고서에는 "GA(진화 탐색)로 찾은 역할 선택 정책"이라고 쓰세요. RL이라고 쓰면 안 됩니다.**

### P6. 정리와 커밋

- `beat_greedy/`, 새 지도 yaml 5개, `artifacts/research/maps/integration/`, 이 문서를 커밋합니다.
  - `runs/`의 큰 체크포인트는 `runs/ms_ga/best.pt`와 `runs/ga/best.pt`만 남기는 것을 권장합니다.
- `sumo/nets/`는 다시 만들 수 있으므로 `.gitignore`에 넣어도 됩니다.
- 커밋 메시지 끝에는 사용자 설정의 attribution을 따르세요.

## 6. 알려진 함정

- `common.py`의 `FUGITIVE_SPEED_MPS`는 환경변수 `BG_FUGITIVE_SPEED`에서 읽고, **기본값이 9.0**입니다. 실험 기준은 16입니다. `multi_eval`은 `ME_VF`(기본 16)로 이 값을 맞춰 줍니다. `eval_policies.py`를 쓸 때는 `BG_SOLVABLE=1 BG_FUGITIVE_SPEED=16`을 꼭 붙이세요.
- 봉인 환경(`replay_episode`)은 경찰 6대만 받습니다. N대 실험은 반드시 `multi_sim`에서 하세요.
- 포위 정책(`EncirclementPolice`)의 진단 레코드는 6대일 때만 검증을 통과합니다. N대일 때는 `multi_eval.AnySizeEncirclement`를 쓰세요(`team_for`가 자동 처리).
- `macro.CandidateScorer`를 저장할 때 `"kind": "macro"` 키가 이미 있습니다. 스코어러 종류는 반드시 `"scorer_kind"` 키로 저장하세요.
- multiprocessing 워커에서 torch를 쓸 때는 `torch.set_num_threads(1)`을 넣으세요. 빠뜨리면 평가가 20배 이상 느려집니다(`multi_eval._init`에 반영돼 있습니다).
- SUMO 엣지 길이는 교차로 형상 때문에 우리 그래프보다 평균 9~14 m 짧습니다. 줄어든 길이는 교차로 내부 차로가 채웁니다. 정상입니다.
- SUMO 검거 판정(0.25 s 간격)은 자체 시뮬레이터(1 s 간격)보다 촘촘합니다. 그래서 두 시뮬레이터의 검거율을 직접 비교하지 말고, **같은 시뮬레이터 안에서 정책끼리 비교**하세요.

## 7. 웹 대시보드 + SUMO 서버 (2026-10-12 마지막 작업, 진행 중)

목표: 사용자가 지도에서 경찰차·도주차를 클릭해 배치하면 SUMO(또는 빠른 자체 시뮬레이터)로 추격을 실행하고, 1초 단위로 스트리밍해 지도 위에서 실시간으로 보여줍니다. 프런트는 Vercel(사용자 GitHub 계정), SUMO 서버는 이 GPU 서버(공인 IP 168.188.128.36, 인바운드 가능 여부는 미확인)에 둡니다.

- **서버** `demo/server/app.py` (FastAPI)
  - 엔드포인트: `GET /api/maps`, `GET /api/map/{name}`, `POST /api/example`, `POST /api/run`(NDJSON 스트림: setup, frame×N, end/error)
  - 실행: `cd demo/server && /workspace/cluster/.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8600`
  - **검증됨**: curl로 point·sumo 두 엔진 모두 스트리밍 정상(예시 배치에서 각각 17초, 14초 검거)
  - 주의: 마지막 수정(frame에 `ox` 열린 출구 좌표 추가) 후 서버를 재시작했지만 재검증은 하지 못했습니다. real 프로필 + traffic 경로(일반 차량 `bg` 포함)도 아직 테스트하지 않았습니다.
- **프런트** `demo/web/` (Vite + TypeScript + Leaflet, CARTO 타일)
  - `npm run build` 통과(tsc 포함). 로컬 개발: `npm run dev`(vite proxy가 `/api`를 :8600으로 넘김)
  - 배포 시 `VITE_API_BASE`에 서버의 HTTPS 주소를 넣습니다. 화면의 '서버 주소' 칸에서 바꿀 수도 있습니다.
  - **브라우저 실제 동작은 아직 확인하지 못했습니다.** 확인하려고 headless 브라우저(playwright chromium)를 scratchpad venv에 설치하던 중에 세션이 끝났습니다.
- **남은 일**
  1. 서버를 재시작하고 headless 브라우저로 대시보드 E2E를 확인합니다(배치 → 추격 시작 → 애니메이션 → 기록). 깨진 UI를 수정합니다.
  2. HTTPS 노출: Vercel(HTTPS) 페이지는 HTTP 서버를 호출할 수 없습니다(mixed content). cloudflared 터널 등으로 HTTPS 주소를 만들어야 합니다. **외부 공개 작업이므로 사용자에게 먼저 확인받으세요.** 서버는 systemd나 tmux로 상시 실행합니다. `PURSUIT_CORS`는 Vercel 도메인으로 제한합니다.
  3. `demo/web`을 사용자 GitHub 저장소에 push하고 Vercel을 연결합니다(Root Directory = `demo/web`, env `VITE_API_BASE`). 계정 작업은 사용자가 합니다.
  4. 서버가 꺼져 있을 때를 대비해, 미리 계산한 SUMO 판(`viewer/sumo_dj.json`)을 정적 파일로 넣는 대체 모드를 만듭니다.

### Codex 후속 확인 (2026-10-12)

- 서버 직접 호출로 point, SUMO ideal, SUMO real+traffic=10 실행을 끝까지 확인했습니다. 세 경로 모두 frame에 `ox`가 있고, real+traffic frame에는 `bg`가 있습니다. UI의 보통 교통(`traffic=2`)도 1판 실행해 18 frame, 최대 일반 차량 150대, 17초 검거, 오류 0건을 확인했습니다. 단, `traffic=0.7`의 고부하 장시간 안정성은 아직 확인하지 않았습니다.
- `tests/test_demo_server_app.py`에 API/프레임 계약 테스트 3개를 추가했고 모두 통과했습니다.
- Headless Chromium에서 지도 클릭·드래그·예시 배치·point/SUMO 실행·기록·되감기를 확인했습니다. 첫 QA에서 발견한 녹화 모드 지도 클릭과 재생 인덱스 오류는 수정하고 재검증했으며 앱 런타임 예외는 0건입니다.
- 서버 장애 시 `demo/web/public/fallback.json`의 **사전 계산된 대전 SUMO ideal GA 1판**을 재생하는 버튼을 추가했습니다. 이 파일은 `demo/web/scripts/build_fallback.py`로 원본 `viewer/sumo_dj.json`에서 다시 만들 수 있습니다. 녹화 모드는 사용자의 새 배치나 설정을 계산하지 않는다고 화면에 명시합니다.
- 웹 `npm run build`와 서버 테스트 3개가 통과했습니다. 외부 HTTPS 공개, GitHub push, Vercel 연결은 진행하지 않았습니다.

### Claude 후속 수정 (2026-10-12)
- 지도 배경: CARTO 타일이 "API KEY REQUIRED"를 반환해서 OpenStreetMap 표준 타일로 바꿨습니다. 다크 모드에서는 타일 영역에 CSS 필터를 적용합니다. 시뮬레이터 도로망도 위에 겹쳐 그려서, 타일이 막혀도 지도가 보이게 했습니다. 시연 수준의 트래픽은 OSM 타일 정책상 괜찮지만, 정식 서비스로 쓰려면 키가 있는 타일 제공자로 바꾸세요.
- 결과 미리 노출 수정: 스트림이 끝나도 재생이 끝나기 전에는 "계산 완료 · 재생 중"을 표시하고, 재생이 끝나면 결과를 보여줍니다.
- 실행 시작과 기록 선택 시 지도를 차량들이 있는 범위로 확대합니다.
- headless 확인: 예시 배치 → SUMO 실행 → 재생에서 런타임 예외 0건. `npm run build`와 `tests/test_demo_server_app.py`(7개) 통과.

### 공개 배포 상태 (2026-10-12 18:41, Claude)
- **완성본 링크**: https://express-krypton-ciz5965.vercel.app (Vercel 프로젝트 `express-krypton-ciz5965`, jaehyunkims-projects). 같은 빌드가 `temporary-express-krypton-ciz5965.vercel.app`에도 걸려 있습니다.
- SUMO 서버와 공개 터널은 `demo/run_public.sh`(setsid 상시 실행)가 관리합니다.
  - uvicorn과 cloudflared quick tunnel을 띄우고, 둘 중 하나가 꺼지면 다시 살립니다.
  - 터널 주소가 바뀌면 `vercel@59.26.0 deploy --prod --build-env VITE_API_BASE=<새 주소>`로 자동 재배포합니다. 재시작 테스트에서 14초 만에 재배포가 끝났습니다.
  - 현재 주소: `demo/logs/tunnel_url`. 로그: `demo/logs/{supervisor,server,tunnel,deploy}.log`.
  - 다시 띄우는 방법: `cd /workspace/cluster && REDEPLOY_ON_URL_CHANGE=1 setsid nohup demo/run_public.sh > demo/logs/supervisor.log 2>&1 < /dev/null &`
  - 종료할 때는 PID로 kill 하세요. `pkill -f`는 자기 셸까지 죽입니다.
- 사용자 요청에 따라 토큰 없이 링크만으로 동작합니다. 보호 장치는 동시 실행 6개 제한(`PURSUIT_MAX_RUNS`)과 CORS(Vercel 도메인만 허용)입니다. 토큰을 다시 켜려면 `REQUIRE_TOKEN=1`로 띄우세요.
- 실제 배포 사이트를 headless 브라우저로 확인했습니다: 예시 배치 → SUMO 실행 → 검거, 런타임 오류 0건. 실제 도로 + 일반 교통(traffic=2)도 터널 경유 API로 18 frame, 일반 차량 150대, 검거까지 확인했습니다.
- 한계:
  - 이 GPU 서버나 스크립트가 꺼지면 SUMO 기능이 멈춥니다. 녹화 시연 버튼은 계속 동작합니다.
  - quick tunnel은 고정 주소가 아닙니다. 고정 주소가 필요하면 Cloudflare 계정에 named tunnel을 만드세요.
- GitHub: 워크트리 `/tmp/pursuit-dashboard-deploy`, 브랜치 `codex/dashboard-deploy-20261012`에 커밋 2개(cc28f01까지)가 있지만 **push는 하지 못했습니다**. 이 서버에 GitHub 인증 정보가 없습니다. Codex 환경이나 사용자가 `git -C /tmp/pursuit-dashboard-deploy push origin codex/dashboard-deploy-20261012`로 push하세요.

### 7.x Barricades + GitHub-linked Vercel (Claude, 2026-10-13)
- Vercel `express-krypton-ciz5965` is linked to GitHub `jjaehyun2/Police_Runner`: Root Directory `demo/web`, **production branch = `codex/dashboard-deploy-20261012`** (main is untouched), env `VITE_API_BASE` = current tunnel URL. A push to that branch rebuilds production.
- `demo/run_public.sh`: on tunnel URL change → `vercel env add VITE_API_BASE ... --force` (production/preview) → POST deploy hook (`demo/logs/deploy_hook`, not in git). Run with `REDEPLOY_ON_URL_CHANGE=1`.
- Barricades (`sumo_sim.Barricade`, `barricade_at`; API `barricades: [[lat,lon]]`, SUMO only; frames carry `bk` = indices the fugitive has seen):
  - closes both directions of the nearest road; in SUMO each lane gets a stopped `barricade` vehicle (orange), so background traffic queues behind it;
  - officers route on `sim.pg` (barricaded roads removed) from the start; escape region in the dashboard also uses `pg`;
  - fugitive uses its own graph `sim.fg`, closes a road only when within `BARRICADE_SIGHT_M`=80 m, then re-plans; if the barricade is ahead on its current road it U-turns (`moveToXY` onto the reverse edge); one-way/failed U-turn → `sim.trapped`, the 2 nearest officers switch to pursue and approach along that road;
  - **the team policies (GA / encircle / ...) do not know about barricades yet** — officers may hold an exit that a barricade already closes. Next step: give the policy the barricaded graph (graph caches are keyed per network id in `multi_sim.graph_of` and `obs_ext.graph_for`, so this needs a per-run graph instead of the shared cache) and, later, let the policy choose barricade placement.

### 7.y Carried barricades prototype (Claude, 2026-10-13)
`SumoPursuit(carry_barricades=True)` / API `carry_barricades` / dashboard checkbox. One barricade per patrol car, placed by rules in `sumo_sim._deploy` (rules via `SUMO_BARRICADE_RULES`, default `leave,stand-in`). SUMO ideal, daejeon_demo, ring start, GA policy, 100 episodes, same seeds (eval script pattern: scratchpad `carry_eval.py`):

| rules | N=4 SR | paired W/L vs none | barricades/ep |
|---|---|---|---|
| none | 87.0 | – | 0 |
| leave | 89.0 | 2/0 | 0.18 |
| leave,stand-in (default) | 89.0 | 2/0 | 0.59 |
| leave,planned | 89.0 | 3/1 | 0.22 |
| planned | 85.0 | 1/3 | 0.02 |
| leave,pass (cut .05/.15) | 82.0 | 3/8 | 2.0 / 1.7 |

N=3: none 80.0, leave 80.0 (1/1), leave,stand-in 79.0 (1/2). N=6: 100 either way. Every failure without barricades is a late escape (126–272 s).
Takeaways: in the Dunsan grid one closed road rarely shuts anything; placing often hurts because officers must route around barricades too; planned trips pull officers out of the cordon. Effect is within noise — a learned/searched placement (GA over placement features, scored on the same paired seeds) is the next step if this is pursued.
Ops: the Vercel CLI token in ~/.local/share/com.vercel.cli/auth.json expired on 2026-10-13; `run_public.sh`'s env update needs `vercel login` again (git pushes and the deploy hook still deploy).

### 7.z Joint role + barricade GA (bar_ga.py), v1 → v2 (2026-10-13)
v1 (runs/bar_ga, 30 gens, report 08:39): learned vs never (success %, ring starts, 1 fugitive)
- validation: daejeon 97.0→100 (all from role; head off = same), incheon 92.3→91.3, suwon2 85.3→86.3
- held-out once: daejeon_test 89.3→90.3, seoul 70.8→59.8 (role-only 67.5, 2.74 barricades/ep), busan 65.0→57.8 (role-only 59.0)
- SUMO daejeon_demo: N=3 88.3→68.3 (role-only 73.3), N=4 88→86, N=6 100→96.7
Diagnosis: role mutations overfit the 3 train maps; head placed no-effect barricades on unseen maps; per-gen max-of-noisy validation selection (winner's curse).
v2 (runs/bar_ga_v2): barricade.IMPACT_ONLY (only roads that shrink the region / shut an exit), role steps 0.01/0.03,
final selection re-scores top-6 + never on fresh validation x2 and SUMO daejeon_demo (N=3,4), never may win.
Held-out maps were looked at once for v1 diagnosis (role-only ablation); v2 changes were not tuned on them.
v2 result (runs/bar_ga_v2, 20 gens, report 11:51): the chosen policy (g7_5) places **0 barricades** on every map — the
head learned not to place once only impact roads were allowed. Remaining differences are the role scorer only:
validation daejeon 97→100, incheon 92.3→92.7, suwon2 85.3→87.0; held-out daejeon_test 89.3→88.3, seoul 70.8→71.0,
**busan 65.0→46.5** (6 W / 80 L); SUMO N=3 88.3→83.3, N=4 88→89, N=6 100→100. Hand rules (leave,stand-in) in SUMO:
N=3 85.0, N=4 88.0, N=6 96.7 — no gain either. Conclusion: in ring starts with mid-chase placement, one barricade per
car does not raise capture; GA role tuning on 3 train maps does not generalize (busan). Dashboard stays on
runs/nvar_ga/best.pt; the carried-barricade option keeps the hand rules (prototype).

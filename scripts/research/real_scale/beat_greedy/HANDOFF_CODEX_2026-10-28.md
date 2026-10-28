# Codex 인계 문서 (2026-10-28 12:30 KST, Claude Code 세션)

> **팀 공유 기준:** 수정 전 SUMO 표는 정책 비교에 사용하지 않는다. 최신 확정 결과와 원인 분리는
> `REPORT_2026-10-28.md` §13을 기준으로 한다. 교통 없는 통영 큰 지도에서는 SUMO와 단순
> 시뮬레이터가 거의 일치했고, 16/19 m/s에서 우리 90.0%, encircle 71.5%였다.

이 문서 하나로 이어받을 수 있게 썼다. 상세 수치와 근거는 같은 폴더의 **`REPORT_2026-10-28.md`** (이하 REPORT)에 있다. 먼저 이 문서의 §0~§3을 읽고, 숫자가 필요하면 REPORT를 본다.
옛 인계 문서 `/workspace/cluster/HANDOFF_CODEX.md`는 10-12 버전이라 일부 사실이 낡았다 (예: 최고 정책, 지도 목록). 충돌하면 이 문서를 따른다.

작업 폴더: `/workspace/cluster/scripts/research/real_scale/beat_greedy` (이하 BG). 모든 경로는 BG 기준.
파이썬: `/workspace/cluster/.venv/bin/python` (매 명령마다 전체 경로로 쓴다. `source activate`는 셸마다 풀린다).

---

## 0. 사용자 규칙 (반드시 지킬 것)

1. **응답은 한국어.** 결과는 정직하게: 실패, 한계, 유의하지 않은 차이도 숫자로 함께 쓴다. 정황 추정은 '가설'이라고 표시한다.
2. **보류 지도(held-out)는 학습·선택·조정에 절대 쓰지 않는다:** `daejeon_test`, `seoul_zero_shot`, `busan_zero_shot`. 최종 확인에 한 번만 쓴다.
   - 예외: PEPCRL(비교 대상)은 보류 지도에서 학습해도 된다고 사용자가 승인했다.
   - 개발 지도(`*_dev`, `*_large_dev`)는 서울·부산 등에 있지만 보류 지도와 다른 영역이며, 개발·조정에 써도 된다.
3. **시나리오:** 도주차 정확히 1대, 경찰 수만 변동 (3~10). 정책은 경찰 수와 무관해야 한다.
4. **보안:** GitHub 토큰은 `~/.git-credentials`에 있다. 메모리·커밋·문서에 절대 쓰지 않는다. 자격 증명을 들여다보거나 토큰을 명령에 넣어 push하지 않는다. 커밋·push는 사용자가 요청할 때만.
5. **사용자 지시 원문 (최근):** "밤새 중단없이 버그없이 잘 진행해", "우선적으로 encircle을 이기는게 중요하니까", "결과 나오면 바로 공유해".
6. 논문 표현 주의: 최고 정책은 GA로 찾은 역할 선택 MLP다 (`runs/nvar_ga/best.pt`). IPPO 버전도 있다 (CLI 이름은 mappo지만 논문에서는 IPPO). PEPCRL은 '재구현'이라고 쓰고 차이점과 공식 코드 버그를 명시한다.

---

## 1. 연구 목표와 현재 위치

- **논문:** IEEE IV 2027 (마감 **2026-11-15**, 6쪽). 대안 ITSC 2027 → T-ITS.
- **문제:** 실제 OSM 도로망에서 도주차 1대가 지도 경계(출구)로 탈출하려 하고, N대의 경찰이 출구를 막거나 붙잡는다 (25 m 이내 = 포획). 단순 시뮬레이터(point-mass, `multi_sim.py`)에서 학습하고 SUMO(`sumo/`)에서 검증한다.
- **우리 방법:** 6초마다 경찰마다 후보(0 = 직접 추격, 1~7 = 도주차 출구 경로 위 차단 지점) 중 하나를 고르는 역할 정책. 후보별 16차원 특징 → 공유 MLP 점수. 상대 단위(도주차~가장 가까운 출구 거리 S)로 정규화해 도시 크기와 무관하다. 코드: `macro.py` (`MacroTeamPolicy`, `CandidateScorer`).
- **기준선:** `encircle` (가까운 2대 추격 + 나머지는 진행 방향 0°/±55°/±110°, 120~380 m 포위 지점; `pursuit_evasion_rl/.../baselines.py:373`), `greedy`, PEPCRL 재구현(`pep:` 스펙, `pepcrl.py`).

### 지금까지의 핵심 결론 (근거는 REPORT)
1. **단순 시뮬레이터, 보류 지도:** 우리 정책이 encircle보다 +5.8 ~ +7.2점 (유의). 처음 보는 도시(서울·부산) 일반화가 가장 강한 근거. PEPCRL 재구현은 23~57%로 크게 낮다.
2. **SUMO, 3 km 지도:** encircle을 **아직 이긴 적이 없다.** 대전은 모두 96~99%로 천장, 감천동(과포화 교통)에서는 −12.5점으로 졌다.
3. **진 원인 (규명됨):**
   - 과포화 배경 교통(1초 간격 투입: 멈춘 차 55~61%) → 정체에 갇혀 시간초과. 우리 방식은 포획이 느려 더 오래 노출된다.
   - 경찰차가 만든 줄은 원인이 아니다 (갇힌 시간의 3%).
   - 교통이 없으면 감천동 SUMO에서 88.9 vs 88.9 동률.
   - 단순 시뮬레이터의 우위가 SUMO에서 사라지는 이유: encircle이 SUMO에서 더 많이 '얻는다' (단순 시뮬레이터에서 놓친 판을 31판 잡음 vs 우리 17판).
   - **유력 가설 (검증 중):** SUMO의 속도 상한(제한속도 × 2배)이 30 km/h 골목에서 모든 차를 16.7 m/s로 묶어, 빠른 도주차(19 m/s)의 속도 우위를 지운다 → 단순 추격만으로 잡힌다 → 계획 기반인 우리 방식의 이점이 사라진다.
4. **큰 지도(약 6 km)에서는 관계가 뒤집힌다 (단순 시뮬레이터):** 통영 큰 지도 16개 조건 모두 우리 > encircle (+2 ~ +16점, 빠른 도주차 + 포위에서 최대). 감천동 큰 지도는 대체로 동률.
5. **v2 교통 반영 이동시간 비용:** 과포화 SUMO 6개 조건 합산 v2 − 도로 거리 = +4.3 ± 2.1점 (유의, 공통 분모 n = 1,191). v1(제한속도+회전)은 효과 없음.
6. **기각된 시도:** '항상 1명 추격'(−16~−18점), 이동시간 무작위화 학습 가설(재현 안 됨), '목표 교차로 도착 시 도로 진입' 조향(효과 없음, 되돌림).

---

## 2. 지금 돌고 있는 작업 (12:27 기준)

모두 `nohup`으로 띄웠고 부모가 PID 1이라 세션이 끊겨도 계속 돈다. **결과가 나오면 분석해서 사용자에게 공유하고 REPORT를 갱신하는 것이 주 업무다.**

| 작업 | 스크립트 (PID) | 출력 | 끝 표시 | 내용 |
|---|---|---|---|---|
| **큰 지도 SUMO** | `runs/large_map/sumo.sh` (3064409) | `runs/large_map/sumo.log` | `runs/large_map/SUMO_DONE` | 통영 큰 지도(3초 간격 교통) → 감천동 큰 지도(2초). 조건: 무작위 N3 / 포위 1.5~2.5 km N3 / 도주차 19 m/s 무작위 N6. 정책: rmacro, cmacro, encircle. 양보 5.6, 신호 무시, 1200초, 판 9000번~. 통영 첫 조건: rmacro 98.0%, cmacro 96.5%, encircle 진행 중 |
| **큰 지도 후속** | `runs/large_map/followup.sh` (3107209) | `sumo_fast.log`, `sumo_yeosu.log`, `yield_sens.log`, `followup.out` | `FOLLOWUP_DONE` | SUMO_DONE 뒤 시작. ① 도로 분류 수정(`RT_CLASS_FIX=1`)으로 통영·감천동 큰 도로망 재생성 ② **빠른 추격** (포위, 16/19와 22/26 m/s, 속도 배율 3) 통영·감천동·여수 ③ 여수 보통 조건 ④ 양보 속도 민감도 (없음 / 3 / 5.6 / 8 m/s) |
| **속도 상한 검증** | `runs/speedcap/run.sh` (3117752) | `runs/speedcap/sf2.log`, `sf3.log`, `sf*.file` (결과 json 이름) | `runs/speedcap/DONE` | 성북동 3 km, 포위 900~1400, 도주차 19, N3, 교통 없음, 판 8000번~. **속도 배율 2 vs 3**에서 rmacro vs encircle |
| 교통 밀도 3단계 (3 km) | `runs/traffic_levels/run.sh` (2977481) | `runs/traffic_levels/sweep.log`, `stall.log` | `runs/traffic_levels/DONE` | 감천동·성북동·통영 3 km, 5/8/2초 간격, 무작위 N3, 양보 5.6, rmacro·cmacro·encircle·greedy. 5초에서 교착 해소 규칙 변형 추가 |
| 어려운 조건 (3 km) | `runs/traffic_levels/hard2.sh` (3117751) | `runs/traffic_levels/hard.log` | `HARD_DONE` | 밀도 실험 뒤. 포위 900~1400, 도주차 19, 5초 간격, 양보, **속도 배율 3** |
| 큰 지도 단순 시뮬레이터 선별 | `runs/large_map/queue_screen.sh` (3042570) → 여수 대기 (3107210) | `runs/large_map/pm_gamcheon.log`, `pm_yeosu.log` | `PM_DONE` | 속도 4단계 × 배치 2 × N 3·6, 200판 |
| 직선거리 IPPO 제거 실험 | `runs/metric_abl/run2.sh` (2577639) | `euclid_s*.log`, `table.log` | `runs/metric_abl/DONE` | seed 3개 185회 중 180/170/155회. 끝나면 보류 지도 표 자동 생성 (PEPCRL+도로거리 `pep_road_s0` 포함) |
| 공식 PEPCRL 재현 | `/workspace/cluster/external/pepcrl_official_run/main.py` (2968895) + 감시 `guard.sh` (3054119) | `train.out`, `log/`, `guard.log` | – | 25 GB 넘으면 감시가 죽이고 저장 가중치에서 재시작. 매우 느림 (1단계 2시간+), 수렴 재현은 불가능. 초기 학습 곡선용 |

자원: CPU 48코어 (부하 60~70), 컨테이너 메모리 한도 100 GB (`/sys/fs/cgroup/memory/memory.usage_in_bytes`로 확인, 평소 ~55 GB). `free -g`는 호스트 전체라 무시. 새 실험은 동시 워커 10~16개 이내로.

---

## 3. 결과가 나오면 하는 일 (분석 절차)

### 3.1 SUMO 결과 읽기
- 로그 줄: `... SUMO SR=0.980 ATS=176.1s esc=0.020 err=0 | point-mass SR=...`. 원 성공률은 오류를 실패로 센다.
- json: `sumo/results/sumo_<map>_<profile>_tr<교통>_red<신호>_P<N>_<배치><범위>_off<시작판>_<시각>.json`. 정책마다 `[판번호, SUMO결과, 시간, 단순시뮬레이터결과, 걸음]`.
- **같은 판 비교 (기본):** `python sumo/paired.py <json> <A> <B> [...]` → A − B 차이 ± 95% CI, W/L.
- **여러 정책을 공통 분모로:** `python sumo/common_table.py <json> <기준> <정책...>` → 모두 오류 없는 판만으로 성공률과 기준 대비 차이. **보고서·논문 표는 이것으로 만든다** (§4.3과 §5.3 불일치의 원인이 분모 차이였다).
- 유의 판단: |차이| > CI 반폭이면 유의. 여러 조건 중 유의한 것만 골라 강조하지 말고, 합산 결과를 함께 쓴다.

### 3.2 판단 기준
| 결과 | 판단 | 다음 행동 |
|---|---|---|
| 속도 상한 검증: 배율 3에서 rmacro − encircle가 배율 2보다 유의하게 커짐 | 가설 확정 | SUMO 표준 설정을 배율 3(또는 차량 최고속도까지 허용)으로 정하고, 논문에 근거와 함께 명시. 3 km 지도 비교 재실행 |
| 차이 없음 | 가설 기각 | REPORT에 기각 기록. 다른 원인 탐색 (예: 교차로 회전 감속이 도주차에 미치는 영향) |
| 큰 지도 빠른 추격에서 rmacro(또는 cmacro) > encircle 유의 | **논문 SUMO 주장 성립** | 큰 지도 중심으로 SUMO 표 재구성. 보류 지도도 같은 크기(0.06°)로 받아 마지막에 한 번만 확인 |
| 큰 지도에서도 동률/열세 | 위험 | 사용자에게 바로 보고. 대안: (a) 포위 후보를 학습 후보에 추가해 재학습 (REPORT §8 설계), (b) 논문 주장을 '일반화 + 단순 시뮬레이터 우위 + SUMO 동등'으로 낮춤 |
| 양보 속도 민감도: 3~8 m/s에서 결론(우리 vs encircle 순위)이 같음 | 양보 모델 견고 | 논문에 민감도로 보고 |
| 교통 밀도 실험: 밀도가 높을수록 v2 − 도로 차이가 커짐 | v2 기여 성립 | '교통 밀도에 따른 v2 이득' 그림 |

### 3.3 공유 형식 (사용자 선호)
- 한국어, 결론 먼저, 표로 숫자. 유의 여부와 판 수를 함께. 모르는 것은 모른다고.
- 끝나면 REPORT의 해당 절(§11 진행 표, §12 큰 지도 등)을 갱신한다.

---

## 4. 남은 작업 (우선순위)

1. **[최우선] 위 실행 중 실험 결과 분석·공유·REPORT 갱신.** 특히 속도 상한 검증 → 큰 지도 SUMO의 encircle 결과 → 빠른 추격 결과.
2. **SUMO 표준 설정 확정** (속도 배율, 교통 밀도 '보통' = 도로 1 km당 차량 수 기준, 양보 5.6 m/s, `RT_CLASS_FIX=1`). 확정 뒤 논문용 SUMO 표를 이 설정으로 일괄 재실행.
3. **논문 최종 표는 모두 `RT_CLASS_FIX=1`로 다시 돌려야 한다.** 지금까지 실행은 옛 도로 분류 (생활도로 30 km/h). 이 스위치를 켤 때는 해당 지도의 SUMO 도로망도 같은 스위치로 다시 만든다 (`RT_CLASS_FIX=1 python sumo/build_net.py <map> real`). **실행 중인 실험이 쓰는 도로망을 중간에 다시 만들지 말 것.**
4. **보류 지도 큰 버전:** 큰 지도에서 이기면, 보류 지도와 같은 도시의 다른 영역이 아니라 **보류 지도 영역 자체를 0.06°로 확장**해 받을지 사용자와 상의 (보류 원칙 유지 방법).
5. **v2 단순화:** 항 제거 비교 결과 지수의 차로 수는 빼도 된다 (REPORT §4.2.1). 논문에는 p = (1−ρ)^g 단순형 권장. 바꾸면 보정(`traffic_calib.py`)과 평가를 다시 해야 하므로 사용자 확인 후.
6. 직선거리 제거 실험 표(`runs/metric_abl/table.log`)가 나오면 REPORT §4.4에 반영.
7. 공식 PEPCRL: 학습 곡선만 주기적으로 확인 (`grep test_episode train.out`, `log/RealMap_6_3/PEPCRL-MVP/test_log.csv`). 패치 5건은 REPORT §3.3.
8. (선택) 포위 지점 후보 추가 재학습 — 큰 지도에서도 encircle을 못 이기면 검토.

---

## 5. 코드 스위치 (모두 기본값 꺼짐 = 지금까지의 결과와 동일)

| 스위치 | 파일 | 의미 |
|---|---|---|
| `SE_YIELD_V` (m/s) | `sumo/sumo_sim.py` `_yield_pass` | 막힌 경찰차 앞 25 m 안의 일반차가 비켜주고 경찰은 이 속도로 통과 |
| `SE_SPEED_FACTOR` | `sumo/eval_sumo.py` | 추적 차량의 제한속도 대비 배율 (기본 2.0). 빠른 조건은 3 |
| `SE_TRAFFIC` (초) | `sumo/eval_sumo.py` | 배경차 투입 간격. 작을수록 많음. 3 km 지도 '보통' = 5, 큰 지도 통영·여수 3, 감천동 2 |
| `SE_IGNORE_RED` | `sumo/eval_sumo.py` | 1 = 둘 다 신호 무시, p = 경찰만, f = 도주차만 |
| `SE_OFFSET` | `sumo/eval_sumo.py` | 판 번호 시작. 새 실험은 겹치지 않게 (8000, 9000 사용 중) |
| `ME_PLACEMENT`, `ME_RING` | `multi_eval.py` | random / ring, 포위 거리 범위 (m) |
| `ME_VP`, `ME_VF` | `multi_eval.py`, `sumo/eval_sumo.py` | 경찰·도주차 최고속도. 정책이 아는 속도도 같이 바뀐다 |
| `ME_MAX_STEPS` | 두 파일 | 최대 시간(초). 큰 지도 1200 |
| `RT_CLASS_FIX=1` | `roadtime.py`, `sumo/build_net.py` | living_street 20 km/h. 정책과 도로망에 함께 |
| `MACRO_STALL_S` | `macro.py` | 도주차가 이 시간 이상 멈춰 있고 추격자가 없으면 가장 가까운 1대 추격 (단순 시뮬레이터에서 무해) |
| `MACRO_MIN_PURSUE=1` | `macro.py` | 항상 1명 추격 — **쓰지 말 것** (−16~−18점) |
| `MS_DR="lo,hi"` | `multi_sim.py` | 구간 이동속도 무작위화 (가설 기각, 진단용) |
| `OSM_TILES=k` | `fetch_osm_api.py` | 큰 영역을 k×k 조각으로 수집 |
| 정책 스펙 | `common.py` `_build` | `rmacro:` 도로 거리, `emacro:` 직선, `tmacro:` v1, `cmacro:` v2 (SUMO에서 실시간 교통 사용) |

예시 (큰 지도 SUMO 한 조건):
```bash
cd /workspace/cluster/scripts/research/real_scale/beat_greedy
env SE_WORKERS=12 SE_IGNORE_RED=1 SE_YIELD_V=5.6 SE_OFFSET=9000 ME_MAX_STEPS=1200 SE_TRAFFIC=3 \
    ME_PLACEMENT=ring ME_RING=1500,2500 ME_VP=16 ME_VF=19 SE_SPEED_FACTOR=3 RT_CLASS_FIX=1 \
    /workspace/cluster/.venv/bin/python sumo/eval_sumo.py tongyeong_large_dev 200 3 real \
    rmacro:runs/nvar_ga/best.pt cmacro:runs/nvar_ga/best.pt encircle > out.log 2>&1
```
단순 시뮬레이터 비교: `runs/large_map/pm_screen.py <map>` 참고 (`multi_eval.evaluate(map, eps, N, 1, spec, workers)`).

---

## 6. 함정 (실제로 겪은 것)

1. **`pkill -f` / `pgrep -f` 패턴이 자기 셸을 잡는다.** 명령문에 같은 문자열이 있으면 셸이 스스로 죽는다 (exit 144, 오늘도 한 번). **PID를 직접 확인해서 `kill <PID>`.** 확인은 `ps -eo pid,args | grep '[p]attern'`.
2. **`eval_sumo.py`는 `common`을 먼저 불러온다.** 정책이 아는 속도는 import 전에 정해진다. 과거 도주차 속도를 9 m/s로 오인한 버그가 있었다 (수정됨). 속도 관련 환경변수는 반드시 실행 시 env로 준다.
3. **실행 중인 스크립트가 불러오는 파이썬 파일을 고칠 때:** 새로 뜨는 프로세스가 바로 새 코드를 읽는다. import 오류 하나로 대기 중인 실험이 전부 깨질 수 있다. 고친 직후 `python -c "import ast; ast.parse(open(F).read())"`와 실제 import를 확인. 동작이 바뀌는 수정은 스위치(기본 꺼짐)로.
4. **실행 중인 bash 스크립트를 편집하지 말 것.** bash는 스크립트를 조금씩 읽는다. 바꿀 땐 PID로 종료 → 새 파일로.
5. **메모리:** 공식 PEPCRL이 메모리를 계속 늘려 오늘 아침 OOM으로 죽었다 (감시 스크립트로 대응). 대형 실험을 더 띄우기 전 `memory.usage_in_bytes` 확인.
6. **OSM 수집:** Overpass는 막혔다. `fetch_osm_api.py` (OSM API)를 쓰고, 큰 영역은 `OSM_TILES`. 서울 0.06° 영역은 공용 전처리(`pursuit_evasion_rl/osm_demo/coarsening.py`)에서 'cyclic virtual routing chain' 오류 — **보류 지도도 쓰는 공용 코드라 고치지 말 것.**
7. **고리 도로** (출발점 = 도착점): 큰 지도에만 있다. SUMO 변환기가 빼므로 `build_net.py`와 `sumo_sim.py`에서 제외 처리했다.
8. **SUMO 경고** `too close to brake`는 코드가 처리하는 정상 상황 (err 아님). `Edge ... is not known` (고리 도로 조회)은 수정됨.
9. GPU(P100)는 설치된 torch가 지원하지 않는다. 전부 CPU. 병목은 시뮬레이션이라 문제없다.
10. 서버 시간은 KST. 오늘은 2026-10-28.

---

## 7. 파일 지도

| 파일 | 내용 |
|---|---|
| `REPORT_2026-10-28.md` | 전체 결과·분석·근거 (12개 절). **결과가 나오면 여기를 갱신** |
| `/workspace/cluster/Police_Runner_v2_교통반영_이동시간_상세해설_2026-10-28.md` | 사용자가 가져온 v2 외부 검토 문서. 지적 대부분 반영함 (REPORT §4) |
| `/workspace/cluster/3804601.3804818.pdf` | GSTAM 원문 (격자 벤치마크, 직접 비교 불가; REPORT §6.3) |
| `macro.py` | 역할 정책 |
| `multi_sim.py`, `multi_eval.py` | 단순 시뮬레이터, 사례 생성·평가 |
| `sumo/sumo_sim.py`, `sumo/eval_sumo.py`, `sumo/build_net.py` | SUMO 시뮬레이션, 평가, 도로망·교통 생성 |
| `sumo/paired.py`, `sumo/common_table.py` | 통계 비교 |
| `roadtime.py`, `traffic_calib.py`, `runs/traffic_metric/calib_ablation.py` | v1/v2 이동시간, 보정, 항 제거 비교 |
| `nvar_ga.py`, `train_macro.py` | GA / IPPO 학습 |
| `runs/diag_gamcheon/` | encircle 패배 원인 진단 도구 (궤적, 대기열, 포화도 등) |
| `runs/large_map/` | 큰 지도 실험 |
| `fetch_osm_api.py` | 지도 수집 |
| `/root/.claude/projects/-workspace-cluster/memory/paper-ieee-iv-2027.md` | Claude 메모리 (참고용) |

---

## 7.5 [최우선, 15:10 수정 완료] 기존 SUMO 정책 비교는 무효

두 개의 독립된 버그가 확인됐다.

1. **도주차 종점 정차 편향**: 매 교차로에서 `[현재, 다음]` 경로 끝에 stop을 걸어 미리 감속했다. 같은 20개 seed에서 정차를 제거하자 도주차의 구간 평균속도가 **14.30 → 16.79 m/s**로 올랐다. 기본 동작을 stop 없는 rolling route로 바꾸고, 과거 동작은 `SE_LEGACY_FUGITIVE_STOPS=1` 진단에만 남겼다.
2. **더 큰 원인 — greedy 정책 연결 오류**: `TargetPolicy.goals()`가 greedy의 한 교차로 행동을 버리고 `pursue`로 바꿨다. 즉 기존 SUMO의 `greedy`는 greedy가 아니라 **도주차까지의 전역 유향 최단경로 추격**이었다. 추천 구간을 그대로 실행하도록 수정했다.

같은 성북동 20개 seed(교통 없음, ring 900~1400 m, 경찰 3대, 도주차 19 m/s, 배율 2) 재검증:

| 정책 | 수정 SUMO | 단순 시뮬레이터 | 판별 일치 | 오류 |
|---|---:|---:|---:|---:|
| 우리 (`rmacro`) | 95% | 80% | 75% | 0 |
| encircle | 95% | 85% | 80% | 0 |
| greedy | **30%** | **30%** | 80% | 0 |

따라서 “SUMO에서 greedy 전략이 갑자기 적절해졌다”는 해석은 틀렸다. 정책 연결 버그 때문에 다른 정책이 실행된 것이다. 기존 98.5% 대 43.5% 괴리는 수정 후 사라졌다. 수정 전의 큰 지도·교통 밀도·속도 상한 등 **모든 SUMO 정책 비교 수치는 폐기하고 다시 실행해야 한다.**

추가 수정:
- 짧은 경계 구간을 한 step 안에 통과해 도주차가 사라질 때 오류로 세던 것을 escape로 판정한다.
- `SE_SPEED_FACTOR=3`은 SUMO 한 step이 장시간 멎는 별도 문제가 있었다. 배율 3 이상은 차량형 배율을 3으로 올리지 않고, 안전거리·가감속 검사를 유지한 채 `vp/vf`를 직접 명령한다. 1판 및 5판 점검과 20판 비교 모두 완료, `err=0`.
- 배율 3, 같은 20 seed: 우리 95%, encircle 95%, greedy 50% (단순 각각 80%, 85%, 30%). 이 작은 확인에서는 배율 2→3이 우리−encircle 차이를 개선하지 않았다.

## 8. 사용자에게 바로 보고할 첫 항목 (예상)
0. **수정 전 SUMO 결과와 실행 스크립트를 그대로 재사용하지 말 것.** 위 §7.5 수정 코드로 새 결과를 생성한다.
1. 속도 상한 20판 점검에서는 배율 2와 3 모두 우리 95%, encircle 95%였다. 논문 표본 수로 재검증하되 속도 상한만으로 역전될 근거는 현재 없다.
2. 통영 큰 지도 무작위 N3의 encircle 결과와 같은 판 비교 (rmacro 98.0%, cmacro 96.5%는 이미 나옴; 천장 가능성).
3. 교통 밀도 실험 첫 조건 (감천동 3 km, 5초 간격).

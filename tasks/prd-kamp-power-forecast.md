# PRD: KAMP 제조 전력예측·최대피크 위험관리 시스템 (GMST-Power Contest Core v1)

- 작성일: 2026-09-22 (v2 수정 1·2차 + KMA 제외 + 컨트롤러 판정 동기화 + 재검토 R1–R11) · 대회 마감: 2026-10-08 23:59
- 스펙(방법·수식·임계값의 권위): `document/KAMP_GMST_Power_project_proposal_v2.md` (이하 v2, 수정 1·2차와 KMA 제외 반영본)
- 구속 결정: `.sdd/decisions.md` (이하 결정), `.sdd/progress.md`의 "Decision (user)"·"Ruling" 줄 (이하 판정), 한전 요금: `.sdd/kepco_tariff_2021.md` (이하 요금표), 대회 공고·보고서 양식: `.sdd/competition_task.txt`, `.sdd/report_template.txt`
- 우선순위: 이 PRD와 v2가 다르면 **v2가 우선**한다. 단 판정이 다르게 정한 항목은 판정이 우선한다. v2 리뷰 수정지시(`[수정-n]`)는 v2 수정 1·2차에 반영되었다.
- **외부 기상자료(KMA 단기예보·ASOS)는 쓰지 않는다**(사용자 결정). 기상의 가치는 데이터 자체의 실측 기상을 완전예보로 쓰는 oracle-weather ablation(US-007)으로만 확인한다.

### 출처 태그

| 태그 | 의미 |
|---|---|
| `[v2 §x]`, `[Ax]`, `[v2 부록B]` | v2 절 / 부록 A 설계결정 / 부록 B 재계산표 |
| `[결정-데이터]` `[결정-프로토콜]` `[결정-외부]` `[결정-코드]` `[결정-대회]` | decisions.md의 해당 절 |
| `[요금표]` | kepco_tariff_2021.md |
| `[수정-n]` | v2 리뷰 수정지시 n번 (1 선택규칙, 2 백본 τ, 3 강수량, 4 ponytail, 5 입력 마스킹·e₀, 6 is_missing 분리·관측슬롯 최대, 7 보정 기준·클리핑, 8 누수/상태안정 테스트, 9 복사 인공물 판정(사용자 결정)·경로최대 부풀림, 10 ≤08-31 설계통계, 11 일정·컷리스트). 모두 v2 수정 1·2차에 반영됨 |
| `[복사판정]` | `claudedocs/research_copy_vs_outlier_20260922.md` (복사 인공물 vs 이상치 판정 보고서, 사용자 결정으로 채택) |
| `[판정]`, `[판정 Qn]` | `.sdd/progress.md`의 "Decision (user)"·"Ruling" 줄. `Qn`은 이 PRD 옛 미해결 질문 n번에 대한 판정 |
| `[재계산]` | PRD 작성 중 원본 CSV를 읽기 전용으로 다시 계산한 값 (설계통계는 ≤ 2021-08-31만) |
| `[PRD]` | v2에 값이 없어 이 PRD가 정한 기본값. 판정 Q4로 승인됨(v2가 값을 정한 곳에서는 v2가 우선) |
| `[환경]` | 현재 `pyproject.toml`·환경 기록(torch 2.14.0+cu126, RTX A6000 ×2) |

---

## 1. 개요 (Introduction/Overview)

KAMP 제6회 경진대회 문제 ⑤(제조 생산데이터 기반 전력사용량 예측 및 최대피크 위험조건 분석)를 위한 예측 시스템을 **새 파이썬 패키지 `gmst/`의 스크립트**로 구현한다 `[결정-코드]`. 입력은 공식 데이터 `okm_augumented_2021.csv` 한 개(2021-01-01~09-14, 시간행 6,168개)이며, 대회가 테스트셋·예측구간·파일형식을 주지 않으므로 v2가 정의한 문제를 그대로 푼다 `[결정-대회, v2 §4.5]`.

- **예측문제**: 매일 00:00 발행, 전일 23:45까지 정보 사용, 당일 15분 96슬롯(H=96) 예측, 정보길이 L=672 `[v2 §4.5]`.
- **피크위험**: 달력일 최대 \(M_d\)가 임계값 \(C\)(fold 학습구간 가동일 일최대의 q50/q75/q90)를 넘을 확률 `[v2 §4.5, A1]`.
- **모델 사다리**: B0(lag-7 naive), B0′(가동달력 naive), **B1(LightGBM, 벤치마크 — 대체 제출 후보 아님)**, BB(백본 단독), B2/B3/B4(백본 + 고정/조건부 HMM, B4 = Contest Core v1: 순환 RW2 백본 + K=3 조건부 Markov + 상태 내 AR(1) + MC 피크위험 + Platt 보정) `[v2 §15.4, §19]`. 제출 모델은 항상 Core B4 계열이다: B4가 §15.6의 성공 기준(CI 게이트)을 못 넘기면 사전등록 개선 루프(B4-H → B4-IO(조건부 디코더) → B4-KAN, §9.6/US-021)를 2026-10-03까지 돌려 최선의 변형을 제출한다(사용자 결정) `[v2 §5.2, §9.6, §15.6]`.
- **두 트랙**: MAIN(A+, 제출) = 과거 + 달력 + 당일 가동플래그 + 공휴일, SCENARIO(B) = A+ + 당일 생산량(계획 대용, 상한) `[v2 §4.6, 결정-프로토콜]`.
- **검증**: 07-01 이후 rolling-origin f1–f4(각 14일, gap 1일, 중복그룹 purge), 봉인 테스트 09-01~09-14는 선택이 끝난 뒤 1회 `[v2 §15, 결정-프로토콜]`.
- **현장활용**: SCENARIO 트랙의 생산일정 이동 what-if와 한전 산업용(을) 2021 요금 기준 ₩ 환산. 래칫 바닥 222(07-19) 때문에 7–8월 기본요금 절감은 사실상 0이고, ₩ 절감의 주 수단은 TOU 전력량요금 이동이다 `[v2 §12.1, 판정]`.
- **외부자료**: 2021 공휴일(하드코딩)과 한전 요금표만. KMA 기상은 쓰지 않는다(사용자 결정). MAIN은 과거 관측 기상만 쓰고, 기상의 최대 기여는 oracle-weather ablation으로만 보인다 `[v2 §4.6, §4.7]`.
- **재현성**: `uv run python -m gmst.run_all` 한 줄로 전처리 → 분할 → 학습 → 추론 → 결과. 노트북은 결과 표시 전용 `[결정-코드, 결정-대회]`.

이 PRD는 구현자용 요구사항이다. v2는 연구 기획서로 남고, 이 PRD는 v2 + 결정 + 판정을 스토리·검증 가능한 수용기준으로 옮긴다. 기존 노트북 산출물(`okm_15min_2021.csv`, `okm_cv_splits_2021.csv`)은 스크립트 산출물로 **대체**된다 `[v2 §18]`.

---

## 2. 목표 (Goals)

- G1. `uv run python -m gmst.run_all`이 원본 CSV에서 시작해 종료코드 0으로 끝나고 `results/test_predictions.csv`(1,344행 × 33열)를 만든다 `[v2 §4.5, 수정-6, 결정-대회]`.
- G2. `uv run pytest -q`가 GPU 머신과 `CUDA_VISIBLE_DEVICES=""`(CPU) 양쪽에서 모두 통과한다 `[환경]`.
- G3. 파이프라인이 기준수치를 재현한다: lag-7(참조일 마스킹 건너뜀) MAE f1–f4 = 6.23 / 25.77 / 65.34 / 9.84 (n = 960 / 1,152 / 1,248 / 1,272) `[v2 §4.3, 부록B]`. B0′(28일 규칙, FR-45) = 14.85 / 10.38 / 14.49 / 15.17 (n = 1,152 / 1,248 / 1,344 / 1,272) `[v2 §4.3, 부록B, 판정 Q10]`. 임계값 C는 확정된 복사 규칙(DC1·DC14) 적용 후 f1 = 181 / 189.5 / 198 (n = 55), 테스트 = 186.5 / 198 / 210.7 (n = 92) `[v2 §4.5, A1]`.
- G4. 동일 조건(같은 fold·마스킹·대상집합)에서 B0, B0′, BB, B1, B2, B3, B4의 MAE/RMSE/CRPS/coverage/PeakMAE/Brier/AUC/F1이 `results/metrics.csv`에 n과 함께 기록된다 `[v2 §15.5, §16, 결정-대회 2]`.
- G5. **B1(LightGBM)은 벤치마크이며 대체 제출 후보가 아니다**(사용자 결정). ΔMAE와 Δ평균Brier(후보 B4 변형 − B1)의 일 블록 부트스트랩 95% CI 상한이 **둘 다** < 0인 것은 제출 스위치가 아니라 **성공 기준**이다. 만족하면 그 변형을 제출하고, 만족하지 못하면 사전등록 개선 루프(① B4-H → ② B4-IO(조건부 디코더) → ③ B4-KAN, US-021)를 f1–f4 OOF + R10 내부검증으로 2026-10-03까지 돌려, 그때까지 평균 OOF MAE가 가장 낮은(동률이면 C50/C75/C90 평균 Brier) 변형을 제출한다. B1은 어떤 경우에도 제출되지 않는다. 판정 근거·시도한 변형 전부·최종 선택이 `results/selection.json`에 남는다. CRPS/pinball은 분위수 열 평가용으로 보고하며 성공 기준·중단 규칙에 들어가지 않는다 `[v2 §5.2, §9.6, §15.6, 판정 Q5]`.
- G6. 위험확률 완료기준이 f1–f4 OOF에서 판정·기록된다: 세 임계값 각각 보정 후 Brier < **가동여부×요일유형 조건부 기후값** Brier, 날짜 간 AUC ≥ 0.70, 조건부 기후값 대비 BSS > 0 (q90은 f1–f4 합산 13사건으로만 판정). 미달이면 숨기지 않고 보고 `[v2 §11.3, A26]`.
- G7. SCENARIO 13개 시나리오(기존 + 4변환 × 이동률 10/20/30%)의 \(E[M]\)·위험·₩ 변화가 `results/scenarios.csv`(일)·`results/scenarios_month.csv`(청구월)에 남는다. 7–8월 기본요금 변화는 래칫 바닥 222 때문에 사실상 0이고, ₩ 변화의 주 성분은 TOU 전력량요금임이 표에서 드러난다 `[v2 §12, §12.1, A24, A28, A29, 판정]`.
- G8. 모든 `# ponytail:` 주석이 `PONYTAIL-DEBT.md`에 빠짐없이 수록된다 `[결정-코드, 수정-4]`.

---

## 2A. 데이터 주의점 (Data Caveats)

원칙: **측정 복원**은 전처리(`gmst/preprocess.py`), **날짜 단위 사용정책**(복사/오염/가동/공휴일/fold 역할)은 날짜표(`okm_cv_splits_2021.csv`, `gmst/splits.py`), **적용**은 로더(`gmst/features.py`)가 한다. ablation은 로더 스위치 하나로 바뀐다 `[결정-데이터, v2 §4.2]`. 로더가 목표를 가린 날은 **입력(lag 참조, HMM warm-up)에서도 가린다** `[수정-5]`.

| # | 주의점 | 증거 수치 | 처리 규칙 | 적용 위치 | 테스트 |
|---|---|---|---|---|---|
| DC1 | **완전 복사일(증강)**: 96점 전력벡터가 다른 날과 완전일치 | 45그룹 / 160일(62%), 복사 115일, 고유 142일. 공변량은 복사 안 됨: 03-01(삼일절, 생산 0)이 02-01(생산 20,368)과 전력 완전동일(같은 그룹에 06-01). "첫 날짜 = 원본" 규칙은 01-24 그룹에서 실패: 01-24(일, 생산 0)가 가동 모양, 02-24(수, 생산 14,979)가 모양과 맞음. 07-01 이후 유일한 복사쌍 07-28 = 07-30 `[v2 §4.2, 결정-데이터, 복사판정, 재계산]` | 그룹마다 원본 1개 = **가장 이른 날**. 단 **강한 모순일 때만 교체**: 가장 이른 날이 완전 가동 모양(가동 슬롯 = 전력 > 40인 슬롯이 ≥ 48)인데 일 생산량이 0이고, 그룹 안에 생산량 > 0인 다른 날이 있으면 그중 가장 이른 날이 원본(→ 교체는 01-24→02-24 1건뿐). 요일·활동량까지 맞추는 넓은 일관성 규칙은 평일 휴무·야간 꼬리·토요일 반일 때문에 5개 그룹을 잘못 옮기므로 쓰지 않는다 `[v2 §4.2, 판정 Q11, 복사판정]`. 완전 복사 115일은 `is_copy=true`, `copy_kind="exact"`, `copy_of`=그룹 원본 날짜. 목표 마스킹(학습·검증·테스트 동일), lag 참조·warm-up 입력도 NaN. 공변량은 유지. ablation `include_copies=True` → 보고서 제2장 | 날짜표 플래그 + 로더 | `test_splits.py::test_copy_days`: `(copy_kind=="exact").sum()==115`, `is_copy.sum()==122`(완전 115 + 편집 7), `event_id.n_unique()==142`, dup_size>1 그룹 45개·160일, 02-01/03-01/06-01 같은 event_id이고 is_copy = F/T/T, `copy_of`(03-01) = 02-01, **01-24 T(`copy_of`=02-24)·02-24 F**, 07-28 F·07-30 T, 원본이 첫 날짜가 아닌 그룹은 01-24 그룹 하나. `test_features.py::test_copy_mask`: 03-01 행 Y 전부 NaN, `include_copies=True`면 유한 |
| DC2 | **07-13·07-15 순서 뒤섞임(suspect)**: 원자료 `시간`이 오름차순 키로 덮어써지고 전력행이 크기순 정렬 | 불일치 48행. Spearman(전력, 행순서) 0.857/0.911(15분 기준; 시간평균 기준 0.864/0.910) = 257일 중 상위 2일. 자정 출구 절벽 188→75 / 186→67. 일평균/일최대 136.1/190, 135.6/202 = 정상 화·목 → 값은 실측, 시각 소실. 생산량 0 = 결측. lag 입력 오염: lag-7 입력으로 쓰면 07-20/07-22 MAE 35.4/33.5 vs lag-14 10.7/9.3. f1 lag-7 MAE 제외 8.96 / 포함 12.31 `[결정-데이터, v2 §4.2, 부록B]` | `is_suspect`. 목표 + 생산량 마스킹(학습·주평가·lag 입력·warm-up 모두). 원자료 보존. **가동일로 취급**(`is_operating=True`). 민감도 ablation `include_suspect=True`("suspect 포함" 점수) | 날짜표 + 로더 | `test_splits.py::test_suspect`: 정확히 2일, 둘 다 is_operating=True. `test_features.py::test_suspect_mask`: 두 날의 Y·생산량 NaN, B1 특징 `y_lag7`이 07-20·07-22에서 전부 NaN |
| DC3 | **정전 0값** | 74점 = 2021-08-28 17:30~08-29 11:15 연속 72점(구간 시작 라벨; 결정의 "17:45–11:30"은 구간 끝 라벨로 같은 72점) + 09-08 12:00, 12:15 2점. 일별 26/46/2점. 유휴 베이스 19–20이므로 0은 공급·계측 손실 `[v2 §4.2, 부록B, 결정-데이터]` | 전처리에서 전력 null + `is_missing=True`. 학습 손실·지표 제외(n 보고), HMM emission ≡1, LightGBM 목표 결측행 drop, 결측을 참조하는 lag는 NaN. 일 피크지표는 `n_missing ≤ 4`인 날만, 실측·예측 일최대는 **같은 관측 슬롯**에서 계산 `[A12, 수정-6]` | 전처리 (+ 날짜표 `n_missing`, 로더) | `test_preprocess.py::test_zero_power`: 전력 null 74, `is_missing` 74, 날짜별 26/46/2, 유한 전력 최솟값 > 0. `test_evaluate.py::test_peak_same_slots` |
| DC4 | **풍속 결측** | 원자료 3시간 = 06-01 01:00–02:45(2시간), 07-04 20:00–20:45(1시간) → 15분 12행 `[v2 §4.2]` | 시간 계열에서 선형보간 후 4슬롯 반복 | 전처리 | `test_preprocess.py::test_wind_interp`: null 0, 06-01 01시 값 = 00시·03시 사이 선형값 |
| DC5 | **강수량 = 당일 누적(리셋 지연)** | ≤08-31 원자료: 00h 값이 전일 23h와 같은 자정 전이 184/242건, 그중 **24건은 값 > 0**(리셋이 01h로 밀린 날). 감소 전이 23h→00h 47건(자정 리셋), 00h→01h 32건(지연 리셋), 일중 4건(04→05, 11→12, 17→18, 19→20). 01-24 00h 결측 1건(15분 4행). 일내 차분이면 위 24일의 00h 누적값이 두 번 계산됨 `[v2 §4.2, 부록B]` | `강수량_증분`: **자정을 가로지르는 연속 시간 계열**의 차분, 음수 차분 = 리셋 → 현재값, 01-24 00h 결측 → 0 `[수정-3]` | 전처리 | `test_preprocess.py::test_precip_increment`: 합성 `[0, .5, 1.0, (익일 00h) 1.0, (01h) .2]` → `[0, .5, .5, 0, .2]`; 실데이터 null 0·음수 0 |
| DC6 | **`공장인원`·`평균` 누수** | `공장인원` = 생산량/S (S = 네 전력값 합) 정확한 항등식, S>0 6,151행에서 오차 ≤ 4.9e-9 (S=0 17행은 null). `평균` = ⌊mean4 + 0.5⌋ 6,168/6,168행 `[v2 §4.1, D3/D4]` | 항등식을 계산해 감사 파일에 기록한 뒤 두 열 삭제. 모든 프로토콜·과거 구간에서도 사용 금지 | 전처리 | `test_preprocess.py::test_leakage_audit`: 출력에 두 열 없음, `data_audit.json`의 `공장인원_max_abs_err ≤ 4.9e-9`, `공장인원_n == 6151`, `평균_match == 6168` |
| DC7 | **`전기요금(계절)`은 요금이 아님** | 1–2월 109.8, 3–5·9월 167.2, 6–8월 191.6(월별 고유값 1개) = 고압A 선택Ⅱ **최대부하**(봄가을 109.3/겨울 166.7/여름 191.1) +0.5, 게다가 겨울↔봄가을 뒤바뀜. 월 경계(1–2 / 3–5·9 / 6–8)만 한전 계절과 일치 `[v2 §4.1, 요금표]` | 월별 **계절 지시변수로만** 사용(한전 월→계절 매핑과 동치). ₩ 계산에 절대 사용 안 함. TOU 단가는 요금표에서 생성 | 로더(특징) + 요금 모듈 금지 | `test_features.py::test_season_indicator`: 월별 고유값 1개, 계절↔값 1:1. `test_scenario.py::test_no_data_tariff`: `gmst/scenario.py` 소스에 문자열 `전기요금` 없음 |
| DC8 | **생산량은 실적(계획 아님)** | corr(전력ₜ, 생산량ₜ₋ℓ) ℓ=−1/0/1: 07-01~08-31 0.527/0.579/0.496 (≤08-31 전체 0.480/0.514/0.440) → 동행지표, 좌우 대칭. 1–6월은 복사로 생산량↔전력 연결 끊김(≤08-31 고유일 corr 0.554 vs 복사 그룹일 0.491) `[v2 §4.6, §12, 부록B]` | MAIN(A+)은 당일 생산량 미사용(과거 생산량만). SCENARIO(B)만 당일 생산량을 계획으로 사용, 항상 "상한"으로 표기, 제출 금지. 시나리오 민감도는 07-01 이후 학습 모델에서만 | 로더(프로토콜) | `test_leakage.py::test_protocol_prod`: 당일 생산량 교란 시 A+ 예측 불변, B 예측 변화 |
| DC9 | **중복그룹 purge** | 학습일의 `event_id`가 그 fold 검증일 event 집합에 있으면 `purged`. 실제로는 전 fold 0일(07-28=07-30이 모두 f2 검증창 안) `[v2 §15.1–15.2]` | purge 로직 유지(복사 정의 ablation에서 작동해야 함). LOEO(fold = event_id)는 보조 진단 | 날짜표 | `test_splits.py::test_purge_synthetic`(합성 표에서 purged 발생), 실표 fold별 purged 0 |
| DC10 | **봉인 테스트** | 09-01~09-14, 14일 × 96 = 1,344행. 테스트 C는 ≤ 08-30 학습으로 186.5/198/210.7(DC1·DC14 규칙 적용). v2 작성 중 기준선 값이 테스트창에서 한 번 계산됐으나 어떤 선택에도 쓰이지 않음(ledger 기록) → 보고서에 공개. 봉인 전 EDA·검증보고서의 일부 기술통계도 09-01~09-14를 포함하므로 비교용으로만 인용 `[v2 머리말, §4.5, §15.2, 수정-10]` | 로더는 `unseal=False`(기본)일 때 테스트창 Y·X를 전부 NaN으로 반환. `gmst/` 안에서 `unseal=True`는 `run_all.py`의 최종 단계에만. **모든 설계통계(임계값·τ·K·p*·빈·기후값)는 ≤ 2021-08-31 데이터만** | 로더 + 러너 | `test_features.py::test_sealed`, `test_style.py::test_unseal_only_in_run_all` |
| DC11 | **요금 공휴일 계량 규칙과 래칫 바닥** | **관공서 공휴일(일요일 포함, 임시공휴일 제외)**은 최대수요·사용량 모두 경부하 계량, 공휴일 아닌 토요일의 최대부하 사용량은 중간부하 계량. 요금적용전력 래칫은 중간·최대부하 시간대만. 08-16(광복절 대체공휴일)은 요금상 공휴일(경부하)로 **가정**. 2020-12 자료 없음, 계약전력 미상. ≤08-31 중간·최대부하 월 최대(일요일·공휴일 제외): 1월 202, 2월 195, 7월 222, 8월 207. 전체 최대 **222 = 07-19(월) 11:15**(여름 최대부하 10–12시), 마스킹 대상 아님. 예: 2021-07-11(일) 11:00, 100 kW → 1,402.5₩(=100×0.25×56.1, 여름 경부하 단가) `[v2 §12.1, 부록B, 요금표, 판정 Q1·Q7, 재검토 R1]` | 공휴일(일요일 포함, `HOLIDAYS_2021` ∪ `daytype=="sun"`) → 경부하(에너지·수요 둘 다), 비공휴 토요일 최대부하 → 에너지만 중간부하 단가(수요는 래칫 포함). 08-16은 스위치 `tariff_0816=True`(기본 공휴일, 가정을 결과에 명시). 래칫 이력 = fold 창이 아니라 **시나리오 월까지의 전체 관측 이력**(마스킹 점 제외) → 7–9월 바닥 222. 30% 하한 미적용·2020-12 부재를 결과에 명시 | 요금 모듈 + 날짜표 `is_holiday`, `daytype` | `test_scenario.py::test_bands`(아래 US-015 사례표), `test_scenario.py::test_floor_222`, `test_scenario.py::test_sunday_offpeak`(07-11 사례) |
| DC12 | **가동플래그 오류** | EDA의 `is_operating`(일 생산량 합>0)은 suspect 2일을 비가동으로 표기 → 64일. 보정 후 비가동 **62일** `[v2 §4.2]` | `is_operating = (일 생산량 합 > 0) or is_suspect` | 날짜표 | `test_splits.py::test_operating`: 비가동 62, 가동 195 |
| DC13 | **공휴일 ≠ 휴무** | 공휴일 8일 중 05-05·05-19·08-16은 생산 있음(08-16 일생산 40,952), 02-11·02-12·02-13·03-01은 복사일. 비가동 62일 = 일 36·토 4·평일 22(공휴일 4 + 비공휴 평일 18) `[v2 §4.3, 부록B]` | 가동플래그가 주 변수, 공휴일은 보조. 요일유형 wk/sat/sun = 183/37/37 `[재계산]` | 날짜표 | `test_splits.py::test_holiday_daytype`: `is_holiday` 8일, daytype 183/37/37 |
| DC14 | **복사 인공물 vs 이상치 (편집된 복사)** — 사용자 결정 | "생산량 0인데 가동수준 전력" 원본 7일(v2 부록B)을 조사한 결과 **이상치가 아니라 복사 인공물**: 그 날의 측정값이 아니므로 이상치 처리(유지·견고손실)가 아니라 마스킹이 맞다(Aguinis 외 2013의 "그 단위의 진짜 관측인가" 기준). 가동구간(전력 > 40) 슬롯의 정확히 같은 값 수: 7–8월 복사 아닌 날짜쌍 1,830개에서 p99 6, **최대 9**(복사판정 보고서는 대상일 집합 차이로 1,769쌍, 같은 결론); 문제의 날 03-07 67, 03-21·03-28 64, 01-09 50, 01-10 48, 01-16 33, 01-01 27. 원본과 다른 칸은 4~12칸, 02~07시에 몰림(복사 후 편집 흔적). 새 규칙 적용 후 ≤08-31 셀평균 잔차 sd 가동 32.36 / 비가동 7.31(01-02 한 날이 4.47 → 7.31로 올림; 01-02는 휴무일의 실제 야간 꼬리로 보고 가동플래그 예외를 두지 않음). 반대로 **진짜 드문 날**(08-16 공휴일 부분가동, 09-08 짧은 차단 전후)은 실측이므로 유지 `[v2 §4.2, §9, 부록B, 복사판정, 판정]` | 편집된 복사 = 완전복사 그룹의 원본이 아닌 날 중, 다른 어떤 날과 가동구간 슬롯 값이 **20개 이상** 정확히 같은 날 → `is_copy=true`, `copy_kind="edited"`, `copy_of` = 가동구간 동일값이 가장 많은 상대 날짜(동률이면 마스킹되지 않은 날, 그다음 가장 이른 날). 전이적 연결 없음(부분복사 01-16이 두 그룹과 겹쳐도 그룹 병합 안 함; 편집복사는 자기 `event_id` 유지). 목표 + 입력(lag, warm-up) 마스킹, `include_copies=True`에 함께 포함. 판정 규칙은 모델 학습 전에 고정. 진짜 드문 날은 유지하고 가동·비가동 층별 지표로 보고 | 날짜표 `is_copy`·`copy_kind`·`copy_of` + 로더 | `test_splits.py::test_edited_copies`: `copy_kind=="edited"` = 정확히 01-01, 01-09, 01-10, 01-16, 03-07, 03-21, 03-28; 각 날과 `copy_of`의 가동구간 동일값이 ≥ 20이고 다른 어떤 날과의 값보다도 작지 않음; 07-01~09-14에 `is_copy` 신규 없음(07-30 `exact`만); 07-01~08-31 비복사 쌍의 가동구간 동일값 최대 9. `test_features.py::test_edited_mask`: 7일의 Y가 목표·`y_lag7` 참조에서 NaN |
| DC15 | **시각 규약·시간 경계 점프** | 구간 시작 라벨(“15분”→HH:00, “60분”→HH:45), 하루 00:00–23:45. 시간 경계 \|Δy\| 12.39 vs 시간 내부 7.45/6.51/6.23 `[v2 §4, D8]` | 슬롯 내 위치 `q % 4`를 특징으로 | 전처리 + 로더 | `test_preprocess.py::test_grid`: 24,672행, 15분 간격 연속, 날짜당 96 |
| DC16 | **정수·반복 베이스부하 / 단위 미상** | 20–30대가 07-01~08-31 사용가능점의 37.2%(2,082/5,592). 전력 단위 없음(값 0–222) `[v2 §4.2, §12.1, A19]` | emission scale 하한 ε=1. ₩는 "15분 평균 kW" 가정 조건부로, 상대변화(%) 병기 | HMM + 요금 모듈 | `test_hmm.py::test_sigma_floor` |

---

## 3. 사용자 스토리 (User Stories)

**모든 스토리 공통 규칙** `[결정-코드, 수정-4]`: ponytail 스타일(최소 코드, stdlib → 이미 설치된 의존성 우선, 한 구현뿐인 인터페이스·팩토리·설정파일 금지). 의도적 지름길은 반드시 `# ponytail: <ceiling>, <upgrade trigger>` 주석. 모든 스토리의 마지막 수용기준은 "`uv run pytest -q` 통과". 구현 모델: fable `[결정-코드]`.

### US-001: 의존성과 폴더 골격
**Description:** 구현자로서 필요한 의존성과 빈 패키지·테스트 골격을 갖춰, 이후 스토리가 같은 규칙 위에서 시작하게 하고 싶다.

**Acceptance Criteria:**
- [ ] `uv add lightgbm`(런타임)과 `uv add --dev pytest`(dev 전용)만 실행, 결과가 `pyproject.toml`·`uv.lock`에 반영됨. torch cu126 `[tool.uv.sources]`/`[[tool.uv.index]]` 블록은 그대로 `[v2 §18, 판정]`
- [ ] `pyproject.toml`에 `[tool.pytest.ini_options]` `pythonpath = ["."]`, `testpaths = ["tests"]`
- [ ] 직접 의존성 목록에 `requests`, `scipy`, `scikit-learn`, `pandas` 없음. `.env`·자격증명을 읽는 코드 없음(외부 수집 없음)
- [ ] `gmst/__init__.py`에 `ROOT`, `DATA`, `RESULTS`(`pathlib.Path`)만 정의, `DATA.name == "5. 자원 최적화 AI 데이터셋"`, `RESULTS == ROOT / "results"`
- [ ] `tests/test_style.py`: (a) `gmst/`·`tests/`의 모든 `ponytail:` 주석이 정규식 `#\s*ponytail:\s*[^,]+,\s*\S+`와 일치, (b) `gmst/` 어디에도 `import scipy`·`from scipy`·`sklearn`·`pandas` 없음, (c) `gmst/` 안에서 문자열 `unseal=True`는 `gmst/run_all.py`에만 (파일이 아직 없으면 통과)
- [ ] `uv run python -c "import lightgbm, torch, polars, gmst"` 성공
- [ ] `uv run pytest -q` 통과

### US-002: 전처리 스크립트 (측정 복원)
**Description:** 분석가로서 원본 시간행 CSV를 결함이 복원된 15분 CSV로 바꾸는 스크립트를 원한다. 노트북 산출물을 대체하고 누수 감사를 남기기 위해서다.

**Acceptance Criteria:**
- [ ] `gmst/preprocess.py`에 `build_15min() -> pl.DataFrame`(파일 쓰기 없음)과 `run()`(파일 쓰기) 존재, `uv run python -m gmst.preprocess`로 실행
- [ ] 출력 `DATA/okm_15min_2021.csv`: 24,672행, 열 순서 정확히 `datetime, 전력, 생산량, 기온, 풍속, 습도, 강수량_증분, 전기요금(계절), 인건비, is_missing`, datetime 문자열 형식 `%Y.%m.%d %H:%M:%S`, 첫 행 `2021.01.01 00:00:00`, 끝 행 `2021.09.14 23:45:00`
- [ ] 원본 `okm_augumented_2021.csv`는 수정되지 않음(sha256 전후 동일)
- [ ] DC3·DC4·DC5·DC6·DC15의 테스트가 `tests/test_preprocess.py`에 있고 통과
- [ ] `RESULTS/data_audit.json`에 최소 키: `n_rows_raw`(6168), `n_rows_15min`(24672), `n_days`(257), `zero_points`(74), `zero_by_day`({"2021-08-28":26,"2021-08-29":46,"2021-09-08":2}), `wind_null_hours`(3), `precip_null_hours`(1), `time_col_mismatch_rows`(48), `time_col_mismatch_dates`(["2021-07-13","2021-07-15"]), `공장인원_max_abs_err`, `공장인원_n`(6151), `평균_match`(6168)
- [ ] `uv run pytest -q` 통과

### US-003: 날짜표와 분할 (플래그 + fold 역할)
**Description:** 분석가로서 날짜당 한 행의 사용정책 표를 원한다. 복사·오염·가동·공휴일·fold 역할을 한곳에서 정의해 로더와 노트북이 같은 사실을 쓰게 하려는 것이다.

**Acceptance Criteria:**
- [ ] `gmst/splits.py`에 상수 `HOLIDAYS_2021`(8일, 출처 주석 "한국천문연구원 특일정보"), `SUSPECT`(07-13, 07-15), `FOLDS`(f1–f4 검증창 + `test`: 09-01~09-14), `build_days(df15) -> pl.DataFrame`, `run()`
- [ ] 출력 `DATA/okm_cv_splits_2021.csv`: 257행 × 17열, 열 순서 정확히 `date, event_id, dup_size, is_copy, copy_kind, copy_of, is_suspect, is_operating, is_holiday, daytype, n_missing, anomaly, f1, f2, f3, f4, test`, date·`copy_of` 형식 `%Y.%m.%d`(복사 아니면 빈 문자열), `copy_kind` ∈ {`""`, `exact`, `edited`}, 불리언 `true/false` `[v2 §15.3, 판정: v2 이름 우선]`
- [ ] 복사 판정(사용자 결정 수용기준) `[v2 §4.2, §15.3, 복사판정]`: 새로 마스킹되는 날(`copy_kind == "edited"`)은 정확히 2021-01-01, 01-09, 01-10, 01-16, 03-07, 03-21, 03-28; 01-24는 `copy_kind="exact"`·`copy_of=2021.02.24`, 02-24는 `is_copy=false`(원본); 07-01~09-14에는 `edited`가 없음(기존 07-30만 `exact`); 복사일은 목표와 lag·HMM warm-up 입력 모두에서 마스킹됨(US-004 테스트와 연결). 이 세 수락기준은 `splits.py`에서 assert로도 고정
- [ ] 역할 개수: f1 train 186 / gap 1(07-06) / val 14 / 빈칸 56; f2 200 / 1(07-20) / 14 / 42; f3 214 / 1(08-03) / 14 / 28; f4 228 / 1(08-17) / 14 / 14; test 242 / 1(08-31) / test 14 / 0; 모든 fold `purged` 0
- [ ] DC1·DC2·DC9·DC12·DC13·DC14 테스트가 `tests/test_splits.py`에 있고 통과; `anomaly` = {07-13, 07-15, 08-28, 08-29, 09-08}; `n_missing` = 08-28 26, 08-29 46, 09-08 2, 나머지 0
- [ ] `uv run pytest -q` 통과

### US-004: 로더 — 마스킹·프로토콜·봉인
**Description:** 모델 구현자로서 마스킹 정책과 프로토콜이 이미 적용된 배열을 받고 싶다. 모든 모델이 같은 조건에서 학습·평가되고 ablation이 스위치 하나가 되게 하려는 것이다.

**Acceptance Criteria:**
- [ ] `gmst/features.py::load_panel(include_copies=False, include_suspect=False, unseal=False) -> dict`, 키 `dates`(257개 `date`), `Y`(257×96 float64), `X`(dict: `생산량, 기온, 풍속, 습도, 강수량_증분` 각 257×96), `is_missing`(257×96 bool), `days`(날짜표 DataFrame)
- [ ] 기본 호출 유한 Y 개수 11,352; ≤07-05 6,240; ≤08-30 11,256(118일, 가동 93) `[v2 §4.2, 부록B]`. `include_copies=True`(완전·편집 복사 모두 포함) 23,064, `include_suspect=True` 11,544, 둘 다 23,256 (모두 봉인 상태) `[재계산]`
- [ ] 봉인: 기본 호출에서 테스트 역할 14일의 Y와 모든 X가 NaN (`test_sealed`)
- [ ] `PROTOCOLS` dict: `"A": set()`, `"A+": {"op","hol"}`, `"A+W*": {"op","hol","wx_obs"}`(oracle-weather, 데이터 자체 대상일 실측 기상; 제출 금지), `"B": {"op","hol","prod"}`. 외부 기상 예보 프로토콜(`A+W`)은 없음 `[v2 §4.6, 사용자결정]`
- [ ] `lgbm_rows(panel, day_idx, protocol, backbone)`가 FR-31의 특징명 목록 그대로 열을 만들고, `y_lag7`이 07-20·07-22에서 전부 NaN (DC2)
- [ ] DC1·DC2·DC7·DC14 로더 테스트가 `tests/test_features.py`에 있고 통과(복사·편집복사·suspect 날이 목표와 `y_lag*`·`prev_*` 참조 양쪽에서 NaN)
- [ ] `uv run pytest -q` 통과

### US-005: 지표·rolling-origin 평가기·naive 기준선
**Description:** 연구자로서 동일 평가조건의 지표 모듈과 fold 루프, 그리고 넘어야 할 naive 기준선을 원한다. v2 §4.3 표를 파이프라인으로 재현하는 것이 첫 검증이다.

**Acceptance Criteria:**
- [ ] `gmst/evaluate.py`: `TAUS`(0.05…0.95, 19개), `mae, rmse, crps, coverage, brier, auc, peak_mae, peak_hit, prf, thresholds, rolling_origin`
- [ ] `gmst/baselines.py`: `b0p`(B0′), `b0`(B0), `lag7_skip_table()`(보고용)
- [ ] `lag7_skip_table()` MAE f1–f4 = 6.23 / 25.77 / 65.34 / 9.84, n = 960 / 1,152 / 1,248 / 1,272 (소수 둘째 자리 일치)
- [ ] B0′(FR-45, 28일 규칙) OOF MAE f1–f4 = 14.85 / 10.38 / 14.49 / 15.17, n = 1,152 / 1,248 / 1,344 / 1,272 `[v2 §4.3, 부록B, 판정 Q10]`; 폴백은 07-31(비가동 토요일 → 07-25)과 08-02(비가동 평일 → 08-01) 두 날뿐; 28일 제한을 끄면 f2 = 12.06(07-31 → 01-02)으로 재현(보고용 민감도)
- [ ] `thresholds`(fold별 자기 train): f1 = (181, 189.5, 198) n=55, f2 = (182, 193, 200.2), f3 = (184, 197.8, 210.1), f4 = (185, 198, 211), 테스트용(≤08-30 학습) = (186.5, 198, 210.7) n=92 `[v2 §4.5, A1]`
- [ ] fold별 자기 C 기준 사건일 수(q50/q75/q90): f1 8/8/4, f2 7/7/7, f3 6/4/2, f4 10/2/**0**, 합계 31/21/13 / 51일. 모든 Brier·F1·AUC 표에 사건 수 병기, f4 q90 판별 지표는 "정의 안 됨"(NaN) `[v2 §4.5, §15.5, §16.4]`
- [ ] 일 피크 사용가능일: f1 12, f2 13, f3 14, f4 12 `[v2 §15.2]`
- [ ] 합성 예제로 CRPS = 2 × 평균 pinball, coverage, AUC(동점 0.5), F1/precision/recall 단위테스트
- [ ] `test_evaluate.py::test_peak_same_slots`: 관측 슬롯 마스크가 있으면 실측·예측 일최대를 같은 슬롯에서 계산
- [ ] `results/oof_slots.csv`, `results/oof_days.csv`, `results/metrics.csv`가 FR-43/44 스키마로 B0·B0′ 행을 담음
- [ ] `uv run pytest -q` 통과

### US-006: 패널티형 순환 RW2 백본
**Description:** 연구자로서 가동여부×요일유형별 96슬롯 매끈한 일프로파일을 폐형해로 얻고 싶다. HMM의 평균 구조와 LightGBM 특징, BB 참고모델로 재사용하기 위해서다.

**Acceptance Criteria:**
- [ ] `gmst/backbone.py::fit_backbone(Y, op, daytype, train_idx, tau, half_life=60, min_days=5) -> (profiles (2,3,96), fallback_classes)`와 `predict_backbone(profiles, op, daytype)`
- [ ] 합=0 제약 없음, `numpy.linalg.solve((W_c + tau*C2.T@C2), b_c)` 사용. `C2 @ ones == 0`, 선형 램프에는 순환 경계에서 0 아님
- [ ] `tau=0, half_life=None, min_days=1`로 07-01~08-31(마스킹) 셀평균 적합 시 R² = 0.923 (n = 5,592), 잔차 sd(ddof=1) 가동 19.95 / 비가동 1.55 (±0.02), 잔차 lag-1 자기상관 0.909 (±0.002). 같은 적합을 ≤08-31 전체(마스킹)에 하면 잔차 sd 가동 32.36 / 비가동 7.31, lag-1 0.961 (01-02를 빼면 비가동 4.47) `[v2 §4.3, §9, §11.2, 부록B]`
- [ ] τ가 매우 크면(1e8) 각 프로파일의 표준편차 < 1e-3 (상수로 수렴)
- [ ] f1 학습에서 `fallback_classes`에 (가동, 일)과 (비가동, 토)가 반드시 포함(사용가능일 < 5)
- [ ] 격자 τ `{0.1, 1, 10, 100, 1000}` × 반감기 h `{30, 60, 120}`일(15조합) 선택 함수가 f1–f4 pooled BB MAE로 (τ, h)를 고르고 `results/backbone_tau.csv`(tau, half_life, mae_pooled)에 기록 `[v2 §6, A6]`
- [ ] B1용 행별 확장창 백본: 대상일 d의 값은 d−2일까지의 데이터로 적합한 프로파일(`backbone_asof(d)`), 학습·검증 행 모두 같은 방식 `[v2 §6, A32]`
- [ ] 불변성: 검증창 Y를 교란해도 fold 학습 프로파일 동일
- [ ] `uv run pytest -q` 통과

### US-007: LightGBM 강기준선 B1과 oracle-weather ablation
**Description:** 연구자로서 반드시 넘어야 할 강기준선(점예측 + 19분위수 + 일 단위 \(M_d\) 분위수 회귀 + 일피크 분류기)을 원한다 `[v2 §5.2, §15.4]`. 같은 스토리에서 외부 기상 제외의 근거가 될 oracle-weather 상한도 잰다 `[v2 §4.6, §4.7, 사용자결정]`.

**Acceptance Criteria:**
- [ ] `gmst/baselines.py::fit_b1(panel, train_idx, protocol, backbone, C)`와 `predict_b1(models, panel, d)` (`backbone` = 선택된 (τ, h))
- [ ] 학습행 = 목표가 유한한 (d, q)만: f1 6,240행, 최종(≤08-30) 11,256행
- [ ] fold당 슬롯 회귀 20개(L2 1 + 분위수 19, `y_median` = τ=0.5 모델) + 일 단위 \(M_d\) 분위수 회귀 19개 + 이진분류 3개(C50/C75/C90), 분위수는 행마다 정렬해 단조
- [ ] `backbone` 특징은 행별 확장창 적합(대상일 d 행 = d−2일까지 적합, US-006 `backbone_asof`) `[A32]`
- [ ] 일 단위 출력 `[v2 §15.4, A14]`: `M_hat_median` = \(M_d\) 분위수 회귀 τ=0.5, `M_hat_mean` = 19개 분위수 평균, `peak_time_mode` = 슬롯 `y_median`의 argmax, `risk_C*` = 이진분류기 + Platt(FR-52). 일 모델의 학습 라벨은 관측 슬롯 기준 최대(FR-38). 학습 사건이 한 클래스뿐이라 분류기를 적합할 수 없는 임계값은 `risk = 1 − F̂_M(C)`(\(M_d\) 분위수 보간)로 대체하고 로그 1줄
- [ ] LightGBM 파라미터: 라이브러리 기본값 + `seed=0, deterministic=True, verbose=-1`, 코드에 `# ponytail: 기본 하이퍼파라미터, B1이 f1–f4에서 B0′를 못 이기면 튜닝` 주석
- [ ] A+ 특징에 당일 생산량·당일 기상 파생 열 없음(과거 관측 기상 `prev_*`만), B에는 당일 생산량 열 있음(교란 테스트)
- [ ] 무선행성: 원점 이후 값(Y[d:], X[d:])을 교란해도 d의 예측 동일
- [ ] OOF 결과가 `oof_slots.csv`·`oof_days.csv`에 model `B1`, variant `main`으로 기록
- [ ] **oracle-weather ablation** (variant `AWstar`, 프로토콜 `A+W*`): 대상일의 데이터 자체 `기온·습도·풍속·강수량_증분` 시간값(4슬롯 반복)을 알려진 값처럼 특징 `ob_temp, ob_hum, ob_wind, ob_rain`으로 더해 같은 f1–f4 OOF를 기록하고, `bootstrap.csv`에 `AWstar vs main`(B1) ΔMAE·Δ평균Brier CI를 쓴다. 결과는 보고서 제2·3장에서 "기상의 최대 기여에 대한 휴리스틱 상한(f1–f4처럼 표본이 작으면 특징을 더하는 것 자체가 손해일 수 있어 엄밀한 상한은 아님) = 외부 기상을 쓰지 않은 근거"로 제시한다(재검토 R9). **제출 모델에는 절대 들어가지 않는다**: `gate_pass`/`select_variant` 후보·최종 적합·`test_predictions.csv`에 쓰이지 않음을 테스트(`test_run_all.py`에서 `ob_*` 열이 테스트 추론 특징에 없음) `[v2 §4.6, §4.7, A17, 판정 Q3]`
- [ ] `uv run pytest -q` 통과

### US-008: 확률보정·판별력·부트스트랩·성공 기준
**Description:** 연구자로서 위험확률을 누수 없이 보정하고, 판별력을 점검하고, 신뢰구간으로 B4가 벤치마크 B1을 이기는 성공 기준을 만족하는지 판정하고 싶다(만족하지 못하면 US-021의 개선 루프로 넘어간다).

**Acceptance Criteria:**
- [ ] `gmst/evaluate.py`: `platt_fit/platt_apply`(입력을 [1/(2N), 1−1/(2N)], N=2,000으로 클리핑 후 logit), `pav_fit/pav_apply`(isotonic), `calibrate_oof(days_df, method)`(leave-one-fold-out), `p_star`, `block_bootstrap`, `dm_test`, `climatology`, `risk_check`, `gate_pass`
- [ ] 합성 데이터에서 Platt가 참 (a, b)를 ±0.1 이내 복원, PAV 출력 단조 + 알려진 예제와 일치
- [ ] leave-one-fold-out: fold j 보정함수는 fold j 라벨을 뒤집어도 불변
- [ ] 기후값: 무조건부(학습 기저율)와 **가동여부×요일유형 조건부**(학습 사용가능일 기준, 클래스 비면 가동여부만; 공휴일은 넣지 않음) 둘 다; `bss_cond = 1 − Brier/Brier_climcond` `[v2 §11.3, 판정 Q8]`
- [ ] `risk_check`: 모델×임계값별로 f1–f4 합산 OOF에서 보정 Brier, `Brier_climcond`, AUC, `bss_cond`, 사건 수를 `results/risk_check.json`에 쓰고, 통과 = 보정 Brier < `Brier_climcond` **그리고** AUC ≥ 0.70 **그리고** `bss_cond` > 0 (사건이 한 클래스뿐인 임계값은 AUC 대신 BSS만). 미통과면 `"discrimination_missing": true` `[v2 §11.3, A26]`
- [ ] `block_bootstrap`(B=2,000, seed 0, 일 블록, pooled f1–f4)이 합성 차이에서 참값을 포함하는 CI 반환, 같은 seed면 결과 동일
- [ ] `gate_pass(variant_metrics, b1_metrics)`: ΔMAE·Δ평균Brier(후보 변형−B1, 세 임계값 평균 보정 Brier) **두** CI 상한이 모두 < 0이면 `True`(성공 기준 충족) — 4개 경우 단위테스트(둘 다 충족/MAE만/Brier만/둘 다 미충족). ΔCRPS는 계산·보고만 하고 판정에 쓰지 않음(분위수 열은 CRPS/pinball로 평가). B1은 후보가 아니므로 이 함수는 `"B4"`/`"B1"` 중 고르지 않고 bool만 반환한다 — 어떤 변형을 제출할지는 US-021의 `select_variant`가 정한다 `[v2 §15.6, 판정 Q5]`
- [ ] `uv run pytest -q` 통과

### US-009: 조건부 HMM 코어 (마스킹 쌍 emission forward, GPU)
**Description:** 연구자로서 v2 §8–§10의 조건부 전이·상태 내 AR(1)·마스킹 forward·2일 블록 학습을 GPU에서 돌리고 싶다.

**Acceptance Criteria:**
- [ ] `gmst/hmm.py`: `DEVICE`, `z_features(panel, protocol)`, `CondHMM(torch.nn.Module)`(인자 `K, dz, cond=True, ar=True`), `forward_logp`, `backward`(smoothing), `filter_last`, `stationary`, `train_hmm(...) -> (model, best_epoch, history)`
- [ ] 파라미터 수(K=3): A+ 전이 90 + emission 15 = 105, A 54 + 15, B 102 + 15; K=2 A+ 30 + 10, K=4 A+ 180 + 20
- [ ] 소형 예제(K=2, 3스텝)에서 `forward_logp`가 모든 상태경로 전수합(쌍 emission)과 1e-6 이내 일치, 마스킹 스텝의 Y 값을 바꿔도 logp 불변
- [ ] 임의 파라미터에서 모든 시점 μ₁<μ₂<μ₃, σ ≥ 1 (`test_sigma_floor`), `stationary(A) @ A == stationary(A)` (1e-6)
- [ ] f1 학습 블록 수 65 (각 블록 = warm-up 전일 96 + 목표일 96)
- [ ] 손실 = 목표일 관측 슬롯당 평균 NLL. 점유 하한 항은 기본 꺼짐(`occ_floor=False`), 켜면 \(\lambda_B=10\), κ=0.02 (FR-68) `[v2 §14, A21]`
- [ ] `CUDA_VISIBLE_DEVICES=""`에서 CPU로, GPU가 있으면 `cuda`로 학습(로그에 장치 출력); 합성 2상태 자료에서 학습 NLL 감소
- [ ] 무선행성: fold 학습을 고정 epoch로 돌리면 검증창 Y 교란이 파라미터를 바꾸지 않음
- [ ] `uv run pytest -q` 통과

### US-010: MC 피크위험·해석식 ablation·사다리 B2/B3/B4·K·일 랜덤효과(진단)
**Description:** 연구자로서 조상샘플링 MC로 \(F_M\)·분위수·피크시각을 얻고, 사다리와 ablation으로 각 구성요소의 기여를 보이고 싶다.

**Acceptance Criteria:**
- [ ] `gmst/hmm.py::forecast(model, panel, d, N=2000, re=False) -> dict`(y_mean, y_median, q, paths (N×96), M_hat_median, M_hat_mean, peak_time_mode, risk_raw (3,)), `analytic_risk(model, panel, d, C)` (φ=0 전용)
- [ ] 마지막 관측 \(Y_{t_0}\)가 마스킹이면 \(e_0\sim N(0,\sigma^2_{S_0,op}/(1-\phi^2_{S_0}))\) (테스트: 해당 경우 e₀ 표본 분산이 이론값의 ±10% 이내, N=20,000)
- [ ] φ=0 모델에서 해석식 위험과 MC 위험(N=20,000)의 차이가 3·√(p(1−p)/N) 이내
- [ ] 위험은 C에 대해 단조감소, 분위수 단조, `peak_time_mode` ∈ [0, 95](동률은 이른 슬롯), 날짜별 seed로 같은 입력이면 같은 출력
- [ ] 사다리: B2(전이 특징 없음, θ 학습 안 함, φ≡0, 해석식 위험), B3(조건부, φ≡0, 해석식), B4(조건부 + AR(1) + MC); f1–f4 OOF가 model `B2/B3/B4`로 기록, 각 5,376 슬롯행·56 일행
- [ ] variant `K2`, `K4`(B4), `re`(B4, 일 랜덤효과, FR-75, **진단 전용 ablation** — 개선 루프의 정식 단계 아님) 기록. §11.5 대응 순서: B3·B4·B4-H 판별력 점검(`risk_check`) → 미달이면 `re`로 원인을 진단(슬롯 잡음의 날짜 간 누적 여부) → 개선 루프는 US-021의 B4-IO(전일 요약으로 같은 역할을 결정론적으로 흡수)·B4-KAN으로 계속 진행 → **B1로 되돌아가지 않는다**(§5.2, §15.6, 사용자 결정) `[v2 §11.5, A30]`
- [ ] 최종 epoch = f1–f4 best epoch 중앙값(정수 반올림)이 `selection.json`에 기록될 값으로 반환
- [ ] `uv run pytest -q` 통과

### US-011: P0-4 누수 방지 테스트 (원점 이후 정보 교란)
**Description:** 검증자로서 어떤 모델의 어떤 입력에도 발행시각 이후 정보가 들어가지 않음을 테스트로 증명하고 싶다 `[수정-8, 검증 P0-4]`.

**Acceptance Criteria:**
- [ ] `tests/test_leakage.py`가 B0, B0′, BB, B1, B2, B3, B4, SCENARIO(B) 각각에 대해: 원점 d(2021-08-18, f4 첫 검증일)에서 Y[d:], 기상 X[d:], 생산량 X[d+1:]을 난수로 바꾸고, 프로토콜이 허용하지 않는 당일 정보(A+의 생산량 X[d], 기상 X[d])도 바꾼 패널로 예측 → 원본과 `np.array_equal`(MC는 같은 seed)
- [ ] 허용 정보 확인: B는 X[d] 생산량 교란 시 예측이 바뀜, A+W*(oracle-weather ablation)는 X[d] 기상 교란 시 B1 예측이 바뀜
- [ ] 적합 불변: 백본 프로파일·임계값 C·B1 모델 예측·(고정 epoch) HMM 파라미터가 fold 학습 종료 이후 값 교란에 불변
- [ ] 보정 불변: fold j의 보정 확률이 fold j 라벨 교란에 불변
- [ ] 봉인: 기본 로더에서 테스트창 전체 NaN
- [ ] `uv run pytest -q` 통과

### US-012: P1-3 상태 순서 안정성 테스트 (3 seed)
**Description:** 검증자로서 seed를 바꿔도 상태 해석(저/정상/고부하 편차)이 같음을 보이고 싶다 `[수정-8, v2 §9, A25]`.

**Acceptance Criteria:**
- [ ] `gmst/hmm.py::state_stability(panel, fold="f1", seeds=(0,1,2)) -> dict`
- [ ] 각 seed에서 op∈{0,1}마다 δ₁<δ₂<δ₃ (상태 순서 동일)
- [ ] seed 쌍마다 허용치 `[v2 §15.7, A31]`: 상태별 학습구간 평균 점유율 차 < 0.05, \(\delta_{k,op}\) 차 < 5, §13.4 결론 동일(= `hmm_transitions`에서 \(P(S_{t+1}=K)\)가 가장 큰 (가동, 요일유형, 시) 셀이 같음)
- [ ] 보고용으로 f1 검증 사용가능점의 smoothed 최빈상태 seed 쌍별 일치율도 계산; run_all이 결과를 `results/state_stability.csv`(seed_a, seed_b, occ_maxdiff, delta_maxdiff, same_order, same_conclusion, agreement)로 저장. 허용치를 넘으면 결과표에 "seed 의존"으로 표기하고, 원인이 점유율 붕괴(< 0.02)면 FR-68의 점유 하한을 켠다
- [ ] `tests/test_state_stability.py` 통과, `uv run pytest -q` 통과

### US-013: (삭제: 사용자 결정 — KMA 제외)
옛 내용: KMA 세션·지점목록·ASOS 파서·공장 지점 식별. 번호는 참조 유지를 위해 비워 둔다.

### US-014: (삭제: 사용자 결정 — KMA 제외)
옛 내용: KMA 최종 다운로드(ASOS·단기예보)와 A+W 변형. oracle-weather ablation은 US-007로 옮겼다(데이터 자체 기상, 외부자료 없음).

### US-015: 한전 산업용(을) 2021 요금 계산기
**Description:** 현장 담당자로서 예측 전력의 ₩ 비용과 요금적용전력을 한전 2021 요금 규칙으로 계산하고 싶다 `[v2 §12.1, 요금표]`.

**Acceptance Criteria:**
- [ ] `gmst/scenario.py`: `TARIFF`(선택Ⅰ/Ⅱ/Ⅲ 기본요금 7,220/8,320/9,810, 계절×시간대 단가 요금표 그대로), `season(month)`, `band(ts, holiday, for_energy)`, `energy_won(y, ts, holidays, option="II")`, `billing_demand(y, ts, holidays, month)`, `ratchet_floor(panel, month, exclude_dates)`, `expected_p_app(paths_mp, floor)`, `delta_cost(...)` (FR-85–86)
- [ ] 100 kW 한 슬롯 에너지요금(선택Ⅱ): 07-07(수) 10:00 → 4,777.5₩; 07-10(토) 11:00 → 2,725.0₩(에너지 중간부하); 07-11(일) 11:00 → 1,402.5₩(일요일 경부하, `daytype=="sun"`만으로 판정, `HOLIDAYS_2021`에는 없음); 08-16 11:00 → 1,402.5₩(공휴일 경부하, `tariff_0816=True`); 01-05(화) 18:00 → 4,167.5₩(겨울 최대); 02-13(토·공휴일) 11:00 → 1,577.5₩(겨울 경부하); 09-08(수) 10:00 → 2,732.5₩(봄가을 최대) `[재검토 R1]`
- [ ] 시간대: 07-07 09:00 중간, 08:45 경부하, 17:00 중간, 23:00 경부하; 01-05 13:00 중간, 21:00 중간, 22:30 최대
- [ ] 래칫: 합성 월 자료에서 경부하 200 kW·최대부하 150 kW면 요금적용전력 150, 공휴일 피크 무시, 07-10(토) 11:00 수요는 래칫에 포함
- [ ] `test_floor_222`: ≤08-31 실데이터(마스킹 점·일요일·공휴일 제외)에서 `ratchet_floor(month=7, exclude_dates=f2 시나리오 대상일)`와 `ratchet_floor(month=8, exclude_dates=f2–f4 시나리오 대상일)`가 모두 **222**이고 그 위치가 2021-07-19 11:15(최대부하)임. 이력은 fold 창이 아니라 대상월까지의 전체 관측 `[v2 §12.1, 판정]`
- [ ] ΔP_app 10 kW → 기본요금 변화 83,200₩
- [ ] `test_no_data_tariff` (DC7)
- [ ] `uv run pytest -q` 통과

### US-016: SCENARIO 트랙과 생산일정 이동 시나리오
**Description:** 현장 담당자로서 생산계획을 바꿨을 때 모델상 피크위험·\(E[M]\)·₩가 어떻게 변하는지 보고 싶다 `[v2 §12, 결정-프로토콜]`.

**Acceptance Criteria:**
- [ ] SCENARIO 모델 = B4 구조 + 프로토콜 B(z 16차원), fold f2–f4 각각 **train 역할 중 ≥ 2021-07-01인 날만**으로 학습(assert). f1은 07-01 이후 학습일이 5일뿐이라 제외 `[v2 §12, §12.1, 판정 Q2]`
- [ ] 변환 4종 × 이동률 {0.1, 0.2, 0.3} + `baseline` = 13 시나리오(FR-88); 모든 변환이 일 생산량 합 보존(1e-9), 음수 없음, `baseline`은 무변경
- [ ] 같은 날의 시나리오들은 공통 난수(common random numbers) 사용: `baseline`을 두 번 돌리면 결과 동일
- [ ] `results/scenarios.csv`(일 단위) 열: `fold, date, scenario, rate, E_M, M_median, risk_C50, risk_C75, risk_C90, risk_ok, peak_time_mode, C_change, d_energy_won` — `C_change` = Σ|ΔP|/(2ΣP), `risk_ok` = \(R^{(a)}(C_{90}) \le R^{(0)}(C_{90})\) `[v2 §12, A28]`
- [ ] `results/scenarios_month.csv`(청구월 단위, FR-86) 열: `month, scenario, rate, n_days, P_floor, E_P_app_base, E_P_app_scn, d_demand_won, d_energy_won, d_kwh, d_total_won, d_total_pct, C_change, eta_star, risk_ok_all` — `d_kwh` = 그 달 예측 전력량(kWh)의 변화(A) − (기준), 기후환경요금·연료비조정요금 같은 kWh당 균일 가산은 `d_kwh ≈ 0`일 때만 상쇄됨을 README·보고서에 명시(재검토 R8). `eta_star` = −`d_total_won`/`C_change`(손익분기, `C_change` = 0이면 NaN), `risk_ok_all` = 그 달 모든 날 `risk_ok` `[v2 §12, §12.1, A28, A29, 재검토 R8]`
- [ ] **래칫 반영** `[v2 §12.1, 판정]`: 7·8월 모든 행의 `P_floor` = 222(07-19 11:15, 시나리오 대상 창 밖 기록값). 따라서 `d_demand_won`은 경로의 월 최대가 222를 넘는 꼬리를 줄일 때만 0이 아니며 사실상 0이다. 테스트: `baseline` 행의 `d_demand_won == 0`, 7·8월 행의 `P_floor == 222`. 노트북 요약표는 월별 `d_demand_won`과 `d_energy_won`을 나란히 보여 준다. README·노트북·보고서 제4장은 "7–8월 기본요금 절감 ≈ 0, ₩ 절감의 주 수단은 TOU 전력량요금 이동"을 먼저 제시하고, 기본요금 절감은 "12·1·2·7·8·9월 전체에 같은 정책을 적용할 때"의 조건부 결과로만 쓴다
- [ ] SCENARIO OOF 지표가 metrics.csv에 variant `B`로 기록되고 README·노트북에서 "계획이 있을 때의 상한"으로 표기
- [ ] `uv run pytest -q` 통과

### US-017: 오류분석·FN/FP 조건·증강 누수 격차
**Description:** 연구자로서 보고서 제3장(잘 되는/실패하는 조건, 미탐지·오경보)과 제1장(증강 누수 크기) 근거 표를 원한다 `[v2 §13, §15.3, 결정-대회 3]`.

**Acceptance Criteria:**
- [ ] `gmst/analysis.py::error_by_condition` → `results/error_by_condition.csv`(`model, condition, bin, n, mae, rmse`), 조건: 생산량 구간(0 / 양수 4분위), 가동/비가동, 요일유형, 시(0–23), 기온 4분위, 휴무 전후(전일 비가동 / 익일 비가동 / 그 외), B4 상태(filtered·smoothed), 상태전환 vs 안정 구간. 빈 경계는 ≤ 08-31 학습자료로 계산
- [ ] `fn_fp` → `results/fn_fp_days.csv`(`model, C, p_star_kind, date, kind(TP/FN/FP/TN), daytype, op_prev, op_next, prev_day_max, temp_mean, prod_sum, risk_cal, M_true`)와 `results/fn_fp_summary.csv`(FN·FP 집합의 조건 분포 vs 전체 사용가능일). 대표 임계값 q90의 F1·FN·FP는 **f1–f4 합산 OOF(13사건)**로 보고하고, fold별 표에는 사건 수를 병기하며 사건 0칸(f4 q90)은 "정의 안 됨" `[v2 §4.5, §16.4]`
- [ ] `leakage_gap` → `results/leakage_gap.csv`: B1 특징·복사 포함 패널(봉인, ≤ 08-31)에서 일 단위 랜덤 5-fold MAE vs LOEO(fold = event_id) MAE
- [ ] `transitions` → `results/hmm_transitions.csv`: f4 B4 모델의 \(P(S_{t+1}=K\mid S_t=i,\mathbf z)\)를 (가동, 요일유형, 시)별로
- [ ] 합성 입력 단위테스트(빈 분할, TP/FN/FP/TN 분류)
- [ ] `uv run pytest -q` 통과

### US-018: 원커맨드 러너와 제출물
**Description:** 평가위원으로서 한 명령으로 전처리부터 결과까지 재현하고 제출 파일을 받고 싶다 `[결정-대회 6, v2 §20]`.

**Acceptance Criteria:**
- [ ] `uv run python -m gmst.run_all`이 `results/`를 지운 상태에서 종료코드 0, 단계별 로그(단계명·소요초·DEVICE)
- [ ] 단계 순서가 FR-94대로이며 `selection.json`이 쓰인 뒤에만 `load_panel(unseal=True)` 호출
- [ ] `results/test_predictions.csv`: 1,344행, 열 33개 순서 정확히 FR-96, datetime 2021.09.01 00:00:00 ~ 2021.09.14 23:45:00, `track` 전부 `MAIN`, `model` 전부 `selection.json`의 제출모델, `C50/C75/C90` = 186.5 / 198 / 210.7, 분위수 열 행마다 단조, 0 ≤ risk ≤ 1, `risk_C50 ≥ risk_C75 ≥ risk_C90`, `is_missing` 열 없음
- [ ] `results/eval_mask.csv`: 1,344행 `datetime, is_missing`, `is_missing` true는 정확히 2행(09-08 12:00, 12:15) `[v2 §4.5, A16]`
- [ ] `results/test_metrics.csv`(fold=`test`), `results/selection.json`(FR-57/FR-110: `submitted`, `gate_met`, `tried`, `remaining_gap_to_b1` 포함), `results/bootstrap.csv`, `results/risk_check.json` 존재. B1은 제출되지 않으므로 조건부 모듈 파일(`test_predictions_B4_module.csv`)은 만들지 않는다 `[v2 §4.5, §15.6, A16, 사용자결정]`
- [ ] `--smoke`(f4만, 최소 epoch·MC 경로)로 같은 산출물 파일이 생성되고, `tests/test_run_all.py`가 smoke 실행 후 위 스키마 검사를 통과
- [ ] `requirements.txt`: 첫 줄 `--extra-index-url https://download.pytorch.org/whl/cu126`, `torch==2.14.0+cu126`·`lightgbm`·`polars`·`numpy`·`matplotlib` 버전 고정 포함, `pytest`·`requests` 미포함
- [ ] `README.md`(현재 빈 파일)를 FR-101 목차로 작성, 소속·로고 없음
- [ ] `uv run python -m gmst.run_all --package` → `dist/kamp_power_src.zip`에 `.env`, `.venv/`, `document/`, `tasks/` 없음(테스트)
- [ ] `uv run pytest -q` 통과

### US-019: 결과 노트북 (표시 전용) 정리
**Description:** 보고서 작성자로서 스크립트 산출물만 읽어 보여주는 노트북을 원한다. 노트북이 데이터를 만들면 재현성이 깨지기 때문이다 `[결정-코드]`.

**Acceptance Criteria:**
- [ ] `notebooks/03_results.ipynb` 신설: `results/*`와 두 데이터 CSV만 읽음(학습·쓰기 없음), 보고서 6개 장 순서의 절(데이터 주의점 표, 날짜표 타임라인, 모델 비교표, 부트스트랩 CI, 신뢰도 곡선 보정 전/후, 조건별 오차·FN/FP, 시나리오 ₩, 테스트 예측 그림)
- [ ] `notebooks/01_preprocess.ipynb`: 더는 CSV를 쓰지 않고 `../5. 자원 최적화 AI 데이터셋/okm_15min_2021.csv`(스크립트 산출물)를 읽어 점검·그림만; 미사용 `import numpy as np` 제거
- [ ] `notebooks/02_eda.ipynb`: 분할 CSV를 쓰지 않고 스크립트 산출물을 읽음; 비가동일 62 출력; "기획서 검증 체크"의 설계통계(R², 모드 수, lag-1, 초과율)는 ≤ 2021-08-31만 사용; §8 셀은 10단위 빈 광의 3봉(07-01~08-31 마스킹: 20–30대 2,082 / 100–110대 569 / 170–180대 326점) `[재계산]`을 출력하고 "100빈+5빈 이동평균의 10개는 잔물결" 주석 `[v2 §4.4]`; §11 두 행의 행 라벨 주의문, H=96 초과율이 자정 정렬이 아니고 레짐을 섞는다는 주의문 `[v2 §4.5]`
- [ ] 세 노트북이 `cd notebooks && uv run --with nbconvert jupyter nbconvert --to notebook --execute --inplace <nb>`로 오류 없이 실행
- [ ] `uv run pytest -q` 통과

### US-020: ponytail 부채 원장
**Description:** 유지보수자로서 의도적 지름길을 한곳에서 보고 싶다 `[결정-코드, 수정-4]`.

**Acceptance Criteria:**
- [ ] `/home/user/manufacture_ai/PONYTAIL-DEBT.md`: `grep -rn "# ponytail:" gmst tests` 결과 전부를 모듈별 표(`file:line | 지름길 | 한계(ceiling) | 업그레이드 조건(trigger)`)로 수록, 항목 수 = grep 줄 수
- [ ] 각 항목에 대응 FR 또는 v2/수정 번호
- [ ] `ponytail:ponytail-debt` 스킬로 생성
- [ ] `uv run pytest -q` 통과

### US-021: 사전등록 개선 루프 (B4-H · B4-IO · B4-KAN)
**Description:** 연구자로서 B4가 US-008의 성공 기준(B1 대비 CI 게이트)을 못 넘길 때, 세 변형을 사전등록된 순서로만 시도하고 그중 최선을 고르고 싶다(사용자 결정: `.sdd/progress.md` Decision (user) + Ruling + "Ruling (B4-IO spec)"). B1은 이 루프의 어느 단계에서도 제출 후보로 돌아오지 않는다 `[v2 §9.6, §15.6]`.

**Acceptance Criteria:**
- [ ] `gmst/hmm.py::CondHMM` 생성자에 `emission_source: "backbone"|"b1" = "backbone"`(B4-H) 추가. `"b1"`이면 μ 계산에 쓰는 \(m_t\)가 `gmst/baselines.py::predict_b1`의 그 fold train 적합 τ=0.5 슬롯 예측(A32 행별 확장창 특징 그대로)이다. 단위테스트: `emission_source="b1"`일 때 모델이 받는 \(m_t\) 배열이 `predict_b1` 출력과 `np.array_equal` `[v2 §9.6 ①, A34]`
- [ ] variant `H`(B4-H, model `B4`)로 f1–f4 OOF 기록. `tests/test_leakage.py`에 `H` variant 추가: 원점 이후 교란 시 B1 부분입력을 포함해 예측 불변(US-011과 같은 절차)
- [ ] `gmst/hmm.py::u_features(panel, protocol, m_t) -> np.ndarray`(T×13, 열 순서 `1, op, sin1, cos1, sin2, cos2, op_sin1, op_cos1, sat, sun, hol, ybar_prev, ymax_prev`)와 `v_features(panel) -> np.ndarray`(T×3, `1, sin1, cos1`) `[v2 §9.6 ②, A35]`
- [ ] `ybar_prev[d]`/`ymax_prev[d]` = 전일(d−1) 관측 슬롯에서 \((Y-m)\)의 평균/최댓값, 관측 슬롯이 0개면 0(감사 로그에 `ybar_prev_missing_days` 카운트만 남기고 u_features 자체는 13열 고정)
- [ ] `CondHMM` 생성자에 `decoder: "const"|"io" = "const"`(B4-IO) 추가. `"io"`이면 파라미터가 `a`(K×13), `b`(K×13), `c`(K×3)이고 \(\delta_{t,1}=\mathbf a_1^\top\mathbf u_t\), \(\delta_{t,k}=\delta_{t,k-1}+\mathrm{softplus}(\mathbf a_k^\top\mathbf u_t)\)(k=2..K), \(\mu_{t,k}=m_t+\delta_{t,k}\), \(\sigma_{t,k}=1+\mathrm{softplus}(\mathbf b_k^\top\mathbf u_t)\), \(\phi_{t,k}=\mathrm{sigmoid}(\mathbf c_k^\top\mathbf v_t)\)로 매 시각 재계산된다. 쌍 emission·마스킹·forward·MC 식은 §9–§11과 동일(상수 \(\delta_{k,op},\sigma_{k,op},\phi_k\) 자리에 시변값을 대입)
- [ ] K=3에서 `decoder="io"`의 학습 파라미터 수 = `a`(39) + `b`(39) + `c`(9) = **87**(단위테스트로 고정 확인). `decoder="const"`의 기존 파라미터 수(105, US-009)와 별개로 집계
- [ ] `a`, `b`의 조화항 열(`sin1,cos1,sin2,cos2,op_sin1,op_cos1`)과 전일요약 열(`ybar_prev,ymax_prev`)에만 L2 벌점 `lambda_io * sum(coef**2)`을 손실에 더하고, 절편·`op`·`sat`·`sun`·`hol` 열 계수는 벌점에서 제외. `lambda_io`는 격자(예 `{0, 0.01, 0.1, 1}`)에서 **내부검증**(R10, fold 학습구간 마지막 7일) NLL로 선택하고 `results/backbone_tau.csv`와 같은 형식으로 `results/io_lambda.csv`에 기록
- [ ] variant `IO`(B4-IO, model `B4`)로 f1–f4 OOF 기록. `H`가 채택되지 않았으면 `decoder="io"`도 `emission_source="backbone"`으로, `H`가 채택됐으면 `emission_source="b1"` 위에서 `decoder="io"`를 적용(순차 탐색)
- [ ] `gmst/hmm.py::z_features`에 `kan: bool = False` 인자 추가(B4-KAN, §9.6 ③). `kan=True`면 전이 특징의 sin/cos 조화항 6차원 + 가동×조화 교호 4차원(총 10차원)을 제거하고, 대신 `g(q, op) = sum_m c[op, m] * B_m(q)`(주기 3차 B-스플라인, 매듭 수 `M ∈ {8,12,16}`, 균등 간격, 순환 경계)를 전이 logit에 더한다. 0/1 플래그(토/일/공휴일/가동) 4차원은 그대로 선형 유지
- [ ] `M`과 매끄러움 벌점 `lambda_spl`은 R10 내부검증 pooled NLL로 함께 선택(격자 3×4, `results/kan_grid.csv`에 (M, lambda_spl, nll) 기록). 벌점 = `lambda_spl * sum((c[op,m+1] - c[op,m])**2)`(m을 M에서 1로 순환)
- [ ] variant `KAN`(B4-KAN, model `B4`)로 f1–f4 OOF 기록. 순서상 `H`·`IO` 채택 결과 위에서 적용(순차 탐색)
- [ ] `select_variant(candidates: dict[str, metrics]) -> dict`: 각 후보(`main`(B4 기본), `H`, `IO`, `KAN` 중 실제로 시도된 것들)에 대해 US-008의 `gate_pass`를 계산하고, 하나라도 `True`면 **그중 평균 OOF MAE가 가장 낮은 것**을 반환, 전부 `False`면(=2026-10-03 도달 시) 시도된 후보 전체 중 평균 OOF MAE가 가장 낮은 것(동률이면 C50/C75/C90 평균 Brier)을 반환. 반환값에 `gate_met: bool`, `tried: [...]`, `remaining_gap_to_b1: {...}` 포함 `[v2 §15.6]`
- [ ] `results/selection.json`에 `select_variant`의 시도 순서·게이트 결과·최종 선택·B1 대비 잔여 격차를 기록(FR-57과 연결)
- [ ] `uv run pytest -q` 통과

---

## 4. 기능 요구사항 (Functional Requirements)

### 4.1 환경·공통
- FR-1: 직접 의존성은 `uv add lightgbm`(런타임)과 `uv add --dev pytest`(dev)로만 추가한다. `requests`는 추가하지 않는다(외부 수집 없음). `scipy`·`scikit-learn`·`pandas`·R을 직접 의존성으로 두거나 `import`하지 않는다(lightgbm이 scipy를 전이 설치하더라도 코드에서 쓰지 않음). Platt·PAV·CRPS·AUC·순환 차분·선형해는 numpy/torch로 쓴다 `[결정-코드, v2 §18, 판정 Q9, 사용자결정]`.
- FR-2: `pyproject.toml`에 pytest 설정(`pythonpath=["."]`, `testpaths=["tests"]`). torch 인덱스 설정 유지 `[환경]`.
- FR-3: 코드는 `gmst/` 패키지. `gmst/__init__.py`는 `ROOT`, `DATA`, `RESULTS`만 정의. 각 모듈은 `uv run python -m gmst.<module>`로 단독 실행 가능. 파이프라인은 네트워크·자격증명을 쓰지 않는다 `[결정-코드, A18, 사용자결정]`.
- FR-4: `gmst/hmm.py`에 `DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")`. 모든 torch 텐서·모델은 `DEVICE`에 둔다. 단일 GPU만 사용 `[환경]`.
- FR-5: 난수는 seed 0 기본: numpy `default_rng(seed)`, `torch.manual_seed(seed)`, LightGBM `seed=0, deterministic=True` `[결정-대회 6]`.
- FR-6: 의도적 지름길은 `# ponytail: <ceiling>, <upgrade trigger>` 주석을 단다 `[결정-코드, 수정-4]`.
- FR-7: 코드·README·결과물에 소속·로고 등 식별정보를 쓰지 않는다 `[결정-대회]`.

### 4.2 전처리 (`gmst/preprocess.py`)
- FR-8: 원본 `DATA/okm_augumented_2021.csv`(BOM 포함, 6,168행 × 18열)를 읽기만 한다 `[v2 §4]`.
- FR-9: 시(hour)는 날짜 내 행 위치(0–23)로 정한다(255일은 `시간`과 일치, 07-13·07-15의 48행은 불일치 → 감사 기록) `[v2 §4.2]`.
- FR-10: wide→long은 구간 시작 규약(`15분`→HH:00, `30분`→HH:15, `45분`→HH:30, `60분`→HH:45)으로 24,672행을 만든다. 시간 단위 공변량은 4슬롯에 반복한다(나누지 않음) `[v2 §4, §8]`.
- FR-11: `전력 == 0`인 74점은 null로 바꾸고 `is_missing = True` `[v2 §4.2, 결정-데이터]`.
- FR-12: 풍속 결측 3시간은 시간 계열에서 시간축 선형보간 후 반복한다 `[v2 §4.2]`.
- FR-13: `강수량_증분`: 시간 누적값을 자정을 가로지르는 연속 계열로 보고 `inc_t = c_t − c_{t−1}`(c_{t−1}은 직전 비결측값), `inc_t < 0`이면 `inc_t = c_t`(리셋), 결측 시간은 `inc = 0`, 첫 시간은 `c_t` `[수정-3, v2 §4.2]`.
- FR-14: `공장인원`·`평균` 항등식을 계산해 감사 파일에 기록하고 `공장인원, 평균, day, d, m, 날짜, 시간` 원열을 출력에서 뺀다 `[v2 §4.1]`.
- FR-15: 출력 `DATA/okm_15min_2021.csv`(US-002 스키마)와 `RESULTS/data_audit.json`. `build_15min()`은 쓰기 없이 DataFrame 반환, `run()`만 쓴다 `[v2 §18]`.

### 4.3 날짜표 (`gmst/splits.py`)
- FR-16: `event_id`: 날짜별 96점 전력벡터(null도 값으로 취급)의 완전일치 그룹, 그룹 첫 날짜 순 `E000`… 번호(142개). `dup_size` = 그룹 크기 `[v2 §15.3]`.
- FR-17: 그룹 원본 선택: "완전 가동 모양" = 전력 > 40인 슬롯(가동 슬롯) ≥ 48. 원본 = 그룹의 가장 이른 날. 단 가장 이른 날이 완전 가동 모양인데 일 생산량 0이고 그룹 안에 생산량 > 0인 다른 날이 있으면, 그중 가장 이른 날이 원본. 완전일치 그룹(크기 > 1)의 원본이 아닌 날 = `is_copy=true`, `copy_kind="exact"`, `copy_of`=원본 날짜(115일; 원본이 바뀌는 그룹은 01-24 → 02-24 하나) `[v2 §4.2, §15.3, 판정 Q11, 복사판정, 결정-데이터]`.
- FR-18: `is_suspect` = {2021-07-13, 2021-07-15} `[결정-데이터]`.
- FR-19: 편집된 복사 = 완전일치 그룹의 원본이 아닌 날(완전 복사 제외) 중, 다른 어떤 날과 "두 날 값이 정확히 같고 그 값이 > 40인 슬롯" 수가 20 이상인 날(7일) → `is_copy=true`, `copy_kind="edited"`, `copy_of` = 그 수가 가장 큰 상대 날짜(동률이면 마스킹되지 않은 날, 그다음 가장 이른 날). 그룹을 전이적으로 잇지 않는다(편집복사는 자기 `event_id`를 유지, purge는 완전일치 event 기준). 임계값 20은 07–08월 비복사 쌍의 자연 최대 9의 2배 이상. 판정 규칙은 모델 학습 전에 고정된 데이터 정합성 규칙이며 결과가 07-01 이후 날짜를 새로 가리지 않음을 테스트한다 `[v2 §4.2, §15.3, 복사판정]`.
- FR-20: `is_operating` = (일 생산량 합 > 0) or `is_suspect` (가동 195 / 비가동 62) `[v2 §4.2]`.
- FR-21: `is_holiday` = `HOLIDAYS_2021` = 01-01, 02-11, 02-12, 02-13, 03-01, 05-05, 05-19, 08-16 (하드코딩 + 출처 주석) `[결정-외부, v2 §4.7]`.
- FR-22: `daytype` ∈ {`wk`, `sat`, `sun`} `[v2 §6]`.
- FR-23: `n_missing` = 그 날 `is_missing` 점 수 `[v2 §15.3]`.
- FR-24: `anomaly` = `is_suspect` or 전력 0 포함일(5일, EDA 정의 유지) `[v2 §15.3]`.
- FR-25: fold 역할: 각 창(f1 07-07~07-20, f2 07-21~08-03, f3 08-04~08-17, f4 08-18~08-31, test 09-01~09-14)에서 창 = `val`(test 열은 `test`), 창 시작 전날 = `gap`, 그 이전 = `train`, 단 `train` 날의 event_id가 창 안 event 집합에 있으면 `purged`, 창 이후 = 빈 문자열 `[v2 §15.2, 결정-프로토콜]`. gap일은 학습 목표에서 빠지지만 검증 첫날 예측의 입력으로는 쓴다(코드 주석 명시) `[v2 §15.2]`.
- FR-26: 출력 `DATA/okm_cv_splits_2021.csv`(US-003 스키마). LOEO fold = `event_id`(별도 열 없음) `[v2 §15.3]`.

### 4.4 로더 (`gmst/features.py`)
- FR-27: `load_panel`(US-004 시그니처)은 다음을 NaN으로 만든다: `is_missing` 점(항상), `is_copy` 날(완전·편집 복사)의 Y(`include_copies=False`), `is_suspect` 날의 Y와 생산량(`include_suspect=False`). 복사일의 공변량(기상·생산량)은 유지한다. 가려진 값은 목표·lag 참조·HMM warm-up 모두에서 NaN이다 `[v2 §4.2, 결정-데이터, 수정-5, 수정-9]`.
- FR-28: `unseal=False`면 `test` 역할 날짜의 Y와 모든 X가 NaN. 테스트 대상 평가는 `unseal=True`에서만 가능하고, `gmst/`에서 그 호출은 `run_all.py` 최종 단계뿐 `[v2 §15.2, 수정-10]`.
- FR-29: 프로토콜은 `PROTOCOLS`(US-004)로 대상일 외생정보를 제한한다. 프로토콜 A는 같은 코드에 `op=1`, `hol=0`을 넣어 실행한다(백본 클래스가 요일유형만으로 축약) `[v2 §4.6]`.
- FR-30: 달력 특징: `q`, `q % 4`, sin/cos(2πrq/96) r=1..3, 요일, `daytype`, 월, `season`(한전 월→계절: 6–8 여름, 3–5·9–10 봄가을, 11–2 겨울), `op`, `hol`. `전기요금(계절)`은 `season`과 동치임을 테스트로만 확인하고 특징으로 직접 쓰지 않는다 `[v2 §4.1, 요금표]`.
- FR-31: B1 슬롯 특징(열 이름 고정): `q, q_mod4, sin1, cos1, sin2, cos2, sin3, cos3, dow, daytype, month, season, hol, op`(A+), `y_lag1, y_lag2, y_lag7`(같은 슬롯), `prev_last4_mean`(전일 23:00–23:45), `prev_mean, prev_max, prev_min`, `recent_same_mean`(최근 7일 중 같은 요일유형·가동여부 사용가능일의 슬롯 평균), `op_lag1, op_lag7`, `backbone`(행별 확장창 적합, d−2까지, A32), `prev_temp, prev_hum, prev_wind`(전일 일평균), `prev_rain`(전일 강수 합 = 시간값 24개 합), `prev_prod`(전일 생산량 합). MAIN의 기상 특징은 이 과거 관측값(`prev_*`)뿐이다. 프로토콜 추가열: B `prod_h, prod_on`, A+W*(oracle-weather ablation 전용) `ob_temp, ob_rain, ob_wind, ob_hum`(대상일 데이터 자체의 `기온, 강수량_증분, 풍속, 습도` 시간값) `[v2 §15.4, §4.6, A32]`.
- FR-32: B1 일 단위 모델(\(M_d\) 분위수 회귀 19개 + 일피크 이진분류기 3개) 특징(행 = 일): `prev_mean, prev_max, prev_min, y_lag7_mean, y_lag7_max, recent_same_max, backbone_max, dow, daytype, month, hol, op, op_lag1` (+ 프로토콜 열의 일 집계) `[v2 §15.4, A14]`.
- FR-33: 대상일 d의 모든 입력은 Y[:d], X[:d], 달력(d), 프로토콜이 허용한 d의 외생정보만의 함수다. fold 적합(백본 프로파일, C, B1, HMM 파라미터)은 그 fold의 train 역할 날짜만 쓴다. 백본 (τ, h)·\(p^*\)·보정함수·HMM early stopping은 **그 fold의 실제 검증일(f1–f4 outer)을 선택 기준으로 쓰지 않는다** — 대신 **내부검증**(각 fold 학습구간의 마지막 7일)으로 고르고, 고른 값으로 그 fold의 전체 학습구간(마지막 7일 포함)을 다시 적합해 그 fold의 검증일 예측에 쓴다. 테스트창은 어떤 경우에도 쓰지 않는다 `[v2 §6 누수 방지, §14, §15, 수정-8, 수정-10, 재검토 R10]`.

### 4.5 지표·평가 (`gmst/evaluate.py`)
- FR-34: 모든 모델의 예측 함수는 dict를 반환한다: `y_mean`(96), `y_median`(96), `q`(19×96 또는 None), `paths`(N×96 또는 None), `M_hat_median`, `M_hat_mean`, `peak_time_mode`(슬롯 0–95), `risk_raw`(3) `[v2 §9, §11]`.
- FR-35: `TAUS = 0.05, 0.10, …, 0.95`(19개). 분위수 열 이름 `q05, q10, …, q95` `[A11]`.
- FR-36: MAE는 `y_median`, RMSE는 `y_mean`으로, 마스킹되지 않은 점만, n 기록 `[v2 §16.1, §15.5]`.
- FR-37: CRPS = 19분위 pinball 평균 × 2. 구간 적중률 50%=(q25,q75), 80%=(q10,q90), 90%=(q05,q95) `[v2 §16.3, A11]`.
- FR-38: 일 피크 대상일 = `n_missing ≤ 4`이고 로더가 목표를 가리지 않은 날(기본 설정에서 비복사·비편집복사·비suspect; ablation 스위치를 켜면 그 날도 포함). \(M^{true}_d\)와 예측 일최대는 **같은 관측 슬롯 집합**에서 계산: 경로가 있으면 경로별 관측슬롯 최대의 중앙값, B1은 일 단위 모델의 `M_hat_median`(학습 라벨 자체가 관측슬롯 기준 최대), 그 외(점예측 기준선)는 관측슬롯에서 `y_median`의 최대. 위험 사건 \(\mathbb 1[M_d>C]\)도 관측슬롯 기준. PeakMAE는 중앙값 사용, 피크시각 적중 = |argmax 실측 − `peak_time_mode`| ≤ 2 슬롯 `[v2 §16.2, A12, A24, 수정-6]`.
- FR-39: 임계값 C(fold별) = 그 fold `train` 역할 날짜 중 가동 & 일피크 대상일의 일최대에 `np.quantile(..., [0.5, 0.75, 0.9])`(기본 linear) `[v2 §4.5, A1, 수정-10]`.
- FR-40: Brier(임계값별 + 평균)를 raw·Platt·isotonic 확률로 계산. 기후값 Brier 두 가지: 무조건부(fold 학습 사용가능일 사건률), 조건부(같은 가동여부×요일유형 학습 사건률, 클래스가 비면 가동여부만; 공휴일은 클래스에 넣지 않음). `bss_cond = 1 − Brier/Brier_climcond` `[v2 §11.3, §16.3, 판정 Q8]`.
- FR-41: 판별력: 임계값별 AUC(Mann–Whitney, 동점 0.5)를 pooled f1–f4 대상일에서. `risk_check` 통과 = 보정 Brier < `Brier_climcond`, AUC ≥ 0.70, `bss_cond` > 0 (세 임계값 각각, q90은 합산으로만; 사건이 한 클래스뿐이면 BSS만). 미통과면 `discrimination_missing = true` `[v2 §11.3, A26]`.
- FR-42: 사건 분류지표: 보정 확률에 \(p^*\) ∈ {0.5, 내부검증 F1 최대값(FR-55)}을 적용해 F1/precision/recall/FN/FP를 임계값별로. 대표 임계값 q90의 F1·FN·FP는 f1–f4 **합산 OOF**로 보고하고, fold별 값은 사건 수와 함께 적되 사건 0칸(f4 q90)은 NaN("정의 안 됨") `[v2 §4.5, §16.4, A24]`.
- FR-43: `rolling_origin(model, panel, folds=("f1","f2","f3","f4"), variant)`: fold마다 train 역할로 한 번 적합, 검증일을 날짜순으로 발행(입력 이력은 매일 갱신, 파라미터 고정). 출력 `results/oof_slots.csv`(`fold, variant, model, date, datetime, y_true, is_missing, y_mean, y_median, q05…q95, state_filt, state_smooth`)와 `results/oof_days.csv`(`fold, variant, model, date, usable_peak, n_obs, M_true, M_hat_median, M_hat_mean, peak_slot_true, peak_time_mode, C50, C75, C90, risk_raw_C50/75/90, risk_platt_C50/75/90, risk_iso_C50/75/90, event_C50/75/90`) `[v2 §15.2]`.
- FR-44: `results/metrics.csv`는 긴 형식 `model, variant, fold, stratum, metric, value, n`(fold ∈ f1–f4, `pooled`, `test`; stratum ∈ `all`, `op`, `nonop` — 08-16 공휴일 부분가동 같은 진짜 드문 날은 유지하고 가동·비가동 층으로 따로 본다 `[복사판정]`). metric 이름: `mae, rmse, crps, cov50, cov80, cov90, peak_mae, peak_hit2, brier_{raw|platt|iso}_C{50|75|90}, brier_mean_{raw|platt|iso}, brier_clim_C*, brier_climcond_C*, bss_cond_C*, auc_C*, f1_C*_{p05|pstar}, precision_*, recall_*, fn_*, fp_*, n_events_C*, nll`(`n_events_C*` = 사건 수, 모든 Brier·F1·AUC 표에 병기; `nll`은 HMM만) `[v2 §15.5, §16]`.

### 4.6 기준선 (`gmst/baselines.py`)
- FR-45: B0′ = d **이전 28일 이내**에서 같은 (`is_operating`, `daytype`)이고 로더가 목표를 가리지 않았으며 `n_missing == 0`인 가장 최근 날의 96점. 없으면 기간 제한 없이 같은 `is_operating`만 일치하는 가장 최근 이전 사용가능일. 미래 날짜 금지, 배열 음수 인덱스 참조 금지(참조 후보는 d보다 이른 날짜만) `[v2 §4.3, §15.4, A27, 판정 Q10]`.
- FR-46: B0 = \(Y_{d-7,q}\), 그 값이 NaN인 슬롯은 B0′ 값으로 대체 `[v2 §15.4]`.
- FR-47: `lag7_skip_table()`: 목표와 \(Y_{d-7,q}\)가 둘 다 유한한 점만 평가한 v2 §4.3 재현표(보고 전용) `[v2 §4.3]`.
- FR-48: 점예측 기준선(B0, B0′, BB)은 `q=None`(CRPS·coverage는 NaN), `M_hat_* = max(경로)`, `peak_time_mode = argmax`, `risk_raw = 1[M_hat_median > C]` `[PRD]`.
- FR-49: B1 회귀: L2 1개(`y_mean`) + 분위수 19개(`y_median` = τ 0.5), 목표 결측행 drop, 분위수 행별 정렬, 라이브러리 기본 하이퍼파라미터 `[v2 §15.4, PRD]`.
- FR-50: B1 일 단위 출력(A14): \(M_d\) 분위수 회귀 19개(행 = 일피크 대상 학습일, 라벨 = 관측슬롯 기준 일최대) → `M_hat_median` = τ 0.5, `M_hat_mean` = 19개 분위수 평균; 임계값별 이진분류기 3개 → `risk_raw` = 예측확률(학습 사건이 한 클래스뿐이면 `1 − F̂_M(C)`로 대체); `peak_time_mode` = argmax `y_median`. 보정 후 위험은 `np.minimum.accumulate`로 C50→C90 단조 강제(모든 모델 공통) `[v2 §15.4, A14, PRD]`.
- FR-51: BB(백본 단독) = `y_mean = y_median =` 백본 \(m_t\) `[v2 §6]`.

### 4.7 보정·비교·선택 (`gmst/evaluate.py`)
- FR-52: Platt: 위험을 [1/(2N), 1−1/(2N)] (N = 2,000)으로 클리핑, logit 위 2모수 로지스틱을 Newton법(numpy)으로 적합. 기본 보정법 `[v2 §11.3, A3, 수정-7]`.
- FR-53: isotonic = PAV(numpy), ablation 전용 `[A3]`.
- FR-54: fold j의 보정 확률은 나머지 3개 fold OOF로 적합한 보정함수로 계산. 테스트에는 f1–f4 전체 OOF로 적합한 보정함수를 한 번 적용. 모델×임계값별 `[v2 §11.3]`.
- FR-55: \(p^*\) = **내부검증**(각 fold 학습구간의 마지막 7일, 그 fold의 실제 검증일은 쓰지 않음)에서 보정 확률의 고유값 후보 중 F1 최대값(임계값별), 0.5와 함께 보고 `[A24, 재검토 R10]`.
- FR-56: f1–f4 검증창은 56일(각 14일)이지만, 메인평가 목표가 전부 가려진 suspect일 07-13·07-15 2일은 일 블록의 후보에서 뺀다 → **사용가능 OOF 일수 = 54일**(부분마스킹일 08-28·08-29는 포함; v2 부록A A13과 동일 정의) `[재계산]`. 이 54일을 일 블록으로 복원추출(B = 2,000, seed 0; 각 지표는 뽑힌 날들의 사용가능 점·일피크 대상일로 계산)하여 B4−B1의 ΔMAE, ΔCRPS, Δ평균Brier(보정)의 95% 백분위 CI, 그리고 oracle-weather ablation `AWstar − main`(B1)의 ΔMAE·Δ평균Brier CI. Diebold–Mariano(일 손실차, 정규근사, `math.erfc`) 보조. `results/bootstrap.csv`(`comparison, metric, delta, ci_lo, ci_hi, dm_stat, dm_p, n_days`) `[v2 §15.6, A13, A17, 재검토 R5]`.
- FR-57: 성공 기준과 개선 루프(사용자 결정, B1은 벤치마크): 후보 B4 변형에 대해 ΔMAE와 Δ평균Brier(변형 − B1) **두** CI 상한이 **모두 < 0**이면 그 변형이 성공 기준을 만족한 것이다(`gate_pass`, FR-... US-008). 만족하면 그 변형을 제출(점예측·분위수·일 단위 열 모두 그 변형), 만족하지 못하면 FR-107–FR-110의 개선 루프(B4-H → B4-IO → B4-KAN)로 2026-10-03까지 최선의 변형을 찾는다(`select_variant`, US-021). ΔCRPS는 기록만 하고 판정에 쓰지 않는다(분위수 열은 CRPS/pinball로 평가). `results/selection.json` = `{submitted, gate_met, tried:[...], criteria:{mae|brier_mean:[delta, lo, hi]}, reported:{crps:[delta, lo, hi]}, remaining_gap_to_b1, tau, half_life, final_epochs, p_star, rule}`. **B1은 어떤 경우에도 제출되지 않는다.** 선택은 봉인 테스트 실행 전에 끝난다 `[v2 §5.2, §15.6, 판정 Q5, 검증 §5.4]`.

### 4.8 백본 (`gmst/backbone.py`)
- FR-58: 클래스 = (가동여부, 요일유형) 6개 × 96. 클래스의 사용가능일 < 5(`min_days`)이면 가동여부만으로 묶은 상위 클래스 프로파일로 대체 `[v2 §6, A5, A23]`.
- FR-59: \(\hat s^{(c)} = (W_c + \tau C_2^\top C_2)^{-1} b_c\), \(C_2\)는 96차 순환 2차차분(행 q: \(s_{q-1} - 2s_q + s_{q+1}\), 인덱스 mod 96). 합=0 제약 없음(τ>0이고 W_c ≠ 0이면 양의 정부호, intrinsic GMRF 사전) `[v2 §6, 수정-2]`.
- FR-60: 가중 \(w_t = 2^{-\Delta_t/h}\), 반감기 h ∈ {30, 60, 120}일(FR-61 격자), \(\Delta_t\) = fold 마지막 train 날짜로부터 경과일, 마스킹 점 w = 0 `[v2 §6, A6]`.
- FR-61: 격자 τ {0.1, 1, 10, 100, 1000}(로그 간격; W_c 대각 = 클래스 가중 일수로 수십~100 규모라 τ/W가 약 1e-3~10을 덮음) × h {30, 60, 120} = 15조합, **내부검증**(각 fold 학습구간의 마지막 7일; 그 fold의 실제 f1–f4 검증일은 쓰지 않음) pooled BB MAE 최소 조합 선택, 테스트 적합(≤08-30 학습구간의 마지막 7일 내부검증)과 B1 행별 확장창 백본(A32)에도 같은 (τ, h). 최적값이 격자 끝이면 로그 경고 `[v2 §6, A6, 재검토 R10]`.
- FR-62: 백본은 fold마다 train 역할 날짜만으로 재적합. posterior sd는 입력에 쓰지 않음. 추세항 없음, \(\lambda_Q = 0\) `[v2 §6, §14]`.

### 4.9 조건부 HMM (`gmst/hmm.py`)
- FR-63: 전이 특징 z: A+ 14차원 = sin/cos(2πrq/96) r=1..3 (6), op (1), op×sin/cos r=1,2 (4), 토, 일, hol (3). A 8차원(sin/cos 6 + 토 + 일). B 16차원 = A+ ⊕ [1(P>0), log(1+P)/10], P는 시간 생산량을 4슬롯 반복 `[v2 §8, A15, A20]`.
- FR-64: \(A_t(i,j) = \mathrm{softmax}_j(b_{ij} + \theta_{ij}^\top z_t)\), 각 행의 \(j=i\) logit = 0 고정. B2는 θ ≡ 0(학습 안 함) `[v2 §8, §15.4]`.
- FR-65: \(\mu_{t,k} = m_t + \delta_{k,op(t)}\), \(\delta_{1,\cdot} = \rho_{1,\cdot}\), \(\delta_{k,\cdot} = \delta_{k-1,\cdot} + \mathrm{softplus}(\rho_{k,\cdot})\); \(\sigma_{k,op} = 1 + \mathrm{softplus}(\cdot)\); \(\phi_k = \mathrm{sigmoid}(\psi_k)\)(B2·B3는 φ ≡ 0). emission 파라미터 5K `[v2 §9, A7, A25]`.
- FR-66: 쌍 emission \(b_t(i,k) = N(Y_t;\ \mu_{t,k} + \phi_k(Y_{t-1} - \mu_{t-1,i}),\ \sigma^2_{k,op})\). \(Y_t\) 마스킹 → \(b \equiv 1\). \(Y_{t-1}\) 마스킹 또는 블록 첫 스텝 → \(N(Y_t;\ \mu_{t,k},\ \sigma^2_{k,op}/(1-\phi_k^2))\) `[v2 §9, A8, 수정-5]`.
- FR-67: 스케일링 forward, \(\log p = \sum_h \log c_h\). 블록 시작 α = 첫 슬롯 전이행렬의 정상분포(선형해) `[v2 §10]`.
- FR-68: 학습 = 2일 블록(warm-up d−1 + 목표 d), train 역할이면서 목표 사용가능점 ≥ 1이고 d−1이 존재하는 d마다 한 블록. 손실 = −(목표일 \(\sum \log c_h\)) / 사용가능점 수(관측 슬롯당 평균 NLL). \(\mathcal L_{balance}\) 없음. 점유 하한 \(\lambda_B \sum_k \max(0, \kappa - \bar\pi_k)^2\)(κ = 0.02, \(\lambda_B = 10\))은 기본 꺼짐(`occ_floor=False`)이고, US-012 3-seed 점검에서 점유율 < 0.02로 붕괴하는 상태가 나올 때만 켠다 `[v2 §14, A21]`. 전체배치 Adam lr 1e-2, ≤ 300 epoch, **내부검증**(해당 fold 학습구간의 마지막 7일, 같은 2일 블록 구조; 그 fold의 실제 검증일은 쓰지 않음) NLL patience 20, best epoch 기록. 테스트용 최종 적합(≤ 08-30) epoch = 같은 방식(≤08-30 학습구간 마지막 7일 내부검증)으로 고른 f1–f4 best epoch 중앙값 `[v2 §10, §14, A21, A22, 재검토 R10]`.
- FR-69: 발행 시 \(\pi_{t_0|t_0}\) = d−1의 96슬롯 forward filter, 전부 마스킹이면 정상분포 `[v2 §8, A9]`.
- FR-70: forward-backward로 filtered·smoothed 상태확률을 OOF에 기록(`state_filt`, `state_smooth` = argmax) `[v2 §13.1]`.
- FR-71: MC: N = 2,000, 날짜별 고정 seed. \(S_0 \sim \pi_{t_0|t_0}\); \(e_0 = Y_{t_0} - \mu_{t_0,S_0}\), \(Y_{t_0}\) 마스킹이면 \(e_0 \sim N(0, \sigma^2_{S_0,op}/(1-\phi^2_{S_0}))\); \(S_h \sim A_h(S_{h-1},\cdot)\), \(e_h = \phi_{S_h} e_{h-1} + \sigma_{S_h,op}\varepsilon_h\), \(Y_h = \mu_{h,S_h} + e_h\) `[v2 §11.1, A10, 수정-5]`.
- FR-72: B2·B3 위험 = 해석식 \(\pi\prod_h[A_h B_h(C)]\mathbf 1\), \(B_h\) = `torch.distributions.Normal.cdf` 대각 `[v2 §11.2]`.
- FR-73: 사다리 기록 model ID: `B2`, `B3`, `B4`. 모든 확률 모델에 raw·Platt·isotonic 열을 계산 `[v2 §15.4]`.
- FR-74: K ablation {2, 3, 4}(variant `K2`, `K4`; K3 = `main`) `[A4]`.
- FR-75: 일 랜덤효과 ablation `re`(B4, MC 전용, A30, **진단 전용 — 개선 루프의 정식 단계 아님**): 경로마다 하루 한 번 \(u_d \sim N(0, \sigma_u^2)\)를 뽑아 96슬롯 전체에 더함. \(\sigma_u\) 초기값 = fold 학습 사용가능일의 일평균 잔차 \((Y-m)\) sd(가동여부별, \(s_{op}\)). 슬롯 잡음과의 재추정 방법은 v2가 정하지 않았으므로 PRD 기본값으로 적률 분해를 쓴다: 슬롯 정상분산 \(v = \sigma^2/(1-\phi^2)\)을 \(v' = \max(v - s^2_{op}, 1)\)로 줄여 \(\sigma' = \sqrt{v'(1-\phi^2)}\) (`# ponytail: 적률 분해로 σ 재추정, re 결과가 보고서 결론을 좌우하면 u_d를 넣은 우도로 재학습`). 항상 실행·보고하고, 판별력 미달의 원인 진단에만 쓴다. 개선 루프 자체는 FR-107–FR-110(B4-H·B4-IO·B4-KAN)을 따르며 B1로 되돌아가지 않는다(사용자 결정) `[v2 §11.5, A30, PRD]`.
- FR-107: B4-H(사용자 결정, US-021): `CondHMM(emission_source="b1")`이면 \(m_t\)를 그 fold train 구간으로 적합한 B1의 fold-safe τ=0.5 슬롯 예측(A32)으로 치환한다. \(\delta,\sigma,\phi\)의 정의·제약은 FR-65와 동일, \(m_t\)만 대체(이중 사용 아님) `[v2 §9.6 ①, A34]`.
- FR-108: B4-IO(사용자 결정 "Ruling (B4-IO spec)", US-021): `CondHMM(decoder="io")`이면 §9의 상수 \(\delta_{k,op},\sigma_{k,op},\phi_k\)를 13차원 입력 \(\mathbf u_t=[1,\mathrm{op}_d,\sin_{r=1},\cos_{r=1},\sin_{r=2},\cos_{r=2},\mathrm{op}_d\sin_{r=1},\mathrm{op}_d\cos_{r=1},\mathbb 1[\text{토}],\mathbb 1[\text{일}],\mathrm{hol}_d,\bar y_{d-1},y^{\max}_{d-1}]\)과 3차원 \(\mathbf v_t=[1,\sin_{r=1},\cos_{r=1}]\)의 시변 함수로 바꾼다: \(\delta_{t,1}=\mathbf a_1^\top\mathbf u_t\), \(\delta_{t,k}=\delta_{t,k-1}+\mathrm{softplus}(\mathbf a_k^\top\mathbf u_t)\), \(\mu_{t,k}=m_t+\delta_{t,k}\), \(\sigma_{t,k}=1+\mathrm{softplus}(\mathbf b_k^\top\mathbf u_t)\), \(\phi_{t,k}=\mathrm{sigmoid}(\mathbf c_k^\top\mathbf v_t)\). \(\bar y_{d-1}\)·\(y^{\max}_{d-1}\) = 전일 관측 슬롯의 \((Y-m)\) 평균·최댓값(관측 슬롯 0개면 0). K=3에서 파라미터 \(\mathbf a\)(3×13)+\(\mathbf b\)(3×13)+\(\mathbf c\)(3×3) = **87개**. 조화항·전일요약 계수에만 L2, \(\lambda\)는 R10 내부검증 선택 `[v2 §9.6 ②, A35]`.
- FR-109: B4-KAN(사용자 결정, US-021): `z_features(kan=True)`이면 전이 특징의 sin/cos 조화항(6) + 가동×조화 교호(4) = 10차원을 제거하고, 가동여부별 주기(순환) 3차 B-스플라인 \(g(q,\mathrm{op})=\sum_{m=1}^{M}c_{\mathrm{op},m}B_m(q)\)(매듭 \(M\in\{8,12,16\}\), R10 내부검증 선택)를 전이 logit에 더한다. 0/1 플래그(토/일/공휴일/가동) 4차원은 선형 유지. 인접 매듭 계수에 매끄러움 벌점(순환), 벌점 강도도 같은 내부검증 격자에서 선택 `[v2 §9.6 ③, A36]`.
- FR-110: 개선 루프 오케스트레이션(사용자 결정): `select_variant`가 `main`(B4) → `H` → `IO` → `KAN` 순으로 US-008의 `gate_pass`를 판정하고, 하나라도 통과하면 그중 평균 OOF MAE 최솟값을, 2026-10-03까지 전부 미통과면 시도된 것 중 평균 OOF MAE 최솟값(동률 시 평균 Brier)을 최종 변형으로 정한다. B1은 후보에 없다. 결과는 `results/selection.json`에 시도 순서·게이트 결과·최종 선택·B1 대비 잔여 격차와 함께 기록된다 `[v2 §15.6, US-021]`.

### 4.10 기상 (외부자료 없음)
- FR-76: (삭제: 사용자 결정 — KMA 제외) 옛 내용: `.env` 자격증명 파싱.
- FR-77: (삭제: 사용자 결정 — KMA 제외) 옛 내용: KMA 로그인 세션.
- FR-78: (삭제: 사용자 결정 — KMA 제외) 옛 내용: ASOS 지점목록·공장 지점 식별.
- FR-79: (삭제: 사용자 결정 — KMA 제외) 옛 내용: ASOS·단기예보 최종 다운로드와 파서.
- FR-80: (삭제: 사용자 결정 — KMA 제외) 옛 내용: KMA 파싱 캐시와 A+W 건너뛰기.
- FR-81: 기상 점검은 **oracle-weather ablation 하나**뿐이며 B1에만 적용한다(US-007): 프로토콜 `A+W*`로 대상일의 데이터 자체 `기온·습도·풍속·강수량_증분`을 알려진 값처럼 넣고(완전예보 가정, 기상 기여의 **휴리스틱 상한** — 표본이 작은 f1–f4에서는 특징 추가 자체가 MAE를 악화시킬 수 있어 엄밀한 이론적 상한은 아님), f1–f4 OOF를 variant `AWstar`로 기록, `AWstar − main` ΔMAE·Δ평균Brier CI를 bootstrap.csv에 쓴다. 제출 모델·최종 적합·테스트 예측에는 절대 쓰지 않는다. 결과는 보고서 제2·3장에서 외부 기상을 쓰지 않은 근거로 제시한다(≤08-31 시간 단위 corr(전력, 기상): 기온 0.05, 습도 −0.10, 풍속 0.12, 강수 −0.02; 공장 위치 미상) `[v2 §4.6, §4.7, A17, 판정 Q3, 사용자결정, 재검토 R9]`.

### 4.11 요금·시나리오 (`gmst/scenario.py`)
- FR-82: `TARIFF` = 요금표의 고압A 선택Ⅰ/Ⅱ/Ⅲ 기본요금·전력량요금 그대로. 기본 선택Ⅱ `[v2 §12.1, A19, 요금표]`.
- FR-83: 계절·시간대는 요금표 표 그대로(여름·봄가을 공통, 겨울 별도). 공휴일(`HOLIDAYS_2021` ∪ **일요일**(`daytype=="sun"`), 08-16은 `tariff_0816` 스위치, 기본 True = 광복절 대체공휴일을 요금상 공휴일로 가정하고 결과에 가정 명시, f3 시나리오 ₩에만 영향)은 에너지·수요 모두 경부하. 비공휴 토요일 최대부하는 에너지만 중간부하 단가. 일요일은 `HOLIDAYS_2021` 목록이 아니라 `daytype`로 판정하므로 별도 하드코딩이 필요 없다 `[v2 §12.1, 요금표, 판정 Q1, 재검토 R1]`.
- FR-84: 에너지요금 = Σ 0.25 · y(kW) · 단가. 단위 "15분 평균 kW" 가정을 결과·README에 명시하고 상대변화(%)를 병기 `[v2 §12.1, A19]`.
- FR-85: 요금적용전력(청구월 단위, 경로별, A29): 경로 n의 월 최대 \(M^{mp,(n)}\) = 그 달 시나리오 대상일(FR-89)들의 중간·최대부하 슬롯(일요일·공휴일 제외) 경로 최대의 최댓값(날짜 간 경로는 같은 인덱스끼리 독립 짝짓기, `# ponytail: 날짜 간 경로 독립 짝짓기, 날짜 간 상관이 크면 일 랜덤효과 경로로 교체`), \(P^{app,(n)} = \max(M^{mp,(n)}, P_{floor})\), \(E[P^{app}]\) = 경로 평균. \(P_{floor}\) = 그 달의 시나리오 대상일을 뺀 **대상월까지의 전체 관측 이력**(fold 창에 한정하지 않음; 래칫월 1·2·7·8·9월 중 대상월 이하 + 대상월)의 중간·최대부하·비공휴일·비일요일 **사용가능 관측점**(마스킹 점 제외)의 최대 → 7·8월 모두 222(2021-07-19 11:15). 2020-12 부재, 계약전력 30% 하한 미적용, 마스킹 점 제외를 결과에 가정으로 명시 `[v2 §12.1, A29, 요금표, 판정 Q7, 판정]`.
- FR-86: Δ비용(a) = 8,320 × (E[P_app(a)] − E[P_app(0)]) + Σ_{월} 0.25 (ŷ⁽ᵃ⁾ − ŷ⁽⁰⁾) × 단가, ŷ = MC 평균경로(`y_mean`), 계산 범위 = 청구월. 래칫 바닥 222 때문에 7–8월 기본요금 항은 경로의 월 최대가 222를 넘는 꼬리를 줄일 때만 0이 아니며 사실상 0이다. ₩ 절감의 주 수단은 전력량요금 항(TOU 이동)이다 `[v2 §12.1, 판정]`.
- FR-87: SCENARIO 모델 = B4 + 프로토콜 B, fold f2–f4의 train 역할 중 ≥ 2021-07-01 날짜만으로 학습(f1은 07-01 이후 학습일이 5일뿐이라 제외). OOF 지표는 variant `B`, 항상 "상한" 표기, 제출 금지 `[v2 §4.6, §12, §12.1, 판정 Q2]`.
- FR-88: 시나리오 변환(시간 생산량 24벡터 P, 이동률 s ∈ {0.1, 0.2, 0.3}, 일 합 보존; 받는 시간이 없으면 해당 변환은 무변경으로 기록) `[v2 §12, A24, PRD]`:
  - `shift_peak`: 최대부하 시간의 s·P를 떼어 같은 날 비최대부하·P>0 시간에 P 비례 배분
  - `stagger_start`: 각 생산 연속구간 첫 시간의 s·P를 그 구간 둘째 시간으로
  - `avoid_high`: `baseline` 예측 중앙값 경로가 C90을 넘는 슬롯이 있는 시간의 s·P를 그 외 P>0 시간에 비례 배분
  - `ease_peak`: `baseline` `peak_time_mode` ±2슬롯이 걸친 시간의 s·P를 그 외 P>0 시간에 비례 배분
- FR-89: 시나리오 대상일 = f2–f4 검증창의 가동 & 일피크 대상일. 같은 날 시나리오는 공통 난수. \(J_d(a)\)는 **날짜 d 단위**로 정의한다(재검토 R7): 일 단위 전력량요금 변화 \(\Delta E_d(a)\) + \(\eta\,\mathcal C_{change,d}(a)\), 위험 제약 `risk_ok` = \(R^{(a)}_d(C_{90}) \le R^{(0)}_d(C_{90})\). 일 단위 성분(A28, `scenarios.csv`): \(E[M]\), \(R_{C90}\), 변경량 \(\mathcal C_{change,d}(a) = \sum_h|\Delta P_{d,h}| / (2\sum_h P_{d,h})\)(열 `C_change`), `d_energy_won` = \(\Delta E_d(a)\). **월 단위 기본요금 변화(`d_demand_won`)는 \(J_d(a)\)와 더하지 않고 `scenarios_month.csv`에 청구월 단위로 따로 저장**한다(월 1회 래칫이라 날짜별로 쪼갤 수 없음). 월 단위 손익분기 \(\eta^* = -\)`d_total_won`\(/\mathcal C_{change}\)(월 합)는 `scenarios_month.csv`에서만 보고한다. η와 λ 가중치는 쓰지 않는다(위험은 제약, η는 가정하지 않고 \(\eta^*\) 보고) `[v2 §12, A28, 재검토 R7]`.

### 4.12 분석 (`gmst/analysis.py`)
- FR-90: 조건별 오차표(US-017), 대상 모델 = 제출모델과 B4(다르면 둘 다) `[v2 §13.1–§13.3]`.
- FR-91: FN/FP 일표와 요약표, 대표 임계값 q90(F1·FN·FP는 f1–f4 합산 OOF, FR-42), \(p^*\) 두 가지 `[v2 §13.4, §4.5, §16.4]`.
- FR-92: 증강 누수 격차: 복사 포함·봉인 패널에서 B1 특징으로 일 단위 랜덤 5-fold `[PRD]` vs LOEO(event_id) MAE `[v2 §15.3]`.
- FR-93: f4 B4 모델의 조건부 전이확률표 `[v2 §13.4]`.

### 4.13 러너·제출 (`gmst/run_all.py`)
- FR-94: 단계 순서: ① preprocess ② splits ③ τ 선택 + f1–f4 OOF(B0, B0p, BB, B1, B2, B3, B4 + ablation) ④ 보정(OOF) ⑤ metrics ⑥ risk_check ⑦ bootstrap ⑧ **개선 루프**(FR-110의 `select_variant`: `gate_pass`가 미통과면 B4-H → B4-IO → B4-KAN을 f1–f4 OOF + R10 내부검증으로 순서대로 시도, 2026-10-03 중단 규칙) ⑨ selection.json ⑩ state_stability ⑪ analysis ⑫ scenarios ⑬ `unseal=True`로 ≤ 08-30 최종 적합(B0, B0p, B1, **선택된 B4 변형**) 및 09-01~09-14 발행 ⑭ test_predictions / eval_mask / test_metrics. B1은 제출되지 않으므로 조건부 모듈 파일 단계는 없다. 단계마다 이름·소요초·DEVICE를 stdout에 `[결정-코드, v2 §15.6, §9.6]`.
- FR-95: ablation 행렬: `main`(전 모델); `copies`(완전+편집 복사 포함), `suspect`, `protoA`(B1, B4); `K2`, `K4`, `re`(B4); `AWstar`(B1, oracle-weather, 제출 금지); `B`(SCENARIO); 보정 none/Platt/isotonic은 열로 `[v2 §15.4, 수정-9]`.
- FR-96: `results/test_predictions.csv`(UTF-8, 1,344행) 열 순서: `datetime, track, model, y_mean, y_median, q05, q10, q15, q20, q25, q30, q35, q40, q45, q50, q55, q60, q65, q70, q75, q80, q85, q90, q95, M_hat_median, M_hat_mean, peak_time_mode, risk_C50, risk_C75, risk_C90, C50, C75, C90`(33열). datetime `%Y.%m.%d %H:%M:%S`, `track = MAIN`, `model` = **`select_variant`가 정한 B4 변형 ID**(`B4`/`B4-H`/`B4-IO`/`B4-KAN` 중 하나, B1 아님), `peak_time_mode`는 `HH:MM`, 일 단위 값은 그날 96행 반복, 위험은 보정 후, M·위험은 96슬롯 전체 기준(발행시점엔 결측을 모름) `[v2 §4.5, A16, 수정-1, 수정-6, PRD]`.
- FR-97: 평가 마스크 `results/eval_mask.csv`(`datetime, is_missing`, 1,344행 — 결측은 사후 정보라 예측파일과 분리)와 `results/test_metrics.csv`(metrics.csv 형식, fold `test`, 일피크는 관측슬롯 기준) `[v2 §4.5, §15.5, A16, 판정: v2 이름 우선]`.
- FR-98: (삭제: 사용자 결정 — B1은 벤치마크로 고정, 제출은 항상 §15.6·FR-57·FR-110의 `select_variant`가 정한 B4 변형이므로 조건부 "B4 모듈 파일" 첨부는 없다. 번호는 참조 유지를 위해 비워 둔다) `[v2 §4.5, §15.6, A16, 사용자결정]`.
- FR-99: `--smoke`: f4만, epoch·MC 경로·부트스트랩 반복을 최소로 줄여 같은 파일 집합을 만든다(테스트 전용) `[PRD]`.
- FR-100: `requirements.txt`는 `uv export --no-hashes --no-dev` 기반, 첫 줄 `--extra-index-url https://download.pytorch.org/whl/cu126` `[환경, 결정-대회]`.
- FR-101: `README.md` 목차: 개요(문제정의 H/L/발행시각/C), 환경(`uv sync` 또는 `pip install -r requirements.txt`), 단일 명령, 산출물 표(파일 → 보고서 장), 데이터 주의점 요약(§2A 표의 처리 규칙), 제출파일 스키마와 `model` 열 의미, 외부자료(2021 공휴일·한전 요금표 하드코딩과 출처; 외부 기상 미사용과 그 근거인 oracle-weather 결과), 재현성(seed, CUDA/CPU), 폴더 구조, 제출 zip `[결정-대회, v2 §20, 사용자결정]`.
- FR-102: `--package`: `dist/kamp_power_src.zip`에 `gmst/, tests/, notebooks/, results/, DATA의 CSV(원본·가공), requirements.txt, README.md, pyproject.toml, uv.lock`만. `.env`, `.venv/`, `document/`, `tasks/`, `claudedocs/`, `__pycache__/`, `dist/` 제외 `[결정-대회]`.

### 4.14 노트북
- FR-103: `notebooks/03_results.ipynb`는 읽기·표시만(US-019) `[결정-코드]`.
- FR-104: `notebooks/01_preprocess.ipynb`는 스크립트 산출물을 읽어 점검·그림만, 쓰기 없음 `[결정-코드, v2 §18]`.
- FR-105: `notebooks/02_eda.ipynb`는 스크립트 산출물을 읽고, 설계통계는 ≤ 08-31, §8 모드 수 교정·§11/H=96 주의문(US-019) `[v2 §4.4, §4.5, 수정-10]`.

### 4.15 부채 원장
- FR-106: `PONYTAIL-DEBT.md`(US-020) `[결정-코드, 수정-4]`.

---

## 5. 비목표 (Non-Goals)

- R/INLA full posterior, TimeXer 인코더(B5 포함), HSMM/explicit duration, Student-\(t\)·logistic emission, 미분가능 Laplace, 결합 일정최적화 — Research Extension v2 `[v2 §7, §19]`.
- K = 5, 6 `[v2 §8]`. 백본 학습파라미터화(\(\lambda_Q > 0\)) `[v2 §14]`. 손실에 Brier 넣기 `[v2 §14]`.
- 07-13·07-15의 원래 시각 복원 시도(마스킹만) `[결정-데이터]`.
- KPX 수요/SMP/DR, 폭염특보, 일출·일몰 `[결정-외부]`.
- KMA 단기예보·ASOS 등 외부 기상자료 수집, 공장 위치(지점) 식별, 기상 예보 변형(A+W) `[사용자결정, v2 §4.7]`.
- LightGBM 하이퍼파라미터 탐색(기본값 + ponytail 주석) `[PRD]`.
- 다중 GPU/분산학습, 웹 UI·대시보드, 2021-09-14 이후 예측.
- 결과보고서 PDF·발표자료 작성(이 PRD의 산출물을 입력으로 별도 작업).
- `main.py`(uv 기본 스텁) 수정.
- **컷 리스트**(일정이 밀리면 이 순서로 제외, 결과표에 "미실행" 기록) `[v2 §18, 판정 Q12]`: ① K = 4 ② isotonic 보정. B5 인코더는 컷 항목이 아니라 애초부터 비목표(위 R/INLA·TimeXer 항목)이므로 목록에 없다. 선택규칙(FR-57), 봉인 테스트, 누수 테스트, 제3·4장 분석, oracle-weather ablation은 자르지 않는다.

---

## 6. 설계 고려사항 (Design Considerations)

### 폴더 구조
```
manufacture_ai/
├── gmst/
│   ├── __init__.py      # ROOT, DATA, RESULTS
│   ├── preprocess.py    # 측정 복원 → okm_15min_2021.csv, data_audit.json
│   ├── splits.py        # 날짜표·fold 역할 → okm_cv_splits_2021.csv
│   ├── features.py      # load_panel(마스킹·봉인), PROTOCOLS, 특징
│   ├── evaluate.py      # 지표, rolling_origin, 보정, 부트스트랩, 선택규칙
│   ├── baselines.py     # B0, B0′, B1(LightGBM)
│   ├── backbone.py      # 순환 RW2 백본(BB)
│   ├── hmm.py           # 조건부 HMM, MC, 해석식, state_stability
│   ├── scenario.py      # 한전 요금 + SCENARIO 시나리오
│   ├── analysis.py      # 제3장 표, 누수 격차, 전이표
│   └── run_all.py       # 단일 명령
├── tests/               # test_<module>.py + test_leakage.py, test_state_stability.py, test_style.py, test_run_all.py
├── notebooks/           # 01, 02(표시 전용으로 전환), 03_results(신설)
├── results/             # run_all이 매번 다시 만듦
├── 5. 자원 최적화 AI 데이터셋/   # 원본(불변) + 스크립트 산출 CSV
├── requirements.txt, README.md, PONYTAIL-DEBT.md
```
v2 A18의 8개 파일 구성에서 `lgbm`은 naive 기준선과 합쳐 `baselines.py`, 요금·분석은 각 1파일로 둔다. 외부 수집 모듈은 없다(KMA 제외). 설정파일은 만들지 않고 상수는 쓰는 모듈 상단에 둔다.

### 단일 명령
`uv run python -m gmst.run_all` 하나가 결정론적으로 `results/`를 다시 만든다. 별도 수집 명령·네트워크·자격증명은 필요 없다.

### 노트북은 표시 전용
노트북은 CSV/JSON을 읽어 표와 그림만 만든다. 데이터 파일을 쓰거나 모델을 학습하지 않는다. 보고서 장 매핑(v2 "보고서 장 매핑" 표)을 03의 절 순서로 쓴다.

### 산출물 → 보고서 장
| 파일 | 장 |
|---|---|
| `data_audit.json`, `okm_cv_splits_2021.csv`, `leakage_gap.csv` | 1 |
| `metrics.csv`, `bootstrap.csv`, `selection.json`, `risk_check.json`, `test_metrics.csv`, `backbone_tau.csv` | 2 |
| `error_by_condition.csv`, `fn_fp_*.csv`, `hmm_transitions.csv` | 3 |
| `scenarios.csv`, `scenarios_month.csv` | 4 |
| `state_stability.csv`, 보정 전/후 `oof_days.csv` | 5 |
| `test_predictions.csv`(항상 선택된 B4 변형), `eval_mask.csv`, README, requirements | 6 |

oracle-weather 결과(`metrics.csv`의 variant `AWstar`, `bootstrap.csv`의 `AWstar − main`)는 제2·3장에서 외부 기상 제외의 근거로 쓴다 `[v2 보고서 장 매핑]`.

---

## 7. 기술 고려사항 (Technical Considerations)

- **환경**: Python ≥ 3.13, uv. torch 2.14.0+cu126(드라이버 535 때문에 cu126 인덱스), RTX A6000 × 2 확인됨. 모든 명령은 `uv run`. 테스트는 CPU에서도 통과해야 한다 `[환경]`.
- **scipy**: lightgbm이 전이 설치할 수 있으나 코드는 쓰지 않는다(FR-1). `test_style.py`가 import를 막는다.
- **Polars만**(pandas 없음). LightGBM에는 numpy 배열을 넘긴다.
- **경로**: 데이터 폴더명에 공백·한글(`5. 자원 최적화 AI 데이터셋`) → `pathlib`만 사용. 원본 CSV는 BOM.
- **dtype**: 백본·지표는 float64(numpy). HMM은 GPU float32 기본, 테스트의 전수합 비교는 float64 CPU.
- **결정론**: MC는 날짜별 seed(`torch.Generator`)로 같은 입력 → 같은 출력. 같은 장치 재실행 시 테스트 파일 동일이 목표(성공지표).
- **보안**: 코드는 `.env`를 읽지 않는다(외부 수집 없음). 저장소 루트의 `.env`는 `.gitignore`에 있고 제출 zip에서 제외한다(FR-102).
- **봉인·설계통계**: 테스트창은 로더 수준에서 가린다. 임계값·τ·K·p*·빈 경계·기후값 등 모든 설계통계는 ≤ 2021-08-31에서만 `[수정-10]`.
- **리스크 — 경로 최대 부풀림** `[수정-9b]`: 컨트롤러의 조악한 3상태 + AR(1) 모의실험(f4)에서 일 MAE는 검증 10일 모두 개선됐지만 모든 날 P(M>198) ≈ 0.96, 실제 사건 2/10일(Brier 0.74 vs 기저율 0.16). 96슬롯 최대가 슬롯 잡음으로 부풀면 위험이 판별력 없이 포화된다. 대응: `risk_check`(AUC ≥ 0.70·조건부 BSS > 0, A26)로 탐지하고 `re` ablation(FR-75, 진단 전용)으로 원인을 확인, Platt가 남은 편향을 교정. 그래도 판별력이 없으면 개선 루프(FR-107–FR-110: B4-H → B4-IO → B4-KAN, 사용자 결정)를 계속 진행한다 — **위험 열이 B1 분류기로 넘어가는 일은 없다**(§5.2, §15.6) `[v2 §11.5, §9.6]`.
- **git**: `manufacture_ai`는 자체 git 저장소(main)이고 작업은 worktree 브랜치에서 한다. 커밋 메시지에 Co-Authored-By 트레일러를 넣지 않는다(사용자 규칙) `[판정]`.
- **일정**: 마감 2026-10-08 23:59, 오늘 2026-09-22. 컷 리스트는 §5. §9.6/US-021의 **개선 루프는 2026-10-03까지**만 돌린다(v2 Phase 4, 사용자 결정) — B4가 US-008의 성공 기준을 못 넘기면 그 시한까지 B4-H → B4-IO → B4-KAN을 순서대로 시도하고, 시한에 도달하면 그때까지 최선의 변형으로 멈춘다. **10-04~10-06**은 선택된 변형의 최종 적합·봉인 테스트·보고서 반영(v2 Phase 5–6), **10-07~10-08**은 순수 버퍼다.

---

## 8. 성공 지표 (Success Metrics)

- `uv run pytest -q` 전부 통과(GPU·CPU), `uv run python -m gmst.run_all` 종료코드 0.
- G3의 재현수치가 소수 둘째 자리까지 일치.
- 같은 장치에서 `run_all` 2회 실행 시 `test_predictions.csv`의 수치 열 최대 절대차 ≤ 1e-6.
- 제출된 위험확률: 세 임계값 각각 보정 후 OOF Brier < 조건부 기후값 Brier, 날짜 간 AUC ≥ 0.70, BSS_cond > 0 (q90은 f1–f4 합산) `[A26]`. 미달이면 `risk_check.json`과 보고서에 그대로 기록(성공지표 미달로 표시).
- oracle-weather ablation(`AWstar`)의 ΔMAE·Δ평균Brier CI가 `bootstrap.csv`에 있고 보고서 제2·3장에 외부 기상 제외 근거로 인용된다.
- B1이 f1–f4 pooled MAE에서 B0′보다 낮음(아니면 B1 튜닝 트리거 발동, ponytail 원장에 기록).
- **B1은 어떤 경우에도 제출되지 않는다**(`selection.json`의 `submitted`가 `B1`이면 실패). 성공 기준(`gate_pass`)을 만족하지 못하면 `selection.json`에 시도한 개선 루프 변형(B4-H/B4-IO/B4-KAN) 전부와 최종 선택, B1 대비 잔여 격차(`remaining_gap_to_b1`)가 기록되고 보고서 제2장에 다중비교 주의문과 함께 공개된다(사용자 결정) `[v2 §5.2, §15.6, §9.6]`.
- DC1–DC16 각 행에 대응 테스트가 1개 이상 존재.
- 보고서 6개 장 각각에 대응하는 `results/` 파일이 1개 이상(§6 표).
- `PONYTAIL-DEBT.md` 항목 수 = `grep -rn "# ponytail:" gmst tests | wc -l`.

---

## 9. 미해결 질문 (Open Questions)

없음.

옛 질문 Q1–Q12는 모두 판정으로 해결되어 기본값으로 옮겼다: Q1 08-16 요금상 공휴일 → DC11·FR-83, Q2 SCENARIO fold f2–f4 → US-016·FR-87, Q3 기상 점검은 oracle-weather만(B1) → US-007·FR-81, Q4 PRD 기본값 승인(v2가 값을 정한 A14·A21·A26·A30·A31은 v2 우선) → 각 FR, Q5 선택규칙 ΔMAE·Δ평균Brier → US-008·FR-57, Q6 KMA 다운로드 → 해당 없음(KMA 제외), Q7 래칫 가정 → FR-85, Q8 조건부 기후값 가동×요일유형 → FR-40, Q9 scipy 미사용 → FR-1, Q10 B0′ 28일 규칙 → G3·FR-45, Q11 좁은 원본 교체 규칙 → DC1·FR-17, Q12 컷 리스트 → §5 `[판정]`.

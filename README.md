# KAMP 공장 전력·피크 위험 예측

## 개요

매일 **00:00**에 다음 96개 15분 슬롯(H=96)을 예측합니다. 과거 7일(L=672)과 발행 시점에 알려진 달력·가동계획(A+)을 사용합니다. 가동계획은 당일 실현 생산량에서 복원한 플래그이므로 결과는 **계획을 알고 있다는 가정하의 성능**입니다. C50·C75·C90은 각 fold 학습 가동일 최대값의 50·75·90% 분위수입니다.

B0/B0p, 주기 RW2 백본(BB), LightGBM(B1), 조건부 HMM(B2/B3/B4)을 비교합니다. B1은 벤치마크이며 제출 모델은 B4 계열입니다. 코드와 스키마 검증, 축소 smoke, 전체 개발 실험, 실제 봉인 평가를 구분합니다. 전체 개발과 실제 9월 평가를 각각 완료했으며, 상세 수치와 한계는 [전체 실험 기록](docs/superpowers/reviews/2026-09-23-full-experiment-validation.md)에 있습니다. smoke 수치는 성능 주장에 사용하지 않습니다.

## 환경

Python 3.13 이상, `uv`를 사용합니다.

```bash
uv sync
# 제출 환경에서 uv를 쓰지 않는 경우
pip install -r requirements.txt
```

`requirements.txt`는 `uv export --no-hashes --no-dev --no-emit-project`로 생성한 런타임 고정 버전입니다. PyTorch cu126 인덱스는 `https://download.pytorch.org/whl/cu126`입니다. CPU 실행은 `CUDA_VISIBLE_DEVICES="" uv run python -m gmst.run_all --smoke`로 선택합니다. 외부자료 취득, 네트워크 호출, 자격증명은 실행에 필요하지 않습니다.

## 단일 명령

```bash
uv run python -m gmst.run_all                         # 개발 단계만, 9월 봉인 유지
uv run python -m gmst.run_all --no-final              # 위와 동일
uv run python -m gmst.run_all --smoke --out /tmp/kamp-smoke
uv run python -m gmst.run_all --occ-floor             # 상태 점유 붕괴 진단 후 개발 재실행
uv run python -m gmst.run_all --package               # 학습 없이 소스 zip만 생성
```

smoke의 기본 출력은 별도 `results_smoke/`이고, 명시적 `--out`이라도 기존 전체 개발 selection을 덮어쓰지 못합니다. smoke는 f4의 **2021-08-18~08-31** 14일을 pseudo-test로 사용합니다. epoch=2, 경로=50, LightGBM=10라운드, bootstrap=20회이며, LOEO와 시나리오 날짜도 줄입니다. `test_predictions.csv`라는 제출 형식을 쓰더라도 9월 평가가 아닙니다.

전체 개발 결과에서 선택을 고정한 후에만 `uv run python -m gmst.run_all --final --out results`를 명시적으로 실행합니다. 이 경로는 OOF 선택을 다시 하지 않으며, 고정 임계값과 참조 OOF를 검사한 뒤 실제 봉인을 한 번 엽니다. `final_started.json`은 진행·성공·실패 상태를 기록합니다. 성공하거나 진행 중인 final은 반복할 수 없습니다. 실행 결함을 고친 뒤에만 같은 selection으로 `--final --retry-reason "수정한 결함과 재실행 사유"`를 명시할 수 있으며 `final_attempts.jsonl`에 이력이 남습니다. `--smoke --final`은 오류이며 smoke의 selection으로 실제 final을 실행할 수 없습니다.

현재 `results/`의 실제 final은 **1회 완료** 상태입니다. 같은 위치에서 개발 명령이나 `--final`을 다시 실행하면 거부됩니다. `selection.json`은 final 전 SHA-256 `a138f8dac423e8940ac405f0adb2070efff56835efd385e172e0314dea2eba3a` 그대로 보존됐습니다.

## AR 축소 사전분포 개발 실험

```bash
uv run python -m gmst.ar_experiment --smoke --out /tmp/kamp-ar-smoke
uv run python -m gmst.ar_experiment --out /tmp/kamp-ar-full
```

첫 명령은 마지막 사전 개발 창과 seed 0만 작은 예산으로 검사합니다. 두 번째 명령은 9월 이전 8개 rolling 창과 seed 0/1/2에서 B1, B3(φ=0), B4, B4-low(낮은 φ 초기값), B4-PC(같은 초기값과 AR(1) PC prior)를 비교합니다. 결과는 `fold_metrics.csv`, `summary.csv`, `decision.json`에 저장하며 기존 경로나 공식 `results/`에는 쓰지 않습니다. `--u`, `--alpha`로 `P(φ>u)=α`를, `--epochs`, `--paths`, `--rounds`로 실행 예산을 바꿀 수 있습니다. B4-PC의 자동 판정은 세 seed 모두에서 평균 MAE가 B4·B4-low·B3보다 낮을 때만 통과하며, 동률은 B3를 우선합니다. 이는 겹치는 개발 창의 탐색 결과이지, 이미 봉인 평가를 끝낸 9월 성능의 갱신이 아닙니다.

## 산출물과 보고서 장

| 파일 | 장 / 용도 |
|---|---|
| `data_audit.json`, 데이터 폴더의 `okm_cv_splits_2021.csv`, `leakage_gap.csv` | 1 / 데이터·누수 |
| `metrics.csv`, `bootstrap.csv`, `selection.json`, `risk_check.json`, `backbone_tau.csv` | 2 / 비교·선택 |
| `oof_slots.csv`, `oof_days.csv`, `inner_days.csv`, `calibration_status.csv` | 2·5 / OOF·보정 근거 |
| `ablation_status.csv`, 실행된 경우 `io_lambda.csv`, `kan_grid.csv` | 2 / 실행 여부·fold별 격자 |
| `error_by_condition.csv`, `fn_fp_days.csv`, `fn_fp_summary.csv`, `hmm_transitions.csv` | 3 / 선택된 변형의 오류 |
| `scenarios.csv`, `scenarios_month.csv` | 4 / 일별 변경·월별 비용 |
| `state_stability.csv` | 5 / 세 seed 상태 안정성 |
| `test_predictions.csv`, `eval_mask.csv`, `test_metrics.csv`, `final_p_star.json`, `final_status.json` | 6 / smoke 또는 명시적 final |

`--out`은 결과 위치를 바꿉니다. 전처리·날짜표 CSV는 데이터 폴더에 갱신됩니다. 개발 명령은 같은 출력 폴더의 이전 pseudo-test 파일을 제거합니다. 실제 final을 시작한 출력 폴더에서는 개발 재선택도 막습니다.

## 데이터 주의점

| 규칙 | 처리 |
|---|---|
| DC1 완전 복사 | 가장 이른 원본 보존, 강한 생산 모순일 때만 원본 교체; 복사 목표·lag·warm-up 마스킹 |
| DC2 07-13·07-15 | 소실된 시각을 추측하지 않고 목표·생산량 마스킹, 가동일 유지 |
| DC3 전력 0 | 정전·계측 손실로 결측 처리; 관측 슬롯끼리 평가, 일 최대는 결측 ≤4슬롯 |
| DC4 풍속 결측 | 시간 단위 선형보간 후 4슬롯 반복 |
| DC5 누적 강수 | 자정을 가로지르는 차분, 음수는 리셋, 원 결측 증분 0 |
| DC6 누수 열 | `공장인원`, `평균`은 항등식 감사 후 제거 |
| DC7 원자료 계절값 | 계절 지시만 사용; 원자료 값을 요금 단가로 사용하지 않음 |
| DC8 생산 실적 | MAIN은 당일 생산량 미사용; SCENARIO(B)만 계획으로 가정한 상한 |
| DC9 중복 purge | 학습·검증 event 겹침 제거, LOEO 보조 진단 |
| DC10 봉인 | 9월 1~14일 Y·X는 기본 로더에서 NaN; 설계 통계는 8월 31일까지 |
| DC11 요금 규칙 | 일요일·공휴일 경부하, 토요일 최대부하 에너지만 중간단가, 월별 래칫 |
| DC12 가동 여부 | 일 생산량 양수 또는 suspect; 수정된 비가동 62일 |
| DC13 공휴일 | 휴무와 동일시하지 않고 가동·요일과 별도로 사용 |
| DC14 편집 복사 | 가동구간 동일값 ≥20개 규칙, 전이적 그룹 병합 없이 마스킹 |
| DC15 시각 | 구간 시작 라벨: 15분→HH:00, 60분→HH:45; 슬롯 내 위치 특징 |
| DC16 단위·베이스 | emission scale 하한 1, 원 단위는 조건부 가정과 상대 변화율 병기 |

날짜표 정책·임계값 계산은 모델 적합과 분리되어 있습니다. f1~f4 bootstrap 모집단은 유한 목표가 있는 날짜에서 도출하며 현재 정책에서는 53일입니다. 점별 지표와 일 피크 지표는 서로 다른 유효 분모를 보고합니다.

## 모델 선택과 해석

각 outer fold의 τ·반감기와 IO/KAN 설정은 해당 학습기간의 내부 시간순 fit/tune/cal 분할로만 선택합니다. 모든 모델과 새 변형에서 보정과 p*를 다시 계산합니다. NLL은 예측 후 고정 모델로 평가합니다. 결측 직후 AR은 정상분산 재시작 근사이므로 정확한 결측 주변우도로 해석하지 않습니다.

게이트는 B1 대비 ΔMAE와 Δ평균 Platt Brier의 paired day bootstrap CI 상한이 모두 0 미만인 경우입니다. 실패 시 **B4-H → B4-IO → B4-KAN**을 순서대로 시도하며 2026-10-03(한국시각) 이후에는 추가 후보를 시작하지 않습니다. 게이트를 통과한 후보가 있으면 그 안에서, 없으면 실행한 B4 후보 중 pooled MAE가 최소인 후보를 선택합니다. 동률이면 평균 Brier를 사용합니다. `selection.json`에 시도 목록·개수·구성요소·잔여 B1 격차를 기록합니다. 같은 OOF에서 반복 비교한 CI는 **탐색적**이며 다중선택 편향이 제거된 우월성 증명이 아닙니다.

K4·isotonic은 승인된 선택적 컷으로 `not_run`을 기록합니다. 미실행 점수는 만들지 않습니다. `risk_check.json`의 AUC·조건부 Brier skill 미달도 그대로 남깁니다. 임계값별 보정 위험은 MC 경로의 하나의 보정된 최대분포를 구성한다고 주장하지 않습니다.

## 제출 파일

`test_predictions.csv`는 UTF-8, 1,344행, 아래 순서의 33열입니다.

```text
datetime,track,model,y_mean,y_median,q05,q10,q15,q20,q25,q30,q35,q40,q45,q50,q55,q60,q65,q70,q75,q80,q85,q90,q95,M_hat_median,M_hat_mean,peak_time_mode,risk_C50,risk_C75,risk_C90,C50,C75,C90
```

datetime은 `YYYY.MM.DD HH:MM:SS`, track은 `MAIN`입니다. model은 `B4`, `B4-H`, `B4-IO`, `B4-KAN` 중 선택된 변형입니다. 일 최대와 위험은 발행 시점의 96슬롯 전체 기준이며 하루 96행에 반복됩니다. `peak_time_mode`는 `HH:MM`입니다. 위험은 선택된 변형의 시간순 보정 후 값입니다. 실제 test 임계값도 사전 계산해 selection에 고정하며 최종 시점에서 변경하지 않습니다.

`eval_mask.csv`의 `datetime,is_missing`은 사후 평가용으로 분리됩니다. `test_metrics.csv`는 관측 슬롯과 평가 가능 일만 사용합니다. 경로·점예측의 평가 위험은 관측 슬롯 마스크를 반영하지만, B1의 일별 분류기는 슬롯 결합분포가 없어 부분 결측일의 공동 확률을 다시 계산할 수 없습니다. 제출 위험은 발행 시점의 전체 슬롯 기준입니다.

## 외부자료와 원 단위 가정

`HOLIDAYS_2021`은 **한국천문연구원 특일정보(2021년)**에서 확정한 8일입니다. 요금은 제공된 **한국전력 산업용(을) 고압A 2021 요금표**, 선택Ⅰ/Ⅱ/Ⅲ를 코드에 옮겼으며 기본은 선택Ⅱ입니다. 네트워크로 최신 요금이나 날씨를 가져오지 않습니다.

외부 기상은 사용하지 않습니다. B1의 `AWstar`는 데이터 자체의 당일 기상을 안다고 가정한 oracle-weather 진단입니다. 실제 비교는 `bootstrap.csv`의 `AWstar-main` ΔMAE·Δ평균Brier와 CI를 사용하십시오. OOF ΔMAE는 +0.115, 95% CI는 −0.234~+0.433으로 개선 근거가 없습니다. 이는 외부 기상 서비스의 효과를 직접 검증한 결과는 아닙니다.

원 단위 계산은 **15분 평균 kW** 가정하의 값입니다. 상대변화(%)도 함께 봅니다. 기준자료의 래칫 바닥 222(07-19 11:15) 때문에 7~8월 기본요금 절감은 대체로 0에 가깝고, 주요 수단은 시간대별 전력량요금 이동입니다. 월별 실제 바닥·비용은 `scenarios_month.csv`를 따릅니다. 2020-12 이력이 없고 계약전력 30% 하한은 적용하지 않았습니다. 08-16은 요금상 공휴일로 가정합니다. kWh당 가산요금이 상쇄된다는 해석은 `d_kwh ≈ 0`일 때만 가능합니다. 일별 전력량 변화와 월 1회 기본요금 변화는 더해서 일별 목적값으로 만들지 않습니다.

## 재현성·검사

seed=0, 날짜별 MC 난수, 같은 장치에서 동일 입력을 사용합니다. CUDA와 CPU 사이의 동일 숫자는 보장하지 않습니다. 같은 장치의 smoke 2회 결과는 수치 열 최대 절대차 ≤1e-6을 목표로 검증하며 실제 봉인창은 재현성 시험에 사용하지 않습니다.

```bash
uv run pytest -q
KAMP_RESULTS=/tmp/kamp-smoke uv run pytest tests/test_results.py -q
```

세 notebook은 파일 읽기·표·그림만 수행합니다. `notebooks/`에서 실행하고 결과 경로를 바꾸려면 `KAMP_RESULTS=/tmp/kamp-smoke`(또는 `GMST_RESULTS`)를 지정합니다. 모델 적합과 CSV 생성은 runner의 역할입니다.

## 폴더 구조와 제출 zip

`gmst/`에는 `contracts`, `preprocess`, `splits`, `features`, `lgbm_features`, `backbone`, `baselines`, `evaluate`, `hmm`, `hmm_core`, `hmm_training`, `hmm_forecast`, `hmm_states`, `scenario`, `analysis`, `run_all`, `run_development`, `run_outputs`가 있습니다. `tests/`는 모듈·누수·CLI 검사, `notebooks/`는 표시 전용, `results/`는 실행 산출물, `5. 자원 최적화 AI 데이터셋/`는 원본과 생성 CSV입니다. `PONYTAIL-DEBT.md`는 코드의 단순화·확장 조건 원장입니다.

`--package`는 `dist/kamp_power_src.zip`에 위 코드·검사·노트북·results·데이터 CSV와 requirements·README·pyproject·uv.lock·부채 원장만 포함합니다. 숨김 파일, 캐시, 가상환경, 기획 문서, 작업 메모, dist 자체는 제외합니다. 다른 `--out`의 결과는 자동으로 공식 results에 복사하지 않습니다. 실제 제출 ZIP은 승인된 최종 results가 준비된 뒤 생성하십시오.

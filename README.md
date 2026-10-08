# KAMP 공장 전력 예측·최대피크 위험조건 분석 (C-BAT)

제6회 K-인공지능 제조데이터 분석 경진대회 문제 ⑤ *제조 생산데이터 기반 전력사용량 예측 및 최대피크 위험조건 분석*.
원자료는 `5. 자원 최적화 AI 데이터셋/`에 있습니다.

## 바로 실행 (제출 예측 재현)

```bash
uv sync
uv run python main.py
```

- 2021-09-01 이전 자료로 최종 C-BAT를 적합하고 2021-09-01~14를 예측합니다. CPU에서 약 1분 걸리며 GPU는 필요 없습니다.
- 산출 위치는 `results_v3/submission/`입니다.
  - `test_predictions.csv`: 1,344행(14일 × 96슬롯)
  - `eval_mask.csv`: 결측 슬롯 표시
  - `test_metrics.csv`, `test_metrics_long.csv`, `test_slots.csv`, `test_days.csv`: 테스트 채점
  - `posterior_summary.csv`, `posterior_trace.csv`, `acceptance.csv`: 사후 표
  - `run_status.json`: 설정과 소스 해시
- 환경은 Python 3.13(`.python-version`)이고 Linux와 Windows에서 돌아갑니다. uv를 쓰지 않으면 `pip install -r requirements.txt`로 설치합니다. 의존성은 numpy·polars·lightgbm·matplotlib뿐이며 GPU가 필요 없습니다.
- 외부자료 취득, 네트워크 호출, 자격증명은 실행에 필요하지 않습니다.

### `test_predictions.csv` 열 (32개)

| 열 | 의미 |
|---|---|
| `datetime` | 슬롯 시작 시각 `YYYY.MM.DD HH:MM:SS` |
| `model` | `BAT_conditional_gaussian_fixed_attention` |
| `y_mean`, `y_median` | 슬롯 전력 예측 경로의 평균과 중앙값(kW). 점예측은 `y_median`입니다 |
| `q05` … `q95` | 5%p 간격 예측 분위수 19개(`q50` = `y_median`) |
| `M_hat_median`, `M_hat_mean` | 그날 일 피크(경로 최댓값) 분포의 중앙값과 평균. 날짜별 값을 96행에 반복합니다 |
| `peak_time_mode` | 경로별 피크 시각의 최빈값(HH:MM) |
| `risk_C50/C75/C90` | P(일 피크 > C). 보정하지 않은 경로 초과확률이며 Platt 보정을 쓰지 않습니다([근거](document/RISK_issuance_preregistration.md)) |
| `C50/C75/C90` | 피크 기준값: 학습 구간 가동일 피크의 50·75·90% 분위수(kW) |

## 최종 모델: C-BAT

`BAT_conditional_gaussian_fixed_attention`(`gmst/conditional_bat.py`)입니다. `ConditionalConfig()` 기본값이 곧 최종 설정입니다.

- 잡음: 가동/비가동 상태별 조건부 가우시안
- 시간 커널: 고정. 폭 1.5시간, 모양 2, 지연 0
- 기준일 규칙: `ref="op"`(B0′)
- 반감기 30일 시간 가중
- 최종 실행: 6,000회 표집, 앞 3,000회 버림

날짜 $d$, 15분 슬롯 $t=1,\dots,96$, 생산계획 시각 $h=0,\dots,23$, 가동유형 $k\in\{0,1,2,3\}$(하루 생산시간 0 / 1–12 / 13–19 / 20시간 이상)에 대해 모형은 다음과 같습니다.

$$
y_{d,t} = \underbrace{\mu_{k,t}}_{\text{기본곡선}} + \underbrace{\gamma_k\, r_{d,t}}_{\text{최근 기준일}} + \underbrace{\sum_{c\in\{\mathrm{on},\log\}} w_{c,k} \sum_{h=0}^{23} \alpha_{t,h}\, x_c(q_{d,h})}_{\text{생산 전달항}} + \underbrace{u_d}_{\text{날짜 효과}} + \underbrace{e_{d,t}}_{\text{슬롯 잡음}}
$$

$$
\alpha_{t,h} = \mathrm{sparsemax}_h\!\left(-\frac{\lvert \Delta_{t,h}-c\rvert^{\beta}}{\sigma}\right),\quad
\Delta_{t,h}=\frac{t-1-4h-1.5}{4},\quad (\sigma,\beta,c)=(1.5,2,0)
$$

$$
u_d \sim \mathcal N(0,\sigma^2_{u,g}),\qquad e_{d,t}=\rho_g\, e_{d,t-1}+\eta_{d,t},\quad \eta_{d,t}\sim\mathcal N(0,\sigma^2_{\eta,g}),\qquad g\in\{\text{비가동},\text{가동}\}
$$

| 항 | 설명 |
|---|---|
| $\mu_{k,t}$ | 가동유형별 하루 곡선. 순환 RW2 평활 사전분포 |
| $r_{d,t}$, $\gamma_k$ | 예측일 이전 28일 안에서 가동 여부와 날짜유형(평일·토·일)이 같은 완전 관측일 중 가장 최근 날의 곡선, 그리고 그 곡선을 따르는 정도 |
| $x_{\mathrm{on}}, x_{\log}$ | 생산 여부 $\mathbf 1[q>0]$, 정규화 로그 생산량 $\log(1+q)/\log(1+q_{\max})$ |
| $w_{c,k}=e^{\omega_{c,k}}$ | 양수 생산 이득. 생산이 늘면 예측 전력이 줄지 않습니다. 비가동일에는 전달항이 없습니다 |
| $\alpha_{t,h}$ | 시간 계획을 15분 슬롯으로 옮기는 고정 sparsemax 커널. 각 슬롯은 인접한 한두 시각만 봅니다 |
| $u_d$, $e_{d,t}$ | 가동 상태별 날짜 효과와 AR(1) 슬롯 잡음. 결측 간격만큼 자기상관을 거듭 적용합니다 |

- **추정**: 기본곡선과 기준일 계수는 정규 조건부분포에서 한꺼번에 뽑습니다. 생산 이득은 주변우도 기반 랜덤워크 Metropolis로 갱신합니다. 날짜 효과, 분산, 자기상관은 차례로 갱신합니다.
- **예측**: 사후표본마다 평균 곡선에 새 날짜 효과와 AR(1) 잡음을 더해 하루 경로를 만들고, 음수는 0으로 자릅니다. 분위수, 일 피크 분포, 초과확률은 모두 같은 경로에서 계산합니다.
- 자세한 사전분포와 수렴 진단은 보고서 원고 `report/chapters/02_method.tex`와 부록 B(`report/chapters/B_appendix.tex`)에 있습니다.

입력은 당일 시간별 생산계획, 달력, d일 이전 관측 전력뿐입니다. 데이터의 생산량은 실적이지만 사전에 알려진 생산계획으로 가정했고, 이 가정은 보고서에 적었습니다.

## 결과

### 개발 구간 f1–f4 (7–8월, 53일 · 5,016슬롯 · 피크 51일)

출처는 `results_v3/report_tables/summary.md`(T1)입니다. 확률 모델은 seed 0–2 평균이고 단위는 kW입니다.

| 모델 | 슬롯 MAE | RMSE | R² | CRPS | 일 피크 MAE | 피크 R² |
|---|---:|---:|---:|---:|---:|---:|
| **C-BAT** | 8.50 | **12.37** | **0.960** | **6.96** | 8.93 | 0.976 |
| BAT (Laplace, 학습형 커널) | **7.82** | 12.61 | 0.959 | 6.99 | 14.50 | 0.952 |
| B0_kind (최근 동일 가동유형일) | 8.51 | 14.98 | 0.942 | – | **6.67** | **0.983** |
| LightGBM(튜닝) | 11.05 | 19.78 | 0.898 | 8.65 | 10.06 | 0.946 |
| B0 (지난주 같은 요일) | 26.84 | 52.62 | 0.280 | – | 42.22 | −0.058 |

- 고피크일 13일(관측 피크 > C90)의 피크 MAE는 C-BAT 4.37, B0_kind 7.54, BAT 12.47, LightGBM(튜닝) 18.27 kW입니다(T2).
- 전체 일 피크 MAE는 단순 기준선 B0_kind가 가장 작습니다. 슬롯 MAE는 기존 BAT가 가장 작습니다.

### 9월 테스트 (2021-09-01~14, 관측 1,342슬롯)

출처는 `document/CBAT_september_test.md`입니다.

| 모델 | 슬롯 MAE | RMSE | R² | CRPS | 90% 구간 적중 | 일 피크 MAE |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 7.877 | 11.183 | 0.9615 | – | – | 7.000 |
| M2 (LightGBM 기본) | **6.600** | 9.751 | 0.9708 | **5.026** | 0.872 | **5.526** |
| BAT | 6.824 | **9.533** | **0.9721** | 5.512 | 0.948 | 7.979 |
| C-BAT | 7.184 | 9.789 | 0.9705 | 5.709 | 0.984 | 8.934 |

- 9월에는 M2가 슬롯 MAE, CRPS, 일 피크 MAE에서 C-BAT보다 좋았습니다. C-BAT의 90% 구간은 필요보다 넓었습니다.
- 9월 14일 중 C90 초과일은 0일이어서 고피크 성능은 이 구간에서 확인할 수 없습니다.
- 9월 구간은 이전 개발 단계(v2)에서 한 번 열린 적이 있으므로 오염되지 않은 홀드아웃이 아닙니다. 모델 선택은 f1–f4 결과로만 했고, 최종 C-BAT는 동결 후 9월을 한 번만 예측했습니다.

## 4장: 월 최대(청구 수요) 갱신 경보

출처는 `document/BILLING_alarm_preregistration.md`입니다. 사전 등록 뒤 실행한 백테스트이며 시뮬레이션이 아닙니다.

- 요금시간대 슬롯에서 그날 00:00에 알려진 기준선 $B_d$를 정합니다. $B_d$는 이번 달 누적 최대와 과거 래칫 월 최대 중 큰 값입니다.
- C-BAT 경로로 $P(M_d > B_d)$와 기대 초과 기본요금을 계산하고, 기대 초과 비용이 3만 원 이상이면 경보합니다(주 규칙).
- 검증 대상 44일 중 월 최대 갱신일은 **7/19 하루**였습니다. 기준선 206 kW, 관측 222 kW, 초과 16 kW, 그 달 기준 133,120원입니다.
  - C-BAT: 7/19를 잡았습니다(P = 0.58, 경보 10회 중 오경보 9회).
  - B0_kind: 기준일 피크 195 kW로 놓쳤습니다.
- 133,120원은 완벽 조치를 가정한 **상한**입니다. 갱신일이 1일뿐이라 성능 추정이 아니라 사례로 봐야 합니다. 조치 효과는 추정하지 않았습니다.
- 생산계획을 옮겨 얻는 모델 반사실 절감액은 관측 유사일 검증에서 지지되지 않아 주장하지 않습니다(`document/CBAT_counterfactual_validation.md`).

## 기타 재현 명령

모든 명령은 저장소 루트에서 실행합니다. 모든 모듈은 스레드 1개 환경변수(`OMP_NUM_THREADS=1` 등)에서 돌리는 것을 권장합니다.

```bash
uv run pytest -q                                   # 검사
uv run python -m gmst.leakage_v3                   # 1장 복사일 누수 진단(무작위 5-fold vs LOEO) → results_v3/ch1/
uv run python -m gmst.robustness --out results_v3/robustness          # 기준선·BAT·튜닝 LightGBM 개발 구간 예측
uv run python -m gmst.run_conditional --attention-ablation --source results_v3/robustness \
  --n-iter 6000 --burn 3000 --out results_v3/attention_ablation       # C-BAT vs 학습형 커널
uv run python -m gmst.run_conditional --ref-comparison --source results_v3/robustness \
  --n-iter 6000 --burn 3000 --out results_v3/ref_ablation             # 기준일 규칙 비교
uv run python -m gmst.run_conditional --noise-comparison --source results_v3/robustness \
  --n-iter 6000 --burn 3000                                           # 잡음 변형 비교
uv run python -m gmst.rescore --source results_v3/attention_ablation  # R²·WAPE·피크 RMSE 등 재채점
uv run python -m gmst.peak_conditions --source results_v3/attention_ablation \
  --model BAT_conditional_gaussian_fixed_attention --out results_v3/peak_conditions_v2   # 3장 피크 조건
uv run python -m gmst.analog_days                  # 모델 없는 관측 유사일 비교(시작 시각만 다른 가동일)
uv run python -m gmst.billing_alarm                # 4장 월 최대 갱신 경보 백테스트
uv run python -m gmst.pooled_eval                 # 4장 시간순 검증 전체(개발+9월) 평가
uv run python -m gmst.report_tables                # 보고서 표 → results_v3/report_tables/
uv run python -m gmst.report_figures               # 보고서 그림 → report/figs/
```

- 플래그 없이 `gmst.run_conditional`을 실행하면 C-BAT만 f1–f4 × seed 0/1/2로 학습해 `results_v3/conditional_fixed/`에 씁니다.
- `--final`은 9월을 여는 실행입니다. `main.py`가 같은 경로(`run_final`)를 씁니다.
- 모든 모델은 as-of 래퍼를 거치므로 d일 예측에서 d일 이후 전력값을 보지 않습니다.

## 데이터 규칙 (요약)

| 규칙 | 처리 |
|---|---|
| 완전 복사일 | 전력 96칸이 소수점까지 같은 날(`okm_augumented`)은 원본 1개만 남기고 목표·기준일에서 마스킹 |
| 07-13·07-15 | 소실 시각을 추측하지 않고 목표·생산량 마스킹. 가동일로는 유지 |
| 전력 0 | 정전·계측 손실로 결측 처리. 관측 슬롯끼리만 평가하고, 일 피크는 결측 4슬롯 이하인 날만 사용 |
| 누수 열 | `공장인원`, `평균`은 항등식 감사 후 제거 |
| 생산량 | 사전 생산계획으로 가정해 입력으로 사용 |
| 봉인 | 9월 1–14일 목표는 기본 로더에서 NaN. `--final`/`main.py`에서만 열림 |
| 요금 | 한국전력 산업용(을) 고압A 2021 요금표, 일요일·공휴일 경부하, 월별 래칫 |
| 공휴일 | 한국천문연구원 특일정보(2021년) 8일 |

도메인 용어는 `CONTEXT.md`에 있습니다.

## 모듈

| 구분 | 파일 |
|---|---|
| 데이터 | `contracts`, `preprocess`, `splits`, `features`, `lgbm_features` |
| 기준선 | `baselines`(B0, B0_kind, M2), `backbone`, `benchmark_models` |
| 모델 | `conditional_bat`(C-BAT), `conditional_ar`, `bat`(기존 BAT) |
| 평가 | `evaluate`, `analysis`, `conditional_diagnostics`, `rescore`, `robustness`, `leakage_v3` |
| 피크·요금 | `peak_conditions`, `plan_sensitivity`, `analog_days`, `billing_alarm`, `pooled_eval`, `scenario`(요금) |
| 러너 | `run_conditional`(C-BAT, `--final`), `run_v3`(기존 BAT·M2 파이프라인), `main.py` |
| 보고 | `report_tables`, `report_figures`, `report_v3`, `paper_figures` |

## 폴더와 제출물 구성

- `report/`: 제출 보고서 원고. 제출 PDF는 공식 hwpx 양식으로 작성합니다.
- `document/`: 사전 등록과 실험 기록
- `paper/`: 이전 연구 초안입니다. 현재 원고가 아니며 최종 결과와 다를 수 있습니다.
- `notebooks/`: 탐색용 노트북
- `results_v3/`: 실행 산출물. 제출 예측은 `results_v3/submission/`에 있습니다.

**소스 압축본 포함 항목:** `gmst/`, `tests/`, `main.py`, `pyproject.toml`, `uv.lock`, `requirements.txt`, `README.md`, `5. 자원 최적화 AI 데이터셋/`, `results_v3/submission/`

**제외 항목:** `.env`, `.venv`, `.git`, `.claude`, 노트북 소켓 파일, 대용량 절제 실험 결과 폴더(`results_v3/`의 나머지)

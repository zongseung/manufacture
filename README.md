# KAMP 공장 전력 예측·피크 저감 (v3: 베이지안 어텐션 전달모형)

제6회 K-인공지능 제조데이터 분석 경진대회 문제 ⑤ *제조 생산데이터 기반 전력사용량 예측 및 최대피크 위험조건 분석*.

## 개요

매일 **00:00**에 당일 96개 15분 슬롯의 전력을 예측합니다. 입력은 **당일 시간별 생산계획**, 달력(요일·공휴일·가동 여부), 그리고 d일 이전의 관측 전력뿐입니다. 생산량은 데이터상 실적이지만 사전에 알려진 생산계획으로 가정합니다. 이 가정은 보고서에 명시합니다.

제안 모델 **BAT(Bayesian Attention Transfer)** 는 날짜 $d$의 슬롯 $t\,(=1,\dots,96)$ 전력을 다음과 같이 둡니다.

$$
y_{d,t} \;=\; \mu_{k,t} \;+\; \gamma_k\, r_{d,t} \;+\; \sum_{c\in\{\mathrm{on},\,\log\}} w_{c,k} \sum_{h=0}^{23} \alpha_{t,h}(\theta_k)\, x_c(q_{d,h}) \;+\; \varepsilon_{d,t},
\qquad \varepsilon_{d,t} \sim \mathrm{Laplace}(0,\, b)
$$

어텐션은 슬롯 $t$와 생산 시각 $h$의 시간 거리 $\Delta_{t,h}$(시간 단위)에 대한 일반화 가우시안 점수를 sparsemax로 정규화합니다.

$$
\alpha_{t,h}(\theta) \;=\; \operatorname{sparsemax}_h\!\left(-\,\frac{\lvert \Delta_{t,h} - c \rvert^{\beta}}{\sigma}\right),
\qquad \theta = (\sigma,\, \beta,\, c)
$$

생산 채널은 가동 여부와 로그 생산량입니다.

$$
x_{\mathrm{on}}(q) = \mathbf{1}[\,q > 0\,], \qquad
x_{\log}(q) = \frac{\log(1+q)}{\log(1+q_{\max})}, \qquad
w_{c,k} = e^{\omega_{c,k}} > 0
$$

| 기호 | 의미 |
|---|---|
| $k$ | 생산계획에서 정한 가동유형 (하루 생산시간 0 / 1–12 / 13–19 / 20시간 이상) |
| $\mu_{k,t}$ | 가동유형별 기본곡선. 순환 2차 확률보행(RW2) 평활 사전 |
| $r_{d,t}$ | 최근 같은 가동·날짜유형 완전일의 곡선 (B0′ 기준일) |
| $\alpha_{t,h}(\theta_k)$ | 슬롯 $t$가 생산 시각 $h$를 보는 희소 국소 어텐션. 폭 $\sigma$, 모양 $\beta$, 지연 $c$를 사후분포로 추정 |
| $w_{c,k}$ | 생산 이득. $w>0,\ \alpha\ge 0$ 이므로 생산이 늘면 예측 전력이 줄지 않음 (단조) |
| $b$ | Laplace 척도. 우도에는 반감기 30일의 시간 가중 $2^{-(d_{\max}-d)/30}$ 적용 |

**추정.** Laplace 오차를 정규–지수 척도 혼합으로 풀어 $(\mu, \gamma)$는 Gibbs(가중 정규방정식)로, $(\theta_k, \omega_k)$는 랜덤워크 Metropolis로 뽑습니다(2,000회, 앞 1,000회 버림).

**예측분포.** 사후 표본 $s$마다 평균곡선 $\hat y^{(s)}_{d,t}$에 날짜 공통 이동과 하루 안 AR(1) 잡음을 더해 경로를 만듭니다.

$$
\tilde y^{(s)}_{d,t} \;=\; \hat y^{(s)}_{d,t} + u^{(s)} + e^{(s)}_t,
\qquad u^{(s)} \sim \mathcal N\!\big(0, \sigma_u^{2}\big),
\qquad e^{(s)}_t = \rho\, e^{(s)}_{t-1} + \eta_t,\ \ \eta_t \sim \mathrm{Laplace}(0, b_\eta)
$$

점예측은 경로의 중앙값이고, 분위수·일 피크·피크 초과 확률 $P(\max_t \tilde y_{d,t} > C)$도 같은 경로에서 계산합니다. 전달행렬 $\partial \hat y / \partial q \in \mathbb R^{96\times 24}$는 해석(3장)과 생산 재배치(4장)에 씁니다.

비교 사다리: **B0**(지난주 같은 요일) → **M2**(생산계획 포함 LightGBM, `baselines.b1_model(protocol="B")`) → **BAT**.

## 환경

Python 3.13 이상, `uv`를 사용합니다.

```bash
uv sync
# uv를 쓰지 않는 경우
pip install -r requirements.txt
```

`requirements.txt`는 `uv export --no-hashes --no-dev --no-emit-project`로 만든 고정 버전입니다(PyTorch 인덱스는 cu126). BAT는 CPU에서 돌아가고 GPU가 필요 없습니다. 외부자료 취득, 네트워크 호출, 자격증명은 실행에 필요하지 않습니다.

## 실행

```bash
uv run python -m gmst.run_v3                  # 개발: f1–f4 검증 fold, 9월 봉인 유지 → results_v3/
uv run python -m gmst.run_v3 --quick --folds f1 --out /tmp/v3q   # 빠른 확인 (Gibbs 400회, 부스팅 20라운드)
uv run python -m gmst.run_v3 --final          # 9월 봉인 테스트를 열어 채점 → results_v3/final/
uv run python -m gmst.report_v3               # 3장 산출물 → results_v3/ch3/
uv run python -m gmst.report_v3 --no-peak-class   # 보조 지표(피크 이벤트 분류)를 뺄 때
uv run python -m gmst.leakage_v3              # 1장 LOEO 누수 진단, 약 9분 → results_v3/ch1/
uv run python -m gmst.realloc_v3              # 4장 생산 재배치 → results_v3/ch4/
uv run python -m gmst.realloc_v3 --quick      # 빠른 확인 (Gibbs 400회, 60스텝, fold당 2일)
uv run python -m gmst.realloc_v3 --shift      # 4장 분포 기반 시간대 이동 → results_v3/ch4/shift_*
```

- `run_v3`는 (fold, 모델) 조합마다 **스레드 1개짜리 프로세스**를 따로 띄워 병렬로 돌립니다. 작은 행렬(392×392)에서는 BLAS 다중 스레드가 오히려 느리기 때문입니다. 32코어 기준으로 개발 실행은 약 10–15분, `--final`은 약 1분 걸립니다.
- 모든 모델의 예측은 as-of 래퍼를 거칩니다. d일 예측에서는 d일 이후의 전력값이 보이지 않습니다.
- `realloc_v3`는 fold마다 BAT를 한 번 학습한 뒤(프로세스 4개), 검증기간 가동일 39일 × ρ 3개 × 인건비 가중 w 4개 = 468개 재배치를 풉니다. 약 40분 걸립니다.
- `report_v3`는 fold별 BAT를 다시 학습하고, 그 사후 draw를 `results_v3/ch3/bat_draws_f*.npz`에 캐시합니다. `bat.py`를 바꿨다면 캐시를 지우고 다시 실행하십시오.

## 산출물과 보고서 장

| 파일 | 장 / 용도 |
|---|---|
| `results_v3/ch1/leakage_gap.csv`, `loeo_days.csv`, `leakage_gap.png`, `summary.md` | 1 / 복사일 포함 자료에서 무작위 5-fold vs LOEO 누수 격차 |
| `results_v3/oof_slots.csv`, `oof_days.csv`, `inner_days.csv` | 2 / fold별 슬롯·일 예측, Platt 보정용 내부일 |
| `results_v3/metrics.csv`, `summary.csv`, `bootstrap.csv` | 2 / 지표, 날짜 짝짓기 부트스트랩 CI |
| `results_v3/ch3/error_by_regime.csv` | 3 / 가동유형 × 생산 on/off, 요금 시간대, 시각별 오차 |
| `results_v3/ch3/peak_events.csv`, `fn_fp_conditions.csv` | 3 / [보조 지표] 피크 초과 분류 F1과 "매일 경보" 기준선, 미탐지·오경보 조건 |
| `results_v3/ch3/attention_k*.csv/png`, `transfer_gain.csv`, `attention_shape.csv` | 3·5 / 어텐션 지도와 생산 이득의 사후분포 |
| `results_v3/ch3/forecast_val.png`, `forecast_test.png`, `pred_vs_actual.png` | 2 / 검증 fold 대표일 예측, 봉인 테스트 14일 예측, 예측값–실측 산점도 |
| `results_v3/ch3/summary.md` | 3 / 핵심 수치 요약 |
| `results_v3/final/test_predictions.csv`, `test_metrics.csv`, `eval_mask.csv` | 2·6 / 9월 봉인 테스트 예측과 채점 |
| `results_v3/ch4/realloc_days.csv` | 4 / 날짜 × ρ × w별 재배치 전후 피크, 사후확률 P(피크 감소), 권고 여부, 요금 변화 |
| `results_v3/ch4/pareto.csv`, `realloc_example_pareto.png` | 4 / 인건비–전력요금 절충 곡선, 재배치 예시 |
| `results_v3/ch4/summary.md` | 4 / 핵심 수치 요약 |
| `results_v3/ch4/shift_days.csv`, `shift_summary.md` | 4 / 날짜 × σ × 후보 계획별 사후 판정, 선택된 시간대 이동 계획 |

`test_predictions.csv`는 UTF-8, 1,344행(14일 × 96슬롯)입니다. 열: `datetime`(YYYY.MM.DD HH:MM:SS), `model`, `y_mean`, `y_median`, `q05`…`q95`, `M_hat_median`, `M_hat_mean`, `peak_time_mode`(HH:MM), `risk_C50/C75/C90`(Platt 보정), `C50/C75/C90`. `eval_mask.csv`는 결측 슬롯 표시입니다.

## 결과

| 구간 | 지표 | BAT | M2 | B0 |
|---|---|---|---|---|
| 검증 f1–f4 (7–8월, 56일) | MAE | **7.82** | 11.64 | 26.84 |
| | CRPS | **6.99** | 8.96 | — |
| | 일 피크 MAE | **14.5** | 17.0 | 42.2 |
| 봉인 테스트 (9/1–9/14) | MAE | 6.82 | **6.60** | 7.88 |
| | RMSE | **9.53** | 9.75 | 11.18 |

- 검증 BAT − M2의 날짜 짝짓기 부트스트랩 95% CI: MAE [−6.35, −1.62], CRPS [−3.87, −0.31]
- 모델 선택은 검증 fold로만 했습니다. 봉인 테스트는 사후 확인입니다. 9월 구간은 v2 때 한 번 열린 적이 있습니다.
- 복사일을 포함해 학습해도 BAT는 MAE 7.59로 거의 같습니다. M2는 13.76으로 나빠집니다.
- 절제 실험: 잠재 날짜효과를 평균식에 넣으면 MAE 19.0(f1), 3차 잔차 보정(M4)을 더하면 10.2, 어텐션을 빼면 MAE 차이는 +0.7%입니다. 다만 어텐션이 없으면 전달행렬과 재배치 계산이 불가능합니다.

## 생산 재배치 (4장)

두 전략 모두 BAT 사후 표본으로 새 계획 $q'$의 일 피크를 계산해, 다음 조건을 만족할 때만 권고합니다.

$$
P\!\left(\max_{t\in\mathcal D} \hat y_t(q') \;<\; \max_{t\in\mathcal D} \hat y_t(q)\right) \;\ge\; 0.95
$$

$\mathcal D$는 기본요금 산정 대상인 중간·최대부하 시간대 슬롯입니다.

### 1) 생산량 재배치 (`gmst/reallocate.py`, `python -m gmst.realloc_v3`)
가동 시각은 그대로 두고 생산량만 옮깁니다.

$$
\min_{q'}\;\; \lambda_D\, \mathrm{LSE}_\tau\!\big(\hat y(q')_{\mathcal D}\big) \;+\; \sum_{t} 0.25\, p_t\, \hat y_t(q') \;+\; w \sum_{h} \ell_h\, q'_h
$$

$$
\text{s.t.}\quad \sum_h q'_h = \sum_h q_h,\qquad
0 \le q'_h \le \bar q_h,\qquad
q'_h = q_h\ \ (h \notin \text{가동 구간}),\qquad
\tfrac12\lVert q' - q \rVert_1 \le \rho \sum_h q_h
$$

- $\mathrm{LSE}_\tau(x) = \tau \log \sum_t e^{x_t/\tau}$는 미분 가능한 최대값, $\lambda_D$는 기본요금/30(원/kW·일), $p_t$는 시간대별 전력량 단가, $\ell_h$는 인건비 할증 배수(1.0/1.5), $\bar q_h$는 학습기간 시간대별 95분위입니다.
- 풀이: Adam 경사 단계 뒤마다 Dykstra 교대 투영(박스·초평면 $\cap$ $L_1$ 공). 새 계획이 원래보다 비싸면 원래 계획을 돌려줍니다.

### 2) 분포 기반 시간대 이동 (`python -m gmst.realloc_v3 --shift`)
가동 시각 자체를 옮깁니다. 용어는 `CONTEXT.md`를 따릅니다. 학습 fold마다 다음 분포를 추정합니다.

$$
S \mid k \;\sim\; \mathrm{Categorical}(\pi_k),\qquad \pi_k \sim \mathrm{Dirichlet}(0.5\cdot\mathbf 1 + n_k)
$$

$$
q_h \mid q_h>0,\ \text{시간대}\ b \;\sim\; \mathrm{Gamma}(a_b,\, \lambda_b),\qquad b \in \{\text{주간 07–20시},\ \text{야간}\}
$$

$S$는 가동 블록의 시작 시각, $n_k$는 가동유형 $k$의 시작 시각 관측 빈도입니다. 시간당 생산량의 KS 통계량은 Gamma 0.043, 로그정규 0.095, 지수 0.133입니다.

- **후보 계획**
  - 블록 전체를 $\pm1$–$2$시간 이동. 새 시작 시각의 사후예측확률 $P(S=s' \mid k) \ge 0.05$
  - 단가가 높은 시각의 생산을 더 싼 인접 가동 시각으로 이동. 이동 후 $q'_h \le F^{-1}_{\mathrm{Gamma}(a_b,\lambda_b)}(0.95)$
- **실행 오차:** $q^{\mathrm{real}}_h = q'_h \cdot \exp(\sigma z_h),\ z_h \sim \mathcal N(0,1)$, $\sigma \in \{0.05, 0.1, 0.2\}$로 민감도 확인
- **선택:** 권고 조건을 통과한 후보 중 기대 요금이 가장 낮은 계획

### 결과 (검증기간 가동일, 모델 기반 반사실 추정)

| 항목 | 생산량 재배치 (ρ=0.2, w=0) | 시간대 이동 (σ=0.1) |
|---|---|---|
| 대상 가동일 | 39 | 38 |
| 권고일 | 10 | **20** |
| 피크 감소 중앙값 / 최댓값 | 0.1 / 1.0 kW | **3.4 / 9.8 kW** |
| 요금 변화 중앙값 | −1,194원/일 | **−5,235원/일** (월 22일 가동 기준 약 −11.5만 원) |
| 인건비 지수 | 1.000 | 1.000 |

- 시간대 이동의 권고일은 σ = 0.05 / 0.1 / 0.2에서 20 / 20 / 19일로, 실행 오차에 거의 영향받지 않습니다.
- 권고 20건 중 19건은 **11시 생산을 12시로 이동**하는 계획입니다. 여름철 11시는 최대부하(191.1원/kWh), 12시는 중간부하(109.0원/kWh)이고, 두 시각 모두 인건비 1.0배 구간입니다. 12시가 휴게 시간이라면 교대 운영을 바꿔야 적용할 수 있습니다.
- 생산량 재배치의 효과가 작은 이유: 전력은 생산량보다 가동 여부(on/off)에 주로 반응하는데(가동 이득 13–16 kW), 이 전략은 가동 시각을 고정합니다.
- 7–8월 래칫 바닥은 222 kW이고 검증기간 피크는 187–201 kW라서, 이 기간의 절감은 대부분 전력량요금입니다.

## 복사일 누수 진단 (1장, `gmst/leakage_v3.py`)

복사일을 포함한 자료(9월 봉인 유지)에서 두 교차검증을 비교합니다. 날짜 단위 무작위 5-fold와 LOEO(복사 그룹 전체를 한 번에 제외, 126개 그룹)입니다. 격차는 같은 날끼리 짝지은 부트스트랩 95% CI로 냅니다.

| 모델 | 무작위 5-fold | LOEO | 격차 (무작위 − LOEO) |
|---|---|---|---|
| BAT | 16.37 | 16.32 | +0.05 [−0.22, +0.42] |
| M2 | 14.91 | 16.64 | **−1.73 [−3.63, −0.04]** |

- 무작위로 나누면 시험일의 복사본이 학습에 섞여서 M2 성능이 약 10% 과대평가됩니다. BAT는 영향을 받지 않습니다.
- BAT는 진단용으로 Gibbs 1,000회(버림 500회)만 돌립니다. 본 실험은 2,000/1,000회입니다.

## 데이터 규칙

| 규칙 | 처리 |
|---|---|
| DC1 완전 복사 | 전력 96칸이 소수점까지 같은 날은 증강 복사로 판정합니다(같은 쌍의 기온·습도는 0% 일치, 파일명 `okm_augumented`). 원본 1개만 남기고 목표·기준일에서 마스킹합니다. 학습기간 186일 중 121일이 여기에 해당합니다 |
| DC2 07-13·07-15 | 소실된 시각을 추측하지 않고 목표·생산량을 마스킹합니다. 가동일로는 유지합니다 |
| DC3 전력 0 | 정전·계측 손실로 결측 처리합니다. 관측 슬롯끼리만 평가하고, 일 최대는 결측 4슬롯 이하인 날만 씁니다 |
| DC4 풍속 결측 | 시간 단위 선형보간 후 4슬롯 반복 |
| DC5 누적 강수 | 자정을 가로지르는 차분, 음수는 리셋, 원 결측 증분은 0 |
| DC6 누수 열 | `공장인원`, `평균`은 항등식 감사 후 제거 |
| DC7 원자료 계절값 | 계절 지시로만 쓰고 요금 단가로 쓰지 않습니다 |
| DC8 생산량 | **사전 생산계획으로 가정해 BAT·M2의 입력으로 씁니다** |
| DC9 중복 purge | 학습·검증 사이의 복사 그룹 겹침을 제거 |
| DC10 봉인 | 9월 1–14일 Y·X는 기본 로더에서 NaN. `--final`에서만 엽니다 |
| DC11 요금 규칙 | 일요일·공휴일은 경부하, 토요일 최대부하 에너지는 중간단가, 월별 래칫 |
| DC12 가동 여부 | 일 생산량 양수 또는 suspect |
| DC13 공휴일 | 휴무와 같다고 보지 않고 가동·요일과 별도로 씁니다 |
| DC14 편집 복사 | 가동구간 동일값 20개 이상이면 편집 복사로 보고 마스킹합니다(전이적 병합 없음) |
| DC15 시각 | 구간 시작 라벨: 15분→HH:00, 60분→HH:45 |

## 외부자료와 가정

- `HOLIDAYS_2021`은 **한국천문연구원 특일정보(2021년)**에서 확정한 8일입니다.
- 요금은 **한국전력 산업용(을) 고압A 2021 요금표**(선택 Ⅰ/Ⅱ/Ⅲ, 기본 Ⅱ)를 코드에 옮긴 것입니다. 08-16은 요금상 공휴일로 가정합니다.
- 인건비 할증은 데이터의 `인건비` 열을 따릅니다(가동 평일 09–18시 1.0, 그 외 1.5, 근로기준법 제56조와 일치).
- 원 단위 계산은 **15분 평균 kW** 가정하의 값입니다.
- 외부 기상 자료는 쓰지 않습니다. 일 피크와 기온의 상관은 −0.005입니다.

## 재현성·검사

seed를 고정하고 날짜별 난수를 씁니다. 같은 장치에서 같은 입력이면 같은 결과가 나옵니다.

전체 재현 순서(32코어 기준 약 1시간):

```bash
uv sync                                   # 1. 환경
uv run pytest -q                          # 2. 검사
uv run python -m gmst.run_v3              # 3. 검증 f1–f4, 약 10–15분 → results_v3/
uv run python -m gmst.run_v3 --final      # 4. 9월 봉인 테스트, 약 1분 → results_v3/final/
uv run python -m gmst.report_v3           # 5. 3장 산출물 → results_v3/ch3/
uv run python -m gmst.realloc_v3          # 6. 4장 재배치, 약 40분 → results_v3/ch4/
uv run python -m gmst.realloc_v3 --shift  # 7. 4장 시간대 이동, 약 3분 → results_v3/ch4/shift_*
cd paper && tectonic -X compile main.tex  # 8. 보고서 PDF (XeLaTeX, Noto CJK KR 폰트)
```

## 폴더 구조

- `gmst/`
  - 데이터: `contracts`, `preprocess`, `splits`, `features`, `lgbm_features`
  - 기준선: `baselines`(B0, M2), `backbone`(M2의 τ·반감기 선택)
  - 제안 모델: `bat`
  - 평가: `evaluate`, `analysis`
  - 요금·재배치: `scenario`(요금), `reallocate`(재배치 풀이), `realloc_v3`(4장 실행)
  - 러너·리포트: `run_v3`, `report_v3`
- `tests/`: 모듈 검사
- `results_v3/`: 실행 산출물
- `5. 자원 최적화 AI 데이터셋/`: 원본 데이터와 생성된 CSV
- `paper/`: 보고서 원고(한국어, AAAI 2026 2단 양식, XeLaTeX)
- `CONTEXT.md`: 도메인 용어집

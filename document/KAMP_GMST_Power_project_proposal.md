# 제조 생산데이터 기반 확률적 전력예측 및 최대피크 위험관리 AI 모델 개발 기획서

## 1. 과제명

**구조적 시간사전과 조건부 상태전이를 결합한 제조 전력사용량 예측 및 최대피크 위험관리 모델 개발**

영문명(안):

**GMST-Power: GMRF-Regularized Markov Switching Transformer for Probabilistic Manufacturing Power Forecasting**

---

## 2. 추진 배경

제조업의 전력사용량은 단순한 시간적 주기만으로 결정되지 않는다. 생산량 변화, 작업시간, 계절, 기상조건 및 여러 생산조건의 상호작용에 따라 짧은 시간에도 부하 수준이 크게 변화할 수 있다. 특히 다수의 공정이 동시에 높은 전력을 사용하는 경우 최대수요전력이 급격하게 상승할 수 있으며, 이는 전력비용뿐 아니라 생산운영 안정성에도 영향을 미친다.

따라서 제조 전력 예측 문제에서는 단순히 미래 전력사용량의 평균값을 정확하게 예측하는 것만으로는 충분하지 않다. 실제 현장 활용을 위해서는 다음 세 가지 문제를 동시에 해결할 필요가 있다.

1. 향후 지정된 시간구간의 전력사용량을 정확하게 예측한다.
2. 높은 예측오차 및 최대전력 피크가 발생하는 조건을 정량적으로 분석한다.
3. 생산조건 또는 생산일정 변화에 따른 미래 피크위험의 변화를 예측하여 운영 의사결정을 지원한다.

본 과제에서는 이를 위해 **구조적 통계모형, 딥러닝 시계열 모델, 조건부 Markov 상태전이, 확률예측을 통합한 하이브리드 모델**을 개발한다.

---

## 3. 연구 목표

본 연구의 최종 목표는 제조 전력 시계열의 안정적인 주기구조와 비선형 생산변동을 동시에 고려하고, 미래 부하상태의 전이를 명시적으로 추정하여 전력사용량뿐 아니라 최대피크 발생위험을 확률적으로 예측하는 것이다.

핵심 연구 질문은 다음과 같다.

> **미래 생산 및 시간조건에 따라 변화하는 잠재 부하상태의 전이를 추정하고, 구조적 시간패턴과 상태별 전력분포의 불확실성을 함께 고려하여 미래 전력사용량과 최대피크 위험을 동시에 예측할 수 있는가?**

세부목표는 다음과 같다.

- 15분 해상도의 제조 전력 시계열 구축
- 데이터 누수와 이상값을 고려한 전처리 체계 구축
- GMRF 기반 구조적 시간패턴 추정
- 외생변수를 고려하는 경량 Transformer 기반 비선형 표현 학습
- 생산조건에 따라 변화하는 Conditional Markov transition 학습
- 상태별 확률적 전력분포 추정
- 미래 구간 최대피크 초과확률 계산
- 예측오차 및 피크 발생조건 분석
- 생산계획 변화에 따른 피크위험 시나리오 평가

---

## 4. 데이터 구성 및 사전 진단

제공 데이터는 시간별 행에 15분, 30분, 45분, 60분의 네 전력값이 존재하는 구조이다. 따라서 시간별 행을 그대로 모델링하기보다 네 값을 시간 순서대로 펼쳐 **15분 단위 column-based time series**로 재구성한다.

시간행 \(r\)의 전력값을

\[
(y_{r,15},y_{r,30},y_{r,45},y_{r,60})
\]

라고 하면,

\[
y_{4r},y_{4r+1},y_{4r+2},y_{4r+3}
=
(y_{r,15},y_{r,30},y_{r,45},y_{r,60})
\]

의 순서로 변환한다.

### 4.1 주요 입력변수

| 구분 | 변수 예시 | 활용 |
|---|---|---|
| 과거 관측 변수 | 과거 전력, 생산량, 기온, 습도, 풍속, 강수량 | Encoder 입력 |
| 미래 확정 변수 | 시간대, 요일, 월, 계절, 전기요금 스케줄 | Forecast 입력 |
| 미래 조건부 변수 | 생산계획, 기상예보 | 실제 제공될 경우만 사용 |
| 사용 금지 변수 | 미래 실측 생산량, 미래 실측 기상 | Data leakage 방지 |

미래 생산량이 사전에 제공되는지 현재 명확하지 않으므로 두 가지 실험 프로토콜을 구성한다.

**Protocol A: Past-only Forecasting**

\[
\mathcal I_t
=
\{Y_{1:t},X_{1:t},C_{t+1:t+H}^{calendar}\}
\]

**Protocol B: Planned-covariate Forecasting**

생산계획이 사전에 제공될 경우

\[
\mathcal I_t
=
\{Y_{1:t},X_{1:t},C_{t+1:t+H}^{calendar},C_{t+1:t+H}^{production}\}.
\]

### 4.2 데이터 누수 방지

초기 데이터 진단 결과, 일부 파생변수는 미래 전력값과 직접적인 관계를 가지므로 예측 입력에서 제외한다.

특히 `공장인원`은 데이터상

\[
\text{공장인원}
\approx
\frac{\text{생산량}}
{\text{15분}+\text{30분}+\text{45분}+\text{60분}}
\]

의 관계를 갖기 때문에 미래 예측변수로 사용할 경우 목표변수 정보를 역산할 가능성이 있다.

또한 `평균`은 네 전력값으로부터 계산되는 파생변수이므로 미래 예측구간에서는 사용하지 않는다.

따라서 주 모델에서는 두 변수를 제외한 **Leakage-free configuration**을 사용한다.

---

## 5. 제안 방법론

### 5.1 전체 구조

\[
\boxed{
\text{GMRF Structural Model}
\rightarrow
\text{Deep Temporal Encoder}
\rightarrow
\text{Conditional Markov}
\rightarrow
\text{State-wise Distribution}
\rightarrow
\text{Peak-risk Prediction}
}
\]

각 모듈의 역할은 다음과 같다.

- **GMRF**: 장기 추세, 하루 주기 및 부드러운 시간구조
- **Deep Encoder**: 생산조건과 전력 사이의 비선형 관계
- **Conditional Markov**: 시간과 생산조건에 따른 부하상태 변화
- **Probabilistic Emission**: 상태별 전력수준과 불확실성
- **Peak Risk**: 미래 전체 시간구간에서 임계전력을 초과할 확률

---

## 6. 구조적 시간모형: GMRF

전력 시계열의 안정적인 시간구조를 다음과 같이 분리한다.

\[
\eta_t^{G}
=
\alpha
+
u_t
+s_{q(t)}
+w_{d(t),q(t)}
+\boldsymbol{\beta}^{\top}x_t.
\]

여기서

- \(u_t\): 장기 추세
- \(s_{q(t)}\): 하루 96개 quarter의 시간대 효과
- \(w_{d(t),q(t)}\): 요일-시간대 상호작용
- \(x_t\): 안전하게 사용할 수 있는 외생변수

이다.

latent field를

\[
\mathbf z
=
[\mathbf u^\top,\mathbf s^\top,\mathbf w^\top]^\top
\]

라고 하면,

\[
\boxed{
\mathbf z\mid\vartheta
\sim
\mathcal N\left(0,Q(\vartheta)^{-1}\right)
}
\]

의 Gaussian Markov Random Field를 사용한다.

장기 추세에는 RW2 구조를 적용한다.

\[
\Delta^2u_t
=u_t-2u_{t-1}+u_{t-2},
\]

\[
Q_u
=
\tau_uD_2^\top D_2.
\]

하루 주기는 23:45와 00:00이 서로 연결되도록 cyclic precision matrix를 구성한다.

\[
Q_{day}
=
\tau_sC_2^\top C_2.
\]

GMRF의 posterior 계산에는 INLA를 활용할 수 있으며, INLA는 모델 그 자체가 아니라 latent Gaussian model을 위한 Bayesian approximation engine으로 사용한다.

GMRF에서 얻은 posterior mean을

\[
\widehat m_t^G
=
E[Y_t\mid\mathcal D_{\mathrm{train}}]
\]

라고 정의하고 구조적 residual을

\[
\boxed{
r_t
=
Y_t-\widehat m_t^G
}
\]

로 계산한다.

---

## 7. Deep Temporal Encoder

GMRF가 설명하지 못하는 비선형 변동은 Transformer 기반 encoder가 학습한다.

본 연구에서는 외생변수가 중요한 제조 전력데이터의 특성을 고려하여 **TimeXer-inspired lightweight architecture**를 사용한다.

입력 window를 길이 \(L\)이라 하면

\[
\mathbf H_t
=
E_\theta\left(
r_{t-L+1:t},
Y_{t-L+1:t},
X_{t-L+1:t},
\widehat m_{t-L+1:t}^{G},
\widehat\sigma_{t-L+1:t}^{G}
\right)
\]

이고,

\[
\mathbf h_t
=
\operatorname{Pool}(\mathbf H_t)
\]

를 현재 생산 및 전력상태를 나타내는 latent representation으로 사용한다.

초기 구현에서는 데이터 크기를 고려하여

\[
d_{\text{model}}=128,
\qquad
N_{\text{layer}}=3,
\qquad
N_{\text{head}}=4
\]

수준의 소형 모델을 기본으로 한다.

---

## 8. 조건부 Markov 부하상태 모델

공장의 부하상태를 잠재변수

\[
S_t\in\{1,\ldots,K\}
\]

로 정의한다.

기본 모델에서는

\[
K=4
\]

를 사용하고 \(K=3,5,6\)을 제거실험으로 비교한다.

고정 transition matrix 대신 미래조건에 따라 변화하는 transition을 사용한다.

\[
\boxed{
A_{t+h|t}(i,j)
=
P\left(
S_{t+h}=j
\mid
S_{t+h-1}=i,\mathcal I_t
\right)
}
\]

전이 logit은

\[
\ell_{hij}
=
b_{ij}
+
g_\psi\left(
e_i,e_j,\mathbf h_t,c_{t+h},\widehat m_{t+h}^{G}
\right)
\]

로 정의하고,

\[
\boxed{
A_{t+h|t}(i,j)
=
\operatorname{softmax}_j(\ell_{hij})
}
\]

로 변환한다.

현재 상태 posterior를

\[
\pi_{t|t}
=
P(S_t\mid\mathcal I_t)
\]

라고 하면 미래 상태확률은

\[
\boxed{
\pi_{t+h|t}
=
\pi_{t+h-1|t}A_{t+h|t}
}
\]

로 계산한다.

---

## 9. 상태별 확률적 전력예측

각 상태에서는 서로 다른 전력분포를 갖도록 한다.

피크와 이상변동을 고려하여 기본 emission으로 Student-\(t\) 분포를 사용한다.

\[
Y_{t+h}
\mid
S_{t+h}=k,\mathcal I_t
\sim
t_{\nu_{h,k}}(\mu_{h,k},\sigma_{h,k}).
\]

location은

\[
\boxed{
\mu_{h,k}
=
\widehat m_{t+h}^{G}
+
f_{\mu,k}(\mathbf h_t,c_{t+h})
}
\]

로 정의한다.

scale은

\[
\sigma_{h,k}
=
\epsilon
+
\operatorname{softplus}[f_{\sigma,k}(\cdot)]
\]

로 두어 양수성을 보장한다.

최종 predictive distribution은

\[
\boxed{
p(Y_{t+h}\mid\mathcal I_t)
=
\sum_{k=1}^{K}
\pi_{t+h|t,k}
p_k(Y_{t+h})
}
\]

의 mixture distribution이 된다.

점예측값은

\[
\boxed{
\widehat Y_{t+h}
=
\sum_{k=1}^{K}
\pi_{t+h|t,k}\mu_{h,k}
}
\]

으로 계산한다.

---

## 10. 상태경로 학습

잠재상태에 사전 label을 생성하지 않고 conditional HMM forward algorithm을 이용하여 state path를 marginalization한다.

상태 \(k\)의 관측 likelihood를

\[
b_{h,k}
=
p(Y_{t+h}\mid S_{t+h}=k,\mathcal I_t)
\]

라고 하면,

\[
\widetilde\alpha_h
=
(\alpha_{h-1}A_h)\odot b_h
\]

이고,

\[
c_h
=
\sum_k\widetilde\alpha_{h,k},
\qquad
\alpha_h
=
\frac{\widetilde\alpha_h}{c_h}.
\]

따라서 전체 forecast horizon의 likelihood는

\[
\boxed{
\log p(Y_{t+1:t+H}\mid\mathcal I_t)
=
\sum_{h=1}^{H}\log c_h
}
\]

로 계산된다.

---

## 11. 최대피크 위험 확률

미래 \(H\)개 구간의 최대전력을

\[
M_{t,H}
=
\max_{1\le h\le H}Y_{t+h}
\]

라고 정의한다.

관리 임계값을 \(C\)라고 하면 관심확률은

\[
R_{t,H}(C)
=
P(M_{t,H}>C\mid\mathcal I_t)
\]

이다.

상태 \(k\)에 대한 emission CDF를

\[
F_{h,k}(C)
=
P(Y_{t+h}\le C\mid S_{t+h}=k,\mathcal I_t)
\]

로 정의하고

\[
B_h(C)
=
\operatorname{diag}[F_{h,1}(C),\ldots,F_{h,K}(C)]
\]

라 하면, 상태경로가 주어진 조건에서 future observation의 조건부 독립을 가정할 때

\[
\boxed{
P(M_{t,H}\le C\mid\mathcal I_t)
=
\pi_{t|t}
A_1B_1(C)
A_2B_2(C)
\cdots
A_HB_H(C)
\mathbf 1
}
\]

이다.

따라서 최종 위험확률은

\[
\boxed{
R_{t,H}(C)
=
1-
\pi_{t|t}
\prod_{h=1}^{H}[A_hB_h(C)]
\mathbf 1
}
\]

로 계산된다.

---

## 12. 생산일정 시나리오 평가

생산계획을 \(a\)라고 하면

\[
\mathbf c_{t+1:t+H}
\rightarrow
\mathbf c_{t+1:t+H}^{(a)}
\]

로 변경하고 다시 모델을 forward하여

\[
R_{t,H}^{(a)}(C)
\]

를 계산한다.

운영 목적함수는 예를 들어

\[
\boxed{
J(a)
=
E[M_{t,H}^{(a)}]
+
\lambda P(M_{t,H}^{(a)}>C)
+
\eta C_{\mathrm{change}}(a)
}
\]

로 정의할 수 있다.

비교 가능한 시나리오는 다음과 같다.

- 기존 생산계획
- 생산량의 일부 시간 이동
- 작업시간 분산
- 특정 고부하 구간 회피
- 피크 발생 예상시간 전후 생산부하 완화

단, 관찰자료만을 사용하는 경우 해당 분석은 **causal intervention 결과가 아니라 model-based what-if scenario**로 표현한다.

---

## 13. 예측오차 및 피크 조건 분석

### 13.1 상태별 오차

\[
E_k
=
E[|Y-\widehat Y|\mid S=k]
\]

를 계산하여 특정 latent regime에서 오류가 집중되는지 확인한다.

### 13.2 상태전환 구간 오차

\[
S_t\neq S_{t+1}
\]

인 transition boundary와 stable regime을 분리하여 성능을 비교한다.

### 13.3 생산조건별 오류

생산량, 기온, 시간대 및 요일 등의 구간별로

\[
RMSE(x\in\mathcal B_j)
\]

를 계산한다.

### 13.4 Peak condition

높은 전력 상태에 대한 posterior probability

\[
P(S_t=k_{\mathrm{peak}}\mid D)
\]

및 conditional transition

\[
P(S_{t+1}=k_{\mathrm{peak}}\mid S_t=i,X_t)
\]

의 변화를 분석하여 높은 peak-risk와 연관되는 조건을 도출한다.

---

## 14. 학습 목적함수

기본 목적함수는 sequence likelihood를 사용한다.

\[
\mathcal L_{\mathrm{NLL}}
=
-
\sum_n
\log p(Y_{t_n+1:t_n+H}\mid\mathcal I_{t_n}).
\]

GMRF regularization과 peak-risk loss를 추가하면

\[
\boxed{
\mathcal L_{\mathrm{total}}
=
\mathcal L_{\mathrm{NLL}}
+
\lambda_Q\mathcal L_{\mathrm{GMRF}}
+
\lambda_P\mathcal L_{\mathrm{peak}}
+
\lambda_B\mathcal L_{\mathrm{balance}}
}
\]

가 된다.

Peak threshold \(C\)에 대해

\[
E_n(C)
=
\mathbf1[M_{n,H}>C]
\]

이고 모델 risk를 \(R_n(C)\)라고 하면

\[
\boxed{
\mathcal L_{\mathrm{peak}}
=
\frac1N
\sum_n
[R_n(C)-E_n(C)]^2
}
\]

의 Brier loss를 적용한다.

---

## 15. 검증전략

데이터의 시간구조와 반복패턴을 고려하여 random split은 사용하지 않는다.

Expanding 또는 rolling time-series validation을 사용한다.

예측 horizon이 \(H\)이면 학습 target과 validation target의 중첩을 방지하기 위해 적절한 gap을 둔다.

| ID | 모델 | 목적 |
|---|---|---|
| B0 | Seasonal Naive | 최소 기준 |
| B1 | LightGBM | 강력한 tabular baseline |
| B2 | TimeXer-lite | Deep-only baseline |
| B3 | GMRF + TimeXer | 구조적 시간 prior 효과 |
| B4 | Fixed HMM + Deep | Markov 효과 |
| B5 | Conditional Markov + Deep | 동적 transition 효과 |
| B6 | GMRF + Conditional Markov + Deep | 제안모델 |
| B7 | GMRF + Conditional HSMM + Deep | duration 효과 |
| B8 | B6/B7 + Peak-risk loss | 최종 후보 |

핵심 제거실험은

\[
\text{Deep}
\rightarrow
\text{GMRF + Deep}
\rightarrow
\text{Fixed Markov}
\rightarrow
\text{Conditional Markov}
\rightarrow
\text{HSMM}
\]

순으로 구성한다.

복잡한 모델이 항상 우수하다고 가정하지 않으며, HSMM이 시간순 validation에서 명확한 개선을 보이지 않을 경우 Conditional Markov 모델을 최종모델로 선택한다.

---

## 16. 성능평가

### 전력 예측

\[
RMSE
=
\sqrt{\frac1N\sum_t(Y_t-\widehat Y_t)^2}
\]

\[
MAE
=
\frac1N\sum_t|Y_t-\widehat Y_t|.
\]

### 피크 예측

\[
M_n^{true}=\max_hY_{n,h},
\qquad
M_n^{pred}=\max_h\widehat Y_{n,h}
\]

\[
PeakMAE
=
\frac1N\sum_n|M_n^{true}-M_n^{pred}|.
\]

Peak timing error도 추가한다.

\[
\Delta t_n^{peak}
=
|\arg\max_hY_{n,h}-\arg\max_h\widehat Y_{n,h}|.
\]

### 확률예측

- Negative Log-Likelihood
- CRPS
- Prediction Interval Coverage
- Brier Score
- Reliability/Calibration Curve

을 평가한다.

---

## 17. 연구의 차별성

본 연구는 각 구성요소 자체의 최초성을 주장하지 않는다.

GMRF와 deep learning의 결합, neural switching model, explicit duration model 및 외생변수 기반 Transformer는 기존 연구에서 각각 제안된 바 있다.

본 연구의 차별성은 다음 요소를 **제조 전력의 peak-risk forecasting이라는 하나의 문제 안에서 구조적으로 결합**한다는 데 있다.

\[
\boxed{
\begin{array}{c}
\text{Structured GMRF temporal prior}\\
+\\
\text{Deep nonlinear exogenous modeling}\\
+\\
\text{Production-conditioned transition}\\
+\\
\text{State-specific probabilistic emission}\\
+\\
\text{Analytical interval peak-risk}
\end{array}
}
\]

특히 Conditional Markov transition을 단순한 auxiliary feature가 아니라 미래 상태경로와 peak exceedance probability를 계산하는 probabilistic operator로 사용한다.

---

## 18. 개발 단계

### Phase 1. 데이터 신뢰성 확보

- 15분 시계열 reconstruction
- timestamp 오류 확인 및 수정
- leakage 변수 제거
- 중복 daily pattern 분석
- 시계열 validation split 생성

### Phase 2. 강력한 baseline 구축

- Seasonal Naive
- LightGBM
- Transformer/TimeXer-lite

### Phase 3. 통계적 구조 결합

- RW2 trend
- cyclic daily GMRF
- INLA posterior mean 및 uncertainty 생성
- GMRF residual forecasting

### Phase 4. Conditional Markov 개발

- initial state posterior
- time-varying transition matrix
- Student-\(t\) state emission
- exact forward likelihood

### Phase 5. Peak-risk model

- threshold별 peak exceedance probability
- Brier calibration
- peak timing 및 magnitude 평가

### Phase 6. 확장모델

Conditional Markov에서 성능개선 가능성이 확인된 경우에만 explicit duration을 가진 Conditional HSMM으로 확장한다.

---

## 19. Contest Core와 Research Extension

### Contest Core v1

\[
\boxed{
\text{GMRF Residual}
+
\text{TimeXer-lite}
+
\text{Conditional Markov}
+
\text{State-wise Student-t}
+
\text{Analytic Peak Risk}
}
\]

대회 단계에서는 예측 정확도, 안정적 학습, 해석 가능성 및 재현성을 우선한다.

### Research Extension v2

\[
\boxed{
\text{GMRF}
+
\text{Conditional HSMM}
+
\text{Neural Emission}
+
\text{Differentiable Laplace Inference}
}
\]

확장 항목은 다음과 같다.

- explicit dwell-time
- differentiable sparse GMRF
- Laplace approximation
- fully Bayesian predictive uncertainty
- joint schedule optimization

---

## 20. 예상 결과물

1. 15분 단위 전력사용량 예측모델
2. 상태별 미래 전력 확률분포
3. 미래 latent load-state transition matrix
4. 최대피크 발생확률
5. 피크 발생 예상시간
6. 생산조건별 예측오차 분석
7. 피크 발생 주요 조건 분석
8. 생산일정 조정 시나리오별 예상 피크위험
9. 모델 제거실험 결과
10. 재현 가능한 전체 학습·추론 코드

---

## 21. 기대효과

본 연구는 전력사용량을 하나의 점으로 예측하는 기존 접근을 넘어, 제조공장의 미래 운전상태 변화와 확률적 전력분포를 함께 고려한다.

따라서 최종 시스템은

> **“향후 전력이 얼마인가?”**

뿐 아니라

> **“어떤 생산조건에서 높은 부하상태로 전환될 가능성이 큰가?”**

> **“향후 일정구간에서 최대전력이 기준을 넘을 확률은 얼마인가?”**

> **“생산일정을 변경했을 때 모델상 피크위험은 어떻게 변하는가?”**

까지 제공하는 의사결정지원 모델을 목표로 한다.

궁극적으로는 제조기업의 전력비용 관리와 생산안정성을 동시에 지원하는 확률적 AI 기반의 **Predict–Diagnose–Mitigate** 체계를 구축하는 것이 본 과제의 최종 목표이다.

# C-BAT 9월 봉인 시험 결과

## 실행 내용

- 모델: `BAT_conditional_gaussian_fixed_attention` (`ConditionalConfig` 기본값 그대로 동결, 조건부 가우시안 잡음 + 사전 고정 attention + `ref="op"`)
- 명령:
  `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 POLARS_MAX_THREADS=1 uv run python -m gmst.run_conditional --final --n-iter 6000 --burn 3000 --seeds 0 --out results_v3/cbat_final`
- 적합은 2021-09-01 이전 데이터만 쓰고, 2021-09-01~14(14일 × 96슬롯)를 예측했다. 위험확률은 내부 보정일로 적합한 Platt 보정을 적용했다.
- 동결한 모델로 9월을 **한 번만** 열었다. 결과를 본 뒤 모델을 조정하거나 다시 돌리지 않았다.
- 실행이 끝난 뒤 소스 해시 검사에서 `gmst/realloc_v3.py`가 달라졌다고 나왔다. 다른 작업이 같은 시각에 이 파일을 고쳤기 때문이다. `realloc_v3`는 `gmst.run_conditional`이 import하지 않으므로(`sys.modules`로 확인) 예측에는 영향이 없다. 산출물은 모두 기록됐고, `run_status.json`에 이 사정을 적었다(`status: complete_source_hash_mismatch`). 9월을 한 번만 연다는 원칙에 따라 다시 실행하지 않았다.

## 산출물 (`results_v3/cbat_final/`)

- `test_predictions.csv`: 제출 파일. 1,344행이며 열 구성은 `results_v3/final/test_predictions.csv`(run_v3 BAT)와 같다. 제출 파일의 M̂는 관측 슬롯으로 자르지 않은 발행값이다.
- `eval_mask.csv`: run_v3의 파일과 똑같다.
- `test_slots.csv`, `test_days.csv`, `test_metrics.csv`, `test_metrics_long.csv`, `posterior_*.csv`, `run_status.json`(`september_opened: true`)

## BAT·M2·B0와 비교 (2021-09-01~14)

다른 모델은 다시 적합하지 않고 `results_v3/final/`에 저장된 예측(`slots.csv`, `test_metrics.csv`)을 그대로 썼다. 네 모델 모두 `gmst.evaluate.point_metrics` 한 가지 정의로 계산했다. 관측 슬롯은 1,342개로 모든 모델이 같고, 실측값·결측 마스크가 같은지도 확인했다. 첨두 지표는 첨두 평가가 가능한 14일을 모두 썼다.

| 모델 | 슬롯 MAE | RMSE | R² | CRPS | 90% 구간 적중률 | 일 첨두 MAE | 첨두 시각 ±2슬롯 적중 | Brier(원시) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 7.877 | 11.183 | 0.9615 | – | – | 7.000 | 0.143 | 0.1190 |
| M2 | **6.600** | 9.751 | 0.9708 | **5.026** | 0.872 | **5.526** | 0.286 | 0.1208 |
| BAT | 6.824 | **9.533** | **0.9721** | 5.512 | 0.948 | 7.979 | 0.286 | **0.1118** |
| C-BAT | 7.184 | 9.789 | 0.9705 | 5.709 | 0.984 | 8.934 | 0.286 | 0.1608 |

- 슬롯 R²는 저장된 슬롯 예측으로 다시 계산했다. MAE·RMSE·CRPS는 `results_v3/final/test_metrics.csv`의 값과 소수점 끝자리까지 같다.
- 일 첨두 MAE는 BAT와 C-BAT 모두 발행값(자르지 않은 M̂)과 관측 슬롯으로 자른 값이 같다. 첨두 평가일 가운데 결측 슬롯이 있는 날은 09-08 하루(결측 2슬롯)뿐이고, 이날은 두 값에 차이가 나지 않았다. 첨두 R²는 C-BAT 0.9650, BAT 0.9706이다. M2와 B0는 일별 예측이 저장돼 있지 않아 첨두 R²를 계산할 수 없다.
- 9월 14일 동안 C-BAT는 슬롯 MAE, CRPS, 첨두 MAE, Brier에서 BAT와 M2보다 나빴다. 90% 구간 적중률(0.984)은 명목값 0.90을 넘어 구간이 넓은 편이다. 14일, 첨두 14개라는 작은 표본이라 차이를 유의하다고 판정할 근거는 없다.

## 주의

9월 구간은 v2 개발 중에 이미 한 번 열린 적이 있다(`results_v3/final`의 BAT·M2·B0 결과도 이 구간에서 나왔다). 따라서 이 비교는 오염되지 않은 홀드아웃이 아니며, 모델 선택 근거는 교차검증 폴드(f1~f4) 결과로 두어야 한다.

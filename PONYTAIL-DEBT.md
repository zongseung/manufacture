# PONYTAIL-DEBT

실제 코드의 `ponytail:` 주석을 수집한 원장입니다. 한 줄당 한 항목이며 코드 위치는 아래와 같습니다.

## gmst/analysis.py

| file:line | 지름길 | 한계 (ceiling) | 업그레이드 조건 (trigger) | 근거 |
|---|---|---|---|---|
| `gmst/analysis.py:207` | 누수 진단은 L2 모델만 사용 | 누수 진단은 L2 모델만 사용 | 분위수·위험 누수 격차가 필요하면 B1 전체를 비교 (FR-92) | FR-92 |

## gmst/baselines.py

| file:line | 지름길 | 한계 (ceiling) | 업그레이드 조건 (trigger) | 근거 |
|---|---|---|---|---|
| `gmst/baselines.py:10` | 기본 하이퍼파라미터 | 기본 하이퍼파라미터 | B1이 f1–f4에서 B0′를 못 이기면 튜닝 (FR-49) | FR-49 |
| `gmst/baselines.py:200` | 학습일 중심은 5개 연속 블록 교차적합 | 학습일 중심은 5개 연속 블록 교차적합 | 시간순 확장창이 필요하면 일별 as-of 재적합 (v2 §9.6 ①) | v2 §9.6 ① |

## gmst/evaluate.py

| file:line | 지름길 | 한계 (ceiling) | 업그레이드 조건 (trigger) | 근거 |
|---|---|---|---|---|
| `gmst/evaluate.py:617` | paired day bootstrap assumes independent dates | paired day bootstrap assumes independent dates | use 7-day sensitivity when lag-1 correlation is significant (FR-56). | FR-56 |

## gmst/hmm_core.py

| file:line | 지름길 | 한계 (ceiling) | 업그레이드 조건 (trigger) | 근거 |
|---|---|---|---|---|
| `gmst/hmm_core.py:117` | stationary AR reset after missing Y | stationary AR reset after missing Y | propagate residual mixtures if gap accuracy matters (review R4) | review R4 |

## gmst/hmm_forecast.py

| file:line | 지름길 | 한계 (ceiling) | 업그레이드 조건 (trigger) | 근거 |
|---|---|---|---|---|
| `gmst/hmm_forecast.py:80` | moment decomposition replaces variance refitting | moment decomposition replaces variance refitting | fit a random-effect likelihood if re changes the conclusion (FR-75) | FR-75 |

## gmst/hmm_training.py

| file:line | 지름길 | 한계 (ceiling) | 업그레이드 조건 (trigger) | 근거 |
|---|---|---|---|---|
| `gmst/hmm_training.py:63` | missing production is zero in transitions | missing production is zero in transitions | add missingness features if suspect-day sensitivity matters (FR-63) | FR-63 |

## gmst/run_all.py

| file:line | 지름길 | 한계 (ceiling) | 업그레이드 조건 (trigger) | 근거 |
|---|---|---|---|---|
| `gmst/run_all.py:26` | uv가 없으면 커밋된 requirements.txt 유지 | uv가 없으면 커밋된 requirements.txt 유지 | 재생성이 필요하면 importlib.metadata 직접 핀 (FR-100) | FR-100 |

## gmst/scenario.py

| file:line | 지름길 | 한계 (ceiling) | 업그레이드 조건 (trigger) | 근거 |
|---|---|---|---|---|
| `gmst/scenario.py:84` | 2020-12 자료 없음·계약전력 30% 하한 미적용·마스킹 점 제외로 래칫 바닥 계산 | 2020-12 자료 없음·계약전력 30% 하한 미적용·마스킹 점 제외로 래칫 바닥 계산 | 실제 청구 이력이 오면 교체 (FR-85) | FR-85 |
| `gmst/scenario.py:199` | 날짜 간 경로 독립 짝짓기 | 날짜 간 경로 독립 짝짓기 | 날짜 간 상관이 크면 일 랜덤효과 경로로 교체 (FR-85) | FR-85 |

## gmst/splits.py

| file:line | 지름길 | 한계 (ceiling) | 업그레이드 조건 (trigger) | 근거 |
|---|---|---|---|---|
| `gmst/splits.py:13` | 2021 공휴일 하드코딩 | 2021 공휴일 하드코딩 | 다른 연도 데이터가 오면 특일정보 목록 교체 (FR-21) | FR-21 |

## 발동된 트리거

전체 개발과 실제 봉인 평가를 실행했습니다. B1의 OOF MAE 13.904가 B0′의 13.720보다 높아 `gmst/baselines.py:10`의 튜닝 검토 조건이 발동했습니다. 봉인 평가가 이미 끝났으므로 이번 선택·최종 결과에는 사후 튜닝을 적용하지 않습니다. 별도로 사전 정의한 후속 실험에서만 검토할 수 있습니다. smoke 결과는 성능 판단에 사용하지 않았습니다.

총 11개 주석, 확장 조건 누락 0개.

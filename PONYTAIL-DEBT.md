# Ponytail 부채 원장

의도적으로 단순화한 곳을 모았습니다. 한계(ceiling)와 다시 볼 조건(upgrade)은 코드의 `ponytail:` 주석에서 그대로 가져왔습니다.

## gmst/bat.py
- `bat.py:26`, Metropolis 보폭 고정. ceiling: 랜덤워크 보폭 고정. upgrade: 수용률이 0.15–0.5를 벗어나면 burn-in 동안 보폭을 적응시킴.
- `bat.py:264`, AR(1) 잡음 시작값. ceiling: 정상분산 가우시안으로 근사. upgrade: 하루 경계 효과가 보이면 전일 말 잔차로 시작.

## gmst/run_v3.py
- `run_v3.py:91`, 봉인 테스트의 부트스트랩 생략. ceiling: `ev.bootstrap_days`가 CV 검증일만 받음. upgrade: 시험 구간 신뢰구간이 필요해지면 그 필터에 fold 인자를 추가.

## gmst/reallocate.py
- `reallocate.py:117`, 최대수요 비용. ceiling: 기본요금을 일 단위(/30)로 환산하고 래칫을 무시. upgrade: 월 래칫(`scenario.ratchet_floor`)이 걸리는 달이면 max(peak, floor)로 교체.

## gmst/scenario.py
- `scenario.py:76`, 래칫 바닥 계산. ceiling: 2020-12 자료 없음, 계약전력 30% 하한 미적용, 마스킹된 점 제외. upgrade: 실제 청구 이력이 오면 교체.

## gmst/evaluate.py
- `evaluate.py:470`, 날짜 짝짓기 부트스트랩. ceiling: 날짜 간 독립을 가정. upgrade: lag-1 상관이 유의하면 7일 블록 민감도를 사용(이미 자동 계산됨).

## gmst/splits.py
- `splits.py:12`, 공휴일 목록. ceiling: 2021년 하드코딩. upgrade: 다른 연도 데이터가 오면 특일정보 목록을 교체.

## gmst/baselines.py
- `baselines.py:9`, LightGBM(M2) 하이퍼파라미터. ceiling: 기본값. upgrade: B1이 f1–f4에서 B0′를 못 이기면 튜닝(2026-09-25 사용자 결정: M2 튜닝은 하지 않음).

8 markers, 0 with no trigger.

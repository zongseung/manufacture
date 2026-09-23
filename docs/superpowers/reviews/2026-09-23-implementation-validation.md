# KAMP 코드 구현·검증 기록 — 2026-09-23

이 문서는 코드 구성 직후의 검증 상태를 기록합니다. 이후 완료된 전체 개발·실제 9월 평가 결과는 [전체 실험 기록](2026-09-23-full-experiment-validation.md)을 따릅니다.

기획 검토는 Sol xhigh, 구현은 Astra xhigh와 Ponytail로 진행했다. 작업 위치는 `manufacture_ai/.claude/worktrees/kamp-core`, 기준 커밋은 `6619558f5bd56adccf422226a310dae0c936b03f`다. 커밋·푸시·외부 제출은 하지 않았다.

## 구현 결과

- 계획·PRD·제안서의 봉인, fold별 튜닝·보정, 53일 OOF, 변형 식별자, 선택적 컷 계약을 동기화했다.
- 전처리·분할·봉인 로더, B0/B0p/BB/B1, B2/B3/B4 및 H/IO/KAN, 평가·보정·bootstrap, 요금·시나리오·오류 분석을 구현했다.
- 개발/스모크/명시적 final CLI, 고정 선택 검증, 실패한 final의 동일 선택·사유 기록 재시도, 제출 스키마, 노트북, README, 의존성 고정 파일과 ZIP을 구성했다.
- 독립 검토에서 찾은 H 교차적합·LOEO 특징의 보류 라벨 재진입, 부분 결측 평가 위험, 후보 추가 후 누락된 상태, 스모크의 정식 결과 덮어쓰기, fold 설정 누락을 수정했다. CUDA 장치 별칭 오류도 재현 테스트로 수정했다.

## 실제 검증

| 검사 | 결과 |
|---|---|
| 전체 검사, CUDA 사용 가능 환경, 실제 스모크 산출물 포함 | 139 passed, 0 skipped |
| CPU 환경의 같은 검사 | 138 passed, CUDA 전용 1 skipped |
| 마지막 노트북 변경 후 CLI·노트북·산출물 검사 | 16 passed |
| Ruff / basedpyright gmst | 통과 / 오류 0 |
| 실제 기본 스모크 | 14단계 완료, `results_smoke/`에 분리 저장 |
| 같은 장치 스모크 2회 | 23 CSV·5 JSON 값 일치, 최대 수치 차이 0.0 |
| 예측 파일 | 1,344행 × 33열, 2021-08-18~08-31, 수치 모두 유한 |
| 봉인 확인 | 9월 14일의 Y·모든 X가 NaN, 실제 final 시작 기록 없음 |
| 노트북 | 3개 실행·출력 저장, 오류 0, 후보 CI 결합 표와 8개 지표·n 비교표 확인 |
| ZIP | 46파일, 압축 무결성·허용 경로·작업파일 바이트 일치 확인 |

실행 예시:

```bash
uv run python -m gmst.run_all --smoke
KAMP_RESULTS="$PWD/results_smoke" uv run pytest -q
uv run python -m gmst.run_all --package
```

최종 코드의 실행 방법은 [README](../../../README.md), 단순화와 확장 조건은 [PONYTAIL-DEBT](../../../PONYTAIL-DEBT.md)에 있다. 소스 ZIP은 `dist/kamp_power_src.zip`이다. 스모크와 ZIP은 git ignore 대상이며 현재 작업 디렉터리에 남겨 두었다.

## 검증 범위와 실험 인계

이번 완료 범위는 코드 구성과 봉인을 유지한 실행 검증이다. 계획 Task 20의 전체 예산 f1–f4 실험과 Task 24의 실제 9월 평가, 대회용 PDF·PPT·포털 제출은 수행하지 않았다. `--final` 구현은 합성 경계 검사로 검증했으며 실제 봉인은 열지 않았다. 제공 ZIP은 소스 검토용이며 최종 대회 결과 패키지가 아니다.

스모크는 epoch 2·50 경로·LightGBM 10회·bootstrap 20회와 축소 진단을 사용한다. 스모크에서 선택된 B4-KAN은 배선 검증 결과이며 성능 우월성의 근거가 아니다. 이 구현은 깊은 인코더가 아닌 조건부 확률상태모형이다. 결측 직후 AR 재시작 근사, A+ 계획 가용성 가정, 선택 후 탐색적 CI, B1 일 분류기의 날짜별 결측 마스크 확률 재구성 한계는 문서와 코드에 남겼다.

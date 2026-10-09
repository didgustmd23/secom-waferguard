# 프로젝트 문서

WaferGuard는 전처리 에이전트, SECOM 머신러닝, WM-811K 딥러닝의 세 영역으로 구성한다. 전체 소개와 각 영역의 구현 상태는 루트 [README](../README.md)를 따른다.

| 영역 | 문서 | 상태 |
| --- | --- | --- |
| SECOM 머신러닝 | [SECOM 문서 목록](README.md#secom-문서), [실행 안내](secom/USAGE.md) | 개발 결과·후보 모델 계약·실행 방법 기록 |
| 전처리 에이전트 | [WM-811K 기초 설계 5절](WM811K/WM811K_BASIC_DESIGN.md#5-맵-전처리-에이전트--양현승) | WM 맵 전처리 작업 관리 설계 |
| WM-811K 딥러닝 | [WM-811K 기초 설계 7절](WM811K/WM811K_BASIC_DESIGN.md#7-wm-딥러닝--이종수) | 작은 CNN·ResNet18 학습·평가 설계 |

에이전트와 WM 딥러닝은 아직 설계 단계이며, 두 데이터셋의 학습·평가는 독립적으로 수행한다. 예정 소스·의존성·폴더는 구현된 것으로 취급하지 않는다.

## SECOM 문서

### 현재 결과와 사용 계약

- [GitHub 공개 결과 요약](../reports/README.md#secom-공개-결과-요약): 센서 축소 비교·90%/80% 개발 결과·저장 및 복원 검증 집계

- [로그 인덱스](secom/LOG_INDEX.md): 실험 로그 분류·이동 전후 경로·저장 모델의 고정 출처 경로

- [머신러닝 결과 요약](secom/SECOM_ML_SUMMARY.md): V2 비교 실험·고정 센서 후보·90%/80% 시나리오의 결과와 한계
- [V2 Model Card](secom/model_card_v2.md): 저장 모델의 센서 목록·학습 범위·성능·입력 계약·환경
- [V2 시나리오 보존 기록](secom/V2_SCENARIO_BUNDLES.md): 저장·복원·20개 센서 추론 데모 절차와 완료 기록
- [실험 보고서](secom/report.md): 초기 탐색부터 V2까지의 이력. 고정 센서 개발 결과는 35.4절, 기존 Test 비교는 35.5절, 팀원 후보 대조는 36절

### 계획과 의사결정

- [상세 실행 안내](secom/USAGE.md): 환경 준비·데이터 생성·개별 실험·V2 추론·분석 노트북
- [Dataset Profile 작성법](secom/DATASET_PROFILE.md): 공통 설정과 데이터셋별 JSON 입력 계약

- [실행 계획·완료 체크](secom/PLAN.md)
- [공동 결정 사항](secom/JOINT_DECISIONS.md)
- [통합 실험 노트](secom/EXPERIMENT_NOTES.md): V1 S/M 설정, V2 가중치·결측·센서 축소, 고정 센서 선택 시점과 OOF, 재현 명령

문서 역할은 PLAN의 작업 상태, 실험 노트의 조건·근거 경로, 보고서의 상세 수치, ML 요약의 결론, Model Card의 저장 모델 계약으로 나눈다. 완료 체크와 실행 안내를 여러 노트에 중복 작성하지 않는다.

### V1 보존 이력

- [V1 Model Card](secom/model_card.md)
- [V1 최종 평가 점검](secom/FINAL_EVALUATION_REVIEW.md)

V1의 성능·문턱을 V2 저장 모델의 값으로 사용하지 않는다. 구간별 센서 재선택 실험과 고정 센서 모델의 평가 범위도 구분한다.

실행 명령의 파일 경로는 **프로젝트 루트 기준**이다. 문서를 이 폴더로 이동해도 학습 코드·데이터·로그·모델 경로는 변경하지 않는다. 그림과 분석 노트북은 `reports/`, 로그는 `logs/`, 모델 객체는 `models/`에 유지한다. 폴더 이동 전 문서의 상대 링크는 새 위치 기준으로 수정했다.

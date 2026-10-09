# WaferGuard 소스 구성

프로젝트 영역은 SECOM 머신러닝, 전처리 에이전트, WM-811K 딥러닝으로 나눕니다. 실제 구현은 SECOM과 재사용 코어이며 나머지 두 폴더는 구현 전 예약 영역입니다. 설계 안내는 이 문서에 통합합니다.

```text
src/
├── modeling_*.py 등   # 재사용 코어·설정·검증·센서 추론
├── sensor_ml/        # 현재 구현된 센서 ML 실행기
│   ├── data_pipeline/
│   ├── experiments/
│   ├── diagnostics/
│   ├── inference/
│   ├── evaluation/
│   └── verification/
├── agents/           # 전처리 작업 관리 예정 영역
└── wafer_dl/         # 맵 처리·딥러닝 예정 영역
```

맵 처리 함수와 딥러닝은 [wm811k](README.md#wm-811k-맵-처리와-딥러닝), 작업 실행·상태·재시도 관리는 [agents](README.md#전처리-에이전트)에서 분리합니다. SECOM 코드는 이 두 영역에 의존하지 않습니다. 현재 코어는 표 형태·이진 분류 계약이며 WM 다중 분류의 지표·맵 변환을 그대로 처리한다고 가정하지 않습니다.

## 공통 코어 — src 바로 아래

여러 실행기가 재사용하는 설정·데이터 구조·전처리·모델·평가·센서 선택·추론 코어만 유지합니다. 실행기는 아래 기능별 폴더에서 관리합니다.

- 코어: `modeling_config`, `dataset_schema`, `modeling_preprocessing`, `modeling_models`, `modeling_metrics`, `split_contract`, `threshold_policy`, `feature_reduction`, `stable_rf_selector`.
- 추론 코어: `sensor_inference`. 저장 모델의 클래스 참조 경로를 보존합니다.

## SECOM 기능별 실행기 — src/sensor_ml

| 폴더 | 역할 | 파일 |
| --- | --- | --- |
| data_pipeline/ | 데이터 병합·품질 점검·분할 | step1_merge_data, step2_data_check, step3_split |
| experiments/ | Baseline·후보·특징 비교·시간 검증·OOF 실험 | step4_baseline~step8_threshold_oof, top20_oof_compare, top20_time_validation, temporal_validation, time_weight_compare, recent_window_compare, feature_time_compare, anomaly_compare |
| experiments/missing_value_compare.py | V2 모델을 유지한 결측 처리 비교 | P0 중앙값 / P1 중앙값+indicator / P2 자체 NaN 처리 |
| inference/ | 모델 묶음 저장·복원·시나리오 보존·CSV 예측 | sensor_bundle, save_v2_scenario, predict_cli |
| evaluation/ | 동결 모델 평가·평가 실행 범위 관리 | evaluate_frozen_model, run_evaluation |

실험 폴더는 학습과 후보 비교, 평가 폴더는 동결 모델 평가 및 실행 관리가 중심입니다. `run_evaluation`은 선택된 변경 범위에 따라 OOF 학습을 호출할 수 있습니다. 역할 분류이지 학습 유무만으로 나눈 것은 아닙니다.

## SECOM 진단·보고서 — src/sensor_ml/diagnostics

| 파일 | 역할 |
| --- | --- |
| sensor_importance.py | 전체 Train 진단용 XGBoost gain 중요도·그래프 |
| sensor_stability.py | 기존 시간 Fold별 중요도 순위·Top-K 등장 횟수·공통 센서 |
| sensor_class_distribution.py | Train 정상·불량별 원본 ECDF·시간 구간별 중앙값·결측률 |
| step3_drift_check.py | Train/Validation 입력 분포 변화 |
| selected_sensor_drift.py | 저장된 센서 목록의 분포 변화 |
| sensor_detail.py | 지정 센서의 상세 통계·그림 생성 함수 |
| score_diagnostics.py | 점수 분포·선택 센서 교집합 보고서 생성 함수 |
| step9_error_analysis.py | FN/FP 그룹과 센서 분포 비교 |
| step5_analysis.py | 기존 PCA·모델 탐색 분석과 시각화 |
| step10_random_vs_time.py | 기존 Random/Time 탐색 비교 |
| step11_drift_check.py | 기존 시간 변화 분석 |
| step12_final_analysis.py | 기존 분석 로그 기반 보고서 그림 |

이 분류는 역할 기준입니다. 중요도·오류 분석 등 일부 진단은 모델을 학습하며, 기존 탐색 스크립트가 모두 최종 평가 프로토콜을 준수한다고 보증하지는 않습니다.

## SECOM 구현 정합성 확인 — src/sensor_ml/verification

`check_sensor_inference.py`는 전체 입력과 축소 입력의 확률·판정 일치 확인 및 후보 묶음 저장을 담당합니다. 성능 평가와는 구분합니다. 단위 테스트 파일은 이동하지 않고 프로젝트 루트의 `tests/`에 유지합니다.

## 실행 예시

프로젝트 루트에서 `python -m src.sensor_ml.<기능 폴더>.<모듈명>`으로 실행합니다. 파일 직접 실행 대신 아래 모듈 경로를 사용하세요.

```powershell
# 데이터 분할 실행 (실제 데이터 준비 작업)
python -m src.sensor_ml.data_pipeline.step3_split

# V2 모델·가중치 비교 (실제 학습 실행)
python -m src.sensor_ml.experiments.time_weight_compare --train data/splits/integrated/time_train.csv --output-dir logs/v2_time_weight_new --n-jobs 2

# 시간순 Fold별 gain 안정성: 학습을 수행하지만 성능 평가는 하지 않음
python -m src.sensor_ml.diagnostics.sensor_stability --train data/splits/integrated/time_train.csv --model v2_m0_ratio --top-k 20 --output-dir logs/v2_sensor_stability --n-jobs 2

# 전체 Train 중요도와 그림
python -m src.sensor_ml.diagnostics.sensor_importance --train data/splits/integrated/time_train.csv --model v2_m0_ratio --output-dir logs/v2_sensor_importance_new --figure-dir reports/secom/figures/v2/v2_sensor_importance_new --n-jobs 2

# 추론 정합성 확인 옵션 조회 (실제 검증·학습 실행은 하지 않음)
python -m src.sensor_ml.verification.check_sensor_inference --help

# 안정성 진단 후 선택한 5개 센서의 원본 분포 (모델 학습 없음)
python -m src.sensor_ml.diagnostics.sensor_class_distribution --train data/splits/integrated/time_train.csv --features sensor_64 sensor_46 sensor_426 sensor_65 sensor_59 --output-dir logs/v2_sensor_class_distribution --figure-dir reports/secom/figures/v2/v2_sensor_class_distribution
```

기존 로그·그림·모델 경로와 학습 설정은 폴더 정리로 변경하지 않습니다. 안정성 결과는 `fold_importances.csv`, `stability.csv`, `overlap.csv`, `fold_summary.csv`, `execution.json`, `stability.md`로 저장됩니다. 진단 결과로 최종 센서 목록을 자동 확정하지 않습니다.

노트북 import도 새 경로를 사용합니다. 프로젝트 루트를 `sys.path`에 등록하는 기존 첫 셀은 유지하세요.

```python
from src.sensor_ml.diagnostics.sensor_importance import fit_importance_candidate
from src.sensor_ml.experiments.time_weight_compare import build_weight_candidates
```

공통 모듈은 기존 import를 유지합니다. 예: `from src.modeling_config import load_modeling_config`.

## 경로 호환과 이동 범위

기존 여섯 실행기 폴더를 `src/sensor_ml/`으로 이동하고 코드·테스트·문서·노트북 셀의 import와 실행 명령을 갱신했습니다. 이동으로 깊이가 늘어난 모듈은 프로젝트 루트 계산도 조정했습니다. 데이터·로그·그림·모델 묶음은 이동하거나 재생성하지 않았습니다.

과거 `python -m src.experiments.time_weight_compare`처럼 작성한 명령도 `src/__init__.py`의 패키지 검색 경로를 통해 이동된 파일을 찾습니다. 호환용 소스 복제는 만들지 않았습니다. 새 코드·import·테스트와 명령은 `src.sensor_ml`을 기준으로 사용하며 옛 경로와 새 경로를 한 실행에서 혼용하지 않는 것이 좋습니다.

직전 `src.secom` 패키지 경로도 호환 조회를 유지합니다. 실제 소스는 `src/sensor_ml/`에만 있으며 새 개발에서는 역할 기반 이름을 사용합니다. 데이터셋별 이름은 Profile·실험 기록·문서·데이터·로그·모델 경로에서 유지합니다.

저장 모델의 클래스가 참조하는 `src.sensor_inference`, `src.modeling_models`, `src.modeling_preprocessing`, `src.stable_rf_selector` 경로는 유지합니다. 이 구조 변경은 모델 설정·센서 목록·문턱·과거 실험 수치 변경이 아닙니다.

## 전처리 에이전트

담당은 양현승이며 아직 실행 가능한 에이전트를 구현하지 않았습니다. `src/agents/`는 작업 실행·상태·실패·재시도·산출물 관리 코드를 둘 영역입니다.

초기 대상은 WM-811K 맵 전처리의 `ingest → audit → split → prepare → validate_contract`입니다. 실제 맵 읽기·검사·분할·변환 함수는 `src/wafer_dl/`에 구현하고 에이전트는 그 함수를 호출합니다. SECOM 학습이나 문턱 선택을 자동 변경하는 역할은 포함하지 않습니다.

- 라벨·제외 기준·split·변환은 고정 설정을 따릅니다.
- 실패한 부분 산출물을 완료 결과로 등록하지 않습니다.
- 에이전트 없이도 전처리 함수를 실행·검증할 수 있도록 분리합니다.
- 아직 프레임워크 의존성·실행 명령·LLM 호출을 추가하지 않았습니다.

예정 진입 모듈은 `src/agents/wm_preprocess_agent.py`입니다. 이름만 정리한 계획이며 파일은 아직 없습니다. 자세한 책임과 입력 계약은 [WM-811K 기초 설계](../WM811K_BASIC_DESIGN.md)를 따릅니다.

## WM-811K 맵 처리와 딥러닝

아직 데이터 준비·딥러닝 학습 모듈을 구현하지 않았습니다. `src/wafer_dl/`는 웨이퍼맵 전처리 함수와 다중 분류 모델·학습·평가·추론 코드를 둘 영역입니다.

| 영역 | 예정 모듈 | 담당 |
| --- | --- | --- |
| 맵 데이터 준비 | ingestion, audit, split, dataset, transforms | 양현승 |
| 딥러닝 | models, train, metrics, evaluate, predict | 이종수 |

전처리 에이전트의 작업 관리 코드는 별도 `src/agents/`에서 위 데이터 준비 함수를 호출하도록 계획합니다. 여기서 다시 에이전트 실행 관리 코드를 중복 구현하지 않습니다.

작은 CNN과 ResNet18, 9종 패턴 분류, Lot 비중첩 분할을 계획합니다. SECOM 이진 분류 설정·지표를 그대로 사용하지 않으며 입력 채널·클래스 매핑·맵 변환·학습 조건은 별도 계약으로 관리합니다. `config.json`의 SECOM Profile을 WM 설정으로 교체하지 않습니다.

구현 전의 예정 모듈은 파일이나 실행 명령으로 제공하지 않습니다. 데이터·모델·전처리 계약과 평가 범위는 [WM-811K 기초 설계](../WM811K_BASIC_DESIGN.md)를 참고하세요.

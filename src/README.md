# 소스 역할과 실행 방법

## 공통 코어 — src 바로 아래

여러 실행기가 재사용하는 설정·데이터 구조·전처리·모델·평가·센서 선택·추론 코어만 유지합니다. 실행기는 아래 기능별 폴더에서 관리합니다.

- 코어: `modeling_config`, `dataset_schema`, `modeling_preprocessing`, `modeling_models`, `modeling_metrics`, `split_contract`, `threshold_policy`, `feature_reduction`, `stable_rf_selector`.
- 추론 코어: `sensor_inference`. 저장 모델의 클래스 참조 경로를 보존합니다.

## 기능별 실행기

| 폴더 | 역할 | 파일 |
| --- | --- | --- |
| data_pipeline/ | 데이터 병합·품질 점검·분할 | step1_merge_data, step2_data_check, step3_split |
| experiments/ | Baseline·후보·특징 비교·시간 검증·OOF 실험 | step4_baseline~step8_threshold_oof, top20_oof_compare, top20_time_validation, temporal_validation, time_weight_compare, recent_window_compare, feature_time_compare, anomaly_compare |
| inference/ | 모델 묶음 저장·복원·CSV 예측 | sensor_bundle, predict_cli |
| evaluation/ | 동결 모델 평가·평가 실행 범위 관리 | evaluate_frozen_model, run_evaluation |

실험 폴더는 학습과 후보 비교, 평가 폴더는 동결 모델 평가 및 실행 관리가 중심입니다. `run_evaluation`은 선택된 변경 범위에 따라 OOF 학습을 호출할 수 있습니다. 역할 분류이지 학습 유무만으로 나눈 것은 아닙니다.

## 진단·보고서 — src/diagnostics

| 파일 | 역할 |
| --- | --- |
| sensor_importance.py | 전체 Train 진단용 XGBoost gain 중요도·그래프 |
| sensor_stability.py | 기존 시간 Fold별 중요도 순위·Top-K 등장 횟수·공통 센서 |
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

## 구현 정합성 확인 — src/verification

`check_sensor_inference.py`는 전체 입력과 축소 입력의 확률·판정 일치 확인 및 후보 묶음 저장을 담당합니다. 성능 평가와는 구분합니다. 단위 테스트 파일은 이동하지 않고 프로젝트 루트의 `tests/`에 유지합니다.

## 실행 예시

프로젝트 루트에서 모듈 방식으로 실행합니다. 과거 `src/<파일명>.py` 경로는 아래 새 경로로 변경됐습니다.

```powershell
# 데이터 분할 실행 (실제 데이터 준비 작업)
python -m src.data_pipeline.step3_split

# V2 모델·가중치 비교 (실제 학습 실행)
python -m src.experiments.time_weight_compare --train data/splits/integrated/time_train.csv --output-dir logs/v2_time_weight_new --n-jobs 2

# 시간순 Fold별 gain 안정성: 학습을 수행하지만 성능 평가는 하지 않음
python -m src.diagnostics.sensor_stability --train data/splits/integrated/time_train.csv --model v2_m0_ratio --top-k 20 --output-dir logs/v2_sensor_stability --n-jobs 2

# 전체 Train 중요도와 그림
python -m src.diagnostics.sensor_importance --train data/splits/integrated/time_train.csv --model v2_m0_ratio --output-dir logs/v2_sensor_importance_new --figure-dir reports/figures/v2_sensor_importance_new --n-jobs 2

# 추론 정합성 확인 옵션 조회 (실제 검증·학습 실행은 하지 않음)
python -m src.verification.check_sensor_inference --help
```

기존 로그·그림·모델 경로와 학습 설정은 폴더 정리로 변경하지 않습니다. 안정성 결과는 `fold_importances.csv`, `stability.csv`, `overlap.csv`, `fold_summary.csv`, `execution.json`, `stability.md`로 저장됩니다. 진단 결과로 최종 센서 목록을 자동 확정하지 않습니다.

노트북 import도 새 경로를 사용합니다. 프로젝트 루트를 `sys.path`에 등록하는 기존 첫 셀은 유지하세요.

```python
from src.diagnostics.sensor_importance import fit_importance_candidate
from src.experiments.time_weight_compare import build_weight_candidates
```

공통 모듈은 기존 import를 유지합니다. 예: `from src.modeling_config import load_modeling_config`.

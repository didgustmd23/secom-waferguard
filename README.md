# SECOM WaferGuard

> UCI SECOM 센서 데이터를 활용한 반도체 공정 불량 웨이퍼 조기 탐지 프로젝트

고차원·결측·불균형 특성을 가진 반도체 제조 공정 데이터를 분석하여, 최종 검사 이전에 불량(Fail) 가능성이 높은 생산 단위를 탐지합니다. 단순 정확도보다 불량 미검(False Negative)을 줄이는 데 초점을 두며, 핵심 센서 선택과 재현 가능한 머신러닝 파이프라인 구축을 목표로 합니다.

실험 일정, 역할 분담, 완료 기준은 [PLAN.md](PLAN.md)에서 관리합니다.

## 현재 구현 상태

| 구분 | 상태 | 내용 |
| --- | --- | --- |
| 범용 설정 코어 | 완료 | `config.json`과 Dataset Profile을 분리해 로드·검증 |
| 데이터 구조 검증 | 완료 | Profile 기준 label·metadata·feature 컬럼 검증 |
| 공통 평가 | 완료 | Profile label 기반 Recall, AP, Precision, F1, ROC-AUC 계산 |
| 자동 테스트 | 완료 | 설정·schema·평가·전처리·split 계약·실험 함수 및 모델 코어 123개 테스트 |
| 데이터 병합·품질 점검 | 완료 | Profile 기반 canonical 병합과 재생성 가능한 품질 로그 |
| Random / Time split | 구현 완료 | 팀원의 `step3_split.py`에 Profile·누수 검사·원본 ID를 통합. Random은 탐색용, 후보 비교·OOF는 Time Train 사용 |
| Baseline | 완료 | Train/Validation 기반 Logistic Regression 평가 및 공통 지표 기록 |
| 후보 모델·불균형 비교 | 비교 완료 | XGBoost 기본·`scale_pos_weight`를 포함한 8개 후보 재실행. 기본 XGBoost CV AP 0.2490; 최종 후보 미확정 |
| 특징 선택 비교 | 비교 완료 | PCA·L1·LightGBM Top-K 및 XGBoost/RF 선택 Top-20 비교. RF Top-20의 CV AP 0.2459, 시간순 AP 0.1177로 성능 유지 미확인·센서 동결 대기 |
| 시간 변화 진단 | 완료 | Time Train/Validation drift 우선순위: 높음 242개, 중간 102개, 낮음 246개 |
| Time Validation | 완료 | LightGBM 전체 특성 및 L1 선택 후보의 시간 구간 일반화 성능 확인 |
| OOF threshold 비교 | 비교 완료 | LightGBM 및 XGBoost 두 후보의 단일 OOF·재검사 부담 기록. XGBoost도 90%·20% 정책 미충족 |
| 최종 Test·모델 배포 | 미완료 | Time Validation 성능 개선·모델 재선정 후 Test 1회 평가, Pipeline 저장·Model Card·예측 CLI 구현 예정 |

기존 Time Validation 235행 중 163행이 Random Train에도 포함되어 있었다. 아래 기존 성능 수치는 **탐색 실험 기록**이며 독립적인 미사용 holdout 평가로 해석하지 않는다. 교정된 실행 경로는 시간 split을 먼저 생성하고, 동일한 Time Train에서 후보 비교·특징 선택·OOF를 수행한다. 이미 관찰한 데이터의 재분할이 새로운 미사용 평가 데이터를 만들지는 않으므로, 엄격한 최종 검증에는 별도의 미사용 데이터가 필요하다.

## 프로젝트 목표

- 불량 클래스의 **Recall**과 **PR-AUC (Average Precision)** 를 중심으로 모델을 평가합니다.
- 약 590개의 공정 특징에서 성능 저하를 최소화하는 핵심 센서를 찾습니다.
- 반복 계층 교차 검증으로 성능의 평균과 편차를 함께 확인합니다.
- Random split과 시간 순서 기반 평가를 비교해 데이터 드리프트 가능성을 점검합니다.
- 전처리부터 추론까지 재현 가능한 Pipeline을 구성합니다.

## 데이터셋

| 항목 | 내용 |
| --- | --- |
| 데이터셋 | [UCI SECOM Data Set](https://archive.ics.uci.edu/dataset/179/secom) |
| 도메인 | Semiconductor Manufacturing Process |
| 샘플 수 | 1,567 |
| 원문 속성 수 | 591 |
| 불량(Fail) 수 | 104 |
| 분류 문제 | 불균형 이진 분류 |
| 라이선스 | CC BY 4.0 |

각 행은 생산 단위 하나를, 각 열은 공정 센서 또는 측정값을 나타냅니다. Label 파일에는 Pass/Fail 결과와 Timestamp가 포함됩니다.

| Label | 의미 |
| ---: | --- |
| `-1` | Pass (정상) |
| `1` | Fail (불량) |

> UCI 설명의 591개 속성과 프로젝트에서 사용하는 센서 열 수는 Label·Timestamp 포함 여부에 따라 다를 수 있습니다. 실제 사용 열 수와 제거 내역은 데이터 점검 결과로 기록합니다.

## 접근 방법

```text
Raw sensor data + label/timestamp
                ↓
Data quality check
                ↓
Random split / Time-based split
                ↓
Baseline model
                ↓
Feature selection
                ↓
Candidate model comparison + repeated CV
                ↓
Threshold selection (OOF prediction)
                ↓
Final test and model bundle
```

### 데이터 전처리

- 센서별 결측률과 상수(Zero-variance) 센서를 점검해 EDA 후보로 기록합니다. 모델 Pipeline의 첫 단계에서는 각 학습 fold에서 결측률 50% 초과 센서를 제거한 뒤, 남은 센서 중 결측을 제외한 값의 종류가 하나 이하인 센서를 제거합니다.
- 전체 데이터에서 센서를 전역 제거하지 않습니다. 실제 제거·선택은 Train/CV 학습 fold 내부에서만 수행합니다.
- 나머지 결측값은 `SimpleImputer(strategy="median")`로 처리합니다.
- Imputation, scaling, 특징 선택은 학습 데이터에만 fit되도록 `scikit-learn Pipeline` 안에서 수행합니다.

### 모델링 및 검증

| 단계 | 방법 |
| --- | --- |
| Baseline | Sensor quality filter → Median imputation → StandardScaler → Logistic Regression |
| 특징 축소 | PCA, L1 Logistic Regression, Feature Importance 기반 Top-K |
| 후보 모델 | Logistic Regression, SVM, Random Forest, LightGBM, XGBoost |
| 불균형 처리 | `class_weight`, `scale_pos_weight` |
| 검증 | Repeated Stratified K-Fold CV |
| 핵심 지표 | Fail Recall, PR-AUC / AP |
| 보조 지표 | Precision, F1-score, ROC-AUC, 학습·추론 시간 |

최종 분류 임계값은 Out-of-Fold 예측에서 Recall, Precision, False Negative, False Positive의 trade-off를 비교하여 결정합니다.

### 현재 실험 결과 요약

2026-10-06 센서 품질 필터 적용 후 같은 Time Train으로 Step 4~9를 재실행했다. 결측률 초과 24개·상수 122개를 제거해 **444개** 센서를 유지했다. Time Train 내부 CV AP는 LightGBM **0.2424**, Random Forest **0.2324**이며, threshold 0.50의 Recall은 L1 balanced **0.2510**이 가장 높았다. 특징 비교에서 LightGBM Top-100 AP는 **0.2407**, Top-50은 **0.2353**, Top-20은 **0.2042**였다. 시간 검증 AP는 Baseline **0.0683**, LightGBM **0.1110**, L1 선택 **0.0668**이었고 모두 Recall **0**이었다. 상세 결과는 [중간 보고서](reports/report.md)의 21절, 새 로그는 `logs/quality_filtered_20261006/`에 기록했다. 기존 Random 실험·필터 전 결과는 탐색 기록으로 보존하며 관찰한 시간 구간은 개발용 검증 데이터로 취급한다.

> 아래 표는 기존 탐색 실험 기록이며 최종 Test 성능이 아닙니다.

필터 적용 후 Step 8 단일 OOF의 격자 내 F1 최대 후보는 **0.00037729474027866034**(Recall **0.3462**, Precision **0.2231**, F1 **0.2714**)였다. 새 비교표의 출처를 확인하고 같은 값을 Time Validation에 적용한 결과 **TP 0·FP 3·FN 17·TN 215**로 불량을 검출하지 못했다. 최종 모델·threshold는 미확정이며 현장 재검사 용량·FN/FP 비용 기준도 필요하다. 필터 전 Time Train 내부 시간순 진단은 보고서 18~20절의 과거 결과다. 필터 적용 후의 시간순 내부 검증은 코드화해 재실행했고, 보고서 22절과 `logs/temporal_quality_filtered_20261006_final/`에 기록했다.

| 실험 | 핵심 결과 | 해석 |
| --- | --- | --- |
| Random CV 후보 비교 | LightGBM 전체 특성: AP `0.1860`, ROC-AUC `0.6863` | 확률 순위 성능이 가장 높았지만 threshold `0.50`에서는 Fail을 예측하지 못함 |
| 특징 선택 비교 | L1 선택 모델: 평균 약 187개 feature, 전체 L1 balanced와 유사한 Random CV 성능 | 경량·해석 후보였으나 최종 후보는 아님 |
| Time Validation | LightGBM 전체 특성: AP `0.0874`, ROC-AUC `0.5666`; L1 선택: AP `0.0637` | 시간 구간에서 성능이 하락했으며 LightGBM이 상대적으로 우세 |
| OOF threshold 비교 | `0.000475`: Recall `0.329`, Precision `0.218`, F1 `0.262`, FN `49`, FP `86` | OOF F1 최대의 프로젝트 평가용 후보값 |
| 고정 threshold Time Validation | LightGBM `0.000475`: TP `0`, FP `4`, FN `17`, Recall `0.0000` | Time 구간에서 Fail을 검출하지 못해 모델·threshold를 아직 동결할 수 없음 |

`lightgbm_all`은 Random CV와 초기 Time Validation의 AP 기준으로 상대적으로 우세했던 후보지만, 고정 threshold `0.000475`를 적용한 Time Validation에서 Fail을 검출하지 못했다. 따라서 현재 최종 후보·최종 threshold는 **미확정**이며, Time Train 내부의 시간 순서 검증으로 개선 실험을 진행한 뒤 다시 선정해야 한다. `0.000475`는 현장 자동 폐기나 Lot hold 기준이 아니라, 현장 비용·재검사 처리 용량 정보가 없는 상태에서 기록한 프로젝트용 OOF 비교값이다.

## 데이터 분할 원칙

- Random split: Train / Validation / Test = **70 / 15 / 15**, 클래스 비율을 보존합니다.
- Time-based split: Timestamp 기준으로 과거 데이터를 학습하고 이후 데이터를 평가합니다.
- Test set은 모델·특징·하이퍼파라미터·임계값을 확정한 뒤 **최종 1회만** 사용합니다.

## 시작하기

현재 저장소는 데이터 준비부터 OOF threshold 비교까지 구현된 상태입니다. 아래 순서로 실행 환경을 준비합니다.

```bash
git clone <repository-url>
cd secom-waferguard
python -m venv .venv
```

Windows PowerShell에서는 가상환경을 활성화한 뒤 의존성을 설치합니다.

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 설정 구조

범용 코어와 데이터셋별 가정을 분리합니다.

- `config.json`: 난수 시드, 후보 모델 비교용 `default_threshold`, CV처럼 데이터셋과 독립적인 실험 조건
- `configs/datasets/secom.json`: SECOM의 원본 source, 입력 경로, label 값, 컬럼 역할, timestamp 형식, feature 타입·선택 규칙, 데이터 품질 기준

`src/modeling_config.py`는 두 설정을 함께 읽어 `ModelingConfig`와 `DatasetSpec`으로 검증합니다. `src/dataset_schema.py`는 Profile 기준으로 label·metadata·feature 컬럼과 수치형 규칙을 검증합니다. 따라서 새 데이터셋에는 코어 코드를 고치지 않고 같은 형식의 Dataset Profile을 추가합니다.

`src/modeling_metrics.py`는 Dataset Profile에서 전달받은 정상·불량 label을 기준으로 Recall, AP, Precision, F1, ROC-AUC와 confusion matrix를 공통 계산합니다.

주요 함수에는 입력 데이터의 역할, 처리 결과, 평가 구간 분리 이유를 한국어 docstring으로 설명합니다. 특히 `src/modeling_preprocessing.py`와 `src/split_contract.py`는 공통 전처리·누수 방어 규칙을, `src/threshold_policy.py`는 threshold 정책 선택 기준을 설명합니다. `src/step5_model_compare.py`부터 `src/step9_error_analysis.py` 및 `src/temporal_validation.py`에는 각 실험의 데이터 사용 범위와 검증 목적을 기록해 코드만 읽어도 평가 흐름을 따라갈 수 있도록 했습니다.

담당 B의 모델 정의·Pipeline 생성·fold별 학습·양성 확률 추출은 `src/modeling_models.py`로 통합했습니다. 각 Step은 실험 실행과 결과 기록을 맡으며, 공통 데이터 분리는 `dataset_schema.py`, 시간 fold 생성은 `split_contract.py`, threshold 출처 검사는 `threshold_policy.py`를 사용합니다. 기존 Step 실행 명령과 CSV 열은 유지합니다. Step 8의 XGBoost에도 `--n-jobs`가 적용되며, Step 9는 `--experiment`로 사전 선택한 모델을 지정할 수 있습니다. 기본값은 기존 `lightgbm_all`입니다.

`quality_rules.missing_ratio_threshold`와 `drop_zero_variance`는 공통 Pipeline의 `quality_filter` 단계에 적용됩니다. Validation에는 Train에서 결정한 센서 목록을 그대로 사용합니다. Baseline·모델 비교·특징 비교·시간 검증 결과 CSV의 `quality_filter_log`는 학습별 제거 개수·센서명·기준을 JSON 배열로 기록합니다. OOF는 점수 CSV 옆의 `<점수파일명>_quality_filter.csv`에 fold별 기록을 저장합니다. 품질 필터 도입 전의 성능·OOF threshold는 과거 결과이며, 필터 적용 후 재실행 결과는 `logs/quality_filtered_20261006/`에 저장했습니다.

### Dataset Profile 추가

새 데이터셋을 적용할 때는 `configs/datasets/`에 JSON Profile을 추가하고 `config.json`의 `dataset_profile` 경로만 변경합니다. Profile에는 최소한 입력 경로, label 컬럼과 정상·불량 값, timestamp 컬럼·형식(사용 시), ID·그룹·제외 컬럼 역할, feature 타입·선택 방식, 데이터 품질 규칙을 정의합니다.

`prefix` 방식은 지정한 접두어를 가진 컬럼만 feature로 사용합니다. `all_except_metadata` 방식은 label·timestamp·ID·그룹·명시적 제외 컬럼을 제외한 모든 컬럼을 feature로 사용합니다. 문자열 feature는 `feature_types.categorical_columns`에 명시해야 하며, 그 외 feature는 수치형이어야 합니다. Profile 검증에 실패하면 split·모델링 전에 오류가 발생합니다.

원본 파일을 병합해야 하는 데이터셋은 Profile의 `ingestion`에 adapter 이름, source 목록, 각 source의 `read_csv_options`, adapter 전용 결합 규칙을 정의합니다. 같은 수집 형식이면 Profile만 추가하고, 형식이 다를 때에만 adapter registry에 처리기를 추가합니다.

#### `configs/datasets/` JSON 설정 방법

1. `configs/datasets/<dataset_id>.json`을 만들고 아래 필수 항목을 정의합니다.
2. 루트 [config.json](config.json)의 `dataset_profile`을 새 JSON의 상대 경로로 변경합니다.
3. `python -m unittest discover -s tests -v`로 Profile 구조와 범용 코어를 검증합니다.

병합이 끝난 canonical CSV가 이미 있다면 `ingestion` 없이 다음처럼 작성할 수 있습니다.

```json
{
  "dataset_id": "factory_a",
  "input_path": "data/processed/factory_a_merged.csv",
  "columns": {
    "label": "result",
    "timestamp": null,
    "timestamp_format": null
  },
  "column_roles": {
    "id_columns": ["wafer_id"],
    "group_columns": ["lot_id"],
    "excluded_feature_columns": ["operator_note"]
  },
  "labels": {
    "negative": "PASS",
    "positive": "FAIL"
  },
  "feature_columns": {
    "selection": "all_except_metadata",
    "prefix": null
  },
  "feature_types": {
    "categorical_columns": ["equipment_type"]
  },
  "quality_rules": {
    "missing_ratio_threshold": 0.5,
    "drop_zero_variance": true
  }
}
```

`positive`는 Fail(불량), `negative`는 Pass(정상) label입니다. `id_columns`, `group_columns`, `excluded_feature_columns`은 feature 선택에서 항상 제외합니다. 명시하지 않은 feature는 수치형이어야 하고, 문자열·category feature는 `categorical_columns`에 넣어야 합니다.

원본 feature 파일과 metadata 파일을 병합하는 경우에는 위 JSON에 다음 `ingestion` 블록을 추가합니다. `feature_metadata_pair`는 같은 행 순서의 두 source를 결합하는 내장 adapter입니다.

```json
{
  "ingestion": {
    "adapter": "feature_metadata_pair",
    "sources": [
      {
        "name": "features",
        "path": "data/raw/factory_a_features.csv",
        "read_csv_options": {"header": 0}
      },
      {
        "name": "metadata",
        "path": "data/raw/factory_a_metadata.csv",
        "read_csv_options": {"header": 0}
      }
    ],
    "adapter_options": {
      "feature_source": "features",
      "metadata_source": "metadata",
      "metadata_columns": ["result"]
    }
  }
}
```

`prefix` 선택 방식과 이름 없는 feature 행렬을 사용할 때는 `feature_columns.prefix`를 지정하고, source의 `read_csv_options`에 `"header": null`을 설정합니다. `feature_metadata_pair`가 아닌 원본 결합 방식은 Profile만으로 처리할 수 없으므로 adapter registry에 별도 처리기를 추가해야 합니다.

### 자동 테스트

현재 구현된 범용 코어는 아래 명령으로 검증합니다.

```powershell
python -m unittest discover -s tests -v
```

UCI의 원본 데이터 CSV는 저장소에 포함하지 않습니다. [UCI SECOM Data Set](https://archive.ics.uci.edu/dataset/179/secom)에서 `secom.data`, `secom_labels.data`를 받아 `data/raw/`에 저장합니다. 원본 파일은 수정하지 않고, 아래 Script로 재현 가능한 산출물을 생성합니다.

### Day 1 데이터 준비

```powershell
python src/step1_merge_data.py
python src/step2_data_check.py
```

`step1_merge_data.py`는 Profile의 `ingestion` source 정의로 원본 파일을 병합하여 canonical 입력(`data/processed/secom_merged.csv`)을 생성합니다. 따라서 원본 경로나 읽기 옵션, label/timestamp 열 순서를 코드에 고정하지 않아 같은 수집 형식의 다른 데이터셋 Profile에도 처리 흐름을 적용할 수 있습니다.

`step2_data_check.py`는 결측률·상수 feature·timestamp·label 분포를 `logs/dataset_log.csv`에 기록합니다. 이 결과는 전체 데이터의 EDA 후보 정보일 뿐이며, 이를 사용해 `secom_cleaned.csv`를 만들거나 모델 feature를 전역에서 제거하지 않습니다. 실제 결측 처리와 feature 선택은 Train/CV 학습 fold 안에서만 fit하여 데이터 누수를 방지합니다. 두 CSV 산출물은 원본과 Profile만 있으면 재생성 가능하므로 Git에서 추적하지 않습니다.

### 모델링 실험

XGBoost의 전체 센서·gain 중요도 Top-20·RF 중요도 Top-20은 아래 명령으로 비교합니다. 최종 분류기는 모두 같은 XGBoost이며 전처리·선택은 학습 fold 내부에서만 fit합니다. `--experiments`를 생략하면 기존 PCA·L1·LightGBM 비교 목록을 실행합니다. `--details-dir`는 비어 있는 새 폴더를 지정하며 fold별 지표·선택 센서·설정을 저장합니다.

```powershell
python src/step6_feature_compare.py --train data/splits/integrated/time_train.csv --experiments xgboost_all xgboost_top_20 xgboost_rf_top_20 --output logs/xgboost_top20_run/feature_compare.csv --details-dir logs/xgboost_top20_run/details --n-jobs 2
```

반복 CV AP는 전체 444개 **0.2490**, RF 선택 20개 **0.2459**, XGBoost 선택 20개 **0.2044**였습니다. 성능 동등성이나 최종 센서 집합은 확정하지 않았습니다. 현재는 최종 분류기에 20개가 들어가는 구조로, 원본 20개만 받는 추론 Pipeline은 아직 준비하지 않았습니다. 실제 결과는 `logs/xgboost_top20_20261006/`와 [보고서 26절](reports/report.md#26-xgboost-센서-선택-방법별-top-20-비교-2026-10-06)에 기록했습니다. 이번 비교에서 외부 Validation·Test 및 threshold 정책은 변경하지 않았습니다.

후속 시간순 검증은 같은 코어에서 전체 XGBoost와 RF Top-20만 지정해 실행할 수 있습니다. 과거 학습→다음 구간 평가 3회와 각 과거 구간의 내부 시간순 OOF 2회를 사용하며, 선택 센서·OOF threshold·정책 미충족을 별도로 기록합니다.

```powershell
python src/feature_time_compare.py --train data/splits/integrated/time_train.csv --experiments xgboost_all xgboost_rf_top_20 --output-dir logs/xgboost_top20_time_run --n-jobs 2
```

실제 시간 구간 평균 AP는 전체 **0.1553**, RF Top-20 **0.1177**이었으며 세 구간 모두 전체 모델이 높았습니다. 과거 OOF의 F1 최대 진단 문턱을 다음 구간에 적용하면 RF Top-20의 합산 Recall은 69.23%지만 양성 판정 비율도 59.47%였습니다. 90%·20% 정책은 두 모델·세 구간 모두 미충족으로, “20개로 같은 성능 유지”는 아직 입증하지 못했습니다. 결과는 `logs/xgboost_top20_time_20261006/`와 [보고서 27절](reports/report.md#27-xgboost-전체-vs-rf-top-20-시간순-검증-2026-10-06)에 기록했으며 외부 Validation·Test는 사용하지 않았습니다.

시간순 센서 축소 비교는 `src/feature_time_compare.py`로 실행합니다. 전체 센서·LightGBM 중요도 Top-50/20·RF 중요도 Top-50/20을 동일한 LightGBM 분류기로 비교합니다. 선택기는 각 학습 fold 내부에서만 fit하며 기존 시간 검증 코어의 외부 3구간·내부 시간순 OOF 2구간을 재사용합니다.

```powershell
python src/feature_time_compare.py --train data/splits/integrated/time_train.csv --output-dir logs/feature_time_run --n-jobs 2
```

`selected_features.csv`는 각 fit의 실제 특징 목록·선택 중요도·순위, `feature_frequency.csv`는 외부 fit 3회·내부 OOF fit 6회 각각의 선택 빈도입니다. `candidate_features.csv`는 전체 Time Train에서 만든 **최종 미확정 후보**이며 이를 이용해 과거 CV 특징을 전역 고정하지 않습니다. 범주형을 One-Hot으로 확장한 특징은 원본 센서 수와 구분합니다. 실제 결과는 `logs/feature_time_20261006/`, 보고서 24절, 분석 노트북 7절에서 확인할 수 있습니다. 외부 Validation·Test는 읽지 않고 기존 실행기 대상도 자동 변경하지 않습니다.

정상만 학습하는 이상 탐지는 `src/anomaly_compare.py`로 실행합니다. 센서 품질 필터·대치·표준화와 Isolation Forest 또는 PCA90을 각 학습 구간의 정상 데이터에만 fit합니다. Isolation Forest는 `-score_samples`, PCA는 표준화된 입력의 평균 제곱 복원오차를 사용해 클수록 이상으로 통일합니다. 두 점수는 확률이 아니므로 기본 확률 threshold 0.5를 적용하지 않습니다. Isolation Forest의 점수 방향은 [공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)를 따릅니다.

```powershell
python src/anomaly_compare.py --train data/splits/integrated/time_train.csv --output-dir logs/anomaly_run --supervised-dir logs/temporal_quality_filtered_20261006_final --n-jobs 2
```

출력 폴더는 새 폴더여야 합니다. 기존과 같은 외부 시간 구간 3개·내부 시간순 OOF 구간 2개를 사용하며 초기 미예측 행은 제외합니다. `fold_results.csv`는 OOF F1 최대 진단 후보의 다음 구간 성능, `policy_selection.csv`는 현재 설정 정책의 선택·미충족 결과입니다. 정책 미충족이어도 AP와 F1 진단 비교는 저장하지만 임의 후보를 정책 통과로 취급하지 않습니다. `anomaly_compare.csv`에는 생성 계약·학습 설정·시간 경계가 맞는 기존 지도학습 진단을 연결합니다. `quality_filter.csv`에는 모든 fit의 정상 표본 ID·제거 개수·PCA 성분 수가 기록됩니다. 점수 규모가 학습 구간에 따라 달라질 수 있어 OOF 문턱의 미래 전이는 별도 검증이 필요합니다. 최종 Test와 외부 Validation은 읽지 않습니다.

변경 후 OOF와 시간 검증을 연결할 때는 `src/run_evaluation.py`를 사용합니다. `--change preprocessing`, `model`, `data`, `split`은 Step 8 OOF·threshold 비교표를 재생성한 뒤 Step 7 시간 검증을 실행합니다. `--change threshold`는 이 실행기로 성공한 같은 폴더의 비교표를 재사용하며 `--threshold`가 필수입니다. OOF 생성이 실패하면 후속 평가를 중단하고, 실패한 실행 폴더의 과거 비교표 재사용도 거부합니다. 상태와 설정은 `execution.json`에 기록합니다.

```powershell
# 전처리 변경: OOF 생성 → 후보 threshold 선택 → 시간 검증
python src/run_evaluation.py --change preprocessing --train data/splits/integrated/time_train.csv --validation data/splits/integrated/time_valid.csv --run-dir logs/evaluation_run --n-jobs 2

# threshold 변경: 같은 실행 폴더의 비교표 후보값을 지정
python src/run_evaluation.py --change threshold --threshold <비교표의_반올림하지_않은_값> --train data/splits/integrated/time_train.csv --validation data/splits/integrated/time_valid.csv --run-dir logs/evaluation_run --n-jobs 2

# config.json의 threshold_policy만 변경: 기존 OOF로 정책 후보 재선택
python src/run_evaluation.py --change policy --train data/splits/integrated/time_train.csv --validation data/splits/integrated/time_valid.csv --run-dir logs/evaluation_run --n-jobs 2
```

threshold 생략 시 `config.json`의 `threshold_policy`로 OOF 후보를 선택합니다. `min_recall: 0.7`은 최소 Recall, `max_reinspection_ratio: 0.2`는 `(TP + FP) / 전체 표본 수`의 상한입니다. 현재 값은 프로젝트용 예시이며 현장 표준이나 공동 확정 기준이 아닙니다. `selection_rule: "min_reinspection"`은 조건을 충족하는 후보 중 재검사 비율 최소 → Recall 최대 → threshold 최대 순으로 선택합니다. `on_infeasible: "report_only"`는 후보가 없으면 F1 최대값으로 대체하지 않고 미충족을 기록하며 시간 검증을 생략합니다. 후보별 판단은 `policy_candidates.csv`, 실행 여부는 `execution.json`에 기록합니다. 생략 시 과거 `time_validation.csv`가 남아 있어도 이번 실행 결과로 해석하지 않습니다.

`--change policy`는 정책만 바뀐 경우 OOF를 재사용하며 `--threshold`는 지정하지 않습니다. 학습 설정·Train 경로가 바뀌면 재사용을 거부합니다. 같은 경로의 파일 내용이나 코드 변경은 자동 감지하지 않으므로 해당 변경 유형으로 OOF부터 재생성해야 합니다. 수동 `--threshold` 지정은 정책을 우회하는 연구용 실행이며 정책 충족을 뜻하지 않습니다. 연결 대상은 `--experiment`로 `lightgbm_all`, `xgboost`, `xgboost_scale_pos_weight` 중에서 지정합니다(기본값 `lightgbm_all`). Baseline·모델 비교·특징 비교·오류 분석은 아래 개별 명령으로 실행합니다. OOF 재사용 시에도 Step 7은 동일 Train으로 모델을 다시 학습합니다. 개별 Step 7 실행에는 실행 상태 검사가 적용되지 않습니다. 최종 Test 평가 전에 정책을 확정하고 Test 결과에 맞춰 바꾸지 않습니다.

`temporal_validation.py`는 기존 F1 최대 진단을 유지하면서 `policy_selection.csv`에 내부 OOF의 정책 충족 여부·선택값과 다음 구간 성능을 별도로 기록합니다. 미충족 구간에는 threshold와 정책 기반 평가값을 만들지 않습니다. 기존 보고서의 수치는 과거 실행 결과이며 새 정책 실험 결과로 간주하지 않습니다.

팀원의 `step3_split.py`를 공식 진입점으로 사용한다. 기존 Random/Time 분할·EDA·클래스 요약·누수 검사 함수를 유지하면서 공통 Profile 검증과 원본 ID·split 계약을 통합했다. 별도 `step3_split_data.py`는 제거했다. 기본 출력은 `data/splits/integrated/`, 점검 로그는 `logs/step3_integrated/`, 그림은 `reports/figures/step3/`이며 기존 split CSV는 덮어쓰지 않는다.

```powershell
# 시간 holdout을 먼저 분리 (70/15/15는 목표 비율이며 동일 timestamp·그룹은 분리하지 않음)
python src/step3_split.py

# Time Train/Validation baseline
python src/step4_baseline.py --train data/splits/integrated/time_train.csv --validation data/splits/integrated/time_valid.csv --split-strategy time --output logs/integrated/baseline.csv

# 동일한 Time Train 내부 반복 CV 후보 모델 비교
python src/step5_model_compare.py --train data/splits/integrated/time_train.csv --output logs/integrated/model_compare.csv

# 위 후보 목록에는 XGBoost 일반·scale_pos_weight 실험도 포함됨

# PCA·L1·LightGBM Top-K 특징 선택 비교
python src/step6_feature_compare.py --train data/splits/integrated/time_train.csv --output logs/integrated/feature_compare.csv --n-jobs -1

# Train 내부 세 시간 구간: 기본 threshold·계층 OOF·시간순 OOF 전이 비교
python src/temporal_validation.py --train data/splits/integrated/time_train.csv --output-dir logs/temporal_validation_run --n-jobs 2

# LightGBM 전체 특성의 OOF 확률·threshold 비교
python src/step8_threshold_oof.py --train data/splits/integrated/time_train.csv --oof-output logs/integrated/oof_predictions.csv --threshold-output logs/integrated/threshold_compare.csv --n-jobs -1

# CV 결과에서 선택한 XGBoost 후보의 OOF threshold 비교
python src/step8_threshold_oof.py --train data/splits/integrated/time_train.csv --experiment xgboost --oof-output logs/integrated/xgboost_oof_predictions.csv --threshold-output logs/integrated/xgboost_threshold_compare.csv --n-jobs -1

# 시간 검증: 생략 시 후보 비교 기본 threshold 사용. 새 Train에서 정한 값만 --threshold로 주입
python src/step7_time_validation.py --train data/splits/integrated/time_train.csv --validation data/splits/integrated/time_valid.csv --experiments lightgbm_all --output logs/integrated/time_validation.csv --n-jobs -1

# 후보 비교에서 XGBoost를 선택한 경우 같은 시간 분할에서 비교
python src/step7_time_validation.py --train data/splits/integrated/time_train.csv --validation data/splits/integrated/time_valid.csv --experiments xgboost --output logs/integrated/xgboost_time_validation.csv --n-jobs -1
```

`--n-jobs -1`은 해당 옵션을 제공하는 Step 6~9 및 시간순 실험에서 트리 모델 내부 병렬 처리를 사용한다. Step 5는 모델 내부 병렬 수 1로 실행한다. 환경 자원이 제한된 경우 기본값인 `1`을 사용한다. XGBoost는 `requirements.txt`에 추가했으므로 기존 환경에서는 `pip install -r requirements.txt`로 의존성을 설치한 뒤 실행한다. XGBoost의 반복 CV·시간순 내부 검증·단일 OOF 결과는 [보고서 25절](reports/report.md#25-xgboost-추가-비교-2026-10-06)과 `logs/xgboost_20261006/`에 기록했다. 기본 모델 CV AP는 0.2490으로 LightGBM 0.2424보다 소폭 높았지만, Recall 91.03%에서 재검사율은 63.96%로 20% 상한을 넘었다. 최종 모델·threshold는 미확정이며 이번 실험에서 외부 Validation·Test는 사용하지 않았다.

`temporal_validation.py`는 Train 파일만 받으며, 고유 timestamp 단위의 3개 외부 시간 구간에서 지정한 후보 모델을 비교합니다. 기본 목록에는 LightGBM·Random Forest·균형 가중 Logistic Regression·XGBoost가 포함됩니다. 내부 계층 OOF는 설정의 fold 수와 1회 반복, 내부 시간순 OOF는 기본 2개 확장형 fold를 사용합니다. 초기 미예측 행은 제외하고 `oof_coverage.csv`에 기록합니다. `folds.csv`, `fold_results.csv`, `summary.csv`, 예측·OOF·threshold·제거 로그와 `temporal_run.json`을 저장하며 기존 결과를 보존하기 위해 비어 있는 출력 폴더를 요구합니다. 모델·구간별 threshold가 다르므로 합산 Recall·Precision은 단일 최종 모델의 성능이 아닙니다.

Step 3을 재실행할 때는 새로운 `--output-dir`과 `--log-dir`을 지정한다. `--config`, `--input`, `--figures-dir`로 경로를 변경할 수 있고, `--skip-eda`는 분할·검사만 수행한다. 그룹이 없는 Random split은 기존 stratify 방식을 유지한다. 그룹이 선언되면 GroupShuffleSplit을 사용하므로 정확한 행 비율·class 비율은 보장되지 않으며 실제 비율을 요약에서 확인한다. 누수 검사 실패 시 split CSV를 저장하지 않고 오류로 종료한다.

위 LightGBM 명령은 실행 예시이며 최종 후보 확정을 뜻하지 않는다. 기존 `0.000475`를 새 평가 threshold로 그대로 승계하지 않는다. Step 8의 기본 `--score-method single`은 1회 K-fold의 샘플별 OOF 예측을 사용한다. 후보 비교의 5-fold × 5-repeats 정책은 그대로 유지한다. `--score-method repeated_mean`은 반복 평균 분석용이며 결과의 `threshold_use=analysis_only`로 구분한다. 단일 OOF도 fold 학습과 전체 Train 재학습의 확률 척도 일치를 보장하지 않으며, 시간순 내부 검증·확률 보정은 후속 점검 대상이다.

새 OOF 비교표에서 threshold를 선택한 뒤 Step 7에 `--threshold-report logs/integrated/threshold_compare.csv --threshold <비교표의_반올림하지_않은_값>`을 전달하면 데이터셋·Train 생성 계약·모델·단일 OOF 방식과 값의 존재 여부를 검사한다. `<…>`는 실제 숫자로 바꿔야 한다. 비교표 없이 실행하는 기존 CLI는 탐색용 호환 경로이며 출처 확인을 보장하지 않는다. 같은 계약이라도 모델 설정을 변경했다면 비교표를 다시 생성해야 한다.

새 split의 `__source_row_id`는 원본 CSV 내 행 위치, `__split_role`은 train/validation/test, `__split_protocol_id`는 원본·Profile·비율의 생성 계약 해시다. 이 컬럼들은 모델 입력에서 제외한다. Baseline·시간 검증은 중복 행과 계약 혼합을 검사하고, 후보 비교·OOF는 Validation/Test 역할의 입력을 거부한다. 기존 metadata 없는 CSV도 사용할 수 있지만 출처까지 보장하지 못하므로 탐색용으로만 취급한다.

모든 모델링 Pipeline은 센서 품질 필터 이후 수치형 median 대치와 선언된 범주형 최빈값 대치/One-Hot Encoding을 공통으로 사용한다. SECOM의 결측률 50% 초과 기준에서는 전부 결측인 컬럼도 제거한다. Profile에서 품질 제거를 비활성화한 경우에는 전부 결측인 수치형 컬럼을 대치 단계에서 0으로 보존한다. Top-K와 특징 수는 변환 후 기준이며 범주형 입력에서는 센서 원본 컬럼 수와 One-Hot 특징 수가 다를 수 있다.

## 프로젝트 디렉터리 구조

```text
secom-waferguard/
├── README.md
├── requirements.txt
├── config.json              # 데이터셋과 독립적인 모델링 실험 조건
├── configs/
│   └── datasets/
│       └── secom.json       # SECOM Dataset Profile
├── data/
│   ├── raw/                 # 수정하지 않는 원본 데이터
│   ├── processed/           # 병합 canonical 데이터
│   └── splits/              # 데이터 분할 정보
├── src/
│   ├── modeling_config.py   # 공통 실험 설정·Dataset Profile 로드·검증
│   ├── dataset_schema.py    # Profile 기반 feature·label 구조 검증
│   ├── modeling_metrics.py  # Dataset Profile label 기반 공통 평가 함수
│   ├── modeling_models.py   # 모델·실험 Pipeline 정의, 공통 fold 학습·양성 확률
│   ├── modeling_preprocessing.py # 모델 간 공통 수치형·범주형 전처리
│   ├── split_contract.py    # split 역할·중복·시간 경계·그룹 CV 검사
│   ├── step1_merge_data.py  # Profile ingestion 기반 원본 병합
│   ├── step2_data_check.py  # EDA용 품질 로그 생성
│   ├── step3_split.py       # 팀원 코드 기반 Random 탐색 / Time holdout 생성·검증
│   ├── step3_drift_check.py # Time Train/Validation 분포 변화 진단
│   ├── step4_baseline.py    # Train/Validation baseline 평가
│   ├── step5_model_compare.py # 후보 모델 반복 CV 비교
│   ├── step6_feature_compare.py # PCA·L1·Top-K 특징 선택 비교
│   ├── step7_time_validation.py # 사전 선택 후보의 시간 구간 검증
│   ├── step8_threshold_oof.py # OOF threshold 비교
│   ├── run_evaluation.py     # 변경 범위별 OOF·시간 검증 실행 관리
│   ├── temporal_validation.py # Train 내부 시간순 검증·OOF 전이 비교
│   ├── anomaly_compare.py    # 정상 전용 Isolation Forest·PCA 복원오차 시간순 비교
│   ├── feature_time_compare.py # 전체·Top-50/20 시간순 비교·선택 센서 및 빈도
│   └── step9_error_analysis.py # Time Validation FN/FP 분석
├── tests/                   # 설정·schema·평가·실험 함수 자동 테스트
├── logs/                    # 데이터·실험 결과 CSV
├── models/                  # model.joblib, model_card.json
└── reports/                 # 프로젝트 분석 보고서 및 결과 시각화
    ├── report.md            # 최종 성능, 오류 분석, 한계 및 개선 방향 보고서
    ├── analysis.ipynb       # 기존 로그를 표·그래프로 확인하는 분석 전용 노트북
    └── figures/             # README, 보고서, 발표에 사용하는 시각화 결과
```

## 산출물

| 구분 | 산출물 |
| --- | --- |
| 데이터 품질 | `dataset_log.csv`, `split_summary.csv` |
| 모델 실험 | `baseline_result.csv`, `feature_compare.csv`, `model_compare.csv`, `time_validation_compare.csv` |
| Threshold | `lightgbm_oof_predictions.csv`, `threshold_compare.csv` |
| 최종 평가 | `test_result.csv`, confusion matrix, PR curve |
| 모델 배포 | `model.joblib`, `model_card.json`, `predict_cli.py` |
| 보고서 | `reports/report.md` |

## 한계와 주의사항

- 불량 표본이 약 7%이므로 Accuracy만으로 모델을 판단하지 않습니다.
- 데이터 규모가 작고 불량 샘플이 적어, 단일 split의 성능만으로 일반화 성능을 보장할 수 없습니다.
- 센서 중요도는 인과관계를 직접 뜻하지 않으며, 공정 전문가의 검토가 필요합니다.
- 시간 순서 평가에서 성능이 하락하면 공정 조건 변화 또는 데이터 드리프트를 추가로 분석해야 합니다.

## Legacy Notebook

예전 `src/final_model.ipynb`는 Git에서 삭제됐으며 현재 실행 경로에 포함하지 않습니다. 병합·품질 점검·모델링은 Script와 범용 코어로 실행합니다.

### 실험 결과 확인용 노트북

`reports/analysis.ipynb`를 VS Code에서 열고 Python 커널을 선택한 뒤 **Run All**을 실행합니다. 커널이 없다면 프로젝트 루트에서 `python -m pip install -r requirements-notebook.txt`로 선택 의존성을 설치합니다.

노트북은 기존 로그를 읽어 Recall–재검사 대상 비율, 70·80·90% 목표별 후보, threshold별 미검·오탐, 모델·특징 CV AP, 시간 구간별 성능을 표시합니다. 첫 설정 셀의 `POLICY_DIR`, `MODEL_DIR`, `TIME_DIR`에서 확인할 실행 폴더를 지정합니다. 정책 미충족으로 생략된 시간 검증은 현재 결과로 표시하지 않습니다. 학습·설정 변경·CSV 저장·최종 Test 읽기는 하지 않으며, 로그가 갱신되면 노트북을 다시 실행해야 화면에 반영됩니다. 화면의 시나리오 비교는 config.json이나 기존 실행 기록을 변경하지 않습니다.

6절에서는 `ANOMALY_DIR`의 이상 탐지 요약·정책 결과·지도학습 비교 그림을 확인합니다. 실제 실행 결과는 `logs/anomaly_20261006/`, 해석은 보고서 23절에 기록했습니다. MLP·Autoencoder 딥러닝 실험은 아직 구현하지 않았습니다.

7절에서는 `FEATURE_DIR`의 시간순 축소 비교·정책 결과·연구용 후보 센서·선택 빈도를 확인합니다. `FEATURE_EXPERIMENT`를 변경하면 선택 방법별 센서 자료를 볼 수 있습니다.

## License

이 프로젝트에서 사용하는 UCI SECOM 데이터셋은 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 라이선스를 따릅니다. 데이터 사용 시 UCI 데이터셋 출처를 함께 표기해 주세요.

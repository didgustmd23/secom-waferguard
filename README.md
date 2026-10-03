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
| 자동 테스트 | 완료 | 설정·schema·평가·병합·품질 점검 16개 테스트 |
| 데이터 병합·품질 점검 | 완료 | Profile 기반 canonical 병합과 재생성 가능한 품질 로그 |
| split·baseline·후속 모델링 | 예정 | Day 2~5 계획에 따라 구현 |

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

- 센서별 결측률을 점검하고, 결측률이 50%를 초과하는 센서를 제거합니다.
- 상수(Zero-variance) 센서를 제거합니다.
- 나머지 결측값은 `SimpleImputer(strategy="median")`로 처리합니다.
- Imputation, scaling, 특징 선택은 학습 데이터에만 fit되도록 `scikit-learn Pipeline` 안에서 수행합니다.

### 모델링 및 검증

| 단계 | 방법 |
| --- | --- |
| Baseline | Median imputation → StandardScaler → Logistic Regression |
| 특징 축소 | PCA, L1 Logistic Regression, Feature Importance 기반 Top-K |
| 후보 모델 | Logistic Regression, SVM, Random Forest, LightGBM |
| 불균형 처리 | `class_weight`, `scale_pos_weight` |
| 검증 | Repeated Stratified K-Fold CV |
| 핵심 지표 | Fail Recall, PR-AUC / AP |
| 보조 지표 | Precision, F1-score, ROC-AUC, 학습·추론 시간 |

최종 분류 임계값은 Out-of-Fold 예측에서 Recall, Precision, False Negative, False Positive의 trade-off를 비교하여 결정합니다.

## 데이터 분할 원칙

- Random split: Train / Validation / Test = **70 / 15 / 15**, 클래스 비율을 보존합니다.
- Time-based split: Timestamp 기준으로 과거 데이터를 학습하고 이후 데이터를 평가합니다.
- Test set은 모델·특징·하이퍼파라미터·임계값을 확정한 뒤 **최종 1회만** 사용합니다.

## 시작하기

현재 저장소는 프로젝트 문서와 실험 설계를 우선 정리한 상태입니다. 구현 후 아래 순서로 재현할 수 있도록 구성합니다.

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

아래 split·모델링 명령은 Day 2~5 구현 후 사용할 예정입니다.

```bash
python src/step3_split.py
python src/step4_baseline.py
python src/step5_feature_selection.py
python src/step6_model_compare.py
python src/step7_threshold.py
python src/step8_test.py
```

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
│   ├── step1_merge_data.py  # Profile ingestion 기반 원본 병합
│   ├── step2_data_check.py  # EDA용 품질 로그 생성
│   └── ...                  # 전처리, 학습, 평가, 추론 코드
├── tests/                   # 설정·schema·평가 함수 자동 테스트
├── logs/                    # 데이터·실험 결과 CSV
├── models/                  # model.joblib, model_card.json
└── reports/                 # 프로젝트 분석 보고서 및 결과 시각화
    ├── report.md            # 최종 성능, 오류 분석, 한계 및 개선 방향 보고서
    ├── analysis.ipynb
    └── figures/             # README, 보고서, 발표에 사용하는 시각화 결과
```

## 산출물

| 구분 | 산출물 |
| --- | --- |
| 데이터 품질 | `dataset_log.csv`, `split_summary.csv` |
| 모델 실험 | `baseline_result.csv`, `feature_compare.csv`, `model_compare.csv` |
| Threshold | `threshold_compare.csv` |
| 최종 평가 | `test_result.csv`, confusion matrix, PR curve |
| 모델 배포 | `model.joblib`, `model_card.json`, `predict_cli.py` |
| 보고서 | `reports/report.md` |

## 한계와 주의사항

- 불량 표본이 약 7%이므로 Accuracy만으로 모델을 판단하지 않습니다.
- 데이터 규모가 작고 불량 샘플이 적어, 단일 split의 성능만으로 일반화 성능을 보장할 수 없습니다.
- 센서 중요도는 인과관계를 직접 뜻하지 않으며, 공정 전문가의 검토가 필요합니다.
- 시간 순서 평가에서 성능이 하락하면 공정 조건 변화 또는 데이터 드리프트를 추가로 분석해야 합니다.

## Legacy Notebook

`src/final_model.ipynb`는 초기 데이터 탐색 과정에서 만든 notebook입니다. 현재 재현 가능한 실행 경로에는 포함하지 않으며, 병합·품질 점검·모델링은 계획된 Script와 범용 코어를 기준으로 구현합니다.

## License

이 프로젝트에서 사용하는 UCI SECOM 데이터셋은 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 라이선스를 따릅니다. 데이터 사용 시 UCI 데이터셋 출처를 함께 표기해 주세요.

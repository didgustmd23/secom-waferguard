# SECOM WaferGuard

> UCI SECOM 센서 데이터를 활용한 반도체 공정 불량 웨이퍼 조기 탐지 프로젝트

고차원·결측·불균형 특성을 가진 반도체 제조 공정 데이터를 분석하여, 최종 검사 이전에 불량(Fail) 가능성이 높은 생산 단위를 탐지합니다. 단순 정확도보다 불량 미검(False Negative)을 줄이는 데 초점을 두며, 핵심 센서 선택과 재현 가능한 머신러닝 파이프라인 구축을 목표로 합니다.

실험 일정, 역할 분담, 완료 기준은 [PLAN.md](PLAN.md)에서 관리합니다.

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

### 실험 설정

모델링의 공통 실험 조건은 루트의 `config.json`에서 관리합니다. 난수 시드, Fail label, 기본 threshold, 반복 교차 검증 횟수, 평가 지표, 특징 선택 Top-K, 후보 모델 목록을 수정할 수 있습니다.

`src/modeling_config.py`는 `config.json`을 읽고 자료형·범위를 검증해 모든 모델링 단계에서 같은 설정을 사용하도록 합니다. `src/modeling_metrics.py`는 Fail label(`1`) 기준의 Recall, AP, Precision, F1, ROC-AUC와 confusion matrix를 공통으로 계산합니다.

UCI의 원본 센서 데이터와 Label/Timestamp 파일은 `data/raw/`에 저장합니다. 원본 파일은 수정하지 않고, 모든 변환은 스크립트로 재현합니다.

구현 완료 후의 실행 순서는 다음과 같습니다.

```bash
python src/step1_merge_data.py
python src/step2_data_check.py
python src/step3_split.py
python src/step4_baseline.py
python src/step5_feature_selection.py
python src/step6_model_compare.py
python src/step7_threshold.py
python src/step8_test.py
```

## 예정 디렉터리 구조

```text
secom-waferguard/
├── README.md
├── requirements.txt
├── config.json              # 모델링 공통 실험 조건
├── data/
│   ├── raw/                 # 수정하지 않는 원본 데이터
│   ├── processed/           # 병합·정제 데이터
│   └── splits/              # 데이터 분할 정보
├── src/
│   ├── modeling_config.py   # config.json 로드·검증
│   ├── modeling_metrics.py  # Fail 중심 공통 평가 함수
│   └── ...                  # 전처리, 학습, 평가, 추론 코드
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

## License

이 프로젝트에서 사용하는 UCI SECOM 데이터셋은 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 라이선스를 따릅니다. 데이터 사용 시 UCI 데이터셋 출처를 함께 표기해 주세요.

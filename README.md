# SECOM WaferGuard

## 반도체 공정 센서 기반 불량 웨이퍼 조기 탐지

> **Semiconductor Process Failure Early Detection with Feature Selection & Imbalanced Learning**

---

# 1. 프로젝트 기본 정보

| 항목 | 내용 |
|---|---|
| **프로젝트명** | SECOM WaferGuard |
| **한글명** | 반도체 공정 센서 기반 불량 웨이퍼 조기 탐지 |
| **영문명** | SECOM WaferGuard: Semiconductor Process Failure Early Detection |
| **GitHub Repository** | `secom-waferguard` |
| **데이터셋** | UCI SECOM Data Set |
| **문제 유형** | 고차원·결측·불균형 이진 분류 |
| **개발 환경** | Python 3.10 / `ml310` |
| **프로젝트 기간** | 5일 |
| **팀 구성** | 2인 |
| **최종 모델** | `model.joblib` |
| **모델 정보** | `model_card.json` |

### GitHub Description

```text
Early detection of semiconductor process failures using UCI SECOM sensor data, feature selection, imbalanced learning, and repeated cross-validation.
```

---

# 2. 프로젝트 개요

## 2.1 프로젝트 배경

현대의 반도체 제조 공정에서는 센서와 공정 측정 지점에서 수집되는 수많은 신호를 지속적으로 모니터링한다.

하지만 모든 센서 신호가 제품의 품질이나 수율을 예측하는 데 동일하게 중요한 것은 아니다. 실제 측정 데이터에는 다음과 같은 정보가 혼합되어 있다.

- 제품 품질과 관련된 유용한 정보
- 관련성이 낮은 정보
- 노이즈
- 결측값

따라서 수많은 센서 중 실제 불량 및 수율에 영향을 주는 핵심 신호를 식별하는 것이 중요하다.

본 프로젝트에서는 **UCI SECOM 데이터셋**을 활용하여 반도체 제조 공정에서 수집된 고차원 센서 데이터를 분석하고, 최종 검사 결과가 불량인 생산 단위를 조기에 탐지하는 머신러닝 모델을 구축한다.

단순히 높은 분류 정확도를 만드는 것보다 다음의 전체 머신러닝 과정을 체계적으로 수행하는 것을 목표로 한다.

```text
데이터 품질 점검
→ 데이터 분할
→ 데이터 누수 방지
→ Baseline
→ 특징 선택
→ 후보 모델 비교
→ 반복 교차검증
→ Threshold 결정
→ 오류 분석
→ 시간 변화 분석
→ 최종 Test
→ 모델 출고
```

---

## 2.2 팀 구성 및 역할

본 프로젝트는 **2인 팀**으로 진행한다.

| 팀원 | 담당 역할 | 주요 업무 |
|---|---|---|
| **팀원 A** | 데이터·분석 담당 | 데이터 수집·병합·정제, EDA, 데이터 분할, 오류 및 Drift 분석, README·리포트 |
| **팀원 B** | 모델링·평가 담당 | Pipeline, Baseline, 특징 선택, 후보 모델 비교, 반복 CV, Threshold, 모델 저장·추론 CLI |
| **공동** | 검토·출고·발표 | 데이터 누수 검토, 최종 모델·Threshold 결정, Test 평가, 재현성 검증, 발표 |

### 역할 분담 흐름

**팀원 A**

```text
Data → EDA → Split → Error Analysis → Documentation
```

**팀원 B**

```text
Baseline → Feature Selection → Modeling → CV → Threshold → Model Bundle
```

**공동**

```text
Model Selection → Final Test → Reproducibility → Presentation
```

---

# 3. 데이터셋

## 3.1 UCI SECOM Data Set (출처 : https://archive.ics.uci.edu/dataset/179/secom)

SECOM은 반도체 제조 공정에서 수집된 실제 공정 데이터를 기반으로 한 다변량 데이터셋이다.

원 데이터셋 설명에 따르면 각 데이터는 하나의 생산 단위를 나타내며, 해당 생산 단위에서 측정된 공정 특징과 Pass/Fail 결과가 제공된다.

| 항목 | 내용 |
|---|---|
| **Title** | SECOM Data Set |
| **분야** | Semiconductor Manufacturing Process |
| **데이터 특성** | Multivariate |
| **인스턴스 수** | 1,567 |
| **원문 Attribute 수** | 591 |
| **Attribute 특성** | Real |
| **관련 Task** | Classification, Causal Discovery |
| **결측값** | 있음 |
| **Fail 수** | 104 |
| **등록일** | 2008-11-19 |
| **저자** | Michael McCann, Adrian Johnston |
| **라이선스** | CC BY 4.0 |

> **주의:** UCI 설명에서 Attribute 수를 591개로 표현하는 부분과 프로젝트에서 센서 변수 수를 590개로 다루는 표현이 혼용될 수 있다. 실제 프로젝트에서는 원본 파일을 로드한 결과를 기준으로 **센서 열, Label, Timestamp를 구분하여 정확한 변수 수를 `dataset_log.csv`에 기록한다.**

---

## 3.2 데이터셋의 목적

반도체 제조 공정에서는 실제로 필요한 신호보다 훨씬 많은 센서와 공정 변수를 수집하는 경우가 많다.

이 중 일부는 수율과 밀접한 관련이 있지만 일부는 관련성이 낮거나 노이즈에 가깝다.

따라서 **Feature Selection**을 이용하여 제품 수율과 관련성이 높은 신호를 식별할 수 있다.

공정 엔지니어는 선택된 주요 신호를 활용하여 이후 공정에서 발생하는 **수율 변동(Yield Excursion)**의 주요 요인을 파악할 수 있다.

이를 통해 기대할 수 있는 효과는 다음과 같다.

- 공정 처리량 증가
- 문제 원인 파악 시간 감소
- 불필요한 센서 및 변수 감소
- 공정 학습 시간 감소
- 제품 단위당 생산 비용 감소

따라서 SECOM 데이터셋의 핵심은 단순 Pass/Fail 예측뿐 아니라 **제품 수율과 관련된 핵심 공정 특징을 찾는 것**에 있다.

---

## 3.3 데이터 구조

원 데이터는 크게 두 개의 파일로 구성된다.

### SECOM 데이터 파일

각 행은 하나의 생산 단위를 의미하며 각 열에는 공정에서 측정된 특징값이 저장되어 있다.

### Label 파일

각 생산 단위에 대응하는 다음 정보가 포함된다.

- Pass/Fail Label
- Date-Time Stamp

프로젝트 시작 단계에서 두 파일을 병합한다.

```text
SECOM Sensor Data
        +
Label / Timestamp
        ↓
secom_merged.csv
```

---

## 3.4 Label

| Label | 의미 |
|---:|---|
| `-1` | Pass / 정상 |
| `1` | Fail / 불량 |

전체 1,567개 데이터 중 Fail은 약 104개로, 불량 비율은 약 7% 수준이다.

따라서 본 프로젝트는 **Class Imbalance가 큰 이진 분류 문제**로 접근한다.

---

## 3.5 결측값

SECOM은 실제 제조 공정 데이터이기 때문에 특징마다 서로 다른 수준의 결측값이 존재한다.

원본 데이터에서는 결측값이 다음과 같이 표현된다.

```text
NaN
```

본 프로젝트에서는 다음과 같이 처리한다.

```text
결측률 확인
    ↓
결측률 > 50% 센서 제거
    ↓
나머지 센서
    ↓
SimpleImputer(median)
```

Imputation은 데이터 누수를 방지하기 위해 모델 학습 Pipeline 내부에서 수행한다.

---

## 3.6 원 데이터셋 Baseline

원 데이터셋 설명에서는 다음 조건으로 초기 Baseline 실험이 수행되었다.

```text
Standardization
        ↓
Constant Feature Removal
        ↓
Feature Selection
        ↓
Top 40 Features
        ↓
Kernel Ridge Classifier
        ↓
10-Fold Cross Validation
```

평가 지표는 **Balanced Error Rate(BER)**를 사용하였다.

| Feature Selection | BER (%) | True + (%) | True - (%) |
|---|---:|---:|---:|
| S2N | 34.5 ± 2.6 | 57.8 ± 5.3 | 73.1 ± 2.1 |
| T-test | 33.7 ± 2.1 | 59.6 ± 4.7 | 73.0 ± 1.8 |
| Relief | 40.1 ± 2.8 | 48.3 ± 5.9 | 71.6 ± 3.2 |
| Pearson | 34.1 ± 2.0 | 57.4 ± 4.3 | 74.4 ± 4.9 |
| F-test | 33.5 ± 2.2 | 59.1 ± 4.8 | 73.8 ± 1.8 |
| Gram Schmidt | 35.6 ± 2.4 | 51.2 ± 11.8 | 77.5 ± 2.3 |

본 프로젝트에서는 이 결과를 직접적인 성능 목표로 사용하기보다, **특징 선택과 Cross Validation이 SECOM 분석의 중요한 요소라는 참고 자료**로 활용한다.

---

# 4. 문제 정의

## 4.1 Input

```text
반도체 제조 공정 센서 데이터
```

## 4.2 Output

```text
정상(Pass) : -1
불량(Fail) :  1
```

## 4.3 핵심 문제

> **다수의 반도체 공정 센서 데이터를 이용하여 최종 검사에서 Fail이 발생할 생산 단위를 조기에 탐지할 수 있는가?**

추가적으로 다음 질문을 검증한다.

> 불량 탐지 성능을 유지하면서 센서 수를 얼마나 줄일 수 있는가?

> Random Split에서 확인된 성능이 시간 순서 평가에서도 유지되는가?

> 적은 불량 데이터를 사용하는 모델의 성능이 반복 검증에서도 안정적인가?

---

# 5. 프로젝트 목표

## 5.1 불량 Recall 확보

불량 제품을 정상으로 판단하는 **False Negative**를 주요 위험으로 본다.

따라서 Accuracy보다 **Fail Recall**을 우선적으로 확인한다.

---

## 5.2 PR-AUC 개선

불량 비율이 매우 낮기 때문에 Accuracy만으로 모델을 평가하지 않는다.

주요 모델 비교 지표:

- Recall
- PR-AUC / Average Precision

보조 지표:

- Precision
- F1-score
- ROC-AUC

---

## 5.3 센서 수 감소

전체 센서를 사용하는 모델과 특징 선택 모델을 비교한다.

도전 목표는 다음과 같다.

```text
약 590개 공정 특징
        ↓
핵심 센서 ≤ 20
```

단, 20개 이하를 절대적인 성공 조건으로 사용하지 않는다.

성능이 크게 하락한다면 **센서 수와 예측 성능의 Trade-off** 자체를 분석 결과로 제시한다.

---

## 5.4 모델 안정성 검증

불량 데이터 수가 적기 때문에 한 번의 검증 결과만으로 모델을 선택하지 않는다.

Repeated Stratified Cross Validation을 이용하여 다음과 같이 평가한다.

```text
CV Score = Mean ± Standard Deviation
```

---

## 5.5 시간 변화 검증

Timestamp를 이용하여 Random Split과 Time-based Split을 비교한다.

시간 순서 평가에서 성능이 크게 감소하면 **공정 조건 변화 또는 Data Drift 가능성**을 추가 분석한다.

---

# 6. 개발 환경

```text
Environment : ml310
Python      : 3.10
```

주요 라이브러리:

```text
pandas
numpy
scikit-learn
lightgbm
xgboost
matplotlib
joblib
```

GPU와 딥러닝은 필수로 사용하지 않는다.

---

# 7. 데이터 전처리

## 7.1 기본 품질 점검

다음 항목을 확인한다.

- 데이터 Shape
- 정상/불량 개수
- 불량 비율
- 센서별 결측률
- Zero Variance
- 중복 데이터
- Timestamp

---

## 7.2 센서 제거 기준

다음 조건에 해당하는 센서를 확인하고 제거 내역을 기록한다.

```text
Missing Ratio > 50%
Zero Variance
```

제거 전·후 센서 수를 `dataset_log.csv`에 기록한다.

---

## 7.3 데이터 누수 방지

다음 작업은 학습 데이터 기준으로 수행한다.

- Imputation
- Scaling
- Feature Selection
- PCA
- Model Training

가능한 전처리는 `scikit-learn Pipeline` 내부에서 관리한다.

---

# 8. 데이터 분할

## 8.1 Random Split

```text
Train : Validation : Test
70    : 15         : 15
```

분류 문제이므로 `stratify`를 적용한다.

---

## 8.2 Time-based Split

Timestamp 기준으로 정렬한 뒤 과거 데이터를 학습하고 이후 데이터를 평가한다.

목적은 Random Split에서는 발견하기 어려운 시간 변화에 대한 성능 저하를 확인하는 것이다.

---

## 8.3 Test 사용 원칙

Test 데이터는 프로젝트 마지막 단계에서 **최종 모델로 1회만 평가**한다.

Test 결과를 이용하여 다음을 변경하지 않는다.

- 모델
- 특징 선택 방법
- 센서 수
- Hyperparameter
- Threshold

---

# 9. 모델링 전략

## 9.1 Baseline

```text
SimpleImputer(median)
        ↓
StandardScaler
        ↓
LogisticRegression
```

---

## 9.2 특징 축소

### PCA

```text
Imputer
→ StandardScaler
→ PCA(90% Variance)
→ Classifier
```

### L1 Feature Selection

```text
Imputer
→ StandardScaler
→ L1 Logistic
→ Selected Features
→ Classifier
```

### Feature Importance

Random Forest 또는 LightGBM 기반 중요도를 사용한다.

```text
Top 100
Top 50
Top 30
Top 20
Top 10
```

---

## 9.3 후보 모델

최소 3개 이상을 동일 조건에서 비교한다.

- L1 Logistic Regression
- SVM
- Random Forest
- LightGBM

필요한 경우 다음 방법으로 불균형을 처리한다.

```text
class_weight
scale_pos_weight
```

---

# 10. 모델 검증

RepeatedStratifiedKFold 기반으로 후보 모델을 평가한다.

주요 기록 항목:

| 항목 | 내용 |
|---|---|
| Model | 모델 |
| Feature Method | 특징 선택 방식 |
| Sensor Count | 사용 센서 수 |
| Recall | Mean ± Std |
| AP | Mean ± Std |
| F1 | Mean ± Std |
| Train Time | 학습 시간 |
| Inference Time | 추론 시간 |

성능 차이가 CV 편차 범위 내에서 유사하다면 **더 단순하고 빠르며 사용하는 센서 수가 적은 모델**을 우선 검토한다.

---

# 11. Threshold 및 이상 탐지

## 11.1 Threshold

OOF Prediction을 이용하여 Threshold에 따른 다음 값을 비교한다.

| Threshold | Recall | Precision | FN | FP |
|---:|---:|---:|---:|---:|
| 0.50 | - | - | - | - |
| 0.40 | - | - | - | - |
| 0.30 | - | - | - | - |
| 0.20 | - | - | - | - |

Fail Recall 목표와 FN/FP Trade-off를 기준으로 최종 Threshold를 결정한다.

---

## 11.2 이상 탐지

지도학습 방식과 다음 이상 탐지 방식을 비교한다.

- Isolation Forest
- PCA Reconstruction Error

정상 데이터만을 이용하여 불량 후보를 탐지할 수 있는 가능성을 확인하고 지도학습 결과와 비교한다.

---

# 12. 프로젝트 일정

각 업무는 `[ ]` → `[x]` 형태로 진행 상황을 관리한다.

| 일차 | 담당 | 할 일 | 예상 산출물 | 완료 |
|---|---|---|---|:---:|
| **Day 1** | A | SECOM 센서·Label 데이터 확보 | `data/raw/` | [ ] |
| Day 1 | A | 센서 + Label + Timestamp 병합 | `secom_merged.csv` | [ ] |
| Day 1 | A | 데이터 Shape·클래스 분포 확인 | `dataset_log.csv` | [ ] |
| Day 1 | A | 센서별 결측률 계산 | `dataset_log.csv` | [ ] |
| Day 1 | A | 결측률 50% 초과 센서 확인·제거 | `dataset_log.csv` | [ ] |
| Day 1 | A | Zero Variance 센서 확인·제거 | `dataset_log.csv` | [ ] |
| Day 1 | B | 평가 지표 정의 | `README.md` | [ ] |
| Day 1 | B | Baseline·후보 모델 설계 | 실험 계획 | [ ] |
| Day 1 | B | 특징 선택 전략 설계 | 실험 계획 | [ ] |
| Day 1 | 공동 | GitHub 구조·실행 환경 구성 | `requirements.txt` | [ ] |
| Day 1 | 공동 | Random/Time Split 계획 확정 | `README.md` | [ ] |
| **Day 2** | A | 클래스·결측·센서 EDA | EDA Figures | [ ] |
| Day 2 | A | Timestamp 분석 | `timestamp_distribution.png` | [ ] |
| Day 2 | A | Random 70/15/15 Split | `split_*.txt` | [ ] |
| Day 2 | A | Time-based Split | `split_time_*.txt` | [ ] |
| Day 2 | B | Baseline Pipeline 구현 | `step4_baseline.py` | [ ] |
| Day 2 | B | Baseline 성능 측정 | `baseline_result.csv` | [ ] |
| Day 2 | 공동 | 데이터 누수 및 Pipeline 검토 | 코드 리뷰 | [ ] |
| **Day 3** | B | PCA 특징 축소 | `feature_compare.csv` | [ ] |
| Day 3 | B | L1 Feature Selection | `feature_compare.csv` | [ ] |
| Day 3 | B | RF/LightGBM Importance | `feature_compare.csv` | [ ] |
| Day 3 | B | Top-K 센서 비교 | `feature_compare.csv` | [ ] |
| Day 3 | B | Logistic/SVM/RF/LightGBM 비교 | `model_compare.csv` | [ ] |
| Day 3 | B | Repeated Stratified CV | `model_compare.csv` | [ ] |
| Day 3 | A | PCA·Feature Importance 시각화 | Figures | [ ] |
| Day 3 | A | 센서 수와 성능 관계 분석 | `sensor_count_vs_ap.png` | [ ] |
| Day 3 | 공동 | 최종 후보 모델 1~2개 선정 | 후보 모델 | [ ] |
| **Day 4** | B | OOF Prediction 생성 | OOF 결과 | [ ] |
| Day 4 | B | Threshold별 Recall/FN/FP 비교 | `threshold_compare.csv` | [ ] |
| Day 4 | B | Isolation Forest 실험 | `anomaly_compare.csv` | [ ] |
| Day 4 | B | PCA Reconstruction Error 실험 | `anomaly_compare.csv` | [ ] |
| Day 4 | B | CLI 구현 | `predict_cli.py` | [ ] |
| Day 4 | A | FN/FP 오류 분석 | `error_analysis.png` | [ ] |
| Day 4 | A | 핵심 센서 Top 20 분석 | `top20_sensors.png` | [ ] |
| Day 4 | A | Random vs Time 성능 비교 | `random_vs_time.png` | [ ] |
| Day 4 | A | 센서 Drift 분석 | `sensor_drift.png` | [ ] |
| Day 4 | 공동 | 최종 센서·모델·Threshold 확정 | `candidate_model.joblib` | [ ] |
| **Day 5** | B | 최종 모델 설정 동결 | 최종 설정 | [ ] |
| Day 5 | B | Test 데이터 최종 1회 평가 | `test_result.csv` | [ ] |
| Day 5 | B | 최종 Pipeline 저장 | `model.joblib` | [ ] |
| Day 5 | B | Model Card 작성 | `model_card.json` | [ ] |
| Day 5 | B | CLI 최종 검증 | `predict_cli.py` | [ ] |
| Day 5 | A | 최종 Confusion Matrix·PR Curve | Figures | [ ] |
| Day 5 | A | 오류·핵심 센서·Drift 결과 정리 | `report.md` | [ ] |
| Day 5 | A | README 최종 작성 | `README.md` | [ ] |
| Day 5 | 공동 | 전체 재현성 검증 | 재현성 확인 | [ ] |
| Day 5 | 공동 | CLI Demo 제작 | Demo | [ ] |
| Day 5 | 공동 | 발표자료·리허설 | `final_presentation.pdf` | [ ] |

---

# 13. 일정별 완료 기준

| 일차 | 완료 기준 | 완료 |
|---|---|:---:|
| **Day 1** | 데이터 병합 및 품질 점검을 완료하고 제거 센서와 이유를 설명할 수 있다. | [ ] |
| **Day 2** | 데이터 Split이 고정되고 Baseline 성능을 확보했다. | [ ] |
| **Day 3** | 반복 CV 결과를 기반으로 최종 후보 모델 1~2개를 선정했다. | [ ] |
| **Day 4** | 특징·센서·모델·Threshold를 확정하고 Test 평가 전 설정을 동결했다. | [ ] |
| **Day 5** | Test 1회 평가와 모델 출고, 문서, Demo, 발표자료를 완성했다. | [ ] |

---

# 14. 최종 산출물

## 14.1 데이터 및 실험

| 완료 | 산출물 | 담당 |
|:---:|---|---|
| [ ] | `dataset_log.csv` | A |
| [ ] | `split_*.txt` | A |
| [ ] | `split_summary.csv` | A |
| [ ] | `baseline_result.csv` | B |
| [ ] | `feature_compare.csv` | B |
| [ ] | `model_compare.csv` | B |
| [ ] | `threshold_compare.csv` | B |
| [ ] | `anomaly_compare.csv` | B |
| [ ] | `test_result.csv` | B |

## 14.2 모델 및 서비스

| 완료 | 산출물 | 담당 |
|:---:|---|---|
| [ ] | `model.joblib` | B |
| [ ] | `model_card.json` | B |
| [ ] | `predict_cli.py` | B |
| [ ] | `requirements.txt` | B |
| [ ] | CLI Demo | 공동 |

## 14.3 분석 및 포트폴리오

| 완료 | 산출물 | 담당 |
|:---:|---|---|
| [ ] | EDA Figures | A |
| [ ] | Confusion Matrix | A |
| [ ] | PR Curve | A |
| [ ] | 핵심 센서 Top 20 | A |
| [ ] | FN/FP 오류 분석 | A |
| [ ] | Random vs Time 분석 | A |
| [ ] | Drift 분석 | A |
| [ ] | `report.md` | A |
| [ ] | `README.md` | A + B |
| [ ] | `final_presentation.pdf` | 공동 |

---

# 15. GitHub Repository 구조

```text
secom-waferguard/
│
├── README.md                         # 프로젝트 개요, 실행 방법, 실험 결과 및 전체 문서
├── requirements.txt                 # Python 패키지 및 실행 환경 의존성 목록
│
├── data/                            # 프로젝트에서 사용하는 데이터 저장
│   ├── raw/                         # UCI에서 받은 원본 SECOM 데이터 (수정 금지)
│   ├── processed/                   # 병합 및 기본 정리가 완료된 데이터
│   └── splits/                      # Train / Validation / Test 분할 정보
│
├── logs/                            # 데이터 처리 및 모델 실험 결과 기록
│   ├── dataset_log.csv              # 데이터 크기, 결측률, 제거 센서 등 데이터 품질 기록
│   ├── baseline_result.csv          # Baseline 모델의 Recall, AP, F1 등 평가 결과
│   ├── feature_compare.csv          # PCA, L1, Importance 등 특징 선택 방법 비교 결과
│   ├── model_compare.csv            # 후보 모델별 반복 CV 성능·편차·속도 비교
│   ├── threshold_compare.csv        # Threshold별 Recall, Precision, FN, FP 비교
│   ├── anomaly_compare.csv          # Isolation Forest, PCA 이상 탐지 실험 결과
│   └── test_result.csv              # 최종 Test 데이터 1회 평가 결과
│
├── src/                             # 데이터 처리, 학습, 평가 및 추론 Python 코드
│   ├── step1_merge_data.py          # 센서 데이터와 Label/Timestamp 데이터 병합
│   ├── step2_data_check.py          # 결측률, Zero Variance, 중복 등 데이터 품질 점검
│   ├── step3_split.py               # Random 및 Time-based 데이터 분할
│   ├── step4_baseline.py            # Logistic Regression Baseline 학습 및 평가
│   ├── step5_feature_selection.py   # PCA, L1, Feature Importance 기반 특징 선택
│   ├── step6_model_compare.py       # LR, SVM, RF, LightGBM 반복 CV 및 모델 비교
│   ├── step7_threshold.py           # OOF Prediction 기반 최종 Threshold 탐색
│   ├── step7_anomaly_detection.py   # Isolation Forest 및 PCA 이상 탐지 비교
│   ├── step8_test.py                # 고정된 최종 모델의 Test 데이터 1회 평가
│   └── predict_cli.py               # 저장된 모델을 이용한 신규 데이터 CLI 추론
│
├── models/                          # 학습 완료 모델 및 모델 메타데이터
│   ├── model.joblib                 # 전처리 + 특징 선택 + 최종 모델 Pipeline
│   └── model_card.json              # 모델 버전, Threshold, 특징, 성능 및 한계 정보
│
├── reports/                         # 프로젝트 분석 보고서 및 결과 시각화
│   ├── report.md                    # 최종 성능, 오류 분석, 한계 및 개선 방향 보고서
│   │
│   └── figures/                     # README, 보고서, 발표에 사용하는 시각화 결과
│       ├── class_distribution.png   # 정상/불량 클래스 분포
│       ├── missing_ratio.png        # 센서별 결측률 분포
│       ├── pca_variance.png         # PCA 누적 설명분산 결과
│       ├── feature_importance.png   # 주요 센서 Feature Importance
│       ├── sensor_count_vs_ap.png   # 사용 센서 수와 AP 성능 관계
│       ├── threshold_recall_precision.png
│       │                            # Threshold 변화에 따른 Recall/Precision 관계
│       ├── error_analysis.png       # False Negative / False Positive 오류 분석
│       ├── final_confusion_matrix.png
│       │                            # 최종 Test 결과 Confusion Matrix
│       ├── final_pr_curve.png       # 최종 모델 Precision-Recall Curve
│       ├── top20_sensors.png        # 최종 선정 핵심 센서 Top 20
│       ├── sensor_drift.png         # 시간에 따른 주요 센서 값 변화 및 Drift 분석
│       └── random_vs_time.png       # Random Split과 Time-based Split 성능 비교
│
└── presentation/                    # 프로젝트 최종 발표 자료
    └── final_presentation.pdf       # 문제→데이터→방법→결과→결론 발표자료
```

## 폴더별 역할 요약

| 폴더 | 역할 | 주요 내용 |
|---|---|---|
| `data/raw/` | 원본 데이터 | 다운로드한 SECOM 원본 파일 보관 |
| `data/processed/` | 가공 데이터 | 센서·Label·Timestamp 병합 결과 |
| `data/splits/` | 데이터 분할 | Train / Validation / Test 대상 기록 |
| `logs/` | 실험 기록 | 데이터 점검 및 모든 모델 실험 결과 CSV |
| `src/` | 실행 코드 | 데이터 처리부터 최종 평가·추론까지 단계별 코드 |
| `models/` | 모델 출고 | 최종 Pipeline과 Model Card |
| `reports/` | 분석 보고 | 프로젝트 결과, 오류 분석, 한계 및 개선 방향 |
| `reports/figures/` | 시각화 | EDA 및 모델 평가 그래프 |
| `presentation/` | 발표 | 최종 프로젝트 발표자료 |

## 실행 순서

파일명의 `step` 번호는 프로젝트 실행 순서를 의미한다.

```text
step1_merge_data.py
        ↓
step2_data_check.py
        ↓
step3_split.py
        ↓
step4_baseline.py
        ↓
step5_feature_selection.py
        ↓
step6_model_compare.py
        ↓
step7_threshold.py
        ├───────────────┐
        ↓               ↓
Threshold        step7_anomaly_detection.py
        │               │
        └───────┬───────┘
                ↓
        최종 모델 결정
                ↓
        step8_test.py
                ↓
          model.joblib
                ↓
          predict_cli.py
```

> **관리 원칙:** `data/raw/`의 원본 데이터는 직접 수정하지 않는다. 모든 데이터 처리 과정은 `src/`의 코드로 재현할 수 있도록 하며, 실험 결과는 `logs/`, 그래프는 `reports/figures/`, 최종 모델은 `models/`에 저장한다.
---

# 16. 전체 Workflow

```text
UCI SECOM
    │
    ▼
Sensor + Label + Timestamp
    │
    ▼
데이터 품질 점검
    │
    ├── Missing > 50%
    └── Zero Variance
    │
    ▼
Random / Time Split
    │
    ▼
Baseline
    │
    ▼
Feature Selection
    │
    ├── PCA
    ├── L1
    └── Feature Importance
    │
    ▼
Candidate Models
    │
    ├── Logistic Regression
    ├── SVM
    ├── Random Forest
    └── LightGBM
    │
    ▼
Repeated Stratified CV
    │
    ▼
성능 + 안정성 + 센서 수 + 속도
    │
    ▼
Final Candidate
    │
    ├─────────────┐
    ▼             ▼
Threshold      Anomaly Detection
    │             │
    └──────┬──────┘
           ▼
      Error Analysis
           │
           ▼
     Random vs Time
           │
           ▼
      Model Freeze
           │
           ▼
       Test 1회
           │
           ▼
      Model Bundle
       /    |     \
model.joblib |  predict_cli.py
        model_card.json
```

---

# 17. 프로젝트 성공 기준

| 항목 | 기준 |
|---|---|
| **Fail Recall** | Baseline 대비 개선 또는 목표 Recall 확보 |
| **PR-AUC** | Baseline 대비 개선 |
| **센서 축소** | 성능을 고려한 최소 센서 구성 탐색 |
| **도전 목표** | 가능하면 핵심 센서 20개 이하 |
| **안정성** | 반복 CV Mean ± Std 확인 |
| **데이터 누수** | Pipeline 기반 전처리 |
| **시간 변화** | Random / Time 성능 비교 |
| **오류 분석** | FN / FP 사례 분석 |
| **최종 평가** | Test 데이터 마지막 1회 사용 |
| **재현성** | README 기반 재실행 가능 |
| **모델 출고** | Model + Threshold + Version 저장 |

---

# 18. 최종 발표 구성

| 슬라이드 | 내용 |
|---|---|
| **1. Problem** | 왜 반도체 공정 불량 조기 탐지가 필요한가? |
| **2. Data** | 1,567 Samples / 고차원 특징 / 104 Fails / Missing |
| **3. Method** | Cleaning → Feature Selection → Models → Repeated CV → Threshold |
| **4. Results** | Baseline vs Final / Recall / AP / Sensors / Random vs Time |
| **5. Conclusion** | 핵심 센서, 오류 분석, 한계, 개선 방향 |

---

# 19. 프로젝트 최종 체크리스트

| 완료 | 확인 항목 |
|:---:|---|
| [ ] | 데이터 출처와 라이선스를 README에 기록했다. |
| [ ] | 원 데이터 파일 구조와 변수 수를 직접 확인했다. |
| [ ] | 데이터 정제 과정과 제거 센서 수를 기록했다. |
| [ ] | Train / Validation / Test 데이터 누수가 없다. |
| [ ] | Imputation·Scaling·특징 선택을 Pipeline에서 처리했다. |
| [ ] | Baseline과 최종 모델을 비교했다. |
| [ ] | 최소 3개 이상의 후보 모델을 동일 조건에서 비교했다. |
| [ ] | 반복 CV의 Mean ± Std를 기록했다. |
| [ ] | 센서 수와 성능의 Trade-off를 분석했다. |
| [ ] | OOF Prediction을 이용해 Threshold를 결정했다. |
| [ ] | False Negative와 False Positive를 분석했다. |
| [ ] | Random Split과 Time Split을 비교했다. |
| [ ] | Test 데이터는 최종 단계에서 1회만 평가했다. |
| [ ] | 최종 모델·Threshold·Version을 저장했다. |
| [ ] | `predict_cli.py`로 신규 데이터를 추론할 수 있다. |
| [ ] | README 실행 방법을 처음부터 다시 검증했다. |
| [ ] | 프로젝트 한계와 향후 개선 방향을 기록했다. |
| [ ] | GitHub 최종 산출물을 모두 점검했다. |

---

# 20. 프로젝트 핵심 메시지

> **SECOM WaferGuard는 고차원·결측·불균형 특성을 가진 반도체 제조 공정 데이터를 대상으로, 데이터 누수를 방지한 Pipeline과 Feature Selection 및 반복 교차검증을 이용하여 불량 생산 단위의 조기 탐지 가능성과 핵심 공정 센서를 분석하는 프로젝트이다.**

본 프로젝트는 단순히 높은 분류 성능을 만드는 데 목적을 두지 않는다.

```text
불량 미검 위험
    ↓
핵심 센서 선택
    ↓
모델 성능 및 안정성
    ↓
Threshold
    ↓
시간 변화 / Drift
    ↓
오류 원인
    ↓
재현 가능한 모델 출고
```

이를 통해 실제 반도체 공정·수율 데이터 분석에서 요구되는 **데이터 품질 관리, 고차원 특징 선택, 불균형 분류, 모델 검증, 오류 분석 및 모델 출고의 전체 머신러닝 프로세스**를 수행하는 것을 최종 목표로 한다.
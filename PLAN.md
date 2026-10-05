# SECOM WaferGuard 실행 계획

이 문서는 프로젝트의 실험 순서, 역할 분담, 완료 기준과 산출물을 관리합니다. 대외 공개용 프로젝트 소개와 사용 방법은 [README.md](README.md)를 참고합니다.

## 원칙

- 원본 데이터는 `data/raw/`에서 수정하지 않습니다.
- Train 데이터로만 전처리기·특징 선택기·모델을 fit합니다.
- Imputation, scaling, PCA, feature selection은 `scikit-learn Pipeline`으로 관리합니다.
- Test set은 모델·특징·하이퍼파라미터·threshold를 동결한 뒤 최종 1회만 사용합니다.
- 모든 비교 실험은 가능한 한 동일한 split과 평가 지표를 사용합니다.

## 역할 분담

| 담당 | 주요 업무 |
| --- | --- |
| A — 이종수 (데이터·분석) | 데이터 확보·병합·정제, EDA, split, 오류·drift 분석, 문서·시각화 |
| B — 양현승 (모델링·평가) | Pipeline, baseline, 특징 선택, 후보 모델 비교, 반복 CV, threshold, 모델 저장·CLI |
| 공동 | 데이터 누수 검토, 최종 모델·threshold 동결, Test 평가, 재현성 검증, 발표 |

## 전체 실행 흐름

```text
Data merge → Quality check → Split → Baseline
                                      ↓
Feature selection → Candidate models → Repeated CV
                                      ↓
Threshold selection + Error/Drift analysis
                                      ↓
Model freeze → Final test (once) → Model bundle
```

## Day 1 — 데이터 확보와 실험 설계

**목표:** UCI SECOM 데이터를 분석 가능한 형태로 만들고 실험 기준을 확정합니다.

### A — 이종수 (데이터·분석)

- [x] Sensor 데이터와 Label/Timestamp 데이터를 `data/raw/`에 저장
- [x] Sensor + Label + Timestamp 병합
- [x] 행·열 수, 클래스 분포, 중복, Timestamp 확인
- [x] 센서별 결측률 산출
- [x] 결측률 50% 초과 및 zero-variance 센서 EDA 후보 기준 확정
- [x] 제거 후보·EDA 기준 잔여 센서 수와 사유를 `logs/dataset_log.csv`에 기록

#### Day 1 데이터 처리 원칙 및 변경 이유

- 원본 데이터 병합은 Dataset Profile의 `ingestion` source 목록과 adapter 정의를 사용해 canonical `secom_merged.csv`를 생성한다. 원본 경로·읽기 옵션·결합 규칙을 코드에 고정하지 않아 같은 수집 형식의 다른 공장 데이터는 Profile만으로 처리할 수 있다.
- `step2_data_check.py`의 결측률 50% 초과·상수 센서 결과는 전체 데이터 EDA용 후보와 근거를 기록하는 용도다. 전역 `secom_cleaned.csv`를 만들거나 그 결과를 모델 입력에 바로 적용하지 않는다. 실제 제거·대치·선택은 Train 또는 CV 학습 fold 안에서 fit해야 validation/test 정보가 학습에 섞이지 않는다.
- `secom_merged.csv`와 `dataset_log.csv`는 원본 파일과 Profile로 재생성하는 결과물이다. 저장소에는 코드·설정·문서만 유지하고, 결과 CSV는 `.gitignore`로 제외한다.

### B — 양현승 (모델링·평가)

- [x] Recall, PR-AUC/AP, Precision, F1, ROC-AUC 평가 기준 확정
- [x] Baseline과 후보 모델의 비교 조건 설계
- [x] PCA·L1·Feature Importance 기반 특징 선택 전략 설계

#### B 결정 사항

- SECOM 실험의 기본 Fail label은 `1`로 사용한다. 평가 함수는 `positive_label` 설정값을 사용하며, 모델 비교에서는 AP와 Fail Recall을 핵심 지표로, Precision·F1·ROC-AUC를 보조 지표로 기록한다.
  - **이유:** 모든 평가 함수와 confusion matrix에서 불량 클래스를 일관되게 해석하기 위해서다. Recall은 불량을 정상으로 놓치는 False Negative 위험을, AP는 불량 비율이 낮은 데이터에서 threshold와 무관한 순위 성능을 확인한다.
- 범용 코어는 데이터셋별 가정을 직접 갖지 않는다. 공통 실험 조건은 `config.json`, 데이터 구조·Label·품질 규칙은 `configs/datasets/<dataset_id>.json` Profile로 분리한다.
  - **이유:** SECOM 외 tabular 데이터에도 같은 split·평가·Pipeline 코어를 적용하고, 새 데이터셋 추가 시 모델 코드가 특정 데이터셋에 종속되지 않게 하기 위해서다.
- 후보 모델 비교는 Test를 제외한 학습 데이터에서 `RepeatedStratifiedKFold(5 folds × 5 repeats, random_state=42)`로 수행한다. 후보 비교의 threshold는 `0.50`으로 고정하며, 최종 threshold는 Day 4 OOF prediction에서만 결정한다.
  - **이유:** 약 104개의 적은 Fail 표본으로 한 번의 split에 의존하지 않기 위해 반복 계층 CV를 사용한다. 모델 비교와 운영 threshold 결정을 분리해 공정한 비교를 유지하고, Test set의 사후 최적화를 막는다.
- Baseline은 `SimpleImputer(median) → StandardScaler → LogisticRegression` Pipeline으로 구현한다. 후보는 L1 Logistic Regression, RBF SVM, Random Forest, LightGBM으로 구성한다.
  - **이유:** Baseline은 결측·스케일 차이가 있는 고차원 수치 데이터에서 빠르고 해석 가능한 기준점이 된다. 후보 모델은 희소 선형 관계, 비선형 경계, 센서 간 상호작용을 각각 검증하기 위해 선택한다.
- PCA는 누적 설명분산 90% 기준으로 비교한다. L1과 Feature Importance 기반 Top-K(100/50/30/20/10) 선택은 반드시 각 CV 학습 fold 내부에서 fit한다.
  - **이유:** PCA·L1·Importance는 각각 분산 보존, 희소 센서 선택, 공정 해석 가능성 측면의 장단점이 다르다. 선택기를 검증 fold까지 포함해 fit하면 센서 선택 자체가 데이터 누수가 되므로 학습 fold 안에서만 fit한다.
- 모든 비교는 같은 split·난수 시드·평가 함수·입력 특징 정의를 사용한다. tree 모델에는 scaling을 강제하지 않지만, 결측치 처리와 특징 선택은 Pipeline 내부에서 수행한다.
  - **이유:** 모델 외 조건을 고정해야 성능 차이를 모델·특징 선택 방식의 차이로 해석할 수 있다. tree 기반 모델은 값의 scale에 민감하지 않지만, Logistic Regression·SVM·PCA는 scale에 민감하므로 model-specific Pipeline을 사용한다.

### 공동

- [x] Random / Time-based split 계획 및 leakage 검토

  **공동 결정된 데이터 분할 기준**

  1. **입력 데이터:** 분할과 모델링에는 `data/processed/secom_merged.csv`를 사용한다. `secom_cleaned.csv`는 전체 데이터를 기준으로 만든 EDA·점검용 산출물이므로 모델 입력으로 사용하지 않는다.
  2. **Random 평가의 범위:** 전체 데이터에서 별도로 만든 기존 Random split은 탐색 실험으로 보존한다. 새 시간 평가에서는 먼저 Time Train·Validation·Test를 분리하고, 후보 비교·특징 선택·OOF는 같은 Time Train 내부에서만 수행한다. 내부 CV는 `random_state=42`와 label 계층화를 적용하며, 그룹이 선언된 데이터는 반복 Stratified Group K-Fold로 같은 그룹의 fold 간 누수를 방지한다.
  3. **Time-based split:** timestamp를 Dataset Profile의 형식으로 파싱한 뒤 시간 오름차순으로 정렬한다. 과거 70%를 Train, 중간 15%를 Validation, 최근 15%를 Test로 사용하는 것을 목표로 하며 shuffle·stratify는 적용하지 않는다. 같은 timestamp·선언된 그룹·복합 ID는 경계를 넘겨 분리하지 않는다. 따라서 실제 비율은 달라질 수 있으며, 안전한 경계가 없거나 timestamp 파싱이 실패하면 중단한다.
  4. **Leakage 방지:** 동일한 생성 계약의 Train·Validation·Test만 사용하며 원본 행 ID와 split 역할을 검증한다. `label`, `timestamp`, Profile의 ID·그룹·제외 컬럼과 예약 split metadata는 모델 입력에서 제외한다. 수치형 median 대치·범주형 최빈값 대치/One-Hot Encoding·scaling·특징 선택은 Train/CV 학습 fold에서만 fit한다. Time Validation은 Train 이후 시간 범위인지도 검사한다.
  5. **Test set 봉인:** 후보 모델 비교, 특징 선택, threshold 결정에는 Test를 사용하지 않는다. 최종 모델과 threshold가 확정된 뒤에만 Test 성능을 한 번 평가한다.
  6. **산출물 및 점검 기록:** 팀원의 `src/step3_split.py`를 공식 진입점으로 유지한다. 기존 분할·EDA·요약·누수 검사 함수에 Profile 검증·그룹 경계·원본 행 ID·저장 전 검사를 통합한다. 기본 출력은 `data/splits/integrated/`의 여섯 split 및 `manifest.json`, `logs/step3_integrated/`의 요약·누수·시간 경계 점검이다. 각 CSV에는 `__source_row_id`, `__split_role`, `__split_protocol_id`를 보존한다. 기존 split 파일은 덮어쓰지 않고 검사 실패 시 저장을 중단한다. 그림은 `reports/figures/step3/`에 저장한다. 별도 `step3_split_data.py`는 중복 구현을 피하기 위해 제거한다.

  **기존 결과 해석의 정정:** 기존 Time Validation 235행 중 163행이 Random Train에 포함돼 후보 비교·OOF 탐색에 사용됐다. 기존 시간 평가 수치를 독립적인 미사용 holdout 성능으로 주장하지 않는다. 위 교정 경로로 재실행하더라도 이미 관찰한 데이터가 다시 미사용 데이터가 되는 것은 아니며, 엄격한 최종 일반화 확인에는 별도의 미사용 데이터가 필요하다.

- [x] Python 환경과 `requirements.txt` 구성 확인

**예정 산출물:** `data/processed/secom_merged.csv`, `logs/dataset_log.csv`, `src/step1_merge_data.py`, `src/step2_data_check.py`

## Day 2 — EDA, 데이터 분할, Baseline

**목표:** 재현 가능한 split을 만들고 비교 기준이 될 baseline을 구축합니다.

### A — 이종수 (데이터·분석)

- [ ] 클래스 분포·결측률·Timestamp EDA 및 시각화
- [ ] Random Train/Validation/Test = 70/15/15 split, stratify 적용
- [ ] Timestamp 기준 time-based split 생성
- [ ] Split별 클래스 비율을 `logs/split_summary.csv`에 기록

### B — 양현승 (모델링·평가)

- [x] `SimpleImputer(median) → StandardScaler → LogisticRegression` baseline 구현
- [x] Recall, AP, Precision, F1, ROC-AUC, 학습 시간을 기록

### 공동

- [ ] Train/Validation/Test 간 데이터 누수와 Pipeline fit 범위 검토
- [ ] Baseline 결과 검토 및 다음 실험 조건 확정

**예정 산출물:** `data/splits/`, `logs/split_summary.csv`, `logs/baseline_result.csv`, `src/step3_split.py`, `src/step4_baseline.py`

## Day 3 — 특징 선택과 후보 모델 비교

**목표:** 센서 수와 성능의 trade-off를 분석하고 최종 후보 1~2개를 고릅니다.

### A — 이종수 (데이터·분석)

- [ ] PCA 누적 설명분산과 Feature Importance 시각화
- [ ] 센서 수와 AP의 관계 분석

### B — 양현승 (모델링·평가)

- [x] PCA(누적 설명 분산 90%) 실험
- [x] L1 Logistic Regression 기반 특징 선택
- [x] Random Forest/LightGBM Feature Importance 기반 Top-K 비교 (100/50/30/20/10)
- [x] Logistic Regression, SVM, Random Forest, LightGBM 비교
- [x] `class_weight` 또는 `scale_pos_weight` 적용 비교
- [x] Repeated Stratified K-Fold로 평균 ± 표준편차 기록

### 공동

- [ ] 성능, 센서 수, 학습·추론 시간을 종합해 후보 선정

**예정 산출물:** `logs/feature_compare.csv`, `logs/model_compare.csv`, `reports/figures/pca_variance.png`, `reports/figures/feature_importance.png`, `reports/figures/sensor_count_vs_ap.png`

## Day 4 — Threshold, 이상 탐지, 오류·드리프트 분석

**목표:** 운영 관점의 분류 기준을 정하고 최종 설정을 동결합니다.

### A — 이종수 (데이터·분석)

- [ ] False Negative/False Positive 사례 분석
- [ ] Random split과 time-based split의 성능 비교
- [ ] 주요 센서의 시간 변화와 drift 가능성 점검

### B — 양현승 (모델링·평가)

- [x] OOF prediction 생성
- [x] Threshold별 Recall, Precision, FN, FP 비교
- [ ] FN/FP trade-off를 바탕으로 최종 threshold 결정
- [ ] Isolation Forest와 PCA reconstruction error를 보조 실험으로 비교

#### Threshold 결정 기준 및 현재 프로젝트 후보

- Threshold는 후보 모델을 고정한 뒤 `data/splits/integrated/time_train.csv` 내부의 OOF 확률로만 비교한다. 외부 Time Validation과 Test는 threshold 탐색에 사용하지 않는다. 그룹이 있으면 그룹 분리를 적용한다.
  - **이유:** Time Validation은 시간 순서 일반화 확인용이고, Test는 최종 성능 확인용이다. 두 데이터를 threshold에 맞추면 평가 데이터에 과적합될 수 있다.
- 최종 모델은 미확정이다. `lightgbm_all`과 `l1_balanced_l1_select`의 기존 비교는 탐색 결과로 보존하고, 교정된 Time Train 내부에서 후보를 다시 선정한다. Time Validation 결과를 활용해 다시 개선하면 해당 구간은 개발용 검증 데이터로 취급한다.
  - **이유:** 기존 Random Train과 Time Validation은 겹치며, 시간 검증 결과도 이미 관찰했으므로 사전 선정·독립 검증으로 소급 해석할 수 없다.
- 기존 Random Train OOF의 F1 최대 비교값 `0.000475`는 탐색 기록으로만 보존한다. 해당 결과는 Recall `0.329`, Precision `0.218`, F1 `0.262`, FN `49`, FP `86`이며 새 평가에 그대로 승계하지 않는다.
  - **이유:** 현재 데이터에는 현장의 재검사 처리 용량, Lot hold 비용, 불량 유출 비용이 없으므로 임의의 현장 비용을 가정하지 않는다. 따라서 OOF F1 최대 지점을 비교 기준 후보로 사용한다.
- `0.000475`는 현장 자동 폐기 또는 Lot hold 기준이 아니라 프로젝트 평가용 후보값이다. 실제 현장 적용 시에는 허용 가능한 재검사 건수, FP 비용, FN 비용을 정의하고 그 제약 아래에서 다시 결정한다.
  - **이유:** 반도체 공정의 FP 허용 수준은 검사 위치와 후속 조치에 따라 달라 단일 비율로 일반화할 수 없다.
- 후보 threshold를 정한 뒤에는 같은 값을 Time Validation에 그대로 적용해 시간 구간의 FN·FP를 확인한다. Time Validation 결과를 보고 threshold를 다시 조정하지 않는다.
  - **이유:** 시간 데이터로 threshold를 재조정하면 시간 검증이 또 다른 튜닝 데이터가 되어 일반화 성능을 과대평가할 수 있다.
- 모델·특성 집합·threshold를 모두 고정한 뒤에만 Test를 한 번 평가한다.
  - **이유:** Test set을 최종 확인용으로 보존해 성능 추정의 낙관 편향을 막는다.
- 반복 OOF의 평균 확률과 단일 재학습 모델의 확률은 척도가 같다고 보장하지 않는다. 평균 방식·확률 보정·시간 순서 내부 검증에 대한 확인은 별도 후속 과제로 남기며, 현재 성능 하락을 drift만의 결과로 단정하지 않는다.

### 공동

- [ ] 특징 선택, 센서 집합, 모델, 하이퍼파라미터, threshold 동결

**예정 산출물:** `logs/threshold_compare.csv`, `logs/anomaly_compare.csv`, `reports/figures/error_analysis.png`, `reports/figures/random_vs_time.png`, `reports/figures/sensor_drift.png`

## Day 5 — 최종 평가와 출고

**목표:** 동결된 설정으로 Test를 한 번 평가하고 재현 가능한 결과물을 준비합니다.

### A — 이종수 (데이터·분석)

- [ ] confusion matrix와 PR curve 생성
- [ ] 오류·핵심 센서·drift 분석 결과 정리
- [ ] 최종 보고서와 README 업데이트

### B — 양현승 (모델링·평가)

- [ ] Test data 단일 최종 평가
- [ ] 최종 Recall, AP, Precision, F1, ROC-AUC 기록
- [ ] 전처리·특징 선택·모델을 포함한 Pipeline 저장
- [ ] threshold, 사용 센서, 버전, 한계를 Model Card에 기록
- [ ] 신규 데이터 추론 CLI 확인

### 공동

- [ ] README 절차로 재현성 검증 및 최종 보고서 작성
- [ ] GitHub 산출물, CLI demo, 발표 자료 최종 점검

**예정 산출물:** `logs/test_result.csv`, `models/model.joblib`, `models/model_card.json`, `src/predict_cli.py`, `reports/report.md`

## 완료 기준

- [ ] 데이터 출처와 라이선스가 문서화되어 있다.
- [ ] 데이터 정제·제거 내역과 split이 기록되어 있다.
- [ ] 누수 없는 Pipeline으로 baseline과 후보 모델을 비교했다.
- [ ] 반복 CV의 평균과 표준편차를 기록했다.
- [ ] 센서 수와 성능의 trade-off를 분석했다.
- [ ] OOF prediction으로 threshold를 결정했다.
- [ ] Random/time-based 평가와 FN/FP 분석을 수행했다.
- [ ] Test는 모든 설정을 동결한 뒤 1회만 평가했다.
- [ ] 모델, threshold, 버전, 재현 방법이 함께 저장되어 있다.

# SECOM WaferGuard 실행 계획

이 문서는 프로젝트의 실험 순서, 역할 분담, 완료 기준과 산출물을 관리합니다. 대외 공개용 프로젝트 소개와 사용 방법은 [README.md](../../README.md)를 참고합니다.

## 진행 상태 갱신 — 2026-10-09

- 체크는 연구·데모 범위의 구현·실험·기록 완료를 뜻하며 목표 성능 달성을 뜻하지 않습니다. 현장 적용 관련 항목은 이 계획의 완료 범위에서 제외합니다.
- V1의 S0·M3 Top-20 연구·데모 모델은 후속 시간 구간 평가까지 수행한 뒤 사용을 중단하고 `models/retired/v1_m3_sensor20/`에 보존했습니다. 당시 완료 이력은 유지하되 현재 V2의 최종 모델·문턱 동결로 해석하지 않습니다.
- V1은 `time_test.csv`를 이미 평가했습니다. 과거 완전한 미사용 여부는 보증하지 못하며, 이후 같은 구간을 개선 판단에 재사용하면 개발 검증으로 구분합니다. 아래 최종 Test 원칙은 향후 미사용 평가 데이터를 확보했을 때 적용할 기준입니다.
- V2 비교 실험과 고정 센서 20개 연구용 후보의 90%·80% 시나리오 저장·복원·추론 데모를 완료했습니다. 두 저장 후보는 같은 센서·median·XGBoost 분류기를 공유하고 문턱만 다릅니다. 독립 최종 평가는 별도이며 [V2 Model Card](model_card_v2.md)에서 범위와 한계를 확인합니다.

## 원칙

- 원본 데이터는 `data/raw/`에서 수정하지 않습니다.
- Train 데이터로만 전처리기·특징 선택기·모델을 fit합니다.
- Imputation, scaling, PCA, feature selection은 `scikit-learn Pipeline`으로 관리합니다.
- Test set은 모델·특징·하이퍼파라미터·threshold를 동결한 뒤 최종 1회만 사용합니다.
- 모든 비교 실험은 가능한 한 동일한 split과 평가 지표를 사용합니다.

## 역할 분담

| 담당 | 주요 업무 |
| --- | --- |
| 이종수 (SECOM 데이터·분석 / WM 딥러닝) | 기존 SECOM 데이터 확보·병합·정제, EDA, split, 오류·drift 분석, 문서·시각화. 확장 과제의 WM-811K CNN·ResNet18 학습, 모델 비교·평가, 저장·추론 |
| 양현승 (SECOM 모델링·평가 / 맵 전처리 에이전트) | 기존 SECOM Pipeline, baseline, 특징 선택, 후보 모델 비교, 반복 CV, threshold, 모델 저장·CLI. 확장 과제의 WM-811K 수집·품질 검사, Lot 분할, 맵 변환·Dataset, 전처리 에이전트 실행·기록 |
| 공동 | 데이터 누수 검토, 입력 시점·라벨 의미·전달 규격 확정, 최종 모델·threshold 동결, Test 평가, 재현성 검증, 발표 |

### WM-811K 확장 업무 — 2026-10-08 추가

- **맵 전처리 에이전트: 양현승.** 데이터 manifest, Lot 분할, 변환 설정과 학습용 Dataset을 이종수에게 전달한다.
- **WM 딥러닝: 이종수.** 전달된 데이터 계약을 사용해 작은 CNN·ResNet18을 비교하고 모델·평가·추론 묶음을 만든다.
- 기존 SECOM 업무 담당은 유지한다. 확장 업무는 **기초 설계 단계**이며 구현·학습 완료로 표시하지 않는다.
- WM-811K 과제는 검사 이후 웨이퍼맵 패턴 분류로 정의한다. 공개 SECOM과 같은 웨이퍼로 연결하거나 전체 2단계 검출 성능을 주장하지 않는다.
- 상세 구조, 담당별 산출물, 전달 규격과 완료 기준은 [WM811K_BASIC_DESIGN.md](../../WM811K_BASIC_DESIGN.md)에서 관리한다.

#### 결과 해석 기준 — 2026-10-09 표현 정리

- 기존 SECOM ML의 전처리·센서 선택·학습·검증과 설정값은 유지한다. 1차 모델의 출력은 불량 확정이 아니라 **후속 확인을 위한 위험 대상 선별**로 표현한다.
- WM-811K는 검사 이후 웨이퍼맵의 **결함 패턴 진단**을 담당한다. 정상 재확정·자동 출하 승인·기본 검사 대체로 표현하지 않는다.
- 미검 우선의 OOF Recall 90% 목표를 주 시나리오, 80% 목표를 선별 부담 비교 시나리오로 보존한다. 이는 비교 목적의 결정이며 기존 설정의 선별 상한이나 최종 threshold 확정을 뜻하지 않는다.
- 선별 비율은 실제 검사가 수행된 비율이 아니라 후속 확인 대상 비율이다. 1차에서 선별되지 않은 불량은 해당 가상 2차 경로로 전달되지 않는다.
- 독립 성능 평가와 가상 연결 데모를 구분한다. 대응 웨이퍼가 없는 두 공개 데이터의 결과로 통합 검출률·정상 복구 건수·불량 유출 감소를 계산하지 않는다. 구현 완료 체크와 과거 실험 수치는 변경하지 않는다.

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

### 이종수 (데이터·분석)

- [x] Sensor 데이터와 Label/Timestamp 데이터를 `data/raw/`에 저장
- [x] Sensor + Label + Timestamp 병합
- [x] 행·열 수, 클래스 분포, 중복, Timestamp 확인
- [x] 센서별 결측률 산출
- [x] 결측률 50% 초과 및 zero-variance 센서 EDA 후보 기준 확정
- [x] 제거 후보·EDA 기준 잔여 센서 수와 사유를 `logs/secom/data/dataset_log.csv`에 기록

#### Day 1 데이터 처리 원칙 및 변경 이유

- 원본 데이터 병합은 Dataset Profile의 `ingestion` source 목록과 adapter 정의를 사용해 canonical `secom_merged.csv`를 생성한다. 원본 경로·읽기 옵션·결합 규칙을 코드에 고정하지 않아 같은 수집 형식의 다른 공장 데이터는 Profile만으로 처리할 수 있다.
- `step2_data_check.py`의 결측률 50% 초과·상수 센서 결과는 전체 데이터 EDA용 후보와 근거를 기록하는 용도다. 전역 `secom_cleaned.csv`를 만들거나 그 결과를 모델 입력에 바로 적용하지 않는다. 실제 제거·대치·선택은 Train 또는 CV 학습 fold 안에서 fit해야 validation/test 정보가 학습에 섞이지 않는다.
- 모델 Pipeline의 첫 `quality_filter` 단계가 Profile의 결측률 기준 초과 센서를 먼저 제거하고, 남은 센서에서 상수를 제거한다. 제거 개수·센서명은 결과 CSV의 `quality_filter_log` 또는 OOF의 별도 `_quality_filter.csv`에 학습 fold별로 기록한다. 기존 성능·threshold 결과는 필터 적용 전의 탐색 기록이며 재실행이 필요하다.
- `secom_merged.csv`와 `dataset_log.csv`는 원본 파일과 Profile로 재생성하는 결과물이다. 저장소에는 코드·설정·문서만 유지하고, 결과 CSV는 `.gitignore`로 제외한다.

### 양현승 (모델링·평가)

- [x] Recall, PR-AUC/AP, Precision, F1, ROC-AUC 평가 기준 확정
- [x] Baseline과 후보 모델의 비교 조건 설계
- [x] PCA·L1·Feature Importance 기반 특징 선택 전략 설계

#### 양현승 결정 사항

- SECOM 실험의 기본 Fail label은 `1`로 사용한다. 평가 함수는 `positive_label` 설정값을 사용하며, 모델 비교에서는 AP와 Fail Recall을 핵심 지표로, Precision·F1·ROC-AUC를 보조 지표로 기록한다.
  - **이유:** 모든 평가 함수와 confusion matrix에서 불량 클래스를 일관되게 해석하기 위해서다. Recall은 불량을 정상으로 놓치는 False Negative 위험을, AP는 불량 비율이 낮은 데이터에서 threshold와 무관한 순위 성능을 확인한다.
- 범용 코어는 데이터셋별 가정을 직접 갖지 않는다. 공통 실험 조건은 `config.json`, 데이터 구조·Label·품질 규칙은 `configs/datasets/<dataset_id>.json` Profile로 분리한다.
  - **이유:** SECOM 외 tabular 데이터에도 같은 split·평가·Pipeline 코어를 적용하고, 새 데이터셋 추가 시 모델 코드가 특정 데이터셋에 종속되지 않게 하기 위해서다.
- 후보 모델 비교는 Test를 제외한 학습 데이터에서 `RepeatedStratifiedKFold(5 folds × 5 repeats, random_state=42)`로 수행한다. 후보 비교의 threshold는 `0.50`으로 고정하며, 최종 threshold는 Day 4 OOF prediction에서만 결정한다.
  - **이유:** 약 104개의 적은 Fail 표본으로 한 번의 split에 의존하지 않기 위해 반복 계층 CV를 사용한다. 모델 비교와 운영 threshold 결정을 분리해 공정한 비교를 유지하고, Test set의 사후 최적화를 막는다.
- Baseline은 `SimpleImputer(median) → StandardScaler → LogisticRegression` Pipeline으로 구현한다. 후보는 L1 Logistic Regression, RBF SVM, Random Forest, LightGBM, XGBoost로 구성한다. XGBoost는 기본 학습과 fold별 `scale_pos_weight` 적용을 비교한다.
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
  6. **산출물 및 점검 기록:** 팀원의 `src/sensor_ml/data_pipeline/step3_split.py`를 공식 진입점으로 유지한다. 기존 분할·EDA·요약·누수 검사 함수에 Profile 검증·그룹 경계·원본 행 ID·저장 전 검사를 통합한다. 기본 출력은 `data/splits/integrated/`의 여섯 split 및 `manifest.json`, `logs/secom/data/step3_integrated/`의 요약·누수·시간 경계 점검이다. 각 CSV에는 `__source_row_id`, `__split_role`, `__split_protocol_id`를 보존한다. 기존 split 파일은 덮어쓰지 않고 검사 실패 시 저장을 중단한다. 그림은 `reports/secom/figures/eda/step3/`에 저장한다. 별도 `step3_split_data.py`는 중복 구현을 피하기 위해 제거한다.

  **기존 결과 해석의 정정:** 기존 Time Validation 235행 중 163행이 Random Train에 포함돼 후보 비교·OOF 탐색에 사용됐다. 기존 시간 평가 수치를 독립적인 미사용 holdout 성능으로 주장하지 않는다. 위 교정 경로로 재실행하더라도 이미 관찰한 데이터가 다시 미사용 데이터가 되는 것은 아니며, 엄격한 최종 일반화 확인에는 별도의 미사용 데이터가 필요하다.

- [x] Python 환경과 `requirements.txt` 구성 확인

**예정 산출물:** `data/processed/secom_merged.csv`, `logs/secom/data/dataset_log.csv`, `src/sensor_ml/data_pipeline/step1_merge_data.py`, `src/sensor_ml/data_pipeline/step2_data_check.py`

## Day 2 — EDA, 데이터 분할, Baseline

**목표:** 재현 가능한 split을 만들고 비교 기준이 될 baseline을 구축합니다.

### 이종수 (데이터·분석)

- [x] 클래스 분포·결측률·Timestamp EDA 및 시각화
- [x] Random Train/Validation/Test = 70/15/15 split, stratify 적용
- [x] Timestamp 기준 time-based split 생성
- [x] Split별 클래스 비율을 `logs/secom/data/split_summary.csv`에 기록

  기존 요약·그림과 통합 split의 `data/splits/integrated/manifest.json`을 확인했습니다. 통합 실행의 기본 요약 경로는 `logs/secom/data/step3_integrated/`이며 같은 timestamp·그룹 보호에 따라 실제 분할 비율은 목표 비율과 다를 수 있습니다.

### 양현승 (모델링·평가)

- [x] `SimpleImputer(median) → StandardScaler → LogisticRegression` baseline 구현
- [x] Recall, AP, Precision, F1, ROC-AUC, 학습 시간을 기록

### 공동

- [x] Train/Validation/Test 간 데이터 누수와 Pipeline fit 범위 검토
- [x] Baseline 결과 검토 및 다음 실험 조건 확정

  기존 Random Train과 Time Validation 중복 이력을 확인하고 교정된 split 계약·학습 fold 내부 전처리 경로를 검토했습니다. 과거 탐색 결과가 독립 미사용 평가로 바뀌었다는 뜻은 아닙니다. Baseline 검토 근거는 보고서 12절입니다.

**예정 산출물:** `data/splits/`, `logs/secom/data/split_summary.csv`, `logs/secom/exploratory/legacy/baseline_result.csv`, `src/sensor_ml/data_pipeline/step3_split.py`, `src/sensor_ml/experiments/step4_baseline.py`

## Day 3 — 특징 선택과 후보 모델 비교

**목표:** 센서 수와 성능의 trade-off를 분석하고 최종 후보 1~2개를 고릅니다.

### 센서 축소 성능 기준 — 2026-10-07 사용자 결정

- 원본 센서 20개 이하 모델은 전체 센서 모델 대비 **AP 상대 하락률 20% 이내를 목표, 30% 이내를 최대 허용**으로 한다. 성능 유지율로는 목표 80% 이상, 최소 허용 70% 이상이다.
- 계산식은 `(전체 AP - 축소 모델 AP) / 전체 AP × 100`이다. 같은 분류기 설정·분할·평가 대상·검증 방식을 사용하며 전체 센서는 학습 구간의 품질 필터 후 남은 센서 전체를 뜻한다. 시간순 3구간 평균 AP를 우선 비교하고 반복 CV는 별도로 보조 비교한다.
- 전체 AP가 0이거나 평가에 불량이 없으면 판정 불가로 기록한다. AP 하락률은 불량 미검출 비율이 아니며 Recall·양성 판정 비율·구간별 저하는 별도로 확인한다.
- 기존 “같은 성능 유지”를 제한적인 성능 손실 허용으로 변경한 프로젝트 기준이다. 최종 Test 결과를 보고 기준을 바꾸지 않는다. 이번 결정만으로 최종 후보 선정·센서 동결·완료 체크를 수행하지 않는다.

상세 결정과 기존 결과의 사후 진단은 [공동 결정 D1](JOINT_DECISIONS.md#d1-성능-유지허용-저하-기준)을 참고한다.

### 이종수 (데이터·분석)

- [x] PCA 누적 설명분산과 Feature Importance 시각화
- [x] 센서 수와 AP의 관계 분석

  Top-K·전체 센서와 RF Top-20의 비교는 보고서 26~33절에 기록했습니다. 시각화 산출물은 `reports/secom/figures/exploratory/day3_pca_cumulative_variance.png`, `reports/secom/figures/exploratory/day3_feature_importance.png`와 `reports/secom/figures/v2/v2_sensor_importance*/`에 보존했습니다.

### 양현승 (모델링·평가)

- [x] PCA(누적 설명 분산 90%) 실험
- [x] L1 Logistic Regression 기반 특징 선택
- [x] Random Forest/LightGBM Feature Importance 기반 Top-K 비교 (100/50/30/20/10)
- [x] Logistic Regression, SVM, Random Forest, LightGBM 비교
- [x] `class_weight` 또는 `scale_pos_weight` 적용 비교
- [x] Repeated Stratified K-Fold로 평균 ± 표준편차 기록
- [x] XGBoost 기본·`scale_pos_weight` 후보를 같은 CV fold에서 비교

  근거: 보고서 25절 및 `logs/secom/exploratory/xgboost_20261006/`. V2의 시간순 가중치 비교는 아래 별도 진행 항목으로 구분합니다.

### 공동

- [x] V1 연구·데모 후보 선정: 전체 과거 학습 + S0 RF Top-20 + M3 XGBoost
- [x] V2 성능·센서 수·미검과 선별 부담을 비교해 연구·데모 후보 선정

  V1 후보 선정은 보고서 31~33절에 보존했습니다. V2는 반복 weighted gain Top-20·깊이 2·규제 1과 90%/80% 문턱 시나리오를 연구용 후보로 선정·저장했습니다. 근거는 보고서 35절과 `docs/secom/model_card_v2.md`입니다. 학습·추론 시간의 종합 정량 비교 완료를 뜻하지 않습니다.

**예정 산출물:** `logs/secom/exploratory/legacy/feature_compare.csv`, `logs/secom/exploratory/legacy/model_compare.csv`, `reports/secom/figures/exploratory/pca_variance.png`, `reports/secom/figures/exploratory/feature_importance.png`, `reports/secom/figures/exploratory/sensor_count_vs_ap.png`

## Day 4 — Threshold, 이상 탐지, 오류·드리프트 분석

**목표:** FN/FP 비교로 연구·데모 시나리오의 분류 기준을 정하고 설정을 고정합니다.

### 이종수 (데이터·분석)

- [x] False Negative/False Positive 사례 분석
- [x] Random split과 time-based split의 성능 비교
- [x] 주요 센서의 시간 변화와 drift 가능성 점검

  오류 사례는 보고서 9·17·21절, 기존 탐색 분할 비교는 4~8절, 시간 변화·월별 센서 진단은 33절에 기록했습니다. Random/Time 비교는 데이터 사용 이력이 있는 탐색 결과이며 drift가 성능 저하의 확정 원인이라는 뜻은 아닙니다.

### 양현승 (모델링·평가)

- [x] OOF prediction 생성
- [x] Threshold별 Recall, Precision, FN, FP 비교
- [x] XGBoost 후보가 선정되면 해당 모델의 OOF threshold 시나리오 비교
- [x] 모델 비교·특징 선택·OOF·시간 검증·오류 분석 및 공통 전처리 코드에 한국어 함수 설명 보강
- [x] FN/FP trade-off를 바탕으로 연구·데모용 threshold 결정: OOF Recall 목표 90%를 주 시나리오, 80%를 부담 비교 시나리오로 보존
- [x] Isolation Forest와 PCA reconstruction error를 보조 실험으로 비교

  XGBoost M3·M0 단일 OOF 시나리오는 `logs/secom/v1/m3_m0_oof_compare/`, 정상 전용 이상 탐지 비교는 보고서 23절과 `logs/secom/exploratory/anomaly_20261006/`에 기록했습니다. V2 고정 후보는 90% 문턱 `0.0029468848183751106`, 80% 문턱 `0.0339033380150795`를 선택·저장했습니다. 개발 구간에서 90%는 80%보다 불량 1건을 더 검출하고 정상 119건을 추가 선별했습니다. 근거는 [V2 Model Card](model_card_v2.md)입니다.

#### Threshold 결정 기준 및 현재 프로젝트 후보

- Threshold는 후보 모델을 고정한 뒤 `data/splits/integrated/time_train.csv` 내부의 OOF 확률로만 비교한다. 외부 Time Validation과 Test는 threshold 탐색에 사용하지 않는다. 그룹이 있으면 그룹 분리를 적용한다.
  - **이유:** Time Validation은 시간 순서 일반화 확인용이고, Test는 최종 성능 확인용이다. 두 데이터를 threshold에 맞추면 평가 데이터에 과적합될 수 있다.
- V2 연구용 후보는 고정 센서 20개·XGBoost 깊이 2·규제 1로 선정·저장했다. `lightgbm_all`과 `l1_balanced_l1_select`의 기존 비교는 탐색 이력이다. 관찰한 시간 검증 구간은 개발용 검증 데이터로 구분한다.
  - **이유:** 기존 Random Train과 Time Validation은 겹치며, 시간 검증 결과도 이미 관찰했으므로 사전 선정·독립 검증으로 소급 해석할 수 없다.
- 기존 Random Train OOF의 F1 최대 비교값 `0.000475`는 탐색 기록으로만 보존한다. 해당 결과는 Recall `0.329`, Precision `0.218`, F1 `0.262`, FN `49`, FP `86`이며 새 평가에 그대로 승계하지 않는다.
  - **이유:** 과거 Random OOF의 F1 진단값과 현재 고정 센서 후보의 시간순 OOF Recall 목표 문턱은 평가 범위와 선택 기준이 다르다.
- 후보 threshold를 정한 뒤에는 같은 값을 Time Validation에 그대로 적용해 시간 구간의 FN·FP를 확인한다. Time Validation 결과를 보고 threshold를 다시 조정하지 않는다.
  - **이유:** 시간 데이터로 threshold를 재조정하면 시간 검증이 또 다른 튜닝 데이터가 되어 일반화 성능을 과대평가할 수 있다.
- 모델·특성 집합·threshold를 모두 고정한 뒤에만 Test를 한 번 평가한다.
  - **이유:** Test set을 최종 확인용으로 보존해 성능 추정의 낙관 편향을 막는다.
- V2 고정 후보의 문턱은 센서 선택 이후 과거 데이터의 시간순 2구간 OOF에서 결정한다. 과거 반복 CV·반복 평균 OOF는 탐색 이력으로 구분하고 현재 저장 문턱에 승계하지 않는다. OOF 모델과 저장 모델의 확률 척도가 같다고 보장하지 않으며 이후 개발 구간 결과와 함께 해석한다.
- 실행 범위는 `src/sensor_ml/evaluation/run_evaluation.py`의 `--change`로 관리한다. 전처리·모델·데이터·split 변경은 OOF부터 다시 생성하고, threshold 변경은 성공한 같은 실행 폴더의 OOF 비교표를 재사용해 시간 검증을 실행한다. 선행 실패 시 평가를 중단한다. 변경 유형은 호출자가 선언하며 해시로 소스 변경을 자동 감지하지 않는다.

### 공동

- [x] V1 연구·데모 평가 대상의 특징 선택, 센서 집합, 모델, 하이퍼파라미터, threshold 동결
- [x] V2 연구·데모 후보의 특징 선택, 고정 센서 집합, 모델, 하이퍼파라미터, 시나리오별 threshold 보존

  사용자 승인으로 V1 센서 20개·저장 전처리 통계·M3 설정·문턱 `0.036539457738399506`을 고정한 뒤 후속 구간을 평가했습니다. 근거는 `docs/secom/FINAL_EVALUATION_REVIEW.md`와 `logs/secom/v1/m3_frozen_time_evaluation/evaluation.json`입니다. 이후 V1 사용 중단은 별도 개선 요청이며 당시 동결·평가 완료 이력을 취소하지 않습니다.

**예정 산출물:** `logs/secom/exploratory/legacy/threshold_compare.csv`, `logs/anomaly_compare.csv`, `reports/secom/figures/exploratory/error_analysis.png`, `reports/secom/figures/exploratory/random_vs_time.png`, `reports/secom/figures/exploratory/sensor_drift.png`

## Day 5 — 최종 평가와 출고

**목표:** 동결된 설정으로 Test를 한 번 평가하고 재현 가능한 결과물을 준비합니다.

### 이종수 (데이터·분석)

- [ ] confusion matrix와 PR curve 생성
- [x] 오류·핵심 센서·drift 분석 결과 정리
- [x] 연구·데모 결과를 보고서와 README에 업데이트

### 양현승 (모델링·평가)

- [x] V1 동결 모델의 Test 역할 후속 시간 구간 단일 평가 (과거 완전한 미사용 여부 보증 불가)
- [x] V1 후속 평가의 Recall, AP, Precision, F1, ROC-AUC 기록
- [x] V2 고정 센서 목록·학습 median·분류기·시나리오별 문턱을 포함한 추론 묶음 저장
- [x] V1의 선택 센서·학습 전처리 통계·분류기를 담은 추론 묶음 저장 및 복원 확인
- [x] V1 threshold, 사용 센서, 버전, 한계를 Model Card에 기록
- [x] 센서 입력 추론 CLI 확인 (Train 유래 데모 입력, 신규 데이터 성능 평가 아님)

  V1 후속 평가: `logs/secom/v1/m3_frozen_time_evaluation/evaluation.json`의 `completed`, `refit=false` 확인. 결과는 Recall 11.11%로 정책 미충족이며 보고서 34절에 기록했습니다. V2 저장 산출물은 `models/candidates/v2_fixed20_recall90/`, `models/candidates/v2_fixed20_recall80/`입니다. 선택기를 다시 실행하는 전체 학습 Pipeline 객체 대신 고정 센서 입력에 필요한 통계·모델을 보존하는 추론 묶음을 사용합니다. V2 독립 최종 평가는 별도입니다.

### 공동

- [x] V1 저장 묶음·별도 프로세스 복원·센서 입력 CLI 데모 확인
- [x] V1 후속 시간 구간 평가와 한계를 보고서·README·Model Card에 기록
- [x] V2 연구·데모 보고서·README·Model Card 작성 및 업데이트
- [x] V2 고정 센서 CLI demo 확인

  V1과 V2의 복원·Train 유래 CLI 데모 확인은 완료했습니다. 프로젝트 전체 재현 검증과 GitHub 공개·발표 최종 점검은 아래 V2 후속 체크리스트에서 관리합니다. 보고서 업데이트 완료와 최종 공개 점검 완료를 구분합니다.

**실제 보존 산출물:** `models/candidates/v2_fixed20_recall90/`, `models/candidates/v2_fixed20_recall80/`, `docs/secom/model_card_v2.md`, `src/sensor_ml/inference/predict_cli.py`, `docs/secom/report.md`. 두 묶음의 `experiment/`에 기존 OOF·개발 평가 기록을 보존한다. 기존 단일 `models/model.joblib`·JSON Model Card 계획은 이 구성으로 대체한다.

## 완료 기준

- [x] 데이터 출처와 라이선스가 문서화되어 있다.
- [x] 데이터 정제·제거 내역과 split이 기록되어 있다.
- [x] 학습 fold 내부 전처리·선택 Pipeline으로 baseline과 후보 모델을 비교했다. (과거 분할 사용 이력은 별도 명시)
- [x] 반복 CV의 평균과 표준편차를 기록했다.
- [x] 센서 수와 성능의 trade-off를 분석했다.
- [x] 선택 이후 시간순 OOF prediction으로 V2 연구용 90%·80% 시나리오 threshold를 결정했다.
- [x] Random/time-based 탐색 평가와 FN/FP 분석을 수행했다.
- [ ] Test는 모든 설정을 동결한 뒤 1회만 평가했다.
- [x] V1 모델, threshold, 버전, 재현 방법이 함께 저장되어 있다. (연구·데모 이력, 현재 사용 중단)

V2 연구용 시나리오 문턱 선택·저장은 완료했습니다. 독립 최종 Test는 미완료입니다. V1의 동결 후 단일 평가 이력은 Day 5에 기록했지만 데이터 전체 사용 이력을 보증할 수 없어 엄격한 미사용 최종 Test 완료로 체크하지 않습니다.

## V2 개선 및 구현 정리 — 후속 작업

- [x] 전체 센서 M0·M3 × 가중치 3수준(1·√비율·비율) 실험 구현 및 프리셋 관리
- [x] 동일한 Train 내부 시간순 외부 3구간·내부 OOF 2구간에서 6개 후보 실행
- [x] 학습 구간별 실제 가중치·센서 제거·OOF 범위와 실행 설정 기록
- [x] 평균·구간별 성능과 Recall 80%·양성 판정 비율 40% 정책 결과 확인
- [x] V2 연구·데모 모델·전처리·고정 센서·90%/80% threshold 선정 및 저장
- [x] P0 중앙값 대치 기준 결과 확보 (`v2_m0_ratio`, M0·비율 가중치)
- [x] P1 중앙값 대치 + 결측 indicator 비교 구현·실행
- [x] P2 NaN 유지 + XGBoost 자체 결측 처리 비교 구현·실행
- [x] `evaluate_frozen_model.py`의 실행 흐름·항목별 동결 검증·저장 함수 분리 (합성 테스트 7개 통과)
- [x] 양현승 실험 CLI 9개 파일의 인자 처리·결과 저장 분리 (관련 합성 테스트 42개 통과)

V2 초기 근거는 `logs/secom/v2/v2_time_weight_compare/`입니다. M0·비율 가중치는 전체 센서 기준으로 유지하며, 6개 후보 × 3개 학습 구간의 내부 OOF에서 당시 80%·40% 정책 후보를 찾지 못했습니다. 이후 P1·P2 및 센서 축소·깊이·규제 비교는 사용자 실행으로 완료했습니다. 실험 실행 완료와 최종 동결·정책 충족은 구분합니다.

### V2 개발 실험 마무리 — 2026-10-09

- [x] 반복 weighted gain Top-20·깊이 2·규제 1을 연구용 주후보로 정리, 규제 5를 부담 비교 후보로 보존
- [x] 미검 우선 OOF Recall 목표 90%와 부담 비교 80%의 Train 내부 시간순 결과 분석
- [x] 결측 처리·모델·센서 축소·깊이·규제 비교와 실험 한계를 [ML 마무리 문서](SECOM_ML_SUMMARY.md)에 정리
- [x] README와 보고서에 개발 실험 종료 및 독립 패턴 진단 확장 범위 반영

추가 ML 후보 탐색은 여기서 마무리한다. 후속 사용자 실행으로 고정 센서·시나리오별 단일 threshold·V2 저장·추론 검증을 완료했다. 독립 평가는 별도이며 기존 설정값과 V1 저장 묶음은 변경하지 않는다.

### V2 고정 센서 후보 보존·인계 완료 — 2026-10-09

- [x] 과거 551행에서 센서 20개를 선택하고 이후 개발 구간에서 고정 목록 평가
- [x] 선택 이후 과거 273행·불량 8건의 시간순 OOF로 90%·80% 문턱 선택
- [x] 과거 824행으로 학습한 후보의 마지막 개발 구간 272행·불량 9건 평가 기록
- [x] 90% 후보 저장: `models/candidates/v2_fixed20_recall90`
- [x] 재학습 없이 80% 문턱 분기 저장: `models/candidates/v2_fixed20_recall80`
- [x] 두 묶음의 별도 프로세스 복원: 각 33행, 최대 확률 차이 0.0, 판정·라벨 일치
- [x] 공통 Train 유래 20개 센서 CLI 데모: 90%는 33행, 80%는 30행 선별
- [x] 원본 OOF 기록 보존 및 각 묶음 `experiment/`에 사본 저장
- [x] 보고서·README·ML 요약·V2 Model Card 업데이트
- [ ] 프로젝트 전체 README 절차의 처음부터 재현 검증
- [ ] GitHub 공개 산출물·데이터 공개 범위·발표 자료 최종 점검
- [ ] 별도 미사용 데이터 확보 및 독립 최종 평가

개발 구간에서 90%는 TP 9·FP 248·FN 0·TN 15, 80%는 TP 8·FP 129·FN 1·TN 134였다. 두 시나리오 AP는 0.323619, ROC-AUC는 0.795944다. 한 개발 구간의 결과이며 목표 검출률 보장이 아니다. 저장·복원·데모는 성능 평가나 Test 재사용이 아니다. [시나리오 보존 기록](V2_SCENARIO_BUNDLES.md)과 [V2 Model Card](model_card_v2.md)를 인계 기준으로 사용한다.

리팩토링한 9개 파일은 `check_sensor_inference.py`, `top20_time_validation.py`, `recent_window_compare.py`, `step6_feature_compare.py`, `feature_time_compare.py`, `anomaly_compare.py`, `time_weight_compare.py`, `top20_oof_compare.py`, `temporal_validation.py`입니다. 팀원 통합 파일 `step3_split.py`는 이번 리팩토링에서 제외했습니다.

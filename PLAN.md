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
| A — 데이터·분석 | 데이터 확보·병합·정제, EDA, split, 오류·drift 분석, 문서·시각화 |
| B — 모델링·평가 | Pipeline, baseline, 특징 선택, 후보 모델 비교, 반복 CV, threshold, 모델 저장·CLI |
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

- [ ] Sensor 데이터와 Label/Timestamp 데이터를 `data/raw/`에 저장
- [ ] Sensor + Label + Timestamp 병합
- [ ] 행·열 수, 클래스 분포, 중복, Timestamp 확인
- [ ] 센서별 결측률 산출
- [ ] 결측률 50% 초과 및 zero-variance 센서 제거 기준 확정
- [ ] 제거 전후 센서 수와 사유를 `logs/dataset_log.csv`에 기록
- [ ] Recall, PR-AUC/AP, Precision, F1, ROC-AUC 평가 기준 확정
- [ ] Random / Time-based split 계획 및 leakage 검토

**예정 산출물:** `data/processed/secom_merged.csv`, `logs/dataset_log.csv`, `src/step1_merge_data.py`, `src/step2_data_check.py`

## Day 2 — EDA, 데이터 분할, Baseline

**목표:** 재현 가능한 split을 만들고 비교 기준이 될 baseline을 구축합니다.

- [ ] 클래스 분포·결측률·Timestamp EDA 및 시각화
- [ ] Random Train/Validation/Test = 70/15/15 split, stratify 적용
- [ ] Timestamp 기준 time-based split 생성
- [ ] Split별 클래스 비율을 `logs/split_summary.csv`에 기록
- [ ] `SimpleImputer(median) → StandardScaler → LogisticRegression` baseline 구현
- [ ] Recall, AP, Precision, F1, ROC-AUC, 학습 시간을 기록

**예정 산출물:** `data/splits/`, `logs/split_summary.csv`, `logs/baseline_result.csv`, `src/step3_split.py`, `src/step4_baseline.py`

## Day 3 — 특징 선택과 후보 모델 비교

**목표:** 센서 수와 성능의 trade-off를 분석하고 최종 후보 1~2개를 고릅니다.

- [ ] PCA(누적 설명 분산 90%) 실험
- [ ] L1 Logistic Regression 기반 특징 선택
- [ ] Random Forest/LightGBM Feature Importance 기반 Top-K 비교 (100/50/30/20/10)
- [ ] Logistic Regression, SVM, Random Forest, LightGBM 비교
- [ ] `class_weight` 또는 `scale_pos_weight` 적용 비교
- [ ] Repeated Stratified K-Fold로 평균 ± 표준편차 기록
- [ ] 성능, 센서 수, 학습·추론 시간을 종합해 후보 선정

**예정 산출물:** `logs/feature_compare.csv`, `logs/model_compare.csv`, `reports/figures/pca_variance.png`, `reports/figures/feature_importance.png`, `reports/figures/sensor_count_vs_ap.png`

## Day 4 — Threshold, 이상 탐지, 오류·드리프트 분석

**목표:** 운영 관점의 분류 기준을 정하고 최종 설정을 동결합니다.

- [ ] OOF prediction 생성
- [ ] Threshold별 Recall, Precision, FN, FP 비교
- [ ] FN/FP trade-off를 바탕으로 최종 threshold 결정
- [ ] Isolation Forest와 PCA reconstruction error를 보조 실험으로 비교
- [ ] False Negative/False Positive 사례 분석
- [ ] Random split과 time-based split의 성능 비교
- [ ] 주요 센서의 시간 변화와 drift 가능성 점검
- [ ] 특징 선택, 센서 집합, 모델, 하이퍼파라미터, threshold 동결

**예정 산출물:** `logs/threshold_compare.csv`, `logs/anomaly_compare.csv`, `reports/figures/error_analysis.png`, `reports/figures/random_vs_time.png`, `reports/figures/sensor_drift.png`

## Day 5 — 최종 평가와 출고

**목표:** 동결된 설정으로 Test를 한 번 평가하고 재현 가능한 결과물을 준비합니다.

- [ ] Test data 단일 최종 평가
- [ ] 최종 Recall, AP, Precision, F1, ROC-AUC 기록
- [ ] confusion matrix와 PR curve 생성
- [ ] 전처리·특징 선택·모델을 포함한 Pipeline 저장
- [ ] threshold, 사용 센서, 버전, 한계를 Model Card에 기록
- [ ] 신규 데이터 추론 CLI 확인
- [ ] README 절차로 재현성 검증 및 최종 보고서 작성

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

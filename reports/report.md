# SECOM WaferGuard 분석 보고서

> 반도체 공정 센서 데이터 기반 불량 웨이퍼 조기 탐지

## 1. 요약

본 보고서는 UCI SECOM 데이터셋을 이용해 불량(Fail) 생산 단위를 조기에 탐지하는 모델의 실험 결과를 정리합니다. 핵심 평가는 불량 미검을 줄이기 위한 **Fail Recall**과 불균형 데이터에 적합한 **PR-AUC / Average Precision**입니다.

| 항목 | 최종 결과 |
| --- | --- |
| 최종 모델 | _실험 후 입력_ |
| 사용 센서 수 | _실험 후 입력_ |
| 결정 Threshold | _실험 후 입력_ |
| Test Recall | _실험 후 입력_ |
| Test AP | _실험 후 입력_ |

## 2. 데이터와 전처리

| 항목 | 결과 |
| --- | --- |
| 전체 샘플 수 | 1,567 |
| Fail 샘플 수 | 104 |
| 원본 특징 수 | 591 |
| 제거된 센서 | _결측률·상수 센서 점검 후 입력_ |
| 최종 후보 특징 수 | _실험 후 입력_ |

- 결측률이 50%를 초과하는 센서는 제거합니다.
- 상수(Zero-variance) 센서는 제거합니다.
- 나머지 결측값은 median imputation으로 처리합니다.
- 전처리와 특징 선택은 데이터 누수를 막기 위해 Pipeline 내부에서 수행합니다.

상세 점검 결과는 `logs/dataset_log.csv`에 기록합니다.

## 3. 실험 설계

### 데이터 분할

- Random split: Train / Validation / Test = 70 / 15 / 15, stratify 적용
- Time-based split: Timestamp 기준으로 과거 데이터 학습, 이후 데이터 평가
- Test set: 모델 설정을 동결한 뒤 최종 1회만 평가

### 비교 모델과 평가 지표

| 구분 | 내용 |
| --- | --- |
| Baseline | Median imputation → StandardScaler → Logistic Regression |
| 특징 선택 | PCA, L1 Logistic Regression, Feature Importance 기반 Top-K |
| 후보 모델 | Logistic Regression, SVM, Random Forest, LightGBM |
| 검증 | Repeated Stratified K-Fold CV |
| 핵심 지표 | Fail Recall, PR-AUC / AP |
| 보조 지표 | Precision, F1-score, ROC-AUC, 학습·추론 시간 |

## 4. 모델 비교 결과

반복 교차 검증 결과를 아래 표에 기록합니다. 값은 평균 ± 표준편차 형식으로 작성합니다.

| 모델 | 특징 선택 | 센서 수 | Recall | AP | F1 | 비고 |
| --- | --- | ---: | --- | --- | --- | --- |
| Baseline | 전체 | _ | _ | _ | _ | _ |
| 후보 1 | _ | _ | _ | _ | _ | _ |
| 후보 2 | _ | _ | _ | _ | _ | _ |

상세 결과는 `logs/baseline_result.csv`, `logs/feature_compare.csv`, `logs/model_compare.csv`에 저장합니다.

## 5. Threshold 결정

Out-of-Fold prediction을 바탕으로 Recall과 오탐의 trade-off를 비교합니다.

| Threshold | Recall | Precision | False Negative | False Positive | 선택 여부 |
| ---: | ---: | ---: | ---: | ---: | --- |
| _ | _ | _ | _ | _ | _ |

**결정:** _최종 threshold와 선택 근거를 입력합니다._

상세 비교 결과는 `logs/threshold_compare.csv`에 저장합니다.

## 6. 최종 Test 평가

> Test 결과는 최종 모델과 threshold를 동결한 뒤 한 번만 기록합니다.

| 지표 | 결과 |
| --- | --- |
| Recall | _ |
| Precision | _ |
| F1-score | _ |
| PR-AUC / AP | _ |
| ROC-AUC | _ |

- Confusion matrix: `reports/figures/final_confusion_matrix.png`
- Precision-Recall curve: `reports/figures/final_pr_curve.png`
- 원본 수치: `logs/test_result.csv`

## 7. 오류 및 시간 변화 분석

### False Negative / False Positive

- False Negative 특성: _분석 후 입력_
- False Positive 특성: _분석 후 입력_
- 대응 방안: _분석 후 입력_

### Random split과 Time-based split 비교

| 평가 방식 | Recall | AP | 해석 |
| --- | ---: | ---: | --- |
| Random split | _ | _ | _ |
| Time-based split | _ | _ | _ |

시간 순서 평가에서 성능이 하락하면 공정 조건 변화 또는 데이터 드리프트 가능성을 함께 검토합니다.

## 8. 결론과 다음 단계

- **핵심 결과:** _실험 후 입력_
- **운영상 고려사항:** 높은 Recall과 False Positive 증가 사이의 trade-off를 명시합니다.
- **한계:** 작은 불량 표본 수, 단일 공개 데이터셋, 센서 중요도와 인과성의 차이
- **개선 방향:** 추가 공정 데이터 검증, 시간 기반 재학습 정책, 공정 전문가 검토

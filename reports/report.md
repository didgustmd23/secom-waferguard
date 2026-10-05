# SECOM WaferGuard 중간 분석 보고서

> 기준 시점: Step 9 Time Validation 오류 사례 분석 완료 후. Test set은 열지 않았으므로 아래 수치는 최종 성능이 아니다.

> 평가 해석 정정: 기존 Time Validation 235행 중 163행이 Random Train에도 포함되어 후보 비교·OOF 탐색에 사용됐다. 아래 수치는 탐색 기록이며 독립적인 미사용 holdout 성능이 아니다. 새 실행 경로는 시간 holdout을 먼저 분리하고 동일한 Time Train만 사용한다. 기존 `0.000475`는 승계하지 않으며, 평균 OOF와 단일 재학습 모델의 확률 척도 차이도 있으므로 성능 하락을 drift만으로 단정하지 않는다. 이미 관찰한 데이터는 재분할해도 미사용 데이터가 되지 않는다.

## 1. 요약

- SECOM 1,567개 생산 단위와 590개 센서 feature를 대상으로 불량(Fail) 조기 탐지 실험을 진행했다.
- Random Validation Baseline은 AP 0.1985, Fail Recall 0.2667을 기록했다.
- Time-based Validation Baseline은 AP 0.0679, Recall 0.0000으로 하락했다.
- Time Train/Validation 사이에 높음 우선순위 drift feature가 242개 확인됐다. Fail 비율 변화는 0.12%p로 작았다.
- 반복 CV에서 LightGBM은 AP·ROC-AUC, L1 Logistic Regression은 고정 threshold Recall이 가장 높았다.
- 특징 선택 비교에서 LightGBM 전체 590개 feature가 AP 기준으로 가장 높았다. L1 선택 모델은 평균 약 187개 feature로 줄여도 Random CV 성능 변화가 작았다.
- 그러나 LightGBM OOF F1 최대 threshold `0.000475`를 Time Validation에 그대로 적용했을 때 TP 0, FN 17이었다. 따라서 모델·threshold는 아직 동결하지 않는다.

## 2. 데이터와 품질 점검

| 항목 | 결과 |
| --- | ---: |
| 전체 샘플 수 | 1,567 |
| Fail 샘플 수 | 104 |
| 모델 입력 후보 feature 수 | 590 |
| 결측률 50% 초과 EDA 후보 | 28 |
| 상수 feature EDA 후보 | 116 |

EDA 후보는 전역 feature 제거 결과가 아니다. 실제 결측 처리와 feature 선택은 Train/CV 학습 fold 내부에서 수행한다.

## 3. 분할과 평가 원칙

- Random split: Train/Validation/Test = 70/15/15, label stratify 적용
- Time-based split: timestamp 오름차순 70/15/15 분할, stratify 미적용
- 후보 모델 비교: `random_train.csv`에서 `RepeatedStratifiedKFold(5 folds × 5 repeats)` 사용
- 후보 비교 threshold: 0.50 고정
- Test set: 모델·feature·threshold 동결 전까지 사용하지 않음

## 4. Baseline Validation 결과

| Split 방식 | Validation Fail 수 | Recall | Precision | F1 | AP | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 15 | 0.2667 | 0.2222 | 0.2424 | 0.1985 | 0.7121 |
| Time-based | 17 | 0.0000 | 0.0000 | 0.0000 | 0.0679 | 0.4584 |

Random Validation에서는 15개 Fail 중 4개를 탐지했다. Time-based Validation에서는 17개 Fail을 모두 놓쳤다.

## 5. Time-based 분포 변화 진단

| drift 우선순위 | feature 수 |
| --- | ---: |
| 높음 | 242 |
| 중간 | 102 |
| 낮음 | 246 |

Fail 비율은 Train 7.12%에서 Validation 7.23%로 0.12%p만 변했다.

| Feature | PSI | 표준화 평균 차이 | 비고 |
| --- | ---: | ---: | --- |
| sensor_78 | 5.8508 | 1.0760 | 가장 큰 PSI |
| sensor_80 | 4.0431 | 0.4108 | 큰 분포 변화 |
| sensor_570 | 3.3330 | -0.0542 | 큰 분포 변화 |
| sensor_109 | 2.6961 | 1.3355 | 결측률 84.03% → 29.79% |

이 결과는 feature 제거 결정이 아니라 시간 변화·공정 조건·측정 환경을 추가 확인할 EDA 후보를 뜻한다.

## 6. 후보 모델 반복 CV 결과

> `random_train.csv` 1,096개 샘플, 5 folds × 5 repeats, threshold 0.50 고정.

| 모델 | AP 평균 ± 표준편차 | Recall 평균 ± 표준편차 | ROC-AUC 평균 ± 표준편차 | 평균 학습 시간(초) |
| --- | --- | --- | --- | ---: |
| LightGBM | 0.1860 ± 0.0518 | 0.0000 ± 0.0000 | 0.6863 ± 0.0453 | 1.6408 |
| Random Forest | 0.1748 ± 0.0483 | 0.0027 ± 0.0131 | 0.6767 ± 0.0646 | 2.8708 |
| LightGBM (`scale_pos_weight`) | 0.1639 ± 0.0515 | 0.0027 ± 0.0131 | 0.6667 ± 0.0464 | 1.6894 |
| L1 Logistic Regression | 0.1473 ± 0.0409 | 0.1836 ± 0.0688 | 0.6236 ± 0.0530 | 0.2171 |
| L1 Logistic Regression (`class_weight="balanced"`) | 0.1403 ± 0.0371 | 0.2657 ± 0.0660 | 0.6105 ± 0.0518 | 0.2495 |
| RBF SVM | 0.1338 ± 0.0315 | 0.0000 ± 0.0000 | 0.6436 ± 0.0557 | 0.2033 |

LightGBM은 확률 순위 성능이 가장 높지만 threshold 0.50에서 Fail을 예측하지 못했다. `scale_pos_weight` 적용은 AP를 낮추고 Recall을 거의 개선하지 못했다. L1 Logistic Regression의 `class_weight="balanced"` 적용은 Recall을 0.1836에서 0.2657로 높였지만 Precision과 AP를 낮췄다.

특징 선택과 Time Validation 전의 상위 후보는 **LightGBM 원본**과 **L1 Logistic Regression (`class_weight="balanced"`)**이었다. 전자는 AP·ROC-AUC 관점, 후자는 고정 threshold Fail Recall 관점의 후보이다.

## 7. 특징 선택 비교

> `random_train.csv` 1,096개 샘플, 5 folds × 5 repeats, threshold 0.50 고정.

| 실험 | 평균 선택 feature 수 | AP 평균 | Recall 평균 | ROC-AUC 평균 |
| --- | ---: | ---: | ---: | ---: |
| LightGBM 전체 | 590.00 | 0.1860 | 0.0000 | 0.6863 |
| LightGBM Top-100 | 100.00 | 0.1799 | 0.0109 | 0.6839 |
| LightGBM Top-50 | 50.00 | 0.1740 | 0.0110 | 0.6814 |
| L1 balanced 전체 | 590.00 | 0.1403 | 0.2657 | 0.6105 |
| L1 balanced + L1 선택 | 186.88 | 0.1403 | 0.2657 | 0.6105 |
| L1 balanced + PCA 90% | 126.56 | 0.1246 | 0.3038 | 0.6125 |

LightGBM Top-K는 전체 feature보다 AP가 낮았고 feature 선택기 학습 비용도 추가됐다. L1 선택은 feature 수를 줄이면서 Random CV 지표가 유지됐지만, Time Validation AP는 0.0637로 LightGBM 전체의 0.0874보다 낮았다.

## 8. Time Validation과 OOF threshold 검증

| 구간·설정 | AP | ROC-AUC | Recall | Precision | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM, Time Validation, threshold 0.50 | 0.0874 | 0.5666 | 0.0000 | 0.0000 | 0 | 0 | 17 |
| LightGBM, Time Validation, threshold 0.000475 | 0.0874 | 0.5666 | 0.0000 | 0.0000 | 0 | 4 | 17 |
| LightGBM, Random Train OOF, threshold 0.000475 | 0.1664 | 0.7367 | 0.3288 | 0.2182 | 24 | 86 | 49 |

OOF에서는 Fail과 정상 점수의 차이가 있었지만, Time Validation에서는 Fail 점수가 전반적으로 낮아졌다. Time Validation Fail 17건의 최고 점수는 0.000033으로 고정 threshold보다 낮았고, threshold를 낮추면 FP가 빠르게 증가했다. 이는 단순 threshold 조정만으로 해결하기 어렵다는 뜻이다.

## 9. Time Validation 오류 사례 분석

| 오류 그룹 | 샘플 수 | 평균 행 결측률 | 점수 중앙값 |
| --- | ---: | ---: | ---: |
| FN | 17 | 2.87% | 0.00000109 |
| FP | 4 | 3.05% | 0.001217 |
| TN | 214 | 3.22% | 0.00000056 |

FN의 평균 행 결측률은 TN보다 낮아 단순 결측률이 Fail 미탐지의 직접 원인이라는 근거는 없다. `sensor_158`, `sensor_293`, `sensor_358`, `sensor_340`, `sensor_55`는 FN/TN 차이와 높은 drift 우선순위가 함께 관찰된 추가 확인 대상이다. 단, FN 17건·FP 4건의 작은 표본 분석이므로 feature 제거 또는 인과관계의 근거로 사용하지 않는다.

## 10. 남은 단계

- Time Train 내부 시간 순서 검증을 통한 drift 대응 개선 실험
- 개선 모델의 Time Validation 재확인과 후보·feature·threshold 동결
- Isolation Forest 또는 PCA reconstruction error 보조 실험 진행 여부 결정
- PCA 분산, feature importance, Random/Time 비교, 오류 사례 시각화
- 동결된 모델의 Test 단 한 번 평가

## 11. 한계

- Fail 샘플 수가 적어 split별 성능 변동이 크다.
- Time-based 성능 저하가 drift의 인과적 결과라고 단정할 수 없다.
- Time Validation 결과를 이미 확인했으므로, 개선 모델 선정은 Time Train 내부의 시간 순서 검증을 우선 사용해야 한다.
- Test set을 아직 평가하지 않았으므로 최종 일반화 성능은 확정할 수 없다.

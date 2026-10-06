# SECOM WaferGuard 중간 분석 보고서

> 전처리 변경: 공통 Pipeline 앞단에 학습 데이터 기준 결측률 초과·상수 센서 제거와 제거 로그를 추가했다. 1~20절은 필터 적용 전의 탐색 기록이며, 필터 적용 후 Step 4~9 재실행 결과는 21절, 코드화한 시간순 내부 검증 결과는 22절에 기록했다.

> 기준 시점: 2026-10-06 센서 품질 필터 적용 후 Step 4~9 및 Time Train 내부 시간순 검증 재실행 완료. XGBoost 추가 비교는 25절, XGBoost 센서 선택 방법별 Top-20 반복 CV 비교는 26절, 전체 vs RF Top-20 시간순 검증은 27절에 기록했다. 아래 수치는 최종 Test 성능이 아니다.

> 평가 해석 정정: 기존 Time Validation 235행 중 163행이 Random Train에도 포함되어 후보 비교·OOF 탐색에 사용됐다. 아래 수치는 탐색 기록이며 독립적인 미사용 holdout 성능이 아니다. 새 실행 경로는 시간 holdout을 먼저 분리하고 동일한 Time Train만 사용한다. 기존 `0.000475`는 승계하지 않으며, 평균 OOF와 단일 재학습 모델의 확률 척도 차이도 있으므로 성능 하락을 drift만으로 단정하지 않는다. 이미 관찰한 데이터는 재분할해도 미사용 데이터가 되지 않는다.

## 1. 요약

4~9절은 기존 탐색 실험 기록이며, 12~18절은 교정된 평가 경로의 재실행 및 추가 진단 결과다.

- SECOM 1,567개 생산 단위와 590개 센서 feature를 대상으로 불량(Fail) 조기 탐지 실험을 진행했다.
- Random Validation Baseline은 AP 0.1985, Fail Recall 0.2667을 기록했다.
- Time-based Validation Baseline은 AP 0.0679, Recall 0.0000으로 하락했다.
- Time Train/Validation 사이에서 분포 변화 우선순위가 ‘높음’인 특징 242개를 확인했다. Fail 비율 변화는 0.12%p로 작았다.
- 반복 CV에서 LightGBM은 AP·ROC-AUC, L1 Logistic Regression은 고정 threshold Recall이 가장 높았다.
- 특징 선택 비교에서 LightGBM 전체 590개 feature가 AP 기준으로 가장 높았다. L1 선택 모델은 평균 약 187개 feature로 줄여도 Random CV 성능 변화가 작았다.
- 기존 LightGBM OOF의 F1 최대 threshold `0.000475`를 Time Validation에 그대로 적용했을 때 TP 0, FN 17이었다. 이 값은 탐색 기록으로만 보존했다.
- 교정된 Time Train의 단일 OOF 후보 약 `0.00002343`을 시간 검증에 적용한 결과는 TP 1·FP 19·FN 16이었다. Time Train 내부 시간 순서 검증에서도 구간별 성능 변동이 컸으므로 최종 모델·threshold는 아직 확정하지 않았다.

## 2. 데이터와 품질 점검

| 항목 | 결과 |
| --- | ---: |
| 전체 샘플 수 | 1,567 |
| Fail 샘플 수 | 104 |
| 모델 입력 후보 feature 수 | 590 |
| 결측률 50% 초과 EDA 후보 | 28 |
| 상수 feature EDA 후보 | 116 |

EDA 후보는 전역 feature 제거 결과가 아니다. 실제 결측 처리와 feature 선택은 Train/CV 학습 fold 내부에서 수행한다.

## 3. 기존 탐색 실험의 분할과 평가 원칙

- Random split: Train/Validation/Test = 70/15/15, label stratify 적용
- Time-based split: timestamp 오름차순 70/15/15 분할, stratify 미적용
- 후보 모델 비교: `random_train.csv`에서 `RepeatedStratifiedKFold(5 folds × 5 repeats)` 사용
- 후보 비교 threshold: 0.50 고정
- Test set: 모델·feature·threshold 동결 전까지 사용하지 않음

교정된 평가 경로에서는 동일한 Time Train에서 후보 비교·특징 선택·단일 OOF threshold 탐색을 수행했다. 기존 Random Train 결과와 교정된 시간 평가 결과를 혼용하지 않으며, 상세 과정은 12~18절에 기록했다.

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

특징 선택과 Time Validation 이전의 상위 후보는 **LightGBM 전체 특징 모델**과 **L1 Logistic Regression (`class_weight="balanced"`)**이었다. 전자는 AP·ROC-AUC 관점에서, 후자는 고정 threshold의 Fail Recall 관점에서 선정한 후보였다.

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

- 동일한 Time Train 내부 시간 검증 구간에서 후보 모델 비교 및 개선 실험
- 개선 모델의 Time Validation 재확인과 후보·feature·threshold 동결
- Isolation Forest 또는 PCA reconstruction error 보조 실험 진행 여부 결정
- PCA 분산, feature importance, Random/Time 비교, 오류 사례 시각화
- 동결된 모델의 Test 단 한 번 평가

## 11. 한계

- Fail 샘플 수가 적어 split별 성능 변동이 크다.
- Time-based 성능 저하가 drift의 인과적 결과라고 단정할 수 없다.
- Time Validation 결과를 이미 확인했으므로, 개선 모델 선정은 Time Train 내부의 시간 순서 검증을 우선 사용해야 한다.
- Test set을 아직 평가하지 않았으므로 최종 일반화 성능은 확정할 수 없다.

## 12. 교정된 split 기준 Baseline·후보 비교 재실행

기존 결과는 보존하고 `data/splits/integrated/`의 동일한 Time Train을 사용했다. Train은 1,096행(Fail 78행), Validation은 235행(Fail 17행)이다. 후보 비교는 Time Train 내부의 Repeated Stratified K-Fold(5 folds × 5 repeats), threshold 0.50에서 수행했으며 Test를 평가하지 않았다.

### Baseline 시간 검증

Logistic Regression Baseline은 AP **0.0679**, ROC-AUC **0.4584**, Recall **0.0000**이었다. TP 0, FN 17, FP 2, TN 216으로 Fail을 검출하지 못했다. 이 구간은 이미 관찰한 개발용 검증 구간이며 새로운 미사용 평가 데이터가 아니다.

### Time Train 내부 후보 비교

| 후보 | AP 평균 ± 표준편차 | ROC-AUC 평균 | Recall 평균 @ 0.50 | Precision 평균 | F1 평균 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 0.2360 ± 0.0621 | 0.7591 | 0.0052 | 0.0800 | 0.0097 |
| Random Forest | 0.2323 ± 0.0658 | 0.7416 | 0.0025 | 0.0400 | 0.0047 |
| LightGBM scale_pos_weight | 0.2285 ± 0.0616 | 0.7487 | 0.0102 | 0.1400 | 0.0189 |
| L1 Logistic Regression | 0.1762 ± 0.0544 | 0.6570 | 0.1685 | 0.1928 | 0.1768 |
| RBF SVM | 0.1758 ± 0.0408 | 0.6832 | 0.0000 | 0.0000 | 0.0000 |
| L1 Logistic Regression balanced | 0.1726 ± 0.0515 | 0.6524 | 0.2635 | 0.1664 | 0.2025 |

- LightGBM의 AP 평균이 가장 높지만 Random Forest와의 차이는 약 0.0037이다. 평균 ± 표준편차 범위가 겹치지만, 범위의 중첩 여부만으로 통계적 우위를 판단할 수는 없다. 별도의 검정을 수행하지 않았으므로 우위를 확정하지 않으며, 표준편차를 신뢰구간으로 해석하지 않는다.
- 고정 threshold Recall은 L1 balanced가 가장 높다. AP 순위와 threshold 0.50의 검출 성능은 서로 다른 기준이므로 함께 검토한다.
- 기존 Random Train과 학습 표본 구성이 다르므로 과거 AP 대비 상승을 모델 자체의 개선 효과로 해석하지 않는다.
- 내부 CV는 시간 순서 검증이 아니라 과거 학습 구간 내부의 계층 CV다. 이 결과만으로 이후 시간 구간 성능을 보장하지 않는다.
- 이 단계에서는 최종 모델·threshold를 확정하지 않았다. 이후 동일한 Time Train에서 Step 6 특징 선택 비교를 재실행했다(13절).

재실행 로그: `logs/integrated/baseline.csv`, `logs/integrated/model_compare.csv`. 후보 비교 로그에는 학습 split 생성 계약을 기록했다.

## 13. 교정된 split 기준 특징 선택 비교 재실행

Step 5와 동일한 Time Train 및 5 folds × 5 repeats CV, threshold 0.50을 사용했다. 대치·scaling·PCA·L1·Importance 선택기는 매 학습 fold 안에서만 fit했으며 외부 Validation·Test는 이 실험에서 사용하지 않았다. 로그는 `logs/integrated/feature_compare.csv`에 저장했다.

| 실험 | 평균 특징 수 | AP 평균 ± 표준편차 | Recall 평균 @ 0.50 | Precision 평균 | ROC-AUC 평균 |
| --- | ---: | ---: | ---: | ---: | ---: |
| LightGBM 전체 | 590 | 0.2360 ± 0.0621 | 0.0052 | 0.0800 | 0.7591 |
| LightGBM Top-100 | 100 | 0.2302 ± 0.0760 | 0.0180 | 0.2333 | 0.7493 |
| LightGBM Top-30 | 30 | 0.2259 ± 0.0790 | 0.0492 | 0.3013 | 0.7315 |
| LightGBM Top-50 | 50 | 0.2202 ± 0.0719 | 0.0317 | 0.2167 | 0.7320 |
| LightGBM Top-20 | 20 | 0.2154 ± 0.0646 | 0.0643 | 0.3233 | 0.7287 |
| LightGBM Top-10 | 10 | 0.1975 ± 0.0689 | 0.0487 | 0.2073 | 0.7193 |
| L1 balanced 전체 | 590 | 0.1726 ± 0.0515 | 0.2635 | 0.1664 | 0.6524 |
| L1 balanced + L1 선택 | 185.56 | 0.1725 ± 0.0515 | 0.2635 | 0.1664 | 0.6522 |
| L1 balanced + PCA 90% | 127.48 | 0.1280 ± 0.0362 | 0.3085 | 0.1110 | 0.5935 |

- AP 기준 전체 LightGBM을 우선 후보로 유지한다. Top-100은 특징 수를 약 83% 줄이면서 AP 평균 차이가 약 0.0058이므로 경량 대안이다. 통계적 동등성이나 비열등성을 검증한 결과는 아니다.
- L1 선택은 특징 수를 약 69% 줄이며 전체 L1 balanced와 AP·Recall이 거의 같았다. 다만 이는 fold별 재선택 결과이며 최종 센서 집합이 고정된 것은 아니다. 선택기를 학습한 뒤 같은 L1 모델을 다시 학습하는 구성이라는 점도 고려한다.
- PCA는 Recall이 가장 높았지만 AP·Precision·F1이 낮아졌다. Recall 하나만으로 전체 특징 방식보다 우수하다고 판단하지 않는다.
- 학습 시간에는 selector 학습이 포함된다. 특징 수를 줄여도 Top-K 학습 시간은 전체 LightGBM보다 길었으며, 추론 속도는 이번 실험에서 측정하지 않았다.
- 이 단계에서는 후보를 최종 확정하지 않았다. 이후 사전 비교 후보의 시간 검증을 수행하고(14절), 동일한 Time Train의 단일 OOF에서 threshold 후보를 새로 생성했다(15절). 이미 관찰한 외부 시간 구간은 개발용 검증 구간으로 취급한다.

## 14. 교정된 split 기준 사전 후보 시간 검증

Step 5·6에서 확인한 LightGBM 전체 특징 모델과 L1 balanced 선택 모델을 동일한 Time Train 1,096행으로 학습하고, Time Validation 235행(Fail 17행)에서 평가했다. threshold는 비교 기준 0.50을 유지했고, Validation에서 threshold를 탐색하거나 Test를 평가하지 않았다. 로그는 `logs/integrated/time_validation.csv`다.

| 후보 | 특징 수 | AP | ROC-AUC | Recall | TP | FP | FN | TN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM 전체 | 590 | 0.0874 | 0.5666 | 0.0000 | 0 | 0 | 17 | 218 |
| L1 balanced + L1 선택 | 214 | 0.0637 | 0.4185 | 0.0000 | 0 | 4 | 17 | 214 |

- 두 후보 모두 threshold 0.50에서는 Fail 17건을 모두 놓쳤다. LightGBM의 AP·ROC-AUC는 상대적으로 높지만, 시간 구간에서의 검출 성능이 확보됐다고 볼 수 없다.
- LightGBM은 내부 CV AP 0.2360에서 시간 구간 AP 0.0874로 낮아졌다. 학습·평가 구간과 학습 표본 수가 다르므로 하락 원인을 drift 하나로 단정하지 않는다.
- L1 선택 수 214개는 전체 Time Train에 fit한 결과다. Step 6의 평균 185.56개는 CV 학습 fold별 평균이므로 서로 일치할 필요는 없다.
- 이후 LightGBM 전체 특징 모델을 **threshold 탐색용 작업 후보**로 두고 같은 Time Train에서 단일 OOF 확률과 threshold 비교표를 생성했다(15절). 이는 최종 모델 확정이나 현장 적용 가능성 판단을 뜻하지 않는다.
- 새 threshold는 Train 내부 OOF에서만 비교한다. 기존 `0.000475`를 승계하거나 이번 Validation 결과를 보고 값을 조정하지 않는다. 해당 시간 구간은 이미 관찰한 개발용 검증 데이터이며 독립적인 최종 holdout으로 해석하지 않는다.

## 15. 교정된 Time Train 단일 OOF threshold 비교

LightGBM 전체 특징 작업 후보에 대해 5-fold **1회** OOF를 생성했다. 모델 비교의 5-fold × 5-repeats와는 별도 과정이며 반복 평균을 사용하지 않았다. 1,096행 모두 OOF 예측 횟수 1회, 원본 ID 1,096개를 확인했다. OOF AP는 **0.1893**, ROC-AUC는 **0.7388**이다.

threshold는 OOF 확률의 101개 분위수 지점과 기본 0.50을 합한 **102개 후보**에서 비교했다. 아래 F1 최대는 이 격자 안의 최대값이지 모든 가능한 threshold의 전역 최적값이나 독립 평가 성능이 아니다.

| 비교 기준 | threshold | Recall | Precision | F1 | TP | FP | FN | TN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 기본 비교값 | 0.50 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 78 | 1018 |
| 격자 내 F1 최대 후보 | 약 0.00002343 | 0.6410 | 0.1689 | 0.2674 | 50 | 246 | 28 | 772 |

- F1 최대 후보값을 적용하면 Fail 78건 중 50건을 검출하지만 정상 246건을 오탐한다. 양성 판정은 296/1,096행, 약 **27.0%**다. Precision 16.9%는 양성 판정 중 실제 Fail 비율이며, 나머지는 오탐이다.
- 이 값은 Train 내부의 프로젝트용 비교 후보다. 현장 검사 용량·FN/FP 비용을 반영한 운영 기준이나 최종 threshold로 확정하지 않는다.
- 단일 OOF AP와 반복 CV의 fold별 AP 평균 0.2360은 산출 방식이 다르다. 하나의 반복에서 모은 pooled OOF AP를 여러 fold의 AP 평균과 동일한 지표 집계로 해석하지 않는다.
- 단일 OOF도 fold별 학습 모델과 전체 Train 재학습 모델의 확률 척도가 일치한다고 보장하지는 않는다. 시간 검증에서 예측 확률 분포의 변화 가능성을 확인해야 한다.
- 기존 Random Train 반복 평균 후보 `0.000475`는 승계하지 않았다. 이번 단계에서는 Time Validation·Test를 읽거나 threshold 탐색에 사용하지 않았다.

로그: `logs/integrated/oof_predictions.csv`, `logs/integrated/threshold_compare.csv`. 생성 계약·단일 OOF 방식·후보값의 출처 검사를 통과했다. 이후 비교표에 저장된 **반올림하지 않은 값**을 그대로 전달하여 Time Validation의 FN·FP를 확인했으며(16절), 해당 결과로 threshold를 재조정하지 않았다.

## 16. 단일 OOF 후보값의 시간 검증

Step 8 비교표의 F1 최대 후보 **0.0000234308486649443**을 그대로 사용했다. 비교표의 데이터셋·Train 생성 계약·모델·단일 OOF 방식·후보값 존재 여부를 검사한 뒤 같은 Time Train으로 LightGBM 전체 특징 모델을 학습했다. 결과는 `logs/integrated/time_validation_oof_threshold.csv`에 별도 기록했으며 기존 threshold 0.50 결과를 보존했다.

| 평가 | Recall | Precision | F1 | TP | FP | FN | TN | 양성 판정 비율 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Time Train 단일 OOF | 0.6410 | 0.1689 | 0.2674 | 50 | 246 | 28 | 772 | 27.0% |
| Time Validation, 동일 후보값 | 0.0588 | 0.0500 | 0.0541 | 1 | 19 | 16 | 199 | 8.5% |

시간 구간 AP는 **0.0874**, ROC-AUC는 **0.5666**으로 threshold 0.50 평가와 동일하다. AP·ROC-AUC는 확률 순위 지표이므로 threshold만 바꾸어도 바뀌지 않는다.

- threshold를 낮춰 Fail 1건은 검출했지만 17건 중 16건을 놓쳤다. 현재 후보를 불량 미검 감소 목표에 적합하다고 판단할 근거가 부족하므로 최종 모델·threshold 확정은 보류한다.
- 양성 판정 20건 중 실제 Fail은 1건이다. 단일 OOF보다 Recall·Precision이 낮아졌지만, 학습량·평가 구간·확률 척도 변화가 함께 있으므로 원인을 drift 하나로 단정하지 않는다.
- 이번 결과를 보고 Validation에서 threshold를 다시 탐색하지 않았고 Test도 평가하지 않았다. 시간 구간은 이미 관찰한 개발용 검증 데이터로 취급한다.
- 이후 **이번 후보값 기준 Step 9 FN/FP 분석**을 수행했다(17절). 기존 `0.000475` 오류 분석은 탐색 기록으로 보존하며 새 결과와 혼용하지 않는다. 개선 실험은 Time Train 내부에서 수행하고, 엄격한 일반화 확인에는 별도의 미사용 데이터가 필요하다.

## 17. 새 OOF 후보값 기준 FN/FP 사례 분석

동일한 Time Train/Validation에 threshold **0.0000234308486649443**을 적용해 오류 사례를 재생성했다. TP 1·FN 16·FP 19·TN 199로 Step 7 결과와 일치했고, 사례 235행의 원본 ID 235개를 확인했다. 모델·threshold를 변경하거나 Test를 평가하지 않았다.

| 오류 그룹 | 샘플 수 | 평균 행 결측률 | 예측 점수 중앙값 |
| --- | ---: | ---: | ---: |
| TP | 1 | 2.71% | 0.0000332829 |
| FN | 16 | 2.88% | 0.0000008456 |
| FP | 19 | 3.28% | 0.0000631319 |
| TN | 199 | 3.21% | 0.0000004914 |

FN의 평균 결측률은 TN보다 낮았다. 이 결과만으로 결측 처리 문제가 없다고 보장할 수는 없지만, 단순 결측량 증가가 미탐지의 원인이라는 근거는 없다. FN 16건의 최고 점수도 약 **0.0000138925**로 후보 threshold보다 낮았다. 이를 근거로 Validation에서 threshold를 낮추지는 않는다.

같은 split의 분포 변화 우선순위는 높음 242개·중간 102개·낮음 246개였다. Fail 클래스 비율 변화는 약 +0.12%p로 작았지만, 이것이 센서 분포의 안정성을 뜻하지는 않는다.

| 센서 | FN/TN 표준화 평균 차이 | 해석 |
| --- | ---: | --- |
| sensor_158 | 1.3920 | FN/TN 차이가 크고 높은 drift 우선순위도 있는 확인 대상 |
| sensor_293 | 1.3362 | FN/TN 차이가 크고 높은 drift 우선순위도 있는 확인 대상 |
| sensor_340 | 0.7126 | FN/TN 차이와 높은 drift 우선순위가 함께 있는 확인 대상 |

전체 FN/TN 절대 차이 순위에는 `sensor_345`, `sensor_100`도 포함된다. 반면 drift 점수 최상위인 `sensor_78`은 FN/TN 표준화 차이가 약 -0.0126으로 작았다. 따라서 **분포 변화 순위와 오류 그룹 차이 순위는 구분**하며, 높은 drift를 곧바로 미탐지 원인으로 해석하지 않는다.

FN 16건·FP 19건의 탐색 분석이며 센서 중요도, 인과관계, 최종 특징 선택 결과가 아니다. 이 시간 구간에서 발견한 센서를 그대로 선택하면 개발용 검증 정보가 모델에 반영되므로 이후 실험의 독립성에도 유의해야 한다.

로그: `logs/integrated/error_cases.csv`, `error_summary.csv`, `error_features.csv`, `temporal_drift_report.csv`, `temporal_drift_label_summary.csv`. 기존 탐색 로그는 보존했다. 이후 **Time Train 내부의 시간 순서 검증**으로 성능 변화가 학습 구간 안에서도 나타나는지 확인했다(18절). 최종 모델·threshold는 아직 미확정이다.

## 18. Time Train 내부 시간 순서 검증

Time Train 1,096행만 사용해 과거 구간을 학습하고 이후 구간을 평가하는 확장형 시간 검증을 3회 수행했다. 동일 timestamp를 하나의 단위로 묶어 `TimeSeriesSplit(n_splits=3)`을 적용했으며, 학습 종료 시각이 평가 시작 시각보다 앞서는지 검사했다. 외부 Time Validation과 Test는 읽지 않았다.

각 시간 구간의 학습 데이터 내부에서 단일 5-fold OOF를 생성하고, 동일한 분위수 격자의 F1 최대 threshold 후보를 구했다. 이 후보값을 해당 구간의 이후 데이터에 그대로 적용했다. 따라서 전체 Time Train에서 구한 threshold를 내부 평가 구간에 재사용하지 않았으며, 이후 구간의 label은 threshold 결정에 사용하지 않았다.

| 구간 | 학습 행 수 | 학습 Fail 수 | 평가 행 수 | 평가 Fail 수 | 구간별 threshold | AP | Recall | Precision | TP | FP | FN | TN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 272 | 39 | 279 | 22 | 0.00107949 | 0.0939 | 0.2727 | 0.0750 | 6 | 74 | 16 | 183 |
| 2 | 551 | 61 | 273 | 8 | 0.00013448 | 0.0315 | 0.0000 | 0.0000 | 0 | 52 | 8 | 213 |
| 3 | 824 | 69 | 272 | 9 | 0.00003612 | 0.2299 | 0.5556 | 0.1020 | 5 | 44 | 4 | 219 |

- 학습 구간 안에서도 이후 구간의 AP·Recall 변동이 컸다. threshold 0.50에서는 세 구간 모두 Recall이 0이었다.
- 구간별 ROC-AUC는 각각 0.4639·0.5085·0.7651이었다. 확률 순위 성능도 일정하지 않으므로 문제를 threshold 하나만으로 설명하기 어렵다.
- 구간별 학습량·Fail 비율·학습 기간이 함께 달라졌다. threshold 변화나 성능 변동을 drift 하나의 효과로 단정하지 않는다.
- 서로 다른 threshold를 적용한 세 평가 구간의 합계는 TP 11·FP 170·FN 28·TN 615다. 이는 하나의 고정 threshold를 사용한 최종 모델의 평가 결과가 아니다.
- 각 구간의 OOF는 해당 과거 학습 데이터 내부의 계층 CV다. 외부 평가 구간은 시간순으로 분리했지만, 내부 threshold 탐색까지 시간순 CV로 수행한 것은 아니다.
- 이 결과는 메모리에서 수행한 진단이며 별도 실행 스크립트나 결과 CSV를 만들지 않았다. 후보 비교는 19절에 기록했다.

## 19. Time Train 내부 시간 구간별 후보 모델 비교

18절과 동일한 Time Train의 세 확장형 시간 구간을 사용해 LightGBM, fold 학습 구간의 `scale_pos_weight`를 적용한 LightGBM, Random Forest, L1 Logistic Regression balanced를 비교했다. 모든 모델은 같은 구간을 평가하고 threshold 0.50을 사용했다. Validation·Test는 읽지 않았다.

| 시간 구간 | 후보 | AP | ROC-AUC | Recall | TP | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LightGBM | 0.0939 | 0.4639 | 0.0000 | 0 | 2 | 22 |
| 1 | LightGBM scale_pos_weight | 0.0804 | 0.4512 | 0.0455 | 1 | 4 | 21 |
| 1 | Random Forest | 0.0886 | 0.4682 | 0.0000 | 0 | 0 | 22 |
| 1 | L1 Logistic Regression balanced | 0.0863 | 0.5350 | 0.2273 | 5 | 65 | 17 |
| 2 | LightGBM | 0.0315 | 0.5085 | 0.0000 | 0 | 0 | 8 |
| 2 | LightGBM scale_pos_weight | 0.0432 | 0.5646 | 0.0000 | 0 | 0 | 8 |
| 2 | Random Forest | 0.1160 | 0.5212 | 0.0000 | 0 | 0 | 8 |
| 2 | L1 Logistic Regression balanced | 0.1698 | 0.5925 | 0.3750 | 3 | 79 | 5 |
| 3 | LightGBM | 0.2299 | 0.7651 | 0.0000 | 0 | 0 | 9 |
| 3 | LightGBM scale_pos_weight | 0.2260 | 0.7651 | 0.0000 | 0 | 0 | 9 |
| 3 | Random Forest | 0.2724 | 0.6367 | 0.0000 | 0 | 0 | 9 |
| 3 | L1 Logistic Regression balanced | 0.0431 | 0.5788 | 0.1111 | 1 | 58 | 8 |

세 구간의 단순 평균은 다음과 같다. 각 구간의 Fail 수가 다르므로, 이 평균은 pooled 전체 예측의 단일 성능 추정치가 아니다.

| 후보 | AP 평균 | ROC-AUC 평균 | Recall 평균 @ 0.50 | Precision 평균 |
| --- | ---: | ---: | ---: | ---: |
| Random Forest | 0.1590 | 0.5420 | 0.0000 | 0.0000 |
| LightGBM | 0.1184 | 0.5792 | 0.0000 | 0.0000 |
| LightGBM scale_pos_weight | 0.1165 | 0.5936 | 0.0152 | 0.0667 |
| L1 Logistic Regression balanced | 0.0997 | 0.5688 | 0.2378 | 0.0417 |

- Random Forest의 평균 AP가 가장 높지만 2·3구간 결과 차이가 크고 Recall은 세 구간 모두 0이었다. 평균 AP만으로 후보를 확정하지 않는다.
- L1 balanced는 세 구간 모두 Fail을 일부 검출했지만 FP가 총 202건(TP 9건)이었다. Recall 개선과 오탐 부담을 함께 봐야 한다.
- LightGBM의 불량 가중치는 구간 1에서만 Fail 1건을 추가 검출했고 나머지 두 구간에서는 Recall이 0이었다. 현재 설정만으로 안정적인 검출 개선이라고 결론 내릴 수 없다.
- 각 구간의 Fail 수가 22·8·9건으로 적고, 학습 표본도 272→551→824행으로 달라진다. 시간 변화 외에 표본 수와 클래스 구성도 결과에 영향을 줄 수 있으므로 원인을 drift 하나로 단정하지 않는다.
- 이 비교는 Time Train에서의 후보 탐색이다. 세 구간을 반복해 보며 후보를 비교했으므로 독립 성능 검증으로 간주하지 않는다. 모델·threshold는 여전히 미확정이며 Test는 평가하지 않았다.

## 20. 시간순 OOF threshold의 다음 구간 전이 비교

각 18절의 시간 순서 학습 구간 안에서 다시 2-fold expanding `TimeSeriesSplit`을 적용해 OOF 점수를 만들고, 102개 분위수·기본 threshold 후보 중 F1이 가장 높은 값을 골랐다. 그 threshold를 해당 학습 구간 전체에 fit한 모델로 바로 다음 시간 구간에 적용했다. 다음 구간의 label은 threshold 선택에 사용하지 않았다. 동일한 세 구간과 절차를 네 후보에 적용했다.

| 구간 | 모델 | 학습 내부 threshold | 다음 구간 Recall | Precision | F1 | TP | FP | FN | TN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LightGBM | 0.0000002065 | 0.9545 | 0.0766 | 0.1419 | 21 | 253 | 1 | 4 |
| 1 | LightGBM scale_pos_weight | 0.0000001705 | 0.9545 | 0.0775 | 0.1433 | 21 | 250 | 1 | 7 |
| 1 | Random Forest | 0.1000 | 0.9545 | 0.0808 | 0.1489 | 21 | 239 | 1 | 18 |
| 1 | L1 Logistic Regression balanced | 0.0003928528 | 0.8636 | 0.0748 | 0.1377 | 19 | 235 | 3 | 22 |
| 2 | LightGBM | 0.0056486961 | 0.0000 | 0.0000 | 0.0000 | 0 | 2 | 8 | 263 |
| 2 | LightGBM scale_pos_weight | 0.0012644740 | 0.1250 | 0.0455 | 0.0667 | 1 | 21 | 7 | 244 |
| 2 | Random Forest | 0.0801333333 | 0.6250 | 0.0242 | 0.0465 | 5 | 202 | 3 | 63 |
| 2 | L1 Logistic Regression balanced | 0.0004095034 | 1.0000 | 0.0340 | 0.0658 | 8 | 227 | 0 | 38 |
| 3 | LightGBM | 0.0026454519 | 0.1111 | 0.1667 | 0.1333 | 1 | 5 | 8 | 258 |
| 3 | LightGBM scale_pos_weight | 0.0000144060 | 0.6667 | 0.0706 | 0.1277 | 6 | 79 | 3 | 184 |
| 3 | Random Forest | 0.1856 | 0.2222 | 0.1429 | 0.1739 | 2 | 12 | 7 | 251 |
| 3 | L1 Logistic Regression balanced | 0.0783714913 | 0.2222 | 0.0217 | 0.0396 | 2 | 90 | 7 | 173 |

서로 다른 세 threshold로 나온 건수를 합산하면 다음과 같다. 세 구간의 평가는 각각 다른 운영 threshold를 사용하므로, 이는 threshold 하나의 성능이 아니라 전이 안정성을 살피기 위한 요약이다.

| 모델 | TP | FP | FN | TN | 전체 양성 판정 비율 | pooled Recall | pooled Precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 22 | 260 | 17 | 525 | 34.2% | 0.5641 | 0.0780 |
| LightGBM scale_pos_weight | 28 | 350 | 11 | 435 | 45.9% | 0.7179 | 0.0741 |
| Random Forest | 7 | 214 | 32 | 571 | 26.8% | 0.1795 | 0.0317 |
| L1 Logistic Regression balanced | 29 | 552 | 10 | 233 | 70.5% | 0.7436 | 0.0499 |

- 한 구간에서 선택한 threshold가 다음 구간으로 안정적으로 이어지지 않았다. 예를 들어 LightGBM threshold는 약 `2.07e-7`에서 `5.65e-3`, `2.65e-3`으로 크게 달라졌고, 첫 구간에서는 정상 257건 중 253건을 양성으로 판정했다.
- 이 F1 최대 탐색은 FP 비용이나 검사 용량을 반영하지 않는다. F1 후보를 적용한 결과 경보 비율은 모델·구간에 따라 크게 달랐다. 사용 가능한 검사 용량이 정의되지 않아 어느 threshold도 운영 후보로 확정하지 않는다.
- 세 학습 구간과 겹치는 시간 사용 및 반복된 후보 검토를 고려하면 이 결과는 내부 개발 진단이다. 모델의 최종 일반화 성능으로 주장하지 않으며 Time Validation/Test 성능도 대체하지 않는다.
- 결과는 메모리에서 산출했으며 전용 스크립트·CSV를 만들지 않았다. 모델·threshold는 미확정 상태다. 다음 의사결정은 현장 재검사 용량 및 FN/FP 비용 기준을 정하는 것이다. 그런 기준이 생기면 각 과거 학습 구간에서 같은 제약을 사용해 threshold를 선택하고, 새로운 미사용 시간 구간으로 검증해야 한다.

## 21. 센서 품질 필터 적용 후 재실행 (2026-10-06)

### 실행 조건과 센서 제거 기록

기존 `data/splits/integrated/time_train.csv` 1,096행(Fail 78행)과 `time_valid.csv` 235행(Fail 17행)을 그대로 사용했다. Step 4 Baseline, Step 5 후보 비교, Step 6 특징 비교, Step 8 단일 OOF, Step 7 시간 검증, Step 9 오류 분석을 재실행했다. 시간순 내부 진단인 18~20절은 이번에 재실행하지 않았다. 외부 시간 검증 구간은 이미 관찰한 개발용 구간이며 Test는 평가하지 않았다.

모든 Pipeline은 품질 필터를 학습 데이터에서 fit한 뒤 결측 대치와 필요한 scaling·특징 선택을 수행했다. 전체 Time Train과 반복 CV 375개 학습 기록(후보 6개·특징 실험 9개 × 25 folds), 단일 OOF 5개 학습 fold에서 동일하게 결측률 50% 초과 센서 24개와 남은 센서 중 상수 122개를 제거했다. 원래 590개 센서 중 146개를 제거해 **444개**를 유지했다. 이 개수는 전체 데이터 EDA의 후보 수 28개·116개와 구별한다.

반복 CV는 5 folds × 5 repeats, seed 42, 비교 threshold 0.50을 유지했다. OOF는 동일한 Train의 단일 5-fold 예측이며, 1,096개 샘플마다 예측이 정확히 한 번 생성됐음을 확인했다. 실행 환경은 Python 3.10.22, scikit-learn 1.5.2, LightGBM 4.5.0이다. Step 6·7·8·9의 LightGBM 내부 병렬 처리 수는 2였다. 동시 실행했으므로 학습 시간은 이전 실행과 직접 비교하지 않는다.

### 후보 모델 비교

| 모델 | 필터 전 CV AP | 필터 후 CV AP 평균 ± 표준편차 | Recall @ 0.50 | ROC-AUC 평균 |
| --- | ---: | ---: | ---: | ---: |
| LightGBM | 0.2360 | 0.2424 ± 0.0665 | 0.0027 | 0.7628 |
| LightGBM scale_pos_weight | 0.2285 | 0.2374 ± 0.0692 | 0.0153 | 0.7523 |
| Random Forest | 0.2323 | 0.2324 ± 0.0590 | 0.0025 | 0.7531 |
| RBF SVM | 0.1758 | 0.1716 ± 0.0361 | 0.0000 | 0.6880 |
| L1 Logistic Regression | 0.1762 | 0.1690 ± 0.0493 | 0.1637 | 0.6576 |
| L1 Logistic Regression balanced | 0.1726 | 0.1688 ± 0.0489 | 0.2510 | 0.6514 |

LightGBM 계열의 AP는 높아졌지만 모든 후보가 개선된 것은 아니다. AP 평균 차이는 기술적인 비교이며 통계적 유의성을 확인한 결론은 아니다. 비교 threshold 0.50에서의 불량 검출은 여전히 제한적이다.

### 특징 축소 비교

| 실험 | 평균 특징 수 | 필터 전 CV AP | 필터 후 CV AP 평균 ± 표준편차 | Recall @ 0.50 |
| --- | ---: | ---: | ---: | ---: |
| LightGBM 품질 필터 후 전체 | 444.00 | 0.2360 | 0.2424 ± 0.0665 | 0.0027 |
| LightGBM Top-100 | 100.00 | 0.2302 | 0.2407 ± 0.0709 | 0.0182 |
| LightGBM Top-50 | 50.00 | 0.2202 | 0.2353 ± 0.0681 | 0.0363 |
| LightGBM Top-30 | 30.00 | 0.2259 | 0.2117 ± 0.0653 | 0.0282 |
| LightGBM Top-20 | 20.00 | 0.2154 | 0.2042 ± 0.0784 | 0.0390 |
| LightGBM Top-10 | 10.00 | 0.1975 | 0.1872 ± 0.0603 | 0.0437 |
| L1 balanced + L1 선택 | 181.16 | 0.1725 | 0.1689 ± 0.0489 | 0.2510 |
| L1 balanced 품질 필터 후 전체 | 444.00 | 0.1726 | 0.1688 ± 0.0489 | 0.2510 |
| L1 balanced + PCA90 | 121.08 | 0.1280 | 0.1393 ± 0.0410 | 0.3335 |

`lightgbm_all`과 `l1_balanced_all`의 all은 품질 필터 후 남은 센서 전체를 뜻한다. PCA의 특징 수는 센서 수가 아닌 성분 수다. Top-50은 원래 590개 센서의 1/10 이하이고 AP 평균이 전체 방식에 비교적 가까웠지만, 별도의 시간 검증이나 성능 동등성 검증을 수행한 것은 아니다. Top-20의 AP 평균은 전체 방식보다 약 0.038 낮아 동일 성능을 유지했다고 주장하지 않는다.

### OOF threshold와 시간 검증

새 LightGBM 단일 OOF의 AP는 **0.1930**, ROC-AUC는 **0.7479**였다. 기존과 동일하게 101개 분위수와 0.50을 합한 102개 후보에서 F1 최대값을 비교했다. 선택한 탐색 후보는 **0.00037729474027866034**이며, 새 비교표의 출처 검사를 통과한 뒤 반올림하지 않은 값을 Time Validation에 적용했다. 이는 현장 Recall 목표·검사 용량을 반영한 최종 threshold가 아니다.

| 평가 구간·모델 | threshold | AP | ROC-AUC | Recall | Precision | TP | FP | FN | TN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Time Validation Baseline | 0.50 | 0.0683 | 0.4741 | 0.0000 | 0.0000 | 0 | 2 | 17 | 216 |
| Time Validation LightGBM | 0.50 | 0.1110 | 0.6576 | 0.0000 | 0.0000 | 0 | 0 | 17 | 218 |
| Time Validation L1 balanced 선택 | 0.50 | 0.0668 | 0.4544 | 0.0000 | 0.0000 | 0 | 8 | 17 | 210 |
| Train OOF LightGBM, 격자 내 F1 최대 후보 | 0.00037729474027866034 | 0.1930 | 0.7479 | 0.3462 | 0.2231 | 27 | 94 | 51 | 924 |
| Time Validation LightGBM, 같은 OOF 후보 | 0.00037729474027866034 | 0.1110 | 0.6576 | 0.0000 | 0.0000 | 0 | 3 | 17 | 215 |

OOF 후보의 F1은 0.2714이며 양성 판정 비율은 약 11.0%였다. 이는 threshold 탐색에 사용한 데이터의 수치로, 독립 성능 추정치가 아니다. 시간 검증 AP는 필터 전 0.0874에서 0.1110으로 높아졌지만, 새 OOF threshold로는 불량 17건을 모두 놓쳤다. 품질 필터 요구사항을 반영한 것과 조기 탐지 성능을 확보한 것은 구별해야 한다. 최종 모델·threshold는 여전히 미확정이다.

Step 9 오류 요약은 TP 0·FP 3·FN 17·TN 215로 시간 검증 결과와 일치했다. 오류 특징 비교와 행 결측률은 원래 590개 센서의 진단 정보이며, 제거 후 모델 입력 444개만의 통계가 아니다. 원본 split이 같으므로 기존 `logs/integrated/temporal_drift_report.csv`를 참고 정보로 재사용했다. 이 진단을 근거로 Validation에서 threshold를 다시 선택하지 않았다.

### 저장한 결과

새 결과는 `logs/quality_filtered_20261006/`에 저장해 기존 `logs/integrated/` 결과를 보존했다. 주요 파일은 `baseline.csv`, `model_compare.csv`, `feature_compare.csv`, `oof_predictions.csv`, `threshold_compare.csv`, `time_validation.csv`, `time_validation_oof_threshold.csv`, `error_cases.csv`, `error_summary.csv`, `error_features.csv`다. 학습별 제거 개수·센서명은 각 결과의 `quality_filter_log`, OOF는 `oof_predictions_quality_filter.csv`에 기록했다.

## 22. 코드화한 시간순 내부 검증 — 센서 필터 적용 후

### 평가 방법과 재현 명령

`src/temporal_validation.py`를 추가해 18~20절의 진단을 재생성 가능한 실험으로 정리했다. 동일한 Time Train 1,096행에서 timestamp 단위 `TimeSeriesSplit(n_splits=3)`로 아래 세 외부 구간을 만들었다. 정상·Fail 표본 수와 외부 경계는 기존 진단과 같으며, 외부 Time Validation과 Test는 읽지 않았다.

```powershell
python src/temporal_validation.py --train data/splits/integrated/time_train.csv --output-dir logs/temporal_validation_run --n-jobs 2
```

출력 폴더는 비어 있어야 한다. 확정 실행 결과는 `logs/temporal_quality_filtered_20261006_final/`에 저장했다. LightGBM·LightGBM scale_pos_weight·Random Forest·L1 Logistic Regression balanced의 공통 Pipeline을 사용했고, 센서 제거·대치·scaling은 각 학습 구간에서만 fit했다. LightGBM 불량 가중치는 해당 학습 구간에서 계산했다.

같은 외부 학습 모델의 예측 점수에 다음 세 threshold 방식을 적용했다.

- `default`: 설정의 0.50.
- `stratified_oof`: 각 과거 학습 구간 내부의 단일 계층 5-fold OOF에서 F1 최대 후보 선택.
- `temporal_oof`: 각 과거 학습 구간 내부의 timestamp 단위 확장형 2-fold OOF에서 F1 최대 후보 선택.

OOF 후보는 동일한 101개 분위수와 0.50의 중복을 제거한 최대 102개 값이다. 이후 평가 구간의 label은 threshold 선택에 사용하지 않았다. 시간순 OOF의 초기 미예측 행은 제외했으며 0점으로 채우지 않았다. 내부 시간순 경계·제외 범위를 코드로 명시했으므로, 과거 메모리 진단과의 차이를 센서 제거 하나의 효과로 단정하지 않는다.

### 시간 구간과 OOF 대상 수

| 구간 | 학습 행 / Fail | 평가 행 / Fail | 학습 종료 | 평가 시작 | 계층 OOF 행 | 시간순 OOF 행 | 초기 제외 행 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| 1 | 272 / 39 | 279 / 22 | 2008-08-18 15:41 | 2008-08-18 16:19 | 272 | 181 | 91 |
| 2 | 551 / 61 | 273 / 8 | 2008-08-29 22:56 | 2008-08-30 00:01 | 551 | 369 | 182 |
| 3 | 824 / 69 | 272 / 9 | 2008-09-13 09:19 | 2008-09-13 10:55 | 824 | 552 | 272 |

외부 학습의 센서 제거는 1구간에서 결측률 초과 20개·상수 122개로 잔여 448개, 2·3구간에서 결측률 초과 24개·상수 122개로 잔여 444개였다. 내부 OOF의 제거 기록도 각 내부 학습 데이터에서 따로 계산해 저장했다.

### 기본 threshold 0.50의 구간별 평균

| 모델 | AP 평균 ± 구간 표준편차 | ROC-AUC 평균 | Recall 평균 | 합산 TP | 합산 FP | 합산 FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.1561 ± 0.0798 | 0.5332 | 0.0000 | 0 | 1 | 39 |
| LightGBM | 0.1233 ± 0.0886 | 0.5710 | 0.0000 | 0 | 2 | 39 |
| LightGBM scale_pos_weight | 0.1201 ± 0.0912 | 0.5722 | 0.0152 | 1 | 4 | 38 |
| L1 Logistic Regression balanced | 0.0811 ± 0.0323 | 0.5605 | 0.2378 | 9 | 206 | 30 |

AP 평균과 표준편차는 서로 다른 세 시간 구간의 기술 통계이며 신뢰구간이나 반복 CV의 통계와 동일하지 않다. 기본 threshold에서도 L1은 검출과 오탐을 함께 늘렸고, AP가 높은 Random Forest는 Fail을 검출하지 못했다.

### 시간순 OOF threshold의 다음 구간 결과

아래 threshold는 표시용 반올림값이며 정확한 값은 `fold_results.csv`에 저장했다.

| 구간 | 모델 | threshold | Recall | Precision | TP | FP | FN | TN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | LightGBM | 2.0652e-7 | 0.9545 | 0.0766 | 21 | 253 | 1 | 4 |
| 1 | LightGBM scale_pos_weight | 1.7045e-7 | 0.9545 | 0.0775 | 21 | 250 | 1 | 7 |
| 1 | Random Forest | 0.103333 | 0.9091 | 0.0797 | 20 | 231 | 2 | 26 |
| 1 | L1 balanced | 0.000385867 | 0.8636 | 0.0748 | 19 | 235 | 3 | 22 |
| 2 | LightGBM | 0.005648696 | 0.0000 | 0.0000 | 0 | 2 | 8 | 263 |
| 2 | LightGBM scale_pos_weight | 0.001264474 | 0.0000 | 0.0000 | 0 | 15 | 8 | 250 |
| 2 | Random Forest | 0.030000 | 1.0000 | 0.0296 | 8 | 262 | 0 | 3 |
| 2 | L1 balanced | 0.000406272 | 0.8750 | 0.0303 | 7 | 224 | 1 | 41 |
| 3 | LightGBM | 0.002645452 | 0.2222 | 0.2857 | 2 | 5 | 7 | 258 |
| 3 | LightGBM scale_pos_weight | 5.8395e-6 | 0.6667 | 0.0545 | 6 | 104 | 3 | 159 |
| 3 | Random Forest | 0.281433 | 0.2222 | 1.0000 | 2 | 0 | 7 | 263 |
| 3 | L1 balanced | 0.095826 | 0.2222 | 0.0222 | 2 | 88 | 7 | 175 |

### OOF 방식별 오탐 부담

| 모델 | OOF 방식 | 합산 Recall | 합산 Precision | 양성 판정 비율 | TP | FP | FN | TN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 계층 | 0.3333 | 0.0619 | 25.5% | 13 | 197 | 26 | 588 |
| LightGBM | 시간순 | 0.5897 | 0.0813 | 34.3% | 23 | 260 | 16 | 525 |
| LightGBM scale_pos_weight | 계층 | 0.2051 | 0.0584 | 16.6% | 8 | 129 | 31 | 656 |
| LightGBM scale_pos_weight | 시간순 | 0.6923 | 0.0682 | 48.1% | 27 | 369 | 12 | 416 |
| Random Forest | 계층 | 0.5385 | 0.0585 | 43.6% | 21 | 338 | 18 | 447 |
| Random Forest | 시간순 | 0.7692 | 0.0574 | 63.5% | 30 | 493 | 9 | 292 |
| L1 balanced | 계층 | 0.4872 | 0.0594 | 38.8% | 19 | 301 | 20 | 484 |
| L1 balanced | 시간순 | 0.7179 | 0.0487 | 69.8% | 28 | 547 | 11 | 238 |

세 구간의 평가 표본은 824행(Fail 39행)이다. 합산 지표는 구간별로 다른 모델 학습·threshold를 적용한 오류 건수를 합친 값이며 하나의 고정 모델·threshold 성능이 아니다. 같은 외부 모델 점수를 사용하므로 threshold 방식에 따라 AP·ROC-AUC는 변하지 않는다.

시간순 OOF 방식에서 Recall이 높아진 모델은 양성 판정 비율도 34~70%로 높았다. LightGBM은 1구간 Recall 0.9545에서 2구간 0으로 떨어졌고, Random Forest는 2구간 정상 265건 중 262건을 양성 판정해 Recall 1.0을 얻었다. 3구간 Random Forest의 Precision 1.0은 검출 2건·오탐 0건이라는 작은 표본의 결과다. 이런 수치를 근거로 안정적인 운영 성능이나 단일 최종 후보를 확정하지 않는다.

### 검증·산출물·다음 결정

전체 96개 자동 테스트가 통과했다. 저장한 96개 학습 기록의 시간 경계·제거 수 합계, 10,996개 OOF 기록의 모델·구간·방식별 행 중복 및 확률 범위, 9,888개 평가 예측 기록을 확인했다. 예측 기록 수에는 같은 평가 행을 네 모델·세 방식으로 비교한 중복 표현이 포함되며, 서로 다른 평가 행 수는 824다.

출력은 `folds.csv`, `fold_results.csv`, `summary.csv`, `predictions.csv`, `oof_predictions.csv`, `oof_coverage.csv`, `threshold_compare.csv`, `quality_filter.csv`, `temporal_run.json`이다. 코드·실행 조건·구간별 모델 파라미터·품질 제거 사유를 함께 남겼다.

담당 B의 시간순 내부 검증 작업은 완료했다. 다음 필수 작업은 목표 Recall 또는 허용 재검사 비율을 공동으로 정하고 그 조건으로 OOF threshold 후보를 비교하는 것이다. 현재 F1 최대값은 검사 용량을 반영하지 않은 탐색 후보이며 최종 모델·threshold는 미확정이다.

## 23. 정상 전용 이상 탐지 비교 (2026-10-06)

### 실행과 평가 조건

`src/anomaly_compare.py`로 Isolation Forest와 PCA 복원오차를 비교했다. 입력은 기존 Time Train 1,096행이며 22절과 같은 세 외부 시간 구간(평가 279·273·272행, Fail 22·8·9행)을 사용했다. 외부 Validation과 최종 Test는 읽지 않았다.

```powershell
python src/anomaly_compare.py --train data/splits/integrated/time_train.csv --output-dir logs/anomaly_run --supervised-dir logs/temporal_quality_filtered_20261006_final --n-jobs 2
```

명령의 출력 폴더는 새 폴더여야 한다. 이번 실제 실행 폴더는 `logs/anomaly_20261006/`이다. 지도학습 비교 로그의 Train 생성 계약·학습 설정·외부 기간 및 표본 수·내부 시간 fold 수를 확인했다.

- 학습 데이터에서 정상만 추출하고 센서 품질 필터·대치·scaling·모델까지 정상만 fit했다. 불량은 OOF·다음 구간 평가의 정답에만 사용한다.
- Isolation Forest는 200개 트리, `contamination="auto"`, seed 42로 학습했다. 원본 `score_samples`의 부호를 뒤집어 클수록 이상으로 통일했다.
- PCA는 정상 데이터의 누적 설명분산 90%를 보존하고 표준화된 입력과 복원값의 평균 제곱 오차를 이상 점수로 사용했다.
- 이상 점수는 불량 확률이 아니므로 확률 threshold 0.5를 적용하거나 평가 데이터로 0~1 정규화하지 않았다. 원본 점수의 OOF 분위수와 전체 미선별 후보를 비교했다.
- 각 외부 학습 구간 내부에서 시간순 OOF 2구간을 생성했다. 초기 미예측 행은 91·182·272행 제외했고 OOF 대상 수는 181·369·552행이었다.
- 현재 `config.json`의 Recall 최소 90%·재검사 최대 20% 정책은 그대로 사용했다. 조건 미충족이면 정책 threshold를 만들지 않으며, 별도로 F1 최대 진단 후보의 다음 구간 성능을 기록했다.

### 동일 구간의 지도학습·이상 탐지 비교

모든 아래 Recall·Precision·재검사 비율은 **과거 내부 시간순 OOF F1 최대 진단 후보**를 다음 구간에 적용한 합산 결과다. 정책 통과 결과가 아니며 구간마다 학습 모델과 threshold가 다르다. 세 구간 합계는 824행, 실제 Fail 39행이다. AP는 구간별 평균이고 AP 표준편차는 신뢰구간이 아니다.

| 접근 | 모델 | 평균 AP | AP 표준편차 | 합산 Recall | 합산 Precision | 재검사 대상 비율 | TP | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 지도학습 | Random Forest | 0.1561 | 0.0798 | 0.7692 | 0.0574 | 63.5% | 30 | 493 | 9 |
| 지도학습 | LightGBM | 0.1233 | 0.0886 | 0.5897 | 0.0813 | 34.3% | 23 | 260 | 16 |
| 지도학습 | LightGBM scale_pos_weight | 0.1201 | 0.0912 | 0.6923 | 0.0682 | 48.1% | 27 | 369 | 12 |
| 이상 탐지 | Isolation Forest | 0.1083 | 0.0805 | 0.2051 | 0.0506 | 19.2% | 8 | 150 | 31 |
| 이상 탐지 | PCA 복원오차 | 0.0924 | 0.0554 | 0.5897 | 0.0529 | 52.8% | 23 | 412 | 16 |
| 지도학습 | L1 balanced | 0.0811 | 0.0323 | 0.7179 | 0.0487 | 69.8% | 28 | 547 | 11 |

### 이상 탐지의 구간별 결과

| 구간 | 모델 | 진단 threshold | AP | Recall | 재검사 대상 비율 | TP | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Isolation Forest | 0.465151 | 0.0574 | 0.0000 | 2.2% | 0 | 6 | 22 |
| 2 | Isolation Forest | 0.416782 | 0.2220 | 0.6250 | 29.3% | 5 | 75 | 3 |
| 3 | Isolation Forest | 0.421063 | 0.0457 | 0.3333 | 26.5% | 3 | 69 | 6 |
| 1 | PCA 복원오차 | 0.330096 | 0.0871 | 0.6364 | 77.4% | 14 | 202 | 8 |
| 2 | PCA 복원오차 | 0.317134 | 0.1628 | 0.7500 | 35.2% | 6 | 90 | 2 |
| 3 | PCA 복원오차 | 0.268513 | 0.0274 | 0.3333 | 45.2% | 3 | 120 | 6 |

표의 threshold는 표시용으로 반올림했으며 실행에는 CSV 원본 값을 사용한다. 두 모델 모두 구간별 점수와 성능의 변동이 컸다. Isolation Forest는 1구간 불량 22건을 모두 놓쳤고 PCA는 1구간 전체의 77.4%를 추가 검사 대상으로 지정했다. PCA의 합산 검출 건수는 LightGBM과 동일한 23건이지만 추가 검사 정상은 412건으로 LightGBM의 260건보다 많았다. 이번 설정만으로는 이상 탐지가 지도학습의 한계를 해결했다는 근거가 없다. 작은 Fail 표본에서 특정 구간의 높은 AP만으로 우수성을 확정하지 않는다.

### 정책·누수 검증·한계

현재 90%·20% 정책은 두 모델 × 세 과거 학습 구간의 내부 OOF에서 모두 미충족이었다. `policy_selection.csv`의 threshold는 비어 있고 정책 기반 다음 구간 평가는 생성하지 않았다. 이는 `fold_results.csv`의 F1 진단 threshold 존재와 모순되지 않는다.

총 18개 fit의 불량 학습 건수는 모두 0이며, 실제 정상 학습 ID·기간·표본 수를 저장했다. 외부 fit의 정상 수는 233·490·755행, 센서 수는 448·444·444개, PCA 성분 수는 88·106·117개였다. OOF 기록 2,204행과 외부 점수 기록 1,648행은 각각 두 모델의 예측을 합한 수다. 모든 fit 기간이 다음 평가 시작보다 앞서고, 품질 제거 개수 합계와 정상 학습 ID를 확인했다. 전체 108개 자동 테스트가 통과했으며 불량 센서값 변경이 학습 결과에 영향을 주지 않는지, 미래 정답 변경이 OOF·threshold 선택을 바꾸지 않는지도 검사했다.

센서·scaling·PCA가 학습 구간마다 달라지므로 원본 이상 점수의 규모도 달라질 수 있다. 여러 내부 fit의 OOF 점수로 선택한 문턱을 다음 fit에 적용하는 것은 전이 가능성 진단이며 안정적인 확률 보정이나 운영 기준을 뜻하지 않는다. 정상에서 벗어난 패턴이 반드시 Fail인 것도 아니다.

산출물은 `folds.csv`, `fold_results.csv`, `summary.csv`, `predictions.csv`, `oof_predictions.csv`, `oof_coverage.csv`, `threshold_compare.csv`, `policy_selection.csv`, `quality_filter.csv`, `anomaly_compare.csv`, `anomaly_run.json`이다. `reports/analysis.ipynb` 6절에서 표·그래프를 확인한다. MLP·Autoencoder는 이번 구현에 포함하지 않았으며 딥러닝 교과 확장으로 별도 비교가 필요하다. 최종 모델·정책·threshold는 아직 미확정이다.

## 24. 시간순 센서 축소와 선택 안정성 (2026-10-06)

### 비교 조건과 실행

`src/feature_time_compare.py`로 전체 특징과 Top-50·Top-20을 22~23절과 같은 외부 시간 구간 3개에서 비교했다. 각 과거 학습 구간 내부의 시간순 OOF 2구간으로 threshold를 선택했다. 공통 시간 검증 코어를 재사용하므로 시간 경계·초기 제외·정책 선택 규칙은 동일하다. 외부 Validation·Test는 읽지 않았다.

```powershell
python src/feature_time_compare.py --train data/splits/integrated/time_train.csv --output-dir logs/feature_time_run --n-jobs 2
```

실제 산출물은 `logs/feature_time_20261006/`에 저장했다. 기존 결과를 보존하려면 실행 시 새 출력 폴더를 지정한다.

모든 분류기는 seed 42·트리 300개의 같은 LightGBM이다. 선택 방법만 LightGBM 중요도와 원본 과제의 RF 중요도로 바꿨다. RF 중요도 실험은 Random Forest 분류기 실험과 달리 **RF 선택기 → LightGBM 분류기**다. RF 선택기도 트리 300개·seed 42이며 기본 불균형 가중치를 유지했다. 이렇게 분류기를 통일해 선택 방법 차이를 비교했다. 품질 필터·대치·선택기는 학습 fold 안에서만 fit했다.

전체 특징 수는 외부 구간별 448·444·444개이며 전체 Time Train의 품질 필터 결과는 444개다. 444개를 미리 전역 고정해 모든 과거 구간에 적용하지 않았다. 범주형 데이터에서는 One-Hot 특징 수가 원본 센서 수와 다를 수 있으나 이번 SECOM 기록은 원본 센서명 기준이다.

### 같은 시간 구간의 성능 비교

다음 Recall·Precision·재검사 비율은 **각 과거 구간 내부 OOF의 F1 최대 진단 후보**를 다음 구간에 적용한 합산 결과다. 현재 90%·20% 정책을 충족한 최종 성능이 아니다. AP는 구간별 평균이고 표준편차는 신뢰구간이 아니다. 평가 합계는 824행·Fail 39행이다.

| 선택 방법 | 특징 수 | 평균 AP | AP 표준편차 | 합산 Recall | 합산 Precision | 재검사 대상 비율 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 전체 | 448/444/444 | 0.1233 | 0.0886 | 0.5897 | 0.0813 | 34.3% | 23 | 260 | 16 |
| LightGBM 중요도 | 50 | 0.0857 | 0.0479 | 0.6923 | 0.0682 | 48.1% | 27 | 369 | 12 |
| LightGBM 중요도 | 20 | 0.1081 | 0.0500 | 0.3333 | 0.0660 | 23.9% | 13 | 184 | 26 |
| RF 중요도 | 50 | 0.1006 | 0.0403 | 0.5897 | 0.0488 | 57.2% | 23 | 448 | 16 |
| RF 중요도 | 20 | 0.1038 | 0.0633 | 0.4615 | 0.0534 | 40.9% | 18 | 319 | 21 |

전체 특징 모델의 평균 AP가 가장 높았고 모든 축소 후보의 평균 AP는 낮았다. 다만 소수 Fail·세 구간의 변동이 있으므로 통계적 유의성이나 일반적인 모델 우수성을 확정하지 않는다.

LightGBM Top-50은 전체 모델보다 불량 4건을 더 찾았지만 정상 109건을 추가 선별해 전체 검사 대상이 113건 늘었다. Top-20은 전체 검사 대상을 86건 줄였으나 불량을 10건 덜 찾았다. RF 중요도 Top-50은 검출 건수가 동일한데 정상 추가 검사 수가 188건 많았다. 따라서 검사량·센서 수·검출률을 함께 비교해야 하며 센서 축소만으로 성능 문제가 해결됐다고 볼 수 없다.

### 구간별 AP와 불안정성

| 모델 | 1구간 AP | 2구간 AP | 3구간 AP |
| --- | ---: | ---: | ---: |
| 전체 | 0.0939 | 0.0326 | 0.2435 |
| LightGBM Top-50 | 0.0845 | 0.0277 | 0.1450 |
| LightGBM Top-20 | 0.0940 | 0.0551 | 0.1752 |
| RF 중요도 Top-50 | 0.0673 | 0.0773 | 0.1574 |
| RF 중요도 Top-20 | 0.0752 | 0.0447 | 0.1916 |

LightGBM Top-20은 1·2구간 AP가 전체 모델보다 높았지만 3구간에서는 낮았다. RF Top-50은 2구간 Recall 1.0을 얻었지만 FP 236건을 만들었고, 3구간에서는 불량 9건을 전부 놓쳤다. 센서 수가 적다는 이유로 안정적인 미래 성능을 주장하지 않는다.

### 선택 센서와 빈도

각 모델의 외부 fit 3회·내부 OOF fit 6회를 별도 기록했다. `selected_features.csv`에는 fit별 특징명·선택 방법·중요도·순위·원본/변환 특징 구분이 있고, `feature_frequency.csv`에는 역할별 분모와 선택 횟수가 있다. 빈도에는 품질 필터에서 제외된 경우도 비선택으로 반영된다. 학습 구간이 겹치므로 독립적인 반복 실험이나 통계 검정으로 해석하지 않는다.

| 선택 방법 | 외부 fit의 선택 특징 합집합 | 세 외부 fit에서 공통 선택한 특징 |
| --- | ---: | ---: |
| LightGBM Top-50 | 108개 | 9개 |
| LightGBM Top-20 | 42개 | 4개 |
| RF 중요도 Top-50 | 91개 | 16개 |
| RF 중요도 Top-20 | 38개 | 5개 |

LightGBM의 기본 split 중요도와 RF의 불순도 감소 중요도는 척도가 달라 숫자 자체를 비교하지 않는다. 선택 빈도와 순위도 불량의 인과 센서를 뜻하지 않는다.

전체 Time Train에서 각 방법을 다시 fit해 444·50·20·50·20개의 연구용 후보 목록을 `candidate_features.csv`에 저장했다. 모든 행은 `is_final=False`다. 이 후보 목록을 과거 CV의 특징으로 전역 고정하지 않았으며, 최종 학습 범위·선정 방법을 합의하면 해당 범위에서 다시 선택해야 한다.

### 정책·검증·남은 결정

현재 90%·20% 정책은 5개 모델 × 3개 내부 OOF에서 모두 미충족이었다. 기본 threshold 0.5 결과와 F1 최대 진단 결과를 별도로 보존했지만 정책 통과 후보를 자동 채택하지 않았다.

평가 fit 45회와 전체 Train 후보 fit 5회의 품질 제거 기록·특징 수를 저장했다. 선택된 센서명이 Pipeline의 실제 입력과 일치하는지, Top-K가 정확히 50·20개인지, 외부/내부 빈도 분모가 3·6인지, 후보가 최종으로 표시되지 않는지 테스트했다. 전체 110개 테스트가 통과했다. `reports/analysis.ipynb` 7절에서 비교 표와 선택 빈도 그래프를 확인한다.

축소 비교와 연구용 센서 자료 준비는 완료했다. 최종 모델·센서 집합 합의와 비전체 모델을 채택할 경우 OOF·시간 검증·오류 분석·실행기의 모델 연결은 남아 있다. 이 축소 실험 당시 `run_evaluation.py`의 대상은 `lightgbm_all`이었으며, 이후 XGBoost 연결과 비교를 추가했다(25절). 딥러닝 교과 평가가 추가됐으므로 MLP 비교도 최종 설정 동결 전에 별도로 수행해야 한다.

## 25. XGBoost 추가 비교 (2026-10-06)

### 실험 조건

동일한 `data/splits/integrated/time_train.csv` 1,096행(정상 1,018행·Fail 78행)만 사용했다. 기존 실험과 split 생성 계약이 같으며, 외부 Time Validation과 최종 Test는 이번에 읽거나 평가하지 않았다.

- 후보 비교: 5-fold × 5-repeats, seed 42, threshold 0.50.
- 품질 필터: 학습 fold에서 결측률 50% 초과·상수 센서를 제거한 뒤 중앙값 대치. 반복 CV에서 두 XGBoost 후보 모두 444개 센서를 사용했다.
- XGBoost: 트리 300개, 깊이 3, learning rate 0.05, subsample·colsample_bytree 0.8, reg_lambda 1.0, `tree_method=hist`. 이번 실험에서는 튜닝·SMOTE를 적용하지 않았다.
- 가중치 후보: 각 학습 구간에서 계산한 정상/Fail 건수 비율을 `scale_pos_weight`로 적용했다.
- threshold 정책: 최소 Recall 90%·최대 재검사율 20%. 미충족 시 임의 문턱으로 대체하지 않는다.
- 환경: Python 3.10.22, scikit-learn 1.5.2, LightGBM 4.5.0, XGBoost 2.1.3, pandas 2.2.3.

모델 설정과 fold별 학습 근거는 `logs/xgboost_20261006/temporal/temporal_run.json` 및 `quality_filter.csv`에 저장했다. 반복 CV는 모델 내부 병렬 수 1, 시간순·단일 OOF는 2로 실행했다. 실험들을 병행했으므로 학습 시간은 엄밀한 속도 비교 자료가 아니다.

### 반복 CV — 기존 후보를 포함한 재실행

| 모델 | AP 평균 ± 표준편차 | Recall 평균 (threshold 0.50) | ROC-AUC 평균 |
| --- | ---: | ---: | ---: |
| XGBoost 기본 | 0.2490 ± 0.0827 | 1.27% | 0.7541 |
| LightGBM 기본 | 0.2424 ± 0.0665 | 0.27% | 0.7628 |
| XGBoost 가중치 | 0.2410 ± 0.0834 | 8.97% | 0.7467 |
| LightGBM 가중치 | 0.2374 ± 0.0692 | 1.53% | 0.7523 |
| Random Forest | 0.2324 ± 0.0590 | 0.25% | 0.7531 |
| RBF SVM | 0.1716 ± 0.0361 | 0.00% | 0.6880 |
| L1 Logistic Regression | 0.1690 ± 0.0493 | 16.37% | 0.6576 |
| L1 Logistic Regression balanced | 0.1688 ± 0.0489 | 25.10% | 0.6514 |

XGBoost 기본 모델의 AP가 가장 높았지만 LightGBM과의 차이는 약 0.0066이다. 표준편차는 신뢰구간이 아니며 반복 CV fold도 독립 표본이 아니므로, 이 순위만으로 통계적 우위나 최종 후보 확정을 주장하지 않는다. XGBoost 가중치는 0.50에서 Recall을 높였지만 AP는 기본 모델보다 낮았다. 기존 여섯 모델의 지표 평균·표준편차는 이전 로그와 모두 일치했다.

원본: `logs/xgboost_20261006/model_compare.csv`.

### Train 내부 시간순 검증

과거 학습 → 다음 시간 구간 평가를 세 번 수행했다. 외부 평가 대상은 합계 824행(Fail 39행)이며, 최초 학습 구간은 평가에 포함하지 않는다. 각 과거 학습 구간 안에서만 계층 단일 OOF와 시간순 OOF를 만들었다. 시간순 OOF의 초기 미예측 91·182·272행은 제외하고 기록했다.

아래 Recall·재검사율은 시간순 내부 OOF의 F1 최대 진단 문턱을 다음 구간에 적용한 합산 결과다. 정책을 충족한 문턱이나 하나의 최종 모델 성능이 아니다.

| 모델 | 시간 구간 AP 평균 ± 표준편차 | 합산 Recall | 합산 Precision | 재검사율 | TP / FP / FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| LightGBM 기본 | 0.1233 ± 0.0886 | 58.97% | 8.13% | 34.34% | 23 / 260 / 16 |
| LightGBM 가중치 | 0.1201 ± 0.0912 | 69.23% | 6.82% | 48.06% | 27 / 369 / 12 |
| XGBoost 기본 | 0.1553 ± 0.0649 | 53.85% | 7.89% | 32.28% | 21 / 245 / 18 |
| XGBoost 가중치 | 0.1749 ± 0.1332 | 71.79% | 6.21% | 54.73% | 28 / 423 / 11 |

XGBoost 가중치의 AP는 세 구간에서 0.1412 → 0.0312 → 0.3522로 크게 변했다. 평균만 보면 가장 높지만 중간 구간에서는 낮았고, 진단 Recall 증가에는 높은 재검사 부담이 따랐다. 기본 XGBoost의 AP 역시 0.1342 → 0.0886 → 0.2432로 변했다. 성능 하락의 원인을 drift 하나로 단정하지 않는다.

네 모델 × 세 외부 구간 × 두 내부 OOF 방식의 정책 선택 24건은 모두 90%·20% 조건 미충족이었다. LightGBM 두 후보의 시간순 요약 지표는 기존 실험과 일치했다.

원본: `logs/xgboost_20261006/temporal/summary.csv`, `fold_results.csv`, `policy_selection.csv`.

### 전체 Time Train의 단일 OOF — 운영 제약 비교

두 XGBoost 후보 모두 1회 5-fold로 샘플당 한 번의 OOF 확률을 생성했다. AP는 기본 0.1877, 가중치 0.1798이었다. 이 AP는 fold별 AP를 평균한 반복 CV 지표와 계산 방식이 다르다.

저장된 101개 분위수와 기본 threshold 후보 중, Recall 90% 이상에서 재검사율이 가장 낮은 후보를 비교했다.

| 모델 | 실제 Recall | 재검사율 | TP | FP | FN | 90%·20% 충족 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| LightGBM 기본 — 기존 단일 OOF | 91.03% | 66.97% | 71 | 663 | 7 | 아니오 |
| XGBoost 기본 | 91.03% | 63.96% | 71 | 630 | 7 | 아니오 |
| XGBoost 가중치 | 91.03% | 67.97% | 71 | 674 | 7 | 아니오 |

기본 XGBoost는 같은 TP 71·FN 7에서 LightGBM보다 정상 재검사를 33건 줄였다. 다만 상한 20%와는 여전히 차이가 크다. 재검사율 20% 이하의 저장 후보 중 최대 Recall은 XGBoost 기본 46.15%(TP 36·FP 162·FN 42), 가중치 42.31%(TP 33·FP 176·FN 45)였다.

이 결과는 저장된 후보 격자 안의 비교이며 모든 가능한 threshold의 정확한 최적값을 뜻하지 않는다. 상세 문턱은 각 `threshold_compare.csv`의 반올림하지 않은 값을 확인한다. 정책 통과 후보가 없으므로 최종 threshold·모델을 확정하거나 외부 Validation을 추가 평가하지 않았다.

원본: `logs/xgboost_20261006/xgboost/`, `logs/xgboost_20261006/xgboost_scale_pos_weight/`.

### 재현 명령과 다음 단계

아래는 이번 실행 조건이다. 시간순 실험을 재실행할 때는 비어 있는 새 출력 폴더를 지정한다.

```powershell
python src/step5_model_compare.py --train data/splits/integrated/time_train.csv --output logs/xgboost_20261006/model_compare.csv
python src/temporal_validation.py --train data/splits/integrated/time_train.csv --output-dir logs/xgboost_20261006/temporal --models lightgbm lightgbm_scale_pos_weight xgboost xgboost_scale_pos_weight --n-jobs 2
python src/step8_threshold_oof.py --train data/splits/integrated/time_train.csv --experiment xgboost --oof-output logs/xgboost_20261006/xgboost/oof_predictions.csv --threshold-output logs/xgboost_20261006/xgboost/threshold_compare.csv --score-method single --n-jobs 2
python src/step8_threshold_oof.py --train data/splits/integrated/time_train.csv --experiment xgboost_scale_pos_weight --oof-output logs/xgboost_20261006/xgboost_scale_pos_weight/oof_predictions.csv --threshold-output logs/xgboost_20261006/xgboost_scale_pos_weight/threshold_compare.csv --score-method single --n-jobs 2
```

현재 판단은 **XGBoost 기본 모델을 후속 비교 후보로 유지하되, 운영 가능 모델로 확정하지 않는다**이다. 가중치도 비교 기록으로 보존한다. 다음 필수 확장은 같은 Train·학습 fold 내부 전처리·시간순 검증을 사용하는 MLP 딥러닝 비교다. 정책 합의와 최종 모델·threshold 동결 전까지 Test를 사용하지 않는다.

## 26. XGBoost 센서 선택 방법별 Top-20 비교 (2026-10-06)

### 실험 조건과 비교 대상

1단계로 같은 Time Train 1,096행(정상 1,018행·Fail 78행)에서 세 실험을 비교했다. `RepeatedStratifiedKFold` 5-fold × 5-repeats, seed 42, threshold 0.50을 사용했다. 이는 시간순으로 분리한 Train **내부의 계층 무작위 CV**이며 미래 시간 구간 평가가 아니다. 외부 Time Validation과 Test는 사용하지 않았다.

품질 필터·중앙값 대치·센서 선택은 매 학습 fold 내부에서만 fit했다. 세 실험의 모든 fold에서 원본 590개 센서 중 결측률 50% 초과 24개, 상수 122개를 제거해 444개가 남았다. 이후 두 축소 실험은 각각 정확히 20개를 선택했다.

| 실험 이름 | 센서 선택 방식 | 최종 분류기 |
| --- | --- | --- |
| `xgboost_all` | 품질 필터 후 전체 444개 | XGBoost 기본 |
| `xgboost_top_20` | XGBoost gain 중요도 상위 20개 | 같은 XGBoost 기본 |
| `xgboost_rf_top_20` | RF 불순도 감소 중요도 상위 20개 | 같은 XGBoost 기본 |

최종 분류기 설정은 세 실험 모두 트리 300개, 깊이 3, learning rate 0.05, subsample·colsample_bytree 0.8, reg_lambda 1.0, `tree_method=hist`, `scale_pos_weight=1`이다. RF 선택기는 트리 300개·Gini 기준을 사용했다. 선택용 모델과 축소 특징으로 학습하는 최종 분류기는 별개의 모델이다. 모델 내부 병렬 수는 2이며 튜닝·SMOTE·threshold 조절은 적용하지 않았다.

### 반복 CV 결과

| 센서 선택 방식 | 최종 분류기 입력 수 | AP 평균 ± 표준편차 | Recall 평균 | Precision 평균 | F1 평균 | ROC-AUC 평균 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 전체 센서 | 444 | 0.2490 ± 0.0827 | 1.27% | 20.00% | 0.0238 | 0.7541 |
| RF 중요도 Top-20 | 20 | 0.2459 ± 0.0523 | 5.85% | 27.91% | 0.0933 | 0.7826 |
| XGBoost gain Top-20 | 20 | 0.2044 ± 0.0561 | 3.28% | 23.33% | 0.0557 | 0.7489 |

Recall·Precision·F1은 threshold 0.50에서의 fold별 평균이다. 합산 confusion matrix로 계산한 지표나 최종 성능이 아니다. AP는 threshold 0.50으로 이진화하기 전의 예측 확률 순위를 평가한다. 전체 센서 모델의 지표 평균·표준편차는 25절의 기본 XGBoost 결과와 일치했다.

같은 fold끼리 AP를 비교하면 다음과 같다. 아래 차이는 `Top-20 AP - 전체 AP`이며 표준편차는 25개 차이의 기술 통계다.

| 축소 방식 | AP 차이 평균 ± 표준편차 | 전체보다 AP가 높은 fold | 전체 평균 AP 대비 비율 |
| --- | ---: | ---: | ---: |
| RF 중요도 Top-20 | -0.0031 ± 0.0683 | 13 / 25 | 98.76% |
| XGBoost gain Top-20 | -0.0446 ± 0.0830 | 8 / 25 | 82.08% |

**RF 중요도 Top-20을 우선 후속 비교 후보로 유지한다.** 평균 AP 감소가 약 0.0031로 작고 ROC-AUC와 고정 문턱 Recall도 전체보다 높았다. 반면 XGBoost gain Top-20은 평균 AP가 약 0.0446 낮았다. 서로 다른 중요도 수치의 크기를 직접 비교하지 않고, 선택 후 동일한 분류기의 성능을 비교한 판단이다.

98.76%는 두 평균 AP의 비율일 뿐 성능 동등성의 증명이 아니다. 허용 성능 저하 기준은 아직 합의하지 않았고 반복 fold는 독립 표본이 아니므로, 표준편차나 13/25 승리만으로 통계적 우위·동등성을 주장하지 않는다. 세 모델 모두 threshold 0.50의 Recall이 낮으며 현재 90%·20% 정책을 통과했다는 뜻도 아니다.

### 선택 센서 안정성과 남은 확인

RF 선택의 25개 fold에서 등장한 센서는 총 63종, XGBoost 선택은 253종이었다. RF는 `sensor_59`, `sensor_64`, `sensor_65`를 25회 모두 선택했고 `sensor_40`, `sensor_67`은 23회 선택했다. XGBoost 선택에서 25회 모두 등장한 센서는 `sensor_59` 하나였다. RF 선택이 이 CV에서 상대적으로 더 일관됐지만, 동일한 데이터를 반복한 빈도이며 센서의 물리적 원인이나 최종 센서 집합을 의미하지 않는다.

중요한 범위 구분은 다음과 같다.

- 이번 결과는 **각 fold에서 선택한 20개**의 성능이다. 고정된 하나의 센서 20개 집합을 검증한 결과가 아니다. 전체 Train의 중요도나 CV 선택 빈도로 목록을 고정한 뒤 같은 CV를 다시 평가하면 선택 누수가 발생할 수 있다.
- 현재 Pipeline은 품질 필터·대치·선택기를 거치므로 추론 입력에는 기존 원본 센서 컬럼이 필요하다. **최종 분류기가 20개를 사용한다는 것과 원본 20개만 입력받는 추론 Pipeline 완성은 다르다.** 방법 확정 후 최종 학습에서 센서를 고정하고, 선택된 열만으로 동작하는 추론 구성을 별도로 준비해야 한다.
- 다음 비교는 전체 XGBoost와 RF Top-20의 **Time Train 내부 시간순 검증**이다. 필요한 threshold는 각 과거 학습 구간의 내부 OOF에서만 결정한다. 미래 구간의 결과에 문턱을 직접 맞추지 않는다.
- 시간 일반화 확인·허용 저하 기준 합의·최종 센서 동결 전에는 “590개를 20개로 줄이고 같은 성능을 냈다”는 포트폴리오 문구를 확정하지 않는다. 필수 딥러닝 MLP 비교도 별도로 남아 있다.

### 산출물과 재현 명령

`src/modeling_models.py`에 두 Top-K 선택 방식과 전체 XGBoost 실험을 추가했다. K는 기존 설정의 `top_k_feature_counts`를 사용하므로 다른 데이터에도 같은 코어를 사용할 수 있다. `src/step6_feature_compare.py`는 `--experiments`로 지정한 실험만 실행하며, 옵션을 생략하면 기존 PCA·L1·LightGBM 실험 목록을 유지한다.

결과는 `logs/xgboost_top20_20261006/`에 기존 로그와 분리해 저장했다.

- `feature_compare.csv`: 세 실험 요약 및 fold별 품질 필터 기록.
- `details/fold_results.csv`: 75개 실험-fold의 지표·특징 수·학습 시간.
- `details/selected_features.csv`: 두 선택 방법의 1,000개 센서 선택 기록(2 × 25 × 20), 순위·중요도·특징 종류.
- `details/feature_run.json`: 입력 경로·split 계약·설정·라이브러리 버전·최종 미확정 상태.

모든 선택 기록이 원본 센서이고 fold별 중복 없이 20개인지 확인했다. 공통 분류기 설정 유지·fold별 기록·학습 시 상수였던 센서의 검증 입력 변화에 대한 방어를 추가 검증했으며 전체 테스트 121개가 통과했다.

재실행 시 상세 결과 폴더는 비어 있는 새 폴더로 지정한다.

```powershell
python src/step6_feature_compare.py --train data/splits/integrated/time_train.csv --experiments xgboost_all xgboost_top_20 xgboost_rf_top_20 --output logs/xgboost_top20_run/feature_compare.csv --details-dir logs/xgboost_top20_run/details --n-jobs 2
```

## 27. XGBoost 전체 vs RF Top-20 시간순 검증 (2026-10-06)

### 검증 범위와 데이터 사용 원칙

26절에서 반복 CV 성능이 가까웠던 `xgboost_all`과 `xgboost_rf_top_20`을 동일한 Time Train 내부의 확장형 시간 구간 3개에서 비교했다. 외부 Time Validation 파일과 최종 Test는 읽지 않았다. 이미 개발에 사용한 Train의 시간순 진단이며 독립적인 최종 성능이 아니다.

각 구간에서 과거 데이터만으로 품질 필터·중앙값 대치·RF 센서 선택·XGBoost 학습을 수행했다. 최종 분류기와 RF 선택기는 앞선 비교와 같은 설정으로 트리 300개·seed 42·내부 병렬 수 2를 사용했다. 튜닝·SMOTE·가중치 변경은 적용하지 않았다. 동일 timestamp는 분리하지 않고 학습 종료 시각이 평가 시작 시각보다 빠른지 검사했다.

각 과거 학습 구간 내부에서 시간순 OOF 2구간으로 threshold 후보를 만들었다. 초기 미예측 91·182·272행은 제외했고 0점으로 채우지 않았다. 외부 구간의 label은 threshold를 정한 후 성능 평가에만 사용했다. 현재 정책인 최소 Recall 90%·최대 양성 판정 비율 20%는 변경하지 않았다.

### 구간별 AP

평가 범위는 2008-08-18 이후부터 2008-09-26까지이며, 합계 824행 중 Fail은 39행이다. 최초 학습 구간 272행은 외부 평가에 포함하지 않았다. 정확한 시간 경계는 `folds.csv`에 기록했다.

| 시간 구간 | 과거 학습 수 / Fail | 다음 구간 평가 수 / Fail | 전체 모델 센서 수 | 전체 AP | RF Top-20 AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 272 / 39 | 279 / 22 | 448 | 0.1342 | 0.0874 |
| 2 | 551 / 61 | 273 / 8 | 444 | 0.0886 | 0.0450 |
| 3 | 824 / 69 | 272 / 9 | 444 | 0.2432 | 0.2206 |
| 구간 평균 | — | — | — | **0.1553 ± 0.0649** | **0.1177 ± 0.0748** |

RF 선택 모델은 모든 외부 fit과 내부 OOF fit에서 원본 센서 20개를 사용했다. 전체 센서 모델은 세 구간의 AP·threshold·confusion matrix가 25절의 기본 XGBoost 시간순 결과와 일치했다.

세 구간 모두 전체 모델의 AP가 높았으며 RF Top-20의 평균 AP는 약 0.0377 낮았다. 평균 비율은 약 75.75%로, 반복 CV의 98.76% 유지 비율을 시간순 평가에서 재현하지 못했다. 허용 저하 기준이나 통계적 동등성 검정 없이 “20개로 같은 성능을 유지했다”고 주장하지 않는다. 구간별 학습량·불량 비율·분포가 다르므로 CV와 시간순 AP 차이의 원인을 drift 하나로 단정하지 않는다.

### Threshold 적용 결과와 정책 충족 여부

기본 threshold 0.50에서 전체 모델은 TP 0·FP 0·FN 39, RF Top-20은 TP 2·FP 9·FN 37이었다. 두 모델 모두 이 기본 문턱으로는 검출률이 낮았다.

아래 표는 각 과거 학습 구간의 시간순 OOF에서 **F1 최대 진단 문턱**을 정하고 다음 구간에 적용한 합산 결과다. 구간마다 학습 모델·센서 목록·문턱이 달라 하나의 최종 모델 성능이 아니다. 90%·20% 정책 통과 문턱으로 대체해 해석하지 않는다.

| 모델 | 합산 Recall | 합산 Precision | 양성 판정 비율 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 전체 XGBoost | 53.85% | 7.89% | 32.28% | 21 | 245 | 18 |
| RF Top-20 → XGBoost | 69.23% | 5.51% | 59.47% | 27 | 463 | 12 |

양성 판정 비율은 `(TP + FP) / 전체 평가 수`이며 실제 추가 검사를 수행했다는 뜻이 아니다. RF Top-20은 전체보다 불량 6건을 더 찾았으나 정상 오탐도 218건 늘었다. 선택 방식뿐 아니라 OOF에서 정한 문턱도 서로 다르므로 Recall 증가를 센서 축소의 단독 효과로 해석하지 않는다.

특히 RF Top-20의 두 번째 시간 구간은 TP 8·FP 263으로 273행 중 271행(99.27%)을 양성으로 판정했다. Recall 100%지만 선별 효과가 낮은 결과이므로 Recall만 보고 우수한 후보로 채택하지 않는다. 첫 번째와 세 번째 구간에서는 각각 TP 16·FP 190·FN 6, TP 3·FP 10·FN 6이었다.

정책 판정 6건(두 모델 × 세 구간)은 내부 OOF에서 모두 **미충족**이었다. `policy_selection.csv`의 threshold는 결측으로 남겼고 정책 기반 미래 성능은 계산하지 않았다. 위 F1 진단 비교는 별도의 연구용 기록으로 보존했다. 기준을 결과에 맞춰 완화하거나 미래 구간에서 threshold를 재탐색하지 않았다.

### 센서 목록과 현재 판단

RF의 외부 학습 3회에서 등장한 센서는 38종이며, 모두 선택된 센서는 `sensor_267`, `sensor_67`, `sensor_562`, `sensor_441`, `sensor_169`의 5개였다. 내부 OOF fit 6회의 선택 목록은 별도로 기록했고 등장 센서는 59종이었다. 학습 구간이 겹치는 빈도 자료이며 인과관계·최종 센서 고정 근거로 단독 사용하지 않는다.

전체 Time Train으로 연구용 Top-20 후보도 만들었지만 `is_final=False`로 기록했다. 이 목록을 과거 CV·시간순 평가에 소급 적용하지 않았고 모델·threshold·센서 집합을 동결하지 않았다. 현재 추론 Pipeline은 원본 센서 입력을 요구하며 원본 20개만 받는 추론 구성도 미완료다.

**판단:** RF Top-20은 센서 축소 비교 후보로 보존하되, 전체 모델과의 성능 동등성 또는 운영 가능성을 주장하지 않는다. 현재 전체 XGBoost가 시간순 AP의 비교 기준이다. 다음 작업은 합의된 허용 저하 기준 아래에서 센서 선택·특징·모델 개선을 비교하고, 필수 딥러닝 MLP 후보를 동일한 누수 방지·시간순 평가 조건에 추가하는 것이다. 최종 후보와 threshold 확정 전까지 Test는 사용하지 않는다.

### 실행 코드·산출물

`src/feature_time_compare.py`에 `--experiments`를 추가해 공통 XGBoost Pipeline을 기존 시간순 코어에 연결했다. 옵션을 생략하면 기존 LightGBM 전체·Top-50/20 목록을 유지한다. XGBoost gain과 RF 중요도는 센서 기록에서도 구분한다. 전체 테스트 123개가 통과했고 실제 실행의 과거→미래 경계·fold별 20개 원본 센서·기존 전체 모델 결과 일치를 확인했다.

결과는 `logs/xgboost_top20_time_20261006/`에 저장했다. `summary.csv`, `fold_results.csv`, `folds.csv`에서 성능과 구간을, `oof_coverage.csv`, `threshold_compare.csv`, `policy_selection.csv`에서 문턱 출처와 미충족을 확인한다. `selected_features.csv`, `feature_frequency.csv`, `candidate_features.csv`에는 선택 센서·빈도·최종 미확정 후보가 있다. `quality_filter.csv`와 `candidate_quality_filter.csv`에는 실제 센서 제거 기록, `feature_run.json`에는 설정·모델 이름·환경 버전·split 계약을 보관했다.

재실행할 때는 비어 있는 새 출력 폴더를 지정한다.

```powershell
python src/feature_time_compare.py --train data/splits/integrated/time_train.csv --experiments xgboost_all xgboost_rf_top_20 --output-dir logs/xgboost_top20_time_run --n-jobs 2
```

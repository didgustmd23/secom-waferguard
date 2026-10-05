# SECOM WaferGuard 중간 분석 보고서

> 기준 시점: 기존 Step 9 탐색 결과, 교정된 Step 3 기반 Step 4~9 재실행 및 Time Train 내부 시간 순서 검증 완료. 아래 수치는 최종 Test 성능이 아니다.

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

# SECOM 센서 20개 이하 모델 — 개선 및 마무리 계획

> 작성일: 2026-10-07. 담당 B: 양현승. 기준 자료: [보고서 25~27절](reports/report.md#25-xgboost-추가-비교-2026-10-06), 현재 소스와 기존 실행 로그.
>
> 사용자 결정: SECOM의 최종 모델은 원본 센서 20개 이하를 사용해야 한다. 이번 계획에서는 딥러닝을 제외한다. 1~4번을 상세히 정리하며, 5~7번은 후속 작업으로 기록만 한다. 이 문서 작성으로 코드 변경·새 학습·최종 Test 평가를 수행하지 않는다.

## 1. SECOM의 필수 조건과 현재 상태

### 확정된 조건

- 최종 모델이 요구하는 **원본 센서는 20개 이하**여야 한다. 현재 비교의 목표는 Top-20이다.
- 전체 센서 모델은 성능 비교 기준으로 유지하지만, 센서 수 조건을 충족하는 최종 결과물로 대체하지 않는다.
- 최종 추론도 선택된 센서만으로 가능해야 한다. 분류기가 20개를 사용하더라도 앞단에서 수백 개 센서를 요구하면 목표가 완성된 것이 아니다.
- 센서 선택·대치·스케일링·튜닝은 학습 구간 내부에서만 수행한다. 외부 평가 구간과 Test로 센서를 고르지 않는다.
- PCA 성분 20개는 원본 센서 20개와 다르므로 이 조건의 대체 수단으로 취급하지 않는다.
- 이 센서 수 제한은 **SECOM 과제의 조건**이다. 범용 코어에 모든 데이터셋의 센서 수를 20개로 고정하지 않는다.
- **2026-10-07 사용자 결정:** 전체 센서 모델 대비 AP 상대 하락률은 **20% 이내를 목표, 30% 이내를 최대 허용**으로 한다. AP 유지율로는 목표 80% 이상, 최소 허용 70% 이상이다. 기존 “같은 성능 유지” 목표를 제한적인 성능 손실 허용으로 변경한다.

`AP 상대 하락률 = (전체 센서 모델 AP - 20개 이하 모델 AP) / 전체 센서 모델 AP × 100`. 같은 분류기 설정·데이터 분할·검증 방식·평가 대상에서 비교한다. 전체 센서는 해당 학습 구간의 품질 필터 후 남은 센서 전체다. 시간순 3구간 평균 AP를 우선 비교하고 반복 CV는 별도로 보조 비교한다. 전체 AP가 0이거나 평가에 불량이 없으면 판정 불가다. 이 비율은 불량 미검출 비율이 아니며 Recall·양성 판정 비율·구간별 저하는 따로 확인한다. 최종 Test를 본 뒤 허용 기준을 바꾸지 않는다.

센서 이름만 익명화된 SECOM에서는 `sensor_*` 열을 원본 센서 단위로 센다. 다른 데이터에 범주형 인코딩이나 파생 특징을 적용할 경우에는 원본 센서 수와 변환 후 특징 수를 따로 기록해야 한다.

### 현재 확인된 성능

| 비교 | 전체 XGBoost | RF Top-20 → XGBoost | XGBoost gain Top-20 → XGBoost |
| --- | ---: | ---: | ---: |
| Time Train 내부 반복 CV AP | 0.2490 ± 0.0827 | 0.2459 ± 0.0523 | 0.2044 ± 0.0561 |
| 시간순 3구간 AP | 0.1553 ± 0.0649 | 0.1177 ± 0.0748 | 이번 시간순 비교에서는 미실행 |

반복 CV에서는 RF Top-20의 평균 AP가 전체 모델에 가까웠으나, 시간순 평가에서는 세 구간 모두 낮았다. **“590개를 20개로 줄여 같은 성능을 유지했다”는 목표는 아직 달성하지 못했다.**

새 기준으로 기존 결과를 사후 해석하면 시간순 평균 AP 유지율 약 75.75%·상대 하락률 약 24.25%로 최대 허용 범위 안이지만 목표 20% 이내에는 미달한다. 구간별 저하와 Recall·양성 판정 비율은 별도 확인이 필요하며, 최종 모델·센서 동결 완료를 뜻하지 않는다. 상세 결정은 [공동 결정 D1](JOINT_DECISIONS.md#d1-성능-유지허용-저하-기준)에 기록한다.

출처: `logs/xgboost_top20_20261006/feature_compare.csv`, `logs/xgboost_top20_time_20261006/summary.csv`.

### 아직 확정하지 않은 기준

- 구간별 심한 저하와 Recall·양성 판정 비율의 보조 조건. 평균 AP 허용 하락률은 위 사용자 결정으로 확정했다.
- 현장 적용 시의 최소 검출률·검사 용량·비용 기준. 프로젝트 임시 정책은 아래 사용자 결정으로 설정했다.
- 개선 실험을 종료할 범위와, 목표 미달 시 연구용 결과로 정리하는 기준.

**현재 정책 — 2026-10-07 사용자 결정:** `config.json`은 최소 Recall 80%·양성 판정 비율 상한 40%로 변경했다. 기존 90%·20%는 과거 도전 시나리오로 기록을 보존하며, 아래 완료 실험의 정책 미충족은 당시 설정 기준이다. 새 정책이 시간 검증에서도 충족됐다고 해석하지 않는다.

M3 단일 OOF의 사용자 공유 결과는 Recall 80.77%·양성 비율 37.14%(TP 63·FP 344·FN 15)다. 37.14%는 관측값이고 40%는 선택한 프로젝트 상한이며 현장 표준이 아니다. 조건 충족 후보 중 양성 비율 최소 → Recall 최대 → threshold 최대 규칙과 미충족 시 기록만 하는 처리는 유지한다. 기존 CSV·JSON은 변경하지 않았으며 새 정책으로 후보를 재선택해야 한다. 고정 threshold 0.50과 AP 하락률 목표 20%·최대 허용 30%는 변경하지 않는다.

## 2. 센서 선택 방식과 모델 설정 개선 방법

### 진행 기록 — 2026-10-07 S0~S3 비교 완료

S0(기존 RF 선택기 → XGBoost)를 전체 센서 XGBoost와 같은 Train에서 재실행했다. CV AP 상대 하락률은 **1.24%로 목표 달성**, 시간순 평균 AP 하락률은 **24.25%로 최대 허용 범위 안·목표 미달**이다. 시간순 1·2구간 하락률은 34.84%·49.18%여서 평균 판정만으로 최종 선택하지 않는다. 상세 AP·FN/FP·정책 미충족과 근거 로그는 [보고서 28절](reports/report.md#28-s0--기존-rf-선택기-기준-실험-2026-10-07)에 기록했다.

이후 S1(최소 leaf 2)·S2(깊이 8·최소 leaf 2)의 CV·시간순 실행도 완료했다. 최종 XGBoost는 유지했다. S1의 CV/시간순 AP 하락률은 **1.13% / 39.36%**, S2는 **0.02% / 24.36%**다. 시간순 평균에서 S1은 허용 초과, S0·S2는 허용 범위·목표 미달이며 세 후보 모두 목표 20%를 달성하지 못했다. S0·S2의 1·2구간 저하는 여전히 크다.

S2는 진단용 OOF 문턱에서 S0 대비 FP 54건을 줄였으나 TP도 1건 줄었다. **권고는 S1의 우선순위를 낮추고 S0·S2를 유지하는 것**이며 최종 선택은 아직 확정하지 않는다. 상세 수치와 다음 결정은 [보고서 29절](reports/report.md#29-s0s2--rf-선택기-복잡도-제한-비교-2026-10-07)에 기록했다. S2 시간순 근거는 사용자 실행 폴더 `logs/s2_rf_top20_time_user/`다. 다음 실험은 사용자 결정 후 진행하며 센서 목록·threshold·모델 동결은 별도 후속 작업이다.

S3(계층 bootstrap 10회 반복 RF 선택)의 CV·시간순 결과도 확인했다. **시간순 AP 개선은 확인되지 않았으며, S0·S2를 우선 검토하는 권고를 유지한다.** 최종 XGBoost는 기존 M0로 같고 센서 선택만 변경했다.

| 실험 | CV AP | CV 상대 하락률 | 시간순 AP | 시간순 상대 하락률 | 시간순 AP 판정 |
| --- | ---: | ---: | ---: | ---: | --- |
| 전체 센서 기준 | 0.248973 | — | 0.155342 | — | 비교 기준 |
| S0 | 0.245881 | 1.24% | 0.117674 | 24.25% | 허용 범위·목표 미달 |
| S1 | 0.246154 | 1.13% | 0.094197 | 39.36% | 허용 초과 |
| S2 | 0.248924 | 0.02% | 0.117505 | 24.36% | 허용 범위·목표 미달 |
| S3 | 0.246246 | 1.10% | 0.114201 | 26.48% | 허용 범위·목표 미달 |

S3의 시간 구간별 AP 하락률은 39.48%·45.08%·12.55%다. 2구간은 S0·S2보다 개선됐지만 1·3구간은 낮아 평균 성능은 개선되지 않았다. 진단용 F1 최대 OOF 문턱에서 Recall 82.05%·양성 판정 비율 68.33%(TP 32·FP 531·FN 7)로, S0보다 TP 5건과 FP 68건이 늘었다. 이 진단 문턱은 Recall 90%·양성 비율 20% 정책 통과값이 아니다. CV fit 평균도 S0 1.72초에서 S3 15.00초로 약 8.73배 늘었다.

네 후보 모두 센서 20개 조건은 지켰으나 시간순 AP 하락률 목표 20%는 달성하지 못했다. 구간별 성능, 목록 안정성, 학습 비용과 근거 로그는 [보고서 30절](reports/report.md#30-s0s3--반복-rf-센서-선택-결과-비교-2026-10-07)에 기록했다.

### 기준 선택기 결정 — 2026-10-07

사용자 결정으로 **S0를 후속 개선 실험의 기준 선택기로 채택하고 S2는 비교 후보로 보존**한다.

- S0: RF 트리 300개, `min_samples_leaf=1`, `max_depth=None`, 반복 선택 없음(`rf_stability_repeats=0`). 각 학습 fold 내부에서 중요도 Top-20을 선택한다.
- S2: `min_samples_leaf=2`, `max_depth=8`, 반복 선택 없음. 기존 코드·실행 기록·결과를 유지하며 필요한 후속 비교에 사용한다.
- 결정 이유: 시간순 AP와 불량 검출을 우선한다. S0·S2의 시간순 AP 차이는 작으므로 S0의 통계적 우월성을 입증한 결정은 아니다. S2의 낮은 진단 선별 부담은 비교 근거로 보존한다.
- 이번 결정은 선택 방법의 채택이다. 고정 센서 목록·최종 XGBoost 설정·threshold·배포 모델을 동결한 것이 아니며 M1~M5 실행 범위도 별도로 결정한다.

### 2.1 비교 원칙

1. 기존 RF Top-20 → 기본 XGBoost를 기준 실험으로 둔다.
2. 먼저 **센서 선택 방식만** 바꾸고 최종 XGBoost 설정은 유지한다.
3. 다음으로 유망한 선택 방식을 유지하고 **최종 모델 설정만** 바꾼다.
4. 전체 센서 모델과 같은 구간의 AP 차이, 구간별 성능, Recall·FP·양성 판정 비율을 함께 기록한다.
5. 사전에 정한 작은 후보 목록을 비교한다. 처음부터 모든 파라미터 조합을 대규모 탐색하지 않는다.

선택기와 전처리를 평가 데이터까지 포함해 학습하면 성능이 부풀려질 수 있으므로, 중첩된 학습 범위를 유지한다. [scikit-learn 누수 방지 문서](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)

### 2.2 센서 선택 개선안

S0~S3의 CV·시간순 비교 결과를 확인했다. 아래 설정은 비교에 사용한 조건이며 최적값으로 확정한 것은 아니다. 결과 요약은 위 진행 기록, 상세 분석은 보고서 30절에 있다.

| 순서 | 방법 | 시작안 | 확인할 점 |
| --- | --- | --- | --- |
| S0 | 기존 RF 중요도 Top-20 | 트리 300개, 현재 설정 | 기존 결과를 기준으로 보존 |
| S1 | RF 선택기 복잡도 제한 | `min_samples_leaf=2`, 나머지 유지 | 작은 표본의 중요도 순위가 덜 흔들리는지 |
| S2 | RF 선택기 깊이도 제한 | `max_depth=8`, `min_samples_leaf=2` | 안정성과 불량 구분 정보 보존의 균형 |
| S3 | 학습 구간 내부 반복 선택 | 과거 학습 데이터만 재표집해 선택 빈도·평균 순위로 Top-20 결정 | 한 번의 RF 학습에 과도하게 의존하는지 |

#### S0~S3 설정 관리와 실행 기록

현재 실험 설정은 공통 JSON·코드 기본값·CLI로 나뉘어 관리한다. **S0~S3는 문서상의 실험 번호이며 `configs/experiments/s3.json` 같은 독립 설정 파일이나 CLI 실험 이름으로 등록된 상태는 아니다.** 전체 비교 경로는 `xgboost_all`, 축소 경로는 모두 `xgboost_rf_top_20`을 사용하고 RF 옵션·결과 폴더·실행 기록으로 구분한다.

| 관리 위치 | 관리 내용 | 이번 비교의 기준 |
| --- | --- | --- |
| `config.json` | seed, CV, 기본 threshold, Top-K 목록, AP 하락률·threshold 정책 | seed 42, 5-fold × 5회, threshold 0.50, AP 하락률 목표 20%·최대 30% |
| `configs/datasets/secom.json` | 센서·label·timestamp 구조, 결측·상수 제거 기준 | 학습 범위에서 결측률 50% 초과·상수 제거 |
| `src/modeling_models.py` | RF·최종 XGBoost의 기본 학습 설정 | RF 트리 300개, 최종 XGBoost는 기존 M0 유지 |
| CLI | RF 최소 leaf·최대 깊이·반복 선택 횟수, 실행 대상·경로·병렬 수 | 아래 S0~S3 옵션으로 선택기만 변경, `--n-jobs 2` |
| 결과 폴더의 `feature_run.json` | 실행 당시 공통 설정·RF 옵션·Train 경로·split 계약·패키지 버전 | 현재 설정 파일이 아니라 해당 실행 기록으로 결과를 해석 |

| 실험 | `--rf-min-samples-leaf` | `--rf-max-depth` | `--rf-stability-repeats` | 의미 |
| --- | ---: | --- | ---: | --- |
| S0 | 1 (기본값) | 생략: 제한 없음 | 0 (기본값) | 기존 RF 한 번으로 중요도 Top-20 선택 |
| S1 | 2 | 생략: 제한 없음 | 0 | 최소 leaf만 제한 |
| S2 | 2 | 8 | 0 | 최소 leaf와 깊이 제한 |
| S3 | 1 (기본값) | 생략: 제한 없음 | 10 | S0 RF 설정으로 계층 bootstrap 반복 선택 |

S3의 10회는 **선택기 내부 재표집 횟수**이며 CV의 5-fold × 5회와 별개다. 반복 선택 횟수 기본값 0은 기존 단일 선택을 뜻한다. 깊이의 “제한 없음”은 CLI 옵션 생략으로 지정하며 `--rf-max-depth 0`은 유효하지 않다. S3에 S2 옵션까지 추가하면 두 개선을 결합한 별도 실험이므로 현재 S3와 구분한다.

| 실험 | CV 실행 기록 | 시간순 실행 기록 |
| --- | --- | --- |
| S0 | `logs/s0_rf_top20_cv/details/feature_run.json` | `logs/s0_rf_top20_time/feature_run.json` |
| S1 | `logs/s1_rf_top20_cv/details/feature_run.json` | `logs/s1_rf_top20_time/feature_run.json` |
| S2 | `logs/s2_rf_top20_cv/details/feature_run.json` | `logs/s2_rf_top20_time_user/feature_run.json` (사용자 실행) |
| S3 | `logs/s3_rf_top20_cv/details/feature_run.json` | `logs/s3_rf_top20_time/feature_run.json` |

RF 옵션은 `rf_selector_parameters`의 `min_samples_leaf`, `max_depth`, `stability_repeats`, `resampling`으로 남긴다. 과거 S0는 이 필드 도입 전, S1·S2는 반복 선택 옵션 도입 전 기록이므로 없는 필드를 사후 추가하지 않고 당시 단일 선택 실행 조건과 함께 해석한다. CV 실행은 `--details-dir`을 지정해야 설정·상세 센서 기록이 저장된다. 시간순 실행은 `--output-dir` 아래에 저장된다.

S3의 `selected_features.csv`에는 센서별 재표집 선택 빈도·평균 순위·반복 횟수, `bootstrap_records.csv`에는 반복별 품질 제거 개수·행 수를 기록한다. 이 빈도는 같은 학습 fold 내부 bootstrap 빈도이며 시간 구간 간 선택 빈도인 `feature_frequency.csv`와 구분한다.

재실행은 비어 있는 새 폴더를 사용하고, 폴더 이름만으로 실험을 판단하지 않는다. 현재 `feature_run.json`은 소스 원문·모델 묶음을 저장하거나 코드 변경을 자동 감지하는 장치는 아니다. 코드 기본값까지 바꾸었다면 해당 변경을 별도 기록하고 새 실험으로 비교한다. 실험별 JSON로 통합하는 구조 변경은 아직 하지 않았다.

S3의 빈도는 **현재 외부 학습 구간 안에서 새로 계산**한다. 기존 25개 CV fold 또는 세 시간 평가 구간의 전체 선택 빈도로 목록을 고정한 뒤 같은 평가를 다시 하면 미래 정보가 섞일 수 있다. 각 내부 fit도 품질 필터·대치를 독립적으로 학습한다. 반복 횟수·재표집 방식·동률 처리 규칙은 실행 전에 기록한다.

필요할 경우 후속 대안으로 상관이 높은 센서의 중복을 줄이는 선택, 과거 학습 구간 내부의 별도 검증 부분을 이용한 permutation importance를 검토한다. 상관·결측률·중요도 역시 외부 평가 데이터에서 계산하지 않는다. Permutation 방식은 추가 학습·반복 예측 비용이 있어 첫 실험군에는 넣지 않는다.

RF 불순도 중요도는 고유값이 많은 특징에 편향될 수 있다. 중요한 센서의 물리적 원인이 증명됐다는 뜻은 아니며 선택 후 실제 성능으로 판단한다. [RF 중요도 공식 예제](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html)

### S3 실행 방법 — 반복 RF 선택 (완료한 실험의 조건 기록)

사용자 요청으로 S3 실행 경로를 준비했다. 첫 실험은 **S0 RF 설정(트리 300개·깊이 제한 없음·최소 leaf 1)을 유지하고 계층 bootstrap 10회**를 적용한다. 각 클래스에서 현재 학습 행 수만큼 복원 추출하며, 각 반복에서 품질 필터·중앙값 대치·RF를 독립적으로 fit한다. 외부 학습 fold에서 유지한 센서 풀 안에서만 재표집하고 검증·Test 데이터는 사용하지 않는다.

반복별 Top-20 선택 빈도 내림차순 → 평균 전체 순위 오름차순 → 원래 입력 열 순서로 최종 20개를 정한다. 재표집에서 품질 기준으로 제외된 센서는 선택 빈도 0, 해당 반복의 순위는 센서 풀 크기 + 1로 기록한다. 선정 뒤 현재 전체 학습 fold에서 선택된 센서의 대치 통계를 fit하고 기존 XGBoost를 학습한다. 현재 S3는 수치형 원본 센서 Profile만 지원하며 One-Hot 변환 특징을 원본 센서로 취급하지 않는다.

`--rf-stability-repeats 10`으로 활성화한다. 기본값 0은 기존 S0~S2 경로를 유지한다. 선택 CSV에는 `selection_method=rf_stability`, `bootstrap_selection_frequency`, `bootstrap_mean_rank`, `bootstrap_repeats`를 기록하고 `bootstrap_records.csv`에는 재표집별 제거 개수·행 수를 남긴다. RF 중요도 평균은 설명용이며 최종 선정 순위는 빈도·평균 순위를 따른다.

```powershell
python -m src.experiments.step6_feature_compare --train data/splits/integrated/time_train.csv --experiments xgboost_all xgboost_rf_top_20 --rf-stability-repeats 10 --output logs/s3_rf_top20_cv/feature_compare.csv --details-dir logs/s3_rf_top20_cv/details --n-jobs 2
```

CV 완료 후 시간순 검증을 실행한다.

```powershell
python -m src.experiments.feature_time_compare --train data/splits/integrated/time_train.csv --experiments xgboost_all xgboost_rf_top_20 --rf-stability-repeats 10 --output-dir logs/s3_rf_top20_time --n-jobs 2
```

반복 수를 줄이거나 RF leaf·깊이를 함께 변경하면 다른 실험 조건으로 별도 기록한다. RF 학습 횟수가 늘어 S0보다 오래 걸릴 수 있다. 재실행은 새 결과 폴더를 사용한다. 사용자 실행 결과를 확인해 진행 기록과 보고서 30절에 정리했다. 이번 문서 정리에서는 실험을 새로 실행하지 않았으며 최종 Test·후보 확정도 수행하지 않았다.

### 2.3 최종 XGBoost 설정 개선안

현재 기준은 트리 300개, 깊이 3, learning rate 0.05, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1`, `scale_pos_weight=1`, `tree_method=hist`이다.

같은 센서 선택 방법을 사용하되 아래처럼 소수의 대표 설정부터 비교한다.

| 후보 | 기준 대비 변경 | 목적 |
| --- | --- | --- |
| M0 | 변경 없음 | 기존 Top-20 기준 |
| M1 | `max_depth=2` | 모델 복잡도를 줄였을 때 시간 일반화 확인 |
| M2 | `reg_lambda=5` | 정규화 강화 효과 확인 |
| M3 | `max_depth=2`, `reg_lambda=5` | 복잡도 제한과 정규화의 결합 확인 |
| M4 | `colsample_bytree=1.0` | 이미 20개로 축소한 특징을 트리마다 추가로 줄이지 않는 경우 비교 |
| M5 | 학습 fold의 정상/불량 비율로 `scale_pos_weight` 계산 | 불량 가중치의 AP·Recall·오탐 영향을 분리해서 확인 |

깊이·정규화·열 표본 추출·불량 가중치의 의미는 [XGBoost 파라미터 문서](https://xgboost.readthedocs.io/en/stable/parameter.html)를 참고한다. 어느 변경도 성능 개선을 보장하지 않는다. 특히 가중치를 높였다고 오탐 부담이 줄어드는 것은 아니다.

필요하면 이후 `min_child_weight`, learning rate와 트리 수의 조합도 검토하되, 우선 위 소규모 실험 결과를 보고 범위를 정한다.

### M1 실행 준비 — S0 선택기 유지·최종 XGBoost 깊이 2

`--topk-xgb-max-depth 2`를 CV·시간순 비교 CLI에 추가했다. 이 옵션은 `xgboost_rf_top_*`의 **최종 XGBoost만** 변경하며 전체 센서 `xgboost_all`은 M0 깊이 3, RF 선택기는 S0 깊이 제한 없음·최소 leaf 1·트리 300개로 유지한다. seed·threshold 정책·센서 수·CV 조건은 기존 설정을 사용한다. 옵션 생략은 기존 실행과 동일하다.

M1은 S0/M0 대비 최종 모델 설정 개선을 비교하는 실험이다. 전체 센서 M0 대비 AP 손실도 함께 기록하므로 이번 AP 손실에는 센서 축소와 모델 깊이 변경 효과가 함께 포함된다. 순수한 깊이 변경 효과는 기존 S0/M0와 비교한다.

실행 조건은 `feature_run.json`의 `topk_xgb_parameters`에 변경 깊이·정규화와 적용 실험 이름으로 기록한다. 시간순 `quality_filter.csv`에는 실제 최종 분류기 파라미터도 남는다. 이후 M2·M3 실행 옵션도 준비했으며 실제 실험은 사용자가 실행한다. M4·M5, 최종 Test, 센서 목록·threshold 동결은 이번 준비 범위에 포함하지 않는다.

1. 반복 CV:

```powershell
python -m src.experiments.step6_feature_compare --train data/splits/integrated/time_train.csv --experiments xgboost_all xgboost_rf_top_20 --rf-min-samples-leaf 1 --rf-stability-repeats 0 --topk-xgb-max-depth 2 --output logs/m1_s0_top20_cv/feature_compare.csv --details-dir logs/m1_s0_top20_cv/details --n-jobs 2
```

2. CV 완료 후 시간순 검증:

```powershell
python -m src.experiments.feature_time_compare --train data/splits/integrated/time_train.csv --experiments xgboost_all xgboost_rf_top_20 --rf-min-samples-leaf 1 --rf-stability-repeats 0 --topk-xgb-max-depth 2 --output-dir logs/m1_s0_top20_time --n-jobs 2
```

새 결과 폴더를 사용한다. 완료 후 기존 `logs/s0_rf_top20_cv/`·`logs/s0_rf_top20_time/`와 평균·구간별 AP, 진단 FN/FP·양성 비율, 학습 비용을 비교한다. 작은 합성 데이터와 설정 전달 테스트로 준비를 검증하며 실제 SECOM 실험은 사용자가 실행한다.

### M2·M3 실행 준비와 보고서 정리 순서

M1~M3의 사용자 실행 결과를 모두 확인해 [보고서 31절](reports/report.md#31-s0-유지--xgboost-m0m3-설정-개선-비교-2026-10-07)에 함께 정리했다. M2는 M1 설정을 승계하지 않고 M0 깊이 3에서 정규화만 변경했다. M3은 깊이 2와 정규화 5를 함께 적용했다. 모든 후보는 S0 RF 선택기와 전체 센서 M0 기준을 유지했다.

| 실험 | Top-20 최종 XGBoost 깊이 | Top-20 최종 XGBoost 정규화 | 상태 |
| --- | ---: | ---: | --- |
| M0 | 3 | 1 | 기존 S0 결과를 비교 기준으로 보존 |
| M1 | 2 | 1 | CV·시간순 결과 확인·보고서 기록 완료 |
| M2 | 3 | 5 | CV·시간순 결과 확인·보고서 기록 완료 |
| M3 | 2 | 5 | CV·시간순 결과 확인·보고서 기록 완료 |

| 후보 | 반복 CV AP | 시간순 AP | 전체 센서 M0 대비 시간순 AP 하락률 | 평균 AP 판정 |
| --- | ---: | ---: | ---: | --- |
| S0/M0 Top-20 | 0.245881 | 0.117674 | 24.25% | 허용 범위·목표 미달 |
| S0/M1 Top-20 | 0.242822 | 0.121453 | 21.82% | 허용 범위·목표 미달 |
| S0/M2 Top-20 | 0.240328 | 0.117148 | 24.59% | 허용 범위·목표 미달 |
| S0/M3 Top-20 | 0.235509 | 0.128986 | 16.97% | 목표 달성 |

M3의 시간순 평균 AP는 Top-20 M0보다 약 9.61% 높고 세 구간 모두 개선됐다. 그러나 CV AP는 감소했고 2구간 AP 하락률 35.33%는 기존 구간별 유지율 70% 보조 조건에 미달한다. 진단용 OOF 문턱에서 Recall 53.85%·양성 비율 53.64%(TP 21·FP 421·FN 18)로, M0보다 FP 42건과 TP 6건이 함께 줄었다. 현재 Recall 90%·양성 비율 20% 정책은 모두 미충족이다.

**사용자 결정 — 2026-10-07:** M3를 후속 기준 모델 설정으로 채택하고 M0는 비교 후보로 보존한다. S0 선택기·S2 비교 후보 보존도 유지한다. 시간순 평균 AP 하락률 목표 달성을 우선한 결정이며, 2구간 저하와 threshold 정책 미충족까지 해결됐다는 뜻은 아니다.

- 기준 경로: **S0 RF Top-20 → M3 XGBoost**. 최종 분류기의 `max_depth=2`, `reg_lambda=5`, 나머지는 M0 설정을 유지한다.
- M0 보존: `max_depth=3`, `reg_lambda=1`. 기존 구현·실험 결과를 유지하며 같은 평가 조건의 후속 비교에 사용한다.
- 실행 시 M3는 `--topk-xgb-max-depth 2 --topk-xgb-reg-lambda 5`를 명시한다. 기존 코드 기본값과 전체 센서 M0 비교 경로는 변경하지 않는다.
- 고정 센서 목록·최종 threshold·배포 모델 동결은 별도 후속 작업이다. 기존 M0의 OOF·threshold를 M3에 그대로 승계하지 않는다.

**후속 사용자 결정 — 2026-10-07:** 시간 검증·분포 진단·최근 기간 학습·센서 고정 대조를 마친 뒤, 전체 과거 학습 + S0·M3 Top-20을 연구·데모용 기준 경로로 유지하기로 했다. 최근 30일과 센서 고정의 평균 AP 개선 근거가 부족하므로 이번 탐색은 현재 결과로 정리한다. 시간순 Validation의 목표 미달을 명시하고 다음은 4절의 20개 입력 추론 기능과 정합성 검증을 진행한다. 임시 80%·40% 정책과 OOF 후보 `0.036539457738399506`은 보존하되 현장 적용·최종 threshold·고정 센서 목록·배포 모델 동결을 승인한 것은 아니다. 근거는 보고서 33절에 기록했다.

아래 명령은 완료한 실험의 조건 기록이다.

M2 반복 CV 후 시간순 검증:

```powershell
python -m src.experiments.step6_feature_compare --train data/splits/integrated/time_train.csv --experiments xgboost_all xgboost_rf_top_20 --rf-min-samples-leaf 1 --rf-stability-repeats 0 --topk-xgb-reg-lambda 5 --output logs/m2_s0_top20_cv/feature_compare.csv --details-dir logs/m2_s0_top20_cv/details --n-jobs 2
python -m src.experiments.feature_time_compare --train data/splits/integrated/time_train.csv --experiments xgboost_all xgboost_rf_top_20 --rf-min-samples-leaf 1 --rf-stability-repeats 0 --topk-xgb-reg-lambda 5 --output-dir logs/m2_s0_top20_time --n-jobs 2
```

M2 완료 후 M3 반복 CV·시간순 검증:

```powershell
python -m src.experiments.step6_feature_compare --train data/splits/integrated/time_train.csv --experiments xgboost_all xgboost_rf_top_20 --rf-min-samples-leaf 1 --rf-stability-repeats 0 --topk-xgb-max-depth 2 --topk-xgb-reg-lambda 5 --output logs/m3_s0_top20_cv/feature_compare.csv --details-dir logs/m3_s0_top20_cv/details --n-jobs 2
python -m src.experiments.feature_time_compare --train data/splits/integrated/time_train.csv --experiments xgboost_all xgboost_rf_top_20 --rf-min-samples-leaf 1 --rf-stability-repeats 0 --topk-xgb-max-depth 2 --topk-xgb-reg-lambda 5 --output-dir logs/m3_s0_top20_time --n-jobs 2
```

각 명령이 성공한 뒤 다음 명령을 실행한다. 새 결과 폴더를 사용하며 M3까지 완료 후 기존 S0/M0와 M1~M3의 평균·구간별 AP, 진단 FN/FP·양성 비율, 정책 충족 여부, 학습 비용을 비교해 보고서에 기록한다. 작은 합성 데이터 테스트는 실행 준비 검증이며 실제 SECOM 실험 결과가 아니다.

### 2.4 구현 시 필요한 보완과 평가 범위

- 현재 XGBoost 세부 설정 대부분은 [modeling_models.py](src/modeling_models.py)의 생성자 기본값이다. 위 후보가 `config.json`만 바꿔 자동 실행되는 상태는 아니다.
- RF 선택기 옵션과 Top-K 최종 XGBoost의 깊이·정규화 옵션을 구분해 전달·검증·실행 기록에 저장하는 기능은 M1~M3 준비 과정에서 구현했다. M4·M5 등 나머지 설정 전달은 별도 보완한다.
- 현재 가중치 자동 계산은 `xgboost_scale_pos_weight` 등 정해진 실험 이름에만 적용된다. Top-20의 M5는 해당 동작을 명시적으로 연결해야 한다. 이름에 가중치 표현을 붙이는 것만으로 적용되지 않는다.
- 모델·선택기를 변경하면 해당 후보의 OOF를 다시 생성한다. 기존 전체 모델의 OOF와 threshold를 Top-20에 그대로 사용하지 않는다.
- 시간순 외부 구간의 점수를 보며 계속 후보를 바꾸면 그 구간은 개발용 검증 데이터다. 이를 새로운 미사용 평가라고 주장하지 않는다.
- 튜닝 결과의 편향을 줄이려면 각 외부 과거 학습 구간 내부에서만 후보를 비교·선정하고, 다음 시간 구간에는 선정 결과를 고정해 적용한다. 전처리·센서 선택도 내부 fold마다 fit한다.
- 소수 불량 표본 때문에 내부 시간 구간에 두 클래스가 있는지 확인하고, 평가가 불가능하면 실패 사유를 기록한다.
- 최종 Test는 후보·센서·파라미터·threshold 확정 전까지 사용하지 않는다.

**현재 진행 순서:** S0~S3 비교 완료 → S0 기준 선택기 채택·S2 비교 후보 보존 → M1~M3 사용자 실행·보고서 31절 기록 완료 → M3 기준 모델 설정 채택·M0 비교 후보 보존 → 전체 Train 단일 OOF 비교 범위 결정. M4·M5는 별도 결정 전까지 실행하지 않는다.

## 3. Train 내부 OOF 비교 자료

### 3.1 서로 다른 자료를 구분한다

| 자료 | 범위 | 용도 |
| --- | --- | --- |
| 반복 CV 지표 | Time Train의 5-fold × 5회 | 후보의 fold별 AP 평균·편차 비교 |
| 전체 Train 단일 계층 OOF | 1회 5-fold, 각 행에 한 번의 미학습 예측 | 전체 Train의 threshold 후보·FN/FP 비교 |
| 구간 내부 시간순 OOF | 각 과거 학습 구간의 확장형 2구간 | 과거에서 정한 threshold의 다음 시간 구간 전이 확인 |

전체 OOF를 합쳐 계산한 AP와 fold별 AP 평균은 계산 방식이 다르다. 예를 들어 전체 XGBoost의 반복 CV AP 0.2490과 단일 OOF AP 0.1877을 같은 지표의 재실행 불일치로 해석하지 않는다. 반복 평균 OOF는 현 코드에서 분석용이며 단일 모델의 최종 threshold 후보용 자료와 구분한다.

### 3.2 기존 전체 Train 단일 OOF 비교

기존 자료는 Time Train 1,096행, Fail 78행, 정상 1,018행을 사용했다. 저장된 threshold 후보 격자에서 조건에 맞는 행을 비교한 결과다.

| 모델 | 단일 OOF AP | Recall 90% 이상일 때의 Recall / 양성 비율 | 양성 비율 20% 이하일 때 최대 Recall |
| --- | ---: | ---: | ---: |
| 전체 XGBoost 기본 | 0.1877 | 91.03% / 63.96% | 46.15% |
| 전체 XGBoost 가중치 | 0.1798 | 91.03% / 67.97% | 42.31% |
| RF Top-20 → XGBoost | 아직 없음 | 추가 생성 필요 | 추가 생성 필요 |

상세 오류 건수는 다음과 같다. 첫 조건은 Recall 90% 이상 후보 중 양성 비율이 가장 낮은 행, 두 번째는 양성 비율 20% 이하 후보 중 Recall이 가장 높은 행이다. 동률은 양성 비율·Recall·threshold로 구분했다.

| 모델 | 비교 조건 | 실제 양성 비율 | TP | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: |
| 전체 XGBoost 기본 | Recall 90% 이상 | 63.96% | 71 | 630 | 7 |
| 전체 XGBoost 가중치 | Recall 90% 이상 | 67.97% | 71 | 674 | 7 |
| 전체 XGBoost 기본 | 양성 비율 20% 이하 | 18.07% | 36 | 162 | 42 |
| 전체 XGBoost 가중치 | 양성 비율 20% 이하 | 19.07% | 33 | 176 | 45 |

두 조건을 따로 비교한 것이며, Recall 90%와 양성 비율 20%를 동시에 충족했다는 뜻이 아니다. 저장된 분위수 격자 안의 결과이지 모든 가능한 문턱의 정확한 최적값도 아니다. 실제 적용할 threshold는 아래 CSV의 반올림하지 않은 값으로 확인한다.

- 기본: `logs/xgboost_20261006/xgboost/threshold_compare.csv` 및 `oof_predictions.csv`.
- 가중치: `logs/xgboost_20261006/xgboost_scale_pos_weight/threshold_compare.csv` 및 `oof_predictions.csv`.

### M3·M0 단일 OOF 비교 실행 준비

`src/experiments/top20_oof_compare.py`로 두 후보를 한 번의 명령으로 비교한다. M3는 S0 RF Top-20 뒤의 XGBoost 깊이 2·정규화 5, M0는 깊이 3·정규화 1이며 RF 선택기는 동일하다. 기존 OOF 코어를 재사용하고 반복 횟수만 1로 제한해 각 Time Train 샘플에 한 번의 미학습 예측을 생성한다. SECOM 설정에서는 단일 계층 5-fold이며 시간 순서 OOF는 아니다. 외부 Validation·Test는 읽지 않는다.

```powershell
python -m src.experiments.top20_oof_compare --train data/splits/integrated/time_train.csv --output-dir logs/m3_m0_oof_compare --recall-targets 0.70 0.80 0.90 --alarm-limit 0.20 --n-jobs 2
```

- `oof_predictions.csv`: 두 후보의 행별 OOF 확률·원본 행 ID·fold·예측 횟수. 예측 횟수는 모두 1이어야 한다.
- `threshold_compare.csv`: 모든 고유 OOF 확률과 0·1·기본 문턱의 지표·현재 정책 충족 여부.
- `scenario_compare.csv`: Recall 70%·80%·90% 이상에서 최소 양성 비율, 양성 비율 20% 이하에서 최대 Recall과 TP·FP·FN. 최소 Recall 목표와 양성 상한을 각각 비교하며 동시에 통과했다는 뜻은 아니다.
- `oof_summary.csv`: 전체 OOF 예측을 합친 AP·ROC-AUC. 반복 CV의 fold 평균 AP와 구분한다.
- `selected_features.csv`, `quality_filter.csv`: fold별 센서 선택·제거 기록. 두 후보의 fold·센서 목록 일치를 검사한다.
- `oof_run.json`: 설정 스냅샷·실제 RF/최종 모델 파라미터·단일 OOF 조건·시나리오·split 계약·패키지 버전.

시나리오 목표는 실행 옵션으로 조절할 수 있으며 현재 threshold 정책은 변경하지 않는다. 문턱은 비교 후보이고 `is_final_threshold=False`다. 기존 `run_evaluation.py`에 이 결과를 바로 전달하는 후속 연결은 이번 준비 범위가 아니며 실제 평가 전에 M3 파라미터와 출처 검사를 연결해야 한다. 새 결과 폴더를 사용하며 코드 준비 검증은 작은 합성 데이터 테스트로만 수행했다. 실제 실험은 사용자가 실행한다.

### S0·M3 고정 문턱 시간순 Validation 실행 준비

현재 80%·40% 정책에서 선택한 OOF 후보는 `0.036539457738399506`이다. `src/experiments/top20_time_validation.py`는 저장된 OOF 확률로 정책 선택을 다시 확인하고, OOF 실행 기록의 S0·M3 파라미터를 복원하여 전체 Time Train에서만 전처리·선택·분류기를 학습한다. Validation에서는 같은 문턱으로 평가만 하며 OOF를 재학습하거나 최종 Test를 읽지 않는다.

```powershell
python -m src.experiments.top20_time_validation --train data/splits/integrated/time_train.csv --validation data/splits/integrated/time_valid.csv --oof-dir logs/m3_m0_oof_compare --threshold 0.036539457738399506 --output-dir logs/m3_policy80_40_time_validation --n-jobs 2
```

새 폴더에 `time_validation.csv`(지표·양성 비율·정책 충족 여부), `selected_features.csv`(전체 Train에서 선택한 센서), `execution.json`(설정·문턱·실제 모델 파라미터·출처)을 저장한다. 전체 Train의 센서 목록은 OOF fold별 목록과 같을 필요가 없다. 기존 로그를 덮어쓰지 않으며 실행은 사용자가 수행한다. 정책 외 설정이나 Train 행·label·생성 계약이 달라지면 기존 OOF 재사용을 차단한다. 이 검사는 센서 값 자체의 변경까지 감지하는 해시 검사는 아니므로 원본 Train 파일을 변경했다면 OOF부터 재생성해야 한다.

### 시간 검증 저하의 확률 분포 진단 준비

사용자가 공유한 고정 문턱 시간 검증 결과는 Recall 5.88%, 양성 판정 비율 8.09%, AP 0.077252, TP 1·FP 18·FN 16이었다. 이 결과만으로 시간 drift나 센서 축소가 원인이라고 단정하지 않는다. 동일한 모델·문턱의 정상·불량별 확률 분포와 OOF fold/전체 Train 선택 센서의 교집합을 먼저 확인한다.

```powershell
python -m src.experiments.top20_time_validation --train data/splits/integrated/time_train.csv --validation data/splits/integrated/time_valid.csv --oof-dir logs/m3_m0_oof_compare --threshold 0.036539457738399506 --output-dir logs/m3_policy80_40_diagnostics --n-jobs 2 --diagnostics --figure-dir reports/figures/m3_policy80_40_diagnostics
```

기존 실행에는 행별 확률이 없으므로 같은 설정으로 전체 Train 학습과 Validation 예측을 한 번 재실행한다. OOF는 재학습하지 않고 최종 Test는 읽지 않는다. 새 `validation_predictions.csv`를 보존하고, `score_metrics.csv`, `score_distribution.csv`, `sensor_overlap.csv`, `diagnostics.md`와 `reports/figures`의 누적분포 그림을 생성한다. 문턱 재탐색이나 개선 모델 비교는 포함하지 않는다. 이 진단은 OOF의 여러 fold 모델과 전체 Train 모델의 차이도 포함하므로 drift 원인 확정 자료가 아니다.

### 동일 M3 설정의 전체 센서·Top-20 시간 검증 비교 준비

`--compare-all`은 기존 S0·M3 경로와 동일한 분류기 파라미터의 전체 센서 M3 경로를 평가한다. 두 모델은 같은 Train·Validation·품질 필터·중앙값 대치를 사용하며 RF Top-20 선택 여부가 다르다. 기존 전체 센서 M0 실험과 구분하고, 이번에는 센서 축소 영향만 먼저 비교한다.

```powershell
python -m src.experiments.top20_time_validation --train data/splits/integrated/time_train.csv --validation data/splits/integrated/time_valid.csv --oof-dir logs/m3_m0_oof_compare --threshold 0.036539457738399506 --output-dir logs/m3_all_vs_top20_time --n-jobs 2 --compare-all
```

두 모델을 각각 전체 Train에서 학습하며 OOF는 재생성하지 않는다. `sensor_compare.csv`, `sensor_compare.md`, `all_validation_predictions.csv`를 추가 저장하고 `execution.json`에 전체 모델의 실제 파라미터도 남긴다. 전체 센서 수는 Train 품질 필터 후의 개수다. 우선 비교 지표는 문턱과 무관한 AP·ROC-AUC다. 공통 문턱은 Top-20 OOF에서 선택한 진단값으로, 전체 모델의 운영 문턱을 결정한 것은 아니다. 운영 Recall·양성 비율을 공정하게 비교하려면 이후 전체 모델 자체의 Train OOF에서 정책 문턱을 별도로 선택해야 한다. 최종 Test는 사용하지 않고 실제 실행은 사용자가 수행한다.

### 선택된 센서의 Train·Validation 입력 분포 진단 준비

`src/diagnostics/selected_sensor_drift.py`는 기존 시간 검증 실행 기록의 Train·Validation 경로와 `selected_features.csv`를 사용한다. 모델을 다시 학습하거나 센서를 재선택하지 않고, 기존 Step 3의 결측률·표준화 평균 차이·PSI 정의를 재사용한다. 생성 계약·평가 역할·설정·선택 센서 출처를 확인한다. 센서 값 변경을 탐지하는 해시 검사는 아니므로 기존 실행에 사용한 split 파일을 유지해야 한다.

```powershell
python -m src.diagnostics.selected_sensor_drift --run-dir logs/m3_all_vs_top20_time --output-dir logs/m3_selected_sensor_drift --figure-dir reports/figures/m3_selected_sensor_drift
```

`selected_sensor_drift.csv`(선택 센서별 변화), `all_sensor_drift.csv`(전체 원본 센서의 EDA 요약), `selected_sensor_monthly.csv`(월·split별 평균·결측률·관측 수), `sensor_diagnostics.md`, `source.json`을 저장한다. 그림은 `reports/figures`에 저장한다. 원본 센서 전체 EDA에는 Train 품질 필터에서 제외된 센서도 포함되므로 전체 모델의 입력 목록과 혼동하지 않는다. 변화 우선순위는 공정 이상이나 통계적 유의성의 판정 기준이 아니다. 원인 후보 확인만 수행하고 threshold 조정·최종 Test 사용은 하지 않는다.

### 우선 확인할 5개 센서의 월별 상세 진단 준비

사용자가 공유한 진단 결과에서 PSI가 컸던 `sensor_539`, `sensor_267`, `sensor_59`, `sensor_40`과 결측률이 24.30%p 감소한 `sensor_562`를 우선 시각적으로 확인한다. 대상 센서는 실행 인자로 지정하며 코드에 SECOM 센서명을 고정하지 않는다.

```powershell
python -m src.diagnostics.selected_sensor_drift --run-dir logs/m3_all_vs_top20_time --output-dir logs/m3_sensor_detail --figure-dir reports/figures/m3_sensor_detail --detail-features sensor_539 sensor_267 sensor_59 sensor_40 sensor_562
```

기존 기본 진단에 더해 `sensor_detail.md`, `detail_value_summary.csv`, `detail_monthly.csv`, `reports/figures/m3_sensor_detail/sensor_detail.png`를 생성한다. 원본 관측값 누적분포·월별 평균·월별 결측률을 비교하고 월별 관측 수와 불량 비율도 남긴다. 같은 달이라도 Train/Validation을 별도로 표시한다. 결측을 대치하거나 센서를 제거하지 않으며 모델 학습·threshold 조정·최종 Test 사용은 없다.

### Train 내부 최근 기간 학습 비교 준비

시간 변화에 대응하는 후보로 전체 과거 구간과 최근 30일·60일 학습을 먼저 Train 내부에서 비교한다. M3 설정(깊이 2·정규화 5·트리 300개)은 유지하고 전체 센서와 S0 RF Top-K 경로를 함께 평가한다. 외부 Validation 결과에 맞춰 기간을 고르지 않는다.

```powershell
python -m src.experiments.recent_window_compare --train data/splits/integrated/time_train.csv --output-dir logs/m3_recent_window_compare --windows 0 30 60 --n-splits 3 --n-jobs 2
```

0일은 전체 과거이며 양수는 각 학습 구간 마지막 시각에서 해당 일수 이내의 행이다. 같은 timestamp는 함께 유지한다. 3개 동일 시간 평가 구간 × 3개 학습 기간 × 2개 특징 경로로 최대 18회 학습한다. 내부 학습 구간에 두 클래스가 없으면 미평가 사유를 기록한다. 최근 기간에서 센서 품질 필터·대치·RF 선택도 다시 학습하며 전체 Train에서 선택한 센서를 고정하여 사용하지 않는다.

`fold_results.csv`, `summary.csv`, `predictions.csv`, `selected_features.csv`, `quality_filter.csv`, `execution.json`, `comparison.md`를 새 폴더에 저장한다. AP·ROC-AUC와 구간별 결과·학습 표본 수를 우선 비교한다. 최근 기간은 drift 대응과 학습 표본 감소를 동시에 일으키므로 개선의 원인을 단정하지 않는다. 기존 전체 Train OOF threshold는 내부 평가 행을 포함해 선택했으므로 사용하지 않고 config의 기본 문턱 0.50을 보조 지표에만 적용한다. 기간 후보가 같아진 fold(`window_truncated=False`)나 미평가 fold는 별도 개선 증거로 해석하지 않는다. 외부 Validation·최종 Test는 사용하지 않으며 실제 실행과 채택은 사용자가 결정한다.

### 최근 기간 실험의 구간별 로그 분석 준비

평균만으로 원인을 추정하지 않고 같은 모델·시간 fold의 전체 과거 결과와 최근 기간 결과를 짝지어 AP 증감과 제외된 학습 표본·불량 수를 확인한다. `--analyze-only`는 기존 결과만 읽고 재학습하지 않는다.

```powershell
python -m src.experiments.recent_window_compare --analyze-only --run-dir logs/m3_recent_window_compare --output-dir logs/m3_recent_window_diagnostics
```

`window_fold_diagnostics.csv`, `window_fold_diagnostics.md`, `source.json`을 생성한다. AP 차이는 최근−전체 과거이며 상대 변화는 기준 AP로 나눈 비율이다. `removed_train_fail`은 학습에서 제외된 불량 수로 평가 FN과 다르다. `same_training_period`는 기록된 학습 기간·행 수의 일치 여부다. 같은 미래 평가 구간의 결과만 대조하고 기준 누락·중복 결과는 오류 처리한다. 미평가·기준 AP=0은 상대 변화를 계산하지 않는다. 학습 불량 표본 감소와 성능 저하가 동시에 나타나도 인과관계를 확정하지 않는다.

### 최근 기간 학습의 센서 선택 변화 분석 준비

`--sensor-overlap`은 기존 `selected_features.csv`에서 전체 과거와 최근 기간의 센서 교집합·추가·제외를 같은 시간 fold끼리 비교한다. 특히 학습 표본을 조금 제외해도 AP가 감소한 구간 1에서 센서 목록도 달라졌는지 확인한다.

```powershell
python -m src.experiments.recent_window_compare --analyze-only --sensor-overlap --run-dir logs/m3_recent_window_compare --output-dir logs/m3_recent_sensor_overlap
```

기존 구간별 분석 파일에 더해 `sensor_overlap.csv`, `sensor_changes.csv`, `sensor_overlap.md`를 생성한다. `s0_m3_topk`는 RF 선택 목록의 변화이며 `m3_all`은 품질 필터 후 특징 목록의 변화다. 센서 수·중복·목록 누락을 검사하고 미평가 구간은 교집합 0으로 대신하지 않는다. 목록 교체와 AP 변화만으로 인과관계나 구현 오류를 확정하지 않는다. 모델 재학습·센서 제거·threshold 조정은 없으며 외부 Validation·최종 Test는 읽지 않는다.

### 센서 고정과 최근 학습의 효과 분리 실험 준비

각 시간 fold의 전체 과거에서 선택한 S0 센서 20개를 고정한 `s0_m3_fixed_sensors`와 각 기간에서 선택기를 다시 학습하는 `s0_m3_topk`를 비교한다. 두 경로 모두 전체 과거(0일)와 최근 30일로 분류기를 학습하며 M3 설정·미래 평가 행은 유지한다. 전체 Train에서 고른 센서를 모든 fold에 사용하는 것은 금지한다.

```powershell
python -m src.experiments.recent_window_compare --train data/splits/integrated/time_train.csv --output-dir logs/m3_fixed_sensor_control --windows 0 30 --n-splits 3 --n-jobs 2 --fixed-sensor-control
```

최대 12회 학습한다. 고정 경로의 품질 제거·선택 목록은 각 fold의 전체 과거 fit 결과이며 median과 M3 분류기는 해당 기간에서 다시 fit한다. 최근 기간에서 센서를 재제거하지 않아 같은 센서 수·순서를 유지한다. 최근 기간의 특정 센서가 전부 결측이면 `keep_empty_features=True`의 0 대치를 사용하며 `empty_input_sensor_count`를 기록한다. `selection_train_samples`는 센서 선택에 사용한 표본 수, `train_samples`는 분류기 학습 표본 수다. 고정 경로는 오래된 과거 정보도 사용한 대조군이므로 순수한 최근 데이터 운영 전략으로 해석하지 않는다.

우선 0일 두 경로의 확률 정합성을 확인하고, 최근 30일의 재선택·고정 경로 AP·ROC-AUC를 같은 미래 구간에서 비교한다. AP 개선만으로 센서 교체가 저하의 유일한 원인이라고 확정하지 않는다. 결과 파일 형식은 기존 최근 기간 실험과 같고 외부 Validation·최종 Test·기존 전체 Train OOF 문턱은 사용하지 않는다. 실제 실험은 사용자가 실행한다.

### 3.3 기존 시간순 내부 OOF의 다음 구간 결과

RF Top-20의 시간순 OOF는 이미 존재한다. 하지만 각 과거 학습 부분의 OOF이며 **전체 Train 1,096행에 대한 단일 계층 OOF 자료를 대신하지 않는다.**

아래는 27절에서 과거 내부 OOF의 F1 최대 진단 문턱을 다음 구간에 적용한 결과다. 평가 합계는 824행, Fail 39행이며 문턱은 구간마다 다르다.

| 모델 | 합산 Recall | 합산 Precision | 양성 판정 비율 | TP / FP / FN |
| --- | ---: | ---: | ---: | ---: |
| 전체 XGBoost | 53.85% | 7.89% | 32.28% | 21 / 245 / 18 |
| RF Top-20 → XGBoost | 69.23% | 5.51% | 59.47% | 27 / 463 / 12 |

현재 90%·20% 정책은 두 모델 × 세 구간의 내부 OOF에서 모두 미충족이다. F1 진단 문턱은 정책 통과값이나 최종 threshold가 아니다.

출처: `logs/xgboost_top20_time_20261006/oof_predictions.csv`, `oof_coverage.csv`, `threshold_compare.csv`, `policy_selection.csv`, `summary.csv`.

### 3.4 추가 작성할 OOF 자료와 선행 작업

1. Top-20 후보를 선택할 때마다 같은 Train·split 계약·OOF 방식으로 새 확률을 만든다. 센서 선택과 가중치 계산도 매 OOF 학습 fold 내부에서 수행한다.
2. 학습 행과 OOF 평가 행의 원본 ID가 겹치지 않는지, 단일 계층 OOF에서 각 행이 한 번 예측됐는지 검사한다.
3. 시간순 OOF는 초기 미예측 부분을 제외하고 대상 수·Fail 수·기간을 기록한다. 미예측을 0점이나 정상으로 바꾸지 않는다.
4. 다음 항목을 같은 형식으로 기록한다: 실험 이름·센서 수·선택 방식·모델 설정·OOF 방식·대상 수·Fail 수·AP·threshold·Recall·Precision·F1·TP·FP·FN·TN·양성 비율·정책 충족 여부.
5. Recall 70/80/90% 조건별 양성 비율 최소 후보, 양성 비율 20% 이하의 Recall 최대 후보, F1 최대 진단 후보를 **별도 시나리오**로 정리한다. 이는 비교용 자료이지 `config.json`의 정책을 자동 변경하는 작업이 아니다.
6. 문턱 탐색에 사용한 OOF 성능은 최종 미사용 성능이 아니다. 모델·파라미터 선정에도 같은 자료를 사용했다면 선택 편향의 한계를 함께 적는다.

**현재 CLI 연결의 빈 부분:** [step8_threshold_oof.py](src/experiments/step8_threshold_oof.py)와 [run_evaluation.py](src/evaluation/run_evaluation.py)의 `--experiment` 선택지는 `lightgbm_all`, `xgboost`, `xgboost_scale_pos_weight`로 제한돼 있다. RF Top-20은 공통 Pipeline 생성이 가능하지만 이 실행 경로의 이름 허용·후속 모델 일치 검사를 연결해야 한다. 아직 지원되지 않는 명령을 실행 예시로 제공하지 않는다.

문서·노트북용 산출물은 OOF 확률표, threshold 비교표, 정책 상태표, Recall–양성 비율 그래프, FN/FP 비교 그래프로 준비한다. 소스·모델·선택 방식이 변경되면 새 실행 폴더에서 OOF부터 생성하고, 정책만 바뀔 때만 같은 성공 실행의 OOF를 재사용한다. 별도의 소스 해시 자동화 도입은 이 계획에 포함하지 않는다.

## 4. 선택된 컬럼만으로 추론할 때의 의존성과 대처

### 4.1 현재 Pipeline을 그대로 쓰면 안 되는 이유

현재 처리 흐름은 다음과 같다.

```text
원본 센서 입력 → 학습된 품질 필터 → 수백 개 센서의 중앙값 대치
             → 학습된 RF 선택기 → 20개 특징 → XGBoost
```

선택기는 예측 시 센서를 다시 고르는 것이 아니라 저장된 mask를 적용한다. 하지만 [SensorQualityFilter](src/modeling_preprocessing.py)는 학습에서 유지한 444~448개 컬럼을 요구하며, 기존 대치기와 선택기 역시 그 입력 공간에 fit돼 있다. 입력 CSV에서 20개만 남겨 전달하면 앞단의 누락 컬럼·특징 수 오류가 발생한다. 최종 XGBoost가 20개로 fit됐다는 사실만으로 전체 Pipeline이 20개 입력을 지원하는 것은 아니다.

목표 추론 흐름은 다음과 같다.

```text
고정 센서 20개 입력 → 이름·순서·타입 검증 → 저장된 20개 센서 전처리
                  → 20개로 학습한 분류기 → 저장된 threshold 적용
```

센서 선택용 RF는 최종 학습 과정에서만 사용하고 추론 때 재학습하지 않는다. 입력 데이터의 결측률·상수 여부로 센서 집합을 다시 바꾸지도 않는다.

### 4.2 주요 문제와 대처

| 문제 | 영향 | 대처 방향 |
| --- | --- | --- |
| 수백 개에 fit된 품질 필터·대치기·선택기 | 20개 입력에서 오류 또는 잘못된 변환 | 기존 객체의 내부 배열을 임의로 잘라 쓰지 말고 20개 입력용 변환 구성과 정합성 확인 |
| 센서명과 입력 순서 불일치 | 오류 없이 잘못된 예측이 발생할 수 있음 | 실제 분류기 학습 순서의 이름 목록을 저장하고 이름 기준으로 재정렬 |
| 중요도 순위와 모델 입력 순서 혼동 | 높은 중요도 순서로 재배열하면 기존 모델과 불일치 | `get_support()`가 유지한 전처리 출력 순서를 기준으로 저장. 순위 표는 설명용으로 분리 |
| 컬럼 누락과 개별 값의 결측 혼동 | 없는 센서를 정상적인 NaN 값처럼 처리 | 필수 센서 컬럼 누락은 오류. 존재하는 컬럼의 NaN만 저장된 대치기로 처리 |
| 입력 배치로 median·scaling 재계산 | 입력 묶음에 따라 같은 제품의 예측이 달라짐 | 최종 학습에서 저장한 통계만 사용하고 추론에서 fit 금지 |
| 센서 고장·과도한 결측·무한대·잘못된 타입 | 대치로 문제가 가려지거나 예측이 무의미해짐 | 타입·유한값 검사와 별도 입력 품질 경고/거부 규칙을 설계. 전체 센서 결측을 조용히 정상 처리하지 않음 |
| 학습용 schema가 label·timestamp를 요구 | 라벨 없는 실제 추론 입력이 거부됨 | 학습 schema를 약화하지 않고, 고정 센서 입력을 검증하는 추론용 계약을 분리 |
| 센서 목록과 모델·threshold의 조합 불일치 | 다른 후보의 문턱이나 전처리가 적용됨 | 모델·전처리·순서 있는 센서 목록·라벨 매핑·threshold를 하나의 묶음으로 관리 |
| 추가 센서 컬럼의 묵시적 유입 | 20개 제한 위반 또는 입력 공간 변경 | 센서 20개 입력 계약에서는 추가 센서를 명시적으로 거부하거나 기록 후 제외. 모델에는 고정 목록만 전달 |

SECOM 센서가 익명 이름이라는 점도 제약이다. 동일한 이름이라도 다른 데이터의 측정 단위·센서 의미가 같다는 보장은 없다. 다른 공장에 재사용할 때는 해당 Profile·센서 매핑과 새 검증이 필요하며 SECOM 모델을 그대로 적용한다고 가정하지 않는다.

### 4.3 학습에서 추론으로 연결하는 방법

1. 평가 중에는 매 fold의 학습 데이터에서 센서를 선택한다. 최종 전체 Train의 목록을 과거 OOF·CV에 소급 적용하지 않는다.
2. 선택 방법·모델 설정·학습 범위를 확정한 후에만 최종 학습에서 하나의 센서 목록을 결정한다. 현재 `candidate_features.csv`의 `is_final=False` 목록은 확정 목록이 아니다.
3. 실제 선택 순서의 20개 입력용 대치기와 분류기를 포함한 추론 구성을 준비한다. 선택된 원본 센서가 수치형이고 median 대치가 센서별로 독립적인 현재 SECOM 조건을 이용할 수 있다.
4. 같은 학습 데이터의 해당 20개로 새 대치기를 fit하는 방식은 기존 median과의 일치 여부를 확인한다. 범주형 인코딩·PCA·센서 간 파생 연산에는 이 독립성 가정을 자동 적용하지 않는다.
5. 기존 전체 입력 Pipeline과 축소 추론 구성에 같은 행을 넣어 **양성 확률·판정·라벨 매핑이 일치하는지** 검증한다. 결측값 포함 입력에서도 비교한다.
6. 변환·분류기 재학습으로 확률이 달라지면 단순한 포장 변경이 아니다. 변경된 학습 절차로 OOF·threshold·시간 검증을 다시 수행한다. 확률이 동일한 구조 변환일 때만 기존 threshold의 재사용 근거가 된다.

전체 입력 경로와 20개 입력 경로가 일치하는지 보는 것은 구현 검증이다. 이미 개발에 사용한 행으로 그 일치를 확인했다고 새로운 일반화 성능이 검증되는 것은 아니다.

### 4.4 라이브러리·코드·설정 의존성

- RF 선택기를 추론 묶음에서 제외하면 선택 모델 객체는 불필요해질 수 있지만 라이브러리 의존성이 자동 제거되는 것은 아니다. 대치기·Pipeline은 scikit-learn을 계속 사용한다.
- 현재 [modeling_models.py](src/modeling_models.py)는 LightGBM도 모듈 상단에서 import한다. 따라서 XGBoost만 예측하더라도 현재 모듈 구조에서는 LightGBM 설치가 필요하다. 추론 의존성을 줄이려면 별도의 경량 모듈이나 지연 import를 검토해야 한다.
- 사용자 정의 `XGBoostClassifierAdapter`를 저장하면 복원 환경에서도 해당 클래스의 안정적인 모듈 경로가 필요하다. `src.modeling_models`와 직접 실행 시의 `modeling_models` 경로 차이도 저장·복원 시험에서 점검한다.
- Python·numpy·pandas·scikit-learn·XGBoost·joblib의 실제 학습 버전을 기록한다. `requirements.txt`의 넓은 범위가 저장된 모델의 모든 버전 간 호환성을 보장하지 않는다.
- 모델에 저장한 센서 목록·전처리·라벨·threshold가 추론 기준이다. 이후 수정된 전역 `config.json`이나 `secom.json`이 배포 모델의 계약을 묵시적으로 바꾸지 않도록 한다. 기존 학습용 Profile은 보존한다.
- 입력 경로와 출력 경로는 실행 인자로 받으며 특정 사용자 PC의 절대 경로를 추론에 고정하지 않는다.
- 이 범용 추론 코어는 Agent·외부 LLM API 없이 동작해야 한다.
- joblib/pickle 모델은 신뢰한 파일만 로드한다. 외부 파일의 로딩은 코드 실행 위험이 있으며, 환경 버전 일치와 사용자 정의 코드 제공이 필요하다. [모델 저장 공식 문서](https://scikit-learn.org/stable/model_persistence.html)

### 4.5 나중에 구현할 때의 검증 목록

- [ ] 원본 센서 20개만으로 label·timestamp 없이 예측한다.
- [ ] 컬럼 순서를 바꿔도 이름으로 맞춰 동일한 확률을 반환한다.
- [ ] 필수 컬럼 누락·중복·추가 센서·잘못된 타입·무한대 처리 규칙을 검증한다.
- [ ] 결측값에 최종 학습 median이 적용되고 입력 배치에 따라 바뀌지 않는다.
- [ ] 실제 분류기 특징 수와 원본 센서 수가 20개 이하인지 확인한다.
- [ ] 기존 전체 입력 경로와 확률·판정이 일치한다.
- [ ] 별도 프로세스에서 저장·복원한 모델로 같은 결과를 얻는다.
- [ ] 센서·모델·threshold·라벨의 다른 조합을 잘못 연결하지 않는다.

### 4.6 추론 코어 구현 (2026-10-07)

`src/sensor_inference.py`에 학습된 수치형 Pipeline을 축소하는 `build_sensor_inference`와 예측 전용 `SensorInference`를 구현했다. 센서 수 상한·양성 label·threshold는 호출자가 명시하며 SECOM 이름이나 20개 제한을 코어에 고정하지 않는다.

- 선택기 입력 순서의 센서명과 센서별 학습 median·선택적 StandardScaler 통계를 가져온다. 기존 sklearn 객체의 내부 배열은 수정하지 않고, 분류기는 독립 복사한다.
- 추론에서 센서 선택이나 통계를 재학습하지 않는다. 컬럼은 이름으로 정렬하고 label·timestamp 없이 예측한다.
- 필수 컬럼 누락·추가·중복·잘못된 타입·무한대·모든 필수 센서가 결측인 행은 오류로 거부한다. 일부 센서의 NaN은 저장된 학습 median으로 대치한다.
- 현재 지원 범위는 수치형 median 대치, 선택적 StandardScaler, SelectFromModel 선택 경로다. PCA·범주형 인코딩·StableRFSelector 등은 별도 변환 설계 없이 지원하지 않는다.
- 합성 데이터 테스트 5개로 원본 Pipeline과의 확률·판정·label 정합성, 열 순서·배치 독립성, 입력 오류, fit 금지, 메모리 직렬화 복원을 검증했다. 실제 XGBoost adapter도 작은 합성 자료로 확인했다.

실제 SECOM의 최종 센서 목록·모델·threshold를 확정하거나 학습하지 않았다. 실제 학습 모델의 정합성 확인과 별도 프로세스 저장·복원은 후속 작업이다. 다음 단계에서도 최종 Test는 사용하지 않는다.

### 4.7 실제 Train의 추론 정합성 확인 명령어

`src/verification/check_sensor_inference.py`는 기존 단일 OOF 설정과 현재 정책 문턱을 확인한 뒤 전체 Train으로 S0·M3 모델을 한 번 학습한다. 같은 모델의 전체 입력 경로와 축소 입력 경로를 비교하며, 선택기 없는 전체 센서 모델과의 성능 비교가 아니다. 실행은 사용자가 수행한다.

```powershell
python -m src.verification.check_sensor_inference --train data/splits/integrated/time_train.csv --oof-dir logs/m3_m0_oof_compare --threshold 0.036539457738399506 --output-dir logs/m3_sensor_inference_check --n-jobs 2
```

- 정상 입력·컬럼 역순·일부 NaN·단일 행·배치 독립성·전체 센서 결측 거부를 확인한다.
- 확률은 절대 오차 `1e-12` 이내, 판정과 label은 완전히 같아야 통과한다.
- 원본 Train의 모든 선택 센서가 NaN인 행은 비교에서 제외하고 거부 개수를 기록한다. 나머지 비교가 통과해도 전체 결측 행까지 원본과 동일하다는 뜻은 아니다.
- 결과는 `inference_check.json`, `inference_check.md`에 저장한다. 불일치 시 보고서를 남기고 오류로 종료하며 기존 결과 폴더는 덮어쓰지 않는다.
- Validation·Test·OOF 재학습·최종 모델 저장은 수행하지 않는다. Train 점수를 성능으로 보고하지 않으며 센서 목록은 여전히 후보 목록이다.
- 실행 검사기의 합성 테스트 3개와 추론 코어 5개, 총 8개 테스트가 통과했다. 실제 SECOM 실행 결과는 아직 확인 전이다.

## 5. 최종 Test 평가 — 기록만, 지금 실행하지 않음

최종 평가 전 점검과 후속 사용자 동결 승인은 [최종 평가 전 점검 문서](FINAL_EVALUATION_REVIEW.md)에 기록했다. 현재 저장 후보·센서 20개·학습 통계·문턱을 연구·데모 평가 대상으로 동결한다. Test 내용은 아직 읽지 않았고 과거 완전한 미사용 여부는 보증하지 못한다. 결과는 기존 데이터의 후속 시간 구간 평가로 표현하며 다음은 저장 후보를 그대로 평가하는 실행 코드 준비다. 현장 배포 승인은 아니다.

모델·센서·전처리·설정·threshold를 동결한 후 1회 평가한다. 기존 Random 탐색 등에서 Test 행이 학습·선정에 사용됐는지도 먼저 확인하고, 사용 이력이 있으면 엄격한 미사용 평가라는 주장을 하지 않는다. 목표 미달 모델을 평가할 경우 연구용 한계를 명시한다.

- [ ] 최종 평가 조건과 데이터 사용 이력을 점검한다.
- [ ] 동결 후 Recall·AP·Precision·F1·ROC-AUC·TP/FP/FN/TN을 기록한다.
- [ ] Test 결과에 맞춰 모델·센서·threshold를 수정하지 않는다.

## 6. 모델 저장·Model Card·예측 CLI — 후보 저장 구현, 최종 확정은 보류

### 6.1 후보 묶음 저장·별도 프로세스 복원 (2026-10-07)

사용자 후속 요청에 따라 후보 저장·복원 검증과 신규 입력 예측 CLI를 구현하고 후보 Model Card를 작성했다. 최종 모델 확정은 아직 완료하지 않았다. 앞서 추론 정합성만 실행했을 때는 모델을 저장하지 않았으므로 아래 저장 명령은 Train 학습을 한 번 다시 수행한다. 이미 저장한 후보로 예측할 때는 재학습하지 않는다. OOF·Validation·Test는 다시 학습하거나 평가하지 않는다.

프로젝트 루트에서 실행한다. `-m src...` 모듈 실행을 사용해야 저장된 사용자 정의 클래스의 경로가 별도 프로세스에서도 유지된다.

```powershell
python -m src.verification.check_sensor_inference --train data/splits/integrated/time_train.csv --oof-dir logs/m3_m0_oof_compare --threshold 0.036539457738399506 --output-dir logs/m3_sensor_bundle_check --bundle-dir models/candidates/m3_sensor20 --n-jobs 2
```

정합성 통과 후 `candidate.joblib`, `manifest.json`을 저장한다. 같은 학습 모델의 센서 순서·median·선택적 scaling·분류기·label·후보 threshold를 보존하며 현재 config를 읽어서 복원 상태를 변경하지 않는다. manifest에는 런타임 버전과 학습 출처·후보 상태를 기록한다.

다음 명령은 **새 Python 프로세스**에서 복원한 예측과 저장 당시 예측을 비교한다. 학습은 하지 않는다.

```powershell
python -m src.inference.sensor_bundle --bundle-dir models/candidates/m3_sensor20 --trusted-local-bundle
```

- `status: passed`와 확률·판정·label 일치 여부를 확인한다. 버전이 다르면 복원을 거부한다.
- joblib 파일은 임의 코드 실행 위험이 있다. `--trusted-local-bundle`은 직접 생성한 신뢰한 파일에만 사용한다. manifest 검증은 보안 서명이나 안전한 역직렬화를 보장하지 않는다.
- 후보 묶음에는 **선택 센서 검증 입력 최대 33행**이 포함된다. 외부 공유·업로드하지 않으며 `/models/candidates/`는 Git 추적에서 제외한다. label·timestamp는 검증 입력에 포함하지 않는다.
- 합성 데이터에서 별도 프로세스 복원·신뢰 확인·버전/계약 검증·불일치 검출을 확인했다. 관련 테스트 총 11개가 통과했으며 실제 SECOM 후보 저장·복원 실행은 사용자 확인 전이다.

### 6.2 후보 예측 CLI (2026-10-07)

`src/inference/predict_cli.py`는 저장된 계약만 사용하며 재학습·설정 파일 로드·성능 평가를 하지 않는다. 신규 센서 CSV 또는 저장된 검증 행의 데모 복사본으로 실행할 수 있다. 입력 조건과 명령어는 [README](README.md#저장된-후보-모델로-센서-csv-예측)에 기록했다.

사용자 실행 결과로 실제 Train 1,096행에서 전체·축소 추론 확률 차이가 0이었고 판정·label이 일치했다. 후보 묶음의 별도 프로세스 복원에서도 검증 입력 33행의 확률 차이가 0이었다. 이는 구현 정합성 확인이며 시간순 성능 미달의 해결이나 최종 모델 승인이 아니다. CLI의 실제 SECOM 데모 실행은 아직 확인 전이다.

### 6.3 후보 Model Card (2026-10-07)

[후보 Model Card](reports/model_card.md)에 저장된 manifest의 센서 순서·학습 설정·후보 문턱·버전과 기존 실험 성능·사용 제한을 기록했다. 실제 후보 묶음의 바이트와 과거 Validation 학습 객체가 동일하다고 보증하지 않으며, 과거 평가 결과와 이번 추론 정합성 결과를 구분한다.

사용자가 수행한 데모 CSV 예측은 33행 중 양성 22행으로 정상 완료했다. 이 자료는 저장된 Train 유래 검증 입력이므로 신규 성능으로 해석하지 않는다. 카드 작성에서는 모델 로딩·학습·Validation/Test 평가·후보 설정 변경을 수행하지 않았다.

### 6.4 남은 최종 산출물

4번의 의존성 검토를 바탕으로 최종 Pipeline과 센서 순서·라벨·threshold·버전·한계를 묶어 저장한다. 신규 입력 CLI에서 실제 선택 컬럼만 받는지 확인한다.

- [ ] 최종 모델 묶음을 저장한다.
- [ ] Model Card에 학습 범위·센서·threshold·성능·제약을 기록한다.
- [ ] 신규 입력 예측 및 저장·복원 CLI를 검증한다.

## 7. 최종 문서·시각화·재현성 — 기록만, 지금 진행하지 않음

최종 센서 Top-20, 성능 저하 여부, 시간 구간 변동, 미검·오탐을 사실대로 정리한다. 실패한 목표를 성공 문구로 바꾸지 않는다. PLAN 체크박스와 최종 README·보고서·발표 자료 정리는 별도 마무리 단계에서 수행한다.

- [ ] 핵심 센서·PR curve·confusion matrix·오류/drift 자료를 정리한다.
- [ ] README 절차로 데이터 준비부터 모델 복원까지 재현성을 확인한다.
- [ ] GitHub 산출물과 발표 자료를 점검한다.

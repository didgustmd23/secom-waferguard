# 다음 회의 안건 — 공동 작업 정리

## 회의 목표

현재까지의 모델링·데이터 분할 결과를 함께 확인하고, 봉인된 Test set을 사용하기 전에 최종 평가 조건과 담당 작업을 합의한다.

## 권장 회의 진행 방식

- **권장 시간:** 45~60분
- **회의 원칙:** 결과가 낮은 원인을 개인 작업의 문제로 단정하지 않는다. 데이터 분할, 시간 변화, 모델 조건을 함께 확인한다.
- **결정 기록:** 각 안건은 `결정`, `담당`, `완료 조건`, `마감`을 한 줄로 남긴다. 결정하지 못한 내용은 가정으로 구현하지 않는다.
- **Test 원칙:** 어떤 안건에서도 `time_test.csv`, `random_test.csv`의 성능을 확인하지 않는다.

### 회의 전 각자 준비할 것

| 담당 | 준비 자료 | 확인할 내용 |
| --- | --- | --- |
| A | `logs/split_summary.csv`, split 생성 코드, drift 결과 | split 재현성, split 간 중복, timestamp 파싱·경계 처리, drift 상위 센서 |
| B | `logs/model_compare.csv`, `logs/feature_compare.csv`, `logs/time_validation_compare.csv`, `logs/threshold_compare.csv` | 후보 모델 근거, feature 수 trade-off, 시간 성능 하락, OOF FN/FP trade-off |
| 공동 | [PLAN.md](PLAN.md), 이 문서 | Test 사용 전 동결해야 할 조건과 각자 후속 작업 |

## 회의 전 용어 정리

| 용어 | 쉬운 설명 | 이번 프로젝트에서의 의미 |
| --- | --- | --- |
| Feature (특징·센서 변수) | 모델이 판단에 사용하는 입력값 | `sensor_0`부터 `sensor_589`까지의 공정 센서 측정값 590개 |
| Label | 정답으로 사용하는 결과값 | `1`은 Fail(불량), `-1`은 Pass(정상) |
| Train / Validation / Test | 학습용 / 중간 점검용 / 최종 시험용 데이터 | Test는 마지막에 한 번만 사용하고, 현재까지 열지 않음 |
| Random split | 데이터를 무작위로 나누는 방법 | 전체 Fail 비율이 각 구간에 비슷하게 남도록 stratify를 적용 |
| Stratify (계층 분할) | 각 split에 정상·불량 비율을 비슷하게 유지하는 방법 | 불량이 적은 데이터에서 특정 split에 Fail이 너무 적게 들어가는 일을 줄임 |
| Time-based split | 시간 순서대로 과거와 미래를 나누는 방법 | 과거 Train으로 학습하고 이후 시점 Validation/Test로 실제 시간 변화에 견딜 수 있는지 확인 |
| Data leakage (데이터 누수) | 미래·검증 데이터의 정보를 학습에 미리 사용하는 오류 | Validation/Test의 결측 처리 통계, feature 선택 결과, label을 Train 학습에 사용하면 안 됨 |
| Pipeline | 전처리와 모델을 한 묶음으로 실행하는 구조 | 결측 처리, scaling, feature 선택, 모델 학습을 Train에서만 fit하도록 관리 |
| Cross Validation (CV) | 학습 데이터를 여러 번 나누어 반복 평가하는 방법 | Random Train에서 5-fold × 5-repeat로 후보 모델 성능의 평균·편차를 계산 |
| OOF prediction | 각 학습 샘플을 자신을 학습에 쓰지 않은 모델로 예측한 확률 | threshold를 정할 때 학습 데이터에 과적합된 확률 대신 사용 |
| Threshold (임계값) | Fail로 경보를 낼 확률 기준값 | 확률이 threshold 이상이면 Fail로 분류. 현재 `0.000475`는 프로젝트용 후보값 |
| FP / FN | FP는 정상인데 Fail 경보, FN은 Fail인데 정상으로 놓침 | FP는 불필요 재검사 비용, FN은 불량 유출 위험을 뜻함 |
| Recall | 실제 Fail 중 모델이 Fail로 잡아낸 비율 | 높을수록 불량을 덜 놓치지만 FP가 늘 수 있음 |
| Precision | 모델이 Fail이라고 한 것 중 실제 Fail 비율 | 높을수록 불필요 경보가 적음 |
| F1 | Recall과 Precision의 균형 지표 | 현장 비용 정보가 없을 때 threshold 후보를 비교하는 보조 기준 |
| AP (Average Precision) | 여러 threshold 전체에서 불량을 앞쪽에 잘 순위화하는 정도 | 불균형 데이터에서 후보 모델의 핵심 비교 지표 |
| ROC-AUC | 정상보다 Fail에 더 높은 확률을 줄 수 있는지 보는 순위 지표 | AP와 함께 모델의 확률 순위 성능을 보조 확인 |
| Drift (분포 변화) | 시간에 따라 센서값이나 데이터 특성이 달라지는 현상 | Time Validation 성능 저하의 가능한 원인. 곧바로 센서 제거 사유는 아님 |
| Feature selection (특징 선택) | 모든 센서 대신 일부 센서만 고르는 방법 | 센서 수·학습 시간·해석 가능성과 성능의 trade-off를 비교 |
| LightGBM | 여러 결정트리를 결합하는 gradient boosting 모델 | 현재 Random CV와 Time Validation 기준의 잠정 최종 후보 모델 |
| Model freeze (모델 동결) | 최종 평가 전에 모든 선택을 더 이상 바꾸지 않는 것 | 모델, feature, 파라미터, threshold를 고정한 뒤 Test를 한 번 평가 |

## 공유할 현재 진행 상황

- 모델링 담당(B)은 후보 모델 비교, 특징 선택 비교, Time Validation, LightGBM OOF threshold 비교를 완료했다.
- 데이터 담당(A)은 데이터 병합·품질 점검·Random/Time split 결과를 생성했다. `step3_split.py`의 재현성과 시간 경계 처리는 보완이 필요하다.
- Test split은 아직 모델 비교, 특징 선택, threshold 결정에 사용하지 않았다.

## 논의 안건

| 안건 | 현재 확인된 내용 | 회의에서 합의할 내용 |
| --- | --- | --- |
| Split 재현성·누수 검증 | split 결과는 생성됐으나 `step3_split.py`의 입력 경로·재실행 처리·시간 경계 그룹 처리를 보완해야 한다. | seed, stratify, 파일 경로, split 간 행 중복, timestamp 중복 그룹 경계 확인 방법을 확정한다. |
| 데이터 품질·시간 변화 | 590개 feature 중 drift 우선순위는 높음 242개, 중간 102개, 낮음 246개다. | 변화가 큰 센서를 즉시 제거하지 않고, 우선 보고·해석 대상으로 둘지 결정한다. |
| 최종 후보 모델 | Random CV와 Time Validation에서 `lightgbm_all`이 L1 선택 모델보다 높은 AP를 보였다. | `lightgbm_all`을 최종 평가 후보로 잠정 고정할지 결정한다. |
| Threshold 후보 | LightGBM OOF에서 F1이 가장 높은 후보는 `0.000475`다. Recall 0.329, Precision 0.218, FN 49, FP 86이다. | 해당 값을 프로젝트 평가용 후보로 기록하고, 현장 운영 기준과 구분할지 확인한다. |
| Time Validation 재확인 | 기존 Time Validation은 비교 기준인 threshold 0.50으로 수행되어 Fail 검출이 없었다. | `0.000475`를 그대로 적용해 Time Validation FN·FP를 확인하되, 결과로 threshold를 재튜닝하지 않기로 합의한다. |
| 오류 사례 분석 | FN/FP 사례 분석은 아직 시작하지 않았다. | sensor 값, 결측률, timestamp, drift를 어떤 표·그래프로 비교할지 정한다. |
| 보조 이상 탐지 실험 | Isolation Forest와 PCA reconstruction error는 미착수다. | 필수 실험으로 진행할지, 시간 여유가 있을 때의 보조 실험으로 둘지 결정한다. |
| 최종 Test 평가 | Test는 봉인 상태다. | 모델·특성·하이퍼파라미터·threshold·버전을 모두 고정한 뒤 한 번만 평가한다는 원칙을 재확인한다. |
| 보고서·시각화 | 중간 결과는 기록 중이며, 시각화는 남아 있다. | PCA 분산, feature importance, Random/Time 비교, 오류 분석 그래프의 담당과 완료 시점을 정한다. |

## 상세 회의 진행안

### 1. Split이 평가에 사용할 수 있는 상태인지 확인 (10~15분)

**확인할 사실**

- Random split은 70/15/15 비율과 label stratify를 사용한다.
- Time-based split은 timestamp 오름차순으로 과거/중간/최근 구간을 나누고 stratify하지 않는다.
- 모델 입력에는 label, timestamp, ID, group, 명시적 제외 컬럼이 포함되지 않는다.
- Train/Validation/Test 사이에 동일 원본 행이 중복되지 않는다.
- 동일 timestamp를 가진 행이 split 경계를 넘는 경우의 처리 정책이 코드와 로그에 남아 있다.

**A에게 확인할 질문**

1. `step3_split.py`를 같은 환경에서 다시 실행하면 동일한 파일과 행 구성이 생성되는가?
2. split 입력 파일 경로와 출력 파일 경로가 CLI 인자로 명확히 관리되는가?
3. Random split의 각 구간 class 비율이 전체 비율과 비교해 허용 가능한 차이인가?
4. Time split에서 timestamp 파싱 실패 행, 동일 timestamp 경계 행, 비정상 시간 순서 행은 어떻게 기록되는가?
5. `logs/split_summary.csv`만 봐도 각 split의 샘플 수·Pass/Fail 수·시간 범위를 확인할 수 있는가?

**회의 결정**

- split 코드 보완이 끝나기 전까지는 현재 결과를 “중간 실험 결과”로 표시한다.
- split 검증 통과 조건을 정한다. 예: 행 중복 0건, feature 구성 동일, timestamp 파싱 실패 0건 또는 별도 기록, seed 재실행 동일 결과.

### 2. 시간 변화와 데이터 품질을 어떻게 해석할지 결정 (10분)

**현재 근거**

- Time Train/Validation 사이에서 drift 우선순위가 높은 feature는 242개다.
- Fail 비율 변화는 약 0.12%p로 작지만, feature 분포 변화는 크다.
- Random Validation보다 Time Validation에서 성능이 크게 낮아졌다.

**논의할 질문**

1. drift 상위 센서를 바로 제거할 근거가 있는가, 아니면 공정 조건 변화 후보로 먼저 기록할 것인가?
2. sensor별 결측률 변화가 장비·측정 정책의 변화와 관련 있는지 확인할 수 있는가?
3. Time Validation 성능 저하를 “모델 실패”가 아니라 “시간 일반화 위험”으로 보고서에 명시할 것인가?
4. 추가 확보 가능한 lot, 장비, 공정 단계 metadata가 있는가?

**회의 결정**

- 현재 drift 결과는 전역 feature 제거 기준으로 사용하지 않는다.
- drift 상위 센서는 오류 사례 분석과 보고서 시각화의 우선 대상 목록으로 사용한다.

### 3. 후보 모델과 feature 집합을 동결할지 결정 (10분)

**현재 근거**

| 후보 | Random CV AP | Time Validation AP | 판단 |
| --- | ---: | ---: | --- |
| LightGBM 전체 feature | 0.1860 | 0.0874 | 현재 최종 평가 후보 |
| L1 선택 Logistic Regression | 0.1403 | 0.0637 | 경량·해석 후보이나 시간 성능이 낮음 |

**논의할 질문**

1. 이번 프로젝트의 우선순위는 최고 순위 성능인가, 적은 sensor 수와 해석 가능성인가?
2. L1 선택 모델을 최종 후보에서 제외하고 비교 결과로만 남길 것인가?
3. LightGBM의 590개 feature를 그대로 사용해도 보고서와 발표에서 설명 가능한가?
4. 추가 하이퍼파라미터 탐색을 할 근거와 시간이 있는가? 없다면 현재 모델 조건을 동결한다.

**회의 결정**

- 기본안은 `lightgbm_all`을 최종 후보로 잠정 고정한다.
- 새 모델·새 feature 집합을 추가하려면 Test를 열기 전에 Random CV와 Time Validation 계획을 다시 합의한다.

### 4. Threshold를 운영 기준과 프로젝트 기준으로 분리 (10분)

**현재 근거**

- `random_train` OOF에서 `0.000475`는 F1 `0.262`로 비교 후보 중 가장 높았다.
- 같은 지점에서 Recall은 `0.329`, Precision은 `0.218`, FN은 `49`, FP는 `86`이다.
- 낮은 threshold일수록 Recall은 올라가지만 재검사 대상(FP 포함)이 빠르게 증가한다.

**논의할 질문**

1. 프로젝트 보고서에서는 F1 최대 후보 `0.000475`를 재현 가능한 기준으로 기록하는 데 동의하는가?
2. 실제 현장 적용이라면 경보가 발생한 wafer/lot을 재검사할 수 있는가, hold만 가능한가, 자동 폐기까지 연결되는가?
3. 한 배치 또는 하루에 처리 가능한 재검사 건수는 얼마인가?
4. 불량 유출(FN) 1건과 정상 경보(FP) 1건의 비용·위험은 어느 쪽이 큰가?

**회의 결정**

- `0.000475`는 현장 운영값이 아니라 프로젝트 평가용 후보로만 기록한다.
- 동일 threshold를 Time Validation에 적용해 FN·FP를 확인한다.
- Time Validation의 결과를 보고 threshold를 다시 조정하지 않는다. 조정이 필요하면 Time Validation을 튜닝 데이터로 사용했다는 사실을 명시하고 별도 검증 구간을 마련한다.

### 5. 최종 Test 전에 남길 산출물과 역할 확정 (10분)

**공동으로 동결할 항목**

| 동결 항목 | 확정 전 확인할 내용 |
| --- | --- |
| 입력 데이터·split 버전 | 원본 파일, Dataset Profile, split 파일, seed |
| feature 집합 | `lightgbm_all`의 590개 입력 feature 및 metadata 제외 규칙 |
| 전처리·모델 | 결측 처리, LightGBM 파라미터, 라이브러리 버전 |
| threshold | OOF 근거, 적용 목적, Time Validation 확인 결과 |
| 평가 방법 | Fail label, AP·Recall 중심 지표, confusion matrix·PR curve |

**역할 초안**

- **A:** split 코드 검증·보완, drift 시각화, 시간·데이터 품질 해석 근거 정리
- **B:** 고정 threshold의 Time Validation FN/FP 확인, OOF FN/FP 사례 분석, 최종 Pipeline·Model Card·CLI 준비
- **공동:** 동결 여부 승인, Test 1회 평가 실행 시점 결정, 보고서·발표 결론 검토

## 회의록에 바로 채울 결정 표

| 결정 항목 | 결정 내용 | 담당 | 완료 조건 | 마감 |
| --- | --- | --- | --- | --- |
| Split 검증 |  |  |  |  |
| 최종 후보 모델 |  |  |  |  |
| Feature 집합 |  |  |  |  |
| Threshold 적용 방식 |  |  |  |  |
| 보조 이상 탐지 실험 |  |  |  |  |
| Test 평가 실행 조건 |  |  |  |  |
| 보고서·시각화 |  |  |  |  |

## 회의 결정 체크리스트

회의에서 합의한 항목만 체크한다. 체크된 결정은 이후 실험과 보고서의 기준으로 사용한다.

### Split·데이터 기준

- [ ] `step3_split.py`의 재실행 결과가 seed·입력 파일 기준으로 동일함을 확인했다.
- [ ] Random/Time split 간 원본 행 중복이 없고, 모델 입력 feature 구성이 동일함을 확인했다.
- [ ] Time split의 timestamp 파싱 실패·동일 timestamp 경계 처리 정책을 기록했다.
- [ ] drift 상위 feature는 전역 제거하지 않고, 오류 분석·시각화 우선 대상으로 사용하기로 했다.

### 모델·feature 기준

- [ ] `lightgbm_all`을 최종 Test 평가의 잠정 후보 모델로 고정한다.
- [ ] L1 선택 모델은 경량·해석 비교 결과로 남기고, 최종 Test 후보에서는 제외한다.
- [ ] LightGBM 전체 590개 feature와 현재 전처리·하이퍼파라미터 조건을 Test 전까지 변경하지 않는다.

### Threshold·평가 기준

- [ ] `0.000475`를 OOF F1 최대 기반의 **프로젝트 평가용 후보 threshold**로 기록한다.
- [ ] `0.000475`는 현장 자동 폐기 또는 Lot hold 기준이 아니며, 실제 현장 적용에는 FP·FN 비용과 재검사 용량이 추가로 필요함을 기록한다.
- [ ] 고정된 `0.000475`를 Time Validation에 그대로 적용해 FN·FP를 확인한다.
- [ ] Time Validation 결과만으로 threshold를 다시 조정하지 않는다.
- [ ] Test split은 모델·feature·전처리·하이퍼파라미터·threshold를 동결한 뒤 한 번만 사용한다.

### 남은 실험·산출물 기준

- [ ] Isolation Forest 또는 PCA reconstruction error를 필수 실험으로 진행할지, 선택 실험으로 둘지 결정한다.
- [ ] FN/FP 사례 분석의 비교 항목(sensor, 결측률, timestamp, drift)을 확정한다.
- [ ] PCA 분산, feature importance, Random/Time 비교, 오류 분석 시각화의 담당·완료 시점을 확정한다.
- [ ] 최종 Test 평가 전에 Pipeline 저장, Model Card, 예측 CLI의 완료 조건을 확정한다.

## 기존 담당 기준 후속 작업

담당은 [PLAN.md](PLAN.md)에 이미 정해져 있다. 아래는 담당을 새로 배정하는 내용이 아니라, 기존 담당 범위 안에서 회의 후 이어갈 작업이다.

### A — 데이터·분석

- `step3_split.py`의 입력 경로, 재현성, timestamp 경계 그룹 처리 보완
- split 간 중복·클래스 비율·시간 범위 검증 결과 기록
- drift 상위 센서와 시간 변화 시각화
- FN/FP 사례 분석용 데이터 확인

### B — 모델링·평가

- `0.000475`를 적용한 Time Validation의 FN·FP 확인
- OOF 기준 FN/FP 사례 분석 구현
- 최종 후보 Pipeline·모델 저장·Model Card·예측 CLI 준비
- 최종 Test 단일 평가 실행 준비

### 공동

- 후보 모델·특성 집합·하이퍼파라미터·threshold 동결
- 오류 분석과 drift 해석의 결론 합의
- 최종 보고서, README, 발표 자료의 담당·마감 확정

## 팀원에게 전달할 메시지 예시

> 현재 모델링은 후보 비교, 특징 선택, Time Validation, OOF threshold 비교까지 진행됐습니다. 내일은 Test를 열기 전에 split 검증, 최종 후보, threshold 검증 방식, 보고서 산출물을 함께 확정하면 좋겠습니다.
>
> split 코드 수정 후에는 재실행해도 동일한 Train/Validation/Test가 생성되는지와 timestamp 경계에서 같은 시점 데이터가 섞이지 않는지 확인 부탁드립니다. 모델링 쪽에서는 고정된 LightGBM·threshold 기준으로 FN/FP 분석과 최종 평가 준비를 이어가겠습니다.

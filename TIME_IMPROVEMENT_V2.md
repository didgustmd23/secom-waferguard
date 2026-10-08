# V2 전체 센서 시간순 개선 실험

## 1. 목적과 현재 상태

V1의 후속 시간 구간 Recall 11.11% 이후, 전체 센서에서 모델 복잡도와 불량 가중치의 영향을 다시 비교합니다. V1 묶음은 `models/retired/v1_m3_sensor20`에 보존하고 새 실험에서 사용하지 않습니다.

현재 상태는 **실험 준비 완료, 실제 데이터 실행 전**입니다. 이번 비교에서 후보 채택이나 최종 문턱을 확정하지 않습니다.

## 2. 여섯 후보

`r = 현재 학습 구간의 정상 수 / 불량 수`입니다. 외부 평가 구간의 label로 가중치를 계산하지 않습니다.

| 후보 | max_depth | reg_lambda | scale_pos_weight |
| --- | ---: | ---: | ---: |
| v2_m0_none | 3 | 1 | 1 |
| v2_m0_sqrt_ratio | 3 | 1 | √r |
| v2_m0_ratio | 3 | 1 | r |
| v2_m3_none | 2 | 5 | 1 |
| v2_m3_sqrt_ratio | 2 | 5 | √r |
| v2_m3_ratio | 2 | 5 | r |

공통 설정은 트리 300개, learning_rate 0.05, subsample·colsample_bytree 0.8입니다. 설정은 `configs/experiments/time_weight_v2.json`에서 관리하며 실제 학습 가중치는 `fit_weights.csv`에 기록합니다.

전처리는 `SensorQualityFilter → SimpleImputer(median) → XGBoost`입니다. 결측 비율 50% 초과·상수 센서를 각 학습 구간에서 제거하고 남은 전체 센서를 사용합니다. RF 선택기·Top-20 제한·PCA·SMOTE는 적용하지 않습니다. 제거 결과와 센서 수는 `quality_filter.csv`에 기록합니다.

## 3. 비교 절차

1. 기존 `time_train.csv`만 읽고 6개 후보에 동일한 외부 시간순 3개 구간을 적용합니다.
2. 각 외부 학습 구간 내부에서 시간순 2개 구간의 OOF 확률을 생성합니다. 초기 미예측 구간은 제외하고 제외량을 기록합니다.
3. 가중치는 외부 학습과 내부 OOF 학습마다 해당 label로 다시 계산합니다.
4. 문턱은 내부 OOF에서만 선택한 뒤 외부 미래 구간에 적용합니다. 현재 `config.json` 정책(Recall 80%·양성 판정 비율 40%)도 별도로 확인합니다.

총 54회 fit입니다: `6개 후보 × 3개 외부 구간 × (외부 학습 1회 + 내부 학습 2회)`. `--n-jobs`는 XGBoost 내부 병렬 수이며, 후보 여섯 개를 동시에 실행한다는 의미는 아닙니다.

기존 외부 Validation·Test는 읽지 않으며 저장 모델을 덮어쓰지 않습니다. 이미 관찰한 Test를 이후 재사용하면 개발 검증으로 구분해야 하며 새로운 미사용 평가로 표현하지 않습니다.

## 4. 실행과 결과 확인

프로젝트 루트에서 실행합니다. 출력 폴더가 이미 비어 있지 않으면 새 이름을 지정하세요.

```powershell
python -m src.time_weight_compare --train data/splits/integrated/time_train.csv --output-dir logs/v2_time_weight_compare --n-jobs 2
```

- `comparison.md`, `summary.csv`: 구간별 AP 평균·편차와 탐지/양성 판정 부담 요약.
- `fold_results.csv`: 특정 구간에서만 좋아지거나 무너지는지 확인.
- `policy_selection.csv`: 정책 문턱 선택 가능 여부와 외부 시간 구간의 정책 충족 여부.
- `fit_weights.csv`, `quality_filter.csv`: 실제 가중치와 전처리 제거 내역.
- `oof_coverage.csv`, `oof_predictions.csv`: OOF 적용 범위와 초기 제외량.
- `execution.json`: 실제 설정·Train 경로·split 식별자·후보 설정 기록.

**주의:** `summary.csv`의 `temporal_oof` Recall은 F1 최대화 문턱을 적용한 진단값입니다. 80%·40% 정책을 만족했다는 뜻이 아니므로 `policy_selection.csv`와 분리해서 해석합니다. 정책 후보가 없더라도 진단 결과는 남기며 목표값을 자동으로 완화하지 않습니다.

평균 AP만으로 채택하지 않고 구간별 AP, 미검(FN), 오탐(FP), 양성 판정 비율과 정책 결과를 함께 비교합니다. 전체 센서 개선을 확인한 뒤 별도 실험에서 Top-20 성능 손실을 다시 검증합니다.

## 5. 전체 센서 중요도 확인

기존 V2 로그에는 학습 객체가 저장돼 있지 않습니다. 아래 명령은 지정한 후보 하나를 Time Train 전체에서 새로 학습하고 중요도를 추출합니다. 기본 후보는 `v2_m0_ratio`이며 최종 채택을 뜻하지 않습니다. 외부 Validation·Test는 읽지 않고 모델도 저장하지 않습니다.

```powershell
python -m src.sensor_importance --train data/splits/integrated/time_train.csv --model v2_m0_ratio --output-dir logs/v2_sensor_importance --figure-dir reports/figures/v2_sensor_importance --n-jobs 2
```

- `feature_importance.csv`: 원본 센서 전체의 이름·학습/제거 상태·Train 결측률·정규화 gain 중요도·순위.
- `reports/figures/v2_sensor_importance/feature_importance_all.png`: 제거 센서까지 포함한 전체 목록을 여러 패널로 표시.
- `feature_importance_top20.png`: 학습 센서의 중요도 상위 20개 그림. 실제 Top-20 모델 학습과 구분.
- `importance.md`, `execution.json`: 그림 연결 보고서와 실제 학습 설정·출처·제거 내역.

`reports/analysis.ipynb`의 **8. V2 전체 센서 Feature Importance**에서 목록과 그림을 읽을 수 있습니다. 로그가 바뀌면 해당 셀을 다시 실행하세요. 기본 폴더와 다른 경로로 실행했다면 셀의 결과 경로도 맞춰야 합니다.

gain 중요도는 평균 분할 손실 개선량의 정규화 값이며 인과적 영향이나 불량 검출률이 아닙니다. 제거 센서는 중요도가 빈 값이고, 학습 센서의 0은 해당 학습에서 분할에 사용되지 않았다는 의미입니다. 전체 Train 진단 순위를 보고 센서를 다시 고르면 이후 시간순 검증에서 각 학습 fold 내부 선택을 다시 수행해야 합니다.

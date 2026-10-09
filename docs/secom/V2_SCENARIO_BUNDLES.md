# V2 90%·80% 시나리오 모델 보존

## 목적과 범위

미검 우선 90% 목표 시나리오를 먼저 저장·복원 검증하고, 그 작업을 마친 뒤 80% 목표를 별도 모델 묶음으로 보존한다. 두 목표는 **과거 OOF Recall 목표**이며 이후 데이터의 검출률 보장이 아니다.

센서 20개·학습 median·XGBoost 분류기는 공유하며 문턱만 다르다. 처음 90% 저장에는 기존 검증의 과거 824행에서 분류기 재학습이 필요하다. 이전 실험에는 학습 객체를 저장하지 않았기 때문이다. 센서는 다시 선택하지 않고 후보 목록을 사용한다. 80%는 신뢰한 90% 묶음을 독립 복사하므로 새 학습·OOF 탐색 없이 문턱만 변경한다.

### 1. 90% 시나리오 저장

```powershell
python -m src.sensor_ml.inference.save_v2_scenario `
  --train data/splits/integrated/time_train.csv `
  --oof-dir logs/v2_fixed_sensor20_oof `
  --target-recall 0.9 `
  --bundle-dir models/candidates/v2_fixed20_recall90 `
  --n-jobs 2
```

기존 마지막 개발 구간의 확률과 새 학습 확률이 일치해야 저장한다. 문턱은 저장된 CSV의 정확한 값으로 읽으며 출력의 반올림값을 입력하지 않는다. 원래 검증 범위인 824행만 학습하고 마지막 272행을 학습에 추가하지 않는다. 새 독립 Test나 추가 성능 개선을 수행하는 명령이 아니다.

### 2. 별도 프로세스로 90% 복원 확인

```powershell
python -m src.sensor_ml.inference.sensor_bundle `
  --bundle-dir models/candidates/v2_fixed20_recall90 `
  --trusted-local-bundle
```

`status: passed`를 확인한다. joblib은 임의 코드 실행 위험이 있으므로 직접 생성한 신뢰한 파일에만 이 옵션을 사용한다. 저장·복원 정합성과 탐지 성능은 별도다.

### 3. 90% 확인 후 80% 분기 저장

```powershell
python -m src.sensor_ml.inference.save_v2_scenario `
  --oof-dir logs/v2_fixed_sensor20_oof `
  --target-recall 0.8 `
  --base-bundle-dir models/candidates/v2_fixed20_recall90 `
  --trusted-local-bundle `
  --bundle-dir models/candidates/v2_fixed20_recall80
```

80%는 90%를 덮어쓰지 않는다. 같은 OOF 실행 기록과 센서 순서를 확인하고 분류기·통계는 유지한 채 문턱만 바꾼다. 추가 실험은 먼저 이 두 저장 시나리오를 기준으로 비교하고, 모델·전처리 변경이 필요하면 별도 버전으로 진행한다.

### 4. 80% 복원 확인

```powershell
python -m src.sensor_ml.inference.sensor_bundle `
  --bundle-dir models/candidates/v2_fixed20_recall80 `
  --trusted-local-bundle
```

## 저장 구조와 상태

각 묶음은 `candidate.joblib`, `manifest.json`, `verification.json`, `experiment/`로 구성한다. `experiment/`에는 원본 OOF CSV·실행 기록 사본을 보존하고, manifest에는 목표 Recall·정확한 문턱·OOF 출처·당시 개발 결과·이번 실행 재학습 여부를 기록한다. 원본 로그는 수정하지 않는다. 비어 있지 않은 모델 폴더는 거부한다.

상태는 **연구용 위험 선별 후보**이며 `is_final_model=false`와 배포 미승인을 유지한다. 검증 입력은 학습 데이터 유래의 로컬 센서 값이며 묶음에 포함된다. 공개 저장소 업로드 여부는 별도 검토한다.

- [x] 90% 모델 저장 완료
- [x] 90% 별도 프로세스 복원 확인
- [x] 80% 모델 분기 저장 완료 (재학습 없음)
- [x] 80% 별도 프로세스 복원 확인

사용자가 두 시나리오의 저장·별도 프로세스 복원 확인을 완료했다. 각 묶음의 검증 입력 33건에서 확률 차이는 0.0이며 확률·판정·라벨이 모두 일치했다. Test는 사용하지 않았다.

## 동일 입력 추론 데모 결과

| OOF 목표 | 정확한 문턱 | 데모 입력 | 위험 대상 선별 |
| --- | ---: | ---: | ---: |
| 90% | 0.0029468848183751106 | 33건 | 33건 |
| 80% | 0.0339033380150795 | 33건 | 30건 |

90%는 낮은 문턱을 사용하므로 같은 입력에서 80%보다 3건을 더 선별했다. 이 입력은 저장된 Train 유래 검증 행의 복사본이며 label이 없는 추론 데모다. 이 3건이 정상인지 불량인지, 데모 Recall·오탐률·신규 데이터 성능은 판단하지 않는다. 두 CSV 간 확률 동일성과 선별 대상 포함 관계는 이 실행 출력만으로 추가 검증했다고 주장하지 않는다.

- 공통 입력: `data/processed/v2_sensor20_demo.csv`
- 90% 예측: `logs/secom/demos/v2/v2_recall90_demo_predictions.csv`
- 80% 예측: `logs/secom/demos/v2/v2_recall80_demo_predictions.csv`

두 묶음은 센서 목록·전처리 통계·분류기를 공유하고 문턱만 다르다. 90% 저장은 기존 범위로 재학습했고 80%는 재학습 없이 분기 저장했다. 연구용 후보 상태와 배포 미승인은 유지한다.

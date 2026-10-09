# Dataset Profile 설정 안내

공통 실험 설정과 데이터셋별 입력 계약을 분리하는 방법입니다. 명령과 설정 경로는 프로젝트 루트 기준입니다.

## 설정 구조

범용 코어와 데이터셋별 가정을 분리합니다.

- `config.json`: 난수 시드, 후보 모델 비교용 `default_threshold`, CV처럼 데이터셋과 독립적인 실험 조건
- `configs/datasets/secom.json`: SECOM의 원본 source, 입력 경로, label 값, 컬럼 역할, timestamp 형식, feature 타입·선택 규칙, 데이터 품질 기준

센서 축소 허용 기준은 `feature_selection.reduction_policy`에서 설정합니다. `max_sensor_count: 20`, `target_ap_loss_ratio: 0.2`, `max_ap_loss_ratio: 0.3`은 각각 센서 수 상한·AP 상대 하락률 목표·최대 허용입니다. `comparison_pairs`는 같은 분류기 설정의 `[전체 실험 이름, 축소 실험 이름]` 쌍을 지정합니다. 정책을 생략하면 다른 데이터셋에 이 기준을 강제하지 않습니다. Recall·양성 판정 비율의 `threshold_policy`와는 독립적입니다.

Step 6는 판정표를 출력하고 `--details-dir` 지정 시 `reduction_assessment.csv`로 저장합니다. 시간순 특징 비교는 출력 폴더에 같은 파일을 저장합니다. `목표 달성`, `허용 범위·목표 미달`, `허용 하락률 초과`, `판정 불가`를 구분하고 원본 센서 수 충족 여부를 별도로 기록합니다. 기준 모델 누락·fold 불일치·정의 불가 AP는 성공으로 처리하지 않으며, 이 표만으로 최종 모델을 자동 선정하지 않습니다.

`src/modeling_config.py`는 두 설정을 함께 읽어 `ModelingConfig`와 `DatasetSpec`으로 검증합니다. `src/dataset_schema.py`는 Profile 기준으로 label·metadata·feature 컬럼과 수치형 규칙을 검증합니다. 따라서 새 데이터셋에는 코어 코드를 고치지 않고 같은 형식의 Dataset Profile을 추가합니다.

`src/modeling_metrics.py`는 Dataset Profile에서 전달받은 정상·불량 label을 기준으로 Recall, AP, Precision, F1, ROC-AUC와 confusion matrix를 공통 계산합니다.

## Dataset Profile 추가

새 데이터셋을 적용할 때는 `configs/datasets/`에 JSON Profile을 추가하고 `config.json`의 `dataset_profile` 경로만 변경합니다. Profile에는 최소한 입력 경로, label 컬럼과 정상·불량 값, timestamp 컬럼·형식(사용 시), ID·그룹·제외 컬럼 역할, feature 타입·선택 방식, 데이터 품질 규칙을 정의합니다.

`prefix` 방식은 지정한 접두어를 가진 컬럼만 feature로 사용합니다. `all_except_metadata` 방식은 label·timestamp·ID·그룹·명시적 제외 컬럼을 제외한 모든 컬럼을 feature로 사용합니다. 문자열 feature는 `feature_types.categorical_columns`에 명시해야 하며, 그 외 feature는 수치형이어야 합니다. Profile 검증에 실패하면 split·모델링 전에 오류가 발생합니다.

원본 파일을 병합해야 하는 데이터셋은 Profile의 `ingestion`에 adapter 이름, source 목록, 각 source의 `read_csv_options`, adapter 전용 결합 규칙을 정의합니다. 같은 수집 형식이면 Profile만 추가하고, 형식이 다를 때에만 adapter registry에 처리기를 추가합니다.

### JSON 작성 예시

1. `configs/datasets/<dataset_id>.json`을 만들고 아래 필수 항목을 정의합니다.
2. 루트 [config.json](../../config.json)의 `dataset_profile`을 새 JSON의 상대 경로로 변경합니다.
3. `python -m unittest discover -s tests -v`로 Profile 구조와 범용 코어를 검증합니다.

병합이 끝난 canonical CSV가 이미 있다면 `ingestion` 없이 다음처럼 작성할 수 있습니다.

```json
{
  "dataset_id": "factory_a",
  "input_path": "data/processed/factory_a_merged.csv",
  "columns": {
    "label": "result",
    "timestamp": null,
    "timestamp_format": null
  },
  "column_roles": {
    "id_columns": ["wafer_id"],
    "group_columns": ["lot_id"],
    "excluded_feature_columns": ["operator_note"]
  },
  "labels": {
    "negative": "PASS",
    "positive": "FAIL"
  },
  "feature_columns": {
    "selection": "all_except_metadata",
    "prefix": null
  },
  "feature_types": {
    "categorical_columns": ["equipment_type"]
  },
  "quality_rules": {
    "missing_ratio_threshold": 0.5,
    "drop_zero_variance": true
  }
}
```

`positive`는 Fail(불량), `negative`는 Pass(정상) label입니다. `id_columns`, `group_columns`, `excluded_feature_columns`은 feature 선택에서 항상 제외합니다. 명시하지 않은 feature는 수치형이어야 하고, 문자열·category feature는 `categorical_columns`에 넣어야 합니다.

원본 feature 파일과 metadata 파일을 병합하는 경우에는 위 JSON에 다음 `ingestion` 블록을 추가합니다. `feature_metadata_pair`는 같은 행 순서의 두 source를 결합하는 내장 adapter입니다.

```json
{
  "ingestion": {
    "adapter": "feature_metadata_pair",
    "sources": [
      {
        "name": "features",
        "path": "data/raw/factory_a_features.csv",
        "read_csv_options": {"header": 0}
      },
      {
        "name": "metadata",
        "path": "data/raw/factory_a_metadata.csv",
        "read_csv_options": {"header": 0}
      }
    ],
    "adapter_options": {
      "feature_source": "features",
      "metadata_source": "metadata",
      "metadata_columns": ["result"]
    }
  }
}
```

`prefix` 선택 방식과 이름 없는 feature 행렬을 사용할 때는 `feature_columns.prefix`를 지정하고, source의 `read_csv_options`에 `"header": null`을 설정합니다. `feature_metadata_pair`가 아닌 원본 결합 방식은 Profile만으로 처리할 수 없으므로 adapter registry에 별도 처리기를 추가해야 합니다.

SECOM의 결측률 기준은 센서 제거 기준입니다. `default_threshold` 및 OOF에서 선택하는 분류 문턱과는 다릅니다. 품질 필터·대치·스케일링·특징 선택은 각 학습 fold에서만 fit합니다.

[실행 안내](USAGE.md) · [문서 목록](../README.md#secom-문서) · [프로젝트 소개](../../README.md)

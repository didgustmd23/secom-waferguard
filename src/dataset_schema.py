# ==========================================
# Dataset Profile 기반 범용 데이터 구조 검증
# - 컬럼명·Label 값·feature 선택 규칙을 DatasetSpec에서 받아 하드코딩을 제거
# - split·EDA·모델링 Script가 같은 feature 정의를 재사용하도록 지원
# - 원본 DataFrame은 수정하지 않고 검증된 feature 컬럼명만 반환
# ==========================================

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

# `python -m src...` 실행과 `python src/<script>.py` 직접 실행을 모두 지원한다.
try:
    from src.modeling_config import DatasetSpec
except ModuleNotFoundError:
    from modeling_config import DatasetSpec


# ==========================================
# DatasetSpec 기준으로 모델 입력 feature 컬럼명 결정
# - prefix: 지정한 접두어를 가진 컬럼만 feature로 선택
# - all_except_metadata: Label·Timestamp를 제외한 모든 컬럼을 feature로 선택
# - 필수 metadata 누락·중복 컬럼·feature 부재를 조기에 차단
# ==========================================
def resolve_feature_columns(
    column_names: Iterable[str], dataset: DatasetSpec
) -> tuple[str, ...]:
    # Iterable 입력을 한 번만 순회할 수 있도록 불변 Tuple로 변환
    columns = tuple(column_names)

    # DataFrame의 중복 컬럼은 선택·로그·Pipeline에서 모호하므로 허용하지 않음
    if len(columns) != len(set(columns)):
        raise ValueError("dataset columns must not contain duplicates")

    # Label과 선택적인 Timestamp 컬럼이 실제 입력에 존재하는지 확인
    required_columns = [dataset.label_column]
    if dataset.timestamp_column is not None:
        required_columns.append(dataset.timestamp_column)
    missing_columns = [column for column in required_columns if column not in columns]
    if missing_columns:
        raise ValueError(f"dataset is missing required columns: {missing_columns}")

    # Profile이 지정한 방식으로 feature 후보를 일관되게 선택
    if dataset.feature_selection_mode == "prefix":
        feature_columns = tuple(
            column
            for column in columns
            if column.startswith(dataset.feature_column_prefix or "")
        )
    elif dataset.feature_selection_mode == "all_except_metadata":
        metadata_columns = set(required_columns)
        feature_columns = tuple(column for column in columns if column not in metadata_columns)
    else:
        # load_dataset_spec이 이미 검증하지만 직접 DatasetSpec을 만들었을 때도 방어
        raise ValueError(f"unsupported feature selection mode: {dataset.feature_selection_mode}")

    # 모델 입력 feature가 하나도 없으면 이후 Pipeline 오류 대신 명확한 원인을 제공
    if not feature_columns:
        raise ValueError("dataset profile did not select any feature columns")

    return feature_columns


# ==========================================
# DatasetSpec과 실제 DataFrame의 metadata·Label 정합성 검증
# - 결측·허용 외 Label을 정상 클래스로 묵시적으로 처리하지 않음
# - 반환된 feature 컬럼만 후속 split·모델링 코드에서 사용
# ==========================================
def validate_dataset_frame(dataframe: pd.DataFrame, dataset: DatasetSpec) -> tuple[str, ...]:
    # 공통 feature 선택 함수로 컬럼 구성과 필수 metadata 존재 여부를 먼저 검증
    feature_columns = resolve_feature_columns(dataframe.columns, dataset)

    # Label 결측은 클래스 비율·평가 지표를 왜곡하므로 즉시 차단
    labels = dataframe[dataset.label_column]
    if labels.isna().any():
        raise ValueError(f"dataset column '{dataset.label_column}' contains missing labels")

    # Dataset Profile에 선언된 정상·불량 Label 외 값이 포함되었는지 확인
    allowed_labels = {dataset.negative_label, dataset.positive_label}
    invalid_labels = labels[~labels.isin(allowed_labels)].unique().tolist()
    if invalid_labels:
        raise ValueError(
            f"dataset column '{dataset.label_column}' contains unsupported labels: "
            f"{invalid_labels}"
        )

    return feature_columns

# ==========================================
# Dataset Profile 기반 데이터 품질 진단과 재현 가능한 로그 생성
# - canonical merged CSV의 schema·Label·Timestamp·결측·상수 feature를 점검
# - 모든 컬럼명·Label·결측 기준은 Dataset Profile에서 읽어 하드코딩을 제거
# - 전체 데이터 품질 점검 결과는 EDA용 후보 정보이며 모델 feature 선택에 사용하지 않음
# - 모델용 결측 제거·상수 feature 제거는 반드시 Train 또는 CV 학습 fold 안에서 fit
# ==========================================

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

# Script 직접 실행과 `src.step2_data_check` 모듈 import를 모두 지원한다.
try:
    from src.dataset_schema import validate_dataset_frame
    from src.modeling_config import DEFAULT_CONFIG_PATH, DatasetSpec, load_modeling_config
except ModuleNotFoundError:
    from dataset_schema import validate_dataset_frame
    from modeling_config import DEFAULT_CONFIG_PATH, DatasetSpec, load_modeling_config


# ==========================================
# 데이터 품질 로그에 단일 진단 항목 추가
# - 모든 진단 결과를 category·item·value·rule·action 형식으로 통일
# - CSV를 다시 읽을 때 shape·Label·Timestamp·feature 결과를 같은 구조로 분석 가능
# ==========================================
def add_log_record(
    records: list[dict[str, Any]],
    category: str,
    item: str,
    value: Any,
    rule: str,
    action: str,
) -> None:
    # CSV 컬럼 구조를 고정해 downstream 보고서·시각화에서 재사용
    records.append(
        {
            "category": category,
            "item": item,
            "value": value,
            "rule": rule,
            "action": action,
        }
    )


# ==========================================
# Dataset Profile 기준 전체 데이터 품질 진단 로그 생성
# - feature별 결측률과 상수 여부는 EDA 후보 정보로만 계산
# - Timestamp는 Profile 형식으로 파싱해 실패 수·시간 범위를 기록
# - 반환값은 로그 DataFrame과 EDA 기준 제거 후보 feature 목록
# ==========================================
def build_quality_log(
    dataframe: pd.DataFrame, dataset: DatasetSpec
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    # Profile과 실제 DataFrame의 metadata·Label·feature 구성을 먼저 검증
    feature_columns = validate_dataset_frame(dataframe, dataset)
    records: list[dict[str, Any]] = []

    # 데이터 규모와 중복 행 수를 기록
    add_log_record(records, "dataset", "row_count", len(dataframe), "all rows", "record")
    add_log_record(
        records,
        "dataset",
        "column_count",
        len(dataframe.columns),
        "all columns",
        "record",
    )
    add_log_record(
        records,
        "dataset",
        "duplicate_row_count",
        int(dataframe.duplicated().sum()),
        "exact full-row duplicate",
        "review",
    )

    # 정상·불량 class별 행 수를 Profile Label 값 기준으로 기록
    # Label 값의 자료형이 달라도 동작하도록 정렬은 강제하지 않는다.
    label_counts = dataframe[dataset.label_column].value_counts()
    for label_value, count in label_counts.items():
        add_log_record(
            records,
            "label_distribution",
            str(label_value),
            int(count),
            f"column={dataset.label_column}",
            "record",
        )

    # Timestamp가 Profile에 정의된 데이터셋만 형식·범위·중복을 점검
    if dataset.timestamp_column is not None and dataset.timestamp_format is not None:
        raw_timestamp = dataframe[dataset.timestamp_column]
        parsed_timestamp = pd.to_datetime(
            raw_timestamp,
            format=dataset.timestamp_format,
            errors="coerce",
        )
        parse_failure_count = int((raw_timestamp.notna() & parsed_timestamp.isna()).sum())

        add_log_record(
            records,
            "timestamp",
            "missing_count",
            int(raw_timestamp.isna().sum()),
            f"column={dataset.timestamp_column}",
            "review",
        )
        add_log_record(
            records,
            "timestamp",
            "parse_failure_count",
            parse_failure_count,
            f"format={dataset.timestamp_format}",
            "review",
        )
        add_log_record(
            records,
            "timestamp",
            "duplicate_count",
            # 파싱 실패(NaT)는 중복 시각이 아니므로 제외한다.
            int(parsed_timestamp.dropna().duplicated().sum()),
            "parsed timestamp duplicate",
            "review",
        )
        add_log_record(
            records,
            "timestamp",
            "minimum",
            parsed_timestamp.min(),
            f"format={dataset.timestamp_format}",
            "record",
        )
        add_log_record(
            records,
            "timestamp",
            "maximum",
            parsed_timestamp.max(),
            f"format={dataset.timestamp_format}",
            "record",
        )

    # feature별 결측률과 Profile의 데이터 품질 기준을 대조
    feature_frame = dataframe.loc[:, feature_columns]
    missing_ratio = feature_frame.isna().mean()
    high_missing_columns = tuple(
        column
        for column, ratio in missing_ratio.items()
        if ratio > dataset.missing_ratio_threshold
    )
    for column, ratio in missing_ratio.items():
        action = "eda_candidate_drop" if column in high_missing_columns else "keep"
        add_log_record(
            records,
            "feature_missing_ratio",
            column,
            float(ratio),
            f"missing_ratio > {dataset.missing_ratio_threshold}",
            action,
        )

    # 결측률 기준을 통과한 feature에서만 상수 여부를 확인
    retained_columns = tuple(
        column for column in feature_columns if column not in high_missing_columns
    )
    zero_variance_columns: tuple[str, ...] = ()
    if dataset.drop_zero_variance:
        zero_variance_columns = tuple(
            column
            for column in retained_columns
            if feature_frame[column].nunique(dropna=True) <= 1
        )

        for column in retained_columns:
            unique_count = int(feature_frame[column].nunique(dropna=True))
            action = "eda_candidate_drop" if column in zero_variance_columns else "keep"
            add_log_record(
                records,
                "feature_constant_check",
                column,
                unique_count,
                "unique_non_missing_count <= 1",
                action,
            )

    # 전체 데이터 기반 후보 목록은 누수 방지를 위해 EDA 전용이라는 사실을 명시
    add_log_record(
        records,
        "feature_summary",
        "feature_count",
        len(feature_columns),
        "Dataset Profile selection",
        "record",
    )
    add_log_record(
        records,
        "feature_summary",
        "high_missing_candidate_count",
        len(high_missing_columns),
        f"missing_ratio > {dataset.missing_ratio_threshold}",
        "eda_only",
    )
    add_log_record(
        records,
        "feature_summary",
        "constant_candidate_count",
        len(zero_variance_columns),
        "unique_non_missing_count <= 1",
        "eda_only",
    )
    add_log_record(
        records,
        "feature_summary",
        "remaining_feature_candidate_count",
        len(feature_columns) - len(high_missing_columns) - len(zero_variance_columns),
        "feature_count - high_missing_candidates - constant_candidates",
        "eda_only",
    )

    # 전역 feature 제거 결과는 반환만 하고 cleaned CSV로 저장하지 않음
    return pd.DataFrame(records), high_missing_columns, zero_variance_columns


# ==========================================
# canonical merged CSV를 읽어 Dataset Profile 기반 품질 로그 파일 저장
# - 입력 경로는 Dataset Profile의 input_path를 사용
# - 로그 경로를 생략하면 config.json이 있는 디렉터리의 logs/dataset_log.csv에 저장
# - 반환된 로그는 재생성 가능하며 Git에 결과 CSV를 저장할 필요가 없음
# ==========================================
def run_quality_check(
    config_path: Path | str, output_path: Path | None = None
) -> Path:
    # 공통 실험 설정과 활성 Dataset Profile을 함께 로드
    resolved_config_path = Path(config_path)
    config = load_modeling_config(resolved_config_path)
    dataset = config.dataset

    # Step 1이 만든 canonical merged CSV를 읽음
    dataframe = pd.read_csv(dataset.input_path)
    quality_log, high_missing_columns, zero_variance_columns = build_quality_log(
        dataframe, dataset
    )

    # 호출자가 경로를 주지 않으면 config.json 위치 기준 logs 폴더에 저장
    destination = output_path or resolved_config_path.parent / "logs" / "dataset_log.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    quality_log.to_csv(destination, index=False, encoding="utf-8-sig")

    # 실행 결과와 EDA 후보 개수를 표시하되, 모델 feature 선택으로 사용하지 않음을 안내
    print(f"dataset_id: {dataset.dataset_id}")
    print(f"input_path: {dataset.input_path}")
    print(f"log_path: {destination}")
    print(f"shape: {dataframe.shape}")
    print(f"high_missing_candidates: {len(high_missing_columns)}")
    print(f"constant_candidates: {len(zero_variance_columns)}")
    print("note: EDA candidates are not model feature-selection results.")

    return destination


# ==========================================
# command line 실행 인자 처리
# - --config로 다른 Dataset Profile을 선택한 config.json을 전달
# - --output으로 데이터 품질 로그 저장 위치를 일시적으로 변경
# ==========================================
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Dataset Profile quality checks")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    run_quality_check(arguments.config, arguments.output)

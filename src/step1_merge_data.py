# ==========================================
# Dataset Profile 기반 SECOM 원본 데이터 병합
# - SECOM의 두 원본 파일을 label·timestamp·sensor가 있는 canonical table로 변환
# - 원본 파일 경로·구분자·metadata 컬럼 순서는 Dataset Profile에서 읽음
# - 행 수·metadata 구조·Label·feature 구성을 저장 전에 검증
# - 다른 원본 형식은 별도 ingestion adapter를 추가하고, 이후 코어 단계는 재사용
# ==========================================

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Script 직접 실행과 `src.step1_merge_data` 모듈 import를 모두 지원한다.
try:
    from src.dataset_schema import validate_dataset_frame
    from src.modeling_config import (
        DEFAULT_CONFIG_PATH,
        DatasetSpec,
        IngestionSpec,
        load_modeling_config,
    )
except ModuleNotFoundError:
    from dataset_schema import validate_dataset_frame
    from modeling_config import (
        DEFAULT_CONFIG_PATH,
        DatasetSpec,
        IngestionSpec,
        load_modeling_config,
    )


# ==========================================
# SECOM whitespace pair 형식의 sensor·metadata 파일 병합
# - sensor 파일은 공백으로 구분된 수치 feature 행렬
# - metadata 파일은 Profile의 metadata_columns 순서대로 Label·Timestamp를 포함
# - 두 파일의 행 수가 다르면 concat 전에 오류를 발생시켜 행 정렬 오류를 차단
# ==========================================
def merge_secom_whitespace_pair(
    dataset: DatasetSpec, ingestion: IngestionSpec
) -> pd.DataFrame:
    # SECOM adapter가 필요한 원본 파일 경로와 공백 구분자를 Profile에서 읽음
    sensor_frame = pd.read_csv(
        ingestion.sensor_path,
        sep=ingestion.separator,
        header=None,
    )
    metadata_frame = pd.read_csv(
        ingestion.metadata_path,
        sep=ingestion.separator,
        header=None,
    )

    # 원본 두 파일은 동일한 wafer 행 순서를 공유해야 하므로 행 수를 먼저 검증
    if len(sensor_frame) != len(metadata_frame):
        raise ValueError(
            "sensor and metadata row counts must match: "
            f"{len(sensor_frame)} != {len(metadata_frame)}"
        )

    # metadata 열 수와 순서가 Profile 정의와 다르면 Label·Timestamp 정렬을 신뢰할 수 없음
    if metadata_frame.shape[1] != len(ingestion.metadata_columns):
        raise ValueError(
            "metadata column count does not match ingestion.metadata_columns: "
            f"{metadata_frame.shape[1]} != {len(ingestion.metadata_columns)}"
        )

    # Label과 timestamp 열의 의미가 Profile 정의와 일치하는지 먼저 확인한다.
    expected_metadata_columns = [dataset.label_column]
    if dataset.timestamp_column is not None:
        expected_metadata_columns.append(dataset.timestamp_column)
    if list(ingestion.metadata_columns) != expected_metadata_columns:
        raise ValueError(
            "ingestion.metadata_columns must match Dataset Profile label/timestamp "
            f"columns: expected={expected_metadata_columns}, "
            f"actual={list(ingestion.metadata_columns)}"
        )

    metadata_frame.columns = ingestion.metadata_columns

    # prefix 방식의 Profile에서만 SECOM sensor 컬럼명을 일관되게 생성
    if dataset.feature_selection_mode != "prefix" or not dataset.feature_column_prefix:
        raise ValueError("secom_whitespace_pair requires a non-empty feature column prefix")
    sensor_frame.columns = [
        f"{dataset.feature_column_prefix}{index}"
        for index in range(sensor_frame.shape[1])
    ]

    # metadata와 sensor를 같은 행 순서로 결합해 canonical table을 생성
    merged_frame = pd.concat([metadata_frame, sensor_frame], axis=1)

    # 일반 schema 검증으로 Profile의 Label·feature 정의와 병합 결과를 대조
    validate_dataset_frame(merged_frame, dataset)

    return merged_frame


# ==========================================
# Profile이 지정한 ingestion adapter로 canonical merged CSV 생성
# - config.json의 Dataset Profile을 실행 시 선택할 수 있음
# - output_path를 생략하면 Dataset Profile의 input_path에 저장
# - 병합 결과는 이후 범용 EDA·split·모델링 단계의 유일한 입력으로 사용
# ==========================================
def merge_dataset(config_path: Path | str, output_path: Path | None = None) -> Path:
    # 공통 실험 설정과 활성 Dataset Profile을 함께 로드
    config = load_modeling_config(config_path)
    dataset = config.dataset
    ingestion = dataset.ingestion

    # canonical table을 만들 원본 ingestion 정의가 없으면 명확하게 중단
    if ingestion is None:
        raise ValueError(f"dataset '{dataset.dataset_id}' does not define ingestion settings")

    # adapter 이름으로 원본 형식별 처리기를 선택
    if ingestion.adapter == "secom_whitespace_pair":
        merged_frame = merge_secom_whitespace_pair(dataset, ingestion)
    else:
        raise ValueError(f"unsupported ingestion adapter: {ingestion.adapter}")

    # 호출자가 경로를 주지 않으면 Profile의 canonical input_path를 사용
    destination = output_path or dataset.input_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    merged_frame.to_csv(destination, index=False)

    # 재실행 확인에 필요한 최소 결과를 콘솔에 기록
    print(f"dataset_id: {dataset.dataset_id}")
    print(f"output_path: {destination}")
    print(f"shape: {merged_frame.shape}")
    print(f"label_counts:\n{merged_frame[dataset.label_column].value_counts()}")

    return destination


# ==========================================
# command line 실행 인자 처리
# - --config로 다른 Dataset Profile을 선택한 config.json을 전달
# - --output으로 canonical merged CSV의 저장 위치를 일시적으로 변경
# ==========================================
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge raw data using a Dataset Profile")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    merge_dataset(arguments.config, arguments.output)

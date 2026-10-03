# ==========================================
# Dataset Profile 기반 DataFrame 구조 검증 테스트
# - SECOM의 sensor_ 접두어 방식과 일반적인 metadata 제외 방식을 함께 검증
# - 결측·허용 외 Label이 모델링 이전에 차단되는지 확인
# ==========================================

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from src.dataset_schema import validate_dataset_frame
from src.modeling_config import DatasetSpec


class DatasetSchemaTest(unittest.TestCase):
    # 일반 tabular 데이터셋용 all_except_metadata Profile을 테스트마다 재사용
    generic_dataset = DatasetSpec(
        dataset_id="generic",
        input_path=Path("unused.csv"),
        label_column="target",
        positive_label="yes",
        negative_label="no",
        timestamp_column=None,
        timestamp_format=None,
        feature_selection_mode="all_except_metadata",
        feature_column_prefix=None,
        missing_ratio_threshold=0.8,
        drop_zero_variance=False,
    )

    # Label을 제외한 일반 수치 컬럼이 feature로 선택되는지 확인
    def test_selects_all_non_metadata_columns(self) -> None:
        dataframe = pd.DataFrame(
            {"age": [20, 30], "income": [1.0, 2.0], "target": ["no", "yes"]}
        )

        features = validate_dataset_frame(dataframe, self.generic_dataset)

        self.assertEqual(features, ("age", "income"))

    # Label 결측을 정상 클래스처럼 처리하지 않는지 확인
    def test_rejects_missing_label(self) -> None:
        dataframe = pd.DataFrame({"feature": [1.0, 2.0], "target": ["no", None]})

        with self.assertRaisesRegex(ValueError, "missing labels"):
            validate_dataset_frame(dataframe, self.generic_dataset)

    # Profile에 없는 Label은 모델링 전에 차단되는지 확인
    def test_rejects_unsupported_label(self) -> None:
        dataframe = pd.DataFrame(
            {"feature": [1.0, 2.0], "target": ["no", "unknown"]}
        )

        with self.assertRaisesRegex(ValueError, "unsupported labels"):
            validate_dataset_frame(dataframe, self.generic_dataset)
